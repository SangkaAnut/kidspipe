"""Lensing kernels, NLA intrinsic alignments and the Limber projection (spec s4-s6).

Everything here is the convergence-like spectrum of spec s6.1,

    C_ij(l) = int_0^chi_max dchi  q_i(chi) q_j(chi) / chi^2  P_nl( (l+1/2)/chi , z(chi) ),
    q_i(chi) = s W_i(chi) + F(z(chi)) n_i(chi),                                  (spec s6.1)

with ``s = (3/2) Omega_m (100/c)^2`` [(h/Mpc)^2], ``c = 299792.458 km/s``, ``chi`` in Mpc/h,
``n_i(chi) = n_i(z) dz/dchi`` [h/Mpc] (spec s4), ``P_nl`` in (Mpc/h)^3, ``k`` in h/Mpc.  Expanding
the product gives GG + GI + IG + II exactly (the "folded" kernel, spec s6.1 / D19).  The C_l are
dimensionless, are NOT multiplied by l(l+1)/2pi and carry no (l-1)l(l+1)(l+2)/(l+1/2)^4 factor.

Numerics ("fast", spec s6.3 (E))
--------------------------------
* The integral is done over ``z`` on ``n_nodes`` uniform nodes in ``[0, z_max]`` with the
  trapezoid rule:  ``int dchi f = int dz (dchi/dz) f``.
* ``W_i = (chi/a)[A_i(z) - chi B_i(z)]`` with ``A_i(z) = int_z^inf n_i dz'`` and
  ``B_i(z) = int_z^inf n_i/chi(z') dz'`` by reverse cumulative trapezoid (spec s4 (E)).
* ``ln P_nl`` is interpolated linearly in ``z`` and (default) with 4-point Lagrange cubics in
  ``ln k``; the integrand is exactly 0 outside the (z, k) table (spec s6.2, D13).
* The expensive, parameter dependent look-up ``Q[m, l] = P_nl((l+1/2)/chi_m, z_m)`` is built once
  per cosmology (:meth:`Limber.set_power`); changes of photo-z shifts / ``A_IA`` only re-do the
  cheap kernel contraction ``C[pair, l] = sum_m KK[pair, m] Q[m, l]`` (:meth:`Limber.cl`).

``fidelity="kcap"`` adds the cheap, reproducible parts of the kcap numerics (spec s6.2, D10, D15):
the power spectrum itself (not ln P) is interpolated, and ``W_i`` is divided by the numerically
integrated ``Z_i = int n_i dz`` of the sampled distribution.  It does NOT emulate GSL Akima/bicubic
splines or the adaptive QAG quadrature, so it differs from kcap by ~1e-3, not bit-wise.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline

__all__ = [
    "C_LIGHT",
    "C1_RHO_CRIT",
    "default_ell",
    "coarse_ell",
    "lensing_prefactor",
    "lensing_efficiency",
    "nla_growth_factor",
    "Limber",
    "cl_shear",
]

C_LIGHT = 299792.458  # km/s
C1_RHO_CRIT = 0.0138731  # dimensionless C1 rho_crit (spec s5)


def default_ell(n: int = 400) -> np.ndarray:
    """kcap multipole grid: ``logspace(log10 0.1, log10 5e5, 400)`` (spec s6.2)."""
    return np.logspace(np.log10(0.1), np.log10(5.0e5), n)


def coarse_ell(n: int = 120) -> np.ndarray:
    """Coarse grid ``logspace(0, log10 3e5, 120)`` (spec D20: Delta chi2 = 0.01)."""
    return np.logspace(0.0, np.log10(3.0e5), n)


def lensing_prefactor(omega_m: float) -> float:
    """``s = (3/2) Omega_m (100/c)^2`` in (h/Mpc)^2, c = 299792.458 km/s (spec s0.1)."""
    return 1.5 * omega_m * (100.0 / C_LIGHT) ** 2


def _rev_cumtrapz(f: np.ndarray, dz: float) -> np.ndarray:
    """``out[..., m] = int_{z_m}^{z_end} f dz`` on a uniform grid (reverse cumulative trapezoid)."""
    seg = 0.5 * dz * (f[..., 1:] + f[..., :-1])
    out = np.zeros_like(f)
    out[..., :-1] = np.cumsum(seg[..., ::-1], axis=-1)[..., ::-1]
    return out


def lensing_efficiency(z: np.ndarray, chi: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Lensing efficiency ``W_i(chi)`` [Mpc/h] on a *uniform* z grid (spec s4 (E)).

    ``W = (chi/a) [A(z) - chi B(z)]``, ``A = int_z^zmax n dz'``, ``B = int_z^zmax n/chi(z') dz'``,
    ``a = 1/(1+z)``.  Equivalent to
    ``W(chi) = (chi/a) int_chi^chi_max n(chi') (chi'-chi)/chi' dchi'`` for ``n`` normalised to 1.
    ``n`` has shape ``(..., nz)`` and is the density in z (not chi).
    """
    z = np.asarray(z, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    dz = (z[-1] - z[0]) / (z.size - 1)
    inv_chi = np.divide(1.0, chi, out=np.zeros_like(chi), where=chi > 0)
    A = _rev_cumtrapz(n, dz)
    B = _rev_cumtrapz(n * inv_chi, dz)
    return chi * (1.0 + z) * (A - chi * B)


def nla_growth_factor(z_pk: np.ndarray, pk_nl: np.ndarray) -> np.ndarray:
    """``D(z) = sqrt(P_nl(k_small, z) / P_nl(k_small, 0))`` with ``k_small = k_h[0]`` (spec s5).

    ``pk_nl`` has shape ``(nz, nk)`` with ``z_pk[0] == 0``.  Returns shape ``(nz,)``.
    """
    if abs(z_pk[0]) > 1e-8:
        raise ValueError("the P_nl redshift grid must start at z = 0 (spec s5)")
    return np.sqrt(pk_nl[:, 0] / pk_nl[0, 0])


@njit(parallel=True, cache=True)
def _fill_q(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp, out):
    """out[m, l] = interp of ``tab`` at (z_m, ln k = ln nu_l - ln chi_m); 0 outside the table.

    ``tab`` is ln P (use_exp=True) or P (use_exp=False), shape (nz_pk, nk).  ``ln_nu`` ascending.
    Linear in z; linear (order=1) or 4-point Lagrange cubic (order=3) in ln k.
    """
    nm = z_nodes.size
    nl = ln_nu.size
    nk = lnk.size
    nzp = z_pk.size
    for m in prange(nm):
        z = z_nodes[m]
        if (not np.isfinite(lnchi[m])) or z < z_pk[0] or z > z_pk[nzp - 1]:
            for l in range(nl):
                out[m, l] = 0.0
            continue
        iz = np.searchsorted(z_pk, z) - 1
        if iz < 0:
            iz = 0
        if iz > nzp - 2:
            iz = nzp - 2
        t = (z - z_pk[iz]) / (z_pk[iz + 1] - z_pk[iz])
        row = np.empty(nk)
        for j in range(nk):
            row[j] = (1.0 - t) * tab[iz, j] + t * tab[iz + 1, j]
        j = 0
        for l in range(nl):
            x = ln_nu[l] - lnchi[m]
            if x < lnk[0] or x > lnk[nk - 1]:
                out[m, l] = 0.0
                continue
            while j < nk - 2 and lnk[j + 1] <= x:
                j += 1
            if order == 3 and nk >= 4:
                j0 = j - 1
                if j0 < 0:
                    j0 = 0
                if j0 > nk - 4:
                    j0 = nk - 4
                x0 = lnk[j0]
                x1 = lnk[j0 + 1]
                x2 = lnk[j0 + 2]
                x3 = lnk[j0 + 3]
                w0 = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
                w1 = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
                w2 = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
                w3 = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2))
                v = w0 * row[j0] + w1 * row[j0 + 1] + w2 * row[j0 + 2] + w3 * row[j0 + 3]
            else:
                v = row[j] + (row[j + 1] - row[j]) * (x - lnk[j]) / (lnk[j + 1] - lnk[j])
            if use_exp:
                out[m, l] = np.exp(v)
            else:
                out[m, l] = v if v > 0.0 else 0.0


class Limber:
    """Limber projection with a reusable power-spectrum look-up.

    Typical use (the cosmology changes rarely, the nuisance parameters often)::

        lim = Limber(ell=default_ell(), fidelity="fast", n_nodes=2000)
        lim.set_power(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl)   # slow: builds Q[m, l]
        C = lim.cl(z_nz, nz_shifted, A_IA)                         # fast: kernels + contraction

    Parameters
    ----------
    ell : (nell,) strictly ascending multipoles (floats allowed); default :func:`default_ell`.
    fidelity : ``"fast"`` (spec E) or ``"kcap"`` (P interpolated linearly in value, W / Z_i).
    n_nodes : number of uniform z nodes of the trapezoid rule (spec s15.4: 2000 -> 1.5e-4).
    interp_order : 3 (cubic Lagrange in ln k, default) or 1 (linear).
    kernel_oversample : the cumulative integrals A_i, B_i defining W_i are evaluated on a grid that is
        this many times finer than the Limber nodes (cheap: O(nbin * nodes)) and sub-sampled, which
        removes the O(dz^2) trapezoid error caused by the kinks of the piecewise-linear n(z).
    """

    def __init__(self, ell=None, fidelity: str = "fast", n_nodes: int = 2000, interp_order: int = 3,
                 kernel_oversample: int = 4):
        if fidelity not in ("fast", "kcap"):
            raise ValueError("fidelity must be 'fast' or 'kcap'")
        self.ell = default_ell() if ell is None else np.ascontiguousarray(ell, dtype=np.float64)
        if self.ell.ndim != 1 or self.ell.size == 0 or not np.all(np.isfinite(self.ell)) or self.ell[0] <= -0.5:
            raise ValueError("ell must be a non-empty 1-D array of finite values > -1/2 (nu = l + 1/2 > 0)")
        if np.any(np.diff(self.ell) <= 0):
            raise ValueError("ell must be strictly ascending")
        if int(n_nodes) < 3:
            raise ValueError("n_nodes must be >= 3")
        self.fidelity = fidelity
        self.n_nodes = int(n_nodes)
        self.interp_order = int(interp_order)
        self.kernel_oversample = max(1, int(kernel_oversample))
        self._ready = False
        self._nz_cache = None

    # ------------------------------------------------------------------ power-spectrum stage
    def set_power(self, chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl, z_max=None):
        """Build everything that depends on the cosmology only.

        Parameters
        ----------
        chi_tab, z_tab : (nzb,) comoving radial distance [Mpc/h] vs redshift (z_tab ascending).
        omega_m : Omega_m including massive neutrinos.
        z_pk : (nz,) redshifts of ``pk_nl`` (ascending, ``z_pk[0] = 0``).
        k_h : (nk,) wavenumbers [h/Mpc] ascending (already extended to 500 h/Mpc, spec s2.6).
        pk_nl : (nz, nk) non-linear matter power [(Mpc/h)^3].
        z_max : upper integration limit; default ``min(z_tab[-1], z_pk[-1])``.
        """
        z_tab = np.asarray(z_tab, dtype=np.float64)
        chi_tab = np.asarray(chi_tab, dtype=np.float64)
        z_pk = np.ascontiguousarray(z_pk, dtype=np.float64)
        k_h = np.ascontiguousarray(k_h, dtype=np.float64)
        pk_nl = np.ascontiguousarray(pk_nl, dtype=np.float64)
        if pk_nl.shape != (z_pk.size, k_h.size):
            raise ValueError("pk_nl must have shape (len(z_pk), len(k_h))")
        if z_tab.ndim != 1 or z_tab.shape != chi_tab.shape or z_tab.size < 4:
            raise ValueError("chi_tab and z_tab must be 1-D arrays of equal length >= 4")
        if np.any(np.diff(z_tab) <= 0) or np.any(np.diff(chi_tab) <= 0) or abs(z_tab[0]) > 1e-8:
            raise ValueError("z_tab must start at 0 and chi_tab must increase strictly with z_tab")
        if np.any(np.diff(z_pk) <= 0) or np.any(np.diff(k_h) <= 0) or k_h[0] <= 0:
            raise ValueError("z_pk and k_h must be strictly ascending (k_h > 0)")
        if not (np.all(np.isfinite(pk_nl)) and np.all(pk_nl > 0.0)):
            raise ValueError("pk_nl must be finite and strictly positive")
        if not np.isfinite(omega_m) or omega_m <= 0:
            raise ValueError("omega_m must be positive and finite")
        self.omega_m = float(omega_m)
        self.s = lensing_prefactor(omega_m)
        self.z_max = float(min(z_tab[-1], z_pk[-1]) if z_max is None else z_max)
        if not (0.0 < self.z_max <= z_tab[-1]):
            raise ValueError("z_max must lie in (0, z_tab[-1]] (the distance table is not extrapolated)")

        z = np.linspace(0.0, self.z_max, self.n_nodes)
        spl = CubicSpline(z_tab, chi_tab)
        chi = spl(z)
        chi[0] = max(chi[0], 0.0)
        r = self.kernel_oversample
        self._zf = np.linspace(0.0, self.z_max, (self.n_nodes - 1) * r + 1)  # fine grid; z = _zf[::r]
        self._chif = spl(self._zf)
        self._chif[0] = max(self._chif[0], 0.0)
        self.z = z
        self.dz = z[1] - z[0]
        self.chi = chi
        self.dchidz = spl(z, 1)  # [Mpc/h] per unit z
        self.w = np.full(z.size, self.dz)
        self.w[[0, -1]] *= 0.5
        self.inv_chi = np.divide(1.0, chi, out=np.zeros_like(chi), where=chi > 0)
        # NLA growth factor from the non-linear spectrum at the smallest k (spec s5)
        D_pk = nla_growth_factor(z_pk, pk_nl)
        self.D = CubicSpline(z_pk, D_pk)(z) if z_pk.size > 3 else np.interp(z, z_pk, D_pk)

        use_exp = self.fidelity == "fast"
        tab = np.log(np.maximum(pk_nl, 1e-300)) if use_exp else pk_nl
        with np.errstate(divide="ignore"):
            lnchi = np.where(chi > 0, np.log(np.where(chi > 0, chi, 1.0)), np.nan)
        ln_nu = np.log(self.ell + 0.5)
        Qt = np.empty((z.size, self.ell.size))
        _fill_q(tab, z_pk, np.log(k_h), z, lnchi, ln_nu, self.interp_order, use_exp, Qt)
        self.Qt = Qt  # (nm, nell)
        self._ready = True
        self._nz_cache = None
        return self

    # ------------------------------------------------------------------ kernel stage
    def nla_factor(self, A_IA: float) -> np.ndarray:
        """``F(z_m) = -A_IA * 0.0138731 * Omega_m / D(z_m)`` on the integration nodes (spec s5)."""
        return -A_IA * C1_RHO_CRIT * self.omega_m / self.D

    def _kernels(self, z_nz, nz_shifted):
        """Return (W, n_chi) on the integration nodes; cached for repeated identical n(z)."""
        key = self._nz_cache
        if key is not None and np.array_equal(key[0], nz_shifted) and np.array_equal(key[1], z_nz):
            return key[2], key[3]
        n_in = np.asarray(nz_shifted, dtype=np.float64)
        z_nz = np.asarray(z_nz, dtype=np.float64)
        if z_nz[-1] > self.z_max + 1e-12 and np.any(n_in[:, z_nz > self.z_max + 1e-12] != 0):
            raise ValueError("n(z) has support above the integration limit z_max")
        r = self.kernel_oversample
        nf = np.stack([np.interp(self._zf, z_nz, row, left=0.0, right=0.0) for row in n_in])
        Wf = lensing_efficiency(self._zf, self._chif, nf)
        W = Wf[:, ::r]
        n = nf[:, ::r]
        if self.fidelity == "kcap":
            W = W / (_rev_cumtrapz(nf, self._zf[1] - self._zf[0])[:, :1])
        n_chi = n / self.dchidz  # [h/Mpc]
        self._nz_cache = (n_in.copy(), z_nz.copy(), W, n_chi)
        return W, n_chi

    def kernels(self, z_nz, nz_shifted, A_IA: float = 0.0):
        """Return ``q (nbin, nm)`` = s W_i + F n_i(chi) on the integration nodes ``self.z``."""
        W, n_chi = self._kernels(z_nz, nz_shifted)
        A = np.asarray(A_IA, dtype=np.float64)
        if A.ndim == 0:
            return self.s * W + self.nla_factor(float(A))[None, :] * n_chi
        # per-bin amplitude array (nbin,): F_i(z) = -A_i C1rho_crit Omega_m / D(z)  (mass-dependent NLA, KiDS-Legacy)
        F = -A[:, None] * C1_RHO_CRIT * self.omega_m / self.D[None, :]
        return self.s * W + F * n_chi

    def cl(self, nz_shifted, A_IA: float = 0.0, z_nz=None) -> np.ndarray:
        """Limber ``C_ij(l)`` for all bin pairs, shape ``(nbin, nbin, nell)`` (symmetric).

        ``C = sum_m w_m (dchi/dz)_m q_i q_j / chi_m^2 Q[m, l]`` -- one matmul over all pairs.
        """
        if not self._ready:
            raise RuntimeError("call set_power first")
        z_nz = self._z_nz if z_nz is None else z_nz
        if z_nz is None:
            raise ValueError("z_nz required")
        q = self.kernels(z_nz, nz_shifted, A_IA)
        nb = q.shape[0]
        ii, jj = np.tril_indices(nb)
        meas = self.w * self.dchidz * self.inv_chi**2
        KK = q[ii] * q[jj] * meas[None, :]  # (npair, nm)
        Cp = KK @ self.Qt  # (npair, nell)
        C = np.empty((nb, nb, self.ell.size))
        C[ii, jj] = Cp
        C[jj, ii] = Cp
        return C

    # allow Limber(z_nz=...) style default
    _z_nz = None

    def set_nz_grid(self, z_nz):
        """Store the default z grid of the n(z) tables so :meth:`cl` can omit it."""
        self._z_nz = np.asarray(z_nz, dtype=np.float64)
        return self


def cl_shear(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl, z_nz, nz_shifted, A_IA, ell,
             fidelity: str = "fast", n_nodes: int = 2000) -> np.ndarray:
    """One-shot Limber convergence spectra ``C_ij(l)`` (GG+GI+IG+II folded), shape (nbin, nbin, nell).

    ``C_ij(l) = int dchi q_i q_j / chi^2 P_nl((l+1/2)/chi, z)``, see the module docstring.
    Inputs: ``chi_tab`` [Mpc/h] at ``z_tab``; ``omega_m``; ``pk_nl[z_pk, k_h]`` in (Mpc/h)^3 with
    k_h in h/Mpc extended to 500 h/Mpc; ``nz_shifted[nbin, len(z_nz)]`` normalised; ``A_IA`` the NLA
    amplitude; ``ell`` ascending multipoles.  For repeated calls use :class:`Limber` to reuse the
    power-spectrum look-up.
    """
    lim = Limber(ell=ell, fidelity=fidelity, n_nodes=n_nodes)
    lim.set_power(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl)
    return lim.cl(nz_shifted, A_IA, z_nz=z_nz)
