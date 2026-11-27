"""Stand-alone Numba flat-LCDM background solver matching CAMB's conventions.

Replaces CAMB's single-threaded ``camb.get_background(p, no_thermo=True)`` inside
:class:`~kidspipe.theory.emulator.EmulatedLinearBackend` (~6-8 ms -> < 0.06 ms, no GIL/OpenMP contention).

Reproduced CAMB conventions (verified against CAMB 2.0.5 and 1.5.8 ``camb/constants.py`` & ``camb/model.py``)
-----------------------------------------------------------------------------------------------------------
1. **Physical constants (PDG 2022 SI, ``camb/constants.py``):**
   ``c = 2.99792458e8 m/s``, ``G = 6.67430e-11 m^3/kg/s^2``, ``h_p = 6.62607015e-34 J*s``,
   ``k_B = 1.380649e-23 J/K``, ``eV = 1.602176634e-19 J``, ``sigma_boltz = 5.670374419e-8 W/m^2/K^4``,
   ``Mpc = 3.085677581e22 m``, ``T_CMB = 2.7255 K`` (Fixsen 2009), ``N_eff_std = 3.044``.
   (Note: both CAMB 1.5.8 and 2.0.5 default to ``default_nnu = 3.044``, reflecting non-instantaneous
   decoupling QED/e+e- heating rather than the older ``3.046`` value.)

2. **Photon and massless-neutrino densities (``grhog``, ``grhornomass``, ``grhor``):**
   CAMB computes ``grhog`` in Fortran from ``4 * sigma_boltz / c`` (matching ``get_Omega('photon', 0) * h^2``
   to machine precision; ``a_rad = 8 pi^5 k_B^4 / (15 c^3 h_p^3)`` differs by ``3.25e-11`` relative because PDG
   ``sigma_boltz`` is rounded to 10 digits). Each massless standard neutrino carries
   ``omega_nu1_h2 = (7/8) * (4/11)^(4/3) * omega_g_h2``.

3. **One massive neutrino + two massless states (``num_massive_neutrinos=1``, ``share_delta_neff=False`` default):**
   ``set_cosmology`` splits ``N_eff = 3.044`` into ``nu_deg = 3.044 / 3 = 1.0146666666666666`` for the massive
   eigenstate (assigning 1/3 of the ``+0.044`` decoupling heating) and ``n_massless = 3.044 - nu_deg = 2.0293333333...``
   massless species. The effective massive-neutrino temperature is
   ``T_nu = (4/11)^(1/3) * nu_deg^(1/4) * T_CMB``, absorbing the extra ``0.01467`` entropy into ``T_nu`` while
   setting the phase-space degeneracy per spin-summed state cleanly.

4. **``omnuh2`` mapping (``camb_major >= 2`` vs ``camb_major == 1``):**
   In both versions, ``constants.neutrino_mass_fac`` is ``94.06410581217612 eV`` (from ``a_rad`` and ``zeta3``),
   so with ``nu_deg^(3/4)`` the thermal non-relativistic mass conversion is ``~93.042 eV`` (frequently approximated as
   ``mnu / 93.14``).
   - **CAMB >= 2.0.3 (default, ``camb_major >= 2``):** Sets ``omnuh2`` at ``z = 0`` from the exact relativistic
     Fermi-Dirac integral ``rho_nu(a * m_nu / (k_B T_nu))`` (adding ``+5.087e-5`` relative kinetic energy vs the
     non-relativistic limit at ``m_nu = 0.06 eV``).
   - **CAMB 1.x (``camb_major == 1``):** Sets ``omnuh2 = (mnu / neutrino_mass_fac) * nu_deg^(3/4)`` and adjusts
     the phase-space mass ratio ``am0 = a_0 m_nu / (k_B T_nu)`` self-consistently in ``rho_nu(am0 * a)``.

5. **Massive-neutrino energy-density table and Hubble / distance evolution:**
   The normalized integral
   ``F(am) = (120 / (7 pi^4)) * int_0^inf q^2 sqrt(q^2 + am^2) / (e^q + 1) dq``
   is evaluated once at import via ``scipy.integrate.quad`` (`epsrel=1e-13`) on ``M_CHEB = 96`` Chebyshev-Lobatto
   nodes in the compactified variable ``s = am / (am + s0)`` (`s0 = 8.0`) after factoring out the exact
   non-relativistic asymptote ``sqrt(1 + (C_NR * am)^2)`` (`C_NR = 180 zeta3 / (7 pi^4)`), then resampled to
   ``N_TAB = 2049`` uniform ``s`` segments with a clamped cubic spline (`< 5e-12` max error across all ``am >= 0``).
   Flatness fixes ``omega_de_h2 = h^2 - (omch2 + ombh2 + omega_rad_massless_h2 + omnuh2)`` (`w = -1`, `omk = 0`).
   Note: CAMB switches internal `MassiveNu` splines around `am ~ 60` (`z ~ 5`), shifting CAMB's own `h_of_z(6)` by
   `~2.3e-9` in CAMB 2.0.5 (`~1.8e-8` in CAMB 1.5.8) relative to direct Fermi-Dirac quadrature.
   Comoving radial distance ``chi(z) = (c/100) int_0^z dz'/E(z')`` [`Mpc/h`] uses 4-point cubic panel quadrature on
   uniform ``z_chi`` grids (exact to degree 3, ``< 6e-10`` vs CAMB on 6000-node ``Z_CHI``) and subdivided 4-point
   Gauss-Legendre on non-uniform or short grids.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

__all__ = ["compute_background", "fd_rho_table", "C_KMS", "T_CMB", "N_EFF_STD"]

# PDG 2022 SI constants from camb/constants.py
_PI = 3.1415926535897932384626433832795
_ZETA3 = 1.2020569031595942853997
_C_MS = 2.99792458e8
C_KMS = _C_MS / 1e3
_C_OVER_100 = C_KMS / 100.0
_H_PLANCK = 6.62607015e-34
_HBAR = _H_PLANCK / (2.0 * _PI)
_G_SI = 6.67430e-11
_SIGMA_BOLTZ = 5.670374419e-8
_K_B = 1.380649e-23
_EV_J = 1.602176634e-19
_MPC_M = 3.085677581e22
_A_RAD = 8.0 * _PI**5 * _K_B**4 / (15.0 * _C_MS**3 * _H_PLANCK**3)
_A_RAD_SB = 4.0 * _SIGMA_BOLTZ / _C_MS

T_CMB = 2.7255
N_EFF_STD = 3.044

_NU_DEG_1 = N_EFF_STD / 3.0
_N_NU_MASSLESS = N_EFF_STD - _NU_DEG_1
_H100_INV_S = 100.0e3 / _MPC_M
_RHO_CRIT_100_FAC = 8.0 * _PI * _G_SI / (3.0 * _H100_INV_S**2)

_OMEGA_G_H2 = _RHO_CRIT_100_FAC * (_A_RAD_SB * T_CMB**4 / _C_MS**2)
_OMEGA_NU1_REL_H2 = _OMEGA_G_H2 * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
_OMEGA_NU1_ARAD_H2 = _RHO_CRIT_100_FAC * (_A_RAD * T_CMB**4 / _C_MS**2) * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
_OMEGA_RAD_MASSLESS_H2 = _OMEGA_G_H2 + _N_NU_MASSLESS * _OMEGA_NU1_REL_H2
_OMEGA_RAD_ALL_MASSLESS_H2 = _OMEGA_G_H2 + N_EFF_STD * _OMEGA_NU1_REL_H2

_T_NU_1 = (4.0 / 11.0) ** (1.0 / 3.0) * (_NU_DEG_1**0.25) * T_CMB
_AM_PER_EV = _EV_J / (_K_B * _T_NU_1)

_INV_NEUTRINO_MASS_FAC = (
    _ZETA3
    * 3.0
    / 2.0
    / _PI**2
    * (4.0 / 11.0)
    * ((_K_B * T_CMB / _HBAR / _C_MS) ** 3 * _RHO_CRIT_100_FAC / (_C_MS**2 / _EV_J))
)
_NEUTRINO_MASS_FAC = 1.0 / _INV_NEUTRINO_MASS_FAC

# Normalized Fermi-Dirac energy density table on compactified s = am / (am + s0)
_NORM_FD = 7.0 * _PI**4 / 120.0
_C_NR = 180.0 * _ZETA3 / (7.0 * _PI**4)
_FD_S0 = 8.0


def _fd_rho_quad(am: float) -> float:
    """Normalized massive-neutrino Fermi-Dirac energy density via scipy quad (used once at import)."""
    if am <= 0.0:
        return 1.0

    def integrand(q: float) -> float:
        em = math.exp(-q)
        return q * q * math.sqrt(q * q + am * am) * em / (1.0 + em)

    val, _ = quad(integrand, 0.0, 60.0, epsabs=0.0, epsrel=1e-13, limit=200)
    return val / _NORM_FD


def _build_fd_cubic_table():
    m_cheb = 96
    n_tab = 2049
    theta = np.linspace(0.0, _PI, m_cheb + 1)
    s_cheb = 0.5 * (1.0 - np.cos(theta))
    g_cheb = np.ones(m_cheb + 1, dtype=np.float64)
    for i in range(1, m_cheb):
        am = _FD_S0 * s_cheb[i] / (1.0 - s_cheb[i])
        g_cheb[i] = _fd_rho_quad(am) / math.sqrt(1.0 + (_C_NR * am) ** 2)
    g_rev = g_cheb[::-1]
    k_idx = np.arange(m_cheb + 1)
    c_mat = np.cos(np.outer(k_idx, theta))
    w_lob = np.ones(m_cheb + 1, dtype=np.float64)
    w_lob[0] = 0.5
    w_lob[-1] = 0.5
    cheb_c = (2.0 / m_cheb) * (c_mat @ (w_lob * g_rev))
    cheb_c[0] *= 0.5
    cheb_c[-1] *= 0.5

    s_grid = np.linspace(0.0, 1.0, n_tab)
    g_dense = np.polynomial.chebyshev.chebval(2.0 * s_grid - 1.0, cheb_c)
    g_dense[0] = 1.0
    g_dense[-1] = 1.0
    cs = CubicSpline(s_grid, g_dense, bc_type=((1, 0.0), (1, 0.0)))
    coeffs = []
    for k in range(4):
        arr = np.ascontiguousarray(cs.c[k], dtype=np.float64)
        arr.setflags(write=False)
        coeffs.append(arr)
    ds = float(s_grid[1] - s_grid[0])
    return coeffs[0], coeffs[1], coeffs[2], coeffs[3], ds, 1.0 / ds


_FD_C3, _FD_C2, _FD_C1, _FD_C0, _FD_DS, _FD_INV_DS = _build_fd_cubic_table()

# 4-point Gauss-Legendre nodes/weights mapped onto [0, 1]
_GL4_X, _GL4_W = np.polynomial.legendre.leggauss(4)
_GL4_U = np.ascontiguousarray(0.5 * (_GL4_X + 1.0), dtype=np.float64)
_GL4_WU = np.ascontiguousarray(0.5 * _GL4_W, dtype=np.float64)
_GL4_U.setflags(write=False)
_GL4_WU.setflags(write=False)


@nb.njit(inline="always")
def _fd_rho_and_deriv(am: float, c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, c0: np.ndarray,
                      s0: float, ds: float, inv_ds: float, c_nr: float):
    """Evaluate normalized Fermi-Dirac energy density F(am) and dF/d(am) from the cubic table."""
    if am <= 0.0:
        return 1.0, 0.0
    denom = am + s0
    s = am / denom
    idx = int(s * inv_ds)
    n_seg = c0.size
    if idx >= n_seg:
        idx = n_seg - 1
    dx = s - idx * ds
    c3i = c3[idx]
    c2i = c2[idx]
    c1i = c1[idx]
    g = ((c3i * dx + c2i) * dx + c1i) * dx + c0[idx]
    dg_ds = (3.0 * c3i * dx + 2.0 * c2i) * dx + c1i
    ds_dam = s0 / (denom * denom)
    t = c_nr * am
    env = math.sqrt(1.0 + t * t)
    denv_dam = c_nr * t / env
    return g * env, dg_ds * ds_dam * env + g * denv_dam


@nb.njit(inline="always")
def _fd_rho_fast(am: float, c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, c0: np.ndarray,
                 s0: float, ds: float, inv_ds: float, c_nr: float) -> float:
    """Fast normalized Fermi-Dirac energy density F(am)."""
    if am <= 0.0:
        return 1.0
    s = am / (am + s0)
    idx = int(s * inv_ds)
    n_seg = c0.size
    if idx >= n_seg:
        idx = n_seg - 1
    dx = s - idx * ds
    g = ((c3[idx] * dx + c2[idx]) * dx + c1[idx]) * dx + c0[idx]
    t = c_nr * am
    return g * math.sqrt(1.0 + t * t)


@nb.njit(cache=True)
def _fd_rho_vec_nb(ams: np.ndarray, c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, c0: np.ndarray,
                   s0: float, ds: float, inv_ds: float, c_nr: float) -> np.ndarray:
    out = np.empty(ams.size, dtype=np.float64)
    for i in range(ams.size):
        out[i] = _fd_rho_fast(ams[i], c3, c2, c1, c0, s0, ds, inv_ds, c_nr)
    return out


@nb.njit(cache=True)
def _invert_fd_rho(target: float, am_guess: float, c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, c0: np.ndarray,
                   s0: float, ds: float, inv_ds: float, c_nr: float) -> float:
    """Solve F(am) = target for am >= 0 using safeguarded Newton-Raphson."""
    if target <= 1.0:
        return 0.0
    lo = 0.0
    hi = max(100.0, 2.0 * am_guess + 10.0)
    while _fd_rho_fast(hi, c3, c2, c1, c0, s0, ds, inv_ds, c_nr) < target:
        lo = hi
        hi *= 2.0
    am = am_guess if (lo < am_guess < hi) else 0.5 * (lo + hi)
    for _ in range(32):
        fval, dfval = _fd_rho_and_deriv(am, c3, c2, c1, c0, s0, ds, inv_ds, c_nr)
        diff = fval - target
        if abs(diff) <= 1e-14 * target:
            return am
        if diff < 0.0:
            lo = am
        else:
            hi = am
        if dfval > 0.0:
            cand = am - diff / dfval
        else:
            cand = 0.5 * (lo + hi)
        if not (lo < cand < hi):
            cand = 0.5 * (lo + hi)
        am = cand
        if hi - lo <= 1e-14 * max(1.0, am):
            return am
    return am


@nb.njit(inline="always")
def _inv_ez_at(zp1: float, om_cb_h2: float, om_rad_h2: float, nu_om_fac: float, om_de_h2: float,
               inv_h2: float, am0: float, c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, c0: np.ndarray,
               s0: float, ds: float, inv_ds: float, c_nr: float) -> float:
    """Compute 1 / E(z) where zp1 = 1 + z."""
    zp1_2 = zp1 * zp1
    zp1_3 = zp1_2 * zp1
    zp1_4 = zp1_2 * zp1_2
    rho_nu_h2 = 0.0
    if nu_om_fac > 0.0:
        rho_nu_h2 = nu_om_fac * _fd_rho_fast(am0 / zp1, c3, c2, c1, c0, s0, ds, inv_ds, c_nr) * zp1_4
    e2 = (om_cb_h2 * zp1_3 + om_rad_h2 * zp1_4 + rho_nu_h2 + om_de_h2) * inv_h2
    return 1.0 / math.sqrt(e2)


@nb.njit(inline="always")
def _gl4_interval(za: float, zb: float, om_cb_h2: float, om_rad_h2: float, nu_om_fac: float, om_de_h2: float,
                  inv_h2: float, am0: float, c3: np.ndarray, c2: np.ndarray, c1: np.ndarray, c0: np.ndarray,
                  s0: float, ds: float, inv_ds: float, c_nr: float,
                  gl_u: np.ndarray, gl_wu: np.ndarray) -> float:
    """Integrate 1/E(z) over [za, zb] with subdivided 4-point Gauss-Legendre quadrature."""
    span = zb - za
    if span <= 0.0:
        return 0.0
    n_sub = int(math.ceil(span / 0.05))
    if n_sub < 1:
        n_sub = 1
    h_sub = span / n_sub
    total = 0.0
    for s_idx in range(n_sub):
        z0 = za + s_idx * h_sub
        panel = 0.0
        for q in range(4):
            zp1 = 1.0 + z0 + h_sub * gl_u[q]
            panel += gl_wu[q] * _inv_ez_at(
                zp1, om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0,
                c3, c2, c1, c0, s0, ds, inv_ds, c_nr,
            )
        total += h_sub * panel
    return total


@nb.njit(cache=True)
def _validate_z_nb(z: np.ndarray):
    """Validate finite >= 0 strictly ascending 1-D array and test uniform spacing in one pass."""
    n = z.size
    if n < 1:
        return False, False, 0.0
    z0 = z[0]
    if not math.isfinite(z0) or z0 < 0.0:
        return False, False, 0.0
    dz = z[1] - z0 if n > 1 else 0.0
    uniform = n >= 4 and dz <= 0.05
    tol = 1e-11 * dz
    prev = z0
    for i in range(1, n):
        cur = z[i]
        if not math.isfinite(cur) or cur <= prev:
            return False, False, 0.0
        if uniform and abs((cur - prev) - dz) > tol:
            uniform = False
        prev = cur
    return True, uniform, dz


@nb.njit(inline="always")
def _cosmo_budget(
    omch2: float,
    ombh2: float,
    h: float,
    mnu: float,
    camb_major: int,
    c3: np.ndarray,
    c2: np.ndarray,
    c1: np.ndarray,
    c0: np.ndarray,
    s0: float,
    ds: float,
    inv_ds: float,
    c_nr: float,
    om_rad_massless_h2: float,
    om_rad_all_h2: float,
    nu_deg_om_nu1_rel: float,
    nu_deg_om_nu1_arad: float,
    nu_deg_075_over_fac: float,
    am_per_ev: float,
):
    """Compute scalar density-budget quantities for flat LCDM + 1 massive neutrino."""
    h2 = h * h
    inv_h2 = 1.0 / h2
    om_cb_h2 = omch2 + ombh2
    if mnu <= 0.0:
        omnuh2 = 0.0
        om_rad_h2 = om_rad_all_h2
        nu_om_fac = 0.0
        am0 = 0.0
    elif camb_major < 2:
        omnuh2 = mnu * nu_deg_075_over_fac
        om_rad_h2 = om_rad_massless_h2
        nu_om_fac = nu_deg_om_nu1_rel
        am0 = _invert_fd_rho(omnuh2 / nu_om_fac, mnu * am_per_ev, c3, c2, c1, c0, s0, ds, inv_ds, c_nr)
    else:
        am0 = mnu * am_per_ev
        om_rad_h2 = om_rad_massless_h2
        nu_om_fac = nu_deg_om_nu1_arad
        omnuh2 = nu_om_fac * _fd_rho_fast(am0, c3, c2, c1, c0, s0, ds, inv_ds, c_nr)
    omega_m = (om_cb_h2 + omnuh2) * inv_h2
    om_de_h2 = h2 - (om_cb_h2 + om_rad_h2 + omnuh2)
    return om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0, omega_m, omnuh2


@nb.njit(cache=True)
def _compute_background_uniform_nb(
    omch2: float,
    ombh2: float,
    h: float,
    mnu: float,
    z_grid: np.ndarray,
    z_chi: np.ndarray,
    dz_chi: float,
    camb_major: int,
    c3: np.ndarray,
    c2: np.ndarray,
    c1: np.ndarray,
    c0: np.ndarray,
    s0: float,
    ds: float,
    inv_ds: float,
    c_nr: float,
    om_rad_massless_h2: float,
    om_rad_all_h2: float,
    nu_deg_om_nu1_rel: float,
    nu_deg_om_nu1_arad: float,
    nu_deg_075_over_fac: float,
    am_per_ev: float,
    c_over_100: float,
):
    """Fast branch-free uniform-grid kernel with 4-point cubic panel quadrature for z_chi[0] == 0."""
    om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0, omega_m, omnuh2 = _cosmo_budget(
        omch2, ombh2, h, mnu, camb_major, c3, c2, c1, c0, s0, ds, inv_ds, c_nr,
        om_rad_massless_h2, om_rad_all_h2, nu_deg_om_nu1_rel, nu_deg_om_nu1_arad, nu_deg_075_over_fac, am_per_ev,
    )
    nz = z_grid.size
    ez = np.empty(nz, dtype=np.float64)
    for i in range(nz):
        ez[i] = 1.0 / _inv_ez_at(
            1.0 + z_grid[i], om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0,
            c3, c2, c1, c0, s0, ds, inv_ds, c_nr,
        )
    nchi = z_chi.size
    chi = np.empty(nchi, dtype=np.float64)
    chi[0] = 0.0
    f = np.empty(nchi, dtype=np.float64)
    for i in range(nchi):
        f[i] = _inv_ez_at(
            1.0 + z_chi[i], om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0,
            c3, c2, c1, c0, s0, ds, inv_ds, c_nr,
        )
    scale = dz_chi / 24.0
    acc = (9.0 * f[0] + 19.0 * f[1] - 5.0 * f[2] + f[3]) * scale
    chi[1] = c_over_100 * acc
    for i in range(1, nchi - 2):
        acc += (-f[i - 1] + 13.0 * f[i] + 13.0 * f[i + 1] - f[i + 2]) * scale
        chi[i + 1] = c_over_100 * acc
    acc += (f[nchi - 4] - 5.0 * f[nchi - 3] + 19.0 * f[nchi - 2] + 9.0 * f[nchi - 1]) * scale
    chi[nchi - 1] = c_over_100 * acc
    return omega_m, omnuh2, ez, chi


@nb.njit(cache=True)
def _compute_background_general_nb(
    omch2: float,
    ombh2: float,
    h: float,
    mnu: float,
    z_grid: np.ndarray,
    z_chi: np.ndarray,
    camb_major: int,
    c3: np.ndarray,
    c2: np.ndarray,
    c1: np.ndarray,
    c0: np.ndarray,
    s0: float,
    ds: float,
    inv_ds: float,
    c_nr: float,
    om_rad_massless_h2: float,
    om_rad_all_h2: float,
    nu_deg_om_nu1_rel: float,
    nu_deg_om_nu1_arad: float,
    nu_deg_075_over_fac: float,
    am_per_ev: float,
    c_over_100: float,
    gl_u: np.ndarray,
    gl_wu: np.ndarray,
):
    """General kernel supporting non-uniform z_chi, z_chi[0] > 0, and short grids via subdivided GL-4."""
    om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0, omega_m, omnuh2 = _cosmo_budget(
        omch2, ombh2, h, mnu, camb_major, c3, c2, c1, c0, s0, ds, inv_ds, c_nr,
        om_rad_massless_h2, om_rad_all_h2, nu_deg_om_nu1_rel, nu_deg_om_nu1_arad, nu_deg_075_over_fac, am_per_ev,
    )
    nz = z_grid.size
    ez = np.empty(nz, dtype=np.float64)
    for i in range(nz):
        ez[i] = 1.0 / _inv_ez_at(
            1.0 + z_grid[i], om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0,
            c3, c2, c1, c0, s0, ds, inv_ds, c_nr,
        )
    nchi = z_chi.size
    chi = np.empty(nchi, dtype=np.float64)
    acc = 0.0
    if z_chi[0] > 0.0:
        acc = _gl4_interval(
            0.0, z_chi[0], om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0,
            c3, c2, c1, c0, s0, ds, inv_ds, c_nr, gl_u, gl_wu,
        )
    chi[0] = c_over_100 * acc
    for i in range(nchi - 1):
        acc += _gl4_interval(
            z_chi[i], z_chi[i + 1], om_cb_h2, om_rad_h2, nu_om_fac, om_de_h2, inv_h2, am0,
            c3, c2, c1, c0, s0, ds, inv_ds, c_nr, gl_u, gl_wu,
        )
        chi[i + 1] = c_over_100 * acc
    return omega_m, omnuh2, ez, chi


def fd_rho_table(am):
    """Evaluate the precomputed normalized Fermi-Dirac energy density F(am) for scalar or array ``am``."""
    arr = np.asarray(am, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("am must be finite and >= 0")
    out = _fd_rho_vec_nb(
        np.ascontiguousarray(arr.ravel()), _FD_C3, _FD_C2, _FD_C1, _FD_C0, _FD_S0, _FD_DS, _FD_INV_DS, _C_NR
    )
    return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)


def _as_1d_f64(z: np.ndarray, name: str) -> np.ndarray:
    if isinstance(z, np.ndarray) and z.dtype == np.float64 and z.ndim == 1 and z.flags.c_contiguous:
        if z.size < 1:
            raise ValueError(f"{name} must be non-empty")
        return z
    arr = np.ascontiguousarray(z, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    return arr


def compute_background(
    omch2: float,
    ombh2: float,
    h: float,
    mnu: float,
    z_grid: np.ndarray,
    z_chi: np.ndarray,
    *,
    camb_major: int = 2,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute flat-LCDM background ``(omega_m, omnuh2, ez, chi_hmpc)`` via Numba.

    Parameters
    ----------
    omch2, ombh2, h, mnu : float
        Physical CDM/baryon densities, reduced Hubble constant ``h = H0 / 100``, and single massive-neutrino
        mass sum ``mnu`` [eV] (``num_massive_neutrinos=1``, ``N_eff=3.044``, ``T_CMB=2.7255 K``).
    z_grid : (nz,) array
        Strictly ascending non-negative redshifts for ``E(z) = H(z) / H0``.
    z_chi : (nchi,) array
        Strictly ascending non-negative redshifts for comoving radial distance ``chi(z)`` [Mpc/h].
    camb_major : int, optional
        CAMB major-version convention for ``omnuh2`` (``2`` for CAMB >= 2.0.3 relativistic Fermi-Dirac,
        ``1`` for CAMB 1.x ``mnu / 93.04`` non-relativistic proxy). Default ``2``.
    """
    try:
        vals = (float(omch2), float(ombh2), float(h), float(mnu))
    except (TypeError, ValueError) as exc:
        raise ValueError("cosmology parameters must be real scalars") from exc
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"cosmology parameters must be finite, got {vals}")
    omch2_f, ombh2_f, h_f, mnu_f = vals
    if omch2_f <= 0.0 or ombh2_f <= 0.0:
        raise ValueError(f"omch2 and ombh2 must be > 0, got omch2={omch2_f}, ombh2={ombh2_f}")
    if not 0.1 <= h_f <= 3.0:
        raise ValueError(f"h must be in [0.1, 3.0], got {h_f}")
    if mnu_f < 0.0:
        raise ValueError(f"mnu must be >= 0, got {mnu_f}")
    if isinstance(camb_major, bool) or camb_major not in (1, 2):
        raise ValueError(f"camb_major must be 1 or 2, got {camb_major!r}")

    zg = _as_1d_f64(z_grid, "z_grid")
    zc = _as_1d_f64(z_chi, "z_chi")

    ok_zg, _, _ = _validate_z_nb(zg)
    if not ok_zg:
        raise ValueError("z_grid must be finite, >= 0, and strictly ascending")
    ok_zc, uniform_zc, dz_zc = _validate_z_nb(zc)
    if not ok_zc:
        raise ValueError("z_chi must be finite, >= 0, and strictly ascending")

    if uniform_zc and zc[0] == 0.0:
        omega_m, omnuh2, ez, chi_hmpc = _compute_background_uniform_nb(
            omch2_f,
            ombh2_f,
            h_f,
            mnu_f,
            zg,
            zc,
            dz_zc,
            camb_major,
            _FD_C3,
            _FD_C2,
            _FD_C1,
            _FD_C0,
            _FD_S0,
            _FD_DS,
            _FD_INV_DS,
            _C_NR,
            _OMEGA_RAD_MASSLESS_H2,
            _OMEGA_RAD_ALL_MASSLESS_H2,
            _NU_DEG_1 * _OMEGA_NU1_REL_H2,
            _NU_DEG_1 * _OMEGA_NU1_ARAD_H2,
            (_NU_DEG_1**0.75) / _NEUTRINO_MASS_FAC,
            _AM_PER_EV,
            _C_OVER_100,
        )
    else:
        omega_m, omnuh2, ez, chi_hmpc = _compute_background_general_nb(
            omch2_f,
            ombh2_f,
            h_f,
            mnu_f,
            zg,
            zc,
            camb_major,
            _FD_C3,
            _FD_C2,
            _FD_C1,
            _FD_C0,
            _FD_S0,
            _FD_DS,
            _FD_INV_DS,
            _C_NR,
            _OMEGA_RAD_MASSLESS_H2,
            _OMEGA_RAD_ALL_MASSLESS_H2,
            _NU_DEG_1 * _OMEGA_NU1_REL_H2,
            _NU_DEG_1 * _OMEGA_NU1_ARAD_H2,
            (_NU_DEG_1**0.75) / _NEUTRINO_MASS_FAC,
            _AM_PER_EV,
            _C_OVER_100,
            _GL4_U,
            _GL4_WU,
        )
    return float(omega_m), float(omnuh2), ez, chi_hmpc
