"""Transfer-function emulator for the slow cosmological parameters (omch2, ombh2, h)  --  drop-in for ``LinearBackend``.

Why / what is emulated
----------------------
The only expensive part of a likelihood call that changes (omch2, ombh2, h) is the CAMB Boltzmann run
(0.5 s at 4 threads, 2 s at 1 thread).  The linear spectrum factorises exactly (flat LCDM, adiabatic modes,
m_nu fixed; ``docs/interface_notes/linear_background.md``)::

    P(k, z; omch2, ombh2, h, A_s, n_s) = (A_s / A_ref) (k / 0.05 Mpc^-1)^(n_s - n_ref) * P_ref(k, z; omch2, ombh2, h)

so ONLY ``P_ref`` (the spectrum for ``ln 10^10 A_s = 3``, ``n_s = 1``) is emulated.  Formulation:

* Independent variable: **physical wavenumber k [1/Mpc] on one fixed grid** (CAMB's native nodes at the box centre,
  ~206 nodes in [5e-5, 21] 1/Mpc).  CAMB's own native k nodes depend on the cosmology (183 ... 212 nodes) and, in
  h/Mpc, on h, so a fixed 1/Mpc grid is the only grid on which the emulated function has a fixed dimension.  The
  emulated object's k_h grid is ``k_mpc / h`` (exactly like CAMB, ``k_h[-1] = 21/h``), so the downstream HMcode /
  Limber code needs no change.
* Dependent variable: ``ln P_Mpc(k, z)`` in Mpc^3 (P_h = h^3 P_Mpc is an exact conversion) for the total-matter
  spectrum, and ``ln(P_cb / P_tot)`` (cold CDM+baryon over total; smooth, small) for the cold spectrum.
  The redshift axis is the "master" grid (union of the 150-node spec grid and the 41-node reduced grid), so both
  production grids are exact subsets (any other ascending z-grid is served by a cubic spline in z of ln P).
* Compression: PCA (SVD via the Gram matrix of the training set) of the flattened ``(nz, nk)`` log-spectra.
* Regression: see the artefact metadata (``meta_json``); selected by measured held-out error
  (``docs/PERFORMANCE.md``).
* NOT emulated (exact and cheap): ``omega_m``, ``omnuh2`` (CAMB ``set_cosmology``), ``E(z)`` and ``chi(z)`` (CAMB
  *background* only, ``get_background(no_thermo=True)``, ~1 ms, bit-identical to the full run), the A_s / n_s
  rescale, and ``sigma8_0`` (quadrature of the emulated z = 0 spectrum with the exact top-hat window on a dense
  spline grid; the n_s tilt is applied inside the integral so it is exact for any (A_s, n_s)).

Fallback
--------
:class:`EmulatedLinearBackend` serves a request with the emulator only if (omch2, ombh2, h) lies inside the training
box and ``mnu == 0.06``; anything else (and any emulator failure) is delegated to the exact CAMB path of
:class:`~kidspipe.theory.linear.LinearBackend` (counted in ``stats['fallback']``).

Artefacts: ``data/emulator/transfer_emulator_v1.npz`` (trained with CAMB 1.5.8) and
``data/emulator/transfer_emulator_v1_camb2.npz`` (same design, trained with CAMB 2.0.5); the one matching the installed
CAMB major version is loaded (:func:`kidspipe.paths.emulator_artefact`; override with ``$KIDSPIPE_EMULATOR`` or ``path=``).
Both are written by ``scripts/train_emulator.py`` with version stamps (CAMB version, box, seeds, sizes).
"""
from __future__ import annotations

import json
import threading
import warnings
from collections import OrderedDict
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
from numpy.polynomial import legendre as _leg
from scipy.interpolate import CubicSpline

from .linear import K_PIVOT_MPC, LinearBackend, LinearResult, Z_CHI, Z_GRID_SPEC, _Z_CHI_RO, _ro

__all__ = ["TransferEmulator", "EmulatedLinearBackend", "default_artefact_path", "load_emulator"]

PARAM_ORDER = ("omch2", "ombh2", "h")


def default_artefact_path() -> Path:
    from .. import paths
    return paths.emulator_artefact()


# ======================================================================================================
# regressors (numpy-only evaluation; training lives in scripts/train_emulator.py)
# ======================================================================================================
def poly_terms(deg: int, dim: int = 3):
    """Exponent tuples of the total-degree-``deg`` tensor basis."""
    return [t for t in product(range(deg + 1), repeat=dim) if sum(t) <= deg]


def poly_design(X: np.ndarray, deg: int) -> np.ndarray:
    """Legendre tensor design matrix (n, nterm) for scaled inputs X in [-1, 1]^3."""
    X = np.atleast_2d(X)
    V = [_leg.legvander(X[:, i], deg) for i in range(X.shape[1])]
    return np.stack([V[0][:, a] * V[1][:, b] * V[2][:, c] for a, b, c in poly_terms(deg)], axis=1)


def gp_kernel(Xa: np.ndarray, Xb: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Squared-exponential ARD kernel exp(-|(xa-xb)/ell|^2 / 2); Xa (n,3), Xb (m,3), ell (3,)."""
    A, B = Xa / ell, Xb / ell
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0))


class _Regressor:
    """Maps scaled parameters x (n, 3) -> PCA coefficients (n, npc).

    kind 'poly'   : Legendre total-degree polynomial.
    kind 'polygp' : polynomial of degree ``deg`` + per-PC squared-exponential ARD Gaussian process on the residual,
                    ``c_j(x) = P(x) coef_j + sum_i exp(-|(x - X_i) / ell_j|^2 / 2) alpha_ij``.
    """

    def __init__(self, d: dict):
        self.kind = str(d["kind"])
        self.deg = int(d["deg"])
        self.coef = np.asarray(d["coef"], dtype=np.float64)                  # (nterm, npc)
        if self.kind == "polygp":
            self.X = np.asarray(d["gp_X"], dtype=np.float64)                 # (N, 3)
            self.alpha = np.asarray(d["gp_alpha"], dtype=np.float64)         # (N, npc)
            self.ell = np.asarray(d["gp_ell"], dtype=np.float64)             # (npc, 3)
            self._inv_ell2 = 0.5 / self.ell ** 2                              # (npc, 3)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        out = poly_design(x, self.deg) @ self.coef
        if self.kind == "polygp":
            for i in range(x.shape[0]):
                diff2 = (x[i][None, :] - self.X) ** 2                         # (N, 3)
                d2 = diff2 @ self._inv_ell2.T                                  # (N, npc)
                out[i] += np.einsum("np,np->p", np.exp(-d2), self.alpha)
        return out


class _PCA:
    """mean + orthonormal basis; ``recon(c) = mean + c @ basis``."""

    def __init__(self, mean, basis):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)                     # (npc, D)

    def recon(self, c: np.ndarray) -> np.ndarray:
        return self.mean + c @ self.basis


class TransferEmulator:
    """Loaded emulator artefact: ``predict(omch2, ombh2, h)`` -> ``(ln P_tot, ln P_cb)`` on ``(z_master, k_mpc)``
    for the reference primordial spectrum (``ln 10^10 A_s = ln_As_ref``, ``n_s = ns_ref``), P in Mpc^3.
    """

    def __init__(self, path=None):
        path = Path(path) if path is not None else default_artefact_path()
        with np.load(path, allow_pickle=False) as d:
            self.meta = json.loads(str(d["meta_json"]))
            self.box = np.asarray(d["box"], dtype=np.float64)               # (3, 2) omch2, ombh2, h
            self.k_mpc = np.asarray(d["k_mpc"], dtype=np.float64)
            self.z = np.asarray(d["z_master"], dtype=np.float64)
            self.ln_As_ref = float(d["ln_As_ref"])
            self.ns_ref = float(d["ns_ref"])
            self.mnu = float(d["mnu"])
            self.nz, self.nk = self.z.size, self.k_mpc.size
            self.pca_tot = _PCA(d["tot_mean"], d["tot_basis"])
            self.pca_rat = _PCA(d["rat_mean"], d["rat_basis"])
            self.reg_tot = _Regressor({k[8:]: d[k] for k in d.files if k.startswith("reg_tot_")})
            self.reg_rat = _Regressor({k[8:]: d[k] for k in d.files if k.startswith("reg_rat_")})
        self.path = path
        self._lo, self._hi = self.box[:, 0], self.box[:, 1]
        self._check_camb_version()

    def _check_camb_version(self):
        """Warn (never raise) if the installed CAMB differs from the one the artefact was trained with.

        The emulated targets are CAMB 1.5.8 transfer functions (N_eff, YHe table, k-sampling are version dependent);
        a different version silently shifts the emulator relative to the exact fallback by an unquantified amount.
        """
        try:
            import camb
        except ImportError:                                   # pragma: no cover
            return
        trained = str(self.meta.get("camb_version", ""))
        if trained and trained != camb.__version__:
            warnings.warn(f"transfer emulator {self.path.name} was trained with CAMB {trained} but CAMB {camb.__version__} "
                          "is installed: emulated and exact (fallback) spectra may differ by more than the stated "
                          "accuracy; retrain with scripts/train_emulator.py", RuntimeWarning, stacklevel=3)

    # ---------------------------------------------------------------- geometry
    def contains(self, omch2: float, ombh2: float, h: float, mnu: float = 0.06) -> bool:
        """True if (omch2, ombh2, h) is inside the training box and mnu is the trained value."""
        x = (omch2, ombh2, h)
        return bool(abs(mnu - self.mnu) < 1e-12 and all(lo <= v <= hi for v, lo, hi in zip(x, self._lo, self._hi)))

    def scale(self, x) -> np.ndarray:
        """(omch2, ombh2, h) -> [-1, 1]^3 in LOGARITHMIC coordinates (the spectra are ~power laws in the densities)."""
        x = np.log(np.asarray(x, dtype=np.float64))
        return 2.0 * (x - np.log(self._lo)) / (np.log(self._hi) - np.log(self._lo)) - 1.0

    # ---------------------------------------------------------------- prediction
    def predict(self, omch2: float, ombh2: float, h: float):
        """``(lnP_tot, lnP_cb)`` each ``(nz_master, nk)`` (ln of P in Mpc^3 at the reference primordial spectrum)."""
        x = self.scale([omch2, ombh2, h])[None, :]
        lnp = self.pca_tot.recon(self.reg_tot(x))[0].reshape(self.nz, self.nk)
        rat = self.pca_rat.recon(self.reg_rat(x))[0].reshape(self.nz, self.nk)
        return lnp, lnp + rat


def load_emulator(path=None) -> TransferEmulator:
    return TransferEmulator(path)


# ======================================================================================================
# sigma8 by quadrature (exact top-hat window, dense spline grid)
# ======================================================================================================
_N_DENSE = 4000


def _tophat_w2(x: np.ndarray) -> np.ndarray:
    w = np.empty_like(x)
    small = x < 1e-2
    xs = x[small]
    w[small] = 1.0 - 0.1 * xs * xs
    xl = x[~small]
    w[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / xl ** 3
    return w * w


class _Sigma8:
    """sigma_8(A_s, n_s) of a reference z=0 spectrum: sigma^2 = int dlnk k^3 P_Mpc / (2 pi^2) W^2(k R), R = 8/h Mpc."""

    def __init__(self, k_mpc: np.ndarray, lnp0: np.ndarray, h: float):
        lnk = np.log(k_mpc)
        self.lnk = np.linspace(lnk[0], lnk[-1], _N_DENSE)
        k = np.exp(self.lnk)
        p = np.exp(CubicSpline(lnk, lnp0)(self.lnk))
        integ = k ** 3 * p / (2.0 * np.pi ** 2) * _tophat_w2(k * 8.0 / h)
        self.integ = integ                                      # (D,)  at the reference (A_ref, n_ref)
        self.lnk_rel = self.lnk - np.log(K_PIVOT_MPC)
        self.dlnk = self.lnk[1] - self.lnk[0]

    def __call__(self, fac: float, dn: float) -> float:
        g = self.integ if dn == 0.0 else self.integ * np.exp(dn * self.lnk_rel)
        s2 = fac * self.dlnk * (g.sum() - 0.5 * (g[0] + g[-1]))
        return float(np.sqrt(s2))


# ======================================================================================================
# the backend
# ======================================================================================================
class _EmuEntry:
    __slots__ = ("k_h", "pk_ref", "pk_cb_ref", "chi", "ez", "omega_m", "omnuh2", "h", "ombh2", "omch2", "mnu",
                 "s8", "sigma8_ref")


class _LazyCambState:
    """``_state`` handle for results from the emulator: ``get_pnl(..., backend='camb')`` falls back to an exact CAMB run
    (done lazily, only if that backend is requested)."""
    __slots__ = ("owner", "args")

    def __init__(self, owner: "EmulatedLinearBackend", args: tuple):
        self.owner, self.args = owner, args

    def camb_nonlinear(self, A_baryon: float, eta_baryon: float) -> np.ndarray:
        exact = LinearBackend.compute(self.owner, *self.args)
        return exact._state.camb_nonlinear(A_baryon, eta_baryon)


class EmulatedLinearBackend(LinearBackend):
    """``LinearBackend`` whose ``compute`` uses the transfer emulator inside its training box and exact CAMB outside.

    Same constructor arguments as :class:`LinearBackend` (they configure the exact fallback and the output ``z_grid``)
    plus ``emulator`` (a :class:`TransferEmulator`, default: the shipped artefact) and ``fallback`` (``True``: delegate
    out-of-box requests to CAMB; ``False``: raise ``ValueError``).  ``stats`` additionally counts ``emu_evals``,
    ``fallback`` and ``emu_seconds``.

    The returned :class:`LinearResult` has the same fields and shapes as the CAMB one (``pk_cb`` included); its
    ``k_h`` is ``emulator.k_mpc / h`` and is a fresh (cached) array per (omch2, ombh2, h).
    """

    def __init__(self, *args, emulator: Optional[TransferEmulator] = None, fallback: bool = True, **kw):
        super().__init__(*args, **kw)
        self.emulator = emulator if emulator is not None else TransferEmulator()
        self.fallback = bool(fallback)
        self.stats.update({"emu_evals": 0, "emu_cache_hits": 0, "fallback": 0, "emu_seconds": 0.0})
        self._emu_cache: "OrderedDict[tuple, _EmuEntry]" = OrderedDict()
        em = self.emulator
        zi = np.abs(em.z[:, None] - self.z_grid[None, :]).argmin(axis=0)       # nearest master node
        self._z_subset = bool(np.all(np.abs(em.z[zi] - self.z_grid) < 1e-9))
        self._zi = zi
        if not self._z_subset and (self.z_grid[-1] > em.z[-1] + 1e-12):
            raise ValueError("requested z_grid extends beyond the emulator's redshift range")

    # ---------------------------------------------------------------- background (exact, cheap)
    @staticmethod
    def _background(omch2, ombh2, h, mnu):
        import camb
        p = camb.CAMBparams()
        p.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=mnu, num_massive_neutrinos=1)
        p.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="fluid")
        b = camb.get_background(p, no_thermo=True)
        return p, b

    def _emu_entry(self, omch2, ombh2, h, mnu) -> _EmuEntry:
        em = self.emulator
        t0 = perf_counter()
        lnp, lnpcb = em.predict(omch2, ombh2, h)
        if self._z_subset:
            lnp, lnpcb = lnp[self._zi], lnpcb[self._zi]
        else:
            lnp = CubicSpline(em.z, lnp, axis=0)(self.z_grid)
            lnpcb = CubicSpline(em.z, lnpcb, axis=0)(self.z_grid)
        h3 = h ** 3
        e = _EmuEntry()
        e.k_h = _ro(em.k_mpc / h)
        e.pk_ref = _ro(np.exp(lnp) * h3)                                  # (Mpc/h)^3 at (A_ref, n_ref)
        e.pk_cb_ref = _ro(np.exp(lnpcb) * h3)
        p, b = self._background(omch2, ombh2, h, mnu)
        e.chi = _ro(b.comoving_radial_distance(np.asarray(Z_CHI), tol=1e-8) * p.h)
        e.ez = _ro(b.h_of_z(np.asarray(self.z_grid)) / b.h_of_z(0.0))
        e.omega_m, e.omnuh2 = float(p.omegam), float(p.omnuh2)
        e.h, e.ombh2, e.omch2, e.mnu = float(h), float(ombh2), float(omch2), float(mnu)
        e.s8 = _Sigma8(em.k_mpc, lnp[0], h)
        e.sigma8_ref = e.s8(1.0, 0.0)
        if not (np.isfinite(e.pk_ref).all() and np.isfinite(e.pk_cb_ref).all() and (e.pk_ref > 0).all()):
            raise RuntimeError("emulator returned a non-finite / non-positive spectrum")
        self.stats["emu_seconds"] += perf_counter() - t0
        return e

    # ---------------------------------------------------------------- public API
    def compute(self, omch2, ombh2, h, n_s, ln_1e10_A_s, mnu: float = 0.06) -> LinearResult:
        omch2, ombh2, h, n_s, ln_1e10_A_s, mnu = self._validate(omch2, ombh2, h, n_s, ln_1e10_A_s, mnu)
        em = self.emulator
        if not em.contains(omch2, ombh2, h, mnu):
            if not self.fallback:
                raise ValueError(f"(omch2, ombh2, h) = ({omch2}, {ombh2}, {h}) outside the emulator box "
                                 f"{em.box.tolist()} (or mnu != {em.mnu})")
            self.stats["fallback"] += 1
            return super().compute(omch2, ombh2, h, n_s, ln_1e10_A_s, mnu)
        self.stats["compute_calls"] += 1
        key = (omch2, ombh2, h)
        with self._lock:
            e = self._emu_cache.get(key)
            if e is None:
                e = self._emu_entry(omch2, ombh2, h, mnu)
                self.stats["emu_evals"] += 1
                self._emu_cache[key] = e
                while len(self._emu_cache) > self.cache_size:
                    self._emu_cache.popitem(last=False)
            else:
                self._emu_cache.move_to_end(key)
                self.stats["emu_cache_hits"] += 1
        As_ref = float(np.exp(em.ln_As_ref) * 1e-10)
        As = float(np.exp(ln_1e10_A_s) * 1e-10)
        fac = As / As_ref
        dn = float(n_s) - em.ns_ref
        if dn != 0.0:
            tilt = (e.k_h * e.h / K_PIVOT_MPC) ** dn
            pk = e.pk_ref * (fac * tilt)[None, :]
            pcb = e.pk_cb_ref * (fac * tilt)[None, :]
        else:
            pk = e.pk_ref * fac
            pcb = e.pk_cb_ref * fac
        s8 = e.s8(fac, dn)
        if not (np.isfinite(s8) and np.isfinite(pk).all() and np.isfinite(pcb).all()):
            raise ValueError("non-finite linear power spectrum (n_s / ln_1e10_A_s out of the representable range): "
                             f"n_s={n_s}, ln_1e10_A_s={ln_1e10_A_s}")
        return LinearResult(
            z=self.z_grid, k_h=e.k_h, pk_lin=pk, chi=e.chi, z_chi=_Z_CHI_RO, omega_m=e.omega_m, sigma8_0=float(s8),
            h=e.h, pk_cb=pcb, ez=e.ez, omnuh2=e.omnuh2, omega_b=e.ombh2 / e.h ** 2, omega_c=e.omch2 / e.h ** 2,
            S8=float(s8 * np.sqrt(e.omega_m / 0.3)), n_s=float(n_s), ombh2=e.ombh2, omch2=e.omch2,
            ln_1e10_A_s=float(ln_1e10_A_s), mnu=e.mnu,
            _state=_LazyCambState(self, (omch2, ombh2, h, float(n_s), float(ln_1e10_A_s), mnu)))

    def clear_cache(self) -> None:
        super().clear_cache()
        with self._lock:
            self._emu_cache.clear()
