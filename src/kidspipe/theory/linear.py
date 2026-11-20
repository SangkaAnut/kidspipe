"""Linear background + linear matter power spectrum (CAMB wrapper with exact caching).

Implements spec s2.1-s2.4 (``docs/MASTER_MATH_SPEC.md``).

What is computed
----------------
For a flat LCDM cosmology with one massive neutrino (mnu = 0.06 eV, spec s2.1) CAMB is run in
*transfer* mode (``WantCls=False``, ``WantTransfer=True``, no reionisation, accuracy defaults,
``Transfer.high_precision=True``) on the redshift grid of spec s2.2 and ``kmax = 20 / Mpc`` (CAMB pads
by 5 %, giving ~194 native k nodes up to 21 / Mpc).  The outputs are

* ``pk_lin``  total-matter linear spectrum,  P(k,z) = 2 pi^2 / k^3 * P_R(k) * T_tot(k,z)^2   [(Mpc/h)^3]
  with P_R = A_s (k / 0.05 Mpc^-1)^(n_s - 1)                                          (spec s2.4)
* ``pk_cb``   same for the cold (CDM + baryon) component, CAMB variable ``delta_nonu``
* ``chi``     comoving radial distance chi(z) = (c/H0) int_0^z dz'/E(z') * h             [Mpc/h]
  on ``z_chi = linspace(0, 6, 6000)``                                                  (spec s2.2/s2.3)
* ``ez``      E(z) = H(z)/H0 on the P(k) redshift grid                                  (spec s2.3)
* ``omega_m`` = (ombh2 + omch2 + omnuh2)/h^2 (includes the massive neutrino, CAMB ``omegam``)
* ``sigma8_0`` = sigma(R = 8 Mpc/h, z = 0) of the total matter spectrum; ``S8 = sigma8 sqrt(Om/0.3)``

Exact caching (efficiency)
--------------------------
CAMB's transfer functions T(k,z) depend on (omch2, ombh2, h, mnu) only.  The primordial spectrum enters
as an exact multiplicative factor

    P(k; A_s, n_s) = P(k; A_ref, n_ref) * (A_s / A_ref) * (k / 0.05 Mpc^-1)^(n_s - n_ref),

so changes of ``ln_1e10_A_s`` (and, optionally, of ``n_s``) are handled by a numpy rescale of the cached
arrays (agreement with a fresh CAMB run ~1e-15, verified in ``tests/test_linear.py``).  The CAMB result
object is kept in an LRU cache; ``sigma8_0`` of a rescaled spectrum is obtained by re-running only CAMB's
cheap transfer->power step (0.3 ms), not the Boltzmann evolution (0.3-2 s).

By default the cache key is ``(omch2, ombh2, h, mnu)``; ``n_s`` is rescaled.  Use
``LinearBackend(key_on_ns=True)`` for the strict key ``(omch2, ombh2, h, n_s, mnu)`` of DESIGN.md.

See ``docs/interface_notes/linear_background.md`` for timings and CAMB-usage notes.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import numpy as np

__all__ = [
    "LinearBackend", "LinearResult", "Z_GRID_SPEC", "Z_GRID_REDUCED", "Z_CHI", "default_nthreads",
    "K_PIVOT_MPC",
]

#: Primordial pivot scale [1/Mpc] (spec s2.1).
K_PIVOT_MPC = 0.05
#: Speed of light [km/s] (spec s0.1).
C_KMS = 299792.458

#: P(k,z) redshift grid of spec s2.2: 150 nodes, spacing 0.02 below z=2 and 4/49 above (ascending).
Z_GRID_SPEC = np.concatenate([np.linspace(0.0, 2.0, 100, endpoint=False), np.linspace(2.0, 6.0, 50)])
#: Reduced 41-node grid: 20 nodes (dz=0.05) on [0,1), 10 nodes (dz=0.1) on [1,2), 11 nodes (dz=0.4) on [2,6].
Z_GRID_REDUCED = np.concatenate([np.linspace(0.0, 1.0, 20, endpoint=False),
                                 np.linspace(1.0, 2.0, 10, endpoint=False),
                                 np.linspace(2.0, 6.0, 11)])
#: Distance table redshifts (spec s2.2): 6000 nodes on [0, 6].
Z_CHI = np.linspace(0.0, 6.0, 6000)
for _a in (Z_GRID_SPEC, Z_GRID_REDUCED, Z_CHI):
    _a.setflags(write=False)


def default_nthreads() -> int:
    """Default number of CAMB OpenMP threads, ``min(4, os.cpu_count())`` (22 threads is slower in practice)."""
    return max(1, min(4, os.cpu_count() or 1))


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclass(eq=False)
class LinearResult:
    """Linear-theory products for one cosmology (all float64, redshifts ascending).

    Positional fields follow DESIGN.md; the rest are keyword extras.

    Attributes
    ----------
    z : (nz,)       P(k,z) redshifts (ascending, z[0] = 0).
    k_h : (nk,)     wavenumbers k/h [h/Mpc] (CAMB native nodes, ~194; k_h[-1] = 21/(h Mpc)).
    pk_lin : (nz,nk) total-matter linear P(k,z) [(Mpc/h)^3] (spec s2.4).
    chi : (nzb,)    comoving radial distance [Mpc/h] on ``z_chi``.
    z_chi : (nzb,)  linspace(0, 6, 6000).
    omega_m : float (ombh2 + omch2 + omnuh2)/h^2.
    sigma8_0 : float sigma_8(z=0), total matter.
    h : float       H0/100.
    pk_cb : (nz,nk) cold-matter (CDM+baryon) linear spectrum [(Mpc/h)^3], CAMB ``delta_nonu``.
    ez : (nz,)      E(z) = H(z)/H0 on ``z``.
    omnuh2, ombh2, omch2 : physical densities;  omega_b = ombh2/h^2, omega_c = omch2/h^2 (dimensionless Omega).
    S8 : sigma8_0 * sqrt(omega_m / 0.3).
    n_s, ln_1e10_A_s, mnu : primordial / neutrino inputs.
    """
    z: np.ndarray
    k_h: np.ndarray
    pk_lin: np.ndarray
    chi: np.ndarray
    z_chi: np.ndarray
    omega_m: float
    sigma8_0: float
    h: float
    pk_cb: Optional[np.ndarray] = None
    ez: Optional[np.ndarray] = None
    omnuh2: float = float("nan")
    omega_b: float = float("nan")
    omega_c: float = float("nan")
    S8: float = float("nan")
    n_s: float = float("nan")
    ombh2: float = float("nan")
    omch2: float = float("nan")
    ln_1e10_A_s: float = float("nan")
    mnu: float = 0.06
    _state: Any = field(default=None, repr=False, compare=False)

    @property
    def A_s(self) -> float:
        """Primordial amplitude A_s = exp(ln_1e10_A_s) * 1e-10."""
        return float(np.exp(self.ln_1e10_A_s) * 1e-10)

    @property
    def omega_nu_frac(self) -> float:
        """Neutrino fraction f_nu = omnuh2 / (omega_m h^2)."""
        return float(self.omnuh2 / (self.omega_m * self.h ** 2))

    @property
    def k_mpc(self) -> np.ndarray:
        """Wavenumbers in 1/Mpc, k_h * h."""
        return self.k_h * self.h


class _Entry:
    """One cached CAMB run: the CAMBdata object plus reference spectra and the cosmology-only products."""
    __slots__ = ("results", "As_ref", "ns_ref", "k_h", "z", "pk_ref", "pk_cb_ref", "chi", "ez", "omega_m",
                 "omnuh2", "h", "ombh2", "omch2", "mnu", "state", "sigma8_ref", "lock")


class LinearBackend:
    """CAMB-backed linear background with an exact A_s / n_s rescaling cache.

    Parameters
    ----------
    z_grid : (nz,) array, optional
        Ascending P(k,z) redshifts, must start at 0.  Default :data:`Z_GRID_SPEC` (150 nodes, spec s2.2);
        pass :data:`Z_GRID_REDUCED` for the 41-node grid.  (The grid has negligible effect on CAMB run time.)
    kmax_per_mpc : float
        ``set_matter_power(kmax=...)`` in 1/Mpc (spec s2.2: 20, NOT h/Mpc; CAMB pads by 5 %).
    nthreads : int, optional
        OpenMP threads for CAMB (global ``camb.config.ThreadNum``), default ``min(4, ncpu)``.
    cache_size : int
        Number of CAMB runs kept (LRU).
    key_on_ns : bool
        If True the cache key includes n_s (DESIGN.md wording); default False because the n_s dependence is
        an exact rescale of the primordial spectrum.
    high_precision : bool
        ``Transfer.high_precision`` (spec s2.1: True).  False is ~1.7x faster but changes the k nodes and
        sigma8 at the 1e-4 level.
    accuracy_boost : float
        ``Accuracy.AccuracyBoost`` (CAMB default 1.0).  0.8 is ~20-25 % faster with |Delta chi2| = 0.027 (max over 12
        prior-box points, docs/PERFORMANCE.md); NOT used by default because it breaks bit-compatibility with the
        reference (kcap-like) settings.
    """

    def __init__(self, z_grid=None, kmax_per_mpc: float = 20.0, nthreads: Optional[int] = None,
                 cache_size: int = 8, key_on_ns: bool = False, high_precision: bool = True,
                 accuracy_boost: float = 1.0):
        z = np.asarray(Z_GRID_SPEC if z_grid is None else z_grid, dtype=np.float64)
        if z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0) or z[0] != 0.0:
            raise ValueError("z_grid must be 1-d, strictly ascending and start at z=0")
        self.z_grid = _ro(z)
        self.kmax_per_mpc = float(kmax_per_mpc)
        self.nthreads = default_nthreads() if nthreads is None else int(nthreads)
        self.cache_size = int(cache_size)
        self.key_on_ns = bool(key_on_ns)
        self.high_precision = bool(high_precision)
        self.accuracy_boost = float(accuracy_boost)
        self._cache: "OrderedDict[tuple, _Entry]" = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"camb_runs": 0, "cache_hits": 0, "camb_seconds": 0.0, "compute_calls": 0,
                      "resync_seconds": 0.0}

    # ------------------------------------------------------------------ CAMB plumbing
    @staticmethod
    def _camb():
        import camb  # lazy: importing CAMB costs ~0.4 s
        return camb

    def _make_params(self, omch2, ombh2, h, n_s, As, mnu):
        """CAMBparams with the settings of spec s2.1 (mode transfer)."""
        camb = self._camb()
        p = camb.CAMBparams()
        p.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0, mnu=mnu, num_massive_neutrinos=1)
        p.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="fluid")
        p.InitPower.set_params(As=As, ns=n_s, r=0.0)          # pivot 0.05/Mpc, adiabatic, no running
        p.Reion.Reionization = False                           # spec s2.1: do_reionization = F
        p.WantCls = False                                      # spec s2.1: WantCls False (2e-3 effect otherwise)
        p.DoLensing = False
        p.set_matter_power(redshifts=[float(x) for x in self.z_grid[::-1]],   # CAMB wants descending
                           kmax=self.kmax_per_mpc, nonlinear=False)
        p.Transfer.high_precision = self.high_precision
        if self.accuracy_boost != 1.0:
            p.Accuracy.AccuracyBoost = self.accuracy_boost
        return p

    def _key(self, omch2, ombh2, h, n_s, mnu):
        return (float(omch2), float(ombh2), float(h), float(mnu)) + ((float(n_s),) if self.key_on_ns else ())

    def _run_camb(self, omch2, ombh2, h, n_s, ln_As, mnu) -> _Entry:
        camb = self._camb()
        As = float(np.exp(ln_As) * 1e-10)
        t0 = perf_counter()
        camb.config.ThreadNum = self.nthreads
        p = self._make_params(omch2, ombh2, h, n_s, As, mnu)
        r = camb.get_results(p)
        kh, zz, pk = r.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
        _, _, pcb = r.get_linear_matter_power_spectrum(var1="delta_nonu", var2="delta_nonu",
                                                       hubble_units=True, k_hunit=True)
        zz = np.asarray(zz)
        if not (np.isfinite(pk).all() and np.isfinite(pcb).all() and (pk > 0).all()):
            raise RuntimeError("CAMB returned a non-finite / non-positive linear power spectrum")   # never cache it
        if zz[0] > zz[-1]:                       # defensive: CAMB returns ascending in 1.5.8
            pk, pcb, zz = pk[::-1], pcb[::-1], zz[::-1]
        if zz.shape != self.z_grid.shape or not np.allclose(zz, self.z_grid, rtol=0, atol=1e-12):
            raise RuntimeError("CAMB returned an unexpected redshift grid")
        e = _Entry()
        e.results = r
        e.As_ref, e.ns_ref = As, float(n_s)
        e.k_h = _ro(kh)
        e.z = self.z_grid
        e.pk_ref = _ro(pk)
        e.pk_cb_ref = _ro(pcb)
        hh = p.h
        # distances: CAMB integrates 1/H exactly (agrees with scipy quad to ~1e-14); chi in Mpc -> Mpc/h
        e.chi = _ro(r.comoving_radial_distance(np.asarray(Z_CHI), tol=1e-8) * hh)
        Hz = r.h_of_z(np.asarray(self.z_grid))
        e.ez = _ro(Hz / r.h_of_z(0.0))
        e.omega_m = float(p.omegam)
        e.omnuh2 = float(p.omnuh2)
        e.h, e.ombh2, e.omch2, e.mnu = float(hh), float(ombh2), float(omch2), float(mnu)
        e.sigma8_ref = float(r.get_sigma8_0())
        e.state = (As, float(n_s), None)         # (A_s, n_s, A_baryon or None) currently loaded in CAMB
        e.lock = threading.RLock()
        self.stats["camb_runs"] += 1
        self.stats["camb_seconds"] += perf_counter() - t0
        return e

    # ------------------------------------------------------------------ public API
    @staticmethod
    def _validate(omch2, ombh2, h, n_s, ln_1e10_A_s, mnu):
        """Reject inputs that would crash CAMB (NaN physical densities corrupt the Fortran heap, negative ones
        segfault/stop), be silently ignored (NaN / negative ``mnu`` is treated as 0), or poison the cache
        (a NaN ``A_s`` on the first call of a cosmology would be stored as its reference amplitude)."""
        try:
            v = [float(x) for x in (omch2, ombh2, h, n_s, ln_1e10_A_s, mnu)]
        except (TypeError, ValueError) as exc:           # arrays of size > 1, None, strings
            raise ValueError("compute() arguments must be real scalars") from exc
        names = ("omch2", "ombh2", "h", "n_s", "ln_1e10_A_s", "mnu")
        for nm, x in zip(names, v):
            if not np.isfinite(x):
                raise ValueError(f"{nm} must be finite, got {x}")
        omch2, ombh2, h, n_s, ln_As, mnu = v
        if omch2 <= 0 or ombh2 <= 0:
            raise ValueError(f"omch2 and ombh2 must be > 0, got omch2={omch2}, ombh2={ombh2}")
        if not 0.1 <= h <= 3.0:
            raise ValueError(f"h must be in [0.1, 3] (H0 = 100 h km/s/Mpc), got {h}")
        if abs(n_s) > 10.0:
            raise ValueError(f"|n_s| must be <= 10 (P ~ k^(n_s-1) overflows / is unphysical), got {n_s}")
        if mnu < 0:
            raise ValueError(f"mnu must be >= 0 eV, got {mnu}")
        if not np.isfinite(np.exp(ln_As) * 1e-10) or np.exp(ln_As) * 1e-10 <= 0:
            raise ValueError(f"ln_1e10_A_s={ln_As} gives a non-representable A_s")
        return omch2, ombh2, h, n_s, ln_As, mnu

    def compute(self, omch2, ombh2, h, n_s, ln_1e10_A_s, mnu: float = 0.06) -> LinearResult:
        """Linear background and spectra for the given cosmology (spec s2.1-s2.4).

        Parameters: ``omch2, ombh2`` physical densities, ``h`` = H0/100, ``n_s``, ``ln_1e10_A_s`` =
        ln(10^10 A_s), ``mnu`` [eV].  Returns a :class:`LinearResult` (arrays are shared read-only views
        for cosmology-only quantities; ``pk_lin``/``pk_cb`` are fresh arrays).
        """
        omch2, ombh2, h, n_s, ln_1e10_A_s, mnu = self._validate(omch2, ombh2, h, n_s, ln_1e10_A_s, mnu)
        self.stats["compute_calls"] += 1
        key = self._key(omch2, ombh2, h, n_s, mnu)
        with self._lock:
            e = self._cache.get(key)
            if e is None:
                e = self._run_camb(omch2, ombh2, h, n_s, ln_1e10_A_s, mnu)
                self._cache[key] = e
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
                self.stats["cache_hits"] += 1
        As = float(np.exp(ln_1e10_A_s) * 1e-10)
        ns = float(n_s)
        fac = As / e.As_ref
        if ns != e.ns_ref:
            tilt = (e.k_h * e.h / K_PIVOT_MPC) ** (ns - e.ns_ref)       # (nk,)
            pk = e.pk_ref * (fac * tilt)[None, :]
            pcb = e.pk_cb_ref * (fac * tilt)[None, :]
        else:
            pk = e.pk_ref * fac
            pcb = e.pk_cb_ref * fac
        if (As, ns) == (e.As_ref, e.ns_ref):
            s8 = e.sigma8_ref
        elif ns == e.ns_ref:
            s8 = e.sigma8_ref * np.sqrt(fac)                               # exact
        else:
            t0 = perf_counter()
            s8 = self._sigma8_via_camb(e, As, ns)                          # exact; 0.3 ms
            self.stats["resync_seconds"] += perf_counter() - t0
        if not (np.isfinite(s8) and np.isfinite(pk).all() and np.isfinite(pcb).all()):
            raise ValueError("non-finite linear power spectrum (n_s / ln_1e10_A_s out of the representable range): "
                             f"n_s={ns}, ln_1e10_A_s={ln_1e10_A_s}")
        om = e.omega_m
        st = _State(self, e, As, ns)
        return LinearResult(
            z=e.z, k_h=e.k_h, pk_lin=pk, chi=e.chi, z_chi=_Z_CHI_RO, omega_m=om, sigma8_0=float(s8), h=e.h,
            pk_cb=pcb, ez=e.ez, omnuh2=e.omnuh2, omega_b=e.ombh2 / e.h ** 2, omega_c=e.omch2 / e.h ** 2,
            S8=float(s8 * np.sqrt(om / 0.3)), n_s=ns, ombh2=e.ombh2, omch2=e.omch2,
            ln_1e10_A_s=float(ln_1e10_A_s), mnu=e.mnu, _state=st)

    def clear_cache(self) -> None:
        """Drop all cached CAMB runs."""
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ CAMB in-place re-powering
    def _apply_state(self, e: _Entry, As: float, ns: float, nl_model=None):
        """Load (A_s, n_s[, non-linear model]) into the CAMB result object and redo transfer->power."""
        camb = self._camb()
        from camb import model
        with e.lock:
            want = (As, ns, None if nl_model is None else (nl_model["A"], nl_model["eta"]))
            if e.state == want:
                return
            camb.config.ThreadNum = self.nthreads
            P = e.results.Params
            P.InitPower.set_params(As=As, ns=ns, r=0.0)
            if nl_model is None:
                P.NonLinear = model.NonLinear_none
            else:
                from camb.nonlinear import Halofit
                P.NonLinear = model.NonLinear_both
                P.NonLinearModel = Halofit(halofit_version="mead", HMCode_A_baryon=float(nl_model["A"]),
                                           HMCode_eta_baryon=float(nl_model["eta"]))
            e.results.calc_power_spectra()
            e.state = want

    def _sigma8_via_camb(self, e: _Entry, As: float, ns: float) -> float:
        with e.lock:
            self._apply_state(e, As, ns, None)
            return float(e.results.get_sigma8_0())


_Z_CHI_RO = _ro(Z_CHI)


class _State:
    """Handle that lets ``nonlinear.get_pnl(..., backend='camb')`` reuse the cached CAMB run."""
    __slots__ = ("backend", "entry", "As", "ns")

    def __init__(self, backend: LinearBackend, entry: _Entry, As: float, ns: float):
        self.backend, self.entry, self.As, self.ns = backend, entry, As, ns

    def camb_nonlinear(self, A_baryon: float, eta_baryon: float) -> np.ndarray:
        """HMcode-2016 P_nl(z,k) [(Mpc/h)^3] on the linear grid via CAMB (Halofit 'mead')."""
        be, e = self.backend, self.entry
        with e.lock:
            be._apply_state(e, self.As, self.ns, {"A": A_baryon, "eta": eta_baryon})
            _, _, pnl = e.results.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
        pnl = np.asarray(pnl)
        if pnl.shape != e.pk_ref.shape:
            raise RuntimeError("CAMB non-linear grid differs from the linear grid")
        return pnl
