"""Staged, cached KV450 cosmic-shear model: parameter vector -> 195-point theory vector (spec s12).

Stages and what invalidates them (so cheap parameter moves skip expensive work):

    1  linear P(k,z), distances    omch2, ombh2, h  (A_s and n_s are exact rescalings, see LinearBackend)
    2  P_nl + high-k extension +   + n_s, ln_1e10_A_s, A_baryon        (Limber look-up Q[chi, l])
       Limber power look-up
    3  n(z) shifts, NLA, C_ell     + dz1..dz5, A_IA
    4  Hankel + theta binning      (linear operator on C_ell, always recomputed: microseconds)

``mu`` returns the PURE binned theory (no delta_c / A_c terms: ``GaussianLikelihood`` adds those, see
docs/interface_notes/params_data_likelihood.md).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from ..likelihood.data import assemble_vector
from ..params import PARAM_NAMES
from .hankel import HankelBinner
from .limber import Limber, default_ell
from .linear import LinearBackend, Z_GRID_REDUCED, Z_GRID_SPEC
from .nonlinear import extend_high_k, get_pnl
from .nz import load_nz

_IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
_I_COSMO = [_IDX[n] for n in ("omch2", "ombh2", "h", "n_s", "ln_1e10_A_s")]
_I_ABARY = _IDX["A_baryon"]
_I_AIA = _IDX["A_IA"]
_I_DZ = [_IDX[f"dz{i}"] for i in range(1, 6)]


class CosmicShearModel:
    """Parameter vector (14,) -> binned (xi+, xi-) theory vector (195,).

    Parameters
    ----------
    fidelity : ``"fast"`` (exact bin-averaged Bessel operator, spec E forms) or ``"kcap"`` (FFT Hankel +
        node-trapezoid binning, spec F forms).
    pnl_backend : ``"hmcode"`` (stand-alone numba HMcode-2016) or ``"camb"`` (CAMB 'mead').
    z_grid : ``"spec"`` (150 nodes, spec s2.2) or ``"reduced"`` (41 nodes, Delta chi2 < 1e-3 per spec s13 D4).
    n_ell : number of log-spaced ell nodes (spec: 400 on 0.1..5e5; 120 on 1..3e5 changes chi2 by 0.01).
    n_nodes : Limber nodes in z (spec s15.4: 2000 -> 1.5e-4).
    linear_backend : ``"camb"`` (exact CAMB, default), ``"emulator"`` (``EmulatedLinearBackend``: transfer emulator
        inside its training box, automatic exact-CAMB fallback outside) or a backend instance with ``.compute``.
    """

    def __init__(self, fidelity: str = "fast", pnl_backend: str = "hmcode", z_grid: str = "spec",
                 n_ell: int = 400, n_nodes: int = 2000, nthreads: int | None = None, cache_size: int = 4,
                 data_dir=None, linear_backend="camb"):
        zg = {"spec": Z_GRID_SPEC, "reduced": Z_GRID_REDUCED}[z_grid]
        self.fidelity, self.pnl_backend = fidelity, pnl_backend
        if linear_backend == "camb":
            self.linear = LinearBackend(z_grid=zg, nthreads=nthreads)
        elif linear_backend == "emulator":      # transfer emulator inside its box, exact CAMB outside (theory/emulator.py)
            from .emulator import EmulatedLinearBackend
            self.linear = EmulatedLinearBackend(z_grid=zg, nthreads=nthreads)
        elif hasattr(linear_backend, "compute"):   # a ready-made backend instance
            self.linear = linear_backend
        else:
            raise ValueError(f"linear_backend must be 'camb', 'emulator' or an object with .compute, got {linear_backend!r}")
        self.nz = load_nz(data_dir)
        self.ell = default_ell(n_ell) if n_ell != 400 else default_ell()
        self.binner = HankelBinner(self.ell, fidelity=fidelity)
        self._n_nodes = n_nodes
        self._cache_size = cache_size
        self._lock = threading.RLock()              # guards the stage caches (review: model shared between threads)
        self._tls = threading.local()               # per-thread last_derived
        self._stage2: OrderedDict = OrderedDict()   # key -> (Limber, derived dict)
        self._wc = {}                               # n(z)-shift cache for dz tuple
        self.t = {"linear": 0.0, "pnl": 0.0, "limber_power": 0.0, "limber_cl": 0.0, "hankel": 0.0}
        self.n_calls = 0
        self.last_derived = {}

    @property
    def last_derived(self) -> dict:
        """sigma8 / omega_m / S8 of the cosmology of the last ``mu`` call *in this thread* (thread-local: a plain
        attribute was overwritten by other threads sharing the model, found in review)."""
        return getattr(self._tls, "derived", {})

    @last_derived.setter
    def last_derived(self, value: dict) -> None:
        self._tls.derived = value

    # ------------------------------------------------------------------
    def _stage2_get(self, theta):
        with self._lock:
            return self._stage2_get_locked(theta)

    def _stage2_get_locked(self, theta):
        key = tuple(float(theta[i]) for i in _I_COSMO) + (float(theta[_I_ABARY]),)
        hit = self._stage2.get(key)
        if hit is not None:
            self._stage2.move_to_end(key)
            return hit
        t0 = time.perf_counter()
        lin = self.linear.compute(*[float(theta[i]) for i in _I_COSMO])
        t1 = time.perf_counter()
        pnl = get_pnl(lin, float(theta[_I_ABARY]), backend=self.pnl_backend)
        k_ext, pnl_ext = extend_high_k(lin.k_h, pnl)
        t2 = time.perf_counter()
        lim = Limber(self.ell, fidelity=self.fidelity, n_nodes=self._n_nodes)
        lim.set_power(lin.chi, lin.z_chi, lin.omega_m, lin.z, k_ext, pnl_ext)
        lim.set_nz_grid(self.nz.z)
        t3 = time.perf_counter()
        self.t["linear"] += t1 - t0
        self.t["pnl"] += t2 - t1
        self.t["limber_power"] += t3 - t2
        derived = {"sigma8": lin.sigma8_0, "omega_m": lin.omega_m, "S8": lin.S8}
        self._stage2[key] = (lim, derived)
        while len(self._stage2) > self._cache_size:
            self._stage2.popitem(last=False)
        return lim, derived

    def mu(self, theta) -> np.ndarray:
        """Binned theory vector (195,) in the data ordering, without delta_c / A_c terms."""
        theta = np.asarray(theta, dtype=np.float64)
        self.n_calls += 1
        lim, self.last_derived = self._stage2_get(theta)
        dz = tuple(float(theta[i]) for i in _I_DZ)
        with self._lock:
            nzs = self._wc.get(dz)
            if nzs is None:
                nzs = self.nz.shifted(np.array(dz))
                if len(self._wc) > 64:
                    self._wc.clear()
                self._wc[dz] = nzs
        t0 = time.perf_counter()
        C = lim.cl(nzs, float(theta[_I_AIA]))
        t1 = time.perf_counter()
        xip, xim = self.binner.xi_binned(C)
        out = assemble_vector(xip, xim)
        t2 = time.perf_counter()
        self.t["limber_cl"] += t1 - t0
        self.t["hankel"] += t2 - t1
        return out

    def cl(self, theta):
        """Limber convergence spectrum for parameter vector ``theta``: ``(ell (nell,), C (5, 5, nell))``.

        Same staged caches and n(z)-shift / NLA handling as :meth:`mu` (which is untouched); used by the COSEBIs
        route (``theory/cosebis.py``).  Returns copies.  delta_c / A_c do not enter C_ell.
        """
        theta = np.asarray(theta, dtype=np.float64)
        lim, self.last_derived = self._stage2_get(theta)
        dz = tuple(float(theta[i]) for i in _I_DZ)
        with self._lock:
            nzs = self._wc.get(dz)
            if nzs is None:
                nzs = self.nz.shifted(np.array(dz))
                if len(self._wc) > 64:
                    self._wc.clear()
                self._wc[dz] = nzs
        C = lim.cl(nzs, float(theta[_I_AIA]))
        return self.ell.copy(), C

    __call__ = mu

    def timing(self) -> dict:
        """Cumulative seconds per stage (+ call count)."""
        d = dict(self.t)
        d["n_calls"] = self.n_calls
        return d

    def reset_timing(self):
        for k in self.t:
            self.t[k] = 0.0
        self.n_calls = 0
