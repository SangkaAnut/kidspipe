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

from ..likelihood.data import N_DATA, assemble_vector
from ..params import N_PARAM, PARAM_NAMES
from .hankel import _PI, _PJ, HankelBinner
from .hmcode import hmcode_pnl_batch
from .limber import Limber, _build_limber_batch_fast, _eval_cl_batch_fast, default_ell
from .linear import LinearBackend, Z_GRID_REDUCED, Z_GRID_SPEC
from .nonlinear import eta_baryon, extend_high_k, extend_high_k_batch, get_pnl
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

    def _make_limber_from_block(self, lin, lim_blk: dict, b_idx: int) -> Limber:
        """Materialize a :class:`Limber` object from a precomputed batch block for ``_stage2`` caching."""
        lim = Limber(self.ell, fidelity=self.fidelity, n_nodes=self._n_nodes)
        lim.omega_m = float(lin.omega_m)
        lim.s = float(lim_blk["s_b"][b_idx])
        lim.z_max = float(lim_blk["z_max"])
        lim._zf = lim_blk["zf"]
        lim._chif = lim_blk["chif_b"][b_idx]
        lim.z = lim_blk["z_nodes"]
        lim.dz = float(lim_blk["dz"])
        lim.chi = lim_blk["chi_b"][b_idx]
        lim.dchidz = lim_blk["dchidz_b"][b_idx]
        lim.w = lim_blk["w_nodes"]
        lim.inv_chi = lim_blk["inv_chi_b"][b_idx]
        lim.D = lim_blk["D_b"][b_idx]
        lim.Qt = lim_blk["Qt_b"][b_idx].copy()
        lim._nz_cache = None
        lim._ready = True
        lim.set_nz_grid(self.nz.z)
        return lim

    def mu_batch(self, thetas, *, block_size: int = 32, limber_block_size: int = 16) -> np.ndarray:
        """Batched binned theory vectors ``(M, 195)`` for ``thetas`` of shape ``(M, 14)`` or ``(14,)``.

        De-duplicates rows by stage-2 key ``(omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon)``, runs
        batched linear emulation, batched HMcode ``prange(B * nz)``, batched Limber ``Qt``/``C_ell``
        in blocks of ``limber_block_size`` to bound memory, and one BLAS-3 Hankel matmul.
        Thread-safe and eviction-safe: entries required inside the batch are pinned locally so
        ``cache_size`` eviction never drops an entry still needed by another row of the same batch.
        """
        th = np.asarray(thetas, dtype=np.float64)
        if th.ndim == 1:
            if th.size != N_PARAM:
                raise ValueError(f"mu_batch expects shape (M, {N_PARAM}), got {th.shape}")
            th = th[None, :]
        elif th.ndim != 2 or th.shape[1] != N_PARAM:
            raise ValueError(f"mu_batch expects shape (M, {N_PARAM}), got {th.shape}")
        m = th.shape[0]
        if m == 0:
            return np.empty((0, N_DATA), dtype=np.float64)

        # 1. De-duplicate stage-2 keys and photo-z shifts across rows
        s2_keys: list[tuple] = []
        s2_index: dict[tuple, int] = {}
        row_s2 = np.empty(m, dtype=np.int64)
        row_dz: list[tuple] = []
        unique_dz: dict[tuple, np.ndarray] = {}

        for r in range(m):
            key = tuple(float(th[r, i]) for i in _I_COSMO) + (float(th[r, _I_ABARY]),)
            uid = s2_index.get(key)
            if uid is None:
                uid = len(s2_keys)
                s2_index[key] = uid
                s2_keys.append(key)
            row_s2[r] = uid
            dz = tuple(float(th[r, i]) for i in _I_DZ)
            row_dz.append(dz)
            if dz not in unique_dz:
                unique_dz[dz] = None  # type: ignore[assignment]

        # 2. Check existing stage-2 and n(z) caches into local dicts (eviction-proof)
        u_total = len(s2_keys)
        cached_s2: dict[int, tuple[Limber, dict]] = {}
        miss_uids: list[int] = []
        with self._lock:
            for uid, key in enumerate(s2_keys):
                hit = self._stage2.get(key)
                if hit is not None:
                    self._stage2.move_to_end(key)
                    cached_s2[uid] = hit
                else:
                    miss_uids.append(uid)
            for dz in list(unique_dz):
                nzs = self._wc.get(dz)
                if nzs is not None:
                    unique_dz[dz] = nzs
        for dz, val in list(unique_dz.items()):
            if val is None:
                unique_dz[dz] = self.nz.shifted(np.array(dz))

        # Group row indices by stage-2 uid
        uid_rows: list[list[int]] = [[] for _ in range(u_total)]
        for r in range(m):
            uid_rows[int(row_s2[r])].append(r)

        nell = self.ell.size
        n_pair = _PJ.size
        C15_all = np.empty((m, n_pair, nell), dtype=np.float64)
        derived_by_uid: dict[int, dict] = {uid: cached_s2[uid][1] for uid in cached_s2}
        new_cache_entries: dict[int, tuple[tuple, Limber, dict]] = {}

        # 3. Evaluate cached stage-2 hits
        t_cl_acc = 0.0
        for uid, (lim, _der) in cached_s2.items():
            t0 = time.perf_counter()
            for r in uid_rows[uid]:
                C = lim.cl(unique_dz[row_dz[r]], float(th[r, _I_AIA]))
                C15_all[r] = C[_PJ, _PI]
            t_cl_acc += time.perf_counter() - t0

        # 4. Compute missing unique cosmologies in batch
        if miss_uids:
            miss_cosmo_5 = np.array([s2_keys[uid][:5] for uid in miss_uids], dtype=np.float64)
            miss_ab = np.array([s2_keys[uid][5] for uid in miss_uids], dtype=np.float64)
            t0 = time.perf_counter()
            if hasattr(self.linear, "compute_batch"):
                lins = self.linear.compute_batch(miss_cosmo_5)
            else:
                lins = [self.linear.compute(*row) for row in miss_cosmo_5]
            t1 = time.perf_counter()

            # Stage 2a: P_nl
            n_miss = len(miss_uids)
            pnls: list[np.ndarray] = [None] * n_miss  # type: ignore[list-item]
            if self.pnl_backend == "hmcode":
                groups: dict[tuple[int, int], list[int]] = {}
                for j, lin in enumerate(lins):
                    gk = (lin.k_h.size, lin.z.size)
                    groups.setdefault(gk, []).append(j)
                for _gk, idxs in groups.items():
                    k_b = np.stack([lins[j].k_h for j in idxs])
                    pk_b = np.stack([lins[j].pk_lin for j in idxs])
                    pcb_b = np.stack([lins[j].pk_cb for j in idxs])
                    om_b = np.array([lins[j].omega_m for j in idxs], dtype=np.float64)
                    fnu_b = np.array([lins[j].omega_nu_frac for j in idxs], dtype=np.float64)
                    h_b = np.array([lins[j].h for j in idxs], dtype=np.float64)
                    ns_b = np.array([lins[j].n_s for j in idxs], dtype=np.float64)
                    ab_b = np.array([miss_ab[j] for j in idxs], dtype=np.float64)
                    eta_b = np.array([eta_baryon(miss_ab[j]) for j in idxs], dtype=np.float64)
                    out_pnl = hmcode_pnl_batch(
                        lins[idxs[0]].z, k_b, pk_b, om_b, fnu_b, h_b, ns_b, ab_b, eta_b,
                        pk_cb_batch=pcb_b, block_size=block_size,
                    )
                    for pos, j in enumerate(idxs):
                        pnls[j] = out_pnl[pos]
            else:
                for j, lin in enumerate(lins):
                    pnls[j] = get_pnl(lin, float(miss_ab[j]), backend=self.pnl_backend)
            t2 = time.perf_counter()

            keep_start = max(0, n_miss - self._cache_size) if self._cache_size > 0 else n_miss
            for j, uid in enumerate(miss_uids):
                lin = lins[j]
                derived_by_uid[uid] = {"sigma8": lin.sigma8_0, "omega_m": lin.omega_m, "S8": lin.S8}

            # Stage 2b/2c + Stage 3 in bounded Limber blocks
            t_lim_pow = 0.0
            if self.fidelity == "fast":
                fast_groups: dict[tuple[int, int, int], list[int]] = {}
                scalar_idxs: list[int] = []
                for j, lin in enumerate(lins):
                    if lin.k_h.size >= 4:
                        fk = (lin.k_h.size, lin.z.size, lin.z_chi.size)
                        fast_groups.setdefault(fk, []).append(j)
                    else:
                        scalar_idxs.append(j)

                lbs = max(1, int(limber_block_size))
                qt_buf = np.empty((min(n_miss, lbs), self._n_nodes, nell), dtype=np.float64)
                for _fk, idxs in fast_groups.items():
                    z_tab = lins[idxs[0]].z_chi
                    z_pk = lins[idxs[0]].z
                    for s in range(0, len(idxs), lbs):
                        blk_js = idxs[s:s + lbs]
                        tp0 = time.perf_counter()
                        k_b = np.stack([lins[j].k_h for j in blk_js])
                        pnl_b = np.stack([pnls[j] for j in blk_js])
                        k_ext_b, pnl_ext_b = extend_high_k_batch(k_b, pnl_b)
                        chi_tab_b = np.stack([lins[j].chi for j in blk_js])
                        om_b = np.array([lins[j].omega_m for j in blk_js], dtype=np.float64)
                        lim_blk = _build_limber_batch_fast(
                            chi_tab_b, z_tab, om_b, z_pk, k_ext_b, pnl_ext_b,
                            self.ell, n_nodes=self._n_nodes, Qt_buf=qt_buf,
                        )
                        tp1 = time.perf_counter()
                        t_lim_pow += tp1 - tp0

                        blk_rows: list[int] = []
                        blk_cids: list[int] = []
                        for b_idx, j in enumerate(blk_js):
                            uid = miss_uids[j]
                            if j >= keep_start:
                                lim_obj = self._make_limber_from_block(lins[j], lim_blk, b_idx)
                                new_cache_entries[uid] = (s2_keys[uid], lim_obj, derived_by_uid[uid])
                            for r in uid_rows[uid]:
                                blk_rows.append(r)
                                blk_cids.append(b_idx)

                        if blk_rows:
                            tc0 = time.perf_counter()
                            nzs_b = np.stack([unique_dz[row_dz[r]] for r in blk_rows])
                            a_ia_b = np.array([float(th[r, _I_AIA]) for r in blk_rows], dtype=np.float64)
                            c15_blk = np.empty((len(blk_rows), n_pair, nell), dtype=np.float64)
                            _eval_cl_batch_fast(
                                lim_blk, np.asarray(blk_cids, dtype=np.int64), nzs_b,
                                self.nz.z, a_ia_b, om_b, c15_blk,
                            )
                            C15_all[blk_rows] = c15_blk
                            t_cl_acc += time.perf_counter() - tc0
            else:
                scalar_idxs = list(range(n_miss))

            for j in scalar_idxs:
                uid = miss_uids[j]
                lin = lins[j]
                tp0 = time.perf_counter()
                k_ext, pnl_ext = extend_high_k(lin.k_h, pnls[j])
                lim = Limber(self.ell, fidelity=self.fidelity, n_nodes=self._n_nodes)
                lim.set_power(lin.chi, lin.z_chi, lin.omega_m, lin.z, k_ext, pnl_ext)
                lim.set_nz_grid(self.nz.z)
                t_lim_pow += time.perf_counter() - tp0
                if j >= keep_start:
                    new_cache_entries[uid] = (s2_keys[uid], lim, derived_by_uid[uid])
                tc0 = time.perf_counter()
                for r in uid_rows[uid]:
                    C = lim.cl(unique_dz[row_dz[r]], float(th[r, _I_AIA]))
                    C15_all[r] = C[_PJ, _PI]
                t_cl_acc += time.perf_counter() - tc0

            self.t["linear"] += t1 - t0
            self.t["pnl"] += t2 - t1
            self.t["limber_power"] += t_lim_pow

        # 5. Stage 4: batched Hankel + assembly
        th0 = time.perf_counter()
        xip, xim = self.binner.xi_binned_batch(C15_all)
        out = assemble_vector(xip, xim)
        t_hk = time.perf_counter() - th0

        # 6. Thread-safe cache update + stats + thread-local derived
        derived_per_row = [derived_by_uid[int(row_s2[r])] for r in range(m)]
        with self._lock:
            if self._cache_size > 0:
                for uid in miss_uids[-self._cache_size:]:
                    entry = new_cache_entries.get(uid)
                    if entry is not None:
                        key, lim_obj, der = entry
                        self._stage2[key] = (lim_obj, der)
                        self._stage2.move_to_end(key)
                        while len(self._stage2) > self._cache_size:
                            self._stage2.popitem(last=False)
            for dz, nzs in unique_dz.items():
                if dz not in self._wc:
                    if len(self._wc) > 64:
                        self._wc.clear()
                    self._wc[dz] = nzs
        self.last_derived = derived_per_row[-1]
        self._tls.last_derived_batch = derived_per_row
        self.n_calls += m
        self.t["limber_cl"] += t_cl_acc
        self.t["hankel"] += t_hk
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
