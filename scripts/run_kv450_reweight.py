"""Importance re-weighting of thinned production samples to a model variant (kcap fidelity, CAMB-mead backend + exact CAMB linear,
41-node z grid, ...).   w_i = exp(lnL_variant(theta_i) - lnL_base(theta_i)),  both evaluated here (base = emulator+hmcode+fast+spec).
usage: run_kv450_reweight.py --tags prod1,prod2 --variants kcap,reduced,camb_mead --n 600 --workers 6
writes runs/kv450_final/reweight.json"""
import argparse
import glob
import multiprocessing as mp
import numpy as np
import run_kv450_common as C

VAR = {"kcap": dict(fidelity="kcap"), "reduced": dict(z_grid="reduced"), "camb_mead": dict(linear="camb", backend="camb"),
       "exact_linear": dict(linear="camb"), "hmcode_exactlin": dict(linear="camb")}
_b = _v = None


def _init(variant):
    global _b, _v
    _b = C.make_post(threads=1)
    _v = C.make_post(threads=1, **VAR[variant])


def _one(th):
    th = np.asarray(th)
    return float(_b.log_like(th)), float(_v.log_like(th))


_VARS: dict = {}


def _init_mpi():
    global _b
    _b = C.make_post(threads=1)


def _one_mpi(job):
    """(variant, theta) on an MPI worker; the variant model is built on first use (variants run one after another, so
    the previous variant's model is dropped)."""
    v, th = job
    if v not in _VARS:
        _VARS.clear()
        _VARS[v] = C.make_post(threads=1, **VAR[v])
    th = np.asarray(th)
    return float(_b.log_like(th)), float(_VARS[v].log_like(th))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="prod1,prod2")
    ap.add_argument("--variants", default="kcap,reduced")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=min(6, C.NCPU), help="worker processes (default min(6, os.cpu_count()))")
    ap.add_argument("--burn-frac", type=float, default=0.1)
    ap.add_argument("--mpi", action="store_true", help="evaluate over MPI ranks: `mpirun -n N python scripts/run_kv450_reweight.py "
                    "--mpi ...`; rank 0 schedules and writes, ranks 1..N-1 evaluate; --workers ignored")
    a = ap.parse_args()
    import sys
    from kidspipe import mpi
    if not a.mpi:
        mpi.refuse_unflagged_launch("--mpi")
    _nw = mpi.require_workers(1) if a.mpi else 0
    _cm = mpi.executor(_init_mpi, (), threads=1) if a.mpi else None
    _ex = _cm.__enter__() if _cm is not None else None
    if a.mpi and _ex is None:                          # worker rank: rank 0 has finished
        _cm.__exit__(None, None, None)
        sys.exit(0)
    if a.mpi:
        _pool = mpi.Pool(_ex, _nw)
    try:
        S, D = [], []
        for t in a.tags.split(","):
            for f in C.result_glob(f"{t}/prod_c*.npz"):
                z = np.load(f)
                b = int(a.burn_frac * len(z["samples"]))
                S.append(z["samples"][b:])
                D.append(z["derived"][b:])
        S, D = np.concatenate(S), np.concatenate(D)
        rng = np.random.default_rng(3)
        idx = rng.choice(len(S), a.n, replace=False)
        S, D = S[idx], D[idx]
        out = {}
        for v in a.variants.split(","):
            if a.mpi:
                ll = np.array(_pool.map(_one_mpi, [(v, th) for th in S.tolist()], chunksize=2))
            else:
                with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(v,)) as pool:
                    ll = np.array(pool.map(_one, S.tolist(), chunksize=2))
            dl = ll[:, 1] - ll[:, 0]
            w = np.exp(dl - dl.max())
            w /= w.sum()
            s8 = D[:, 2]
            m0, m1 = s8.mean(), w @ s8
            sd1 = np.sqrt(w @ (s8 - m1) ** 2)
            out[v] = dict(n=a.n, mean_dchi2=float(np.mean(-2 * dl)), sd_dchi2=float(np.std(-2 * dl)), max_abs_dchi2=float(np.abs(2 * dl).max()),
                          S8_unweighted=float(m0), S8_reweighted=float(m1), S8_sd_reweighted=float(sd1), ess_w=float(1 / np.sum(w ** 2)),
                          omega_m_shift=float(w @ D[:, 1] - D[:, 1].mean()), sigma8_shift=float(w @ D[:, 0] - D[:, 0].mean()))
            print(v, out[v], flush=True)
        C.jdump(C.OUT / "reweight.json", out)
    except BaseException:
        if _cm is not None:                            # MPI root: cancel queued tasks, stop the workers
            _cm.__exit__(*sys.exc_info())
            _cm = None
        raise
    if _cm is not None:
        _cm.__exit__(None, None, None)
