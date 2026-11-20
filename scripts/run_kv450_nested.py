"""Nautilus nested sampling of the full 14-D KV450 posterior with the emulator, prior truncated to the emulator box
(omch2 in [0.03, 0.34]; ombh2, h boxes of the emulator coincide with the prior bounds).
ln Z is for the box-truncated prior; to refer it to the full prior add ln(0.31/0.98) (flat omch2 prior) and the (MCMC-estimated,
negligible) posterior mass outside the box.
usage: run_kv450_nested.py --workers 12 [--z-grid reduced] --n-live 500 --n-eff 2000 --tag nautilus1"""
import argparse
import json
import multiprocessing as mp
import time
import numpy as np
import run_kv450_common as C
from run_kv450_common import params, PARAM_NAMES, IDX
from kidspipe.inference.nested import run_nautilus

_post = None
OM_LO, OM_HI = 0.03, 0.34


def _init(z_grid):
    global _post
    _post = C.make_post(threads=1, z_grid=z_grid)


def loglike(theta):
    return _post.log_like(np.asarray(theta))


def prior_transform(u):
    th = params.prior_transform(np.asarray(u, float))
    th[0] = OM_LO + float(u[0]) * (OM_HI - OM_LO)
    return th


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(12, C.NCPU), help="worker processes (default min(12, os.cpu_count()))")
    ap.add_argument("--z-grid", default="spec")
    ap.add_argument("--n-live", type=int, default=500)
    ap.add_argument("--n-eff", type=float, default=2000)
    ap.add_argument("--n-like-max", type=float, default=1e9)
    ap.add_argument("--timeout", type=float, default=1e9, help="seconds")
    ap.add_argument("--tag", default="nautilus1")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = C.OUT / a.tag
    out.mkdir(exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.z_grid,)) as pool:
        r = run_nautilus(loglike, prior_transform, 14, n_live=a.n_live, n_eff=a.n_eff, f_live=0.01, seed=a.seed, pool=pool,
                         filepath=str(out / "nautilus.hdf5"), resume=True, verbose=True, n_like_max=a.n_like_max, timeout=a.timeout)
    post = C.make_post(threads=1, z_grid=a.z_grid)
    from kidspipe import api
    D = api.derived_columns(post, r.samples)
    np.savez(out / "nested.npz", samples=r.samples, weights=r.weights, log_like=r.log_like, derived=D)
    C.jdump(out / "nested_summary.json", dict(log_z_box=r.log_z, n_eval=r.n_eval, ess=r.ess, wall=time.time() - t0, info=r.info,
                                              log_z_full_estimate=r.log_z + np.log((OM_HI - OM_LO) / 0.98), z_grid=a.z_grid))
    w = r.weights
    s8 = D[:, 2]
    m = w @ s8
    sd = np.sqrt(w @ (s8 - m) ** 2)
    print(f"logZ(box) = {r.log_z:.3f}  n_eval {r.n_eval}  ESS {r.ess:.0f}  S8 = {m:.4f} +- {sd:.4f}  wall {time.time()-t0:.0f}s")
