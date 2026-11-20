"""Global MAP search: many diverse starts with the emulator, exact-CAMB confirmation of the best candidates.

usage: OMP_NUM_THREADS=1 python3 scripts/run_kv450_map.py [--starts 36] [--workers 12]
writes runs/kv450_final/map_starts.json and map_best.json
"""
import argparse
import json
import multiprocessing as mp
import time
import numpy as np
import run_kv450_common as C
from run_kv450_common import api, params, PARAM_NAMES, BOUNDS, IDX

_post = None


def _init(z_grid="spec"):
    global _post
    _post = C.make_post(threads=1, z_grid=z_grid)


def make_starts(n, seed=1234):
    rng = np.random.default_rng(seed)
    starts = [("scan_preset", api.preset("scan")), ("start_preset", api.preset("start")), ("test_preset", api.preset("test"))]
    # boundary solution from the earlier MAP run (h=0.82, lnAs=5, A_baryon=2, omch2~0.04-0.05)
    b = api.preset("start")
    b[:6] = [0.0523, 0.0204, 0.82, 1.178, 5.0, 2.0]
    b[6] = 0.8
    starts.append(("boundary_prev", b))
    for k in range(n - len(starts)):
        th = params.prior_transform(rng.uniform(0.02, 0.98, size=params.N_PARAM))     # nuisance from the prior
        lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
        if k % 4 == 3:       # a quarter: fully diverse over the whole cosmological box
            th[:6] = lo[:6] + rng.uniform(0.02, 0.98, 6) * (hi[:6] - lo[:6])
            th[0] = rng.uniform(0.03, 0.5)
        else:
            th[0] = rng.uniform(0.04, 0.30)
            th[1] = rng.uniform(0.0195, 0.0255)
            th[2] = rng.uniform(0.65, 0.815)
            th[3] = rng.uniform(0.75, 1.25)
            th[4] = rng.uniform(2.0, 4.9)
            th[5] = rng.uniform(2.0, 3.13)
        starts.append((f"random{k:02d}", th))
    return starts


def run_one(job):
    name, x0, seed, maxiter, rounds, polish = job
    t0 = time.time()
    th, lp, r = C.optimise(_post, x0, maxiter=maxiter, rounds=rounds, polish=polish, seed=seed)
    chi2 = C.chi2_of(_post, th)
    s8 = C.derived(_post, th)
    return dict(name=name, x0=x0, theta=th, lnpost=lp, chi2=chi2, sigma8=s8[0], omega_m=s8[1], S8=s8[2], n_eval=r.n_eval,
                time=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=36)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stage", choices=("explore", "polish"), default="explore",
                    help="explore: all starts on the 41-node z grid (L-BFGS-B only); polish: top --top candidates, spec grid, + Powell/NM")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--polish", default="powell", help="derivative-free polish after L-BFGS-B in the polish stage (none|powell|nelder-mead|both)")
    a = ap.parse_args()
    t0 = time.time()
    res = []
    if a.stage == "explore":
        starts = make_starts(a.starts)
        jobs = [(n, x, i, 200, 2, "none") for i, (n, x) in enumerate(starts)]
        zg, outf = "reduced", "map_explore.json"
    else:
        prev = json.load(open(C.OUT / "map_explore.json"))["results"]
        jobs = [(r["name"], np.array(r["theta"]), i, 300, 3, a.polish) for i, r in enumerate(prev[: a.top])]
        zg, outf = "spec", "map_starts.json"
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs)), initializer=_init, initargs=(zg,)) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            res.append(r)
            print(f"{r['name']:14s} chi2={r['chi2']:.3f} lnP={r['lnpost']:.3f} S8={r['S8']:.3f} Om={r['omega_m']:.3f} "
                  f"s8={r['sigma8']:.3f} evals={r['n_eval']} t={r['time']:.0f}s", flush=True)
    res.sort(key=lambda r: -r["lnpost"])
    C.jdump(C.OUT / outf, dict(results=res, wall=time.time() - t0, names=PARAM_NAMES, z_grid=zg))
    print("wall", time.time() - t0)
