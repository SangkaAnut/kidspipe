"""Profile chi2 along ln_1e10_A_s (parameter frozen) and along S8 (quadratic penalty, sigma 0.002, removed in the reported chi2).
Optimised on the 41-node z grid (L-BFGS-B + Powell, warm start from the best MAP), final chi2/S8 re-evaluated on the spec grid.
usage: run_kv450_profile.py [--workers 12]   -> runs/kv450_final/profile.json"""
import argparse
import json
import multiprocessing as mp
import time
import numpy as np
import run_kv450_common as C
from run_kv450_common import IDX

_red = _spec = None


def _init():
    global _red, _spec
    _red = C.make_post(z_grid="reduced", threads=1)
    _spec = C.make_post(z_grid="spec", threads=1)


def job(j):
    kind, val, x0 = j
    t0 = time.time()
    x0 = np.array(x0)
    if kind == "lnAs":
        th, lp, r = C.optimise(_red, x0, fixed={"ln_1e10_A_s": val}, maxiter=200, rounds=2, polish="powell")
    else:
        th, lp, r = C.optimise(_red, x0, s8_target=val, s8_sigma=0.002, maxiter=200, rounds=2, polish="powell")
    d = C.derived(_spec, th)
    return dict(kind=kind, target=val, theta=th, chi2_spec=C.chi2_of(_spec, th), chi2_red=C.chi2_of(_red, th), sigma8=d[0],
                omega_m=d[1], S8=d[2], time=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=C.NCPU, help="worker processes (default os.cpu_count())")
    ap.add_argument("--map-file", default="map_best.json")
    a = ap.parse_args()
    x0 = np.array(json.load(open(C.find(a.map_file)))["theta"])
    jobs = [("lnAs", v, x0) for v in np.round(np.linspace(2.0, 5.0, 16), 3)]
    jobs += [("S8", v, x0) for v in np.round(np.linspace(0.55, 0.90, 15), 3)]
    res = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print(r["kind"], r["target"], f"chi2={r['chi2_spec']:.3f} S8={r['S8']:.4f} Om={r['omega_m']:.3f} s8={r['sigma8']:.3f} t={r['time']:.0f}", flush=True)
    C.jdump(C.OUT / "profile.json", res)
