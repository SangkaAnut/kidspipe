"""Maximum-LIKELIHOOD search (flat inside the bounds, i.e. the Gaussian nuisance priors are NOT included), spec grid + emulator.
Starts: the MAP, the minimum-chi2 chain sample and 4 more chain samples with chi2 < 183.   -> runs/kv450_final/ml.json
usage: run_kv450_ml.py [--workers 6]"""
import argparse
import glob
import multiprocessing as mp
import time
import numpy as np
import run_kv450_common as C
from run_kv450_common import params, PARAM_NAMES

_p = None


class _LL:
    def __init__(self, post):
        self.post = post

    def log_post(self, th):
        return self.post.log_like(th)


def _init():
    global _p
    _p = _LL(C.make_post(threads=1))


def job(j):
    name, x0 = j
    t0 = time.time()
    th, lp, r = C.optimise(_p, np.array(x0), maxiter=300, rounds=3, polish="powell")
    chi2 = C.chi2_of(_p.post, th)
    d = C.derived(_p.post, th)
    lpost = float(_p.post.log_post(th))
    return dict(name=name, theta=th, chi2=chi2, lnpost=lpost, sigma8=d[0], omega_m=d[1], S8=d[2], time=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=C.NCPU, help="worker processes (default os.cpu_count())")
    a = ap.parse_args()
    X, LP = [], []
    for f in C.result_glob("prod*/prod_c*.npz"):
        z = np.load(f)
        X.append(z["samples"]); LP.append(z["log_post"])
    X, LP = np.concatenate(X), np.concatenate(LP)
    chi2 = -2 * (LP - params.log_prior(X))
    order = np.argsort(chi2)
    import json
    starts = [("map", json.load(open(C.find("map_best.json")))["theta"]), ("chain_minchi2", X[order[0]])]
    rng = np.random.default_rng(1)
    good = np.nonzero(chi2 < 183)[0]
    for k in rng.choice(good, 4, replace=False):
        starts.append((f"chain_{k}", X[k]))
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        res = []
        for r in pool.imap_unordered(job, starts):
            res.append(r)
            print(r["name"], f"chi2={r['chi2']:.3f} (lnpost {r['lnpost']:.3f}) S8={r['S8']:.3f} Om={r['omega_m']:.3f} t={r['time']:.0f}", flush=True)
    res.sort(key=lambda r: r["chi2"])
    C.jdump(C.OUT / "ml.json", dict(results=res, names=PARAM_NAMES, min_chain_chi2=float(chi2.min())))
