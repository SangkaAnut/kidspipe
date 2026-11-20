"""Systematic-sensitivity study: MAP + short adaptive chains for model / parameter-treatment variants.

variants (name: model options, fixed parameters):
  baseline, abar2 (A_baryon=2.0), abar313 (A_baryon=3.13), aia0 (A_IA=0), nomcorr (m-bias correction off),
  camb_mead (CAMB 'mead' non-linear backend + exact CAMB linear), kcap (kcap fidelity), reduced (41-z grid)
usage: python3 scripts/run_kv450_sens.py --variants all --workers 12 --cycles 3000 --burn 1500 [--no-chain]
writes runs/kv450_final/sens/<variant>.json (+ chain npz)
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import run_kv450_common as C
from run_kv450_common import api, params, PARAM_NAMES, BOUNDS, IDX
from kidspipe.inference.mcmc import blocked_metropolis, split_rhat, effective_sample_size

VARIANTS = {
    "baseline": dict(model=dict(), fixed={}),
    "abar2": dict(model=dict(), fixed={"A_baryon": 2.0}),
    "abar313": dict(model=dict(), fixed={"A_baryon": 3.13}),
    "aia0": dict(model=dict(), fixed={"A_IA": 0.0}),
    "nomcorr": dict(model=dict(mcorr=False), fixed={}),
    "camb_mead": dict(model=dict(linear="camb", backend="camb"), fixed={}),
    "kcap": dict(model=dict(fidelity="kcap"), fixed={}),
    "reduced": dict(model=dict(z_grid="reduced"), fixed={}),
}
CHAIN_VARIANTS = ("abar2", "abar313", "aia0", "nomcorr")
SL, MD, FA = (api.block_indices(k) for k in ("slow", "medium", "fast"))

_posts = {}


def get_post(name):
    if name not in _posts:
        _posts[name] = C.make_post(threads=1, **VARIANTS[name]["model"])
    return _posts[name]


def run_variant(job):
    name, x0, do_chain, cycles, burn, seed, chain_id, cov_full = job
    t0 = time.time()
    post = get_post(name)
    fixed = VARIANTS[name]["fixed"]
    out = dict(name=name, chain_id=chain_id)
    x0 = np.array(x0, float)
    for k, v in fixed.items():
        x0[IDX[k]] = v
    if chain_id == 0:
        # MAP: several starts (the baseline MAP + shifted points)
        best = None
        rng = np.random.default_rng(seed)
        starts = [x0] + [np.clip(x0 + 0.05 * (BOUNDS[:, 1] - BOUNDS[:, 0]) * rng.standard_normal(14), BOUNDS[:, 0] + 1e-9, BOUNDS[:, 1] - 1e-9) for _ in range(1)]
        slow_model = name == "camb_mead"
        for s in (starts[:1] if slow_model else starts):
            th, lp, r = C.optimise(post, s, fixed=fixed, maxiter=(40 if slow_model else 200), rounds=(1 if slow_model else 2), polish="none", seed=seed)
            if best is None or lp > best[1]:
                best = (th, lp)
        th, lp = best
        out.update(map_theta=th, map_lnpost=lp, map_chi2=C.chi2_of(post, th), map_derived=C.derived(post, th))
    if do_chain and chain_id >= 1:
        free = np.array([i for i, n in enumerate(PARAM_NAMES) if n not in fixed])
        def sub(idx):
            return np.array([np.nonzero(free == i)[0][0] for i in idx if i in free], int)
        sched = [(sub(np.concatenate([SL, MD])), 1), (sub(MD), 2), (sub(FA), 6)]
        def f(y):
            t = x0.copy()
            t[free] = y
            return post.log_post(t)
        cov0 = cov_full[np.ix_(free, free)]
        r = blocked_metropolis(f, x0[free], sched, n_cycles=cycles, n_burn_cycles=burn, cov0=cov0, bounds=BOUNDS[free],
                               seed=seed, cov_prior_weight=50.0)
        X = np.tile(x0, (len(r.samples), 1))
        X[:, free] = r.samples
        out.update(samples=X, log_post=r.log_post, derived=api.derived_columns(post, X), accept=r.accept, time_chain=time.time() - t0)
    out["time"] = time.time() - t0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all")
    ap.add_argument("--workers", type=int, default=min(12, C.NCPU), help="worker processes (default min(12, os.cpu_count()))")
    ap.add_argument("--cycles", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--no-chain", action="store_true")
    ap.add_argument("--pilot", default="pilot1")
    ap.add_argument("--map-file", default="map_best.json")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    outdir = C.OUT / "sens"
    outdir.mkdir(exist_ok=True)
    mp_ = json.load(open(C.find(a.map_file)))
    x_map = np.array(mp_["theta"])
    P = np.load(C.find(Path(a.pilot) / "pilot.npz"))
    cov = P["cov"]
    mean = P["mean"]
    # chain starts: pilot mean (inside the posterior bulk) for chains; MAP for the optimiser
    jobs = []
    for n in names:
        if True:
            jobs.append((n, x_map, False, a.cycles, a.burn, a.seed, 0, cov))
        if not a.no_chain and n in CHAIN_VARIANTS:
            for c in range(1, a.chains + 1):
                jobs.append((n, mean, True, a.cycles, a.burn, a.seed + c, c, cov))
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_variant, jobs):
            res.setdefault(r["name"], []).append(r)
            print(r["name"], r["chain_id"], f"t={r['time']:.0f}s", {k: (round(v, 3) if isinstance(v, float) else None) for k, v in r.items() if k in ("map_chi2",)}, flush=True)
    summ = {}
    for n, rs in res.items():
        d = dict(model=VARIANTS[n]["model"], fixed=VARIANTS[n]["fixed"])
        m = [r for r in rs if r["chain_id"] == 0]
        if m:
            m = m[0]
            d.update(map_chi2=m["map_chi2"], map_lnpost=m["map_lnpost"], map_theta=m["map_theta"], map_derived=m["map_derived"])
        ch = [r for r in rs if r["chain_id"] >= 1]
        if ch:
            S8 = np.array([r["derived"][:, 2] for r in ch]).T
            sig8 = np.array([r["derived"][:, 0] for r in ch]).T
            om = np.array([r["derived"][:, 1] for r in ch]).T
            d.update(S8_mean=float(S8.mean()), S8_std=float(S8.std()), S8_rhat=float(split_rhat(S8[:, :, None])[0]),
                     S8_ess=float(effective_sample_size(S8[:, :, None])[0]),
                     S8_q=np.percentile(S8, [15.87, 50, 84.13]), sigma8_mean=float(sig8.mean()), omega_m_mean=float(om.mean()),
                     accept=[r["accept"] for r in ch], n_samples=int(S8.size),
                     lnpost_max_chain=float(max(r["log_post"].max() for r in ch)))
            np.savez(outdir / f"{n}_chains.npz", samples=np.array([r["samples"] for r in ch]),
                     derived=np.array([r["derived"] for r in ch]), log_post=np.array([r["log_post"] for r in ch]))
        summ[n] = d
        print(n, {k: v for k, v in d.items() if k in ("map_chi2", "S8_mean", "S8_std", "S8_rhat", "S8_ess")}, flush=True)
    C.jdump(outdir / "sens_summary.json", summ)
    print("wall", time.time() - t0)
