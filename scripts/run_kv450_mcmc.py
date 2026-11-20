"""Blocked-Metropolis sampling of the full 14-D KV450 posterior with the transfer emulator (hmcode, 'fast', spec grids).

Two stages (one OS process per chain, 1 core each):
  pilot : adaptive blocked Metropolis from dispersed starts -> pooled covariance + block scales  (runs/kv450_final/<tag>/pilot.npz)
  prod  : fixed-proposal blocked Metropolis (valid Markov chains), chunked with checkpoints; the proposal is the pilot
          covariance (conditional covariance per block) times the pilot scales.
usage:
  python3 scripts/run_kv450_mcmc.py pilot --tag pilot1 --chains 12 --burn 3000 --cycles 3000 --starts runs/kv450_final/starts.json
  python3 scripts/run_kv450_mcmc.py prod  --tag prod1 --pilot pilot1 --chains 12 --minutes 60 --starts-from pilot1
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import run_kv450_common as C
from run_kv450_common import api, params, PARAM_NAMES, BOUNDS, IDX
from kidspipe.inference.mcmc import blocked_metropolis, _conditional_cov, _safe_chol

_post = None
SL, MD, FA = api.block_indices("slow"), api.block_indices("medium"), api.block_indices("fast")
JOINT = np.concatenate([SL, MD])
REPS = (1, 1, 8)       # joint block x1, medium x1, fast x8 per cycle (HMcode re-evaluation dominates the cost, so fewer medium moves than the CLI default)


def _init(linear, backend, fidelity, z_grid):
    global _post
    _post = C.make_post(linear=linear, backend=backend, fidelity=fidelity, z_grid=z_grid, threads=1, cache_size=4)


def derived_for(samples):
    return api.derived_columns(_post, samples)


def _stats():
    st = dict(getattr(_post.model.linear, "stats", {}))
    return {k: (float(v) if not isinstance(v, (int, float)) else v) for k, v in st.items()}


def pilot_job(job):
    i, x0, burn, cycles, seed, cov0, outdir = job
    t0 = time.time()
    sched = [(JOINT, REPS[0]), (MD, REPS[1]), (FA, REPS[2])]
    r = blocked_metropolis(_post.log_post, x0, sched, n_cycles=cycles, n_burn_cycles=burn, cov0=cov0, bounds=BOUNDS,
                           seed=seed, cov_prior_weight=100.0)
    d = derived_for(r.samples)
    np.savez(Path(outdir) / f"pilot_c{i}.npz", samples=r.samples, log_post=r.log_post, derived=d, burn=r.burn_samples,
             burn_lp=r.burn_log_post, cov=r.cov, accept=r.accept_blocks and np.array(list(r.accept_blocks.values())),
             scales=np.array(list(r.scales.values())), time=time.time() - t0)
    return dict(i=i, time=time.time() - t0, accept=r.accept_blocks, scales=r.scales, cov=r.cov, n_eval=r.n_eval,
                lp_end=float(r.log_post[-1]), stats=_stats())


def _tune_scales(C_, scales):
    return np.asarray(scales, float)


def prod_job(job):
    """Fixed-proposal blocked MH, chunked; stops after `minutes` or `max_cycles`, writes checkpoints every `ckpt` s."""
    i, x0, cov, scales, seed, outdir, minutes, max_cycles, chunk, ckpt, thin, tune = job
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    lp = _post.log_post(x)
    sched = [(JOINT, REPS[0]), (MD, REPS[1]), (FA, REPS[2])]
    L0 = [_safe_chol(_conditional_cov(cov, idx)) for idx, _ in sched]
    log_s = np.log(np.asarray(scales, float))
    targets = [0.25, 0.30, 0.30]
    # --- tuning phase (discarded): Robbins-Monro on the per-block scale with the covariance fixed
    n_tune = int(tune)
    t_ad = np.zeros(3)
    for _c in range(n_tune):
        for b, (idx, r) in enumerate(sched):
            for _j in range(r):
                y = x.copy()
                y[idx] += np.exp(log_s[b]) * (L0[b] @ rng.standard_normal(len(idx)))
                lpy = _post.log_post(y)
                a_ = lpy - lp
                if lpy > -np.inf and np.log(rng.uniform()) < a_:
                    x, lp = y, lpy
                t_ad[b] += 1
                p_acc = 0.0 if lpy == -np.inf else min(1.0, np.exp(min(a_, 0.0)))
                log_s[b] += 2.0 * (p_acc - targets[b]) / (t_ad[b] + 10.0) ** 0.6
    Ls = [L0[b] * np.exp(log_s[b]) for b in range(3)]
    tuned = np.exp(log_s)
    out_x, out_lp = [], []
    acc = np.zeros(3)
    prop = np.zeros(3)
    path = Path(outdir) / f"prod_c{i}.npz"
    last_ck = time.time()
    n_cyc = 0

    def save(final=False):
        X = np.array(out_x)
        d = derived_for(X) if len(X) else np.zeros((0, 3))
        np.savez(path, samples=X, log_post=np.array(out_lp), derived=d, accept=acc / np.maximum(prop, 1), n_cycles=n_cyc,
                 time=time.time() - t0, final=final, thin=thin, scales=tuned, x_start=np.asarray(x0), stats=json.dumps(_stats()))

    while (time.time() - t0) < minutes * 60 and n_cyc < max_cycles and not (Path(outdir) / "STOP").exists():
        for _ in range(chunk):
            for b, (idx, r) in enumerate(sched):
                for _j in range(r):
                    y = x.copy()
                    y[idx] += Ls[b] @ rng.standard_normal(len(idx))
                    lpy = _post.log_post(y)
                    prop[b] += 1
                    if lpy > -np.inf and np.log(rng.uniform()) < lpy - lp:
                        x, lp = y, lpy
                        acc[b] += 1
            n_cyc += 1
            if n_cyc % thin == 0:
                out_x.append(x.copy())
                out_lp.append(lp)
            if n_cyc >= max_cycles:
                break
        if time.time() - last_ck > ckpt:
            save()
            last_ck = time.time()
    save(final=True)
    return dict(i=i, n_cycles=n_cyc, time=time.time() - t0, accept=(acc / np.maximum(prop, 1)).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pilot", "prod"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chains", type=int, default=12)
    ap.add_argument("--burn", type=int, default=3000)
    ap.add_argument("--cycles", type=int, default=3000)
    ap.add_argument("--starts", default=None, help="json list of start vectors (pilot)")
    ap.add_argument("--pilot", default=None, help="pilot tag providing cov + scales (prod)")
    ap.add_argument("--starts-from", default=None, help="pilot tag whose chain ends / spread supply the prod starts")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--max-cycles", type=int, default=10**9)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--tune", type=int, default=1000, help="prod: discarded scale-tuning cycles per chain")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--ckpt", type=float, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--linear", default="emulator")
    ap.add_argument("--backend", default="hmcode")
    ap.add_argument("--fidelity", default="fast")
    ap.add_argument("--z-grid", default="spec")
    ap.add_argument("--cov-inflate", type=float, default=1.0)
    a = ap.parse_args()
    outdir = C.OUT / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "STOP").unlink(missing_ok=True)
    ctx = mp.get_context("spawn")
    init = dict(initializer=_init, initargs=(a.linear, a.backend, a.fidelity, a.z_grid))
    if a.stage == "pilot":
        starts = np.array(json.loads(Path(a.starts).read_text()))[: a.chains]
        cov0 = np.diag(((BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.02) ** 2)
        jobs = [(i, starts[i], a.burn, a.cycles, a.seed * 1000 + i, cov0, str(outdir)) for i in range(len(starts))]
        with ctx.Pool(len(jobs), **init) as pool:
            res = pool.map(pilot_job, jobs, chunksize=1)
        S = np.concatenate([np.load(outdir / f"pilot_c{i}.npz")["samples"] for i in range(len(jobs))])
        covp = np.cov(S.T)
        scales = np.mean([[v for v in r["scales"].values()] for r in res], axis=0)
        np.savez(outdir / "pilot.npz", cov=covp, scales=scales, mean=S.mean(0),
                 covs_adapted=np.array([r["cov"] for r in res]))
        for r in res:
            print(r["i"], f"t={r['time']:.0f}s", {k: round(v, 3) for k, v in r["accept"].items() if "burn" not in k},
                  "scales", [round(v, 3) for v in r["scales"].values()], "lp_end", round(r["lp_end"], 2), r["stats"], flush=True)
        print("pooled sd:", dict(zip(PARAM_NAMES, np.sqrt(np.diag(covp)).round(4))))
    else:
        P = np.load(C.OUT / a.pilot / "pilot.npz")
        cov = P["cov"] * a.cov_inflate
        scales = P["scales"]
        src = a.starts_from or a.pilot
        import glob
        nsrc = len(glob.glob(str(C.OUT / src / "pilot_c*.npz")))
        ends = np.array([np.load(C.OUT / src / f"pilot_c{i % nsrc}.npz")["samples"][-1] for i in range(a.chains)])
        jobs = [(i, ends[i], cov, scales, a.seed * 1000 + i, str(outdir), a.minutes, a.max_cycles, a.chunk, a.ckpt, a.thin, a.tune)
                for i in range(a.chains)]
        with ctx.Pool(len(jobs), **init) as pool:
            res = pool.map(prod_job, jobs, chunksize=1)
        for r in res:
            print(r, flush=True)
