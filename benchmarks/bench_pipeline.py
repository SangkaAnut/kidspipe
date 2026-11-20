#!/usr/bin/env python
"""ms/evaluation of the kidspipe likelihood pipeline per parameter-move type, for several configurations.

Usage::

    python benchmarks/bench_pipeline.py                         # all configs, 4 threads, writes runs/bench_pipeline.json
    python benchmarks/bench_pipeline.py --configs ii iii --moves 40
    python benchmarks/bench_pipeline.py --threads-scan 1 2 4 8  # thread scaling of config iii (subprocesses)
    python benchmarks/bench_pipeline.py --mcmc [runs/bench_t4.json]   # expected-MCMC accounting from a saved JSON (no timing run)
    python benchmarks/bench_pipeline.py --configs i ii iii iv v vi vii --acc-only   # chi2 decomposition only (no timing)

Configurations (all with the spec z grid unless stated)::

    i    kcap-fidelity (FFT Hankel, node-trapezoid binning) + CAMB 'mead' non-linear   (the faithful reference path)
    ii   fast + stand-alone numba HMcode
    iii  fast + HMcode + transfer emulator (exact-CAMB fallback outside the box)
    iv   fast + HMcode + emulator + reduced 41-node z grid
    v    fast + HMcode + reduced 41-node z grid (no emulator)
    vi   kcap-fidelity + HMcode           (accuracy decomposition only: isolates HMcode vs CAMB 'mead' under kcap binning)
    vii  fast + CAMB 'mead'               (accuracy decomposition only: isolates the exact-binning difference of 'fast')

Move types (every timed move goes to a NEW point so no cache can serve it; the previous point is the base)::

    A_IA/dz          fast block: 6 parameters that only touch the C_ell / Hankel stage
    lnAs/A_baryon    medium block: exact P_lin rescale, new P_nl and Limber look-up
    n_s              exact tilt of P_lin, new P_nl, Limber look-up (+ sigma8 re-power for CAMB)
    omch2|ombh2|h    slow single-parameter moves (new transfer functions)
    slow block       omch2, ombh2, h, n_s all move at once (what a blocked Metropolis step does)
    cold             first call of a fresh model (all caches empty; JIT already warm in the process)

Timing is wall-clock of ``CosmicShearModel.mu`` (the likelihood itself costs 0.015 ms).  The accuracy of each
configuration is |Delta chi2| against configuration i at ``--n-acc`` prior-box points.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np  # noqa: E402

CONFIGS = {
    "i": dict(fidelity="kcap", pnl_backend="camb", z_grid="spec", linear_backend="camb"),
    "ii": dict(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="camb"),
    "iii": dict(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator"),
    "iv": dict(fidelity="fast", pnl_backend="hmcode", z_grid="reduced", linear_backend="emulator"),
    "v": dict(fidelity="fast", pnl_backend="hmcode", z_grid="reduced", linear_backend="camb"),
    "vi": dict(fidelity="kcap", pnl_backend="hmcode", z_grid="spec", linear_backend="camb"),
    "vii": dict(fidelity="fast", pnl_backend="camb", z_grid="spec", linear_backend="camb"),
}
LABEL = {"i": "kcap + CAMB mead", "ii": "fast + HMcode", "iii": "fast + HMcode + emulator",
         "iv": "fast + HMcode + emulator + 41 z", "v": "fast + HMcode + 41 z",
         "vi": "kcap + HMcode", "vii": "fast + CAMB mead"}
MOVES = ["A_IA/dz", "lnAs/A_baryon", "n_s", "omch2|ombh2|h", "slow block"]
BASE = None


def _imports():
    global BASE
    from kidspipe.params import PARAM_NAMES, START
    from kidspipe.theory.model import CosmicShearModel
    from kidspipe.likelihood.gauss import GaussianLikelihood
    BASE = np.asarray(START, float).copy()
    BASE[:5] = [0.12, 0.0222, 0.70, 0.97, 2.8]          # inside the emulator box, near the KV450 best fit
    BASE[5] = 2.8
    return PARAM_NAMES, CosmicShearModel, GaussianLikelihood


def make_move(kind: str, base: np.ndarray, rng, names, step=0.02):
    """A new point differing from ``base`` by a small random move of the given type."""
    idx = {n: i for i, n in enumerate(names)}
    th = base.copy()
    if kind == "A_IA/dz":
        th[idx["A_IA"]] += rng.normal(0, 0.3)
        for i in range(1, 6):
            th[idx[f"dz{i}"]] += rng.normal(0, 0.005)
    elif kind == "lnAs/A_baryon":
        th[idx["ln_1e10_A_s"]] += rng.normal(0, 0.05)
        th[idx["A_baryon"]] = np.clip(th[idx["A_baryon"]] + rng.normal(0, 0.1), 2.0, 3.13)
    elif kind == "n_s":
        th[idx["n_s"]] += rng.normal(0, 0.02)
    elif kind == "omch2|ombh2|h":
        n = ("omch2", "ombh2", "h")[rng.integers(3)]
        th[idx[n]] *= 1.0 + rng.normal(0, step)
    elif kind == "slow block":
        for n in ("omch2", "ombh2", "h", "n_s"):
            th[idx[n]] *= 1.0 + rng.normal(0, step / 2)
    else:
        raise ValueError(kind)
    return th


def time_config(cfg_name, n_moves, threads, seed=1):
    names, Model, _ = _imports()
    kw = dict(CONFIGS[cfg_name])
    rng = np.random.default_rng(seed)
    # warm-up model: pays imports, numba JIT (disk-cached), BLAS thread pools; not timed
    w = Model(nthreads=threads, **kw)
    w.mu(BASE)
    w.mu(make_move("slow block", BASE, rng, names))
    out = {}
    # cold: fresh model, first call
    cold, cold_cpu = [], []
    for j in range(5):
        m = Model(nthreads=threads, **kw)
        th = make_move("slow block", BASE, rng, names)
        t0, c0 = time.perf_counter(), time.process_time()
        m.mu(th)
        cold.append(time.perf_counter() - t0)
        cold_cpu.append(time.process_time() - c0)
    out["cold"] = dict(ms_min=1e3 * min(cold), ms_median=1e3 * float(np.median(cold)), cpu_ms_median=1e3 * float(np.median(cold_cpu)))
    for kind in MOVES:
        m = Model(nthreads=threads, **kw)
        cur = BASE.copy()
        m.mu(cur)
        ts, tc = [], []
        for i in range(n_moves):
            new = make_move(kind, cur, rng, names)
            t0, c0 = time.perf_counter(), time.process_time()
            m.mu(new)
            ts.append(time.perf_counter() - t0)
            tc.append(time.process_time() - c0)
            cur = new if kind not in ("omch2|ombh2|h", "slow block") or (i % 2 == 0) else cur    # random-walk, occasional reject
        ts, tc = np.array(ts[1:]), np.array(tc[1:])          # drop the first move of each series
        out[kind] = dict(ms_median=1e3 * float(np.median(ts)), ms_mean=1e3 * float(ts.mean()), ms_p90=1e3 * float(np.quantile(ts, .9)),
                         ms_min=1e3 * float(ts.min()), cpu_ms_median=1e3 * float(np.median(tc)), n=len(ts))
        print(f"  [{cfg_name}] {kind:15s} median {1e3 * np.median(ts):9.2f} ms  min {1e3 * ts.min():9.2f}  mean {1e3 * ts.mean():9.2f} ms"
              f"  cpu-median {1e3 * np.median(tc):9.2f} ms (n={len(ts)})", flush=True)
    print(f"  [{cfg_name}] cold            min {out['cold']['ms_min']:9.1f} ms  median {out['cold']['ms_median']:9.1f} ms", flush=True)
    return out


def chi2_points(cfg_names, n_acc, seed=11, threads=4):
    """chi2 at ``n_acc`` prior-box points for each configuration (same points)."""
    from common import draw_points, chi2_of
    names, Model, Like = _imports()
    like = Like()
    pts = draw_points(n_acc, seed)
    res = {}
    for c in cfg_names:
        m = Model(nthreads=threads, **CONFIGS[c])
        t0 = time.perf_counter()
        res[c] = np.array([chi2_of(m, like, th) for th in pts])
        print(f"  [{c}] chi2 at {n_acc} points: {time.perf_counter() - t0:.1f} s", flush=True)
    return pts, res


def mcmc_accounting(bench: dict, cfg: str, r_fast_grid=(1, 4, 6, 8, 16), rep_medium: int = 2):
    """Expected cost of one blocked-Metropolis cycle ``slow(1x) + medium(rep_medium x) + fast(r x)`` with the defaults of
    ``kidspipe mcmc`` (``--rep-medium 2 --rep-fast 6``; the first block is the joint update of the six cosmological
    parameters, costed as the 'slow block' move; medium = ln A_s, A_baryon (+ n_s); fast = A_IA, delta_c, A_c, dz1..5)."""
    b = bench["configs"][cfg]["moves"]
    c_slow, c_med, c_fast = b["slow block"]["ms_median"], b["lnAs/A_baryon"]["ms_median"], b["A_IA/dz"]["ms_median"]
    rows = []
    for r in r_fast_grid:
        cyc = c_slow + rep_medium * c_med + r * c_fast
        n_eval = 1 + rep_medium + r
        rows.append(dict(r_fast=r, rep_medium=rep_medium, ms_per_cycle=cyc, evals_per_cycle=n_eval, ms_per_eval=cyc / n_eval))
    return rows


def print_mcmc(path):
    """Expected cost of one blocked-Metropolis cycle (default schedule: 1 joint slow update, 2 medium, 6 fast) per configuration."""
    bench = json.loads(Path(path).read_text())
    print(f"expected MCMC cost from {path} ({bench['threads']} threads, {bench.get('date', '?')})")
    for c, v in bench["configs"].items():
        if "slow block" not in v["moves"]:
            continue
        r6 = mcmc_accounting(bench, c, r_fast_grid=(6,))[0]
        cyc = r6["ms_per_cycle"] / 1e3
        print(f"  [{c}] {v['label']:34s} cycle (1 slow + 2 medium + 6 fast) = {r6['ms_per_cycle']:8.1f} ms, "
              f"{r6['ms_per_eval']:7.1f} ms/eval;  10^4 cycles = {1e4 * cyc / 60:7.1f} min per chain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--moves", type=int, default=30)
    ap.add_argument("--moves-slowcfg", type=int, default=8, help="moves for config i (1-2 s each)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    ap.add_argument("--n-acc", type=int, default=20)
    ap.add_argument("--no-acc", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "bench_pipeline.json"))
    ap.add_argument("--threads-scan", nargs="*", type=int, default=None)
    ap.add_argument("--cfg-scan", default="iii")
    ap.add_argument("--acc-only", action="store_true", help="skip the timing runs, only the chi2 comparison (--n-acc points)")
    ap.add_argument("--mcmc", nargs="?", const=str(ROOT / "runs" / "bench_pipeline.json"), default=None, metavar="JSON",
                    help="print the expected-MCMC accounting from a saved benchmark JSON and exit")
    a = ap.parse_args()

    if a.mcmc:
        print_mcmc(a.mcmc)
        return

    if a.threads_scan:                       # run this script in subprocesses with different thread counts
        scan = {}
        for t in a.threads_scan:
            env = dict(os.environ, OMP_NUM_THREADS=str(t), NUMBA_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            out = ROOT / "runs" / f"bench_threads_{a.cfg_scan}_{t}.json"
            subprocess.run([sys.executable, __file__, "--configs", a.cfg_scan, "--threads", str(t), "--no-acc", "--moves", str(a.moves),
                            "--out", str(out)], env=env, check=True)
            scan[t] = json.loads(out.read_text())["configs"][a.cfg_scan]
        (ROOT / "runs" / f"bench_thread_scan_{a.cfg_scan}.json").write_text(json.dumps(scan, indent=1))
        return

    os.environ.setdefault("NUMBA_NUM_THREADS", str(a.threads))
    res = {"threads": a.threads, "configs": {}, "date": time.strftime("%Y-%m-%d %H:%M"),
           "machine": f"{os.cpu_count()} cores"}
    for c in a.configs:
        if a.acc_only:
            res["configs"][c] = {"label": LABEL[c], "spec": CONFIGS[c], "moves": {}}
            continue
        print(f"== config {c}: {LABEL[c]} ({CONFIGS[c]}), {a.threads} threads", flush=True)
        n = a.moves_slowcfg if c == "i" else a.moves
        res["configs"][c] = {"label": LABEL[c], "spec": CONFIGS[c], "moves": time_config(c, n, a.threads)}
    if not a.no_acc:
        ref = "i" if "i" in a.configs else a.configs[0]
        pts, chi = chi2_points(a.configs, a.n_acc, threads=a.threads)
        for c in a.configs:
            d = chi[c] - chi[ref]
            res["configs"][c]["accuracy_vs_" + ref] = dict(max_abs_dchi2=float(np.abs(d).max()), mean_dchi2=float(d.mean()),
                                                           rms_dchi2=float(np.sqrt((d ** 2).mean())), n_points=int(len(d)))
            res["configs"][c]["chi2"] = chi[c].tolist()
        res["points"] = pts.tolist()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
