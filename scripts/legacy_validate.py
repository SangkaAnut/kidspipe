"""KiDS-Legacy COSEBIs validation against the official chain (Wright et al. 2025).

Stages (run in this order; see docs/KIDS_LEGACY_VALIDATION.md for exact commands):
  eval    sample-by-sample chi2 of our pipeline on top-posterior + weighted-random official chain rows (multiprocessing)
  plots   all figures plots/legacy_*.png and the summary json runs/legacy_val/summary.json
  timing  ms per E_n evaluation (runs/legacy_val/timing.json)

Keep OMP_NUM_THREADS / NUMBA_NUM_THREADS = 1 for `eval` (one single-threaded CAMB per worker); --workers defaults to min(10, os.cpu_count()).

Paths (kidspipe.paths): results in <runs>/legacy_val/ and the operator cache in <runs>/legacy/cache with <runs> = $KIDSPIPE_RUNS
(default <root>/runs); figures in $KIDSPIPE_PLOTS (default <root>/plots).  `eval` resumes from <runs>/legacy_val/chi2.npz (your own run
only); `plots` reads chi2.npz / selection.npz from <runs>, else from the copy shipped in <root>/runs (no 2 h eval needed).
"""
import argparse, contextlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kidspipe import paths  # noqa: E402
OUT = paths.runs_dir() / "legacy_val"
PLOTS = Path(os.environ["KIDSPIPE_PLOTS"]).expanduser() if os.environ.get("KIDSPIPE_PLOTS") else ROOT / "plots"
CACHE = str(paths.runs_dir() / "legacy" / "cache")
COL = dict(S8=20, sigma8=21, Om=23, post=43, prior=42, logw=41)

_M = None
_L = None


def valid_rows(chain):
    ok = np.isfinite(chain[:, [21, 23, 41, 42, 43]]).all(1)
    return np.where(ok)[0]


def pick(chain, n_top, n_draw, seed):
    idx = valid_rows(chain)
    top = idx[np.argsort(chain[idx, COL["post"]])[::-1][:n_top]]
    lw = chain[idx, COL["logw"]]
    w = np.exp(lw - lw.max()); w /= w.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(idx, size=n_draw, replace=True, p=w)
    return top, draws


def _init(nthreads):
    global _M, _L
    from kidspipe.legacy import LegacyCosebisModel, LegacyCosebisLikelihood
    _M = LegacyCosebisModel(nthreads=nthreads, cache_dir=CACHE, cache_size=1)
    _L = LegacyCosebisLikelihood()


def _work(s):
    t0 = time.time()
    try:
        mu = _M.mu(s)
        c2 = _L.chi2(mu)
        return s["row"], c2, _M.last_derived["sigma_8"], _M.last_derived["omega_m"], time.time() - t0
    except Exception as e:  # noqa
        return s["row"], np.nan, np.nan, np.nan, time.time() - t0


def stage_eval(a):
    import multiprocessing as mp
    from kidspipe.legacy import load_chain, chain_samples
    chain = load_chain()
    top, draws = pick(chain, a.n_top, a.n_draw, a.seed)
    uniq, counts = np.unique(draws, return_counts=True)
    rows = np.unique(np.concatenate([top, uniq]))
    fn = OUT / "chi2.npz"
    done = {}
    if fn.exists():
        z = np.load(fn)
        done = {int(r): (c, s, o, t) for r, c, s, o, t in zip(z["row"], z["chi2"], z["sigma8"], z["omega_m"], z["t"])}
    todo = [int(r) for r in rows if int(r) not in done]
    np.random.default_rng(a.seed + 1).shuffle(todo)
    print(f"{len(rows)} rows ({len(top)} top, {len(uniq)} unique draws of {a.n_draw}); {len(done)} done, {len(todo)} todo", flush=True)
    np.savez(OUT / "selection.npz", top=top, draws=draws, rows=rows)

    def save():
        r = np.array(list(done)); v = np.array([done[k] for k in r])
        np.savez(fn, row=r, chi2=v[:, 0], sigma8=v[:, 1], omega_m=v[:, 2], t=v[:, 3])

    S = chain_samples(chain, todo)
    t0 = time.time()
    if getattr(a, "pool", None) is not None:          # --mpi: the worker ranks were initialised by __main__
        pool = contextlib.nullcontext(a.pool)
        it = lambda p: p.imap_unordered(_work, S)
    else:
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(1,))
        it = lambda p: p.imap_unordered(_work, S, chunksize=1)
    with pool as p:
        for k, (row, c2, s8, om, dt) in enumerate(it(p)):
            done[row] = (c2, s8, om, dt)
            if (k + 1) % 50 == 0:
                save()
                print(f"{k+1}/{len(todo)}  {time.time()-t0:.0f}s  mean t/sample {np.mean([v[3] for v in done.values()]):.1f}s", flush=True)
    save()
    print("eval finished", time.time() - t0, flush=True)


# ------------------------------------------------------------------------------------------------
def stage_timing(a):
    from dataclasses import replace
    from kidspipe.legacy import LegacyCosebisModel, LegacyOptions, load_chain, chain_samples
    chain = load_chain()
    best = int(np.argmax(np.where(np.isfinite(chain[:, 43]), chain[:, 43], -np.inf)))
    s = chain_samples(chain, [best])[0]
    nth = int(os.environ.get("OMP_NUM_THREADS", "1"))
    m = LegacyCosebisModel(nthreads=nth, cache_dir=CACHE, cache_size=2)
    m.mu(s)
    res = {"threads": nth}
    # cold: new cosmology each time (CAMB + Limber setup + cl + operator)
    t = []
    rng = np.random.default_rng(0)
    for i in range(a.n_time):
        p = dict(s); p["omch2"] = s["omch2"] * (1 + 0.02 * rng.standard_normal()); p["A_s"] = s["A_s"] * (1 + 0.02 * rng.standard_normal())
        t0 = time.perf_counter(); m.mu(p); t.append(time.perf_counter() - t0)
    res["cold_ms_mean"], res["cold_ms_std"] = 1e3 * np.mean(t), 1e3 * np.std(t)
    # warm: only nuisance (IA, dz) change, cosmology cached
    t = []
    for i in range(50):
        p = dict(s); p["a"] = s["a"] * (1 + 0.1 * rng.standard_normal()); p["bias_3"] = s["bias_3"] + 0.01 * rng.standard_normal()
        t0 = time.perf_counter(); m.mu(p); t.append(time.perf_counter() - t0)
    res["warm_ms_mean"], res["warm_ms_std"] = 1e3 * np.mean(t), 1e3 * np.std(t)
    # operator application alone
    _, C = m.cl(s)
    t0 = time.perf_counter()
    for _ in range(200):
        (C[np.array([p[0] for p in __import__('kidspipe.legacy.data', fromlist=['PAIRS']).PAIRS]), np.array([p[1] for p in __import__('kidspipe.legacy.data', fromlist=['PAIRS']).PAIRS])] @ m.M.T)
    res["operator_ms"] = 1e3 * (time.perf_counter() - t0) / 200
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / f"timing_t{nth}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "plots", "timing"])
    ap.add_argument("--n-top", type=int, default=150)
    ap.add_argument("--n-draw", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1), help="eval worker processes (default min(10, os.cpu_count()))")
    ap.add_argument("--n-time", type=int, default=8)
    ap.add_argument("--mpi", action="store_true", help="eval over MPI ranks: `mpirun -n N python scripts/legacy_validate.py eval "
                    "--mpi`; rank 0 schedules and saves, ranks 1..N-1 evaluate (one single-threaded model each); --workers ignored")
    a = ap.parse_args()
    from kidspipe import mpi
    if a.mpi and a.stage != "eval":
        raise SystemExit("--mpi applies to the eval stage only")
    if not a.mpi:
        mpi.refuse_unflagged_launch("--mpi")
    if a.mpi:
        nw = mpi.require_workers(1)
        with mpi.executor(_init, (1,), threads=1) as ex:
            if ex is None:                             # worker rank: rank 0 has finished
                sys.exit(0)
            OUT.mkdir(parents=True, exist_ok=True)
            a.pool = mpi.Pool(ex, nw)
            print(f"MPI: {nw} worker ranks", flush=True)
            stage_eval(a)
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    if a.stage == "eval":
        stage_eval(a)
    elif a.stage == "timing":
        stage_timing(a)
    else:
        from legacy_plots import stage_plots
        stage_plots(a)
