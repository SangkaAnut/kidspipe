"""Command line interface: ``python -m kidspipe {check,loglike,map,mcmc,nested,bench} ...``.

Every sub-command accepts the shared model options (``--backend``, ``--fidelity``, ...), a deterministic
``--seed`` and writes its products to ``--out`` (default ``<repo>/runs/<name>/``, ``--name`` defaults to the
sub-command).  See ``docs/USAGE.md`` for worked examples and ``python -m kidspipe <cmd> --help``.

Parallelism (``--workers N``): independent worker *processes*, each with its own model (CAMB cache) and
``--worker-threads`` threads (default 1).  Chains / MAP starts run one per worker; the ensemble and nested
samplers evaluate their batches over the pool.  ``--threads`` is the CAMB thread count of the main process.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# default thread count for BLAS/numba/OpenMP unless the user set it (must happen before numpy import)
os.environ.setdefault("OMP_NUM_THREADS", "4")
# numba (HMcode / Limber kernels) sizes its thread pool from NUMBA_NUM_THREADS (default: all cores), NOT from OMP_NUM_THREADS;
# follow the OpenMP setting so that "N threads per process" really bounds the CPU use (results are bit-identical, tested).
_NUMBA_USER_SET = "NUMBA_NUM_THREADS" in os.environ
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import numpy as np  # noqa: E402

from . import api, params  # noqa: E402
from .params import PARAM_NAMES  # noqa: E402

BAD_LL = -1.0e300


# =============================================================================================
# helpers
# =============================================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else (None if math.isnan(f) else ("inf" if f > 0 else "-inf"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")


class _Say:
    def __init__(self, quiet=False):
        self.quiet = quiet

    def __call__(self, *a, **k):
        if not self.quiet:
            print(*a, **k, flush=True)


def _outdir(args) -> Path:
    if getattr(args, "out", None):
        d = Path(args.out)
    else:
        d = api.default_runs_dir() / (args.name or args.cmd)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_kw(args, nthreads=None) -> dict:
    return dict(fidelity=args.fidelity, pnl_backend=args.backend, z_grid=args.z_grid, n_ell=args.n_ell,
                n_nodes=args.n_nodes, nthreads=args.threads if nthreads is None else nthreads,
                cache_size=args.cache_size)


def _lik_kw(args) -> dict:
    return dict(apply_m=not args.no_mcorr, use_cterm=not args.no_cterm)


def _build(args, nthreads=None):
    return api.build_posterior(api.build_model(**_model_kw(args, nthreads)), api.build_likelihood(**_lik_kw(args)))


def _theta_from_args(args, default_preset="start") -> np.ndarray:
    """theta from --theta-file / --preset / --set (in increasing priority)."""
    base = getattr(args, "preset", None) or default_preset
    th = api.preset(base)
    if getattr(args, "theta_file", None):
        th = api.load_theta_file(args.theta_file, base=th)
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        k, v = item.split("=", 1)
        th = api.theta_from({k.strip(): float(v)}, base=th)
    return th


def _theta_table(th, post=None) -> str:
    lo, hi = params.BOUNDS[:, 0], params.BOUNDS[:, 1]
    return "\n".join(f"  {n:<12s} {v:14.8g}   [{lo[i]:g}, {hi[i]:g}]" for i, (n, v) in enumerate(zip(PARAM_NAMES, th)))


def _add_model_args(p):
    g = p.add_argument_group("model")
    g.add_argument("--backend", choices=("hmcode", "camb"), default="hmcode",
                   help="non-linear P(k): stand-alone numba HMcode-2016 or CAMB 'mead' (default hmcode)")
    g.add_argument("--fidelity", choices=("fast", "kcap"), default="fast",
                   help="'fast' = exact bin-averaged Hankel operator, 'kcap' = FFT Hankel + node-trapezoid binning")
    g.add_argument("--z-grid", choices=("spec", "reduced"), default="spec", help="P(k,z) redshift grid (150 or 41 nodes)")
    g.add_argument("--n-ell", type=int, default=400, help="number of log-spaced ell nodes (default 400)")
    g.add_argument("--n-nodes", type=int, default=2000, help="Limber integration nodes in z (default 2000)")
    g.add_argument("--cache-size", type=int, default=4, help="model stage-2 cache entries (default 4)")
    g.add_argument("--threads", type=int, default=None, help="CAMB threads of the main process (default min(4, ncpu))")
    g.add_argument("--no-mcorr", action="store_true", help="do not apply the m-bias correction to the data")
    g.add_argument("--no-cterm", action="store_true", help="drop the A_c^2 c-term")


def _add_common(p, out=True, seed=True):
    if seed:
        p.add_argument("--seed", type=int, default=0, help="random seed (default 0); runs are deterministic given the seed")
    if out:
        p.add_argument("--out", default=None, help="output directory (default <repo>/runs/<name>/)")
        p.add_argument("--name", default=None, help="run name -> runs/<name>/ (default: the sub-command)")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")


def _add_theta_args(p, default_preset="start"):
    g = p.add_argument_group("parameter point")
    g.add_argument("--preset", choices=sorted(api.PRESETS), default=None,
                   help=f"base point: test (spec s4.1 cosmology, A_baryon=2), scan (spec s4.2 base at lnAs=2.8), "
                        f"start (KV450 start values); default {default_preset}")
    g.add_argument("--theta-file", default=None,
                   help="JSON ({name: value} or 14-list or a `kidspipe map` result) or ini-like 'name = value' file")
    g.add_argument("--set", action="append", metavar="NAME=VALUE", help="override one parameter (repeatable)")


def _add_pool_args(p):
    g = p.add_argument_group("parallelism")
    g.add_argument("--workers", type=int, default=1, help="worker processes (default 1 = serial in this process)")
    g.add_argument("--worker-threads", type=int, default=1, help="CAMB/OpenMP threads per worker (default 1)")
    g.add_argument("--mpi", action="store_true",
                   help="use MPI ranks instead of local worker processes: launch with `mpirun -n N python -m kidspipe ...`; "
                        "rank 0 runs the command, ranks 1..N-1 are the workers (N >= 3); --workers is then N-1")


# =============================================================================================
# worker processes
# =============================================================================================
_W: dict = {}


def _w_init(model_kw, lik_kw, nthreads):
    mk = dict(model_kw, nthreads=nthreads)
    _W["post"] = api.build_posterior(api.build_model(**mk), api.build_likelihood(**lik_kw))


def _w_eval(theta):
    """(ln L or BAD_LL, (sigma8, omega_m, S8)) at theta; for the pool wrappers."""
    post = _W["post"]
    t = np.asarray(theta, float)
    ll = post.log_like(t)
    if not np.isfinite(ll):
        return BAD_LL, (math.nan,) * 3
    d = post.recorder.lookup(t)
    return float(ll), tuple(d[n] for n in api.DERIVED_NAMES)


def _w_loglike(theta):
    return _w_eval(theta)[0]


def _w_logpost(theta):
    """(ln post, derived) at theta; -inf allowed."""
    post = _W["post"]
    t = np.asarray(theta, float)
    lp = post.log_post(t)
    if not np.isfinite(lp):
        return -np.inf, (math.nan,) * 3
    d = post.recorder.lookup(t)
    return float(lp), tuple(d[n] for n in api.DERIVED_NAMES)


def _w_derived(thetas):
    post = _W["post"]
    return api.derived_columns(post, np.asarray(thetas, float))


def _w_chain(job):
    """One blocked-Metropolis chain in a worker; returns plain arrays."""
    return _run_blocked(_W["post"], job)


def _w_map(job):
    return _run_map_start(_W["post"], job)


_MPI: dict = {}     # set by main() under --mpi: {"ex": MPIPoolExecutor (root only), "nworkers": int, "init": (model_kw, lik_kw, threads)}


class _ExecPool:
    """Minimal ``map``-style pool over spawned worker processes (dynesty / generic use).

    Under ``--mpi`` it wraps the one MPI executor that ``main()`` opened for the whole command (its workers were
    initialised with the same model/likelihood options); ``close()`` then leaves that executor running."""

    def __init__(self, nworkers, model_kw, lik_kw, worker_threads=1):
        if _MPI.get("ex") is not None:
            if (model_kw, lik_kw, worker_threads) != _MPI["init"]:
                raise RuntimeError("MPI workers were initialised with different model/likelihood options than requested")
            self._ex, self._owned = _MPI["ex"], False
            self.size = self._processes = _MPI["nworkers"]
            return
        self._owned = True
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        os.environ["OMP_NUM_THREADS"] = str(worker_threads)       # inherited by the spawned workers
        os.environ["NUMBA_NUM_THREADS"] = str(worker_threads)
        self.size = self._processes = int(nworkers)
        self._ex = ProcessPoolExecutor(max_workers=self.size, mp_context=mp.get_context("spawn"),
                                       initializer=_w_init, initargs=(model_kw, lik_kw, worker_threads))

    def map(self, func, iterable):
        items = list(iterable)
        if not items:
            return []
        return list(self._ex.map(func, items, chunksize=max(1, len(items) // (4 * self.size))))

    def submit(self, *a, **k):
        return self._ex.submit(*a, **k)

    def close(self):
        if self._owned:
            self._ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _EvalPool(_ExecPool):
    """Pool for nautilus: ``map`` ignores the passed callable and evaluates ln L (finite) in the workers,
    recording sigma8/Omega_m/S8 in ``table`` (keyed by the 5 cosmology parameters) for free."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.table: dict = {}

    def map(self, func, iterable):
        pts = [np.asarray(x, float) for x in iterable]
        res = super().map(_w_eval, pts)
        out = []
        for t, (ll, d) in zip(pts, res):
            if np.isfinite(ll) and ll > BAD_LL:
                self.table[api.DerivedRecorder._key(t)] = d
            out.append(ll)
        return out


# =============================================================================================
# check
# =============================================================================================
def cmd_check(args) -> int:
    say = _Say(args.quiet)
    ok_all = True
    rows = []

    def record(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        rows.append({"check": name, "ok": bool(ok), "detail": detail})
        say(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    t0 = time.perf_counter()
    post = _build(args)
    t_build = time.perf_counter() - t0
    say(f"build model + likelihood: {t_build:.2f} s   (backend={args.backend}, fidelity={args.fidelity}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    th = api.preset("test")
    t0 = time.perf_counter()
    post.model.mu(th)
    t_first = time.perf_counter() - t0
    say(f"first model evaluation (JIT warm-up, numba cache on disk afterwards): {t_first:.2f} s")
    th2 = th.copy()
    th2[PARAM_NAMES.index("omch2")] = 0.1001           # new CAMB transfer: cold-physics cost
    t0 = time.perf_counter()
    post.model.mu(th2)
    t_slow = time.perf_counter() - t0
    th3 = th2.copy()
    th3[PARAM_NAMES.index("A_IA")] = 0.5
    t0 = time.perf_counter()
    post.model.mu(th3)
    t_fast = time.perf_counter() - t0
    say(f"warm evaluation: omch2 move {1e3 * t_slow:.1f} ms, A_IA move {1e3 * t_fast:.2f} ms")

    chi2 = -2.0 * post.log_like(th)
    err = chi2 - api.ANCHOR_TEST_CHI2
    record("chi2 at spec test cosmology", abs(err) < args.tol,
           f"chi2 = {chi2:.3f}, spec anchor {api.ANCHOR_TEST_CHI2} (A_baryon=2), diff {err:+.3f}, tol {args.tol}")
    d = api.derived_for(post, th)
    record("derived quantities", abs(d["sigma8"] - 0.709092) < 2e-4 and abs(d["omega_m"] - 0.250296) < 2e-5,
           f"sigma8 = {d['sigma8']:.6f} (anchor 0.709092), Omega_m = {d['omega_m']:.6f} (0.250296), S8 = {d['S8']:.6f} (0.647692)")

    # A_c / delta_c are forwarded to the likelihood (silent-fixing bug guard)
    thc = th.copy()
    thc[PARAM_NAMES.index("A_c")] = 1.2
    thd = th.copy()
    thd[PARAM_NAMES.index("delta_c")] = 3e-4
    c_a, c_d = -2 * post.log_like(thc), -2 * post.log_like(thd)
    record("A_c and delta_c reach the likelihood", abs(c_a - chi2) > 1e-3 and abs(c_d - chi2) > 1e-6,
           f"chi2(A_c=1.2) = {c_a:.3f}, chi2(delta_c=3e-4) = {c_d:.4f} vs {chi2:.3f}")
    bad = th.copy()
    bad[0] = -1.0
    nanth = th.copy()
    nanth[2] = float("nan")
    record("out-of-bounds / NaN theta -> -inf", post.log_post(bad) == -np.inf and post.log_post(nanth) == -np.inf,
           "log_post = -inf, no exception")

    if args.scan:
        errs = []
        for lnA, exp in api.ANCHOR_SCAN.items():
            t = api.theta_from(dict(api.AMP_SCAN_BASE, ln_1e10_A_s=lnA))
            errs.append((lnA, -2 * post.log_like(t), exp))
        worst = max(abs(c - e) for _, c, e in errs)
        record("amplitude scan vs spec s4.2", worst < args.scan_tol,
               "; ".join(f"lnAs={a}: {c:.2f} (spec {e})" for a, c, e in errs) + f"; max |diff| = {worst:.2f}, tol {args.scan_tol}")
    if args.camb:
        alt = api.build_posterior(api.build_model(**dict(_model_kw(args), pnl_backend="camb" if args.backend == "hmcode" else "hmcode")),
                                  api.build_likelihood(**_lik_kw(args)))
        c2 = -2.0 * alt.log_like(th)
        record("second non-linear backend agrees", abs(c2 - chi2) < 0.3,
               f"chi2 = {c2:.3f} vs {chi2:.3f}, diff {c2 - chi2:+.3f}")
    say("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    if args.out or args.name:
        _write_json(_outdir(args) / "check.json", {"ok": ok_all, "checks": rows, "t_build_s": t_build,
                                                   "t_first_eval_s": t_first, "t_slow_move_s": t_slow, "t_fast_move_s": t_fast})
    return 0 if ok_all else 1


# =============================================================================================
# loglike
# =============================================================================================
def cmd_loglike(args) -> int:
    say = _Say(args.quiet)
    th = _theta_from_args(args, "test")
    post = _build(args)
    t0 = time.perf_counter()
    lp, ll, d = api.log_post_with_derived(post, th)
    dt = time.perf_counter() - t0
    lprior = post.log_prior(th)
    res = {"theta": api.params.to_dict(th), "chi2": -2.0 * ll, "ln_like": ll, "ln_prior": lprior, "ln_post": lp,
           "derived": d, "time_s": dt, "backend": args.backend, "fidelity": args.fidelity}
    say("parameters:\n" + _theta_table(th))
    say(f"chi2      = {res['chi2']:.4f}   (N_data = 195)")
    say(f"ln L      = {ll:.4f}")
    say(f"ln prior  = {lprior:.4f}   (normalised)")
    say(f"ln post   = {lp:.4f}")
    say("derived   = " + ", ".join(f"{k} = {v:.5f}" for k, v in d.items()))
    say(f"time      = {dt:.2f} s (includes JIT/CAMB cold start)")
    if args.mu:
        mu = post.model.mu(th)
        np.savetxt(_outdir(args) / "mu.txt", mu, header="pure binned theory vector (195), data ordering")
    if args.json or args.out or args.name:
        _write_json(_outdir(args) / "loglike.json", res)
    if args.json_stdout:
        print(json.dumps(_jsonable(res)))
    return 0 if np.isfinite(lp) else 2


# =============================================================================================
# map
# =============================================================================================
def _start_points(args, n, post) -> list:
    rng = np.random.default_rng(args.seed)
    pts = [_theta_from_args(args, "scan")]
    if n > 1:
        pts.append(api.preset("start"))
    k = 0
    while len(pts) < n:
        cand = api.random_prior_draws(1, seed=int(rng.integers(1 << 30)), box=args.start_box)[0]
        cand[PARAM_NAMES.index("A_baryon")] = max(cand[PARAM_NAMES.index("A_baryon")], 2.0)
        if np.isfinite(post.log_post(cand)) or k > 50:
            pts.append(cand)
        k += 1
    return pts[:n]


def _run_map_start(post, job):
    """One MAP start.  With ``staged`` the cheap block (medium + fast parameters, ~1-17 ms per move) is optimised
    first with the three CAMB-transfer parameters frozen, which removes most of the work before the expensive
    full-space finite-difference gradients (3 new CAMB transfers per gradient) are needed."""
    from .inference.maxlike import find_map
    t0 = time.perf_counter()
    x0 = np.asarray(job["x0"], float).copy()
    n_eval = 0
    stage_a = None
    if job.get("staged"):
        free = np.concatenate([api.block_indices("medium"), api.block_indices("fast")])

        def sub(y):
            x = x0.copy()
            x[free] = y
            return post.log_post(x)

        ra = find_map(sub, params.BOUNDS[free], x0=x0[free], n_start=1, seed=job["seed"], maxiter=job["maxiter"],
                      polish=job["polish"], n_rounds=job["rounds"])
        x0[free] = ra.theta
        n_eval += ra.n_eval
        stage_a = float(ra.neg_log_post)
    r = find_map(post, params.BOUNDS, x0=x0, n_start=1, seed=job["seed"], maxiter=job["maxiter"],
                 polish=job["polish"], n_rounds=job["rounds"])
    return {"theta": r.theta, "neg_log_post": r.neg_log_post, "n_eval": r.n_eval + n_eval, "time": time.perf_counter() - t0,
            "x0": np.asarray(job["x0"]), "stage": r.starts[0]["stage"], "neg_log_post_after_stage_a": stage_a}


def cmd_map(args) -> int:
    from .inference.maxlike import numerical_hessian
    say = _Say(args.quiet)
    out = _outdir(args)
    post = _build(args)
    x0s = _start_points(args, args.starts, post)
    jobs = [dict(x0=x, seed=args.seed + i, maxiter=args.maxiter, polish=args.polish, rounds=args.rounds,
                 staged=not args.no_staged) for i, x in enumerate(x0s)]
    say(f"MAP: {len(jobs)} start(s), workers={args.workers}, polish={args.polish}, maxiter={args.maxiter}, rounds={args.rounds}")
    t0 = time.perf_counter()
    if args.workers > 1:
        with _ExecPool(args.workers, _model_kw(args), _lik_kw(args), args.worker_threads) as pool:
            results = pool.map(_w_map, jobs)
    else:
        results = []
        for j in jobs:
            r = _run_map_start(post, j)
            say(f"  start: -lnP = {r['neg_log_post']:.5f}  ({r['n_eval']} evals, {r['time']:.1f} s)")
            results.append(r)
    t_map = time.perf_counter() - t0
    best = min(results, key=lambda r: r["neg_log_post"])
    theta = np.asarray(best["theta"])
    funs = np.array([r["neg_log_post"] for r in results])
    say(f"best -lnP = {best['neg_log_post']:.5f}; starts within 0.05: {int(np.sum(funs < best['neg_log_post'] + 0.05))}/{len(funs)}; "
        f"spread of optima {funs.max() - funs.min():.4f}; {t_map:.1f} s")
    lp, ll, dv = api.log_post_with_derived(post, theta)
    say(f"chi2 = {-2 * ll:.3f}  S8 = {dv['S8']:.4f}  sigma8 = {dv['sigma8']:.4f}  Omega_m = {dv['omega_m']:.4f}")
    hess = None
    if not args.no_hessian:
        t1 = time.perf_counter()
        hr = numerical_hessian(post.log_post, theta, params.BOUNDS)
        hess = {"sigma": hr.sigma, "cov": hr.cov, "n_repaired": hr.n_repaired, "n_eval": hr.n_eval,
                "at_boundary": hr.at_boundary, "time_s": time.perf_counter() - t1}
        say(f"Hessian: {hr.n_eval} evals, {hess['time_s']:.1f} s, eigenvalues repaired: {hr.n_repaired}")
    say("MAP parameters (sigma from Hessian):")
    for i, n in enumerate(PARAM_NAMES):
        s = f"  +- {hess['sigma'][i]:.3g}" if hess else ""
        say(f"  {n:<12s} {theta[i]:14.8g}{s}")
    res = {"theta": params.to_dict(theta), "theta_vector": theta, "names": list(PARAM_NAMES), "chi2": -2 * ll,
           "ln_like": ll, "ln_post": lp, "ln_prior": lp - ll, "derived": dv, "hessian": hess,
           "starts": [{"x0": r["x0"], "theta": r["theta"], "neg_log_post": r["neg_log_post"], "n_eval": r["n_eval"],
                       "time_s": r["time"], "stage": r["stage"],
                       "neg_log_post_after_stage_a": r.get("neg_log_post_after_stage_a")} for r in results],
           "n_eval_total": int(sum(r["n_eval"] for r in results)), "time_s": t_map, "seed": args.seed,
           "settings": {"backend": args.backend, "fidelity": args.fidelity, "polish": args.polish, "staged": not args.no_staged,
                        "maxiter": args.maxiter, "rounds": args.rounds}}
    _write_json(out / "map.json", res)
    say(f"wrote {out / 'map.json'}")
    return 0


# =============================================================================================
# mcmc
# =============================================================================================
def _schedule(rep_slow, rep_med, rep_fast, joint=True):
    """Blocked-Metropolis schedule.  ``joint``: the first block moves the three CAMB-transfer parameters TOGETHER with
    (n_s, ln_1e10_A_s, A_baryon) in one proposal (same cost: one transfer function), which handles the omch2 - A_s - h
    degeneracy; otherwise (omch2, ombh2, h) move alone and the medium block follows."""
    sl, md, fa = api.block_indices("slow"), api.block_indices("medium"), api.block_indices("fast")
    first = np.concatenate([sl, md]) if joint else sl
    return [(first, rep_slow), (md, rep_med), (fa, rep_fast)]


def _run_blocked(post, job):
    from .inference.mcmc import blocked_metropolis
    post.reset_stats()
    t0 = time.perf_counter()
    r = blocked_metropolis(post.log_post, job["x0"], _schedule(*job["reps"], joint=job.get("joint", True)), n_cycles=job["cycles"],
                           n_burn_cycles=job["burn"], cov0=job["cov0"], bounds=params.BOUNDS, seed=job["seed"],
                           thin=job["thin"])
    dt = time.perf_counter() - t0
    d = api.derived_columns(post, r.samples)
    return {"samples": r.samples, "log_post": r.log_post, "burn_samples": r.burn_samples, "burn_log_post": r.burn_log_post,
            "derived": d, "accept": r.accept, "accept_blocks": r.accept_blocks, "scales": r.scales, "n_eval": r.n_eval,
            "time": dt, "cov": r.cov, "timing": post.timing_report()}


def _init_points(args, post, n):
    """n chain starting points: MAP/theta file (or preset 'scan'), then jittered copies."""
    x = _theta_from_args(args, "scan")
    if getattr(args, "map", None):
        x = api.load_theta_file(args.map, base=x)
    rng = np.random.default_rng(args.seed + 12345)
    width = params.BOUNDS[:, 1] - params.BOUNDS[:, 0]
    pts = [x]
    while len(pts) < n:
        for _ in range(200):
            y = np.clip(x + args.jitter * width * rng.standard_normal(len(x)), params.BOUNDS[:, 0], params.BOUNDS[:, 1])
            if np.isfinite(post.log_post(y)):
                pts.append(y)
                break
        else:
            raise SystemExit("could not find a finite-posterior jittered starting point")
    return pts


def _proposal_cov(args, say=None):
    """Proposal covariance prior: the MAP Hessian covariance if it is trustworthy, else 2 % of the box width.

    The Hessian is rejected when any parameter sits on a prior boundary at the MAP (``at_boundary``): the finite-difference
    curvature is then one-sided and the covariance can exceed the box (observed for the KV450 MAP, which lies at
    h = 0.82, ln_1e10_A_s = 5 and A_baryon = 2).  ``--force-hessian`` overrides.
    """
    from .inference.mcmc import default_proposal_cov
    if args.map:
        txt = json.loads(Path(args.map).read_text())
        h = txt.get("hessian") if isinstance(txt, dict) else None
        if h and h.get("cov") is not None:
            if any(h.get("at_boundary") or []) and not args.force_hessian:
                if say:
                    say("  MAP lies on a prior boundary: Hessian covariance not used for the proposal (use --force-hessian to override)")
            else:
                return np.asarray(h["cov"], float)
    return default_proposal_cov(params.BOUNDS, frac=0.02)


def _chain_stats(samples_by_chain, names):
    from .inference.mcmc import split_rhat, effective_sample_size
    x = np.stack(samples_by_chain, axis=1)               # (steps, chains, params)
    out = {}
    try:
        out["rhat"] = dict(zip(names, np.atleast_1d(split_rhat(x)).tolist()))
        out["ess"] = dict(zip(names, np.atleast_1d(effective_sample_size(x)).tolist()))
    except Exception as e:  # noqa: BLE001  (e.g. chains too short for split R-hat)
        out["diagnostics_error"] = str(e)
    return out


def _save(out: Path, stem, samples, names, weights, log_post, meta, log_like=None):
    from .inference.chains import Chain, save_chain
    c = Chain(samples, names, weights, log_post, log_like, meta)
    return save_chain(out / stem, c)


def cmd_mcmc(args) -> int:
    from .inference.chains import Chain, summary_table
    say = _Say(args.quiet)
    out = _outdir(args)
    post = _build(args)
    names = PARAM_NAMES + api.DERIVED_NAMES
    summary: dict = {"seed": args.seed, "method": args.method}
    cov0 = _proposal_cov(args, say)
    if args.method in ("blocked", "both"):
        nch = args.chains
        pts = _init_points(args, post, nch)
        reps = (1, args.rep_medium, args.rep_fast)
        jobs = [dict(x0=pts[i], seed=args.seed + 1000 * i, reps=reps, cycles=args.cycles, burn=args.burn,
                     cov0=cov0, thin=args.thin, joint=not args.separate_slow) for i in range(nch)]
        say(f"blocked Metropolis: {nch} chain(s), {args.burn}+{args.cycles} cycles each, repeats slow/med/fast = {reps}, joint slow+medium first block = {not args.separate_slow}, workers={args.workers}")
        t0 = time.perf_counter()
        if args.workers > 1 and nch > 1:
            with _ExecPool(min(args.workers, nch), _model_kw(args), _lik_kw(args), args.worker_threads) as pool:
                results = pool.map(_w_chain, jobs)
        else:
            results = []
            for i, j in enumerate(jobs):
                results.append(_run_blocked(post, j))
                say(f"  chain {i}: {results[-1]['time']:.1f} s, accept {results[-1]['accept']:.3f}")
        wall = time.perf_counter() - t0
        per = [np.hstack([r["samples"], r["derived"]]) for r in results]
        stats = _chain_stats(per, names)
        allx = np.vstack(per)
        allx_lp = np.concatenate([r["log_post"] for r in results])
        meta = {"sampler": "blocked_metropolis", "seed": args.seed, "chains": nch, "cycles": args.cycles,
                "burn": args.burn, "thin": args.thin, "repeats": reps, "joint_slow_medium_block": not args.separate_slow, "chain_lengths": [len(p) for p in per],
                "backend": args.backend, "fidelity": args.fidelity}
        npz, txt = _save(out, "chain", allx, names, None, allx_lp, meta)
        for i, r in enumerate(results):
            _save(out, f"chain_c{i}", per[i], names, None, r["log_post"], dict(meta, chain=i))
        bl = results[0]["timing"]
        summary["blocked"] = {"wall_s": wall, "accept": [r["accept"] for r in results],
                              "accept_blocks": [r["accept_blocks"] for r in results], "scales": [r["scales"] for r in results],
                              "n_eval": [r["n_eval"] for r in results], "chain_time_s": [r["time"] for r in results],
                              "ms_per_call_by_kind_chain0": bl["ms_per_call_by_kind"], "n_by_kind_chain0": bl["by_kind_n"],
                              "n_samples": int(len(allx)), **stats, "files": [str(npz), str(txt)]}
        say(f"  wall {wall:.1f} s; acceptance {', '.join(f'{r['accept']:.3f}' for r in results)}")
        if nch > 1 and "rhat" in stats:
            rh = stats["rhat"]
            say("  split R-hat: " + ", ".join(f"{k}={v:.3f}" for k, v in rh.items()))
        say(summary_table(Chain(allx, names, None, allx_lp)))
        (out / "summary_blocked.txt").write_text(summary_table(Chain(allx, names, None, allx_lp)) + "\n")
    if args.method in ("ensemble", "both"):
        from .inference.mcmc import ensemble_sampler
        nw = args.walkers
        if nw % 2:
            nw += 1
        rng = np.random.default_rng(args.seed + 777)
        x0 = _init_points(args, post, 1)[0]
        width = params.BOUNDS[:, 1] - params.BOUNDS[:, 0]
        sig = np.sqrt(np.diag(cov0)) if cov0.shape == (14, 14) else 0.01 * width
        p0 = []
        while len(p0) < nw:
            y = np.clip(x0 + 0.3 * sig * rng.standard_normal(14), params.BOUNDS[:, 0], params.BOUNDS[:, 1])
            if np.isfinite(post.log_post(y)):
                p0.append(y)
        p0 = np.array(p0)
        say(f"ensemble: {nw} walkers x {args.steps} steps, workers={args.workers}")
        t0 = time.perf_counter()
        table = {}
        if args.workers > 1:
            with _ExecPool(args.workers, _model_kw(args), _lik_kw(args), args.worker_threads) as pool:
                def f(Y):
                    res = pool.map(_w_logpost, list(Y))
                    for t, (lp, d) in zip(Y, res):
                        if np.isfinite(lp):
                            table[api.DerivedRecorder._key(t)] = d
                    return np.array([r[0] for r in res])
                r = ensemble_sampler(f, p0, args.steps, vectorized=True, seed=args.seed)
        else:
            r = ensemble_sampler(post.log_post, p0, args.steps, vectorized=False, seed=args.seed)
        wall = time.perf_counter() - t0
        burn = int(args.burn_frac * args.steps)
        flat, lp = r.flat(burn=burn)
        if table:
            post.recorder.table.update({k: v for k, v in table.items()})
        d = api.derived_columns(post, flat)
        allx = np.hstack([flat, d])
        meta = {"sampler": "ensemble", "seed": args.seed, "walkers": nw, "steps": args.steps, "burn": burn,
                "backend": args.backend, "fidelity": args.fidelity}
        npz, txt = _save(out, "chain_ensemble", allx, names, None, lp, meta)
        from .inference.mcmc import integrated_autocorr_time
        try:
            tau = np.atleast_1d(integrated_autocorr_time(r.chain[burn:][..., :14], tol=0))
        except Exception as e:  # noqa: BLE001
            tau = [float("nan")]
        summary["ensemble"] = {"wall_s": wall, "n_eval": r.n_eval, "acceptance_mean": float(np.mean(r.acceptance_fraction)),
                               "tau_max_steps": float(np.nanmax(tau)), "n_samples": int(len(allx)), "files": [str(npz), str(txt)]}
        say(f"  wall {wall:.1f} s, {r.n_eval} evals, mean acceptance {np.mean(r.acceptance_fraction):.3f}, max tau {np.nanmax(tau):.1f} steps")
        txt_tab = summary_table(Chain(allx, names, None, lp))
        say(txt_tab)
        (out / "summary_ensemble.txt").write_text(txt_tab + "\n")
    _write_json(out / "mcmc_summary.json", summary)
    say(f"wrote {out}")
    return 0


# =============================================================================================
# nested
# =============================================================================================
def cmd_nested(args) -> int:
    from .inference.chains import Chain, summary_table
    from .inference.nested import run_dynesty, run_nautilus
    say = _Say(args.quiet)
    out = _outdir(args)
    post = _build(args)
    ndim = post.ndim
    t0 = time.perf_counter()
    pool = None
    table = None
    try:
        if args.workers > 1:
            cls = _EvalPool if args.sampler == "nautilus" else _ExecPool
            pool = cls(args.workers, _model_kw(args), _lik_kw(args), args.worker_threads)
        say(f"nested sampling with {args.sampler}, workers={args.workers}")
        if args.sampler == "nautilus":
            res = run_nautilus(post.log_like, post.prior_transform, ndim, n_live=args.nlive, n_eff=args.n_eff,
                               f_live=args.f_live, seed=args.seed,
                               pool=(pool, None) if pool is not None else None, n_like_max=args.max_calls or np.inf,
                               timeout=args.timeout or np.inf, verbose=not args.quiet,
                               filepath=str(out / "nautilus.hdf5") if args.checkpoint else None, resume=args.checkpoint)
            table = pool.table if pool is not None else None
        else:
            kw = {}
            if pool is not None:
                kw = dict(pool=pool, queue_size=args.workers)
            res = run_dynesty(_w_loglike if pool is not None else post.log_like,
                              params.prior_transform if pool is not None else post.prior_transform, ndim, nlive=args.nlive,
                              dynamic=args.dynamic, dlogz=args.dlogz, seed=args.seed, maxcall=args.max_calls or None,
                              verbose=not args.quiet, **kw)
        wall = time.perf_counter() - t0
        if table:
            post.recorder.table.update(table)
        samples = res.samples
        need = {api.DerivedRecorder._key(t) for t in samples} - set(post.recorder.table)
        if need and pool is not None:
            say(f"computing derived quantities for {len(need)} distinct cosmologies on the workers ...")
            keys = list(need)
            arrs = [np.array(k + (0.0,) * 9) for k in keys]
            chunks = np.array_split(np.array(arrs), max(1, 4 * args.workers))
            if pool is not None:
                outs = _ExecPool.map(pool, _w_derived, [c for c in chunks if len(c)])
                vals = np.vstack(outs)
                for k, v in zip(keys, vals):
                    post.recorder.table[k] = tuple(v)
        d = api.derived_columns(post, samples)
        lpost = res.log_post(post.log_prior)
        names = PARAM_NAMES + api.DERIVED_NAMES
        meta = {"sampler": res.sampler, "log_z": res.log_z, "log_z_err": res.log_z_err, "n_eval": res.n_eval,
                "ess_kish": res.ess, "seed": args.seed, "nlive": args.nlive, "backend": args.backend, "fidelity": args.fidelity,
                "note": "weights are posterior weights; log_post = ln prior + ln L; log_z uses the normalised priors of params.py"}
        allx = np.hstack([samples, d])
        npz, txt = _save(out, "chain", allx, names, res.weights, lpost, meta, log_like=res.log_like)
        ch = Chain(allx, names, res.weights, lpost, res.log_like, meta)
        tab = summary_table(ch)
        (out / "summary.txt").write_text(tab + "\n")
        _write_json(out / "nested_summary.json", {**meta, "wall_s": wall, "n_samples": len(allx), "info": res.info,
                                                  "max_loglike": float(res.log_like.max()), "min_chi2": float(-2 * res.log_like.max()),
                                                  "files": [str(npz), str(txt)]})
        say(f"ln Z = {res.log_z:.3f} +- {res.log_z_err:.3f}   n_eval = {res.n_eval}   Kish ESS = {res.ess:.0f}   wall {wall:.1f} s")
        say(tab)
    finally:
        if pool is not None:
            pool.close()
    return 0


# =============================================================================================
# bench
# =============================================================================================
def cmd_bench(args) -> int:
    say = _Say(args.quiet)
    th0 = _theta_from_args(args, "scan")
    t0 = time.perf_counter()
    post = _build(args)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    post.log_post(th0)
    t_cold = time.perf_counter() - t0
    say(f"build {t_build:.2f} s; first (cold, includes JIT load) ln-post evaluation {t_cold:.2f} s")
    width = params.BOUNDS[:, 1] - params.BOUNDS[:, 0]
    rows = []
    kind_of = {n: k for k, ns in api.BLOCKS.items() for n in ns}
    for i, n in enumerate(PARAM_NAMES):
        post.model.reset_timing()
        post.reset_stats()
        post.log_post(th0)
        post.model.reset_timing()
        ts = []
        nrep = args.n_slow if kind_of[n] == "slow" else (args.n_medium if kind_of[n] == "medium" else args.n_fast)
        for r in range(nrep):
            th = th0.copy()
            th[i] = th0[i] + (-1) ** r * 1e-3 * width[i] * (1 + r)     # every point new, differs from the previous in i only
            th[i] = min(max(th[i], params.BOUNDS[i, 0]), params.BOUNDS[i, 1])
            t1 = time.perf_counter()
            post.log_post(th)
            ts.append(time.perf_counter() - t1)
        tm = post.model.timing()
        rows.append({"param": n, "block": kind_of[n], "n": nrep, "mean_ms": 1e3 * float(np.mean(ts)), "median_ms": 1e3 * float(np.median(ts)),
                     "stage_ms": {k: 1e3 * tm[k] / nrep for k in ("linear", "pnl", "limber_power", "limber_cl", "hankel")}})
        say(f"  {n:<12s} {kind_of[n]:<7s} mean {rows[-1]['mean_ms']:9.3f} ms  median {rows[-1]['median_ms']:9.3f} ms   "
            + "  ".join(f"{k}={v:.2f}" for k, v in rows[-1]["stage_ms"].items()))
    # likelihood-only cost
    mu = post.model.mu(th0)
    tl = time.perf_counter()
    for _ in range(2000):
        post.likelihood.log_like(mu, 1.0, 0.0)
    t_like = (time.perf_counter() - tl) / 2000
    say(f"  likelihood alone: {1e3 * t_like:.4f} ms")
    res = {"build_s": t_build, "cold_first_eval_s": t_cold, "per_param": rows, "likelihood_ms": 1e3 * t_like,
           "backend": args.backend, "fidelity": args.fidelity, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
           "camb_threads": post.model.linear.nthreads if hasattr(post.model.linear, "nthreads") else None}
    blk = {}
    for k in ("slow", "medium", "fast"):
        v = [r["mean_ms"] for r in rows if r["block"] == k]
        blk[k] = float(np.mean(v))
    res["block_mean_ms"] = blk
    say("block means [ms]: " + ", ".join(f"{k} {v:.3f}" for k, v in blk.items()))
    if args.workers > 1:
        # throughput of the worker pool on new cosmologies
        n = args.workers * 3
        pts = api.random_prior_draws(n, seed=args.seed, box=0.3)
        pts[:, PARAM_NAMES.index("A_baryon")] = 3.0
        with _ExecPool(args.workers, _model_kw(args), _lik_kw(args), args.worker_threads) as pool:
            pool.map(_w_loglike, list(pts[: args.workers]))                    # warm-up (JIT load)
            t1 = time.perf_counter()
            pool.map(_w_loglike, list(pts[args.workers:]))
            dt = time.perf_counter() - t1
        res["pool"] = {"workers": args.workers, "threads_per_worker": args.worker_threads, "n_new_cosmologies": n - args.workers,
                       "wall_s": dt, "throughput_per_s": (n - args.workers) / dt}
        say(f"worker pool: {res['pool']['throughput_per_s']:.2f} new-cosmology evaluations/s with {args.workers} workers")
    if args.out or args.name:
        _write_json(_outdir(args) / "bench.json", res)
    return 0


# =============================================================================================
# parser
# =============================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kidspipe", description="kidspipe: clean-room KV450 cosmic-shear (xi+/-) inference pipeline.",
        epilog="Run `python -m kidspipe <command> --help` for the options of each command. See docs/USAGE.md.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{check,loglike,map,mcmc,nested,bench}")

    c = sub.add_parser("check", help="self-test: JIT warm-up, chi2 at the spec test cosmology vs the anchor, timings",
                       description="Self-test. Exit status 0 if all checks pass.")
    _add_model_args(c)
    _add_common(c, seed=False)
    c.add_argument("--tol", type=float, default=0.5, help="allowed |chi2 - 196.06| at the test cosmology (default 0.5)")
    c.add_argument("--scan", action="store_true", help="also check the amplitude scan of the spec (4 evaluations)")
    c.add_argument("--scan-tol", type=float, default=1.0, help="tolerance for --scan (default 1.0)")
    c.add_argument("--camb", action="store_true", help="also compare to the other non-linear backend (|diff| < 0.3)")
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("loglike", help="chi2, ln L, ln post (and sigma8/Omega_m/S8) at one parameter point",
                       description="Evaluate the posterior at a parameter point. Base point: the spec test cosmology "
                                   "(--preset test, nominal systematics) unless --preset says otherwise; --theta-file and "
                                   "--set override it.")
    _add_model_args(c)
    _add_common(c, seed=False)
    _add_theta_args(c, "test")
    c.add_argument("--json", action="store_true", help="write loglike.json into the output directory")
    c.add_argument("--json-stdout", action="store_true", help="print the result as one JSON line")
    c.add_argument("--mu", action="store_true", help="also write the 195-vector theory prediction to mu.txt")
    c.set_defaults(func=cmd_loglike)

    c = sub.add_parser("map", help="multi-start MAP + Hessian covariance -> map.json",
                       description="Multi-start bounded L-BFGS-B (+ polish) maximisation of ln posterior, then a "
                                   "finite-difference Hessian. Each start is one job; --workers runs them in parallel.")
    _add_model_args(c)
    _add_common(c)
    _add_theta_args(c, "scan")
    _add_pool_args(c)
    c.add_argument("--starts", type=int, default=4, help="number of starts: 'scan' preset, KV450 start, random prior draws (default 4)")
    c.add_argument("--start-box", type=float, default=0.6, help="central fraction of the prior for random starts (default 0.6)")
    c.add_argument("--maxiter", type=int, default=150, help="L-BFGS-B iterations per round (default 150)")
    c.add_argument("--rounds", type=int, default=2, help="L-BFGS-B/polish rounds per start (default 2)")
    c.add_argument("--polish", choices=("none", "nelder-mead", "powell", "both"), default="powell",
                   help="derivative-free polish after L-BFGS-B (default powell)")
    c.add_argument("--no-staged", action="store_true",
                   help="skip the cheap-subspace pre-optimisation (medium+fast parameters with omch2, ombh2, h frozen)")
    c.add_argument("--no-hessian", action="store_true", help="skip the Hessian")
    c.set_defaults(func=cmd_map)

    c = sub.add_parser("mcmc", help="blocked adaptive Metropolis (fast/medium/slow blocks) and/or ensemble sampler",
                       description="Chains are written as chain.npz/.txt (columns: 14 parameters + sigma8, omega_m, S8; "
                                   "ln post per sample). Blocked: one cycle = one joint update of the "
                                   "6 parameters (omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon) (one CAMB transfer; --separate-slow moves "
                                   "omch2, ombh2, h alone), then --rep-medium updates of (n_s, ln_1e10_A_s, A_baryon), then --rep-fast "
                                   "updates of the 8 fast parameters.")
    _add_model_args(c)
    _add_common(c)
    _add_theta_args(c, "scan")
    _add_pool_args(c)
    c.add_argument("--method", choices=("blocked", "ensemble", "both"), default="blocked")
    c.add_argument("--map", default=None, help="map.json from `kidspipe map`: start point and proposal covariance (Hessian)")
    c.add_argument("--force-hessian", action="store_true", help="use the --map Hessian covariance even if the MAP is on a prior boundary")
    c.add_argument("--chains", type=int, default=1, help="independent blocked chains (default 1); with --workers they run in parallel")
    c.add_argument("--cycles", type=int, default=500, help="production cycles per chain (default 500)")
    c.add_argument("--burn", type=int, default=200, help="burn-in cycles (adaptation) per chain; returned separately, not saved in chain.npz")
    c.add_argument("--thin", type=int, default=1)
    c.add_argument("--separate-slow", action="store_true",
                   help="move (omch2, ombh2, h) alone in the expensive block instead of jointly with (n_s, ln_1e10_A_s, A_baryon)")
    c.add_argument("--rep-medium", type=int, default=2, help="updates of the medium block per cycle (default 2)")
    c.add_argument("--rep-fast", type=int, default=6, help="updates of the fast block per cycle (default 6)")
    c.add_argument("--jitter", type=float, default=0.01, help="start-point scatter of extra chains, fraction of the box width (default 0.01)")
    c.add_argument("--walkers", type=int, default=36, help="ensemble walkers (default 36)")
    c.add_argument("--steps", type=int, default=200, help="ensemble steps (default 200)")
    c.add_argument("--burn-frac", type=float, default=0.5, help="ensemble: fraction of steps discarded (default 0.5)")
    c.set_defaults(func=cmd_mcmc)

    c = sub.add_parser("nested", help="nested sampling (nautilus or dynesty) -> weighted chain, ln Z",
                       description="Samples the prior-transform cube. Every nested-sampling point is effectively a new "
                                   "cosmology, so cost is ~ n_eval x (CAMB transfer time); use --workers.")
    _add_model_args(c)
    _add_common(c)
    _add_pool_args(c)
    c.add_argument("--sampler", choices=("nautilus", "dynesty"), default="nautilus")
    c.add_argument("--nlive", type=int, default=500, help="live points (nautilus n_live / dynesty nlive; default 500)")
    c.add_argument("--n-eff", type=float, default=2000, help="nautilus: target effective sample size (default 2000)")
    c.add_argument("--f-live", type=float, default=0.01, help="nautilus: live evidence fraction to stop exploration (default 0.01)")
    c.add_argument("--dlogz", type=float, default=0.5, help="dynesty: stopping tolerance on remaining evidence (default 0.5)")
    c.add_argument("--dynamic", action="store_true", help="dynesty: dynamic nested sampling")
    c.add_argument("--max-calls", type=int, default=0, help="stop after this many likelihood calls (0 = unlimited)")
    c.add_argument("--timeout", type=float, default=0, help="nautilus: stop after this many seconds (0 = unlimited)")
    c.add_argument("--checkpoint", action="store_true", help="nautilus: write/resume <out>/nautilus.hdf5")
    c.set_defaults(func=cmd_nested)

    c = sub.add_parser("bench", help="timing per parameter / per block (fast, medium, slow moves) and per model stage",
                       description="Moves one parameter at a time away from a base point, every point new, so each "
                                   "timing is a cold evaluation of that block with all other stages cached.")
    _add_model_args(c)
    _add_common(c)
    _add_theta_args(c, "scan")
    _add_pool_args(c)
    c.add_argument("--n-slow", type=int, default=6)
    c.add_argument("--n-medium", type=int, default=30)
    c.add_argument("--n-fast", type=int, default=300)
    c.set_defaults(func=cmd_bench)
    return p


def _run_mpi(args) -> int:
    """Every rank parses the same argv.  Rank 0 runs the command with an MPI pool; the other ranks serve it."""
    from . import mpi
    nworkers = mpi.require_workers(2)
    init = (_model_kw(args), _lik_kw(args), args.worker_threads)
    with mpi.executor(_w_init, init, threads=args.worker_threads) as ex:
        if ex is None:                       # worker rank: the root has finished the command
            return 0
        if args.workers not in (1, nworkers):
            print(f"note: --workers {args.workers} ignored under --mpi; using the {nworkers} worker ranks", file=sys.stderr)
        args.workers = nworkers
        _MPI.update(ex=ex, nworkers=nworkers, init=init)
        try:
            return int(args.func(args) or 0)
        finally:
            _MPI.clear()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "threads", None):
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        if not _NUMBA_USER_SET:
            os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    if getattr(args, "mpi", False):
        return _run_mpi(args)
    from . import mpi
    mpi.refuse_unflagged_launch("--mpi")
    return int(args.func(args) or 0)


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
