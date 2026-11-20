"""Shared helpers for the scripts/run_kv450_*.py analysis drivers (emulator + hmcode, 'fast' fidelity, spec grids).

Not a public API: builds the posterior with the options used throughout runs/kv450_final/, a bounded
fixed-parameter / penalised optimiser wrapper, and small chain helpers.

Paths (kidspipe.paths): results are written to ``OUT`` = ``<runs>/kv450_final`` with ``<runs>`` = $KIDSPIPE_RUNS (default <root>/runs).  Results used
as INPUTS (MAP, pilot, production chains, ...) are read with ``find(rel)`` / ``result_glob(pattern)``: ``OUT`` first, else the copy shipped in
<root>/runs/kv450_final.  ``NCPU`` = os.cpu_count() is the default for every ``--workers`` option of the drivers.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one core per worker process (set before numpy / numba are imported)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths                          # noqa: E402

OUT = paths.runs_dir() / "kv450_final"
OUT.mkdir(parents=True, exist_ok=True)
NCPU = os.cpu_count() or 1                          # default worker count of the drivers


def find(rel) -> Path:
    """Existing result ``kv450_final/<rel>`` used as an input: yours (``OUT``), else the shipped copy."""
    return paths.find_result(Path("kv450_final") / rel)


def result_glob(pattern) -> list:
    """Sorted files matching ``pattern`` (relative to kv450_final): yours if any, else the shipped ones."""
    for base in (OUT, paths.shipped_runs_dir() / "kv450_final"):
        fs = sorted(glob.glob(str(base / pattern)))
        if fs:
            return fs
    return []


from kidspipe import api, params                    # noqa: E402
from kidspipe.params import PARAM_NAMES, BOUNDS     # noqa: E402

IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
EMU_BOX = {"omch2": (0.03, 0.34), "ombh2": (0.019, 0.026), "h": (0.64, 0.82)}


def make_post(linear="emulator", backend="hmcode", fidelity="fast", z_grid="spec", mcorr=True, cterm=True, threads=1,
              cache_size=4):
    """Posterior with the requested model options (defaults = the production 'fast' configuration)."""
    model = api.build_model(linear_backend=linear, pnl_backend=backend, fidelity=fidelity, z_grid=z_grid,
                            nthreads=threads, cache_size=cache_size)
    lik = api.build_likelihood(apply_m=mcorr, use_cterm=cterm)
    return api.build_posterior(model, lik)


def in_emu_box(theta) -> np.ndarray:
    th = np.atleast_2d(theta)
    ok = np.ones(len(th), bool)
    for n, (lo, hi) in EMU_BOX.items():
        ok &= (th[:, IDX[n]] >= lo) & (th[:, IDX[n]] <= hi)
    return ok


def derived(post, th):
    d = api.derived_for(post, th)
    return d["sigma8"], d["omega_m"], d["S8"]


def optimise(post, x0, fixed=None, s8_target=None, s8_sigma=0.002, maxiter=300, rounds=3, polish="powell", seed=0,
             eps=1e-6):
    """Maximise ln P from x0 with an optional dict ``fixed`` {name: value} of frozen parameters and an optional
    Gaussian penalty -0.5 ((S8 - s8_target)/s8_sigma)^2 (profile in S8).  Returns (theta, lnP_without_penalty)."""
    from kidspipe.inference import find_map
    x0 = np.array(x0, float)
    fixed = dict(fixed or {})
    for n, v in fixed.items():
        x0[IDX[n]] = v
    free = np.array([i for i, n in enumerate(PARAM_NAMES) if n not in fixed])

    def full(y):
        t = x0.copy()
        t[free] = y
        return t

    def f(y):
        t = full(y)
        lp = post.log_post(t)
        if not np.isfinite(lp):
            return -np.inf
        if s8_target is not None:
            lp -= 0.5 * ((derived(post, t)[2] - s8_target) / s8_sigma) ** 2
        return lp

    r = find_map(f, BOUNDS[free], x0=x0[free], n_start=1, seed=seed, maxiter=maxiter, n_rounds=rounds, polish=polish)
    t = full(r.theta)
    return t, float(post.log_post(t)), r


def chi2_of(post, th):
    return float(-2.0 * post.log_like(th))


def jdump(path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))
