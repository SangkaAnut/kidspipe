"""End-to-end tests: spec anchors, staged-cache correctness, fidelity / backend agreement, robustness, CLI.

Fast tests build a few models (~1-2 s each, numba code is cached on disk after the first run); the long
variants are marked ``slow``.  Run all fast ones with ``pytest -q -m "not slow" tests/test_e2e.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kidspipe import api, params
from kidspipe.params import PARAM_NAMES
from kidspipe.theory.model import CosmicShearModel

ROOT = Path(__file__).resolve().parents[1]
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def chi2_of(post, theta):
    return -2.0 * post.log_like(theta)


def near_fit_draws(n, seed):
    """Parameter vectors scattered around the amplitude-scan optimum (chi2 ~ 185-230): all 14 parameters vary,
    systematics are drawn from their priors.  (Whole-prior draws have chi2 ~ 1e3-1e6 where the agreement of two
    numerical variants in absolute delta-chi2 is not a meaningful criterion.)"""
    rng = np.random.default_rng(seed)
    base = api.preset("scan")
    u = rng.uniform(size=(n, 14))
    pri = params.prior_transform(u)
    box = {"omch2": (0.09, 0.15), "ombh2": (0.0215, 0.0235), "h": (0.64, 0.76), "n_s": (0.9, 1.05),
           "ln_1e10_A_s": (2.65, 2.95), "A_baryon": (2.0, 3.13), "A_IA": (-1.0, 1.5)}
    out = np.tile(base, (n, 1))
    for j in range(n):
        for k, (lo, hi) in box.items():
            out[j, IDX[k]] = lo + (hi - lo) * u[j, IDX[k]]
        for k in ("delta_c", "A_c", "dz1", "dz2", "dz3", "dz4", "dz5"):
            out[j, IDX[k]] = pri[j, IDX[k]]
    return out


@pytest.fixture(scope="module")
def post():
    return api.build_posterior()


# =============================================================================================
# (a) spec anchors
# =============================================================================================
def test_chi2_spec_test_cosmology(post):
    """spec s4.1: chi2 = 196.06 (mead, A_baryon=2) at the test cosmology; tolerance 0.5."""
    c = chi2_of(post, api.preset("test"))
    assert abs(c - api.ANCHOR_TEST_CHI2) < 0.5, c


def test_derived_quantities_test_cosmology(post):
    d = api.derived_for(post, api.preset("test"))
    assert d["sigma8"] == pytest.approx(0.7090923, abs=2e-5)
    assert d["omega_m"] == pytest.approx(0.2502956, abs=2e-6)
    assert d["S8"] == pytest.approx(0.6476924, abs=2e-5)
    assert d["S8"] == pytest.approx(d["sigma8"] * np.sqrt(d["omega_m"] / 0.3), rel=1e-12)


def test_amplitude_scan(post, record_property):
    """spec s4.2 / s15.2 amplitude scan; tolerance ~1 in chi2 (observed max |diff| ~0.56 at lnAs=3.2)."""
    diffs = {}
    for lnA, exp in api.ANCHOR_SCAN.items():
        th = api.theta_from(dict(api.AMP_SCAN_BASE, ln_1e10_A_s=lnA))
        diffs[lnA] = chi2_of(post, th) - exp
    record_property("scan_diffs", json.dumps(diffs))
    assert max(abs(v) for v in diffs.values()) < 1.0, diffs


def test_a_c_and_delta_c_reach_likelihood(post):
    """Guard for the silent-fixing bug: A_c and delta_c in theta must change ln L (and by the right amount)."""
    th = api.preset("test")
    mu = post.model.mu(th)
    lik = post.likelihood
    base = post.log_like(th)
    for name, val, kw in (("A_c", 1.15, dict(A_c=1.15)), ("delta_c", 4e-4, dict(delta_c=4e-4))):
        t = th.copy()
        t[IDX[name]] = val
        assert post.log_like(t) == pytest.approx(lik.log_like(mu, **kw), rel=1e-12, abs=1e-12)
        assert post.log_like(t) != base


# =============================================================================================
# (b) staged-cache correctness
# =============================================================================================
def _fresh_mu(theta, **kw):
    return CosmicShearModel(**kw).mu(theta)


def _variants(th, rng):
    """(identical, fast-perturbed, medium-perturbed) copies of a draw."""
    fast = th.copy()
    fast[IDX["A_IA"]] = np.clip(th[IDX["A_IA"]] + 0.7, -5.9, 5.9)
    fast[IDX["dz2"]] = th[IDX["dz2"]] + 0.01
    fast[IDX["A_c"]] = th[IDX["A_c"]] + 0.05
    med = th.copy()
    med[IDX["n_s"]] = th[IDX["n_s"]] + 0.02
    med[IDX["ln_1e10_A_s"]] = th[IDX["ln_1e10_A_s"]] - 0.05
    med[IDX["A_baryon"]] = min(3.13, th[IDX["A_baryon"]] + 0.1) if th[IDX["A_baryon"]] < 3.0 else th[IDX["A_baryon"]] - 0.1
    return th.copy(), fast, med


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


def test_staged_cache_matches_fresh_model(record_property):
    """15 prior draws (central 50 % box), a long interleaved history through ONE model (cache_size=4 forces
    evictions of the P_nl stage, 8 CAMB entries are exceeded -> linear-stage evictions as well), then 15 checks
    against freshly constructed models.

    Tolerances: points whose (omch2, ombh2, h, n_s, lnAs, A_baryon) were seen first at the same theta and fast
    (A_IA, A_c, dz) variants must agree to 1e-12 relative; points reached through the exact A_s / n_s rescale of a
    cached CAMB transfer function (medium variants) to 1e-10 (observed 6e-16 here, up to 1.4e-11 with larger medium-block changes: HMcode's iterative non-linear-scale
    solve amplifies the ~1e-16 rescale difference)."""
    rng = np.random.default_rng(11)
    draws = api.random_prior_draws(15, seed=3, box=0.5)
    long = CosmicShearModel()
    triples = [_variants(t, rng) for t in draws]
    # history: first visits in shuffled order, then random revisits of random variants, interleaved
    for i in rng.permutation(15):
        long.mu(triples[i][0])
    hist = [(i, k) for i in range(15) for k in range(3)]
    for j in rng.permutation(len(hist)):
        i, k = hist[j]
        long.mu(triples[i][k])
    # checks: end of a messy history, random order, a different variant kind per draw
    kinds = np.array([0, 1, 2] * 5)
    rng.shuffle(kinds)
    worst = {0: 0.0, 1: 0.0, 2: 0.0}
    for i in rng.permutation(15):
        k = int(kinds[i])
        t = triples[i][k]
        mu_long = long.mu(t)
        mu_fresh = _fresh_mu(t)
        worst[k] = max(worst[k], _rel(mu_long, mu_fresh))
    record_property("worst_rel_identical_fast_medium", json.dumps(worst))
    assert worst[0] <= 1e-12 and worst[1] <= 1e-12, worst
    assert worst[2] <= 1e-10, worst


def test_cache_cheap_moves_do_not_touch_expensive_stages():
    """A_IA / dz / A_c moves must not trigger linear or P_nl work (stage timers do not advance)."""
    m = CosmicShearModel()
    th = api.preset("test")
    m.mu(th)
    before = dict(m.timing())
    for k in range(5):
        t = th.copy()
        t[IDX["A_IA"]] = 0.1 * (k + 1)
        t[IDX["dz3"]] = 0.002 * k
        m.mu(t)
    after = m.timing()
    assert after["linear"] == before["linear"] and after["pnl"] == before["pnl"]
    assert after["limber_power"] == before["limber_power"]
    assert after["limber_cl"] > before["limber_cl"]


def test_pure_theory_vector_independent_of_c_term_parameters():
    """mu() must be pure theory: unaffected by A_c / delta_c (they are added by the likelihood)."""
    m = CosmicShearModel()
    th = api.preset("test")
    a = m.mu(th)
    t = th.copy()
    t[IDX["A_c"]], t[IDX["delta_c"]] = 1.3, 5e-4
    assert np.array_equal(m.mu(t), a)


# =============================================================================================
# (c) fast vs kcap fidelity
# =============================================================================================
def test_fast_vs_kcap_fidelity(post, record_property):
    """|delta chi2| < 0.1 between the exact-operator ('fast') and FFT-Hankel ('kcap') paths at points with chi2 < 198
    (within ~18 of the best fit); the difference grows to 0.2 at chi2 ~ 204 and ~0.5 at chi2 ~ 240 (reported, bounded by 1)."""
    kc = api.build_posterior(api.build_model(fidelity="kcap"))
    pts = [api.preset("test"), api.preset("scan")] + list(near_fit_draws(6, seed=21))
    far = api.theta_from(dict(api.AMP_SCAN_BASE, ln_1e10_A_s=3.2))      # chi2 ~ 239: well outside the posterior
    report = []
    for t in pts + [far]:
        mu_f, mu_k = post.model.mu(t), kc.model.mu(t)
        c_f, c_k = chi2_of(post, t), chi2_of(kc, t)
        rel = np.abs(mu_f - mu_k) / np.abs(mu_k)
        report.append({"chi2_fast": c_f, "chi2_kcap": c_k, "dchi2": abs(c_f - c_k), "rel_max": float(rel.max()),
                       "rel_median": float(np.median(rel)), "rel_p95": float(np.percentile(rel, 95))})
    record_property("fidelity", json.dumps(report))
    print("\nfast vs kcap:", json.dumps(report, indent=1))
    # measured: |dchi2| ~ 0.03 for chi2 ~ 189-196, 0.21 at 204, 0.38 at 224, 0.5 at 239-249 (the bin-average vs node-trapezoid
    # difference of ~0.2 % median / 0.7 % max in mu is coherent, so d chi2 scales with the residual).
    good = [r for r in report if r["chi2_fast"] < 198.0]
    assert len(good) >= 3
    assert max(r["dchi2"] for r in good) < 0.1, good
    assert max(r["dchi2"] for r in report) < 1.0, report
    assert max(r["rel_max"] for r in report) < 0.02 and max(r["rel_median"] for r in report) < 0.005
    # known/expected: the difference grows with chi2 (at chi2 ~ 239, d chi2 ~ 0.5); only documented, not asserted


# =============================================================================================
# (d) hmcode vs camb backend
# =============================================================================================
def test_hmcode_vs_camb_backend(post, record_property):
    cb = api.build_posterior(api.build_model(pnl_backend="camb"))
    pts = [api.preset("test"), api.preset("scan")] + list(near_fit_draws(6, seed=5))
    rows = []
    for t in pts:
        c1, c2 = chi2_of(post, t), chi2_of(cb, t)
        rows.append({"chi2_hmcode": c1, "chi2_camb": c2, "diff": c1 - c2})
    record_property("backend", json.dumps(rows))
    print("\nhmcode vs camb:", json.dumps(rows, indent=1))
    assert max(abs(r["diff"]) for r in rows) < 0.3, rows


# =============================================================================================
# (e) robustness of Posterior
# =============================================================================================
def test_posterior_returns_minus_inf_cleanly(post):
    good = api.preset("test")
    ref = post.log_post(good)
    assert np.isfinite(ref)
    bad = []
    for i, n in enumerate(PARAM_NAMES):
        for edge, fac in ((0, -1e-3), (1, 1e-3)):
            t = good.copy()
            lo, hi = params.BOUNDS[i]
            t[i] = (lo + fac * (hi - lo)) if edge == 0 else (hi + fac * (hi - lo))
            bad.append(t)
        for v in (np.nan, np.inf, -np.inf):
            t = good.copy()
            t[i] = v
            bad.append(t)
    bad.append(np.full(14, np.nan))
    bad.append(np.full(14, 1e300))
    for t in bad:
        assert post.log_post(t) == -np.inf
        assert post.log_like(t) == -np.inf
        e = post.eval(t)
        assert e == (-np.inf, -np.inf)
        assert post.log_prior(t) == -np.inf
    # the stream of -inf must not corrupt state
    assert post.log_post(good) == pytest.approx(ref, abs=1e-12)
    # batch interface
    out = post.log_post_batch(np.vstack([good, bad[0], good]))
    assert out[0] == out[2] and out[1] == -np.inf
    # wrong length is a programming error and documented to raise ValueError (not silently broadcast)
    with pytest.raises(ValueError):
        post.log_post(good[:5])


def test_posterior_maps_model_failures_to_minus_inf(post):
    """A model that raises / returns NaN must give ln L = -inf, never propagate."""
    from kidspipe.inference import Posterior

    def boom(theta):
        raise RuntimeError("camb failed")

    p1 = Posterior(boom, post.likelihood, params)
    assert p1.log_post(api.preset("test")) == -np.inf and p1.stats.n_fail == 1
    p2 = Posterior(lambda t: np.full(195, np.nan), post.likelihood, params)
    assert p2.log_post(api.preset("test")) == -np.inf


def test_edge_of_box_is_finite_and_inside(post):
    """Points on the closed box boundary are valid (prior support is closed) as long as the physics is fine."""
    t = api.preset("test")
    t[IDX["A_baryon"]] = 3.13        # upper edge
    t[IDX["A_c"]] = params.BOUNDS[IDX["A_c"], 0]
    assert np.isfinite(post.log_post(t))


# =============================================================================================
# api helpers
# =============================================================================================
def test_theta_file_formats(tmp_path):
    j = tmp_path / "t.json"
    j.write_text(json.dumps({"omch2": 0.15, "A_IA": 1.5}))
    th = api.load_theta_file(j, base="test")
    assert th[IDX["omch2"]] == 0.15 and th[IDX["A_IA"]] == 1.5 and th[IDX["h"]] == 0.7
    i = tmp_path / "t.ini"
    i.write_text("# comment\n[cosmology]\nomch2 = 0.16  ; trailing\nn_s=1.0\n")
    th2 = api.load_theta_file(i, base="test")
    assert th2[IDX["omch2"]] == 0.16 and th2[IDX["n_s"]] == 1.0
    lst = tmp_path / "l.json"
    lst.write_text(json.dumps(list(range(14))))
    assert np.array_equal(api.load_theta_file(lst), np.arange(14.0))
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"theta": {"h": 0.66}}))
    assert api.load_theta_file(m, base="test")[IDX["h"]] == 0.66
    with pytest.raises(KeyError):
        api.theta_from({"nonsense": 1.0})
    with pytest.raises(ValueError):
        i.write_text("garbage line\n")
        api.load_theta_file(i)


def test_derived_recorder_matches_direct_computation(post):
    th = api.random_prior_draws(1, seed=8, box=0.4)[0]
    th[IDX["A_baryon"]] = 2.5
    post.log_post(th)
    d = api.derived_for(post, th)
    lin = post.model.linear.compute(*th[:5])
    assert d["sigma8"] == pytest.approx(lin.sigma8_0, rel=1e-12)
    cols = api.derived_columns(post, np.vstack([th, th]))
    assert cols.shape == (2, 3) and np.allclose(cols[0], cols[1])
    assert cols[0, 2] == pytest.approx(d["S8"], rel=1e-12)


def test_blocks_cover_all_parameters():
    allnames = sum((list(v) for v in api.BLOCKS.values()), [])
    assert sorted(allnames) == sorted(PARAM_NAMES)


# =============================================================================================
# (f) CLI smoke tests (subprocess)
# =============================================================================================
def run_cli(*argv, timeout=300):
    r = subprocess.run([sys.executable, "-m", "kidspipe", *argv], capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=timeout)
    return r


@pytest.mark.parametrize("cmd", ["check", "loglike", "map", "mcmc", "nested", "bench"])
def test_cli_help(cmd):
    r = run_cli(cmd, "--help", timeout=60)
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_cli_top_level_help():
    r = run_cli("--help", timeout=60)
    assert r.returncode == 0
    for c in ("check", "loglike", "map", "mcmc", "nested", "bench"):
        assert c in r.stdout


def test_cli_check(tmp_path):
    r = run_cli("check", "--out", str(tmp_path / "chk"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout
    js = json.loads((tmp_path / "chk" / "check.json").read_text())
    assert js["ok"]


def test_cli_loglike_json_and_theta_file(tmp_path):
    r = run_cli("loglike", "--json-stdout", "--quiet")
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert abs(res["chi2"] - 196.06) < 0.5
    assert res["ln_post"] == pytest.approx(res["ln_like"] + res["ln_prior"], abs=1e-9)
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"A_IA": 1.0, "ln_1e10_A_s": 2.8, "omch2": 0.12, "ombh2": 0.0222, "n_s": 0.97, "A_baryon": 3.13}))
    r2 = run_cli("loglike", "--theta-file", str(f), "--preset", "test", "--set", "A_IA=0", "--json-stdout", "--quiet")
    res2 = json.loads(r2.stdout.strip().splitlines()[-1])
    assert abs(res2["chi2"] - 188.77) < 1.0
    assert res2["theta"]["A_IA"] == 0.0


def test_cli_loglike_out_of_bounds_exit_code():
    r = run_cli("loglike", "--set", "omch2=5", "--quiet")
    assert r.returncode == 2


def test_cli_bench(tmp_path):
    r = run_cli("bench", "--out", str(tmp_path / "b"), "--n-slow", "1", "--n-medium", "2", "--n-fast", "3")
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "b" / "bench.json").read_text())
    bm = js["block_mean_ms"]
    assert bm["fast"] < bm["medium"] < bm["slow"]


def test_cli_map_tiny(tmp_path):
    r = run_cli("map", "--out", str(tmp_path / "m"), "--starts", "1", "--maxiter", "2", "--rounds", "1", "--polish", "none",
                "--no-hessian", "--quiet")
    # (staged pre-optimisation is on by default; with maxiter=2 / polish none it costs a few dozen cheap evaluations)
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "m" / "map.json").read_text())
    assert np.isfinite(js["chi2"]) and set(js["theta"]) == set(PARAM_NAMES)


def test_cli_mcmc_tiny_deterministic(tmp_path):
    outs = []
    for k in range(2):
        d = tmp_path / f"c{k}"
        r = run_cli("mcmc", "--out", str(d), "--cycles", "4", "--burn", "4", "--rep-fast", "2", "--rep-medium", "1",
                    "--seed", "5", "--quiet")
        assert r.returncode == 0, r.stdout + r.stderr
        outs.append(np.load(d / "chain.npz"))
    a, b = outs
    assert tuple(a["names"]) == PARAM_NAMES + api.DERIVED_NAMES
    assert a["samples"].shape == (4, 17)
    assert np.array_equal(a["samples"], b["samples"]) and np.array_equal(a["log_post"], b["log_post"])
    S8 = a["samples"][:, -1]
    assert np.allclose(S8, a["samples"][:, -3] * np.sqrt(a["samples"][:, -2] / 0.3), rtol=1e-12)
    assert (d / "chain.txt").exists() and (d / "mcmc_summary.json").exists()


@pytest.mark.slow
def test_cli_mcmc_ensemble_tiny(tmp_path):
    r = run_cli("mcmc", "--method", "ensemble", "--walkers", "4", "--steps", "2", "--burn-frac", "0", "--out", str(tmp_path / "e"),
                "--quiet")
    assert r.returncode == 0, r.stdout + r.stderr
    z = np.load(tmp_path / "e" / "chain_ensemble.npz")
    assert z["samples"].shape == (8, 17)


@pytest.mark.slow
def test_cli_mcmc_parallel_chains_tiny(tmp_path):
    r = run_cli("mcmc", "--chains", "2", "--workers", "2", "--cycles", "3", "--burn", "3", "--rep-fast", "1", "--out",
                str(tmp_path / "p"), "--quiet", timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    z = np.load(tmp_path / "p" / "chain.npz")
    assert z["samples"].shape == (6, 17)
    assert (tmp_path / "p" / "chain_c1.npz").exists()


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["nautilus", "dynesty"])
def test_cli_nested_tiny(tmp_path, sampler):
    r = run_cli("nested", "--sampler", sampler, "--nlive", "40", "--max-calls", "250", "--n-eff", "10", "--out",
                str(tmp_path / sampler), "--quiet", timeout=1500)
    assert r.returncode == 0, r.stdout + r.stderr
    z = np.load(tmp_path / sampler / "chain.npz")
    assert z["samples"].shape[1] == 17 and np.isclose(z["weights"].sum(), 1.0)
