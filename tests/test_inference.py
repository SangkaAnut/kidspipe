"""Tests of kidspipe.inference on analytic toy problems (no cosmology involved)."""
import json
import os
import sys
import time

import numpy as np
from kidspipe._compat import trapezoid
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kidspipe.inference import chains as ch
from kidspipe.inference import maxlike as ml
from kidspipe.inference import mcmc
from kidspipe.inference.posterior import (DEFAULT_PARAM_NAMES, FAST_PARAMS, MEDIUM_PARAMS, SLOW_PARAMS,
                                          ParamBlocks, Posterior, SimpleParams)

D = 14
NAMES = DEFAULT_PARAM_NAMES


# ----------------------------------------------------------------------------- toy problem
class Toy:
    """Correlated 14-d Gaussian posterior, flat prior on bounds (some of them tight)."""

    def __init__(self, seed=3, tight=(0, 5)):
        rng = np.random.default_rng(seed)
        self.mean = rng.uniform(-1, 1, D)
        sig = np.exp(rng.uniform(np.log(0.05), np.log(1.0), D))        # 20x dynamic range
        A = rng.normal(size=(D, D))
        M = A @ A.T / D + 0.3 * np.eye(D)                                 # SPD, moderately ill-conditioned
        R = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
        self.sig = sig
        self.cov = R * np.outer(sig, sig)
        self.prec = np.linalg.inv(self.cov)
        lo = self.mean - 7 * sig
        hi = self.mean + 7 * sig
        for i in tight:                                                  # truncate two parameters near 1.5 sigma
            lo[i] = self.mean[i] - 1.5 * sig[i]
        self.bounds = np.column_stack([lo, hi])
        self.params = SimpleParams(NAMES, self.bounds, start=self.mean + 0.5 * sig)
        self.n_slow_calls = 0

    def chi2(self, mu):
        r = mu - self.mean
        return r @ self.prec @ r

    def posterior(self, **kw):
        return Posterior(lambda th: th, lambda mu: -0.5 * self.chi2(mu), self.params, **kw)

    def lp_fast(self, th):
        """Plain callable (flat prior in bounds)."""
        if np.any(th < self.bounds[:, 0]) or np.any(th > self.bounds[:, 1]):
            return -np.inf
        r = th - self.mean
        return -0.5 * r @ self.prec @ r

    def lp_vec(self, X):
        ok = np.all((X >= self.bounds[:, 0]) & (X <= self.bounds[:, 1]), axis=1)
        r = X - self.mean
        return np.where(ok, -0.5 * np.einsum("ij,jk,ik->i", r, self.prec, r), -np.inf)

    def truncated_reference(self, n=1_500_000, seed=11):
        rng = np.random.default_rng(seed)
        x = rng.multivariate_normal(self.mean, self.cov, size=n)
        ok = np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1]), axis=1)
        x = x[ok]
        return x.mean(axis=0), np.cov(x.T)


@pytest.fixture(scope="module")
def toy():
    return Toy()


def check_moments(samples, ref_mean, ref_cov, ess, nsig=5.0):
    """Mean within nsig MC errors; marginal std and correlations within MC-error based tolerances.

    The tolerances scale with the effective sample size of each parameter (Gaussian estimates:
    relative error of a std ~ 1/sqrt(2 ESS), error of a correlation ~ (1 - r^2)/sqrt(ESS)), plus a small
    floor for the finite size of the rejection-sampled reference.
    """
    ess = np.asarray(ess, float)
    sd = np.sqrt(np.diag(ref_cov))
    dm = np.abs(samples.mean(axis=0) - ref_mean) / (sd / np.sqrt(ess))
    assert dm.max() < nsig, f"mean off by {dm.max():.1f} MC sigma"
    sd_s = samples.std(axis=0)
    dsd = np.abs(sd_s / sd - 1)
    tol_sd = nsig / np.sqrt(2 * ess) + 0.01
    assert np.all(dsd < tol_sd), (dsd / tol_sd).max()
    C = np.cov(samples.T)
    Rs = C / np.outer(sd_s, sd_s)
    Rr = ref_cov / np.outer(sd, sd)
    tol_r = nsig * (1 - Rr ** 2 + 0.05) / np.sqrt(np.sqrt(np.outer(ess, ess))) + 0.01
    assert np.all(np.abs(Rs - Rr) < tol_r), (np.abs(Rs - Rr) / tol_r).max()


# ----------------------------------------------------------------------------- posterior
def test_posterior_prior_nan_exception_counters():
    p = SimpleParams(["a", "b"], [[0, 1], [0, 2]])
    state = {"mode": "ok"}

    def model(th):
        if state["mode"] == "raise":
            raise RuntimeError("camb failed")
        return th

    def like(mu):
        return float("nan") if state["mode"] == "nan" else -0.5 * float(np.sum(mu ** 2))

    post = Posterior(model, like, p)
    assert post.log_post([0.5, 0.5]) == pytest.approx(-0.5 * 0.5 - np.log(2.0))   # flat prior 1/(1*2)
    n0 = post.stats.n_like
    assert post.log_post([1.5, 0.5]) == -np.inf and post.stats.n_like == n0      # model not called
    assert post.log_like([1.5, 0.5]) == -np.inf
    state["mode"] = "raise"
    assert post.log_post([0.3, 0.3]) == -np.inf
    assert isinstance(post.last_error, RuntimeError)
    state["mode"] = "nan"
    assert post.log_post([0.31, 0.3]) == -np.inf
    assert post.stats.n_fail == 2
    state["mode"] = "ok"
    # memo: identical point -> no new evaluation
    post.log_post([0.2, 0.2])
    n1 = post.stats.n_like
    post.log_post([0.2, 0.2])
    post.log_like([0.2, 0.2])
    assert post.stats.n_like == n1 and post.stats.n_memo_hits == 2
    post2 = Posterior(model, like, p, memo=False)
    post2.log_post([0.2, 0.2]); post2.log_post([0.2, 0.2])
    assert post2.stats.n_like == 2
    rep = post.timing_report()
    assert rep["n_like"] == post.stats.n_like and rep["t_model"] >= 0 and rep["t_prior"] >= 0


def test_blocks_and_change_classification():
    b = ParamBlocks.from_names(NAMES)
    assert [NAMES[i] for i in b.fast] == list(FAST_PARAMS)
    assert [NAMES[i] for i in b.medium] == list(MEDIUM_PARAMS)
    assert [NAMES[i] for i in b.slow] == list(SLOW_PARAMS)
    allidx = np.concatenate([b.slow, b.medium, b.fast])
    assert sorted(allidx) == list(range(14))
    # also the real parameter order of DESIGN.md
    assert NAMES[:4] == ("omch2", "ombh2", "h", "n_s") or set(NAMES) == set(SLOW_PARAMS + MEDIUM_PARAMS + FAST_PARAMS)
    a = np.zeros(14)
    c = a.copy(); c[b.fast[0]] = 1
    assert b.classify_change(a, c) == "fast"
    c[b.medium[0]] = 1
    assert b.classify_change(a, c) == "medium"
    c[b.slow[0]] = 1
    assert b.classify_change(a, c) == "slow"
    assert b.classify_change(a, a.copy()) == "same"


def test_inference_blocks_match_api_blocks():
    """n_s is an exact rescale of the cached transfer function: MEDIUM, not SLOW (review fix)."""
    api = pytest.importorskip("kidspipe.api")
    assert tuple(api.BLOCKS["slow"]) == SLOW_PARAMS == ("omch2", "ombh2", "h")
    assert tuple(api.BLOCKS["medium"]) == MEDIUM_PARAMS
    assert tuple(api.BLOCKS["fast"]) == FAST_PARAMS
    assert "n_s" in MEDIUM_PARAMS and "n_s" not in SLOW_PARAMS


def test_kidspipe_params_module_compatible():
    params = pytest.importorskip("kidspipe.params")
    post = Posterior(lambda th: th, lambda mu: 0.0, params)
    assert post.ndim == 14 and post.names == tuple(params.PARAM_NAMES)
    assert np.isfinite(post.log_post(params.START))
    th = post.prior_transform(np.full(14, 0.5))
    assert post.in_bounds(th)
    assert post.log_post(params.BOUNDS[:, 1] + 1) == -np.inf


def test_posterior_forwards_A_c_and_delta_c_to_likelihood():
    """GaussianLikelihood.log_like(mu, A_c, delta_c) takes the PURE theory vector: the Posterior must
    pass A_c and delta_c from theta, otherwise they would be silently fixed at 1 and 0."""
    p = SimpleParams(["x", "A_c", "delta_c"], [[-1, 1], [0, 2], [-1, 1]])
    seen = []

    def like(mu, A_c=1.0, delta_c=0.0):
        seen.append((A_c, delta_c))
        return -0.5 * (mu[0] - 0.3) ** 2 - A_c ** 2 - 10 * delta_c ** 2

    post = Posterior(lambda th: th, like, p)
    assert post.like_kwargs == {"A_c": "A_c", "delta_c": "delta_c"}
    val = post.log_like([0.1, 1.5, 0.2])
    assert seen[-1] == (1.5, 0.2)
    assert val == pytest.approx(-0.5 * 0.2 ** 2 - 1.5 ** 2 - 10 * 0.2 ** 2)
    # a plain one-argument likelihood is untouched; explicit override / opt-out work
    assert Posterior(lambda th: th, lambda mu: 0.0, p).like_kwargs == {}
    assert Posterior(lambda th: th, like, p, like_kwargs={}).like_kwargs == {}
    seen.clear()
    Posterior(lambda th: th, like, p, like_kwargs={"A_c": "x"}).log_like([0.7, 1.5, 0.2])
    assert seen[-1] == (0.7, 0.0)
    with pytest.raises(ValueError):
        Posterior(lambda th: th, like, p, like_kwargs={"A_c": "nope"})


def test_posterior_with_real_gaussian_likelihood_uses_systematics_parameters():
    params = pytest.importorskip("kidspipe.params")
    gauss = pytest.importorskip("kidspipe.likelihood.gauss")
    try:
        lik = gauss.GaussianLikelihood()
    except Exception as e:      # data files not available
        pytest.skip(f"KV450 data not available: {e}")
    mu0 = lik.d.copy()
    post = Posterior(lambda th: mu0, lik, params)
    th = np.array(params.START)
    assert post.log_like(th) == pytest.approx(lik.log_like(mu0, th[8], th[7]), rel=1e-12)
    th2 = th.copy(); th2[8] = 0.8; th2[7] = 5e-4
    assert post.log_like(th2) == pytest.approx(lik.log_like(mu0, 0.8, 5e-4), rel=1e-12)
    assert post.log_like(th2) != pytest.approx(lik.log_like(mu0), rel=1e-3)     # would be the bug
    assert post.log_post(th2) == pytest.approx(params.log_prior(th2) + lik.log_like(mu0, 0.8, 5e-4), rel=1e-12)


def test_posterior_rejects_wrong_theta_length():
    p = SimpleParams(["a", "b", "c"], [[0, 1]] * 3)
    post = Posterior(lambda th: th, lambda mu: 0.0, p)
    for bad in ([0.5], [0.5, 0.5], [0.5] * 4):
        with pytest.raises(ValueError):
            post.log_post(bad)
        with pytest.raises(ValueError):
            post.log_like(bad)


def test_simple_params_prior_transform_consistent():
    p = SimpleParams(["g", "f"], [[-1.0, 2.0], [0.0, 4.0]], gauss_mean=[0.0, np.nan], gauss_sigma=[0.5, np.nan])
    u = np.random.default_rng(0).uniform(size=(200000, 2))
    th = p.prior_transform(u)
    assert np.all(th[:, 0] >= -1) and np.all(th[:, 0] <= 2)
    a, b = (-1 - 0) / 0.5, (2 - 0) / 0.5
    assert stats.kstest(th[:, 0], stats.truncnorm(a, b, loc=0, scale=0.5).cdf).pvalue > 1e-3
    assert abs(th[:, 1].mean() - 2.0) < 0.02
    # normalisation: integral of exp(log_prior) over the box = 1
    xs = np.linspace(-1, 2, 30001)
    lpv = np.array([p.log_prior([x, 1.0]) for x in xs]) + np.log(4.0)    # flat part 1/4
    assert trapezoid(np.exp(lpv), xs) == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------------------- maxlike
def test_map_and_hessian_recover_truth(toy):
    post = toy.posterior()
    res = ml.find_map(post, toy.bounds, x0=toy.params.START, n_start=4, seed=1)
    assert res.success
    sig = np.sqrt(np.diag(toy.cov))
    assert np.max(np.abs(res.theta - toy.mean) / sig) < 1e-3
    assert res.n_agree >= 2
    assert res.neg_log_post == pytest.approx(-post.log_post(toy.mean), abs=1e-6)
    hr = ml.numerical_hessian(post, res.theta, toy.bounds)
    # tight-bound parameters are 1.5 sigma away: stencil stays inside, so result is exact for a quadratic
    assert hr.n_repaired == 0
    np.testing.assert_allclose(hr.cov, toy.cov, rtol=1e-3, atol=1e-5 * np.outer(sig, sig).max())
    np.testing.assert_allclose(hr.hess, toy.prec * 1.0, rtol=1e-3, atol=1e-3 * np.abs(toy.prec).max() * 1e-2)
    assert np.all(np.linalg.eigvalsh(hr.cov) > 0)


def test_map_random_starts_nonconvex():
    """Bimodal 2-d posterior with a deeper mode: multi-start must find the global one."""
    def lp(t):
        a = -0.5 * np.sum((t - np.array([-2.0, -2.0])) ** 2) / 0.3 ** 2 - 1.0
        b = -0.5 * np.sum((t - np.array([2.0, 2.0])) ** 2) / 0.3 ** 2
        return np.logaddexp(a, b)
    bounds = np.array([[-5, 5], [-5, 5.0]])
    res = ml.find_map(lp, bounds, n_start=12, seed=4)
    np.testing.assert_allclose(res.theta, [2.0, 2.0], atol=1e-3)
    assert res.n_agree >= 2


def test_map_at_boundary_and_failures():
    """Truth outside the box: MAP sits on the bound; NaN regions are treated as invalid."""
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    def lp(t):
        if t[1] > 0.9:
            return np.nan
        return -0.5 * ((t[0] + 0.5) / 0.4) ** 2 - 0.5 * ((t[1] - 0.3) / 0.1) ** 2
    res = ml.find_map(lp, bounds, x0=[0.5, 0.5], n_start=3, seed=0)
    assert res.theta[0] == pytest.approx(0.0, abs=1e-6)
    assert res.theta[1] == pytest.approx(0.3, abs=1e-4)
    hr = ml.numerical_hessian(lp, res.theta, bounds)
    assert np.all(hr.at_boundary == [True, False])
    assert np.all(np.isfinite(hr.cov)) and np.all(np.linalg.eigvalsh(hr.cov) > 0)
    assert hr.sigma[1] == pytest.approx(0.1, rel=2e-2)


def test_hessian_raises_when_stencil_hits_invalid_region():
    """x sits on an isolated finite point (every neighbour invalid): no covariance can be formed."""
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    lp = lambda t: -0.5 * np.sum(t ** 2) if np.all(np.abs(t - 0.5) < 1e-12) else -np.inf
    with pytest.raises(ValueError):
        ml.numerical_hessian(lp, [0.5, 0.5], bounds)


def test_hessian_repair_positive_definite():
    H = np.diag([4.0, 9.0, -1e-3, 1e-14]) * 1e2
    Hr, n = ml.repair_positive_definite(H, scale=np.ones(4))
    assert n == 2 and np.all(np.linalg.eigvalsh(Hr) > 0)
    np.testing.assert_allclose(Hr[:2, :2], H[:2, :2])
    cov = ml.hessian_to_cov(Hr)
    np.testing.assert_allclose(cov @ Hr, np.eye(4), atol=1e-9)
    # step control: very different curvature scales are handled
    def lp(t):
        return -0.5 * ((t[0] / 1e-4) ** 2 + (t[1] / 10.0) ** 2)
    hr = ml.numerical_hessian(lp, np.zeros(2), np.array([[-1.0, 1.0], [-50.0, 50.0]]))
    assert hr.sigma[0] == pytest.approx(1e-4, rel=1e-3) and hr.sigma[1] == pytest.approx(10.0, rel=1e-3)


# ----------------------------------------------------------------------------- MCMC
def test_adaptive_metropolis_truncated_gaussian(toy):
    ref_mean, ref_cov = toy.truncated_reference()
    post = toy.posterior()
    res = ml.find_map(post, toy.bounds, x0=toy.params.START, n_start=1)
    hr = ml.numerical_hessian(post, res.theta, toy.bounds)
    # deliberately mis-scaled seed covariance: adaptation must repair it
    out = mcmc.adaptive_metropolis(toy.lp_fast, toy.params.START, n_steps=120_000, n_burn=15_000,
                                   cov0=hr.cov * 0.05, seed=5)
    assert np.all(out.samples >= toy.bounds[:, 0]) and np.all(out.samples <= toy.bounds[:, 1])
    assert 0.1 < out.accept < 0.5
    ess = mcmc.effective_sample_size(out.samples)
    assert ess.min() > 300
    check_moments(out.samples, ref_mean, ref_cov, ess)
    assert mcmc.split_rhat(out.samples).max() < 1.05


def test_adaptive_metropolis_with_map_seed_and_posterior_object(toy):
    post = toy.posterior()
    res = ml.find_map(post, toy.bounds, x0=toy.params.START, n_start=1)
    hr = ml.numerical_hessian(post, res.theta, toy.bounds)
    out = mcmc.adaptive_metropolis(post, res.theta, n_steps=20_000, n_burn=3_000, cov0=hr.cov, seed=2)
    assert post.stats.n_like >= 23_000 - post.stats.n_memo_hits
    assert 0.1 < out.accept < 0.5
    # chain is a proper post-burn Markov chain: frozen proposal, so the acceptance of production ~ burn-end
    assert abs(out.accept - out.accept_blocks["all_burn"]) < 0.15


def test_blocked_metropolis_oversampling(toy):
    ref_mean, ref_cov = toy.truncated_reference()
    post = toy.posterior()
    b = post.blocks
    sched = [(np.concatenate([b.slow, b.medium]), 1), (b.fast, 6)]
    res = ml.find_map(post, toy.bounds, x0=toy.params.START, n_start=1)
    hr = ml.numerical_hessian(post, res.theta, toy.bounds)
    post.reset_stats()
    out = mcmc.blocked_metropolis(post, res.theta, sched, n_cycles=30_000, n_burn_cycles=3_000,
                                  cov0=hr.cov, seed=8)
    st = post.stats
    assert st.by_kind_n["fast"] > 2 * st.by_kind_n["slow"] > 0     # cheap moves dominate the evaluations
    assert out.n_eval == 1 + 33_000 * 7
    assert np.all(out.samples >= toy.bounds[:, 0]) and np.all(out.samples <= toy.bounds[:, 1])
    ess = mcmc.effective_sample_size(out.samples)
    check_moments(out.samples, ref_mean, ref_cov, ess)
    assert set(out.accept_blocks) >= {"block0", "block1", "all"}
    for k in ("block0", "block1"):
        assert 0.05 < out.accept_blocks[k] < 0.7


def test_blocked_metropolis_saves_expensive_calls():
    """Staged toy model: ESS of the cheap block per expensive recomputation is much higher."""
    d = 6
    rng = np.random.default_rng(1)
    A = rng.normal(size=(d, d))
    cov = A @ A.T + d * np.eye(d)
    prec = np.linalg.inv(cov)
    slow = np.array([0, 1])
    fast = np.array([2, 3, 4, 5])
    calls = {"slow": 0}
    cache = {}

    def lp(th):
        key = tuple(th[slow])
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            calls["slow"] += 1
            cache[key] = True
        return -0.5 * th @ prec @ th

    x0 = np.zeros(d)
    plain = mcmc.adaptive_metropolis(lp, x0, 20_000, 2_000, cov0=cov * 0.3, seed=1)
    n_plain = calls["slow"]
    calls["slow"] = 0
    cache.clear()
    blk = mcmc.blocked_metropolis(lp, x0, [(slow, 1), (fast, 8)], 20_000, 2_000, cov0=cov * 0.3, seed=1)
    n_blk = calls["slow"]
    ess_plain = mcmc.effective_sample_size(plain.samples)[fast].mean() / n_plain
    ess_blk = mcmc.effective_sample_size(blk.samples)[fast].mean() / n_blk
    assert ess_blk > 2.0 * ess_plain, (ess_blk, ess_plain)
    np.testing.assert_allclose(blk.samples.mean(axis=0), 0, atol=6 * np.sqrt(np.diag(cov).max() / 500))


def test_blocked_metropolis_requires_full_coverage():
    f = lambda x: -0.5 * float(x @ x)
    with pytest.raises(ValueError):
        mcmc.blocked_metropolis(f, np.zeros(3), [([0], 1), ([1], 1)], 10, 5)
    with pytest.raises(ValueError):
        mcmc.blocked_metropolis(f, np.zeros(3), [([0, 1, 5], 1)], 10, 5)
    out = mcmc.blocked_metropolis(f, np.zeros(3), [([0], 1), ([1], 1)], 200, 50, allow_partial=True)
    assert np.all(out.samples[:, 2] == 0.0)


def _banana_marginal_cdf():
    """y | x ~ N(x^2, 0.3^2), x ~ N(0,1): marginal CDF of y by brute-force quadrature over x and y."""
    xs = np.linspace(-9, 9, 36001)
    ys = np.linspace(-2.5, 14, 3301)
    py = np.array([trapezoid(stats.norm.pdf(xs) * stats.norm.pdf(v, xs ** 2, 0.3), xs) for v in ys])
    cdf = np.concatenate([[0], np.cumsum(0.5 * (py[1:] + py[:-1]) * np.diff(ys))])
    return lambda q: np.interp(q, ys, cdf)


def _banana_lp(t):
    return -0.5 * t[0] ** 2 - 0.5 * ((t[1] - t[0] ** 2) / 0.3) ** 2


def test_samplers_reproduce_non_gaussian_marginal_by_quadrature():
    """Curved (banana) 2-d target whose y-marginal is obtained by quadrature: each sampler's empirical
    CDF must agree within a few Monte-Carlo errors (a Gaussian target cannot expose a wrong
    acceptance ratio / Jacobian, a curved one does)."""
    cdf = _banana_marginal_cdf()
    qs = np.array([-0.2, 0.3, 0.8, 1.5, 3.0, 5.0])
    ref = cdf(qs)
    cov0 = np.diag([1.0, 1.0])

    def check(y, ess, name):
        emp = (y[:, None] <= qs).mean(axis=0)
        err = np.sqrt(ref * (1 - ref) / ess) * 1.5          # factor for the non-i.i.d. indicator ESS
        assert np.all(np.abs(emp - ref) < 5 * err + 2e-3), (name, emp, ref)

    r = mcmc.adaptive_metropolis(_banana_lp, [0.0, 0.0], 200_000, 10_000, cov0=cov0, seed=3)
    check(r.samples[:, 1], mcmc.effective_sample_size(r.samples)[1], "adaptive")
    r = mcmc.blocked_metropolis(_banana_lp, [0.0, 0.0], [([0], 1), ([1], 3)], 80_000, 5_000, cov0=cov0, seed=3)
    check(r.samples[:, 1], mcmc.effective_sample_size(r.samples)[1], "blocked")
    f = lambda X: -0.5 * X[:, 0] ** 2 - 0.5 * ((X[:, 1] - X[:, 0] ** 2) / 0.3) ** 2
    o = mcmc.ensemble_sampler(f, np.random.default_rng(1).normal(size=(64, 2)), 12_000, vectorized=True, seed=4)
    c = o.chain[1500:]
    check(c[:, :, 1].ravel(), mcmc.effective_sample_size(c)[1], "ensemble")


def test_samplers_reproduce_quartic_1d_cdf():
    """1-d asymmetric non-Gaussian target ln P = -x^4 + x with a quadrature CDF; many effectively independent
    samples (tau ~ 10), so even a mildly wrong acceptance rule (e.g. a tempered ratio) is excluded."""
    lp1 = lambda x: float(-x[0] ** 4 + x[0])
    xs = np.linspace(-4, 4, 80001)
    p = np.exp(-xs ** 4 + xs)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(xs))])
    cdf /= cdf[-1]
    qs = np.array([-0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
    ref = np.interp(qs, xs, cdf)

    def check(y, ess, name):
        emp = (y[:, None] <= qs).mean(axis=0)
        err = np.sqrt(ref * (1 - ref) / ess)
        assert np.all(np.abs(emp - ref) < 5 * err + 1e-3), (name, emp - ref, 5 * err)

    r = mcmc.adaptive_metropolis(lp1, [0.0], 300_000, 5_000, cov0=np.eye(1) * 0.3, seed=1)
    check(r.samples[:, 0], mcmc.effective_sample_size(r.samples)[0], "adaptive")
    r = mcmc.blocked_metropolis(lp1, [0.0], [([0], 2)], 150_000, 5_000, cov0=np.eye(1) * 0.3, seed=1)
    check(r.samples[:, 0], mcmc.effective_sample_size(r.samples)[0], "blocked")
    o = mcmc.ensemble_sampler(lambda X: -X[:, 0] ** 4 + X[:, 0], np.random.default_rng(0).normal(size=(32, 1)) * 0.5,
                              20_000, vectorized=True, seed=5)
    c = o.chain[500:]
    check(c[:, :, 0].ravel(), mcmc.effective_sample_size(c)[0], "ensemble")


def test_samplers_uniform_box_boundary_handling():
    """Flat target on [0,1]^d: hard boundaries must not bias the sample (uniform marginals, var 1/12)."""
    d = 4
    lp = lambda x: 0.0 if np.all((x >= 0) & (x <= 1)) else -np.inf
    outs = {
        "adaptive": mcmc.adaptive_metropolis(lp, np.full(d, 0.5), 60_000, 3_000, cov0=0.05 * np.eye(d), seed=1).samples,
        "blocked": mcmc.blocked_metropolis(lp, np.full(d, 0.5), [([0], 1), ([1, 2, 3], 2)], 30_000, 3_000,
                                           cov0=0.05 * np.eye(d), seed=1).samples,
    }
    f = lambda X: np.where(np.all((X >= 0) & (X <= 1), axis=1), 0.0, -np.inf)
    o = mcmc.ensemble_sampler(f, np.random.default_rng(0).uniform(size=(32, d)), 6_000, vectorized=True, seed=2)
    outs["ensemble"] = o.chain[300:].reshape(-1, d)
    for name, s in outs.items():
        for j in range(d):
            xj = s[:, j]
            assert np.all((xj >= 0) & (xj <= 1))
            tau = mcmc.integrated_autocorr_time(xj)[0] if name != "ensemble" else 40
            ess = len(xj) / tau
            assert abs(xj.mean() - 0.5) < 5 * np.sqrt(1 / 12 / ess), (name, j, xj.mean())
            assert abs(xj.var() * 12 - 1) < 6 * np.sqrt(0.8 / ess) + 0.02, (name, j, xj.var() * 12)


@pytest.fixture(scope="module")
def ensemble_run(toy):
    """One 64-walker, 12000-step ensemble run on the truncated toy Gaussian: (p0, chain output, wall time in s)."""
    nw, ns = 64, 12000
    rng = np.random.default_rng(0)
    p0 = toy.mean + 0.05 * rng.multivariate_normal(np.zeros(D), toy.cov, size=nw)
    p0 = np.clip(p0, toy.bounds[:, 0] + 1e-9, toy.bounds[:, 1] - 1e-9)
    t0 = time.time()
    out = mcmc.ensemble_sampler(toy.lp_vec, p0, ns, vectorized=True, seed=1)
    return p0, out, time.time() - t0


@pytest.mark.timing
def test_ensemble_sampler_wall_time(ensemble_run):
    assert ensemble_run[2] < 20


def test_ensemble_sampler_truncated_gaussian(toy, ensemble_run):
    ref_mean, ref_cov = toy.truncated_reference()
    p0, out, _ = ensemble_run
    assert 0.1 < out.acceptance_fraction.mean() < 0.9
    c = out.chain[2000:]                                    # discard burn-in
    assert np.all(c >= toy.bounds[:, 0]) and np.all(c <= toy.bounds[:, 1])
    assert mcmc.split_rhat(c).max() < 1.1
    ess = mcmc.effective_sample_size(c)
    assert ess.min() > 300, ess.min()
    flat = c.reshape(-1, D)
    check_moments(flat, ref_mean, ref_cov, ess)
    # non-vectorised path gives identical stretch-move statistics for the same seed
    o2 = mcmc.ensemble_sampler(toy.lp_fast, p0, 5, vectorized=False, seed=1)
    o3 = mcmc.ensemble_sampler(toy.lp_vec, p0, 5, vectorized=True, seed=1)
    np.testing.assert_allclose(o2.chain, o3.chain)


def test_ensemble_affine_invariance_banana_free():
    """Highly anisotropic Gaussian (cond. number 1e6): ensemble mixes where plain RW would not."""
    d = 4
    cov = np.diag([1.0, 1e-2, 1e-4, 1e-6])
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(d, d)))[0]
    cov = R @ cov @ R.T
    prec = np.linalg.inv(cov)
    f = lambda X: -0.5 * np.einsum("ij,jk,ik->i", X, prec, X)
    p0 = np.random.default_rng(1).normal(size=(32, d)) * 1e-3
    out = mcmc.ensemble_sampler(f, p0, 6000, vectorized=True, seed=3)
    flat = out.chain[1500:].reshape(-1, d)
    ev = np.linalg.eigvalsh(np.cov(flat.T))
    np.testing.assert_allclose(ev, np.linalg.eigvalsh(cov), rtol=0.15)


# ----------------------------------------------------------------------------- diagnostics
def test_rhat_and_autocorr_sanity():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4000, 4, 3))
    r = mcmc.split_rhat(iid)
    assert np.all(np.abs(r - 1) < 0.01)
    assert np.all(np.abs(mcmc.split_rhat(iid, rank_normalize=True) - 1) < 0.01)
    shifted = iid.copy()
    shifted[:, 0, :] += 3.0
    assert np.all(mcmc.split_rhat(shifted) > 1.5)
    trend = np.cumsum(rng.normal(size=(4000,)) * 0.0 + 0.01) + rng.normal(size=4000) * 0.1   # drift
    assert mcmc.split_rhat(trend)[0] > 1.5
    # AR(1): tau = (1+phi)/(1-phi)
    phi, n = 0.9, 200_000
    e = rng.normal(size=(n, 2))
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    tau, rel = mcmc.integrated_autocorr_time(x, return_flag=True)
    assert np.all(rel)
    np.testing.assert_allclose(tau, (1 + phi) / (1 - phi), rtol=0.15)
    ess = mcmc.effective_sample_size(x)
    np.testing.assert_allclose(ess, n / tau)
    assert mcmc.integrated_autocorr_time(iid)[0] == pytest.approx(1.0, abs=0.15)
    rho = mcmc.autocorr_function(x)
    assert rho[0, 0] == pytest.approx(1.0) and rho[1, 0] == pytest.approx(phi, abs=0.02)


# ----------------------------------------------------------------------------- nested sampling
def _gauss_problem(d=3, L=8.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    cov = 0.25 * (A @ A.T / d + np.eye(d))
    prec = np.linalg.inv(cov)
    logn = -0.5 * np.linalg.slogdet(2 * np.pi * cov)[1]
    mu = rng.uniform(-1, 1, d)
    ll = lambda x: logn - 0.5 * (x - mu) @ prec @ (x - mu)
    pt = lambda u: -L + 2 * L * np.asarray(u)
    return ll, pt, mu, cov, -d * np.log(2 * L)       # analytic ln Z (prior box >> posterior)


@pytest.mark.slow
def test_nautilus_gaussian_evidence():
    pytest.importorskip("nautilus")
    from kidspipe.inference.nested import run_nautilus
    ll, pt, mu, cov, lnz = _gauss_problem()
    res = run_nautilus(ll, pt, 3, n_live=500, n_eff=3000, seed=1)
    assert abs(res.log_z - lnz) < 0.1, (res.log_z, lnz)
    assert res.weights.sum() == pytest.approx(1.0)
    m = np.sum(res.weights[:, None] * res.samples, axis=0)
    sd = np.sqrt(np.diag(cov))
    assert np.all(np.abs(m - mu) < 0.1 * sd + 3 * sd / np.sqrt(res.ess))
    C = ch.weighted_cov(res.samples, res.weights)
    np.testing.assert_allclose(np.sqrt(np.diag(C)), sd, rtol=0.1)
    assert res.sampler == "nautilus" and res.n_eval > 0


def test_dynesty_gaussian_evidence_static_and_dynamic():
    pytest.importorskip("dynesty")
    from kidspipe.inference.nested import run_dynesty
    ll, pt, mu, cov, lnz = _gauss_problem()
    res = run_dynesty(ll, pt, 3, nlive=300, dlogz=0.05, seed=2)
    assert abs(res.log_z - lnz) < max(4 * res.log_z_err, 0.1), (res.log_z, res.log_z_err, lnz)
    assert res.weights.sum() == pytest.approx(1.0) and res.log_z_err > 0
    sd = np.sqrt(np.diag(cov))
    m = np.sum(res.weights[:, None] * res.samples, axis=0)
    assert np.all(np.abs(m - mu) < 0.15 * sd + 3 * sd / np.sqrt(res.ess))
    eq = res.equal_weight(2000)
    assert eq.shape == (2000, 3)
    dyn = run_dynesty(ll, pt, 3, nlive=150, dynamic=True, dlogz=0.1, nlive_batch=100, maxbatch=3, seed=3)
    assert abs(dyn.log_z - lnz) < max(4 * dyn.log_z_err, 0.15), (dyn.log_z, lnz)


def test_nested_result_log_post_adds_prior():
    from kidspipe.inference.nested import NestedResult
    p = SimpleParams(["a"], [[-3.0, 3.0]], gauss_mean=[0.0], gauss_sigma=[1.0])
    x = np.linspace(-2, 2, 7)[:, None]
    r = NestedResult(samples=x, weights=np.full(7, 1 / 7), log_like=-0.5 * x[:, 0] ** 2, log_z=0.0, log_z_err=0.0,
                     n_eval=1, time=0.0, sampler="x")
    np.testing.assert_allclose(r.log_post(p.log_prior), -0.5 * x[:, 0] ** 2 + [p.log_prior(t) for t in x])


def test_nested_wrappers_handle_failures_and_posterior_object():
    pytest.importorskip("dynesty")
    from kidspipe.inference.nested import make_nested_functions, run_dynesty, safe_loglike
    f = safe_loglike(lambda t: float("nan"))
    assert f(np.zeros(2)) < -1e299
    g = safe_loglike(lambda t: 1 / 0)
    assert g(np.zeros(2)) < -1e299
    # Posterior as the source of loglike / prior_transform: Gaussian likelihood x truncated-Gaussian prior
    p = SimpleParams(["a", "b"], [[-3.0, 3.0], [-3.0, 3.0]], gauss_mean=[0.0, 0.0], gauss_sigma=[1.0, 1.0])
    post = Posterior(lambda th: th, lambda mu: -0.5 * np.sum((mu - 0.5) ** 2 / 0.5 ** 2) - np.log(2 * np.pi * 0.25), p)
    ll, pt, nd = make_nested_functions(post)
    res = run_dynesty(ll, pt, nd, nlive=300, dlogz=0.05, seed=4)
    # analytic: product of 1-d convolutions N(0.5; 0, sqrt(1+0.25)), truncation at 3 sigma of prior
    # per axis: Z1 = int N(x;0.5,0.5) N(x;0,1)/T dx = N(0.5;0,sqrt(1.25)) * P(posterior in box)/T,
    # posterior N(0.4, sqrt(0.2)); the likelihood is normalised in 2-d (log(2 pi 0.25)) so it factorises
    T = stats.norm.cdf(3) - stats.norm.cdf(-3)
    z1 = stats.norm.pdf(0.5, scale=np.sqrt(1.25)) / T * (
        stats.norm.cdf((3 - 0.4) / np.sqrt(0.2)) - stats.norm.cdf((-3 - 0.4) / np.sqrt(0.2)))
    assert abs(res.log_z - 2 * np.log(z1)) < max(4 * res.log_z_err, 0.1)


# ----------------------------------------------------------------------------- chains utilities
def _brute_pjhpd(x, w, lp, q):
    w = w / w.sum()
    best = None
    for t in np.sort(np.unique(lp)):
        sel = lp >= t
        if w[sel].sum() >= q - 1e-12:
            best = sel                      # keeps the highest threshold that still holds mass q
    return np.column_stack([x[best].min(axis=0), x[best].max(axis=0)])


def test_pjhpd_matches_brute_force_small_n():
    rng = np.random.default_rng(0)
    for trial in range(5):
        n, d = 150, 3
        x = rng.normal(size=(n, d))
        lp = -0.5 * np.sum(x ** 2, axis=1) + 1e-9 * rng.normal(size=n)
        w = rng.uniform(0.1, 3.0, n) if trial % 2 else None
        wn = np.ones(n) if w is None else w
        ext = ch.pjhpd_extents(x, lp, w, (0.3, 0.68, 0.95))
        for a, q in enumerate((0.3, 0.68, 0.95)):
            np.testing.assert_allclose(ext[a], _brute_pjhpd(x, wn, lp, q))
        m = ch.pjhpd_mask(lp, w, 0.68)
        assert wn[m].sum() / wn.sum() >= 0.68 - 1e-12
        # minimality: removing the lowest-density member drops below the level
        idx = np.nonzero(m)[0][np.argmin(lp[m])]
        m2 = m.copy(); m2[idx] = False
        assert wn[m2].sum() / wn.sum() < 0.68


def test_pjhpd_gaussian_analytic():
    """Joint HPD of a d-dim Gaussian is the ellipsoid r^2 <= chi2_d.ppf(q): projection = +-sqrt(chi2)*sigma."""
    rng = np.random.default_rng(1)
    sg = np.array([0.5, 2.0])
    x = rng.normal(size=(1_000_000, 2)) * sg
    lp = -0.5 * np.sum((x / sg) ** 2, axis=1)
    for q in (0.68, 0.95):
        ext = ch.pjhpd_extents(x, lp, None, (q,))[0]
        np.testing.assert_allclose(0.5 * (ext[:, 1] - ext[:, 0]), np.sqrt(stats.chi2.ppf(q, 2)) * sg, rtol=0.01)


def _check_pjhpd_14d(rng):
    """14-d Gaussian: the sample extent can only undershoot the ellipsoid.  Returns the wall time of ``pjhpd_extents``."""
    d, n = 14, 400_000
    sg = np.linspace(0.5, 2.0, d)
    x = rng.normal(size=(n, d)) * sg
    lp = -0.5 * np.sum((x / sg) ** 2, axis=1)
    t0 = time.perf_counter()
    ext = ch.pjhpd_extents(x, lp, None, (0.68, 0.95))
    dt = time.perf_counter() - t0
    for a, q in enumerate((0.68, 0.95)):
        half = 0.5 * (ext[a, :, 1] - ext[a, :, 0])
        expected = np.sqrt(stats.chi2.ppf(q, d)) * sg
        assert np.all(half <= expected * 1.001) and np.all(half > 0.7 * expected)
    # PJ-HPD is much wider than the marginal 68 % interval (that is the point of it)
    marg = ch.marginal_stats(x[:, :2], None, ["a", "b"], 0.68, lp)
    assert marg["a"]["pj_hi"] - marg["a"]["pj_lo"] > 2 * (marg["a"]["hi"] - marg["a"]["lo"])
    return dt


def test_pjhpd_14d_extents():
    _check_pjhpd_14d(np.random.default_rng(3))


@pytest.mark.timing
def test_pjhpd_14d_speed():
    """O(N log N): 400k samples in 14-d in under 5 s."""
    assert _check_pjhpd_14d(np.random.default_rng(3)) < 5.0


def test_hpd_1d_and_quantiles_vs_brute_force():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 1, 120), rng.normal(5, 0.5, 60)])
    w = rng.uniform(0.2, 2, len(x))
    lo, hi = ch.hpd_interval_1d(x, w, 0.68)
    # brute force over all (i, j)
    o = np.argsort(x); xs, ws = x[o], w[o] / w.sum()
    best = (np.inf, None)
    for i in range(len(xs)):
        s = 0.0
        for j in range(i, len(xs)):
            s += ws[j]
            if s >= 0.68 - 1e-12:
                if xs[j] - xs[i] < best[0]:
                    best = (xs[j] - xs[i], (xs[i], xs[j]))
                break
    assert (lo, hi) == pytest.approx(best[1])
    # equal-weight quantiles agree with numpy (to sample spacing)
    big = rng.normal(size=100001)
    q = ch.weighted_quantile(big, None, [0.16, 0.5, 0.84])
    np.testing.assert_allclose(q, np.quantile(big, [0.16, 0.5, 0.84]), atol=2e-3)
    # weights act as repeat counts
    xs2 = np.array([1.0, 2.0, 3.0]); w2 = np.array([1, 0, 3.0])
    np.testing.assert_allclose(ch.weighted_mean(xs2[:, None], w2), [2.5])
    np.testing.assert_allclose(ch.weighted_cov(np.array([[0.0], [1.0], [2.0], [2.0]]))[0, 0],
                               np.var([0, 1, 2, 2], ddof=1) * 1.0, rtol=1e-12)


def test_marginal_stats_and_s8():
    rng = np.random.default_rng(3)
    s8 = rng.normal(0.74, 0.04, 200000)
    st = ch.marginal_stats(s8[:, None], None, ["S8"], 0.68)["S8"]
    assert st["mean"] == pytest.approx(0.74, abs=3e-4) and st["std"] == pytest.approx(0.04, rel=0.01)
    assert st["lo"] == pytest.approx(0.74 - 0.04, abs=1e-3) and st["hi"] == pytest.approx(0.74 + 0.04, abs=1e-3)
    assert st["hpd_hi"] - st["hpd_lo"] == pytest.approx(0.08, rel=0.02)
    assert st["mode"] == pytest.approx(0.74, abs=0.004)
    assert ch.s8_column(0.8, 0.3) == pytest.approx(0.8)
    assert ch.s8_column(0.8, 0.12) == pytest.approx(0.8 * np.sqrt(0.4))
    x = np.array([[0.8, 0.3], [0.7, 0.12]])
    y, nm = ch.append_s8(x, ["sigma8", "omega_m"])
    assert nm[-1] == "S8" and y.shape == (2, 3)
    np.testing.assert_allclose(y[:, 2], [0.8, 0.7 * np.sqrt(0.4)])


def test_burn_in_removal():
    lp = np.concatenate([np.linspace(-500, -10, 300), -10 + 0.3 * np.random.default_rng(0).normal(size=3000)])
    k = ch.estimate_burn_in(lp)
    assert 200 < k < 340
    x = np.arange(len(lp) * 2.0).reshape(-1, 2)
    assert ch.remove_burn_in(x, n_burn=100).shape[0] == len(lp) - 100
    assert ch.remove_burn_in(x, frac=0.5).shape[0] == len(lp) - len(lp) // 2
    assert ch.remove_burn_in(x, log_post=lp).shape[0] == len(lp) - k
    cw = np.zeros((len(lp), 4, 2))
    assert ch.remove_burn_in(cw, log_post=np.repeat(lp[:, None], 4, 1)).shape[0] == len(lp) - k


def test_chain_io_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    names = ("a", "b", "c")
    c = ch.Chain(rng.uniform(0.1, 1.0, size=(50, 3)), names, rng.uniform(1, 2, 50), rng.normal(size=50), rng.normal(size=50),
                 meta={"sampler": "test", "log_z": -3.25, "arr": np.arange(3)})
    npz, txt = ch.save_chain(tmp_path / "sub" / "chain1", c)
    assert os.path.exists(npz) and os.path.exists(txt)
    head = open(txt).read().splitlines()[:6]
    assert head[0].startswith("# kidspipe chain") and any("columns: weight log_post log_like a b c" in h for h in head)
    for path in (tmp_path / "sub" / "chain1", npz, txt):
        c2 = ch.load_chain(path)
        assert c2.names == names
        np.testing.assert_allclose(c2.samples, c.samples, rtol=1e-9)
        np.testing.assert_allclose(c2.weights, c.weights, rtol=1e-9)
        np.testing.assert_allclose(c2.log_post, c.log_post, rtol=1e-9)
        np.testing.assert_allclose(c2.log_like, c.log_like, rtol=1e-9)
        assert c2.meta["sampler"] == "test" and c2.meta["log_z"] == -3.25
    # no log_post / log_like
    c3 = ch.Chain(c.samples, names)
    ch.save_chain(tmp_path / "plain", c3)
    c4 = ch.load_chain(tmp_path / "plain.txt")
    assert c4.log_post is None and c4.log_like is None and c4.samples.shape == (50, 3)
    assert c.with_s8("a", "b").names[-1] == "S8"
    with pytest.raises(ValueError):
        ch.save_chain(tmp_path / "bad", ch.Chain(c.samples, ("a b", "b", "c")))


def test_summary_tables():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    x = rng.multivariate_normal([1.0, 2.0], cov, size=20000)
    lp = -0.5 * np.einsum("ij,jk,ik->i", x - [1, 2], np.linalg.inv(cov), x - [1, 2])
    c = ch.Chain(x, ("sigma8", "omega_m"), log_post=lp)
    c2 = ch.Chain(np.abs(x) + 0.1, ("sigma8", "omega_m"), log_post=lp).with_s8()
    assert c2.names == ("sigma8", "omega_m", "S8")
    t = ch.summary_table(c)
    assert "sigma8" in t and "omega_m" in t and "pj_lo" in t and "MAP" in t
    t2 = ch.correlation_table(x, None, ["sigma8", "omega_m"])
    assert "0.80" in t2 and "1.00" in t2
    st = c.stats()
    assert st["sigma8"]["mean"] == pytest.approx(1.0, abs=0.03)
    np.testing.assert_allclose(c.map_point(), [1, 2], atol=0.3)
