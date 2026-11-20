"""Tests for kidspipe.params (spec s1): bounds, priors, exact prior transform."""
import numpy as np
import pytest
from scipy import integrate, stats

from kidspipe import params as P


def test_names_bounds_start():
    assert P.PARAM_NAMES == ("omch2", "ombh2", "h", "n_s", "ln_1e10_A_s", "A_baryon", "A_IA",
                             "delta_c", "A_c", "dz1", "dz2", "dz3", "dz4", "dz5")
    assert P.BOUNDS.shape == (14, 2) and P.START.shape == (14,)
    assert np.all(P.BOUNDS[:, 0] < P.BOUNDS[:, 1])
    assert np.all(P.START >= P.BOUNDS[:, 0]) and np.all(P.START <= P.BOUNDS[:, 1])
    # spec s1.1 table
    assert np.allclose(P.BOUNDS[3], [0.7, 1.3]) and np.allclose(P.START[3], 1.09)
    assert np.allclose(P.BOUNDS[8], [0.62, 1.4]) and np.allclose(P.START[8], 1.13)
    assert np.allclose(P.M_BIAS, [-0.0128, -0.0104, -0.0114, 0.0072, 0.0061])
    assert np.allclose(P.SIGMA_DZ, [0.039, 0.023, 0.026, 0.012, 0.011])


def test_derived_helpers():
    assert np.isclose(P.eta0(3.13), 0.6044) and np.isclose(P.eta0(2.0), 0.74)
    assert np.isclose(P.A_s(3.0), np.exp(3.0) * 1e-10)


def test_to_from_dict():
    d = P.to_dict(P.START)
    assert list(d) == list(P.PARAM_NAMES) and d["h"] == 0.70
    assert np.array_equal(P.from_dict(d), P.START)
    assert P.from_dict({"h": 0.75})[2] == 0.75
    with pytest.raises(ValueError):
        P.to_dict(np.zeros(3))


def test_log_prior_bounds():
    assert np.isfinite(P.log_prior(P.START))
    for i in range(14):
        for side in (0, 1):
            th = P.START.copy()
            th[i] = P.BOUNDS[i, side]
            assert np.isfinite(P.log_prior(th))       # closed box
            th[i] = P.BOUNDS[i, side] + (-1 if side == 0 else 1) * 1e-9
            assert P.log_prior(th) == -np.inf
    th = P.START.copy()
    th[4] = np.nan
    assert P.log_prior(th) == -np.inf


def test_log_prior_flat_value():
    # at the Gaussian means the Gaussian exponent vanishes: compare against scipy truncnorm
    th = P.START.copy()
    th[P._G_IDX] = P._G_MU
    expect = sum(-np.log(P.BOUNDS[i, 1] - P.BOUNDS[i, 0]) for i in P._FLAT_IDX)
    for k, i in enumerate(P._G_IDX):
        expect += stats.truncnorm.logpdf(P._G_MU[k], P._G_A[k], P._G_B[k], loc=P._G_MU[k], scale=P._G_SIG[k])
    assert np.isclose(P.log_prior(th), expect, rtol=1e-12)


@pytest.mark.parametrize("name", ["delta_c", "A_c", "dz1", "dz3", "dz5", "h", "A_IA"])
def test_marginal_prior_normalised(name):
    """exp(ln pi) integrates to 1 over the bounds for each parameter separately."""
    i = P.PARAM_NAMES.index(name)
    base = P.START.copy()
    # remove the other parameters' normalisation: ln pi_i = lp(x_i) - lp(x_i = reference-other)
    other = P.log_prior(base) - _single(i, base[i])

    def f(x):
        th = base.copy()
        th[i] = x
        return np.exp(P.log_prior(th) - other)

    lo, hi = P.BOUNDS[i]
    pts = np.linspace(lo, hi, 9)[1:-1]
    val, _ = integrate.quad(f, lo, hi, points=pts, epsabs=1e-12, epsrel=1e-10, limit=200)
    assert np.isclose(val, 1.0, rtol=1e-7)


def _single(i, x):
    """normalised 1-D log prior of parameter i at x (independent formula)."""
    name = P.PARAM_NAMES[i]
    lo, hi = P.BOUNDS[i]
    if name in P.GAUSSIAN_PRIORS:
        mu, sig = P.GAUSSIAN_PRIORS[name]
        return stats.truncnorm.logpdf(x, (lo - mu) / sig, (hi - mu) / sig, loc=mu, scale=sig)
    return -np.log(hi - lo)


def test_log_prior_unnormalised_difference():
    a, b = P.START.copy(), P.START.copy()
    b[8] = 1.0
    d_norm = P.log_prior(a) - P.log_prior(b)
    d_unn = P.log_prior(a, normalized=False) - P.log_prior(b, normalized=False)
    assert np.isclose(d_norm, d_unn, rtol=1e-12)
    assert np.isclose(d_unn, -0.5 * (((1.13 - 1.01) / 0.13) ** 2 - ((1.0 - 1.01) / 0.13) ** 2))


def test_prior_transform_endpoints_and_median():
    lo = P.prior_transform(np.zeros(14))
    hi = P.prior_transform(np.ones(14))
    assert np.allclose(lo, P.BOUNDS[:, 0], rtol=0, atol=1e-12)
    assert np.allclose(hi, P.BOUNDS[:, 1], rtol=0, atol=1e-12)
    mid = P.prior_transform(np.full(14, 0.5))
    flat = P._FLAT_IDX
    assert np.allclose(mid[flat], P.BOUNDS[flat].mean(axis=1))
    # symmetric truncation (delta_c, dz_i): median = mean; A_c asymmetric (1.01 not mid-point 1.01 -> symmetric too)
    for name in ("delta_c", "A_c", "dz1", "dz2", "dz3", "dz4", "dz5"):
        i = P.PARAM_NAMES.index(name)
        assert np.isclose(mid[i], P.GAUSSIAN_PRIORS[name][0], atol=1e-12)


def test_prior_transform_matches_scipy_truncnorm():
    rng = np.random.default_rng(1)
    u = rng.uniform(size=(2000, 14))
    th = P.prior_transform(u)
    for k, i in enumerate(P._G_IDX):
        ref = stats.truncnorm.ppf(u[:, i], P._G_A[k], P._G_B[k], loc=P._G_MU[k], scale=P._G_SIG[k])
        assert np.allclose(th[:, i], ref, rtol=0, atol=1e-9 * P._G_SIG[k])
    for i in P._FLAT_IDX:
        assert np.allclose(th[:, i], P.BOUNDS[i, 0] + u[:, i] * np.diff(P.BOUNDS[i])[0], rtol=1e-14)
    assert np.all(np.isfinite(P.log_prior(th)))


def test_prior_transform_batch_equals_loop_and_roundtrip():
    rng = np.random.default_rng(2)
    u = rng.uniform(size=(5, 7, 14))
    th = P.prior_transform(u)
    assert th.shape == u.shape
    assert np.array_equal(th[3, 2], P.prior_transform(u[3, 2]))
    assert np.allclose(P.prior_cdf(th), u, atol=1e-9)
    assert P.prior_transform(u[0, 0]).shape == (14,)
    with pytest.raises(ValueError):
        P.prior_transform(np.zeros(13))


def test_prior_transform_extreme_u_inside_bounds():
    for uu in (1e-300, 1e-17, 1 - 1e-16, 0.0, 1.0):
        th = P.prior_transform(np.full(14, uu))
        assert np.all(th >= P.BOUNDS[:, 0]) and np.all(th <= P.BOUNDS[:, 1])
        assert np.isfinite(P.log_prior(th))


def test_prior_transform_sample_moments():
    """Monte Carlo moments of transformed uniform samples match the truncated normal's."""
    rng = np.random.default_rng(3)
    th = P.prior_transform(rng.uniform(size=(200000, 14)))
    for k, i in enumerate(P._G_IDX):
        m, v = stats.truncnorm.stats(P._G_A[k], P._G_B[k], loc=P._G_MU[k], scale=P._G_SIG[k], moments="mv")
        assert abs(th[:, i].mean() - m) < 5 * np.sqrt(v / 2e5)
        assert abs(th[:, i].std() / np.sqrt(v) - 1) < 0.01


# ---------------------------------------------------------------- robustness (review additions)
def test_prior_transform_out_of_range_and_nan():
    """u outside [0,1] is clipped identically for flat and Gaussian params; NaN raises (no silent NaN)."""
    lo = P.prior_transform(np.full(14, -0.3))
    hi = P.prior_transform(np.full(14, 1.7))
    assert np.allclose(lo, P.BOUNDS[:, 0], atol=1e-12) and np.allclose(hi, P.BOUNDS[:, 1], atol=1e-12)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    u = np.full(14, 0.5)
    u[9] = np.nan
    with pytest.raises(ValueError):
        P.prior_transform(u)


def test_log_prior_shape_and_nonfinite():
    for bad in (np.inf, -np.inf, np.nan):
        th = P.START.copy()
        th[7] = bad
        assert P.log_prior(th) == -np.inf
    with pytest.raises(ValueError):
        P.log_prior(P.START[:13])
    with pytest.raises(ValueError):
        P.log_prior(P.START.reshape(14, 1))
    with pytest.raises(ValueError):
        P.prior_cdf(P.START[:5])
    # batch: -inf only for the offending row
    th = np.stack([P.START, P.START])
    th[1, 2] = 0.9
    out = P.log_prior(th)
    assert np.isfinite(out[0]) and out[1] == -np.inf


def test_log_prior_independent_formula_at_random_points():
    """Independent re-derivation (scipy truncnorm.logpdf / uniform) at random in-bounds points."""
    rng = np.random.default_rng(11)
    th = P.prior_transform(rng.uniform(size=(50, 14)))
    got = P.log_prior(th)
    ref = np.zeros(50)
    for i, name in enumerate(P.PARAM_NAMES):
        lo, hi = P.BOUNDS[i]
        if name in P.GAUSSIAN_PRIORS:
            mu, s = P.GAUSSIAN_PRIORS[name]
            ref += stats.truncnorm.logpdf(th[:, i], (lo - mu) / s, (hi - mu) / s, loc=mu, scale=s)
        else:
            ref += stats.uniform.logpdf(th[:, i], lo, hi - lo)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-9)


def test_spec_bound_sigma_multiples():
    """Spec s1.1: dz bounds are +-2.56, 4.3, 3.8, 8.3, 9.1 sigma; delta_c and A_c are +-3 sigma."""
    k = (P.BOUNDS[9:14, 1] / P.SIGMA_DZ)
    assert np.allclose(k, [2.56, 4.35, 3.85, 8.33, 9.09], atol=0.06)
    assert np.isclose(P.BOUNDS[7, 1] / 2e-4, 3.0)
    assert np.isclose((P.BOUNDS[8, 1] - 1.01) / 0.13, 3.0) and np.isclose((1.01 - P.BOUNDS[8, 0]) / 0.13, 3.0)
