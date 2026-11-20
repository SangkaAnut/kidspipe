"""Synthetic tests of the adopted A1 (A1b-noise) and A3 gates (kidspipe.measure.a1check)."""
import numpy as np

from kidspipe.measure.a1check import A1B_THRESH, A3_MAX_ABS_OVER_SIGMA, a1b_noise, a3_pass

NP, NB = 21, 1000


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    th = np.exp(np.linspace(np.log(2), np.log(300), NB))
    sig = 1e-6 * (th / 2) ** -0.8
    base = {k: np.outer(1 + 0.1 * np.arange(NP), 2e-4 * th ** -0.8 * f) for k, f in (("xip", 1.0), ("xim", 0.3))}
    asc = {"xip": base["xip"] + sig * rng.standard_normal((NP, NB)),
           "xim": base["xim"] + sig * rng.standard_normal((NP, NB)),
           "sigma_xip": np.tile(sig, (NP, 1)), "sigma_xim": np.tile(sig, (NP, 1)),
           "npairs_weighted": np.tile(th ** 2, (NP, 1))}
    # "their" = same galaxies without the blinding factor: ascii = 0.85 * raw, plus binning jitter.
    # Two estimates from the SAME galaxies differ by anti-correlated bin-to-bin jitter (real data: lag-1
    # autocorrelation -0.43); model it as MA(1) with rho1 ~ -0.44.
    def jitter():
        e = rng.standard_normal((NP, NB + 1))
        return 0.3 * sig * (e[:, 1:] - 0.6 * e[:, :-1])
    their = {k: asc[k] / 0.85 + jitter() for k in ("xip", "xim")}
    sd = {k: np.full(NP, 0.01) for k in ("xip", "xim")}
    return th, asc, their, sd, rng


def test_pass_same_sample_global_constant():
    th, asc, their, sd, _ = _setup()
    r = a1b_noise(their["xip"], their["xim"], asc, sd)
    assert r["pass"]
    assert abs(r["xip"]["C"] - 1 / 0.85) < 0.01 and abs(r["xim"]["C"] - 1 / 0.85) < 0.01


def test_fail_pair_scale_scatter():
    th, asc, their, sd, rng = _setup()
    fac = np.exp(0.1 * rng.standard_normal(NP))[:, None]
    assert not a1b_noise(their["xip"] * fac, their["xim"] * fac, asc, sd)["pass"]


def test_fail_per_bin_m_like_pattern():
    th, asc, their, sd, _ = _setup()
    m = np.array([-0.023, -0.016, -0.011, 0.020, 0.030, 0.045])
    pairs = [(i, j) for i in range(6) for j in range(i, 6)]
    fac = np.array([1 / ((1 + m[i]) * (1 + m[j])) for i, j in pairs])[:, None]
    assert not a1b_noise(their["xip"] * fac, their["xim"] * fac, asc, sd)["pass"]


def test_fail_coherent_shape_excess():
    th, asc, their, sd, _ = _setup()
    bump = 1 + 0.5 * np.exp(-0.5 * ((np.log(th) - np.log(20)) / 0.3) ** 2)
    assert not a1b_noise(their["xip"] * bump, their["xim"] * bump, asc, sd)["pass"]


def test_white_residual_fails_adopted_threshold_documented():
    """Known limitation (see a1check docstring): a WHITE residual has boxcar/white ~ 1 (max over 42
    curves ~1.2), so the adopted threshold 1.03 rejects it; the white-noise 99th percentile is 1.33."""
    th, asc, their, sd, rng = _setup()
    sig = asc["sigma_xip"]
    tp = asc["xip"] / 0.85 + 0.3 * sig * rng.standard_normal((NP, NB))
    tm = asc["xim"] / 0.85 + 0.3 * sig * rng.standard_normal((NP, NB))
    assert not a1b_noise(tp, tm, asc, sd)["pass"]
    assert a1b_noise(tp, tm, asc, sd, {"boxcar_over_white_max": 1.33, "const_pte_min": 0.01})["pass"]


def test_a3_threshold():
    assert A3_MAX_ABS_OVER_SIGMA == 0.1
    assert a3_pass(0.053) and a3_pass(0.0999) and not a3_pass(0.1) and not a3_pass(0.5)


# ---------------------------------------------------------------------------------------------------
# Standardised (per-curve null) option of criterion (i)
# ---------------------------------------------------------------------------------------------------
import warnings

import pytest

from kidspipe.measure import a1check as A1


def _het_data(seed, asc, het=0.0, amp=0.3):
    """Same-sample 'their' on a FIXED ascii: MA(1) jitter whose z-amplitude grows as (theta/2)**het."""
    rng = np.random.default_rng(seed)
    th = np.exp(np.linspace(np.log(2), np.log(300), NB))
    out = []
    for key in ("xip", "xim"):
        e = rng.standard_normal((NP, NB + 1))
        out.append(asc[key] / 0.85 + amp * asc["sigma_" + key] * (th / 2) ** het * (e[:, 1:] - 0.6 * e[:, :-1]))
    return out


@pytest.fixture(scope="module")
def std_null():
    th, asc, their, sd, _ = _setup(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        null = A1.calibrate_standardised(their["xip"], their["xim"], asc, n_tab=1500, n_thr=600, size=0.05, seed=3)
    return th, asc, sd, null


def test_std_default_gate_unchanged():
    th, asc, their, sd, _ = _setup()
    r = a1b_noise(their["xip"], their["xim"], asc, sd)
    assert r["thresholds"]["boxcar_over_white_max"] == 1.03 and "standardised" not in r


def test_std_null_passes_at_roughly_the_right_rate(std_null):
    th, asc, sd, null = std_null
    fails = 0
    n = 60
    for s in range(n):
        tp, tm = _het_data(500 + s, asc)
        fails += not A1.a1b_noise_standardised(tp, tm, asc, sd, null=null)["pass"]
    assert fails / n < 0.2          # nominal 5 %; loose for a 60-draw test


def test_std_catches_single_curve_defect_that_constant_misses(std_null):
    th, asc, sd, null = std_null
    tp, tm = _het_data(900, asc)
    bump = np.exp(-0.5 * ((np.log(th) - np.log(20)) / 0.3) ** 2)
    tp = tp.copy(); tp[10] *= 1 + 0.0006 * bump
    const = a1b_noise(tp, tm, asc, sd)
    assert const["xip"]["max_boxcar_over_white"] <= A1B_THRESH["boxcar_over_white_max"]   # adopted constant does not see it
    r = A1.a1b_noise_standardised(tp, tm, asc, sd, null=null)
    assert not r["pass"] and r["standardised"]["argmin_p"] == 10


def test_std_heteroscedastic_null_and_prediction():
    th, asc, their, sd, _ = _setup(0)
    tp, tm = _het_data(11, asc, het=0.5)
    z, _ = A1.curves_z(tp, tm, asc)
    st_het = A1.standardised_stats(z, "hp15", het=True)
    st_hom = A1.standardised_stats(z, "hp15", het=False)
    assert np.abs(st_het["r_pred"] / st_hom["r_pred"] - 1).max() < 0.05      # heteroscedasticity is a small correction
    assert z[:, -100:].std() > 2 * z[:, :100].std()                          # data really are heteroscedastic
    # predicted ratio tracks the realised ratio on average (null)
    assert abs(np.median(st_het["ratio"] / st_het["r_pred"]) - 1) < 0.15


def test_std_highpass_lags_resist_smooth_shape_error():
    th, asc, their, sd, _ = _setup(0)
    tp, tm = _het_data(21, asc)
    tilt = 1 + 0.004 * np.log(th / 25) / np.log(12)
    z0, _ = A1.curves_z(tp, tm, asc)
    z1, _ = A1.curves_z(tp * tilt, tm * tilt, asc)
    d_plain = np.mean(A1.jitter_rho(z1, "plain")[0][:, 0] - A1.jitter_rho(z0, "plain")[0][:, 0])
    d_hp = np.mean(A1.jitter_rho(z1, "hp15")[0][:, 0] - A1.jitter_rho(z0, "hp15")[0][:, 0])
    assert d_plain > 0.05 and abs(d_hp) < 0.02
