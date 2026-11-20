"""Tests for kidspipe.likelihood.data / gauss (spec s9-s11)."""
import numpy as np
import pytest

from kidspipe.likelihood import GaussianLikelihood, KV450Data
from kidspipe.likelihood import data as D
from kidspipe.params import M_BIAS


@pytest.fixture(scope="module")
def kv():
    return KV450Data.load()


@pytest.fixture(scope="module")
def lik(kv):
    return GaussianLikelihood(kv)


# ---------------------------------------------------------------- ordering / cuts
def test_pairs_and_columns():
    assert D.PAIRS[:6] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1)]
    assert len(D.PAIRS) == 15 and D.PAIRS[-1] == (4, 4)
    # data-file column formula of spec s10.2 is 1..15 in enumeration order
    assert [D.data_column(p) for p in range(15)] == list(range(1, 16))
    assert D.pair_index(0, 1) == D.pair_index(1, 0) == 1


def test_sizes_and_cuts(kv):
    assert D.N_DATA == 195 == kv.d.size
    assert kv.cov.shape == (195, 195)
    assert D.N_PLUS == 7 and D.N_MINUS == 6
    assert D.PLUS_MASK.sum() == 15 * 7
    idx = D.full_index()
    assert idx.size == 195 and np.all(np.diff(idx[:13]) > 0) and len(set(idx)) == 195


def test_assemble_layout_coded():
    """Encode (pair, sign, bin) in the values and check the 195 layout exactly."""
    p = np.arange(15)[:, None]
    b = np.arange(9)[None, :]
    xip = 1000 + 10 * p + b + 0.0 * b      # (15, 9)
    xim = 2000 + 10 * p + b + 0.0 * b
    v = D.assemble_vector(xip, xim)
    assert v.shape == (195,)
    for pp in (0, 6, 14):
        blk = v[13 * pp:13 * pp + 13]
        assert np.array_equal(blk[:7], 1000 + 10 * pp + np.arange(0, 7))
        assert np.array_equal(blk[7:], 2000 + 10 * pp + np.arange(3, 9))
    # same index set as the covariance cut
    full = np.concatenate([xip, xim], axis=1).ravel()           # not the cov layout
    lay = np.stack([xip, xim], axis=1).reshape(-1)              # [pair][xi+(9), xi-(9)]
    assert np.array_equal(lay[D.full_index()], v)
    # batch
    vb = D.assemble_vector(np.stack([xip, xip + 1]), np.stack([xim, xim + 1]))
    assert vb.shape == (2, 195) and np.array_equal(vb[1], v + 1)
    xs, ms = D.split_vector(vb)
    assert xs.shape == (2, 15, 7) and ms.shape == (2, 15, 6)
    del full


def test_data_matches_file(kv):
    raw = np.loadtxt(D.default_data_dir() / "kv450" / "xipm_data_vector.dat")
    # pair (j,i)=(4,4) is column 15, first xi+ row; pair (1,0) is column 2, last kept xi- (bin 8, row 17)
    assert kv.d[13 * 14 + 0] == raw[0, 15]
    assert kv.d[13 * 1 + 7 + 5] == raw[17, 2]
    assert kv.d[13 * 1 + 6] == raw[6, 2]               # last kept xi+ (bin 6)
    assert np.allclose(kv.theta_plus, raw[:7, 0]) and np.allclose(kv.theta_minus, raw[3:9, 0])
    assert np.isclose(kv.theta[0], 0.78613)


def test_covariance_cut(kv):
    assert np.allclose(kv.cov, kv.cov.T, rtol=0, atol=0)
    idx = D.full_index()
    assert np.array_equal(kv.cov, 0.5 * (kv.cov_full[np.ix_(idx, idx)] + kv.cov_full[np.ix_(idx, idx)].T))
    w = np.linalg.eigvalsh(kv.cov)
    assert w[0] > 0
    assert abs(w[-1] / w[0] / 2.4e5 - 1) < 0.05
    # data scatter versus errors: |d|/sigma at small scales is O(1..few)
    sig = np.sqrt(np.diag(kv.cov))
    assert np.all(sig > 0) and np.abs(kv.d / sig).max() < 40


def test_cterm(kv):
    assert np.allclose(kv.cterm_plus[:3], [5.38e-7, 4.98e-7, 3.98e-7], rtol=2e-3)
    assert np.allclose(kv.cterm_minus[:3], [1.04e-9, 2.43e-9, 7.22e-9], rtol=2e-3)
    c = kv.cterm
    assert c.shape == (195,)
    for p in (0, 7, 14):
        assert np.array_equal(c[13 * p:13 * p + 7], kv.cterm_plus[:7])
        assert np.array_equal(c[13 * p + 7:13 * p + 13], kv.cterm_minus[3:])


def test_m_factor(kv):
    f = kv.m_factor(M_BIAS)
    assert f.shape == (195,)
    # pair (4,2) -> index?  (1+m4)(1+m2)
    p = D.pair_index(4, 2)
    assert np.allclose(f[13 * p:13 * p + 13], (1 + M_BIAS[4]) * (1 + M_BIAS[2]))
    p = D.pair_index(0, 0)
    assert np.allclose(f[:13], (1 + M_BIAS[0]) ** 2)


# ---------------------------------------------------------------- Flinc validation vector
def test_flinc_theory_vector_and_layout(lik):
    v = D.load_flinc_theory()
    assert v.shape == (195,) and np.all(np.isfinite(v))
    assert np.allclose(v[:3], [2.6291e-5, 1.8423e-5, 1.1581e-5], rtol=1e-3)
    w = D.from_pm_blocks(v)
    assert np.array_equal(D.to_pm_blocks(w), v)
    # Flinc is a different cosmology, but its pair ordering must be consistent with the KV450
    # data/covariance ordering: chi2 against KV450 data is O(dof), while the unconverted
    # layout and a reversed pair order are catastrophically worse.
    c_ok = lik.chi2(w)
    xs, ms = D.split_vector(w)
    c_rev = lik.chi2(D.assemble_vector(_pad(xs[::-1], 7, 0), _pad(ms[::-1], 6, 3)))
    assert c_ok < 400
    assert lik.chi2(v) > 100 * c_ok
    assert c_rev > 2.5 * c_ok


def _pad(a, n, start):
    """put kept bins (15, n) back into an uncut (15, 9) array at columns start..start+n."""
    out = np.zeros((15, 9))
    out[:, start:start + n] = a
    return out


# ---------------------------------------------------------------- Gaussian likelihood
def test_null_chi2_and_condition(lik):
    assert abs(lik.condition_number() / 2.4e5 - 1) < 0.05
    # spec s15.2: null chi2 (data vs zero, with m-correction) = 269.3
    assert abs(lik.null_chi2() - 269.3) < 0.1
    # without the m correction it is different (267.2) -> the correction is really applied
    assert abs(GaussianLikelihood(lik.data, apply_m=False).null_chi2() - 267.18) < 0.1


def test_chi2_cholesky_vs_explicit_inverse(lik):
    rng = np.random.default_rng(5)
    for _ in range(5):
        mu = lik.d * (1 + 0.3 * rng.standard_normal(195)) + 1e-6 * rng.standard_normal(195)
        a = lik.chi2(mu, A_c=1.1, delta_c=2e-4)
        b = lik.chi2_explicit_inverse(mu, A_c=1.1, delta_c=2e-4)
        assert abs(a / b - 1) < 1e-10
    assert abs(lik.chi2(np.zeros(195)) / lik.chi2_explicit_inverse(np.zeros(195)) - 1) < 1e-10


def test_chi2_batch_matches_single(lik):
    rng = np.random.default_rng(6)
    n = 16
    mus = lik.d * (1 + 0.2 * rng.standard_normal((n, 195)))
    Ac = rng.uniform(0.7, 1.3, n)
    dc = rng.uniform(-5e-4, 5e-4, n)
    batch = lik.chi2_batch(mus, Ac, dc)
    single = np.array([lik.chi2(mus[k], Ac[k], dc[k]) for k in range(n)])
    assert np.allclose(batch, single, rtol=1e-12)
    assert np.allclose(lik.chi2_batch(mus[0]), lik.chi2(mus[0]))     # 1-D input
    mus[3, 5] = np.nan
    out = lik.chi2_batch(mus, Ac, dc)
    assert np.isinf(out[3]) and np.isfinite(out[[0, 1, 2, 4]]).all()
    assert np.allclose(np.delete(out, 3), np.delete(single, 3), rtol=1e-12)
    assert lik.log_like(mus[3]) == -np.inf


def test_systematics_terms(lik):
    mu = np.zeros(195)
    out = lik.add_systematics(mu, A_c=1.2, delta_c=3e-4)
    c = lik.data.cterm
    assert np.allclose(out[D.PLUS_MASK], 9e-8 + 1.44 * c[D.PLUS_MASK], rtol=1e-12, atol=0)
    assert np.allclose(out[~D.PLUS_MASK], 1.44 * c[~D.PLUS_MASK], rtol=1e-12, atol=0)
    # delta_c enters squared: sign-symmetric
    assert lik.chi2(mu, 1.0, 4e-4) == lik.chi2(mu, 1.0, -4e-4)
    # delta_c shift is xi+ only: minus-entries unchanged
    d0 = lik.add_systematics(mu, 1.0, 0.0)
    d1 = lik.add_systematics(mu, 1.0, 5e-4)
    assert np.allclose((d1 - d0)[~D.PLUS_MASK], 0) and np.allclose((d1 - d0)[D.PLUS_MASK], 2.5e-7)


def test_data_m_division(kv):
    lk = GaussianLikelihood(kv)
    assert np.allclose(lk.d * kv.m_factor(M_BIAS), kv.d, rtol=1e-14)
    # m-correction hand-check for pair (1,0) first xi+ entry
    assert np.isclose(lk.d[13], kv.d[13] / ((1 + M_BIAS[1]) * (1 + M_BIAS[0])))


def test_log_like_is_minus_half_chi2(lik):
    mu = lik.d * 0.97
    assert lik.log_like(mu) == -0.5 * lik.chi2(mu)
    assert np.allclose(lik.log_like_batch(np.stack([mu, mu * 1.02])),
                       [lik.log_like(mu), lik.log_like(mu * 1.02)])


def test_chi2_minimised_at_data_and_quadratic(lik):
    """Gaussian: with mu = d - c*e (systematics off) chi2 is exactly quadratic, = e^T C^-1 e."""
    lk = GaussianLikelihood(lik.data, use_cterm=False)
    e = np.random.default_rng(7).standard_normal(195) * 1e-6
    ref = e @ np.linalg.solve(lk.cov, e)
    assert abs(lk.chi2(lk.d - e) / ref - 1) < 1e-10
    assert abs(lk.chi2(lk.d - 2 * e) / (4 * ref) - 1) < 1e-10
    assert lk.chi2(lk.d) < 1e-20


# ---------------------------------------------------------------- review additions
def _loop_reference(root):
    """Independent brute-force construction of (d, cov, c-term, m-factor) with explicit python loops."""
    raw = np.loadtxt(root / "xipm_data_vector.dat")
    C = np.loadtxt(root / "cov_analytic.txt")
    c = np.loadtxt(root / "c_term_treecorr.out")[:, [3, 4]]
    m = M_BIAS
    vec, cv, mf, idx = [], [], [], []
    p = 0
    for i in range(5):
        for j in range(i, 5):
            for k in range(9):
                if k in (7, 8):
                    continue
                vec.append(raw[k, 1 + p]); cv.append(c[k, 0]); mf.append((1 + m[i]) * (1 + m[j])); idx.append(18 * p + k)
            for k in range(9):
                if k in (0, 1, 2):
                    continue
                vec.append(raw[9 + k, 1 + p]); cv.append(c[k, 1]); mf.append((1 + m[i]) * (1 + m[j])); idx.append(18 * p + 9 + k)
            p += 1
    idx = np.array(idx)
    return np.array(vec), C[np.ix_(idx, idx)], np.array(cv), np.array(mf)


def test_everything_vs_bruteforce_loops(kv, lik):
    vec, C, cv, mf = _loop_reference(D.default_data_dir() / "kv450")
    assert np.array_equal(kv.d, vec) and np.array_equal(kv.cov, C)
    assert np.array_equal(kv.cterm, cv) and np.array_equal(kv.m_factor(M_BIAS), mf)
    assert np.array_equal(lik.d, vec / mf)
    rng = np.random.default_rng(9)
    mu = vec * (1 + 0.1 * rng.standard_normal(195))
    A_c, dc = 1.17, 3.3e-4
    tot = mu + A_c ** 2 * cv + np.where(D.PLUS_MASK, dc ** 2, 0.0)
    r = vec / mf - tot
    ref = r @ np.linalg.inv(C) @ r
    assert abs(lik.chi2(mu, A_c, dc) / ref - 1) < 1e-9
    # null chi2 variants (spec s15.2 = 269.3 with m-correction, no c-term)
    Ci = np.linalg.inv(C)
    assert abs((vec / mf) @ Ci @ (vec / mf) - 269.30) < 0.01
    assert abs(lik.null_chi2() - 269.30) < 0.01
    assert abs(lik.chi2(np.zeros(195), A_c=0.0, delta_c=0.0) - 269.30) < 0.01


def test_covariance_pair_order_consistent_with_noise_structure(kv):
    """Shot-noise structure: Var(xi+_ab)^2 ~ Var(xi+_aa) Var(xi+_bb) / 4 at small theta.

    Independent of the data vector, this checks that the covariance blocks follow PAIRS order
    and the [xi+(9), xi-(9)] layout (a wrongly ordered covariance gives O(1..10) scatter)."""
    var = np.diag(kv.cov_full)
    v = np.array([var[18 * p + 1] for p in range(15)])      # xi+, theta bin 1
    for (j, i) in D.PAIRS:
        if j != i:
            ratio = v[D.pair_index(j, i)] ** 2 / (v[D.pair_index(j, j)] * v[D.pair_index(i, i)])
            assert 0.2 < ratio < 0.4, (j, i, ratio)
    # xi- has the same small-scale variance as xi+ (same shot noise) and sits in the second half
    for p in range(15):
        assert abs(var[18 * p + 9 + 1] / var[18 * p + 1] - 1) < 0.25


def test_anchor_pairs_chi2_against_data(lik):
    """Spec-implementation xi for pairs 1_1, 3_3, 5_5, 5_1 (anchors json, different cosmology than the data
    best fit but a sensible one) vs the matching KV450 data columns incl. cross-covariances:
    chi2 ~ N_dof (52); wrong data-column assignments are typically much worse."""
    import json
    a = json.load(open(D.default_data_dir() / "validation" / "anchors_test_cosmology.json"))["takahashi"]["xi"]
    keys = list(a)
    own = [D.pair_index(*[int(x) - 1 for x in k.split("_")]) for k in keys]
    mus = [np.concatenate([np.array(a[k]["xi_plus_binned"])[:7], np.array(a[k]["xi_minus_binned"])[3:]]) for k in keys]

    def chi(cols):
        idx = np.concatenate([np.arange(13 * q, 13 * q + 13) for q in cols])
        mu = np.concatenate([mus[n] + lik._cterm[13 * q:13 * q + 13] for n, q in enumerate(cols)])
        r = lik.d[idx] - mu
        return r @ np.linalg.solve(lik.cov[np.ix_(idx, idx)], r)

    c_ok = chi(own)
    assert 30 < c_ok < 90
    assert chi(own[::-1]) > 1.5 * c_ok                       # reversed assignment
    assert chi([own[0], own[1], own[3], own[2]]) > c_ok      # swapped 5_5 <-> 5_1


def test_nonfinite_nuisance_gives_inf_never_nan(lik):
    mu = lik.d * 0.9
    for bad in (np.nan, np.inf, -np.inf):
        assert lik.chi2(mu, bad, 0.0) == np.inf and lik.chi2(mu, 1.0, bad) == np.inf
        assert lik.log_like(mu, bad, 0.0) == -np.inf
        out = lik.chi2_batch(np.stack([mu, mu]), np.array([bad, 1.0]), 0.0)
        assert out[0] == np.inf and np.isfinite(out[1])
        ll = lik.log_like_batch(np.stack([mu, mu]), np.array([bad, 1.0]), 0.0)
        assert ll[0] == -np.inf and np.isfinite(ll[1])


def test_shape_validation_and_immutability(kv, lik):
    with pytest.raises(ValueError):
        lik.chi2(np.zeros(194))
    with pytest.raises(ValueError):
        lik.chi2_batch(np.zeros((3, 194)))
    with pytest.raises(ValueError):
        kv.m_factor(M_BIAS[:3])
    with pytest.raises(ValueError):
        D.assemble_vector(np.zeros((14, 9)), np.zeros((14, 9)))
    with pytest.raises(ValueError):
        lik.d[0] = 0.0                # internal arrays are read-only (no accidental mutation)


def test_no_state_dependence(lik):
    """chi2 is a pure function of its arguments (n_calls aside): repeated calls identical."""
    mu = lik.d * 0.95
    a = [lik.chi2(mu, 1.1, 1e-4) for _ in range(3)]
    assert a[0] == a[1] == a[2]
    assert lik.chi2_batch(np.stack([mu] * 3), 1.1, 1e-4).tolist() == pytest.approx(a, rel=1e-12)
