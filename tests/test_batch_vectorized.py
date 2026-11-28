"""Tests for Prompt 10: vectorised batch evaluation (mu_batch / log_like_batch) and nested wiring."""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numba
import numpy as np
import pytest
from scipy.stats import qmc

from kidspipe import api, params
from kidspipe.inference.nested import make_nested_functions, run_dynesty, run_nautilus, safe_loglike_batch
from kidspipe.inference.posterior import Posterior, SimpleParams
from kidspipe.likelihood.gauss import GaussianLikelihood
from kidspipe.theory import hmcode, nonlinear
from kidspipe.theory.emulator import EmulatedLinearBackend, TransferEmulator
from kidspipe.theory.model import CosmicShearModel

BASELINE_NPZ = Path("/home/anut/kidspipe_sync/runs/agent_reports/baseline_mu_64.npz")


def _in_box_sobol_thetas(n: int = 64, seed: int = 11) -> np.ndarray:
    """Deterministic in-box, in-bounds 14-D Sobol vectors (matches baseline_mu_64 if available)."""
    if BASELINE_NPZ.exists() and n == 64 and seed == 11:
        return np.load(BASELINE_NPZ)["thetas"]
    sob = qmc.Sobol(d=14, scramble=True, seed=seed).random(n)
    th = params.prior_transform(sob)
    th[:, 0] = 0.04 + (0.30 - 0.04) * sob[:, 0]
    return th


@pytest.fixture(scope="module")
def thetas_64() -> np.ndarray:
    return _in_box_sobol_thetas(64, seed=11)


@pytest.fixture(scope="module")
def emu() -> TransferEmulator:
    return TransferEmulator()


# --------------------------------------------------------------------------------------
# 1. prior_transform
# --------------------------------------------------------------------------------------
def test_prior_transform_vectorized_shapes_and_exactness():
    rng = np.random.default_rng(123)
    for shape in ((14,), (1, 14), (7, 14), (64, 14), (3, 5, 14)):
        u = rng.uniform(size=shape)
        got = params.prior_transform(u)
        assert got.shape == shape
        ref = np.array([params.prior_transform(r) for r in u.reshape(-1, 14)]).reshape(shape)
        np.testing.assert_allclose(got, ref, rtol=1e-14, atol=0.0)
        assert np.all(params.in_bounds(got))
    empty = params.prior_transform(np.empty((0, 14)))
    assert empty.shape == (0, 14)
    lo = params.prior_transform(np.zeros((2, 14)))
    hi = params.prior_transform(np.ones((2, 14)))
    np.testing.assert_allclose(lo, np.broadcast_to(params.BOUNDS[:, 0], (2, 14)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(hi, np.broadcast_to(params.BOUNDS[:, 1], (2, 14)), rtol=0, atol=1e-12)
    bad = np.full((3, 14), 0.5)
    bad[1, 7] = np.nan
    with pytest.raises(ValueError):
        params.prior_transform(bad)


# --------------------------------------------------------------------------------------
# 2. Emulator (_Regressor, TransferEmulator.predict_batch, EmulatedLinearBackend)
# --------------------------------------------------------------------------------------
def test_emulator_predict_batch_exact_and_block_invariant(emu, thetas_64):
    pts = thetas_64[:17]
    lnp_ref = np.empty((len(pts), emu.nz, emu.nk))
    pcb_ref = np.empty((len(pts), emu.nz, emu.nk))
    for i, t in enumerate(pts):
        lnp_ref[i], pcb_ref[i] = emu.predict(float(t[0]), float(t[1]), float(t[2]))

    for bs in (1, 7, 32, 64):
        lnp_b, pcb_b = emu.predict_batch(pts, block_size=bs)
        np.testing.assert_allclose(np.exp(lnp_b), np.exp(lnp_ref), rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(np.exp(pcb_b), np.exp(pcb_ref), rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(lnp_b, lnp_ref, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(pcb_b, pcb_ref, rtol=1e-13, atol=1e-14)

    # Also test (M, 3), (3,), and (0, 3)
    lnp_3, pcb_3 = emu.predict_batch(pts[:, :3], block_size=5)
    np.testing.assert_allclose(np.exp(lnp_3), np.exp(lnp_ref), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(np.exp(pcb_3), np.exp(pcb_ref), rtol=1e-13, atol=0.0)

    lnp_1, _ = emu.predict_batch(pts[0, :3])
    assert lnp_1.shape == (1, emu.nz, emu.nk)
    np.testing.assert_allclose(lnp_1[0], lnp_ref[0], rtol=1e-13, atol=0.0)

    lnp_0, pcb_0 = emu.predict_batch(np.empty((0, 3)))
    assert lnp_0.shape == (0, emu.nz, emu.nk) and pcb_0.shape == (0, emu.nz, emu.nk)

    # _Regressor block_size invariance directly
    x = emu.scale(pts[:, :3])
    c1 = emu.reg_tot(x, block_size=1)
    c7 = emu.reg_tot(x, block_size=7)
    c64 = emu.reg_tot(x, block_size=64)
    np.testing.assert_allclose(c1, c7, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(c1, c64, rtol=1e-14, atol=0.0)


def test_emulated_linear_compute_batch_dedup_and_fallback(emu, thetas_64):
    be_scalar = EmulatedLinearBackend(emulator=emu, nthreads=4, cache_size=2)
    be_batch = EmulatedLinearBackend(emulator=emu, nthreads=4, cache_size=2)
    # Build batch with duplicates and out-of-box point; ensure cache_size=2 does not break reuse
    rows = np.vstack([thetas_64[0], thetas_64[1], thetas_64[2], thetas_64[0], thetas_64[1]])
    rows[3, 3] += 0.03   # change n_s only
    rows[4, 4] += 0.15   # change lnAs only

    res_batch = be_batch.compute_batch(rows)
    res_scalar = [be_scalar.compute(*r[:5]) for r in rows]
    assert len(res_batch) == len(rows)
    assert be_batch.stats["emu_evals"] == 3  # 3 unique (omch2, ombh2, h)
    for a, b in zip(res_batch, res_scalar):
        np.testing.assert_allclose(a.pk_lin, b.pk_lin, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(a.pk_cb, b.pk_cb, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(a.chi, b.chi, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(a.ez, b.ez, rtol=1e-13, atol=0.0)
        assert a.sigma8_0 == pytest.approx(b.sigma8_0, rel=1e-13, abs=0.0)


# --------------------------------------------------------------------------------------
# 3. HMcode batch & thread invariance
# --------------------------------------------------------------------------------------
def test_hmcode_pnl_batch_exact_and_thread_invariant(emu, thetas_64):
    assert "reassoc" not in hmcode._FASTMATH
    be = EmulatedLinearBackend(emulator=emu, nthreads=4)
    pts = thetas_64[:8]
    lins = be.compute_batch(pts)
    k_b = np.stack([l.k_h for l in lins])
    pk_b = np.stack([l.pk_lin for l in lins])
    pcb_b = np.stack([l.pk_cb for l in lins])
    om_b = np.array([l.omega_m for l in lins])
    fnu_b = np.array([l.omega_nu_frac for l in lins])
    h_b = np.array([l.h for l in lins])
    ns_b = np.array([l.n_s for l in lins])
    ab_b = pts[:, 5]
    eta_b = np.array([nonlinear.eta_baryon(float(a)) for a in ab_b])

    ref = np.stack([
        hmcode.hmcode_pnl(
            lins[i].z, lins[i].k_h, lins[i].pk_lin, lins[i].omega_m,
            lins[i].omega_nu_frac, lins[i].h, lins[i].n_s, float(ab_b[i]), float(eta_b[i]),
            lins[i].pk_cb,
        )
        for i in range(len(pts))
    ])

    prev_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        got_t1 = hmcode.hmcode_pnl_batch(
            lins[0].z, k_b, pk_b, om_b, fnu_b, h_b, ns_b, ab_b, eta_b, pk_cb_batch=pcb_b, block_size=3
        )
        numba.set_num_threads(min(4, prev_threads))
        got_t4 = hmcode.hmcode_pnl_batch(
            lins[0].z, k_b, pk_b, om_b, fnu_b, h_b, ns_b, ab_b, eta_b, pk_cb_batch=pcb_b, block_size=5
        )
    finally:
        numba.set_num_threads(prev_threads)

    np.testing.assert_allclose(got_t1, ref, rtol=1e-13, atol=0.0)
    np.testing.assert_array_equal(got_t1, got_t4)


# --------------------------------------------------------------------------------------
# 4. CosmicShearModel.mu_batch & log_like_batch exactness, invariance, cache & threads
# --------------------------------------------------------------------------------------
def test_mu_and_loglike_batch_exactness_64_emulator(thetas_64):
    m_batch = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4)
    m_scalar = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4)
    lik = GaussianLikelihood()
    post_batch = api.build_posterior(m_batch, lik)
    post_scalar = api.build_posterior(m_scalar, lik)

    mus_b = m_batch.mu_batch(thetas_64)
    mus_s = np.array([m_scalar.mu(t) for t in thetas_64])
    # BLAS DGEMM (M=64) vs DGEMV (M=1) in TransferEmulator differs by ~2e-15 in lnP,
    # which is <= 7.6e-19 abs in mu (rtol <= 1e-13 away from J_0 zero-crossings).
    np.testing.assert_allclose(mus_b, mus_s, rtol=1e-13, atol=1e-18)
    non_zero = np.abs(mus_s) >= 1e-7
    np.testing.assert_allclose(mus_b[non_zero], mus_s[non_zero], rtol=1e-13, atol=0.0)

    ll_lik_b = lik.log_like_batch(mus_b, A_c=thetas_64[:, 8], delta_c=thetas_64[:, 7])
    ll_lik_s = np.array([lik.log_like(mus_s[i], A_c=float(thetas_64[i, 8]), delta_c=float(thetas_64[i, 7])) for i in range(64)])
    np.testing.assert_allclose(ll_lik_b, ll_lik_s, rtol=1e-13, atol=0.0)

    # Also test model=... convenience keyword and Posterior.log_like_batch
    ll_model_kw = lik.log_like_batch(thetas_64, model=m_batch)
    np.testing.assert_allclose(ll_model_kw, ll_lik_s, rtol=1e-13, atol=0.0)

    ll_post_b = post_batch.log_like_batch(thetas_64)
    ll_post_s = np.array([post_scalar.log_like(t) for t in thetas_64])
    np.testing.assert_allclose(ll_post_b, ll_post_s, rtol=1e-13, atol=0.0)

    lp_post_b = post_batch.log_post_batch(thetas_64)
    lp_post_s = np.array([post_scalar.log_post(t) for t in thetas_64])
    np.testing.assert_allclose(lp_post_b, lp_post_s, rtol=1e-13, atol=0.0)

    if BASELINE_NPZ.exists():
        # Baseline NPZ was recorded prior to Picks B1 (HMcode 1-halo) and B3 (Numba background).
        base = np.load(BASELINE_NPZ)
        np.testing.assert_allclose(mus_b, base["mu_emu"], rtol=1e-6, atol=1e-11)
        np.testing.assert_allclose(ll_lik_b, base["log_like_emu"], rtol=0.0, atol=1e-4)


def test_emulator_camb_background_batch_parity(thetas_64):
    """EmulatedLinearBackend with background_backend='camb' matches scalar compute to <= 1e-13."""
    from kidspipe.theory.emulator import EmulatedLinearBackend

    b_batch = EmulatedLinearBackend(z_grid=None, background_backend="camb", nthreads=4)
    b_scalar = EmulatedLinearBackend(z_grid=None, background_backend="camb", nthreads=4)
    pts = thetas_64[:8]
    res_b = b_batch.compute_batch(pts)
    res_s = [b_scalar.compute(r[0], r[1], r[2], r[3], r[4]) for r in pts]
    for rb, rs in zip(res_b, res_s):
        np.testing.assert_allclose(rb.pk_lin, rs.pk_lin, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(rb.chi, rs.chi, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(rb.ez, rs.ez, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(rb.sigma8_0, rs.sigma8_0, rtol=1e-13, atol=0.0)


def test_mu_and_loglike_batch_exactness_4_camb(thetas_64):
    pts4 = thetas_64[:4]
    m_batch = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="camb", nthreads=4)
    m_scalar = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="camb", nthreads=4)
    lik = GaussianLikelihood()

    mus_b = m_batch.mu_batch(pts4)
    mus_s = np.array([m_scalar.mu(t) for t in pts4])
    np.testing.assert_allclose(mus_b, mus_s, rtol=1e-13, atol=0.0)

    ll_b = lik.log_like_batch(mus_b, A_c=pts4[:, 8], delta_c=pts4[:, 7])
    ll_s = np.array([lik.log_like(mus_s[i], A_c=float(pts4[i, 8]), delta_c=float(pts4[i, 7])) for i in range(4)])
    np.testing.assert_allclose(ll_b, ll_s, rtol=1e-13, atol=0.0)

    if BASELINE_NPZ.exists():
        # Baseline NPZ was recorded prior to Pick B1 (HMcode 1-halo).
        base = np.load(BASELINE_NPZ)
        np.testing.assert_allclose(mus_b, base["mu_camb_4"], rtol=1e-11, atol=1e-16)
        np.testing.assert_allclose(ll_b, base["log_like_camb_4"], rtol=1e-12, atol=0.0)


def test_batch_size_invariance_and_duplicates(thetas_64):
    m = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4)
    lik = GaussianLikelihood()
    mu_64 = m.mu_batch(thetas_64)
    mu_7 = m.mu_batch(thetas_64[:7])
    mu_1 = m.mu_batch(thetas_64[:1])
    np.testing.assert_allclose(mu_7, mu_64[:7], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(mu_1, mu_64[:1], rtol=1e-13, atol=0.0)

    # Duplicate rows must give identical outputs and evaluate the unique cosmology once
    m_dup = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4)
    dup_pts = np.stack([thetas_64[0], thetas_64[3], thetas_64[0], thetas_64[3]])
    mu_dup = m_dup.mu_batch(dup_pts)
    assert m_dup.linear.stats["emu_evals"] == 2
    np.testing.assert_array_equal(mu_dup[0], mu_dup[2])
    np.testing.assert_array_equal(mu_dup[1], mu_dup[3])
    np.testing.assert_allclose(mu_dup[0], mu_64[0], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(mu_dup[1], mu_64[3], rtol=1e-13, atol=0.0)

    ll_dup = lik.log_like_batch(mu_dup, A_c=dup_pts[:, 8], delta_c=dup_pts[:, 7])
    assert ll_dup[0] == ll_dup[2] and ll_dup[1] == ll_dup[3]


def test_cache_eviction_safety_inside_batch(thetas_64):
    """M=16 with cache_size=2 and repeated row at [0] and [15] must not drop or corrupt entries."""
    m = CosmicShearModel(
        fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4, cache_size=2
    )
    pts = np.vstack([thetas_64[:15], thetas_64[0:1]])
    mu_b = m.mu_batch(pts, limber_block_size=4, block_size=4)
    np.testing.assert_array_equal(mu_b[0], mu_b[15])
    # Scalar evaluation afterwards must still match
    mu_0 = m.mu(pts[0])
    np.testing.assert_allclose(mu_b[0], mu_0, rtol=1e-13, atol=0.0)
    assert len(m._stage2) <= 2


def test_mixed_inbox_outofbox_and_kcap_reduced(thetas_64):
    m_fast = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4)
    m_ref = CosmicShearModel(fidelity="fast", pnl_backend="hmcode", z_grid="spec", linear_backend="emulator", nthreads=4)
    pts = thetas_64[:3].copy()
    pts[1, 0] = 0.45  # out of emulator box -> exact CAMB fallback
    mu_b = m_fast.mu_batch(pts)
    mu_s = np.array([m_ref.mu(t) for t in pts])
    np.testing.assert_allclose(mu_b, mu_s, rtol=1e-13, atol=0.0)
    assert m_fast.linear.stats["fallback"] == 1

    # kcap + reduced z_grid
    m_kcap_b = CosmicShearModel(fidelity="kcap", pnl_backend="hmcode", z_grid="reduced", linear_backend="emulator", nthreads=4)
    m_kcap_s = CosmicShearModel(fidelity="kcap", pnl_backend="hmcode", z_grid="reduced", linear_backend="emulator", nthreads=4)
    np.testing.assert_allclose(
        m_kcap_b.mu_batch(thetas_64[:4]),
        np.array([m_kcap_s.mu(t) for t in thetas_64[:4]]),
        rtol=1e-13, atol=0.0,
    )


def test_thread_safety_concurrent_mu_batch_and_mu(thetas_64):
    pts = thetas_64[:8]
    ref_m = CosmicShearModel(z_grid="reduced", n_ell=100, n_nodes=300, linear_backend="emulator", nthreads=1, cache_size=2)
    ref_mu = ref_m.mu_batch(pts)

    shared = CosmicShearModel(z_grid="reduced", n_ell=100, n_nodes=300, linear_backend="emulator", nthreads=1, cache_size=2)
    errs: list[str] = []
    bad: list[int] = []

    def worker(tid: int):
        rng = np.random.default_rng(100 + tid)
        for step in range(6):
            idx = rng.integers(0, len(pts), size=3)
            try:
                got = shared.mu_batch(pts[idx])
                if not np.allclose(got, ref_mu[idx], rtol=1e-10, atol=0.0):
                    bad.append(tid * 10 + step)
                s_got = shared.mu(pts[idx[-1]])
                if not np.allclose(s_got, ref_mu[idx[-1]], rtol=1e-10, atol=0.0):
                    bad.append(tid * 10 + step)
            except Exception as exc:  # noqa: BLE001
                errs.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errs and not bad, (errs, bad)


def test_posterior_batch_handles_oob_and_nan(thetas_64):
    m = CosmicShearModel(z_grid="reduced", n_ell=100, n_nodes=300, linear_backend="emulator", nthreads=4)
    lik = GaussianLikelihood()
    post = api.build_posterior(m, lik)
    pts = thetas_64[:4].copy()
    pts[1, 0] = 2.0    # out of prior bounds
    pts[2, 7] = np.nan # NaN -> out of bounds -> -inf
    ll_b = post.log_like_batch(pts)
    lp_b = post.log_post_batch(pts)
    assert np.isfinite(ll_b[0]) and np.isfinite(ll_b[3])
    assert ll_b[1] == -np.inf and ll_b[2] == -np.inf
    assert lp_b[1] == -np.inf and lp_b[2] == -np.inf
    assert ll_b[0] == pytest.approx(post.log_like(pts[0]), rel=1e-13, abs=0.0)


# --------------------------------------------------------------------------------------
# 5. Nested sampling integration (dynesty vectorized + mocked nautilus vectorized)
# --------------------------------------------------------------------------------------
def test_safe_loglike_batch_and_make_nested_functions():
    p = SimpleParams(["a", "b"], [[-2.0, 2.0], [-2.0, 2.0]])
    post = Posterior(lambda th: th, lambda mu: -0.5 * np.sum(mu ** 2), p)
    ll_s, pt_s, nd_s = make_nested_functions(post, vectorized=False)
    ll_b, pt_b, nd_b = make_nested_functions(post, vectorized=True)
    assert nd_s == nd_b == 2
    assert ll_s == post.log_like and ll_b == post.log_like_batch

    def flaky(X):
        X = np.asarray(X)
        if np.any(X[:, 0] > 1.0):
            raise RuntimeError("boom")
        return np.where(X[:, 1] > 1.0, np.nan, -np.sum(X ** 2, axis=1))

    s = safe_loglike_batch(flaky)
    out = s(np.array([[0.2, 0.3], [1.5, 0.0], [0.1, 1.5]]))
    assert np.isfinite(out[0]) and out[1] < -1e299 and out[2] < -1e299


def test_dynesty_vectorized_gaussian_evidence():
    pytest.importorskip("dynesty")
    rng = np.random.default_rng(7)
    d = 3
    L = 6.0
    A = rng.normal(size=(d, d))
    cov = 0.25 * (A @ A.T / d + np.eye(d))
    prec = np.linalg.inv(cov)
    logn = -0.5 * np.linalg.slogdet(2 * np.pi * cov)[1]
    mu = rng.uniform(-1, 1, d)

    def ll_batch(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        diff = X - mu[None, :]
        return logn - 0.5 * np.einsum("ij,jk,ik->i", diff, prec, diff)

    def pt_batch(u):
        u = np.asarray(u, dtype=np.float64)
        return -L + 2.0 * L * u

    res = run_dynesty(
        ll_batch, pt_batch, d, nlive=200, dlogz=0.1, seed=5,
        vectorized=True, prior_vectorized=True, queue_size=16,
    )
    lnz_true = -d * np.log(2.0 * L)
    assert abs(res.log_z - lnz_true) < max(4 * res.log_z_err, 0.15), (res.log_z, lnz_true)


def test_nautilus_vectorized_wiring_mock(monkeypatch):
    seen = {}

    class _FakeSampler:
        def __init__(self, prior, likelihood, n_dim, n_live, n_networks, vectorized, seed, pool, filepath, resume, **kw):
            seen["vectorized"] = vectorized
            seen["n_dim"] = n_dim
            u_batch = np.linspace(0.1, 0.9, 5 * n_dim).reshape(5, n_dim)
            th_batch = prior(u_batch)
            ll_batch = likelihood(th_batch)
            seen["th_shape"] = th_batch.shape
            seen["ll_shape"] = ll_batch.shape
            seen["ll_vals"] = ll_batch
            self.log_z = -4.2
            self.n_like = 5
            self.n_eff = 100.0

        def run(self, **kw):
            return True

        def posterior(self):
            pts = np.zeros((4, seen["n_dim"]))
            logw = np.zeros(4)
            logl = np.full(4, -1.0)
            return pts, logw, logl

    fake_mod = types.ModuleType("nautilus")
    fake_mod.Sampler = _FakeSampler
    monkeypatch.setitem(sys.modules, "nautilus", fake_mod)

    m = CosmicShearModel(z_grid="reduced", n_ell=80, n_nodes=200, linear_backend="emulator", nthreads=2)
    lik = GaussianLikelihood()
    post = api.build_posterior(m, lik)
    ll, pt, nd = make_nested_functions(post, vectorized=True)
    res = run_nautilus(ll, pt, nd, n_live=50, n_eff=100, vectorized=True, prior_vectorized=True, seed=1)
    assert seen["vectorized"] is True
    assert seen["th_shape"] == (5, 14)
    assert seen["ll_shape"] == (5,)
    assert np.all(np.isfinite(seen["ll_vals"]))
    assert res.sampler == "nautilus" and res.n_eval == 5
