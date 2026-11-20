"""Tests for kidspipe.theory.hankel (spec s7 FFT Hankel, s8 binning, fast linear operator)."""
import json
import os
import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gamma, ive, jv

from kidspipe.theory import hankel as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

ELL = np.logspace(-1, np.log10(5e5), 400)          # kcap's Limber ell grid (spec s6.2)
GLX, GLW = np.polynomial.legendre.leggauss(16)


# ----------------------------------------------------------------------------------------
# helpers: independent direct quadrature of (1/2pi) int l C(l) J_mu(l theta) dl
# ----------------------------------------------------------------------------------------
def cfun_faithful(ell, c):
    """Callable C(l) = the spec s7.2 interpolant (ln C linear in ln l; slopes +1 / -3 outside)."""
    lnl, lnc = np.log(ell), np.log(c)

    def f(x):
        lx = np.log(x)
        out = np.interp(lx, lnl, lnc)
        out = np.where(lx < lnl[0], lnc[0] + (lx - lnl[0]), out)
        out = np.where(lx > lnl[-1], lnc[-1] - 3.0 * (lx - lnl[-1]), out)
        return np.exp(out)
    return f


def direct_xi(theta, mu, ell, cfun, lmax=5e7):
    """Composite 16-point Gauss-Legendre on panels aligned with the ell nodes and no wider than
    pi/theta (>= ~8 points per half-oscillation); no FFT, no operator code involved."""
    br = np.concatenate([[0.0], ell, [lmax]])
    total = 0.0
    for a, b in zip(br[:-1], br[1:]):
        n = max(1, int(np.ceil((b - a) * theta / np.pi)))
        e = np.linspace(a, b, n + 1)
        x = 0.5 * (e[1:] - e[:-1])[:, None] * GLX[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]
        w = 0.5 * (e[1:] - e[:-1])[:, None] * GLW[None, :]
        total += np.sum(w * x * cfun(x) * jv(mu, x * theta))
    return total / (2.0 * np.pi)


def lagrange_free_toy(lc=100.0):
    """Smooth positive toy with the same asymptotics as the extrapolation rules (+1 / -3)."""
    return ELL / (1.0 + (ELL / lc) ** 4)


def realistic_spectra():
    """15 cosmic-shear-like spectra built from the spec's C_l anchors (data/validation): a
    natural cubic spline in ln-ln through l = 10, 100, 1e3, 1e4 for pairs 1-1, 3-3, 5-5, 5-1."""
    with open(os.path.join(DATA, "validation", "anchors_test_cosmology.json")) as fh:
        A = json.load(fh)["mead_A2.0"]
    lnl = np.log(ELL)
    lo, hi = np.log(10.0), np.log(1e4)

    def shape(name):
        pts = np.array([10, 100, 1000, 10000])
        y = np.log([A[f"Cl_{name}"][str(p)] for p in pts])
        s = CubicSpline(np.log(pts), y, bc_type="natural")
        out = np.where(lnl < lo, y[0] + s(lo, 1) * (lnl - lo), y[-1] + s(hi, 1) * (lnl - hi))
        m = (lnl >= lo) & (lnl <= hi)
        out[m] = s(lnl[m])
        return np.exp(out)

    S = {1: shape("1_1"), 3: shape("3_3"), 5: shape("5_5")}
    S[2] = np.sqrt(S[1] * S[3])
    S[4] = np.sqrt(S[3] * S[5])
    C = np.zeros((5, 5, ELL.size))
    for a in range(5):
        for b in range(5):
            C[a, b] = np.sqrt(S[a + 1] * S[b + 1]) * (1.0 if a == b else 0.9)
    C[4, 0] = C[0, 4] = shape("5_1")
    return C, A


def kv450_cut_cov():
    cov = np.loadtxt(os.path.join(DATA, "kv450", "cov_analytic.txt"))
    idx = []
    for p in range(15):
        idx += [p * 18 + b for b in range(7)] + [p * 18 + 9 + b for b in range(3, 9)]
    idx = np.array(idx)
    return cov[np.ix_(idx, idx)]


def cut_vec(xp, xm):
    return np.concatenate([np.concatenate([xp[p, :7], xm[p, 3:]]) for p in range(15)])


# ----------------------------------------------------------------------------------------
# edges / grid arithmetic
# ----------------------------------------------------------------------------------------
def test_edges_match_spec():
    e = H.THETA_EDGES_ARCMIN
    ref = [0.5, 1.01777898, 2.0717481, 4.21716333, 8.58428037, 17.4738002, 35.56893304,
           72.40262468, 147.37973879, 300.0]
    assert e.size == 10
    np.testing.assert_allclose(e, ref, rtol=2e-8)
    np.testing.assert_allclose(e[1:] / e[:-1], 600.0 ** (1.0 / 9.0), rtol=1e-13)
    hb = H.HankelBinner(ELL, fidelity="kcap")
    np.testing.assert_allclose(hb.edges_rad, e * np.pi / 10800.0, rtol=1e-14)
    assert H.PAIRS[0] == (0, 0) and H.PAIRS[1] == (1, 0) and H.PAIRS[-1] == (4, 4) and len(H.PAIRS) == 15


def test_fft_grid_and_theta_nodes():
    l, dl = H.fft_grid()
    assert l.size == 1024
    assert dl == pytest.approx(0.0240814163, abs=1e-10)
    assert l[0] == pytest.approx(9.645e-5, rel=1e-3) and l[-1] == pytest.approx(4.8226e6, rel=1e-4)
    theta = 1.0 / l[::-1]
    arc = theta / H.ARCMIN
    assert arc[0] == pytest.approx(7.1e-4, rel=0.02) and arc[-1] == pytest.approx(3.6e7, rel=0.02)
    inside = np.nonzero((arc >= 0.5) & (arc <= 300.0))[0]
    assert inside.size == 265 and inside[0] == 273 and inside[-1] == 537      # spec s7.2 step 5


def test_kcap_bin_node_counts_and_weights():
    hb = H.HankelBinner(ELL, fidelity="kcap")
    counts = [(hb.W_kcap[b] != 0).sum() for b in range(9)]
    assert counts == [29, 30, 29, 30, 29, 30, 29, 30, 29]                       # spec s8
    # weights reproduce trapz(theta*xi)/trapz(theta) for xi=const exactly and sum to 1
    np.testing.assert_allclose(hb.W_kcap @ np.ones(1024), 1.0, rtol=1e-13)
    # a pure power law theta^-1: node trapezoid of theta*theta^-1 is exactly the interval length
    th = hb.theta_fft
    for b in range(9):
        s = np.nonzero(hb.W_kcap[b])[0]
        ref = (th[s[-1]] - th[s[0]]) / (0.5 * (th[s[-1]] ** 2 - th[s[0]] ** 2))
        assert hb.W_kcap[b] @ (1.0 / th) == pytest.approx(ref, rel=1e-12)


def test_exact_bin_operator_vs_quad():
    hb = H.HankelBinner(ELL, fidelity="kcap")
    th = hb.theta_fft
    f = lambda t: 1.0 / (t * (1.0 + (t / 3e-3) ** 1.5))        # smooth, power-law like
    W = hb.W_exact
    e = hb.edges_rad
    for b in range(9):
        ref = quad(lambda t: t * f(t), e[b], e[b + 1], epsabs=0, epsrel=1e-12)[0] / (0.5 * (e[b + 1] ** 2 - e[b] ** 2))
        assert W[b] @ f(th) == pytest.approx(ref, rel=2e-7)
    np.testing.assert_allclose(W @ np.ones(th.size), 1.0, rtol=1e-12)


# ----------------------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------------------
def test_kernel_unit_modulus_cached_and_matches_complex_gamma():
    for mu in (0, 4):
        K = H.hankel_kernel(mu)
        assert K.shape == (513,)
        assert H.hankel_kernel(mu) is K                       # cached
        np.testing.assert_allclose(np.abs(K), 1.0, atol=1e-14)
        assert K[0] == pytest.approx(1.0)
        # independent check with explicit complex Gamma via loggamma difference (small k)
        from scipy.special import loggamma
        _, dl = H.fft_grid()
        m = np.arange(513)
        k = 2 * np.pi * m / (dl * 1024)
        sel = k < 20
        a = 0.5 * (1 + mu)
        ref = np.exp(1j * k[sel] * H.LN2_KCAP) * np.exp(loggamma(a + 0.5j * k[sel]) - loggamma(a - 0.5j * k[sel]))
        np.testing.assert_allclose(K[sel], ref, atol=1e-12)
    # Taylor check of the phase of the mu=0 kernel at the first mode:
    # arg K = k ln2 + 2 Im lnGamma(1/2 + ik/2) = k [ln2 + psi(1/2)] - k^3 psi''(1/2)/24 + O(k^5)
    from scipy.special import digamma, polygamma
    k1 = 2 * np.pi / (dl * 1024)
    expect = k1 * (H.LN2_KCAP + digamma(0.5)) - k1 ** 3 * polygamma(2, 0.5) / 24.0
    assert np.angle(H.hankel_kernel(0)[1]) == pytest.approx(expect, rel=2e-3)


# ----------------------------------------------------------------------------------------
# interpolation rules (spec s7.2 step 1)
# ----------------------------------------------------------------------------------------
def test_interpolation_rules():
    l, _ = H.fft_grid()
    I = H.ClInterpolator(ELL)
    # exact power law inside, slope +1 below 0.1, -3 above 5e5
    c = 2.0 * ELL ** -1.3
    out = I(c[None])[0]
    inside = (l >= ELL[0]) & (l <= ELL[-1])
    np.testing.assert_allclose(out[inside], 2.0 * l[inside] ** -1.3, rtol=1e-12)
    lo = l < ELL[0]
    np.testing.assert_allclose(out[lo], c[0] * (l[lo] / ELL[0]), rtol=1e-12)
    hi = l > ELL[-1]
    np.testing.assert_allclose(out[hi], c[-1] * (l[hi] / ELL[-1]) ** -3, rtol=1e-12)
    # negative spectrum: sign restored, same magnitudes
    np.testing.assert_allclose(I((-c)[None])[0], -out, rtol=1e-12)
    # mixed sign: linear in ln l, constant extrapolation
    cm = np.sin(np.log(ELL)) * 1e-9
    om = I(cm[None])[0]
    assert np.all(om[lo] == cm[0]) and np.all(om[hi] == cm[-1])
    j = 200
    lm = np.log(l[inside][50])
    ref = np.interp(lm, np.log(ELL), cm)
    assert om[inside][50] == pytest.approx(ref, rel=1e-12, abs=1e-20)
    # exact zero anywhere -> mixed branch (linear), no NaN
    cz = c.copy()
    cz[10] = 0.0
    oz = I(cz[None])[0]
    assert np.all(np.isfinite(oz))
    # rows are independent: a batch gives the same result as single rows
    batch = I(np.stack([c, -c, cm]))
    np.testing.assert_allclose(batch[0], out, rtol=1e-14)
    np.testing.assert_allclose(batch[2], om, rtol=1e-14)


# ----------------------------------------------------------------------------------------
# FFT Hankel transform vs direct quadrature
# ----------------------------------------------------------------------------------------
def test_fft_exact_samples_vs_direct_quadrature():
    """Strict test of the Hankel machinery itself (kernel, normalisation, index reversal): feed a
    smooth analytic C sampled exactly on the 1024-node FFT grid (so the log-linear interpolation
    is the identity) and compare with direct Gauss-Legendre quadrature of the same analytic C.
    Spec s7.2 quotes ~1e-8 (mu=0, 1') and 4e-6 (mu=4, 10'); we require 1e-7 everywhere."""
    l, _ = H.fft_grid()
    br = np.logspace(-3, np.log10(5e7), 60)
    for lc in (100.0, 1000.0):
        f = lambda x: x / (1.0 + (x / lc) ** 4)
        theta, xp, xm = H.cl_to_xi_fft(l, f(l)[None])
        for arc in (1.0, 10.0):
            j = np.argmin(abs(theta - arc * H.ARCMIN))
            r0 = direct_xi(theta[j], 0, br, f)
            r4 = direct_xi(theta[j], 4, br, f)
            assert xp[0, j] == pytest.approx(r0, rel=1e-7), (lc, arc)
            assert abs(xm[0, j] - r4) < 1e-7 * abs(r4) + 1e-8 * abs(r0), (lc, arc, xm[0, j] / r4 - 1)


@pytest.mark.parametrize("arcmin", [1.0, 10.0])
@pytest.mark.parametrize("name", ["pow1", "pow15", "toy"])
def test_fft_vs_direct_quadrature(name, arcmin):
    """FFT route (incl. the kcap log-linear C interpolation on the 400-node ell grid) against
    direct quadrature of the *same* piecewise interpolant.  The residual is the trapezoid error of
    the kinked interpolant (~1e-6) plus, for the pure power laws whose l*C does not decay at the end
    of the FFT grid, wrap-around (3e-4 at 1', 3e-6 at 10').  Errors are measured against |xi_+|
    (for mu=4 the FFT error floor is set by the xi_+ scale, not by the small xi_- value)."""
    if name == "pow1":
        c, tol = ELL ** -1.0, (1e-3 if arcmin < 5 else 1e-5)
    elif name == "pow15":
        c, tol = 3.0 * ELL ** -1.5, (1e-3 if arcmin < 5 else 1e-5)
    else:
        c, tol = lagrange_free_toy(100.0), 2e-6
    cf = cfun_faithful(ELL, c)
    theta, xp, xm = H.cl_to_xi_fft(ELL, c[None])
    j = np.argmin(abs(theta - arcmin * H.ARCMIN))
    r0 = direct_xi(theta[j], 0, ELL, cf)
    r4 = direct_xi(theta[j], 4, ELL, cf)
    assert abs(xp[0, j] - r0) < tol * abs(r0), (name, arcmin, xp[0, j] / r0 - 1)
    assert abs(xm[0, j] - r4) < tol * abs(r0), (name, arcmin, (xm[0, j] - r4) / r0)
    if name == "toy":
        assert xm[0, j] == pytest.approx(r4, rel=(3e-4 if arcmin < 5 else 1e-5))


def test_fft_power_law_analytic():
    """C = l^-1 -> xi_+ = 1/(2 pi theta) exactly (up to the l<0.1 slope-+1 extrapolation,
    which removes ~ 0.1/(2pi) of the integral)."""
    theta, xp, _ = H.cl_to_xi_fft(ELL, (ELL ** -1.0)[None])
    for arc, tol in ((1.0, 1e-3), (10.0, 1e-3), (60.0, 3e-3)):
        j = np.argmin(abs(theta - arc * H.ARCMIN))
        exact = 1.0 / (2 * np.pi * theta[j])
        assert xp[0, j] == pytest.approx(exact, rel=tol)
    # Mellin transform: int_0^inf l^-n J_4(l th) l dl = th^(n-2) 2^(1-n) G(3-n/2)/G(n/2+2)... check n=1.5
    n = 1.5
    c = ELL ** -n
    theta, _, xm = H.cl_to_xi_fft(ELL, c[None])
    j = np.argmin(abs(theta - 5 * H.ARCMIN))
    s = 2.0 - n
    exact = theta[j] ** (n - 2.0) * 2.0 ** (s - 1) * gamma((4 + s) / 2) / gamma((4 - s) / 2 + 1) / (2 * np.pi)
    assert xm[0, j] == pytest.approx(exact, rel=2e-3)


def test_fft_batch_single_call_consistency():
    rng = np.random.default_rng(1)
    base = lagrange_free_toy(300.0)
    C = np.stack([base * (1 + 0.3 * rng.random()) for _ in range(15)])
    th, xp, xm = H.cl_to_xi_fft(ELL, C)
    assert xp.shape == (15, 1024) and xm.shape == (15, 1024)
    for p in (0, 7, 14):
        th1, xp1, xm1 = H.cl_to_xi_fft(ELL, C[p:p + 1])
        np.testing.assert_allclose(xp[p], xp1[0], rtol=1e-9, atol=1e-12 * np.abs(xp[p]).max())
        np.testing.assert_allclose(xm[p], xm1[0], rtol=1e-9, atol=1e-12 * np.abs(xm[p]).max())
    # linearity
    th2, xp2, _ = H.cl_to_xi_fft(ELL, 2.5 * C)
    np.testing.assert_allclose(xp2, 2.5 * xp, rtol=1e-9, atol=1e-12 * np.abs(xp).max())


# ----------------------------------------------------------------------------------------
# exact bin-averaged Bessel functions and the fast operator
# ----------------------------------------------------------------------------------------
def test_bessel_antiderivative_vs_quad():
    x = np.array([1e-4, 1e-2, 0.3, 1.0, 1.99, 2.01, 3.0, 7.5, 40.0, 300.0])
    for mu in (0, 4):
        F = H.bessel_antiderivative(x, mu)
        ref = np.array([quad(lambda t: t * jv(mu, t), 0, xx, epsabs=0, epsrel=1e-12, limit=2000)[0] for xx in x])
        np.testing.assert_allclose(F, ref, rtol=1e-9, atol=1e-30)


@pytest.mark.parametrize("mu", [0, 4])
def test_bessel_bin_average_vs_quad(mu):
    ta, tb = 4.217 * H.ARCMIN, 8.584 * H.ARCMIN
    for l in (1e-3, 0.5, 30.0, 400.0, 3000.0, 2e4):
        ref = quad(lambda t: t * jv(mu, l * t), ta, tb, epsabs=0, epsrel=1e-12, limit=4000)[0] / (0.5 * (tb ** 2 - ta ** 2))
        got = H.bessel_bin_average(np.array([l]), ta, tb, mu)[0]
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-14)
    assert H.bessel_bin_average(np.array([1e-9]), ta, tb, 0)[0] == pytest.approx(1.0, abs=1e-12)


def test_operator_matches_direct_quadrature_of_interpolant():
    """The fast operator must reproduce the exact continuous definition + exact bin average.
    Reference: direct quadrature of the *same* power-law-extended interpolant at 40 theta points
    per bin (Gauss-Legendre in theta), for a smooth positive toy spectrum."""
    hb = H.HankelBinner(ELL)
    c = lagrange_free_toy(100.0)
    xp, xm = hb.xi_binned(np.tile(c, (15, 1)))
    cf = cfun_faithful(ELL, c)
    gx, gw = np.polynomial.legendre.leggauss(12)
    for b in (0, 2, 5):                                     # a few bins (cost)
        ta, tb = hb.edges_rad[b], hb.edges_rad[b + 1]
        th = 0.5 * (tb - ta) * gx + 0.5 * (tb + ta)
        w = 0.5 * (tb - ta) * gw * th / (0.5 * (tb ** 2 - ta ** 2))
        for mu, got in ((0, xp[0, b]), (4, xm[0, b])):
            ref = sum(wi * direct_xi(ti, mu, ELL, cf) for wi, ti in zip(w, th))
            # interpolation difference (cubic Lagrange vs log-linear) is ~1e-4 at most
            assert got == pytest.approx(ref, rel=5e-4), (b, mu, got / ref - 1)


def test_operator_shape_linearity_and_pair_ordering():
    hb = H.HankelBinner(ELL)
    assert hb.M_plus.shape == (9, ELL.size) and hb.M_minus.shape == (9, ELL.size)
    rng = np.random.default_rng(3)
    C = np.zeros((5, 5, ELL.size))
    base = lagrange_free_toy(200.0)
    for j, i in H.PAIRS:
        C[j, i] = C[i, j] = base * (1 + 0.1 * rng.random() + 0.2 * (i + j))
    xp, xm = hb.xi_binned(C)
    assert xp.shape == (15, 9) and xm.shape == (15, 9)
    for p, (j, i) in enumerate(H.PAIRS):
        np.testing.assert_allclose(xp[p], hb.M_plus @ C[j, i], rtol=1e-12)
        np.testing.assert_allclose(xm[p], hb.M_minus @ C[j, i], rtol=1e-12)
    # only j >= i is read
    C2 = C.copy()
    for j, i in H.PAIRS:
        if j != i:
            C2[i, j] = 0.0
    xp2, _ = hb.xi_binned(C2)
    np.testing.assert_array_equal(xp, xp2)
    # (15, nell) input accepted
    xp3, _ = hb.xi_binned(C[H._PJ, H._PI])
    np.testing.assert_array_equal(xp, xp3)


def test_operator_convergence_with_quadrature_density():
    hb = H.HankelBinner(ELL)
    Mp2, Mm2 = H.bin_operator(ELL, hb.edges_rad, ppp=0.5, tail_stretch=0.5)
    for c in (ELL ** -1.0, lagrange_free_toy(300.0), ELL ** -1.5 * np.exp(-ELL / 1e4)):
        np.testing.assert_allclose(hb.M_plus @ c, Mp2 @ c, rtol=1e-8)
        np.testing.assert_allclose(hb.M_minus @ c, Mm2 @ c, rtol=1e-8)


def test_operator_vs_fft_route_realistic():
    """Fast operator vs FFT route (exact binning) on a cosmic-shear-like C(l): relative xi
    difference and chi2 difference with the KV450 covariance."""
    C, _ = realistic_spectra()
    fast = H.HankelBinner(ELL)
    kc = H.HankelBinner(ELL, fidelity="kcap")
    xp, xm = fast.xi_binned(C)
    ep, em = kc.xi_binned_fft(C, "exact")
    assert np.max(np.abs(xp / ep - 1)) < 2e-4
    assert np.max(np.abs(xm / em - 1)) < 5e-4
    Ci = kv450_cut_cov()
    d = cut_vec(xp, xm) - cut_vec(ep, em)
    dchi2 = d @ np.linalg.solve(Ci, d)
    print(f"[hankel] fast vs FFT(exact bin): max|dxi+/xi+|={np.max(np.abs(xp / ep - 1)):.2e}, "
          f"max|dxi-/xi-|={np.max(np.abs(xm / em - 1)):.2e}, dchi2={dchi2:.2e}")
    assert dchi2 < 1e-3
    # and vs the full faithful kcap route (includes the node-trapezoid binning bias <~0.7%)
    kp, km = kc.xi_binned(C)
    d2 = cut_vec(xp, xm) - cut_vec(kp, km)
    dchi2k = d2 @ np.linalg.solve(Ci, d2)
    print(f"[hankel] fast vs kcap (FFT + node trapezoid): dchi2={dchi2k:.3e}, "
          f"max rel xi+={np.max(np.abs(xp / kp - 1)):.2e}")
    assert np.max(np.abs(xp / kp - 1)) < 1.5e-2
    assert dchi2k < 0.1


def test_kcap_equals_fft_plus_trapezoid():
    C, _ = realistic_spectra()
    kc = H.HankelBinner(ELL, fidelity="kcap")
    th, xp, xm = kc.xi_theta(C)
    p = 3
    j, i = H.PAIRS[p]
    kp, km = kc.xi_binned(C)
    for b in (0, 4, 8):
        s = np.nonzero((th >= kc.edges_rad[b]) & (th < kc.edges_rad[b + 1]))[0]
        ref = np.trapz(th[s] * xp[p, s], th[s]) / np.trapz(th[s], th[s])
        assert kp[p, b] == pytest.approx(ref, rel=1e-12)
        ref = np.trapz(th[s] * xm[p, s], th[s]) / np.trapz(th[s], th[s])
        assert km[p, b] == pytest.approx(ref, rel=1e-12)


def test_validation_anchor_sanity():
    """Order-of-magnitude anchor check against data/validation (spec-implementation xi, pair 1-1,
    5-5, 5-1).  Only 4 C_l anchors exist, so the C(l) here is a crude ln-ln spline: amplitude and
    shape must agree to ~15 % in the well-constrained bins (guards normalisation / pi factors)."""
    C, A = realistic_spectra()
    kc = H.HankelBinner(ELL, fidelity="kcap")
    kp, km = kc.xi_binned(C)
    for name, p in (("1_1", H.PAIRS.index((0, 0))), ("5_5", H.PAIRS.index((4, 4))), ("5_1", H.PAIRS.index((4, 0)))):
        ref = np.array(A["xi"][name]["xi_plus_binned"])
        assert np.all(np.abs(kp[p, :7] / ref[:7] - 1) < 0.2), (name, kp[p, :7] / ref[:7] - 1)
    ref = np.array(A["xi"]["1_1"]["xi_minus_binned"])
    assert np.all(np.abs(km[0, 2:8] / ref[2:8] - 1) < 0.2)


def test_subsets_of_grid_and_other_ell_grids():
    """Operator works for other (even non-log-uniform) ell grids; results agree with the 400-pt grid."""
    c = lambda l: l / (1.0 + (l / 200.0) ** 4)
    ref = H.HankelBinner(ELL).xi_binned(np.tile(c(ELL), (15, 1)))
    ell2 = np.concatenate([np.logspace(-1, 2, 250, endpoint=False), np.logspace(2, np.log10(5e5), 450)])
    got = H.HankelBinner(ell2).xi_binned(np.tile(c(ell2), (15, 1)))
    np.testing.assert_allclose(got[0], ref[0], rtol=2e-4)
    np.testing.assert_allclose(got[1], ref[1], rtol=2e-4)
    with pytest.raises(ValueError):
        H.HankelBinner(ELL[::-1])
    with pytest.raises(ValueError):
        H.HankelBinner(ELL).xi_binned(np.ones((5, 5, 10)))


def test_dense_exact_binning_vs_kcap_bias_size():
    """kcap node-trapezoid vs exact bin average: documented bias up to ~0.6-0.7 % (spec s8)."""
    C, _ = realistic_spectra()
    kc = H.HankelBinner(ELL, fidelity="kcap")
    kp, _ = kc.xi_binned_fft(C, "kcap")
    ep, _ = kc.xi_binned_fft(C, "exact")
    r = np.abs(kp / ep - 1)
    assert r.max() < 1.0e-2
    assert r.max() > 1e-4        # the effect is real, not a bug that hides it


# ----------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------
def test_timing_fast_operator_and_fft():
    """The fast operator must beat the FFT path by >= 10x, both timed in this run.  The old fixed limit (2 ms) failed
    under CPU contention (MKL splits the 15x400 @ 400x18 product over every core and waits for the slowest: 7-9 ms with
    22 busy cores, 42 us on one thread) and could not see a fall-back to the FFT path (0.5-1.8 ms)."""
    try:
        from threadpoolctl import threadpool_limits
        one_blas_thread = threadpool_limits(limits=1)
    except ImportError:                       # optional; the relative check below still holds without it
        import contextlib
        one_blas_thread = contextlib.nullcontext()
    C, _ = realistic_spectra()
    fast = H.HankelBinner(ELL)
    kc = H.HankelBinner(ELL, fidelity="kcap")
    def best(f, n, blocks=5):                 # best of several blocks: robust to a block that was descheduled
        out = []
        for _ in range(blocks):
            t0 = time.perf_counter()
            for _ in range(n):
                f(C)
            out.append((time.perf_counter() - t0) / n)
        return min(out)

    with one_blas_thread:
        fast.xi_binned(C)
        kc.xi_binned(C)
        t_fast = best(fast.xi_binned, 40)
        t_fft = best(kc.xi_binned, 4)
    print(f"[hankel] timing: fast operator {t_fast * 1e6:.0f} us, FFT+trapezoid {t_fft * 1e3:.2f} ms, "
          f"ratio {t_fft / t_fast:.0f}")
    assert t_fast < t_fft / 10
    assert t_fft < 0.1


def test_operator_build_time():
    t0 = time.perf_counter()
    H.HankelBinner(ELL)
    dt = time.perf_counter() - t0
    print(f"[hankel] operator build: {dt:.2f} s")
    assert dt < 10.0


@pytest.mark.slow
def test_fft_vs_direct_quadrature_many_theta():
    c = lagrange_free_toy(500.0)
    cf = cfun_faithful(ELL, c)
    theta, xp, xm = H.cl_to_xi_fft(ELL, c[None])
    # absolute FFT error floor is set by the peak amplitude xi_+(0.5'), xi falls steeply at large theta
    peak = abs(xp[0, np.argmin(abs(theta - 0.5 * H.ARCMIN))])
    for arc in (0.5, 2.0, 5.0, 30.0, 100.0, 250.0):
        j = np.argmin(abs(theta - arc * H.ARCMIN))
        for mu, xi in ((0, xp[0, j]), (4, xm[0, j])):
            ref = direct_xi(theta[j], mu, ELL, cf)
            assert abs(xi - ref) < 2e-6 * peak, (arc, mu, xi / ref - 1)


# ----------------------------------------------------------------------------------------
# review additions: independent analytic references, tails, ordering, input validation, end-to-end
# ----------------------------------------------------------------------------------------
def test_pair_order_is_montepython_spec_s10_2():
    """Spec s10.2: for i in 0..4: for j in i..4: pair (j, i); theory read from bin_{j+1}_{i+1}."""
    ref = [(j, i) for i in range(5) for j in range(i, 5)]
    assert H.PAIRS == ref
    assert [H.PAIRS.index(p) for p in [(0, 0), (1, 0), (4, 0), (1, 1), (2, 1), (2, 2), (3, 3), (4, 4)]] \
        == [0, 1, 4, 5, 6, 9, 12, 14]
    # data-file column rule 1 + i*(2n-i+1)/2 + (j-i) (1-based column of pair (j,i)) = 1 + position
    for p, (j, i) in enumerate(H.PAIRS):
        assert 1 + i * (2 * 5 - i + 1) // 2 + (j - i) == 1 + p
    # the pair actually read from a (5,5,nell) array is C[j, i]: make C asymmetric and check
    hb = H.HankelBinner(ELL)
    C = np.zeros((5, 5, ELL.size))
    base = lagrange_free_toy(200.0)
    C[3, 1] = base                       # pair (j=3, i=1), position 8
    xp, _ = hb.xi_binned(C)
    assert np.all(xp[H.PAIRS.index((3, 1))] != 0) and np.count_nonzero(xp.any(axis=1)) == 1


def _xi_gauss(th, a, mu):
    """Closed form of (1/2pi) int_0^inf l exp(-a l^2) J_mu(l th) dl for mu = 0, 4 (Weber's formula)."""
    if mu == 0:
        return np.exp(-th * th / (4 * a)) / (2 * a) / (2 * np.pi)
    z = th * th / (8 * a)
    return np.sqrt(np.pi) * th / (8 * a ** 1.5) * (ive(1.5, z) - ive(2.5, z)) / (2 * np.pi)


@pytest.mark.parametrize("sigma", [300.0, 3000.0])
def test_analytic_gaussian_binned_and_unbinned(sigma):
    """Independent closed-form reference (Gaussian C(l) = exp(-l^2/sigma^2)): no quadrature of
    Bessel functions and no interpolation rule of the module is involved.  Errors are quoted
    relative to the peak xi_+(0) = sigma^2/(4 pi) because xi_- and the far tail are tiny."""
    a = 1.0 / sigma ** 2
    peak = 1.0 / (2 * a) / (2 * np.pi)
    c = np.exp(-a * ELL ** 2)                         # underflows to exact 0 at large l -> kcap 'mixed' branch
    hb = H.HankelBinner(ELL)
    xp, xm = hb.xi_binned(np.tile(c, (15, 1)))
    e = hb.edges_rad
    for mu, got in ((0, xp[0]), (4, xm[0])):
        ref = np.array([quad(lambda t: t * _xi_gauss(t, a, mu), e[b], e[b + 1], epsabs=0, epsrel=1e-11)[0]
                        / (0.5 * (e[b + 1] ** 2 - e[b] ** 2)) for b in range(9)])
        assert np.max(np.abs(got - ref)) < 2e-6 * peak, (mu, (got - ref) / peak)
    # FFT route: limited by kcap's own linear-in-ln(l) interpolation of the (zero-containing) Gaussian
    theta, fp, fm = H.cl_to_xi_fft(ELL, c[None])
    for arc in (0.5, 1.0, 3.0, 10.0, 30.0, 100.0):
        j = np.argmin(np.abs(theta - arc * H.ARCMIN))
        assert abs(fp[0, j] - _xi_gauss(theta[j], a, 0)) < 1e-3 * peak
        assert abs(fm[0, j] - _xi_gauss(theta[j], a, 4)) < 1e-3 * peak


def test_fast_operator_upper_tail_slope():
    """C = l^-1 keeps l*C*J_0 non-negligible up to the end of the grid, so the l^-3 tail beyond
    ell[-1] (integrated to l = 5e6) matters at 1' (a wrong tail exponent was NOT caught before).
    Reference: direct quadrature of the very same extended interpolant, same upper cut."""
    hb = H.HankelBinner(ELL)
    c = ELL ** -1.0
    xp, xm = hb.xi_binned(np.tile(c, (15, 1)))
    cf = cfun_faithful(ELL, c)
    # the sharp cut at l = 5e6 makes xi(theta) ripple with ~100 oscillations across bin 0: use many nodes
    gx, gw = np.polynomial.legendre.leggauss(160)
    for b in (0,):
        ta, tb = hb.edges_rad[b], hb.edges_rad[b + 1]
        th = 0.5 * (tb - ta) * gx + 0.5 * (tb + ta)
        w = 0.5 * (tb - ta) * gw * th / (0.5 * (tb ** 2 - ta ** 2))
        for mu, got in ((0, xp[0, b]), (4, xm[0, b])):
            ref = sum(wi * direct_xi(ti, mu, ELL, cf, lmax=H.L_MAX_FFT) for wi, ti in zip(w, th))
            assert got == pytest.approx(ref, rel=2e-5), (b, mu, got / ref - 1)
    # Narrow bins do not average the tail ripple away: a 0.1 % wide bin is sensitive to the tail
    # exponent at the several-percent level (the -55 of 729 tail piece), so this pins it down.
    edges = [0.9, 0.9009, 1.7, 1.7017]
    hn = H.HankelBinner(ELL, edges)
    xn, xmn = hn.xi_binned(np.tile(c, (15, 1)))
    gx, gw = np.polynomial.legendre.leggauss(8)
    for b in (0, 2):
        ta, tb = hn.edges_rad[b], hn.edges_rad[b + 1]
        th = 0.5 * (tb - ta) * gx + 0.5 * (tb + ta)
        w = 0.5 * (tb - ta) * gw * th / (0.5 * (tb ** 2 - ta ** 2))
        for mu, got in ((0, xn[0, b]), (4, xmn[0, b])):
            ref = sum(wi * direct_xi(ti, mu, ELL, cf, lmax=H.L_MAX_FFT) for wi, ti in zip(w, th))
            assert got == pytest.approx(ref, rel=1e-5), (b, mu, got / ref - 1)
    Mp, _ = H.bin_operator(ELL, hn.edges_rad, l_cap=ELL[-1])
    assert abs((Mp @ c)[0] / (hn.M_plus @ c)[0] - 1) > 1e-2       # the tail is really exercised


def test_interpolator_floor_and_exact_sample_identity():
    I = H.ClInterpolator(ELL)
    c = 2.0 * ELL ** -1.3
    c[7] = 1e-45                                         # positive but below the 1e-30 floor (spec s7.2 step 1)
    out = I(c[None])[0]
    l, _ = H.fft_grid()
    j = np.argmin(np.abs(np.log(l) - np.log(ELL[7])))
    assert out[j] == pytest.approx(1e-30, rel=1e-2)
    # sampled exactly on the FFT grid the interpolation is the identity
    f = l / (1 + (l / 100.0) ** 4)
    np.testing.assert_allclose(H.ClInterpolator(l)(f[None])[0], f, rtol=1e-12)


def test_input_validation_and_no_hidden_state():
    bad_ell = ELL.copy()
    bad_ell[5] = np.nan
    for fid in ("fast", "kcap"):
        with pytest.raises(ValueError):
            H.HankelBinner(bad_ell, fidelity=fid)
        for edges in ([1.0, 0.5, 3.0], [0.5], [1.0, 2.0, np.nan], [-1.0, 1.0, 2.0], [0.5, 0.5, 1.0]):
            with pytest.raises(ValueError):
                H.HankelBinner(ELL, edges, fidelity=fid)
    with pytest.raises(ValueError):
        H.HankelBinner(ELL, fidelity="exact")
    kc = H.HankelBinner(ELL, fidelity="kcap")
    C = np.tile(lagrange_free_toy(100.0), (15, 1))
    with pytest.raises(ValueError):
        kc.xi_binned_fft(C, "typo")                     # used to fall through silently to 'exact'
    # NaN / inf in C must raise, not return NaN rows silently, on every public route
    for hb in (H.HankelBinner(ELL), kc):
        for badv in (np.nan, np.inf):
            Cb = C.copy()
            Cb[3, 10] = badv
            with np.errstate(all="ignore"), pytest.raises(ValueError):
                hb.xi_binned(Cb)
    Cb = C.copy()
    Cb[3, 10] = np.nan
    with np.errstate(all="ignore"), pytest.raises(ValueError):
        kc.xi_theta(Cb)
    # the caller's ell array is copied (in-place edits afterwards cannot desynchronise the operator)
    e2 = ELL.copy()
    hb = H.HankelBinner(e2)
    before = hb.xi_binned(C)[0].copy()
    e2 *= 2.0
    np.testing.assert_array_equal(hb.ell, ELL)
    np.testing.assert_array_equal(hb.xi_binned(C)[0], before)
    assert not H.THETA_EDGES_ARCMIN.flags.writeable
    assert not H.fft_grid()[0].flags.writeable and not H.hankel_kernel(0).flags.writeable
    # deterministic: two independent builds agree bit for bit
    np.testing.assert_array_equal(H.HankelBinner(ELL).xi_binned(C)[0], before)


def test_exact_ln2_changes_xi_below_1e7():
    """Documentation claim: kcap's 8-digit ln 2 vs the exact one moves xi by < 1e-7 relative."""
    c = lagrange_free_toy(100.0)[None]
    _, a, b = H.cl_to_xi_fft(ELL, c)
    _, a2, b2 = H.cl_to_xi_fft(ELL, c, ln2=np.log(2.0))
    sl = slice(273, 538)
    assert np.max(np.abs(a2[0, sl] / a[0, sl] - 1)) < 1e-7
    assert np.max(np.abs(b2[0, sl] - b[0, sl])) < 1e-7 * np.max(np.abs(a[0, sl]))


def test_end_to_end_camb_limber_hankel_vs_spec_anchors():
    """Spec-anchor ladder step 3: CAMB (Takahashi halofit) -> Limber -> this module vs the stored
    xi anchors of the spec implementation (FFT + node trapezoid, the 'kcap' route).  Observed
    agreement is <= 2.8e-3 (the worst case is xi_- in bin 0 of pair 1-1, which depends on the Limber l > 1e4 tail;
    the Hankel step itself reproduces the spec's reference transcription to 1e-9); the fast route differs by the documented node-trapezoid bias (<~ 1.2 %)."""
    camb = pytest.importorskip("camb")
    from kidspipe.theory.limber import cl_shear, default_ell
    from kidspipe.theory.nz import load_nz
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.1, mnu=0.06, num_massive_neutrinos=1)
    pars.InitPower.set_params(As=np.exp(3.0) * 1e-10, ns=0.96)
    zs = np.concatenate([np.linspace(0, 2, 100, endpoint=False), np.linspace(2, 6, 50)])
    pars.set_matter_power(redshifts=list(zs[::-1]), kmax=20.0, nonlinear=True)
    pars.NonLinearModel.set_params(halofit_version="takahashi")
    pars.NonLinear = camb.model.NonLinear_both
    res = camb.get_results(pars)
    kh, zz, pk = res.get_matter_power_spectrum(minkh=7.1e-5, maxkh=28.5, npoints=250)
    x = np.log(kh)
    xn = np.linspace(x[-1], np.log(500.0), 200)[1:]
    b = np.array([np.polyfit(x[-3:], np.log(p[-3:]), 1) for p in pk])
    pke = np.hstack([pk, np.exp(b[:, 1:2] + b[:, 0:1] * xn[None, :])])
    khe = np.concatenate([kh, np.exp(xn)])
    z_tab = np.linspace(0, 6, 6000)
    chi_tab = res.comoving_radial_distance(z_tab) * 0.7
    nz = load_nz()
    ell = np.logspace(-1, np.log10(5e5), 400)
    C = cl_shear(chi_tab, z_tab, pars.omegam, zz, khe, pke, nz.z, nz.n, 0.0, ell)
    with open(os.path.join(DATA, "validation", "anchors_test_cosmology.json")) as fh:
        A = json.load(fh)["takahashi"]["xi"]
    kc = H.HankelBinner(ell, fidelity="kcap")
    fa = H.HankelBinner(ell)
    kp, km = kc.xi_binned(C)
    fp, fm = fa.xi_binned(C)
    for name, pair in (("1_1", (0, 0)), ("3_3", (2, 2)), ("5_5", (4, 4)), ("5_1", (4, 0))):
        p = H.PAIRS.index(pair)
        rp, rm = np.array(A[name]["xi_plus_binned"]), np.array(A[name]["xi_minus_binned"])
        assert np.max(np.abs(kp[p] / rp - 1)) < 5e-3, (name, kp[p] / rp - 1)
        assert np.max(np.abs(km[p] / rm - 1)) < 5e-3, (name, km[p] / rm - 1)
        assert np.max(np.abs(fp[p] / rp - 1)) < 1.5e-2 and np.max(np.abs(fm[p] / rm - 1)) < 1.5e-2
        assert np.max(np.abs(fp[p, :7] / rp[:7] - 1)) < 8e-3          # kept bins only: bias <~ 0.7 %
