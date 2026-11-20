"""Tests for COSEBIs: filters (tables vs mpmath regeneration), W_n(l), E_n from C_ell vs from xi, FITS data/likelihood."""
import json
import os
import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

from kidspipe.theory import cosebis as C
from kidspipe.likelihood import cosebis as LC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "data", "cosebis")


@pytest.fixture(scope="module")
def F300():
    return C.CosebisFilters(0.5, 300.0, 20)


@pytest.fixture(scope="module")
def F100():
    return C.CosebisFilters(0.5, 100.0, 20)


# --------------------------------------------------------------------------------------------- tables / filters
def test_table_parsing():
    r = C.parse_roots_table(os.path.join(CD, "roots_norms", "Root_0.50-300.00.table"))
    nm, ns = C.parse_norm_table(os.path.join(CD, "roots_norms", "Normalization_0.50-300.00.table"))
    assert sorted(r) == list(range(1, 21)) and sorted(nm) == list(range(1, 21))
    assert all(r[n].size == n + 1 for n in r)
    np.testing.assert_allclose(r[1], [5.0883097, 6.2059688], rtol=0, atol=1e-12)
    assert abs(nm[1] - 0.37596958385331224) < 1e-15
    assert len(ns[1].replace(".", "")) > 40                      # full-precision string kept


def test_roots_norms_vs_mpmath_regeneration(F300):
    """Independent construction from the defining conditions reproduces the tables (roots to the 8-digit rounding)."""
    t0 = time.time()
    roots, norms, _ = C.regenerate_roots_norms(8, 0.5, 300.0, dps=70)
    for n in range(1, 9):
        r = np.array([float(x) for x in roots[n]])
        assert np.abs(r - F300.roots[n]).max() < 6e-8            # table has 8 significant digits (roots < 10)
        assert abs(float(norms[n]) / F300.norms[n] - 1.0) < 1e-12
    assert time.time() - t0 < 20


@pytest.mark.slow
def test_roots_norms_vs_mpmath_regeneration_n12(F300):
    roots, norms, _ = C.regenerate_roots_norms(12, 0.5, 300.0, dps=100)
    for n in (10, 11, 12):
        r = np.array([float(x) for x in roots[n]])
        assert np.abs(r - F300.roots[n]).max() < 6e-8
        assert abs(float(norms[n]) / F300.norms[n] - 1.0) < 1e-10


@pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
def test_tplus_tminus_vs_tabulated(F100, n):
    dp = np.loadtxt(os.path.join(CD, "Tplus_minus", f"Tplus{n}_0.50-100.00.table"))
    dm = np.loadtxt(os.path.join(CD, "Tplus_minus", f"Tminus{n}_0.50-100.00.table"))
    Tp = F100.Tplus(dp[:, 0], 20)[n - 1]
    Tm = F100.Tminus(dm[:, 0], 20)[n - 1]
    assert np.abs(Tp - dp[:, 1]).max() / np.abs(dp[:, 1]).max() < 1e-12
    # the tabulated T_- is only accurate to ~1e-8 (n=1) .. 1.4e-5 (n=20) of its peak; ours is exact (next test)
    assert np.abs(Tm - dm[:, 1]).max() / np.abs(dm[:, 1]).max() < 3e-5


def test_tminus_exact_vs_mpmath_quad(F100):
    import mpmath as mp
    mp.mp.dps = 25
    n = 4
    r = [mp.mpf(float(x)) for x in F100.roots[n]]
    N = mp.mpf(F100.norms[n])
    Tp = lambda y: N * mp.fprod([y - ri for ri in r])
    for th in (0.7, 3.0, 20.0, 99.0):
        z = mp.log(mp.mpf(th) / mp.mpf("0.5"))
        I = mp.quad(lambda y: 4 * Tp(y) * (mp.exp(2 * (y - z)) - 3 * mp.exp(4 * (y - z))), mp.linspace(0, z, 6))
        ex = float(Tp(z) + I)
        assert abs(F100.Tminus(np.array([th]), 20)[n - 1][0] - ex) < 1e-11 * max(1, abs(ex))


def test_orthonormality_and_constraints(F300):
    G = F300.gram(20)
    assert np.abs(G - 299.5 * np.eye(20)).max() / 299.5 < 1e-6  # limited by the 8-digit roots of the table
    a, b, s = F300.constraints(20)
    assert np.abs(a / s).max() < 1e-5 and np.abs(b / s).max() < 1e-5
    assert np.all(np.sign([F300.norms[n] for n in range(1, 21)]) > 0)


def test_pure_E_mode_has_zero_B(F300):
    """For xi+/- generated from an E-mode C_ell the B_n must vanish: this tests T_- against T_+ independently."""
    l = np.logspace(-1, np.log10(5e5), 400)
    Cl = 1e-9 * (l / 100.0) ** -1.0 / (1.0 + (l / 3000.0) ** 1.5)
    from kidspipe.theory.hankel import HankelBinner
    th, xp, xm = HankelBinner(l).xi_theta(np.tile(Cl, (15, 1)))
    E, B = C.E_from_xi(th, xp[:1], xm[:1], F300, 6)
    assert np.abs(B).max() / np.abs(E).max() < 1e-4


# --------------------------------------------------------------------------------------------- W_n(l)
@pytest.mark.parametrize("l", [3.0, 200.0, 2000.0])
def test_wn_vs_adaptive_quad(F300, l):
    f = lambda th: F300.Tplus(np.array([th / C.ARCMIN]), 3)[2][0] * th * j0(l * th)
    pts = np.linspace(0.5 * C.ARCMIN, 300 * C.ARCMIN, 400)
    ref = sum(quad(f, a, b, epsabs=1e-15, epsrel=1e-12)[0] for a, b in zip(pts[:-1], pts[1:]))
    w = C.wn_of_l(np.array([l]), F300, 3)[2, 0]
    assert abs(w - ref) < 1e-9 * np.abs(C.wn_of_l(np.array([l]), F300, 3)).max() + 1e-12


# --------------------------------------------------------------------------------------------- model fixtures
@pytest.fixture(scope="module")
def model_cl():
    from kidspipe.api import build_model
    m = build_model(linear_backend="emulator")
    th = np.array(json.load(open(os.path.join(ROOT, "runs", "kv450_final", "map_best.json")))["theta"])
    ell, Cl = m.cl(th)
    return m, th, ell, Cl


def test_model_cl_hook_consistent_with_mu(model_cl):
    m, th, ell, Cl = model_cl
    from kidspipe.likelihood.data import assemble_vector
    xip, xim = m.binner.xi_binned(Cl)
    np.testing.assert_allclose(assemble_vector(xip, xim), m.mu(th), rtol=1e-12)


def test_E_from_cl_vs_E_from_xi(F300, model_cl):
    m, th, ell, Cl = model_cl
    nm = 5
    M = C.cosebis_operator(ell, F300, nm, use_cache=False)
    E = C.E_from_cl(Cl, M)
    thr, xp, xm = m.binner.xi_theta(Cl)
    Ex, Bx = C.E_from_xi(thr, xp, xm, F300, nm)
    assert E.shape == (15, nm)
    assert np.abs(E - Ex).max() / np.abs(Ex).max() < 2e-4
    assert np.abs(Bx).max() / np.abs(Ex).max() < 1e-4


# --------------------------------------------------------------------------------------------- data / likelihood
def test_fits_ordering_and_covariance():
    d = LC.CosebisData.load()
    assert d.value.shape == (300,) and d.cov.shape == (300, 300)
    assert abs(d.value[0] - (-8.60461612e-11)) < 1e-18
    from kidspipe.likelihood.data import PAIRS
    assert [PAIRS[p] for p in d.pair[::20]][:3] == [(0, 0), (1, 0), (2, 0)]
    assert np.linalg.eigvalsh(d.cov)[0] > 0
    assert d.select(5).size == 75 and d.select(20).size == 300


def test_scaled_chi2_matches_explicit_inverse():
    L = LC.CosebisLikelihood(5)
    rng = np.random.default_rng(1)
    E = L.d + rng.normal(size=75) * np.sqrt(np.diag(L.cov))
    r = L.d - E
    assert abs(L.chi2(E) - r @ np.linalg.solve(L.cov, r)) < 1e-6 * L.chi2(E)
    assert L.chi2(L.d) == 0.0 and L.chi2(np.full(75, np.nan)) == np.inf


def test_chi2_at_xi_bestfit(model_cl):
    m, th, ell, Cl = model_cl
    cm = LC.CosebisModel(m, 5)
    E = cm.mu(th)
    assert E.shape == (75,) and np.all(np.isfinite(E))
    c = {}
    for am in (True, False):
        L = LC.CosebisLikelihood(5, apply_m=am)
        c[am] = L.chi2(E)
        assert 30 < c[am] < 130                 # 75 points, parameters fitted to xi+/-; published KiDS-1000 COSEBIs: 82/75
        assert 150 < L.null_chi2() < 260
    assert abs(c[True] - c[False]) < 5.0
    post = LC.CosebisPosterior(cm, LC.CosebisLikelihood(5))
    assert np.isfinite(post.log_post(th))
