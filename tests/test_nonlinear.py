"""Tests for kidspipe.theory.nonlinear: CAMB 'mead' backend (HMcode-2016), registry, high-k extension (spec s2.6)."""
import os

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

camb = pytest.importorskip("camb")

from kidspipe.theory import nonlinear as nl  # noqa: E402
from kidspipe.theory.linear import LinearBackend, Z_GRID_REDUCED  # noqa: E402
from kidspipe.theory.nonlinear import BACKENDS, eta_baryon, extend_high_k, get_pnl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "validation")
SANITY = dict(omch2=0.10, ombh2=0.022, h=0.70, n_s=1.09, ln_1e10_A_s=3.09)


@pytest.fixture(scope="module")
def backend():
    return LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED)


@pytest.fixture(scope="module")
def lin(backend):
    return backend.compute(**SANITY)


# --------------------------------------------------------------------------------- extend_high_k
def test_extend_exact_powerlaw():
    k = np.geomspace(1e-3, 30.0, 40)
    z = np.array([0.0, 1.0, 3.0])
    slopes = np.array([-2.5, -2.0, -1.7])
    pk = 1e3 * (k[None, :] / 1.0) ** slopes[:, None]
    ke, pe = extend_high_k(k, pk)
    assert ke.size == 40 + 199 and pe.shape == (3, 239)
    assert ke[-1] == pytest.approx(500.0, rel=1e-13)
    assert np.array_equal(ke[:40], k)
    assert np.allclose(np.diff(np.log(ke[39:])), np.log(500 / 30.0) / 199, rtol=1e-12)   # uniform in ln k
    assert np.max(np.abs(pe / (1e3 * ke[None, :] ** slopes[:, None]) - 1)) < 1e-12
    assert np.array_equal(pe[:, :40], pk)


def test_extend_matches_polyfit_last3():
    rng = np.random.default_rng(1)
    k = np.geomspace(1e-4, 20.0, 60)
    pk = np.exp(rng.normal(0, 0.3, (4, 60)) - 2.3 * np.log(k)[None, :])
    ke, pe = extend_high_k(k, pk, khmax=500.0, nnew=200)
    for i in range(4):
        b, a = np.polyfit(np.log(k[-3:]), np.log(pk[i, -3:]), 1)
        xnew = np.linspace(np.log(k[-1]), np.log(500.0), 200)[1:]
        assert np.allclose(pe[i, 60:], np.exp(a + b * xnew), rtol=1e-11)
    # 1-d input
    ke1, pe1 = extend_high_k(k, pk[0])
    assert pe1.shape == (259,) and np.allclose(pe1, pe[0])
    with pytest.raises(ValueError):
        extend_high_k(k, pk, khmax=10.0)


def test_extend_spec_numbers(lin):
    """spec s2.6: 393 nodes for kmax=30 h/Mpc, slope(lin) = -2.57, slope(nl) -2.16 (z=0) ... -1.68 (z=6)."""
    pn = get_pnl(lin, 3.0, "camb")
    ke, pl = extend_high_k(lin.k_h, lin.pk_lin)
    _, pnn = extend_high_k(lin.k_h, pn)
    assert ke.size == 194 + 199 == 393
    s = lambda p, i: np.log(p[i, -1] / p[i, -2]) / np.log(ke[-1] / ke[-2])
    assert s(pl, 0) == pytest.approx(-2.57, abs=0.01) and s(pl, -1) == pytest.approx(-2.57, abs=0.01)
    assert s(pnn, 0) == pytest.approx(-2.16, abs=0.01)
    assert s(pnn, -1) == pytest.approx(-1.68, abs=0.01)
    # P_nl/P_lin at the last CAMB node: 61.6 (z=0), 15.5 (z=6)
    assert pn[0, -1] / lin.pk_lin[0, -1] == pytest.approx(61.6, rel=2e-3)
    assert pn[-1, -1] / lin.pk_lin[-1, -1] == pytest.approx(15.5, rel=5e-3)


# --------------------------------------------------------------------------------- camb backend
def test_eta_map():
    assert eta_baryon(2.0) == pytest.approx(0.74) and eta_baryon(3.13) == pytest.approx(0.6044)


def test_spec_anchor_pnl(backend):
    """VALIDATION s4.3: A=3.0, eta0=0.62: P_nl(z=0,k=1.00298)=350.03, P_lin=69.57 (linear interp in k, 150-z grid)."""
    r = LinearBackend(nthreads=4).compute(**SANITY)
    pn = get_pnl(r, 3.0, "camb")
    k = r.k_h
    i = np.searchsorted(k, 1.00298)
    w = (1.00298 - k[i - 1]) / (k[i] - k[i - 1])
    assert pn[0, i - 1] * (1 - w) + pn[0, i] * w == pytest.approx(350.03, rel=1e-4)
    assert r.pk_lin[0, i - 1] * (1 - w) + r.pk_lin[0, i] * w == pytest.approx(69.57, rel=1e-4)


def test_pnl_vs_fresh_camb_incl_rescale(backend, lin):
    """Cached/rescaled path (new A_s, n_s, A_baryon applied in place) == independent fresh CAMB 'mead' run."""
    from camb.nonlinear import Halofit
    d = dict(SANITY)
    d.update(n_s=0.95, ln_1e10_A_s=3.3)
    r = backend.compute(**d)                                       # cache hit, rescaled
    for A in (2.0, 3.13):
        pn = get_pnl(r, A, "camb")
        p = camb.CAMBparams()
        p.set_cosmology(H0=70.0, ombh2=0.022, omch2=0.10, mnu=0.06, num_massive_neutrinos=1, omk=0)
        p.InitPower.set_params(As=np.exp(3.3) * 1e-10, ns=0.95)
        p.Reion.Reionization = False
        p.WantCls = False
        p.set_matter_power(redshifts=list(Z_GRID_REDUCED[::-1]), kmax=20.0, nonlinear=True)
        p.NonLinearModel = Halofit(halofit_version="mead", HMCode_A_baryon=A,
                                   HMCode_eta_baryon=0.98 - 0.12 * A)
        camb.config.ThreadNum = 4
        c = camb.get_results(p)
        kh, z, pk = c.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
        assert np.array_equal(kh, r.k_h)
        assert np.max(np.abs(pn / pk - 1)) < 1e-8
    # interleaving with linear queries must not corrupt either result
    r2 = backend.compute(**SANITY)
    pn2 = get_pnl(r2, 3.0, "camb")
    assert get_pnl(r, 2.0, "camb")[0, -1] != pn2[0, -1]
    assert backend.compute(**d).sigma8_0 == pytest.approx(r.sigma8_0, rel=1e-14)


def test_pnl_vs_stored_pk_mead():
    """VALIDATION L4: CAMB 'mead' with A=2, eta0=0.74 reproduces data/validation/pk_mead.txt to < 1 % (rtol of repo)."""
    zf = np.loadtxt(os.path.join(DATA, "z.txt"))
    kf = np.loadtxt(os.path.join(DATA, "k_h.txt"))
    pf = np.loadtxt(os.path.join(DATA, "pk_mead.txt"))
    be = LinearBackend(nthreads=4, z_grid=zf)
    r = be.compute(omch2=0.1181838, ombh2=0.02228089, h=0.679565, n_s=0.9679701,
                   ln_1e10_A_s=np.log(2.170229e-9 * 1e10))
    pn = get_pnl(r, 2.0, "camb")
    m = (kf > r.k_h[0]) & (kf < r.k_h[-1])
    worst, ss, n = 0.0, 0.0, 0
    for iz in range(0, 100, 7):
        g = CubicSpline(np.log(r.k_h), np.log(pn[iz]))
        d = np.exp(g(np.log(kf[m]))) / pf[iz, m] - 1
        worst = max(worst, np.max(np.abs(d)))
        ss += np.sum(d ** 2)
        n += d.size
    assert worst < 1e-2                  # measured 0.64 %
    assert np.sqrt(ss / n) < 4e-3        # measured ~0.3 %
    # the stored array is NOT the eta0=0.603 variant mentioned in the repo scripts (VALIDATION s8, hazard 2)
    from camb.nonlinear import Halofit
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.9565, ombh2=0.02228089, omch2=0.1181838, mnu=0.06, num_massive_neutrinos=1, omk=0)
    p.InitPower.set_params(As=2.170229e-9, ns=0.9679701)
    p.set_matter_power(redshifts=[0.0], kmax=20.0, nonlinear=True)
    p.NonLinearModel = Halofit(halofit_version="mead", HMCode_A_baryon=2.0, HMCode_eta_baryon=0.603)
    camb.config.ThreadNum = 4
    c = camb.get_results(p)
    kh, _, p603 = c.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    g = CubicSpline(np.log(kh), np.log(p603[0]))
    mm = (kf > kh[0]) & (kf < kh[-1])
    d603 = np.exp(g(np.log(kf[mm]))) / pf[0, mm] - 1
    assert np.max(np.abs(d603)) > 5e-2


def test_pnl_physics(lin):
    pl = lin.pk_lin
    p2 = get_pnl(lin, 2.0, "camb")
    p313 = get_pnl(lin, 3.13, "camb")
    assert p2.shape == pl.shape and np.all(p2 > 0)
    # linear on large scales (HMcode small-scale corrections switched off below 0.005 h/Mpc; residual <7e-4)
    m = lin.k_h < 5e-3
    assert np.max(np.abs(p2[:, m] / pl[:, m] - 1)) < 2e-3
    # non-linear boost at small scales, bigger at low z
    assert p2[0, -1] / pl[0, -1] > 20 and p2[0, -1] / pl[0, -1] > p2[-1, -1] / pl[-1, -1]
    # A_baryon: larger A (more concentrated, DM-like) -> more power at high k (spec s2.5)
    assert np.all(p313[0, lin.k_h > 5] > p2[0, lin.k_h > 5])
    assert np.max(np.abs(p313[:, m] / p2[:, m] - 1)) < 1e-4


# --------------------------------------------------------------------------------- registry / dispatch
def test_registry_and_dispatch(lin):
    assert "camb" in BACKENDS and callable(BACKENDS["camb"])
    with pytest.raises(KeyError):
        get_pnl(lin, 2.0, "nope")
    calls = []

    def dummy(l, A, **kw):
        calls.append((A, kw))
        return l.pk_lin * 2

    BACKENDS["dummy"] = dummy
    try:
        out = get_pnl(lin, 2.5, "dummy", foo=1)
        assert calls == [(2.5, {"foo": 1})] and np.allclose(out, 2 * lin.pk_lin)
    finally:
        del BACKENDS["dummy"]
    # params dict path builds a LinearResult via the shared default backend (uses CAMB; reduced check on shape)
    assert nl.BACKENDS["camb"] is nl._pnl_camb


def test_camb_backend_needs_state(lin):
    import dataclasses
    bare = dataclasses.replace(lin, _state=None)
    with pytest.raises(ValueError):
        get_pnl(bare, 2.0, "camb")
