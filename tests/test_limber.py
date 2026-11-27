"""Tests for kidspipe.theory.limber (spec s4-s6).

Uses a self-contained toy P_nl (BBKS-like shape times a k-independent growth factor) and a
toy flat-LCDM distance table, plus one CAMB cross-check against the stored spec anchors.
"""
import json
import time
from pathlib import Path

import numpy as np
from kidspipe._compat import trapezoid
import pytest
from scipy import integrate
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from kidspipe.theory.limber import (C1_RHO_CRIT, C_LIGHT, Limber, cl_shear, coarse_ell,
                                    default_ell, lensing_efficiency, lensing_prefactor,
                                    nla_growth_factor)
from kidspipe.theory.nz import load_nz

OM = 0.3
S = 1.5 * OM * (100.0 / C_LIGHT) ** 2


# ----------------------------------------------------------------------------- toy cosmology
def toy_growth(z):
    return (1.0 + z) ** -0.7


def toy_p0(k):
    q = k / 0.2
    T = np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3
                                             + (6.71 * q) ** 4) ** -0.25
    return 2.0e4 * k ** 0.96 * T**2


@pytest.fixture(scope="module")
def toy():
    z_tab = np.linspace(0, 6, 6000)
    E = np.sqrt(OM * (1 + z_tab) ** 3 + 1 - OM)
    chi_tab = np.concatenate([[0], integrate.cumulative_trapezoid(1 / E, z_tab)]) * (C_LIGHT / 100.0)
    z_pk = np.concatenate([np.linspace(0, 2, 100, endpoint=False), np.linspace(2, 6, 50)])
    k_h = np.logspace(-5, np.log10(500.0), 350)
    pk = toy_growth(z_pk)[:, None] ** 2 * toy_p0(k_h)[None, :]
    return dict(z_tab=z_tab, chi_tab=chi_tab, z_pk=z_pk, k_h=k_h, pk=pk)


@pytest.fixture(scope="module")
def nz():
    return load_nz()


# ----------------------------------------------------------------------------- basics
def test_constants_and_prefactor():
    assert lensing_prefactor(OM) == pytest.approx(1.5 * OM * (100 / 299792.458) ** 2)
    assert C_LIGHT == 299792.458 and C1_RHO_CRIT == 0.0138731
    ell = default_ell()
    assert ell.size == 400 and ell[0] == pytest.approx(0.1) and ell[-1] == pytest.approx(5e5)
    np.testing.assert_allclose(ell[1] / ell[0], 1.0394, atol=1e-4)
    assert coarse_ell().size == 120


def test_nla_growth_factor_uses_smallest_k(toy):
    D = nla_growth_factor(toy["z_pk"], toy["pk"])
    np.testing.assert_allclose(D, toy_growth(toy["z_pk"]) / toy_growth(0.0), rtol=1e-13)
    with pytest.raises(ValueError):
        nla_growth_factor(toy["z_pk"] + 0.1, toy["pk"])


# ----------------------------------------------------------------------------- (e) W vs quad
def test_W_analytic_polynomial_n():
    """chi = R z, n = 6 z (zs - z)/zs^3  ->  W = chi (1+z) ((zs-z)/zs)^3 exactly."""
    R, zs = 3000.0, 1.3
    z = np.linspace(0, zs, 3001)
    n = 6 * z * (zs - z) / zs**3
    W = lensing_efficiency(z, R * z, n)
    ref = R * z * (1 + z) * ((zs - z) / zs) ** 3
    np.testing.assert_allclose(W, ref, rtol=0, atol=2e-6 * ref.max())
    assert W[0] == 0.0 and W[-1] == 0.0


def test_W_vs_scipy_quad(toy, nz):
    """W_i(chi_m) = (chi/a) int_z^zmax n(z')[1 - chi/chi(z')] dz' by adaptive quadrature."""
    spl = CubicSpline(toy["z_tab"], toy["chi_tab"])
    zf = np.linspace(0, 6, 8001)  # kernel grid, 4x oversampled w.r.t. the 2001 nodes below
    chif = spl(zf)
    z = zf[::4]
    chi = chif[::4]
    shifted = nz.shifted([0.02, -0.01, 0.015, 0.0, -0.03])
    for ib in (0, 2, 4):
        n_of_z = lambda zz, row=shifted[ib]: np.interp(zz, nz.z, row, left=0.0, right=0.0)
        W = lensing_efficiency(zf, chif, np.interp(zf, nz.z, shifted[ib], left=0.0, right=0.0))[::4]
        for m in (60, 200, 420, 700, 1000):
            zm = z[m]
            f = lambda zz: n_of_z(zz) * (1.0 - chi[m] / spl(zz))
            knots = nz.z[nz.z > zm]
            val = integrate.quad(f, zm, nz.z[-1], points=knots[::4], limit=2000, epsabs=0, epsrel=1e-11)[0]
            ref = chi[m] * (1 + zm) * val
            assert W[m] == pytest.approx(ref, rel=5e-5, abs=1e-9)


def test_W_boundary_properties(toy, nz):
    lim = Limber(ell=coarse_ell(20)).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    W, nchi = lim._kernels(nz.z, nz.n)
    assert np.all(W[:, 0] == 0.0) and np.all(W >= -1e-12)
    # int n dchi = int n dz = 1 ; W(chi_max) ~ 0
    np.testing.assert_allclose(trapezoid(nchi, lim.chi, axis=1), 1.0, atol=2e-4)
    assert np.all(W[:, -1] < 1e-8 * W.max())


# ----------------------------------------------------------------------------- (a) analytic
def tent(z_nz, zs, half):
    n = np.clip(1 - np.abs(z_nz - zs) / half, 0, None)
    return n / trapezoid(n, z_nz)


def test_flat_sky_power_law_delta_n():
    """P = A k^-1.5 (1+z)^-1, chi = R z, delta-like n: the Limber integral is a 1D quadrature.

    Auto spectrum: C = s^2 int_0^chi_s dchi (1+z)^2 (1-chi/chi_s)^2 P(nu/chi, z)  (chi^2/chi^2 cancels).
    Cross spectrum with GI (A_IA != 0): extra term s F(z_a) W_b(chi_a) P(nu/chi_a, z_a)/chi_a^2.
    """
    R, A, p = 3000.0, 50.0, 1.5
    z_tab = np.linspace(0, 6, 6001)
    z_pk = np.linspace(0, 6, 301)
    k_h = np.logspace(-6, 4, 400)
    pk = A * k_h[None, :] ** -p / (1 + z_pk)[:, None]
    z_nz = np.linspace(0, 2, 4001)
    za, zb, half = 0.5, 1.2, 0.004
    nzs = np.stack([tent(z_nz, za, half), tent(z_nz, zb, half)])
    ell = np.array([5.0, 30.0, 200.0, 1500.0])
    A_IA = 1.7
    lim = Limber(ell=ell, n_nodes=4000).set_power(R * z_tab, z_tab, OM, z_pk, k_h, pk, z_max=2.0)
    C0 = lim.cl(nzs, 0.0, z_nz=z_nz)
    C1 = lim.cl(nzs, A_IA, z_nz=z_nz)
    P = lambda k, z: A * k**-p / (1 + z)
    chi_a, chi_b = R * za, R * zb
    D = lambda z: (1 + z) ** -0.5  # sqrt(P(k_small,z)/P(k_small,0))
    F = lambda z: -A_IA * C1_RHO_CRIT * OM / D(z)
    for il, l in enumerate(ell):
        nu = l + 0.5
        aa = S**2 * integrate.quad(lambda c: (1 + c / R) ** 2 * (1 - c / chi_a) ** 2 * P(nu / c, c / R), 0, chi_a,
                                   epsabs=0, epsrel=1e-10)[0]
        bb = S**2 * integrate.quad(lambda c: (1 + c / R) ** 2 * (1 - c / chi_b) ** 2 * P(nu / c, c / R), 0, chi_b,
                                   epsabs=0, epsrel=1e-10)[0]
        ab = S**2 * integrate.quad(lambda c: (1 + c / R) ** 2 * (1 - c / chi_a) * (1 - c / chi_b) * P(nu / c, c / R),
                                   0, chi_a, epsabs=0, epsrel=1e-10)[0]
        assert C0[0, 0, il] == pytest.approx(aa, rel=1e-3)
        assert C0[1, 1, il] == pytest.approx(bb, rel=1e-3)
        assert C0[0, 1, il] == pytest.approx(ab, rel=1e-3)
        assert C0[1, 0, il] == C0[0, 1, il]
        # GI: source a is in front of b, so only (n_a x W_b) survives; II (n_a n_b) = 0
        Wb_at_a = chi_a * (1 + za) * (1 - chi_a / chi_b)
        gi = S * F(za) * Wb_at_a * P(nu / chi_a, za) / chi_a**2
        assert C1[0, 1, il] == pytest.approx(ab + gi, rel=2e-3)
        assert (gi / ab) < 0  # A_IA > 0 -> negative GI


def test_zero_outside_table(toy, nz):
    """kmax = 1 h/Mpc: for l = 1e5 every k = nu/chi > chi_max^-1... >> 1 so C == 0 exactly (spec D13)."""
    m = toy["k_h"] <= 1.0
    lim = Limber(ell=np.array([1e5, 3e5])).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"],
                                                     toy["k_h"][m], toy["pk"][:, m])
    assert np.all(lim.cl(nz.n, 0.0, z_nz=nz.z) == 0.0)
    # and nonzero at low l
    lim2 = Limber(ell=np.array([30.0, 1e5])).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"],
                                                       toy["k_h"][m], toy["pk"][:, m])
    C = lim2.cl(nz.n, 0.0, z_nz=nz.z)
    assert np.all(C[..., 1] == 0.0) and np.all(C[..., 0] > 0)


# ----------------------------------------------------------------------------- (b) convergence
@pytest.mark.slow
def test_node_convergence_vs_12000(toy, nz):
    """Spec s15.4: 2000 trapezoid nodes agree with a 12000-node reference to ~1.5e-4 (10<l<2e4)."""
    ell = np.logspace(np.log10(10.0), np.log10(2e4), 60)
    sh = nz.shifted([0.01, -0.02, 0.01, 0.0, 0.0])

    def run(nn):
        return cl_shear(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"], nz.z, sh, 0.5, ell,
                        n_nodes=nn)

    ref = run(12000)
    errs = {nn: np.max(np.abs(run(nn) / ref - 1)) for nn in (500, 1000, 2000, 4000)}
    print("max rel err vs 12000 nodes:", errs)
    assert errs[500] > errs[1000] > errs[2000] > errs[4000]
    assert errs[500] < 1e-3 and errs[1000] < 3e-4
    assert errs[2000] < 1e-4  # spec s15.4 quotes 1.5e-4 for its (E) implementation
    assert errs[4000] < 5e-5


def test_node_convergence_quick(toy, nz):
    ell = np.logspace(1, np.log10(2e4), 25)
    lim = lambda nn: cl_shear(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"], nz.z, nz.n, 0.0,
                              ell, n_nodes=nn)
    ref = lim(8000)
    assert np.max(np.abs(lim(2000) / ref - 1)) < 3e-4


# ----------------------------------------------------------------------------- (c) folded kernel
def _setup(toy, nz, ell, A_IA, interp_order=3, n_nodes=2000, dz=(0.01, 0, -0.01, 0.02, 0)):
    sh = nz.shifted(dz)
    lim = Limber(ell=ell, n_nodes=n_nodes, interp_order=interp_order)
    lim.set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    return lim, sh


def test_fold_equals_sum_of_GG_GI_IG_II(toy, nz):
    """Same discretisation: C = GG + GI + IG + II with separately contracted kernels (machine precision)."""
    ell = coarse_ell(40)
    A_IA = 1.3
    lim, sh = _setup(toy, nz, ell, A_IA)
    W, nchi = lim._kernels(nz.z, sh)
    F = lim.nla_factor(A_IA)
    meas = lim.w * lim.dchidz * lim.inv_chi**2
    Qt = lim.Qt
    sW, In = lim.s * W, F[None, :] * nchi
    tot = lim.cl(sh, A_IA, z_nz=nz.z)
    for i in range(5):
        for j in range(5):
            gg = (sW[i] * sW[j] * meas) @ Qt
            gi = (sW[i] * In[j] * meas) @ Qt
            ig = (In[i] * sW[j] * meas) @ Qt
            ii = (In[i] * In[j] * meas) @ Qt
            np.testing.assert_allclose(tot[i, j], gg + gi + ig + ii, rtol=1e-11)
    # P_GI = F P, P_II = F^2 P structure: C(A) is quadratic in A_IA
    c = lambda a: lim.cl(sh, a, z_nz=nz.z)
    c0, c1, cm1, c2 = c(0.0), c(1.0), c(-1.0), c(2.0)
    np.testing.assert_allclose(c2 - 2 * c1 + c0, c1 + cm1 - 2 * c0, rtol=1e-7, atol=1e-30 + 1e-10 * np.abs(c1).max())
    assert np.all(np.einsum("iil->il", c1 + cm1 - 2 * c0) > 0)  # II auto spectra are positive
    assert np.all(np.einsum("iil->il", c1) < np.einsum("iil->il", c0))  # A_IA>0: GI<0 dominates in autos


def test_fold_vs_independent_separate_integration(toy, nz):
    """Independent reference: separate GG, GI, IG, II integrals on another z grid, W by direct O(N^2)
    quadrature of its definition, ln P by scipy RegularGridInterpolator (bilinear = interp_order 1)."""
    ell = np.array([20.0, 100.0, 700.0, 4000.0, 15000.0])
    A_IA = -1.1
    dz = (0.02, 0.0, -0.01, 0.0, 0.03)
    lim, sh = _setup(toy, nz, ell, A_IA, interp_order=1, dz=dz)
    folded = lim.cl(sh, A_IA, z_nz=nz.z)

    N = 2500
    z = np.linspace(0, 5.975, N)
    spl = CubicSpline(toy["z_tab"], toy["chi_tab"])
    chi, g = spl(z), spl(z, 1)
    n = np.stack([np.interp(z, nz.z, r) for r in sh])
    # W_i(chi_m) = chi_m (1+z_m) trapz_{z' >= z_m} n(z') (1 - chi_m/chi(z'))
    W = np.zeros((5, N))
    for m in range(1, N):
        f = n[:, m:] * (1.0 - chi[m] / chi[m:])[None, :]
        W[:, m] = chi[m] * (1 + z[m]) * trapezoid(f, z[m:], axis=1)
    nc = n / g
    D = np.sqrt(toy["pk"][:, 0] / toy["pk"][0, 0])
    F = -A_IA * C1_RHO_CRIT * OM / np.interp(z, toy["z_pk"], D)
    rgi = RegularGridInterpolator((toy["z_pk"], np.log(toy["k_h"])), np.log(toy["pk"]), bounds_error=False,
                                  fill_value=-np.inf)
    ref = np.zeros_like(folded)
    wz = np.full(N, z[1] - z[0]); wz[[0, -1]] *= 0.5
    for il, l in enumerate(ell):
        P = np.zeros(N)
        P[1:] = np.exp(rgi(np.c_[z[1:], np.log((l + 0.5) / chi[1:])]))
        meas = wz * g * P
        meas[1:] /= chi[1:] ** 2
        meas[0] = 0
        for i in range(5):
            for j in range(5):
                GG = S * S * np.sum(W[i] * W[j] * meas)
                GI = S * np.sum(F * W[i] * nc[j] * meas)
                IG = S * np.sum(F * nc[i] * W[j] * meas)
                II = np.sum(F * F * nc[i] * nc[j] * meas)
                ref[i, j, il] = GG + GI + IG + II
    np.testing.assert_allclose(folded, ref, rtol=6e-4, atol=0)


def test_symmetry_and_cl_wrapper(toy, nz):
    ell = coarse_ell(30)
    sh = nz.shifted([0.01, 0.0, 0.0, -0.02, 0.0])
    C = cl_shear(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"], nz.z, sh, 0.8, ell)
    assert C.shape == (5, 5, 30)
    np.testing.assert_array_equal(C, C.transpose(1, 0, 2))
    lim = Limber(ell=ell).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    np.testing.assert_array_equal(C, lim.cl(sh, 0.8, z_nz=nz.z))
    # redshift ordering: higher source bins are more strongly lensed (A_IA = 0, no shifts)
    C0 = lim.cl(nz.n, 0.0, z_nz=nz.z)
    assert np.all(np.diff(C0[np.arange(5), np.arange(5), 10]) > 0)
    assert np.all(C0[...] > 0)


def test_interp_order_and_kcap_variant(toy, nz):
    ell = np.logspace(0, np.log10(3e4), 40)
    sh = nz.shifted(np.zeros(5))
    args = (toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"], nz.z, sh, 0.5, ell)
    fast = cl_shear(*args)
    lin = Limber(ell=ell, interp_order=1).set_power(*args[:6]).cl(sh, 0.5, z_nz=nz.z)
    assert np.max(np.abs(lin / fast - 1)) < 1e-3
    kc = cl_shear(*args, fidelity="kcap")
    assert np.max(np.abs(kc / fast - 1)) < 3e-3
    assert np.max(np.abs(kc / fast - 1)) > 0  # genuinely a different code path
    with pytest.raises(ValueError):
        Limber(fidelity="bogus")
    with pytest.raises(ValueError):
        Limber(ell=np.array([5.0, 3.0]))


def test_ia_constant_pieces(toy):
    lim = Limber(ell=coarse_ell(5)).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    F = lim.nla_factor(2.0)
    np.testing.assert_allclose(F, -2.0 * 0.0138731 * OM * (1 + lim.z) ** 0.7, rtol=1e-6)


# ----------------------------------------------------------------------------- performance
@pytest.mark.timing
def test_warm_update_is_fast(toy, nz):
    lim = Limber().set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    lim.set_nz_grid(nz.z)
    lim.cl(nz.n, 0.0)
    t0 = time.perf_counter()
    n = 20
    for i in range(n):
        lim.cl(nz.shifted(np.full(5, 0.002 * i)), 0.1 * i)
    dt = (time.perf_counter() - t0) / n
    print(f"warm cl update (dz + A_IA change): {dt * 1e3:.2f} ms")
    assert dt < 0.05
    # A_IA-only change reuses the cached W
    t0 = time.perf_counter()
    for i in range(n):
        lim.cl(nz.n, 0.1 * i)
    print(f"warm A_IA-only update: {(time.perf_counter() - t0) / n * 1e3:.2f} ms")


# ----------------------------------------------------------------------------- CAMB anchors
@pytest.mark.slow
def test_camb_takahashi_vs_spec_anchors():
    camb = pytest.importorskip("camb")
    h = 0.7
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.1, mnu=0.06, num_massive_neutrinos=1)
    pars.InitPower.set_params(As=np.exp(3.0) * 1e-10, ns=0.96)
    zs = np.concatenate([np.linspace(0, 2, 100, endpoint=False), np.linspace(2, 6, 50)])
    pars.set_matter_power(redshifts=list(zs[::-1]), kmax=20.0, nonlinear=True)
    pars.NonLinearModel.set_params(halofit_version="takahashi")
    pars.NonLinear = camb.model.NonLinear_both
    res = camb.get_results(pars)
    kh, zz, pk = res.get_matter_power_spectrum(minkh=7.1e-5, maxkh=28.5, npoints=250)
    # spec s2.6 high-k extension: log-linear fit on the last 3 nodes, to 500 h/Mpc
    x = np.log(kh)
    xn = np.linspace(x[-1], np.log(500.0), 200)[1:]
    b = np.array([np.polyfit(x[-3:], np.log(p[-3:]), 1) for p in pk])
    pke = np.hstack([pk, np.exp(b[:, 1:2] + b[:, 0:1] * xn[None, :])])
    khe = np.concatenate([kh, np.exp(xn)])
    z_tab = np.linspace(0, 6, 6000)
    chi_tab = res.comoving_radial_distance(z_tab) * h
    anchors = json.load(open(Path(__file__).resolve().parents[1] / "data/validation/anchors_test_cosmology.json"))
    a = anchors["takahashi"]
    assert chi_tab[np.argmin(abs(z_tab - 1.0))] == pytest.approx(a["chi_z1_Mpc_over_h"], rel=2e-3)
    nz = load_nz()
    ell = np.array([10.0, 100.0, 1000.0, 10000.0])
    C = cl_shear(chi_tab, z_tab, pars.omegam, zz, khe, pke, nz.z, nz.n, 0.0, ell)
    worst = 0.0
    for (i, j) in [(1, 1), (3, 3), (5, 5), (5, 1)]:
        for il, l in enumerate(ell):
            ref = a[f"Cl_{i}_{j}"][str(int(l))]
            r = C[i - 1, j - 1, il] / ref - 1
            worst = max(worst, abs(r))
    print("worst rel deviation from takahashi anchors:", worst)
    assert worst < 2e-3
    # node convergence with a realistic (CAMB Takahashi) P_nl, spec s15.4: 2000 nodes -> 1.5e-4
    ell2 = np.logspace(1, np.log10(2e4), 40)
    c = lambda nn: cl_shear(chi_tab, z_tab, pars.omegam, zz, khe, pke, nz.z, nz.n, 0.0, ell2, n_nodes=nn)
    ref = c(12000)
    assert np.max(np.abs(c(2000) / ref - 1)) < 1.5e-4
    assert np.max(np.abs(c(1000) / ref - 1)) < 6e-4


# ----------------------------------------------------------------------------- robustness / state
def test_input_validation(toy, nz):
    a = (toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    lim = Limber(ell=np.array([10.0, 100.0]))
    with pytest.raises(ValueError):  # descending k would silently give C == 0
        lim.set_power(a[0], a[1], OM, a[3], a[4][::-1], a[5][:, ::-1])
    bad = a[5].copy(); bad[3, 4] = np.nan
    with pytest.raises(ValueError):
        lim.set_power(a[0], a[1], OM, a[3], a[4], bad)
    bad = a[5].copy(); bad[3, 4] = -1.0
    with pytest.raises(ValueError):
        lim.set_power(a[0], a[1], OM, a[3], a[4], bad)
    with pytest.raises(ValueError):  # z_max beyond the distance table would extrapolate chi
        lim.set_power(*a, z_max=7.0)
    with pytest.raises(ValueError):
        lim.set_power(a[0], a[1][::-1], OM, a[3], a[4], a[5])
    with pytest.raises(ValueError):
        lim.set_power(a[0], a[1], OM, a[3] + 0.1, a[4], a[5])  # z_pk must start at 0 (spec s5)
    with pytest.raises(ValueError):
        Limber(ell=np.array([-1.0, 5.0]))  # nu <= 0 gives NaN
    with pytest.raises(ValueError):
        Limber(ell=np.array([1.0, np.nan]))
    with pytest.raises(RuntimeError):
        Limber(ell=np.array([10.0])).cl(nz.n, 0.0, z_nz=nz.z)  # set_power not called


def test_kernel_cache_is_not_stale_and_results_deterministic(toy, nz):
    ell = coarse_ell(25)
    lim = Limber(ell=ell).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    shA, shB = nz.shifted(np.full(5, 0.03)), nz.shifted(np.full(5, -0.03))
    cA = lim.cl(shA, 0.7, z_nz=nz.z)
    cB = lim.cl(shB, 0.7, z_nz=nz.z)
    cA2 = lim.cl(shA, 0.7, z_nz=nz.z)  # back to A after B: must not return B's cached kernels
    assert not np.allclose(cA, cB, rtol=1e-3, atol=0)
    np.testing.assert_array_equal(cA, cA2)
    fresh = Limber(ell=ell).set_power(toy["chi_tab"], toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    np.testing.assert_array_equal(cA, fresh.cl(shA, 0.7, z_nz=nz.z))
    # in-place mutation of the n(z) array after caching must invalidate the cache
    mut = shA.copy()
    lim.cl(mut, 0.0, z_nz=nz.z)
    mut[2] = shB[2]
    np.testing.assert_array_equal(lim.cl(mut, 0.0, z_nz=nz.z), fresh.cl(mut, 0.0, z_nz=nz.z))
    # set_power again (new cosmology) must also reset the kernel cache (chi(z) changes)
    toy2_chi = toy["chi_tab"] * 1.05
    lim.set_power(toy2_chi, toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    f2 = Limber(ell=ell).set_power(toy2_chi, toy["z_tab"], OM, toy["z_pk"], toy["k_h"], toy["pk"])
    np.testing.assert_array_equal(lim.cl(shA, 0.7, z_nz=nz.z), f2.cl(shA, 0.7, z_nz=nz.z))
    # s scales as Omega_m: C_GG ~ Omega_m^2 at fixed P and distances
    lim3 = Limber(ell=ell).set_power(toy["chi_tab"], toy["z_tab"], 2 * OM, toy["z_pk"], toy["k_h"], toy["pk"])
    np.testing.assert_allclose(lim3.cl(shA, 0.0, z_nz=nz.z), 4 * fresh.cl(shA, 0.0, z_nz=nz.z), rtol=1e-12)


# ----------------------------------------------------------------------------- CAMB: IA + shifts, independent W
@pytest.mark.slow
def test_camb_with_ia_and_photoz_shift_vs_brute_force():
    """Realistic k-dependent P_nl (so D(z) from the smallest k matters), A_IA != 0 and dz != 0.

    Reference: Gauss-Legendre per n(z) segment (kinks at the nodes), chi straight from CAMB, W by direct
    quadrature of its definition at every outer node, bicubic spline of ln P(z, ln k)."""
    camb = pytest.importorskip("camb")
    from numpy.polynomial.legendre import leggauss
    from scipy.interpolate import RectBivariateSpline
    h = 0.7
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
    Om = pars.omegam
    z_tab = np.linspace(0, 6, 6000)
    chi_tab = res.comoving_radial_distance(z_tab) * h
    nzd = load_nz()
    sh = nzd.shifted([0.03, -0.02, 0.01, 0.0, -0.04])
    ell = np.array([10.0, 200.0, 3000.0])
    A = 1.5
    C = cl_shear(chi_tab, z_tab, Om, zz, khe, pke, nzd.z, sh, A, ell)

    xg, wg = leggauss(8)
    edges = np.arange(0, 5.975 + 1e-9, 0.025)  # the n(z) nodes (kinks) are segment boundaries

    def nodes(e):
        return (((e[:-1, None] + e[1:, None]) / 2 + (e[1:, None] - e[:-1, None]) / 2 * xg[None, :]).ravel(),
                ((e[1:, None] - e[:-1, None]) / 2 * wg[None, :]).ravel())

    # outer nodes: 8-point Gauss-Legendre per segment, restricted to z < 5.9
    zo, wo = nodes(edges)
    ok = zo < 5.9
    zo, wo = zo[ok], wo[ok]
    chio = res.comoving_radial_distance(zo) * h
    # W(z_m) = chi_m (1+z_m) [ int_{z_m}^{edge_up} + sum of full segments above ] n (1 - chi_m/chi') dz'
    # full segments: sum_seg (A_seg - chi_m B_seg) via reverse cumulative sums; partial one by its own GL rule
    zseg, wseg = nodes(edges)
    cseg = res.comoving_radial_distance(zseg) * h
    seg_idx = np.repeat(np.arange(edges.size - 1), 8)
    seg_up = edges[1:]
    iseg_of = np.searchsorted(edges, zo, side="right") - 1  # segment containing zo
    ez = np.concatenate([zo[:, None], seg_up[iseg_of][:, None]], axis=1)  # partial interval [z_m, upper edge]
    xp = (ez[:, :1] + ez[:, 1:]) / 2 + (ez[:, 1:] - ez[:, :1]) / 2 * xg[None, :]
    wp = (ez[:, 1:] - ez[:, :1]) / 2 * wg[None, :]
    cp = res.comoving_radial_distance(xp.ravel()).reshape(xp.shape) * h
    Wo = np.empty((5, zo.size))
    for ib, row in enumerate(sh):
        nseg = np.interp(zseg, nzd.z, row)
        A_seg = np.bincount(seg_idx, wseg * nseg, minlength=edges.size - 1)
        B_seg = np.bincount(seg_idx, wseg * nseg / cseg, minlength=edges.size - 1)
        A_above = np.concatenate([np.cumsum(A_seg[::-1])[::-1], [0.0]])[iseg_of + 1]
        B_above = np.concatenate([np.cumsum(B_seg[::-1])[::-1], [0.0]])[iseg_of + 1]
        npart = np.interp(xp, nzd.z, row)
        A_p = np.sum(wp * npart, axis=1)
        B_p = np.sum(wp * npart / cp, axis=1)
        Wo[ib] = chio * (1 + zo) * ((A_p + A_above) - chio * (B_p + B_above))
    chio = res.comoving_radial_distance(zo) * h
    dchidz = (res.comoving_radial_distance(zo + 1e-5) - res.comoving_radial_distance(zo - 1e-5)) / 2e-5 * h
    rb = RectBivariateSpline(zz, np.log(khe), np.log(pke), kx=3, ky=3)
    Dz = CubicSpline(zz, np.sqrt(pke[:, 0] / pke[0, 0]))(zo)
    s = 1.5 * Om * (100 / C_LIGHT) ** 2
    q = s * Wo + (-A * C1_RHO_CRIT * Om / Dz)[None, :] * (np.stack([np.interp(zo, nzd.z, r) for r in sh]) / dchidz)
    worst = 0.0
    for il, l in enumerate(ell):
        kk = (l + 0.5) / chio
        inside = (kk >= khe[0]) & (kk <= khe[-1])
        P = np.where(inside, np.exp(rb.ev(zo, np.log(np.where(inside, kk, 1.0)))), 0.0)
        mu = wo * dchidz * P / chio**2
        ref = np.einsum("im,jm,m->ij", q, q, mu)
        worst = max(worst, np.max(np.abs(C[..., il] / ref - 1)))
    print("CAMB IA+shift vs brute force:", worst)
    assert worst < 5e-4


# ----------------------------------------------------------------------------- _fill_q look-up kernel (prompt 07)
from numba import njit  # noqa: E402

from kidspipe.theory import limber as _lim  # noqa: E402


@njit(cache=False)
def _fill_q_reference(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp, out, vraw, scale):
    """The pre-prompt-07 kernel, verbatim apart from the serial loop: per-node row, divisions in the weights.

    Also returns the unclipped interpolant ``vraw`` and ``scale = sum_i |w_i r_i|``, the magnitude that bounds the
    rounding error of evaluating ``v = sum_i w_i r_i`` in any order (0 where the result is identically 0)."""
    nm = z_nodes.size
    nl = ln_nu.size
    nk = lnk.size
    nzp = z_pk.size
    for m in range(nm):
        z = z_nodes[m]
        if (not np.isfinite(lnchi[m])) or z < z_pk[0] or z > z_pk[nzp - 1]:
            for l in range(nl):
                out[m, l] = 0.0
                vraw[m, l] = 0.0
                scale[m, l] = 0.0
            continue
        iz = np.searchsorted(z_pk, z) - 1
        if iz < 0:
            iz = 0
        if iz > nzp - 2:
            iz = nzp - 2
        t = (z - z_pk[iz]) / (z_pk[iz + 1] - z_pk[iz])
        row = np.empty(nk)
        for j in range(nk):
            row[j] = (1.0 - t) * tab[iz, j] + t * tab[iz + 1, j]
        j = 0
        for l in range(nl):
            x = ln_nu[l] - lnchi[m]
            if x < lnk[0] or x > lnk[nk - 1]:
                out[m, l] = 0.0
                vraw[m, l] = 0.0
                scale[m, l] = 0.0
                continue
            while j < nk - 2 and lnk[j + 1] <= x:
                j += 1
            if order == 3 and nk >= 4:
                j0 = j - 1
                if j0 < 0:
                    j0 = 0
                if j0 > nk - 4:
                    j0 = nk - 4
                x0 = lnk[j0]
                x1 = lnk[j0 + 1]
                x2 = lnk[j0 + 2]
                x3 = lnk[j0 + 3]
                w0 = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
                w1 = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
                w2 = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
                w3 = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2))
                v = w0 * row[j0] + w1 * row[j0 + 1] + w2 * row[j0 + 2] + w3 * row[j0 + 3]
                sc = abs(w0 * row[j0]) + abs(w1 * row[j0 + 1]) + abs(w2 * row[j0 + 2]) + abs(w3 * row[j0 + 3])
            else:
                v = row[j] + (row[j + 1] - row[j]) * (x - lnk[j]) / (lnk[j + 1] - lnk[j])
                f = (x - lnk[j]) / (lnk[j + 1] - lnk[j])
                sc = abs(row[j]) + abs(row[j + 1]) * abs(f) + abs(row[j]) * abs(f)
            vraw[m, l] = v
            scale[m, l] = sc
            if use_exp:
                out[m, l] = np.exp(v)
            else:
                out[m, l] = v if v > 0.0 else 0.0


def _reference(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp):
    shp = (z_nodes.size, ln_nu.size)
    out, vraw, scale = np.empty(shp), np.empty(shp), np.empty(shp)
    _fill_q_reference(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp, out, vraw, scale)
    return out, vraw, scale


def _fq_case(lnk, nm=301, nl=97, z_lo=0.05, seed=0):
    """A look-up problem whose x = ln nu - ln chi range overhangs the k table on both sides, hits table nodes
    exactly (chi = 1 at one node, ln nu containing ln k nodes), has a chi = 0 node and nodes outside z_pk."""
    rng = np.random.default_rng(seed)
    z_pk = np.concatenate([[z_lo], np.sort(rng.uniform(z_lo, 3.0, 28)), [3.0]])
    z_pk = np.unique(z_pk)
    k = np.exp(lnk)
    tab = np.log((1.0 + z_pk)[:, None] ** -1.4 * 2e4 * k ** 0.9 / (1.0 + (k / 0.02) ** 2.6) + 1e-3 * k ** -0.5)
    z_nodes = np.linspace(0.0, 3.3, nm)                         # [0, z_lo) and (3, 3.3] lie outside z_pk
    chi = 3000.0 * z_nodes / (1.0 + 0.4 * z_nodes)
    chi[nm // 2] = 1.0                                         # ln chi = 0: x == ln nu exactly
    lnchi = np.where(chi > 0, np.log(np.where(chi > 0, chi, 1.0)), np.nan)   # chi(0) = 0 -> NaN row
    c = np.nanmedian(lnchi)                                    # typical ln chi: most x land inside the table
    ln_nu = np.sort(np.concatenate([np.linspace(lnk[0] - 1.0 + c, lnk[-1] + 1.0 + c, nl - 6),
                                    lnk[np.unique([0, 1, min(2, lnk.size - 1), lnk.size // 2, lnk.size - 2, lnk.size - 1])]]))
    return z_pk, tab, z_nodes, lnchi, ln_nu


def _assert_close_fq(new, ref, rtol=1e-13):
    assert np.array_equal(new == 0.0, ref == 0.0), "zero masks differ"
    rel = np.abs(new - ref) / np.maximum(np.abs(ref), 1e-30)
    assert rel.max() < rtol, rel.max()


def _assert_close_fq_cond(new, ref, vraw, scale, use_exp, rtol=1e-13):
    """``use_exp``: |dQ/Q| < rtol.  P interpolated in value (kcap): |dQ| < rtol sum_i |w_i r_i| -- on coarse grids
    v = sum_i w_i r_i cancels and a 1-ulp change of a weight is amplified by sum|w r| / |v| in relative terms (the
    former kernel had rounding errors of the same size) -- and identical zero masks wherever the sign of the
    unclipped interpolant is resolved, |v| > rtol sum|w r|."""
    if use_exp:
        _assert_close_fq(new, ref, rtol)
        return
    assert np.all(np.abs(new - ref) <= rtol * scale), np.max(np.abs(new - ref) / np.maximum(scale, 1e-300))
    resolved = np.abs(vraw) > rtol * scale
    assert np.array_equal((new == 0.0)[resolved], (ref == 0.0)[resolved]), "zero masks differ"
    assert np.all(new[~resolved] <= rtol * scale[~resolved])


_GRIDS = {
    "nonuniform": lambda: np.log(np.sort(np.concatenate([np.logspace(-4, 1, 37), np.logspace(1.02, 2.7, 25),
                                                         [3e-3, 3.1e-3, 0.5]]))),
    "uniform": lambda: np.log(np.exp(np.linspace(np.log(1e-4), np.log(500.0), 120))),
    "nk4": lambda: np.log(np.array([1e-3, 1e-2, 0.3, 30.0])),
    "nk5": lambda: np.log(np.array([1e-3, 4e-3, 0.1, 3.0, 100.0])),
    "nk3_linear_fallback": lambda: np.log(np.array([1e-3, 0.1, 50.0])),
    "nk2_linear_fallback": lambda: np.log(np.array([1e-3, 50.0])),
}


@pytest.mark.parametrize("grid", list(_GRIDS))
@pytest.mark.parametrize("order", [3, 1])
@pytest.mark.parametrize("use_exp", [True, False])
def test_fill_q_matches_reference(grid, order, use_exp):
    """New pre-indexed kernels == the former kernel to < 1e-13 relative with identical zero masks, on uniform and
    non-uniform ln k grids, the 4-point edge stencils (nk = 4, 5), the linear fallback (nk < 4), x on table nodes,
    x outside the table on both sides, chi = 0 and z outside the P(k, z) table."""
    lnk = _GRIDS[grid]()
    z_pk, tab, z_nodes, lnchi, ln_nu = _fq_case(lnk)
    if not use_exp:
        tab = np.exp(tab)
    ref, vraw, scale = _reference(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp)
    new = np.full_like(ref, np.nan)
    _lim._fill_q(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp, new)
    assert np.all(np.isfinite(new))
    assert np.count_nonzero(scale) > 0.4 * ref.size and np.count_nonzero(scale == 0.0) > ln_nu.size   # inside / outside
    _assert_close_fq_cond(new, ref, vraw, scale, use_exp)


def test_fill_q_clipping_matches_reference():
    """kcap (P, not ln P): negative cubic overshoots of a step-like P are clipped to 0 exactly as before."""
    lnk = np.log(np.logspace(-3, 2, 60)) + 0.05 * np.sin(np.arange(60.0))   # non-uniform: the cursor path
    z_pk, tab, z_nodes, lnchi, ln_nu = _fq_case(lnk)
    P = np.exp(tab)
    P[:, 30:] *= 1e-6                                          # a step: the cubic undershoots below 0 after it
    ref, vraw, scale = _reference(P, z_pk, lnk, z_nodes, lnchi, ln_nu, 3, False)
    new = np.empty_like(ref)
    _lim._fill_q(P, z_pk, lnk, z_nodes, lnchi, ln_nu, 3, False, new)
    assert np.count_nonzero(vraw < -1e-3 * scale) > 10        # clipping really happens (clearly negative cubic)
    _assert_close_fq_cond(new, ref, vraw, scale, False)


def test_fill_q_uniform_detection_and_paths():
    """The uniform-grid path is taken only for grids uniform to UNIFORM_LNK_RTOL; on a uniform grid it agrees with
    the monotone-cursor path to < 1e-13 for a smooth table.  (Its fractional position ``s = (x - lnk[0]) / dlnk``
    carries a rounding error of ~|s| 2^-52 node spacings, so next to a near-discontinuity of the table the two paths
    can differ by that shift times the local slope -- as the former kernel's result moves with the rounding of x.)"""
    lnk = _GRIDS["uniform"]()
    dl = lnk[1] - lnk[0]
    z_pk, tab, z_nodes, lnchi, ln_nu = _fq_case(lnk)
    prep = lambda g: _lim._fill_q_prepare(z_pk, z_nodes, lnchi, g, _lim.UNIFORM_LNK_RTOL)
    assert prep(lnk)[2] is True
    for eps, expect in ((1e-13, True), (1e-9, False)):
        g = lnk.copy()
        g[57] += eps * dl
        assert prep(g)[2] is expect, eps
    assert prep(_GRIDS["nonuniform"]())[2] is False
    iz, t, uniform, inv_dlnk, inv_dk, inv_den = prep(lnk)
    work = np.empty((7, lnk.size))
    for use_exp in (True, False):
        tb = tab if use_exp else np.exp(tab)
        a = np.empty((z_nodes.size, ln_nu.size))
        b = np.empty_like(a)
        _lim._fill_q_uniform(tb, lnk, iz, t, lnchi, ln_nu, use_exp, inv_dlnk, work, a)
        _lim._fill_q_cursor(tb, lnk, iz, t, lnchi, ln_nu, use_exp, inv_den, work, b)
        _assert_close_fq(a, b)


def test_fill_q_independent_of_threads_and_chunks(monkeypatch):
    """Bit-identical for 1 vs all numba threads and for any number of node chunks."""
    import numba
    lnk = _GRIDS["nonuniform"]()
    z_pk, tab, z_nodes, lnchi, ln_nu = _fq_case(lnk, nm=503)
    run = lambda: _lim._fill_q(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, 3, True, np.empty((z_nodes.size, ln_nu.size)))
    nt0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = run()
        numba.set_num_threads(nt0)
        many = run()
        monkeypatch.setattr(_lim, "_CHUNKS_PER_THREAD", 1)
        few = run()
        monkeypatch.setattr(_lim, "_CHUNKS_PER_THREAD", 1000)   # more chunks than nodes -> one node per chunk
        finest = run()
    finally:
        numba.set_num_threads(nt0)
    assert np.array_equal(one, many) and np.array_equal(one, few) and np.array_equal(one, finest)


def test_fill_q_degenerate_tables():
    """A single-redshift table gives exact zeros (the former kernel read past the table); results stay finite."""
    lnk = _GRIDS["nonuniform"]()
    z_pk, tab, z_nodes, lnchi, ln_nu = _fq_case(lnk)
    out = np.full((z_nodes.size, ln_nu.size), np.nan)
    _lim._fill_q(tab[:1], z_pk[:1], lnk, z_nodes, lnchi, ln_nu, 3, True, out)
    assert np.all(out == 0.0)


def test_fill_q_allocations_do_not_scale_with_nodes():
    """No heap allocation per node or per chunk inside the parallel kernels (NRT allocation counter): the count per
    call is a constant (parfor scheduling) for 64 / 640 nodes and 1 / 50 chunks.  numba hoists loop-invariant
    ``np.empty`` calls out of ``prange`` bodies (it did so for the former per-node ``row``), so what this guards
    against is an allocation numba cannot hoist; the positive control checks that the counter sees such a
    per-node allocation."""
    import subprocess
    import sys
    code = r'''
import json
import numpy as np
from numba.core.runtime import rtsys
from kidspipe.theory import limber as L
lnk = np.log(np.logspace(-4, 2.7, 150)) + np.r_[0.0, 1e-3 * np.sin(np.arange(149))]
z_pk = np.linspace(0.0, 3.0, 40)
tab_ln = np.log(np.outer(1 + z_pk, np.exp(lnk)) ** -1.0 + 1.0)
tab_p = np.exp(tab_ln)
h1, h2, ih1, ih21, ih32, ih2, ih31, ih3 = L._prep_newton_geom(lnk)
d0, d1, d2, d3 = L._build_newton_from_tab(tab_ln, ih1, ih21, ih32, ih2, ih31, ih3)
counts = []
for nm in (64, 640):
    for nch in (1, 50):
        z = np.linspace(0.0, 2.9, nm); chi = 3000 * z / (1 + 0.4 * z)
        lnchi = np.where(chi > 0, np.log(np.where(chi > 0, chi, 1.0)), np.nan)
        ln_nu = np.log(np.logspace(-1, 5, 80) + 0.5); out = np.empty((nm, 80))
        iz, t, uni, idl, idk, iden = L._fill_q_prepare(z_pk, z, lnchi, lnk, L.UNIFORM_LNK_RTOL)
        las, lbs, j0s, us = L._precompute_cubic_stencils(lnk, iz, lnchi, ln_nu)
        work = np.empty((nch, lnk.size))
        wbuf = np.empty((nch, ln_nu.size))
        wj0 = np.empty((nch, ln_nu.size), dtype=np.int32)
        wu = np.empty((nch, ln_nu.size))
        kerns = (
            lambda: L._fill_q_newton_cursor_fast(d0, d1, d2, d3, h1, h2, lnk, iz, t, lnchi, ln_nu, wbuf, wj0, wu, out),
            lambda: L._fill_q_newton_cached_fast(d0, d1, d2, d3, h1, h2, iz, t, las, lbs, j0s, us, wbuf, out),
            lambda: L._fill_q_uniform_fast(tab_ln, lnk, iz, t, lnchi, ln_nu, idl, work, wbuf, out),
            lambda: L._fill_q_cursor_kcap(tab_p, lnk, iz, t, lnchi, ln_nu, iden, work, out),
            lambda: L._fill_q_cached_kcap(tab_p, lnk, iden, iz, t, lnchi, ln_nu, las, lbs, j0s, work, out),
            lambda: L._fill_q_uniform_kcap(tab_p, lnk, iz, t, lnchi, ln_nu, idl, work, out),
            lambda: L._fill_q_linear_cursor_fast(tab_ln, lnk, iz, t, lnchi, ln_nu, idk, work, wbuf, out),
            lambda: L._fill_q_linear_cursor_kcap(tab_p, lnk, iz, t, lnchi, ln_nu, idk, work, out),
            lambda: L._fill_q_linear_uniform_fast(tab_ln, lnk, iz, t, lnchi, ln_nu, idl, work, wbuf, out),
            lambda: L._fill_q_linear_uniform_kcap(tab_p, lnk, iz, t, lnchi, ln_nu, idl, work, out),
            lambda: L._fill_q_cursor(tab_ln, lnk, iz, t, lnchi, ln_nu, True, iden, work, out),
            lambda: L._fill_q_uniform(tab_ln, lnk, iz, t, lnchi, ln_nu, True, idl, work, out),
            lambda: L._fill_q_linear(tab_ln, lnk, iz, t, lnchi, ln_nu, True, False, idl, idk, work, out),
        )
        for kern in kerns:
            kern()
            s0 = rtsys.get_allocation_stats(); kern(); s1 = rtsys.get_allocation_stats()
            counts.append(s1.alloc - s0.alloc)
from numba import njit, prange
@njit(parallel=True)
def per_node_alloc(out):                 # positive control: a size that depends on m cannot be hoisted
    for m in prange(out.shape[0]):
        tmp = np.empty(m % 7 + 1)
        tmp[:] = m
        out[m, 0] = tmp.sum()
ctrl = []
for nm in (64, 640):
    o = np.empty((nm, 1)); per_node_alloc(o)
    s0 = rtsys.get_allocation_stats(); per_node_alloc(o); s1 = rtsys.get_allocation_stats()
    ctrl.append(s1.alloc - s0.alloc)
print(json.dumps({"counts": counts, "control": ctrl, "nkern": 13}))
'''
    import os
    env = dict(os.environ, NUMBA_NRT_STATS="1")
    src = str(Path(__file__).resolve().parents[1] / "src")      # as tests/test_e2e.py: the package need not be installed
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    got = json.loads(res.stdout.strip().splitlines()[-1])
    counts, ctrl, nkern = got["counts"], got["control"], got["nkern"]
    assert ctrl[0] >= 64 and ctrl[1] >= 640, ctrl            # the counter does see per-node allocations
    per_kernel = [counts[i::nkern] for i in range(nkern)]    # (nm, nchunk) = (64,1), (64,50), (640,1), (640,50)
    for c in per_kernel:
        assert len(set(c)) == 1 and c[0] < 64, counts         # a small constant per-call overhead (parfor scheduling)


def test_vexp1_accuracy_and_fallback():
    """Branch-free ``_vexp1`` achieves < 5e-16 rel across [-708, 709] and ``_vexp_1d`` falls back on outliers."""
    @njit
    def _eval_vexp1(arr):
        out = np.empty_like(arr)
        for i in range(arr.size):
            out[i] = _lim._vexp1(arr[i])
        return out

    @njit
    def _eval_vexp_1d(arr, n):
        out = arr.copy()
        _lim._vexp_1d(out, n)
        return out

    xs = np.concatenate([np.array([-708.0, -50.0, -1e-8, 0.0, 1e-8, 50.0, 709.0]),
                         np.linspace(-708.0, 709.0, 50000)])
    got = _eval_vexp1(xs)
    ref = np.exp(xs)
    rel = np.max(np.abs(got - ref) / ref)
    assert rel < 5e-16, rel

    # In-range 1-D vector matches to < 5e-16 relative
    ok_buf = np.linspace(-100.0, 100.0, 64)
    got_ok = _eval_vexp_1d(ok_buf, ok_buf.size)
    assert np.max(np.abs(got_ok - np.exp(ok_buf)) / np.exp(ok_buf)) < 5e-16

    # Outliers trigger fallback to exact np.exp semantics (underflow, overflow, +/-inf, NaN)
    for bad_val in (-800.0, 800.0, -np.inf, np.inf, np.nan):
        mixed = np.array([-2.0, 0.0, bad_val, 3.0], dtype=np.float64)
        got_bad = _eval_vexp_1d(mixed, mixed.size)
        with np.errstate(over="ignore"):
            ref_bad = np.exp(mixed)
        assert np.array_equal(got_bad, ref_bad, equal_nan=True), (bad_val, got_bad, ref_bad)

    # Partial prefix n < buf.size leaves tail untouched
    tail = np.array([0.0, 1.0, 42.0, -99.0], dtype=np.float64)
    got_tail = _eval_vexp_1d(tail, 2)
    assert np.allclose(got_tail[:2], np.exp(tail[:2]), rtol=5e-16, atol=0)
    assert got_tail[2] == 42.0 and got_tail[3] == -99.0


def test_set_power_cache_equivalence_and_invalidation(toy, nz):
    """Module-level LRU caches are bit-identical to cold computation, thread-safe, and never serve stale data."""
    from concurrent.futures import ThreadPoolExecutor

    ell = coarse_ell(30)
    chi_w = toy["chi_tab"].copy()
    z_w = toy["z_tab"].copy()
    zpk_w = toy["z_pk"].copy()
    kh_w = toy["k_h"].copy()
    pk_w = toy["pk"].copy()

    # 1. Cached medium-move set_power is bit-identical to uncached (fast, kcap, uniform, linear)
    for fidelity in ("fast", "kcap"):
        for order in (3, 1):
            _lim._clear_limber_caches()
            lim = Limber(ell=ell, fidelity=fidelity, n_nodes=400, interp_order=order)
            lim.set_power(chi_w, z_w, OM, zpk_w, kh_w, pk_w)
            # k-independent move hits _DIST_CACHE, _STENCIL_CACHE, and _GROWTH_CACHE
            pk_move1 = pk_w * 1.15
            lim.set_power(chi_w, z_w, OM * 1.02, zpk_w, kh_w, pk_move1)
            q1_cached, d1_cached = lim.Qt.copy(), lim.D.copy()
            # k-dependent move hits _DIST_CACHE and _STENCIL_CACHE, misses _GROWTH_CACHE
            pk_move2 = pk_w * (1.0 + 0.05 * np.tanh(kh_w))[None, :]
            lim.set_power(chi_w, z_w, OM * 0.98, zpk_w, kh_w, pk_move2)
            q2_cached, d2_cached = lim.Qt.copy(), lim.D.copy()

            _lim._clear_limber_caches()
            f1 = Limber(ell=ell, fidelity=fidelity, n_nodes=400, interp_order=order).set_power(
                chi_w, z_w, OM * 1.02, zpk_w, kh_w, pk_move1)
            _lim._clear_limber_caches()
            f2 = Limber(ell=ell, fidelity=fidelity, n_nodes=400, interp_order=order).set_power(
                chi_w, z_w, OM * 0.98, zpk_w, kh_w, pk_move2)
            assert np.array_equal(q1_cached, f1.Qt) and np.array_equal(d1_cached, f1.D)
            assert np.array_equal(q2_cached, f2.Qt) and np.array_equal(d2_cached, f2.D)

    # Uniform k_h grid cached equivalence
    kh_u = np.exp(np.linspace(np.log(kh_w[0]), np.log(kh_w[-1]), 120))
    pk_u = np.exp(np.stack([np.interp(np.log(kh_u), np.log(kh_w), np.log(row)) for row in pk_w]))
    _lim._clear_limber_caches()
    lim_u = Limber(ell=ell, fidelity="fast", n_nodes=300).set_power(chi_w, z_w, OM, zpk_w, kh_u, pk_u)
    lim_u.set_power(chi_w, z_w, OM, zpk_w, kh_u, pk_u * 1.2)
    qu_cached = lim_u.Qt.copy()
    _lim._clear_limber_caches()
    fu = Limber(ell=ell, fidelity="fast", n_nodes=300).set_power(chi_w, z_w, OM, zpk_w, kh_u, pk_u * 1.2)
    assert np.array_equal(qu_cached, fu.Qt)

    # 2. In-place mutations of writeable input arrays invalidate caches
    def check_mut(mut_fn):
        c = chi_w.copy()
        z = z_w.copy()
        zp = zpk_w.copy()
        k = kh_w.copy()
        p = pk_w.copy()
        _lim._clear_limber_caches()
        lim_m = Limber(ell=ell, n_nodes=300)
        lim_m.set_power(c, z, OM, zp, k, p)
        q_before = lim_m.Qt.copy()
        mut_fn(c, z, zp, k, p)
        lim_m.set_power(c, z, OM, zp, k, p)
        q_mut = lim_m.Qt.copy()
        _lim._clear_limber_caches()
        fresh_m = Limber(ell=ell, n_nodes=300).set_power(c, z, OM, zp, k, p)
        assert not np.array_equal(q_mut, q_before)
        assert np.array_equal(q_mut, fresh_m.Qt) and np.array_equal(lim_m.D, fresh_m.D)

    check_mut(lambda c, z, zp, k, p: c.__setitem__(slice(100, None), c[100:] * 1.03))
    check_mut(lambda c, z, zp, k, p: z.__setitem__(slice(1, -1), z[1:-1] + 5e-4))
    check_mut(lambda c, z, zp, k, p: zp.__setitem__(slice(1, -1), zp[1:-1] + 5e-4))
    check_mut(lambda c, z, zp, k, p: k.__setitem__(120, k[120] * 1.01))
    check_mut(lambda c, z, zp, k, p: p.__setitem__(10, p[10] * 1.5))

    # 3. Varying z_max, n_nodes, kernel_oversample, ell, interp_order, fidelity never serves stale data
    _lim._clear_limber_caches()
    base = Limber(ell=ell, fidelity="fast", n_nodes=300, interp_order=3, kernel_oversample=4).set_power(
        chi_w, z_w, OM, zpk_w, kh_w, pk_w, z_max=5.5)
    variants = [
        dict(ell=ell, fidelity="fast", n_nodes=300, interp_order=3, kernel_oversample=4, z_max=5.0),
        dict(ell=ell, fidelity="fast", n_nodes=350, interp_order=3, kernel_oversample=4, z_max=5.5),
        dict(ell=ell, fidelity="fast", n_nodes=300, interp_order=3, kernel_oversample=2, z_max=5.5),
        dict(ell=ell * 1.15, fidelity="fast", n_nodes=300, interp_order=3, kernel_oversample=4, z_max=5.5),
        dict(ell=ell, fidelity="fast", n_nodes=300, interp_order=1, kernel_oversample=4, z_max=5.5),
        dict(ell=ell, fidelity="kcap", n_nodes=300, interp_order=3, kernel_oversample=4, z_max=5.5),
    ]
    for kw in variants:
        zm = kw.pop("z_max")
        l_var = Limber(**kw).set_power(chi_w, z_w, OM, zpk_w, kh_w, pk_w, z_max=zm)
        cl_var = l_var.cl(nz.n, 0.3, z_nz=nz.z)
        _lim._clear_limber_caches()
        l_ref = Limber(**kw).set_power(chi_w, z_w, OM, zpk_w, kh_w, pk_w, z_max=zm)
        cl_ref = l_ref.cl(nz.n, 0.3, z_nz=nz.z)
        assert np.array_equal(l_var.Qt, l_ref.Qt)
        assert np.array_equal(l_var._chif, l_ref._chif)
        assert np.array_equal(cl_var, cl_ref)
        if kw["kernel_oversample"] == 4:
            assert not np.array_equal(l_var.Qt, base.Qt)
        else:
            assert l_var._zf.size != base._zf.size

    # 4. Concurrent multi-threaded set_power produces bit-identical results
    _lim._clear_limber_caches()
    ref_t = [
        Limber(ell=ell, fidelity="fast", n_nodes=300).set_power(chi_w, z_w, OM, zpk_w, kh_w, pk_w * (1.0 + 0.1 * i)).Qt
        for i in range(4)
    ]
    _lim._clear_limber_caches()

    def worker(idx):
        return Limber(ell=ell, fidelity="fast", n_nodes=300).set_power(
            chi_w, z_w, OM, zpk_w, kh_w, pk_w * (1.0 + 0.1 * (idx % 4))).Qt

    with ThreadPoolExecutor(max_workers=8) as ex:
        got_t = list(ex.map(worker, range(16)))
    for idx, qt in enumerate(got_t):
        assert np.array_equal(qt, ref_t[idx % 4])

    # 5. Invalid inputs still raise ValueError after warm cache hits without corrupting existing state
    lim_v = Limber(ell=ell, fidelity="fast", n_nodes=300).set_power(chi_w, z_w, OM, zpk_w, kh_w, pk_w)
    qt_ok = lim_v.Qt.copy()
    for bad_pk in (np.nan, 0.0, -1.0):
        bpk = pk_w.copy()
        bpk[3, 4] = bad_pk
        with pytest.raises(ValueError):
            lim_v.set_power(chi_w, z_w, OM, zpk_w, kh_w, bpk)
        bpk0 = pk_w.copy()
        bpk0[0, 0] = bad_pk
        with pytest.raises(ValueError):
            lim_v.set_power(chi_w, z_w, OM, zpk_w, kh_w, bpk0)
    with pytest.raises(ValueError):
        lim_v.set_power(chi_w, z_w, OM, zpk_w, kh_w[::-1], pk_w[:, ::-1])
    with pytest.raises(ValueError):
        lim_v.set_power(chi_w, z_w[::-1], OM, zpk_w, kh_w, pk_w)
    with pytest.raises(ValueError):
        lim_v.set_power(chi_w, z_w[:2], OM, zpk_w, kh_w, pk_w)
    with pytest.raises(ValueError):
        lim_v.set_power(chi_w, z_w, OM, zpk_w[:0], kh_w, pk_w[:0])
    assert lim_v._ready and np.array_equal(lim_v.Qt, qt_ok)
