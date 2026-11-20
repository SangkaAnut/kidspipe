"""Tests of the stand-alone numba HMcode-2016 (src/kidspipe/theory/hmcode.py).

Independent references: scipy (Ci/Si, quad, ODE), CAMB 1.5.8 ``halofit_version='mead'`` (driven here
directly, no kidspipe module), and the stored array ``data/validation/pk_mead.txt`` (VALIDATION_TARGETS s1.2).
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.special import sici

camb = pytest.importorskip("camb")

from kidspipe.theory import hmcode  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "validation"

# tolerances of VALIDATION_TARGETS s1.6 (test_meadcb): 3e-3 for z<2, 1.5e-2 for z>=2
TOL_LOZ, TOL_HIZ = 3.0e-3, 1.5e-2


# ------------------------------------------------------------------------------------------
# CAMB driver (tests only)
# ------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def camb_run(omch2, ombh2, h, ns, As, mnu, A, eta0, zs):
    """CAMB 'mead' run -> linear total & cold P, non-linear P on CAMB's native k nodes (h/Mpc, (Mpc/h)^3)."""
    p = camb.CAMBparams()
    if mnu > 0:
        p.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=mnu, num_massive_neutrinos=1, omk=0)
    else:
        p.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.0, num_massive_neutrinos=0, omk=0)
    p.InitPower.set_params(As=As, ns=ns)
    p.set_dark_energy(w=-1.0)
    p.NonLinearModel = camb.nonlinear.Halofit()
    p.NonLinearModel.set_params(halofit_version="mead", HMCode_A_baryon=A, HMCode_eta_baryon=eta0)
    p.set_matter_power(redshifts=list(zs[::-1]), kmax=20.0, nonlinear=True)
    p.NonLinear = camb.model.NonLinear_both
    p.WantCls = False
    r = camb.get_results(p)
    kh, z, pl = r.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    pcb = r.get_linear_matter_power_spectrum(var1="delta_nonu", var2="delta_nonu",
                                             hubble_units=True, k_hunit=True)[2]
    pnl = r.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)[2]
    return dict(kh=kh, z=z, pl=pl, pcb=pcb, pnl=pnl, om=p.omegam, h=h, ns=ns, fnu=p.omnuh2 / (p.omegam * h * h),
                sigma8=r.get_sigma8()[::-1])


def run_hm(o, A, eta0, **kw):
    return hmcode.hmcode_pnl(o["z"], o["kh"], o["pl"], o["om"], o["fnu"], o["h"], o["ns"], A, eta0, o["pcb"], **kw)


# dense enough that the worst-z of the extreme corners (z=0, 0.25, 1.9) is covered
ZS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9, 2.0, 3.0, 4.0, 6.0)


def cosmo(omch2=0.1, ombh2=0.022, h=0.7, ns=1.09, lnA=3.09, mnu=0.06, A=2.0, eta0=None, zs=ZS):
    eta0 = 0.98 - 0.12 * A if eta0 is None else eta0
    return camb_run(omch2, ombh2, h, ns, float(np.exp(lnA) * 1e-10), mnu, A, eta0, tuple(zs))


def max_dev_by_z(pm, pc):
    return np.abs(pm / pc - 1.0).max(axis=1)


def assert_vs_camb(o, A, eta0, tol_lo=TOL_LOZ, tol_hi=TOL_HIZ, **kw):
    dev = max_dev_by_z(run_hm(o, A, eta0, **kw), o["pnl"])
    lo = o["z"] < 2.0
    assert dev[lo].max() <= tol_lo, f"z<2 deviation {dev[lo].max():.2e} > {tol_lo}"
    assert dev[~lo].max() <= tol_hi, f"z>=2 deviation {dev[~lo].max():.2e} > {tol_hi}"
    return dev


# ------------------------------------------------------------------------------------------
# building blocks
# ------------------------------------------------------------------------------------------
def test_nfw_window_matches_scipy_ci_si():
    x = np.logspace(-3, 2, 800)
    for c in (1.0, 3.0, 10.0, 40.0, 300.0):
        si1, ci1 = sici((1 + c) * x)
        si0, ci0 = sici(x)
        w = (np.cos(x) * (ci1 - ci0) + np.sin(x) * (si1 - si0) - np.sin(c * x) / ((1 + c) * x)) / (
            np.log(1 + c) - c / (1 + c))
        w = np.clip(w, 0, 1)
        assert np.abs(hmcode.nfw_window(x, c) - w).max() < 2e-5
    assert hmcode.nfw_window(1e-8, 5.0) == pytest.approx(1.0, abs=1e-12)
    assert np.all(hmcode.nfw_window(np.logspace(-4, 3, 50), 7.0) <= 1.0)


def test_nfw_window_matches_profile_integral():
    """W(k) = int_0^{R_v} 4 pi r^2 rho(r) sin(kr)/(kr) dr / M for rho ~ 1/[(r/rs)(1+r/rs)^2]."""
    for c, x in [(4.0, 0.3), (4.0, 2.0), (10.0, 1.0), (10.0, 6.0), (25.0, 0.5)]:
        # units r_s = 1: R_v = c, k = x
        num = integrate.quad(lambda r: np.sin(x * r) / (x * (1 + r) ** 2), 0, c, limit=400)[0]
        den = np.log(1 + c) - c / (1 + c)
        assert hmcode.nfw_window(x, c) == pytest.approx(num / den, abs=2e-6)


def test_growth_table_matches_ode():
    from scipy.integrate import solve_ivp

    for om in (0.1, 0.3, 0.8):
        lna, lng = hmcode.growth_table(om)
        # D'' + (2 + dlnH/dlna) D' - 1.5 Om a^-3/E^2 D = 0   in ln a, matter + Lambda
        def rhs(t, y):
            a = np.exp(t)
            e2 = om * a ** -3 + 1 - om
            dlnh = -1.5 * om * a ** -3 / e2
            return [y[1], -(2 + dlnh) * y[1] + 1.5 * om * a ** -3 / e2 * y[0]]
        a0 = 1e-4
        sol = solve_ivp(rhs, [np.log(a0), 0.0], [a0, a0], rtol=1e-10, atol=1e-14, dense_output=True)
        d = sol.sol(lna[lna > np.log(1e-3)])[0]
        d = d / sol.y[0, -1]
        assert np.abs(np.exp(lng[lna > np.log(1e-3)]) / d - 1).max() < 2e-4
    lna, lng = hmcode.growth_table(1.0)
    assert np.allclose(np.exp(lng), np.exp(lna), rtol=1e-12)          # EdS: g = a
    assert lng[-1] == pytest.approx(0.0, abs=1e-12)


def test_omega_m_of_z():
    z = np.array([0.0, 1.0, 5.0])
    assert hmcode.omega_m_of_z(0.0, 0.3)[()] == pytest.approx(0.3)
    om = hmcode.omega_m_of_z(z, 0.3)
    assert np.allclose(om, 0.3 * (1 + z) ** 3 / (0.3 * (1 + z) ** 3 + 0.7))
    assert hmcode.omega_m_of_z(z, 1.0) == pytest.approx(1.0)


def test_sigma8_and_sigma_v_vs_independent_quadrature():
    o = cosmo(zs=(0.0, 1.0))
    _, diag = hmcode.hmcode_pnl_diag(o["z"], o["kh"], o["pl"], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74, o["pcb"])
    k = o["kh"]
    for iz in range(2):
        cs = CubicSpline(np.log(k), np.log(k ** 3 * o["pl"][iz] / (2 * np.pi ** 2)))
        d2 = lambda lk: np.exp(cs(lk))  # noqa: E731
        lo, hi = np.log(k[0]), np.log(k[-1])
        wth = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3  # noqa: E731
        s8 = np.sqrt(integrate.quad(lambda lk: d2(lk) * wth(8 * np.exp(lk)) ** 2, lo, hi, limit=800)[0])
        sv = np.sqrt(integrate.quad(lambda lk: d2(lk) / np.exp(2 * lk) / 3, lo, hi, limit=400)[0])
        sv100 = np.sqrt(integrate.quad(lambda lk: d2(lk) / np.exp(2 * lk) / 3 * wth(100 * np.exp(lk)) ** 2,
                                       lo, hi, limit=800)[0])
        assert diag[iz, 0] == pytest.approx(s8, rel=2e-4)
        assert diag[iz, 1] == pytest.approx(sv, rel=2e-3)     # extrapolated tail beyond kmax adds ~1e-3
        assert diag[iz, 2] == pytest.approx(sv100, rel=2e-4)
    # CAMB's own sigma8 (same total-matter P_lin): agree to the extrapolation/truncation level
    assert diag[0, 0] == pytest.approx(o["sigma8"][0], rel=2e-3)


def test_pk_cb_approx_accuracy():
    o = cosmo(omch2=0.12, ombh2=0.0222, h=0.7, ns=0.97, lnA=3.04)
    approx = hmcode.pk_cb_approx(o["z"], o["kh"], o["pl"], o["om"], o["fnu"])
    assert np.abs(approx / o["pcb"] - 1).max() < 2.5e-3
    assert np.array_equal(hmcode.pk_cb_approx(o["z"], o["kh"], o["pl"], o["om"], 0.0), o["pl"])
    # end-to-end with the approximate cold spectrum: still within the CAMB tolerance
    pn = hmcode.hmcode_pnl(o["z"], o["kh"], o["pl"], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74)
    assert max_dev_by_z(pn, o["pnl"])[o["z"] < 2].max() < TOL_LOZ


# ------------------------------------------------------------------------------------------
# structural properties
# ------------------------------------------------------------------------------------------
def test_low_k_limit_is_linear_and_damped():
    o = cosmo(zs=(0.0, 1.0))
    pn = run_hm(o, 2.0, 0.74)
    low = o["kh"] < 2e-3
    assert np.abs(pn[:, low] / o["pl"][:, low] - 1).max() < 5e-4
    # LPT-like damping of the 2-halo term: the ratio dips below one around k ~ 3e-3 .. 1e-2 (z=0: ~ -1e-3)
    mid = (o["kh"] > 3e-3) & (o["kh"] < 8e-3)
    assert np.all(pn[:, mid] < o["pl"][:, mid])
    assert (pn[0, mid] / o["pl"][0, mid]).min() > 0.995
    # strongly non-linear: boost at k = 10 h/Mpc of order 20-80 at z=0
    i = np.searchsorted(o["kh"], 10.0)
    assert 20 < pn[0, i] / o["pl"][0, i] < 120
    assert np.all(np.isfinite(pn)) and np.all(pn > 0)


def test_reduced_z_grid_equals_subset_of_full_grid():
    zs = tuple(np.linspace(0, 4, 17))
    o = cosmo(zs=zs)
    full = run_hm(o, 2.0, 0.74)
    idx = [0, 3, 8, 16]
    red = hmcode.hmcode_pnl(o["z"][idx], o["kh"], o["pl"][idx], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74,
                            o["pcb"][idx])
    assert np.array_equal(red, full[idx])
    # order of redshifts is irrelevant
    rev = hmcode.hmcode_pnl(o["z"][::-1], o["kh"], o["pl"][::-1], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74,
                            o["pcb"][::-1])
    assert np.array_equal(rev[::-1], full)
    # single scalar redshift allowed
    one = hmcode.hmcode_pnl(0.5, o["kh"], o["pl"][2:3], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74, o["pcb"][2:3])
    assert one.shape == (1, o["kh"].size)


def test_numerical_convergence():
    o = cosmo(omch2=0.12, ombh2=0.0222, h=0.7, ns=0.97, lnA=3.05, A=3.13)
    ref = run_hm(o, 3.13, 0.604, ms=48, dl=0.02)
    base = run_hm(o, 3.13, 0.604)                       # defaults ms=24, dl=0.04
    assert np.abs(base / ref - 1).max() < 3e-4
    coarse = run_hm(o, 3.13, 0.604, ms=16)
    assert np.abs(coarse / ref - 1).max() < 1.5e-3      # documented speed/accuracy option
    # window-pruning tolerance is harmless
    assert np.abs(run_hm(o, 3.13, 0.604, tol_win=0.0) / base - 1).max() < 1e-8 + 1e-6


def test_input_validation():
    k = np.logspace(-3, 1, 20)
    p = np.ones((2, 20))
    args = ([0.0, 1.0], k, p, 0.3, 0.0, 0.7, 0.96, 2.0, 0.74)

    def call(**over):
        a = list(args)
        names = ["z", "k_h", "pk_lin", "omega_m", "omega_nu_frac", "h", "n_s", "A_baryon", "eta0"]
        kw = {}
        for key, val in over.items():
            if key in names:
                a[names.index(key)] = val
            else:
                kw[key] = val
        return hmcode.hmcode_pnl(*a, **kw)

    bad_p = p.copy(); bad_p[1, 5] = np.nan
    neg_p = p.copy(); neg_p[0, 3] = -1.0
    zero_p = p.copy(); zero_p[0, 3] = 0.0
    for over in (dict(pk_lin=bad_p), dict(pk_lin=neg_p), dict(pk_lin=zero_p), dict(pk_cb=bad_p),
                 dict(z=[-0.5, 1.0]), dict(z=[np.nan, 1.0]), dict(omega_m=0.0), dict(omega_m=-0.3),
                 dict(omega_m=np.nan), dict(omega_nu_frac=-0.1), dict(omega_nu_frac=1.0), dict(h=0.0),
                 dict(A_baryon=0.0), dict(A_baryon=-1.0), dict(A_baryon=np.nan), dict(eta0=np.nan),
                 dict(n_s=np.inf), dict(ms=0), dict(dl=0.0), dict(mmin=1e18, mmax=1.0), dict(mmin=0.0),
                 dict(k_h=np.concatenate([[0.0], k[1:]])), dict(k_h=k[:3], pk_lin=p[:, :3])):
        with pytest.raises(ValueError):
            call(**over)
    with pytest.raises(ValueError):
        hmcode.hmcode_pnl([0.0, 1.0], k, p[:1], 0.3, 0.0, 0.7, 0.96, 2.0, 0.74)
    with pytest.raises(ValueError):
        hmcode.hmcode_pnl([0.0, 1.0], k[::-1], p, 0.3, 0.0, 0.7, 0.96, 2.0, 0.74)
    with pytest.raises(ValueError):
        hmcode.hmcode_pnl([0.0, 1.0], k, p, 0.3, 0.0, 0.7, 0.96, 2.0, 0.74, pk_cb=p[:1])


def test_deterministic_and_thread_independent():
    import numba

    o = cosmo(zs=tuple(np.linspace(0, 3, 12)))
    a = run_hm(o, 2.5, 0.68)
    nthr = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        b = run_hm(o, 2.5, 0.68)
    finally:
        numba.set_num_threads(nthr)
    assert np.array_equal(a, b)
    assert np.array_equal(a, run_hm(o, 2.5, 0.68))
    # dtype / memory layout of the inputs is irrelevant
    c = hmcode.hmcode_pnl(o["z"], o["kh"], np.asfortranarray(o["pl"]), o["om"], o["fnu"], o["h"], o["ns"], 2.5, 0.68,
                          np.asfortranarray(o["pcb"]))
    assert np.array_equal(a, c)


def test_halo_model_parameters_vs_independent_quadrature():
    """delta_c, Delta_v, eta, alpha, neff (nu(R_nl)=1) from the spec formulas with scipy quad + brentq."""
    from scipy import optimize

    o = cosmo(omch2=0.12, ombh2=0.0222, h=0.7, ns=0.97, lnA=3.05, zs=(0.0, 1.0, 3.0))
    _, dg = hmcode.hmcode_pnl_diag(o["z"], o["kh"], o["pl"], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74, o["pcb"])
    k, om, fnu = o["kh"], o["om"], o["fnu"]
    wth = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3  # noqa: E731
    for iz, z in enumerate(o["z"]):
        om_z = om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om)       # radiation (3.6e-5/h^2) shifts this by <~1e-3
        cs = CubicSpline(np.log(k), np.log(k ** 3 * o["pcb"][iz] / (2 * np.pi ** 2)))
        lkm = np.log(k[-1])
        slope = (cs(lkm) - cs(np.log(k[-2]))) / (lkm - np.log(k[-2]))
        d2 = lambda lk: np.exp(cs(lk)) if lk <= lkm else np.exp(cs(lkm) + slope * (lk - lkm))  # noqa: E731

        def sig(R):
            f = lambda lk: d2(lk) * wth(np.exp(lk) * R) ** 2  # noqa: E731
            return np.sqrt(integrate.quad(f, np.log(k[0]), np.log(1e5 / R), limit=2000, points=[lkm])[0])

        s8 = dg[iz, 0]
        dc = (1.59 + 0.0314 * np.log(s8)) * (1 + 0.262 * fnu) * (1 + 0.0123 * np.log10(om_z))
        dv = 418 * om_z ** -0.352 * (1 + 0.916 * fnu)
        lr = optimize.brentq(lambda x: np.log(sig(np.exp(x)) / dc), np.log(1e-3), np.log(50))
        hh = 1e-3
        neff = -3 - (np.log(sig(np.exp(lr + hh))) - np.log(sig(np.exp(lr - hh)))) / hh
        assert dg[iz, 6] == pytest.approx(dc, rel=2e-4)
        assert dg[iz, 7] == pytest.approx(dv, rel=1e-3)
        assert dg[iz, 8] == pytest.approx(0.74 - 0.3 * s8, abs=1e-12)
        assert dg[iz, 4] == pytest.approx(neff, abs=3e-3)
        assert dg[iz, 5] == pytest.approx(3.24 * 1.85 ** dg[iz, 4], rel=1e-12)
        assert dg[iz, 9] == pytest.approx(0.584 / dg[iz, 1], rel=1e-12)


def test_kgrid_resampling_is_benign():
    """P_nl must not depend on the node set at the 3e-3 level when P_lin is resampled to a coarser/shifted
    grid (the contract is CAMB's native nodes; this guards against hidden dependence on the node count)."""
    o = cosmo(omch2=0.12, ombh2=0.0222, h=0.7, ns=0.97, lnA=3.05, zs=(0.0, 1.0, 3.0))
    kh = o["kh"]
    ref = run_hm(o, 2.0, 0.74)
    k2 = np.geomspace(kh[0], kh[-1], 100)
    f = lambda p: np.exp(CubicSpline(np.log(kh), np.log(p), axis=1)(np.log(k2)))  # noqa: E731
    pm = hmcode.hmcode_pnl(o["z"], k2, f(o["pl"]), o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74, f(o["pcb"]))
    pref = np.exp(CubicSpline(np.log(kh), np.log(ref), axis=1)(np.log(k2)))
    assert np.abs(pm / pref - 1).max() < 3e-3


# ------------------------------------------------------------------------------------------
# validation targets
# ------------------------------------------------------------------------------------------
def test_stored_array_pk_mead():
    """VALIDATION_TARGETS s1.2: A=2.0, eta0 = 0.98 - 0.12 A = 0.74, rtol 1e-2 for 1e-4 < k < 30 h/Mpc, all z."""
    kf = np.loadtxt(DATA / "k_h.txt")
    zf = np.loadtxt(DATA / "z.txt")
    pf = np.loadtxt(DATA / "pk_mead.txt")
    o = camb_run(0.1181838, 0.02228089, 0.679565, 0.9679701, 2.170229e-9, 0.06, 2.0, 0.74, tuple(zf))
    pn = run_hm(o, 2.0, 0.74)
    sel = (kf > 1e-4) & (kf < 30.0) & (kf > o["kh"][0]) & (kf < o["kh"][-1])
    assert sel.sum() > 150
    pm = np.exp(CubicSpline(np.log(o["kh"]), np.log(pn), axis=1)(np.log(kf[sel])))
    rel = pm / pf[:, sel] - 1
    assert np.abs(rel).max() < 1e-2
    assert np.sqrt((rel ** 2).mean()) < 4e-3
    # anchors listed in VALIDATION_TARGETS (linear regime and non-linear)
    for z_i, kk, val in [(0, 0.00968, 22860.84), (0, 0.1005, 5674.30), (0, 1.00298, 355.118),
                         (0, 8.96295, 9.08861)]:
        mine = np.exp(CubicSpline(np.log(o["kh"]), np.log(pn[z_i]))(np.log(kk)))
        assert mine == pytest.approx(val, rel=1e-2)
    # the eta0 = 0.603 quoted in the repository scripts does NOT reproduce the array (up to ~20 %)
    bad = run_hm(o, 2.0, 0.603)
    pm_bad = np.exp(CubicSpline(np.log(o["kh"]), np.log(bad), axis=1)(np.log(kf[sel])))
    assert np.abs(pm_bad / pf[:, sel] - 1).max() > 5e-2


def test_vs_camb_pk_mead_cosmology_both_baryon_corners():
    for A in (2.0, 3.13):
        o = camb_run(0.1181838, 0.02228089, 0.679565, 0.9679701, 2.170229e-9, 0.06, A, 0.98 - 0.12 * A,
                     (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0))
        assert_vs_camb(o, A, 0.98 - 0.12 * A)


CASES = [
    # id, kwargs for cosmo(), tolerance override for z<2
    ("kv450_start_A2", dict(omch2=0.10, ombh2=0.022, h=0.70, ns=1.09, lnA=3.09, A=2.0), TOL_LOZ),
    ("kv450_start_A313", dict(omch2=0.10, ombh2=0.022, h=0.70, ns=1.09, lnA=3.09, A=3.13), TOL_LOZ),
    ("planck_like_A25", dict(omch2=0.12, ombh2=0.0222, h=0.70, ns=0.97, lnA=3.04, A=2.5), TOL_LOZ),
    ("high_Om_high_h", dict(omch2=0.25, ombh2=0.025, h=0.80, ns=1.2, lnA=4.0, A=2.0), TOL_LOZ),
    ("mid_Om_blue", dict(omch2=0.20, ombh2=0.022, h=0.70, ns=1.1, lnA=3.5, A=2.5), TOL_LOZ),
    ("Om_gt_1_low_h", dict(omch2=0.40, ombh2=0.022, h=0.64, ns=1.0, lnA=3.0, A=3.13), TOL_LOZ),
    ("red_tilt_high_As", dict(omch2=0.12, ombh2=0.022, h=0.70, ns=0.7, lnA=5.0, A=2.0), TOL_LOZ),
    ("Om_0.05_blue_high_As", dict(omch2=0.01, ombh2=0.022, h=0.82, ns=1.3, lnA=5.0, A=2.0), TOL_LOZ),
    ("massless_nu", dict(omch2=0.12, ombh2=0.0222, h=0.70, ns=0.96, lnA=3.05, A=2.0, mnu=0.0), TOL_LOZ),
    # extreme low-amplitude/red corner: R_nl lies below the mass range for z>~1 -> clamped neff
    ("low_Om_red_low_As_corner", dict(omch2=0.05, ombh2=0.02, h=0.65, ns=0.8, lnA=2.5, A=3.13), TOL_LOZ),
]


@pytest.mark.parametrize("name,kw,tol_lo", CASES, ids=[c[0] for c in CASES])
def test_vs_camb_mead_cosmologies(name, kw, tol_lo):
    """CAMB 'mead' with (A_baryon, eta0 = 0.98 - 0.12 A): <=3e-3 for z<2, <=1.5e-2 for z>=2 (s1.6 criteria)."""
    o = cosmo(**kw)
    A = kw["A"]
    # z>=2: the spec allows 1.5e-2, the implementation achieves ~1e-3 (continuation slope above the last
    # CAMB node was the limiting factor); guard it at 4e-3 so that a regression is noticed.
    assert_vs_camb(o, A, 0.98 - 0.12 * A, tol_lo=tol_lo, tol_hi=4.0e-3)


def test_free_eta0_and_A_response_matches_camb():
    """Independent A and eta0 (not tied by eta0 = 0.98-0.12A): the response ratios must agree with CAMB."""
    base = dict(omch2=0.12, ombh2=0.0222, h=0.70, ns=0.97, lnA=3.05)
    o1 = cosmo(**base, A=2.0, eta0=0.74)
    o2 = cosmo(**base, A=3.13, eta0=0.74)           # only the concentration changes
    o3 = cosmo(**base, A=2.0, eta0=0.45)            # only the window mass dependence changes
    h1 = run_hm(o1, 2.0, 0.74)
    h2 = run_hm(o2, 3.13, 0.74)
    h3 = run_hm(o3, 2.0, 0.45)
    for (ha, hb, oa, ob) in [(h1, h2, o1, o2), (h1, h3, o1, o3)]:
        rm = hb / ha
        rc = ob["pnl"] / oa["pnl"]
        assert np.abs(rm / rc - 1).max() < 2e-3
        assert np.abs(rc - 1).max() > 0.02          # the response is not trivial
    # physical sign: a more concentrated halo (larger A) adds power at high k
    i = np.searchsorted(o1["kh"], 10.0)
    assert h2[0, i] > h1[0, i]


# ------------------------------------------------------------------------------------------
# performance
# ------------------------------------------------------------------------------------------
def test_performance_full_grid_single_thread():
    import numba

    zs = np.concatenate([np.linspace(0, 2, 100, endpoint=False), np.linspace(2, 6, 50)])
    p = camb.CAMBparams()
    p.set_cosmology(H0=70, ombh2=0.022, omch2=0.1, mnu=0.06, num_massive_neutrinos=1)
    p.InitPower.set_params(As=2e-9, ns=0.96)
    p.set_matter_power(redshifts=list(zs[::-1]), kmax=20.0, nonlinear=False)
    p.WantCls = False
    r = camb.get_results(p)
    kh, _, pl = r.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    pcb = r.get_linear_matter_power_spectrum(var1="delta_nonu", var2="delta_nonu", hubble_units=True,
                                             k_hunit=True)[2]
    om, fnu = p.omegam, p.omnuh2 / (p.omegam * 0.49)
    nthr = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        args = (zs, kh, pl, om, fnu, 0.7, 0.96, 2.0, 0.74, pcb)
        hmcode.hmcode_pnl(*args)                      # JIT / table warm-up
        best = min(_time(lambda: hmcode.hmcode_pnl(*args)) for _ in range(5))
    finally:
        numba.set_num_threads(nthr)
    assert pl.shape == (150, kh.size)
    assert best < 0.30, f"single-thread full grid took {best:.3f} s"
    # reduced z grid (e.g. 24 nodes + interpolation) is proportionally cheaper
    sub = np.arange(0, 150, 6)
    numba.set_num_threads(1)
    try:
        t_sub = min(_time(lambda: hmcode.hmcode_pnl(zs[sub], kh, pl[sub], om, fnu, 0.7, 0.96, 2.0, 0.74, pcb[sub]))
                    for _ in range(5))
    finally:
        numba.set_num_threads(nthr)
    assert t_sub < 0.5 * best


def _time(f):
    t = time.perf_counter()
    f()
    return time.perf_counter() - t
