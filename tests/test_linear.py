"""Tests for kidspipe.theory.linear (spec s2.1-s2.4): sigma8 / Omega_m anchors, exact A_s & n_s rescaling,
distances vs an independent scipy quad, cb spectrum identity, grids, cache behaviour."""
import json
import os
import time

import numpy as np
from kidspipe._compat import trapezoid
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

camb = pytest.importorskip("camb")

from kidspipe.theory.linear import (LinearBackend, LinearResult, Z_CHI, Z_GRID_REDUCED, Z_GRID_SPEC,  # noqa: E402
                                    default_nthreads)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "validation")
SANITY = dict(omch2=0.10, ombh2=0.022, h=0.70, n_s=1.09, ln_1e10_A_s=3.09)       # spec s2.2 / VALIDATION s4.3
TESTCOSMO = dict(omch2=0.10, ombh2=0.022, h=0.70, n_s=0.96, ln_1e10_A_s=3.0)      # spec s4.1


@pytest.fixture(scope="module")
def backend():
    return LinearBackend(nthreads=4)


@pytest.fixture(scope="module")
def lin_sanity(backend):
    return backend.compute(**SANITY)


def fresh_camb(omch2, ombh2, h, n_s, ln_1e10_A_s, z, var=None, mnu=0.06):
    """Independent CAMB run written without the wrapper (separate code path)."""
    p = camb.CAMBparams()
    p.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=mnu, num_massive_neutrinos=1, omk=0)
    p.InitPower.set_params(As=np.exp(ln_1e10_A_s) * 1e-10, ns=n_s)
    p.Reion.Reionization = False
    p.WantCls = False
    p.set_matter_power(redshifts=list(z[::-1]), kmax=20.0, nonlinear=False)
    camb.config.ThreadNum = 4
    r = camb.get_results(p)
    return p, r


# ------------------------------------------------------------------------------------------ anchors
def test_spec_sanity_point(lin_sanity):
    """spec s2.2: sigma8=0.779557, omega_m=0.250296, omnuh2=6.44867e-4, nk=194, kmax=30 h/Mpc, kmin=7.1429e-5."""
    r = lin_sanity
    assert r.sigma8_0 == pytest.approx(0.779557, abs=1e-6)
    assert r.omega_m == pytest.approx(0.250296, abs=1e-6)
    assert r.omnuh2 == pytest.approx(6.44867e-4, rel=1e-5)
    assert r.k_h.size == 194
    assert r.k_h[0] == pytest.approx(7.1429e-5, rel=1e-4)
    assert r.k_h[-1] == pytest.approx(30.0, rel=1e-6)
    assert r.pk_lin.shape == (150, 194) and r.pk_cb.shape == (150, 194)
    assert r.S8 == pytest.approx(0.779557 * np.sqrt(0.250296 / 0.3), rel=1e-6)
    assert r.omega_b == pytest.approx(0.022 / 0.49) and r.omega_c == pytest.approx(0.10 / 0.49)
    assert r.omega_m == pytest.approx((0.022 + 0.10 + r.omnuh2) / 0.49, rel=1e-12)
    assert r.omega_nu_frac == pytest.approx(r.omnuh2 / (0.49 * r.omega_m))


def test_vs_spec_anchor_test_cosmology(backend):
    """spec s4.1 anchors (anchors_test_cosmology.json): sigma8, S8, omega_m, chi, P_lin."""
    a = json.load(open(os.path.join(DATA, "anchors_test_cosmology.json")))["mead_A2.0"]
    r = backend.compute(**TESTCOSMO)
    assert r.sigma8_0 == pytest.approx(a["sigma8"], rel=1e-6)
    assert r.S8 == pytest.approx(a["S8"], rel=1e-6)
    assert r.omega_m == pytest.approx(a["omega_m"], rel=1e-6)
    assert r.k_h.size == a["nk"] and r.k_h[-1] == pytest.approx(a["kmax_h"], rel=1e-6)
    chi = CubicSpline(r.z_chi, r.chi)
    assert chi(0.5) == pytest.approx(a["chi_z0.5"], rel=1e-6)
    assert chi(1.0) == pytest.approx(a["chi_z1_Mpc_over_h"], rel=1e-6)
    # P_lin(z=0, k=0.1005): the anchor is the LINEAR-in-k interpolation between CAMB nodes (verified: it agrees
    # to ~1e-14; log-log linear and cubic-spline conventions differ from it by 2.5e-4 and 1e-3).
    assert np.interp(0.1005, r.k_h, r.pk_lin[0]) == pytest.approx(a["Plin_z0_k0.1"], rel=1e-9)


def test_pofk_linear_regime_anchor(backend):
    """VALIDATION L3: stored pk_mead (HMcode-2016 output, linear for k < ~0.004 h/Mpc) vs our P_lin.

    (At k=0.00968 HMcode already deviates from linear by -0.36 %, so only k < 0.004 is used here; the
    CAMB-version offset is ~3e-4.)"""
    ini = dict(omch2=0.1181838, ombh2=0.02228089, h=0.679565, n_s=0.9679701,
               ln_1e10_A_s=np.log(2.170229e-9 * 1e10))
    zf = np.loadtxt(os.path.join(DATA, "z.txt"))
    kf = np.loadtxt(os.path.join(DATA, "k_h.txt"))
    pf = np.loadtxt(os.path.join(DATA, "pk_mead.txt"))
    r = LinearBackend(nthreads=4, z_grid=zf).compute(**ini)
    m = (kf > r.k_h[0]) & (kf < 4e-3)
    for iz in (0, 40, 99):
        f = CubicSpline(np.log(r.k_h), np.log(r.pk_lin[iz]))
        d = np.exp(f(np.log(kf[m]))) / pf[iz, m] - 1
        assert np.max(np.abs(d)) < 1e-3


# ------------------------------------------------------------------------------------------ rescaling
def test_As_rescale_equals_fresh_camb(backend):
    """A_s change: cached rescale == fresh CAMB run to < 1e-6 (pk_lin, pk_cb, sigma8, S8)."""
    be = LinearBackend(nthreads=4)
    base = dict(omch2=0.12, ombh2=0.0222, h=0.69, n_s=0.97)
    r0 = be.compute(**base, ln_1e10_A_s=3.0)
    assert be.stats["camb_runs"] == 1
    r1 = be.compute(**base, ln_1e10_A_s=3.4)
    assert be.stats["camb_runs"] == 1, "A_s change must not trigger CAMB"
    p, c = fresh_camb(**base, ln_1e10_A_s=3.4, z=Z_GRID_SPEC)
    kh, z, pk = c.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    _, _, pcb = c.get_linear_matter_power_spectrum(var1="delta_nonu", var2="delta_nonu", hubble_units=True,
                                                   k_hunit=True)
    assert np.array_equal(kh, r1.k_h)
    assert np.max(np.abs(r1.pk_lin / pk - 1)) < 1e-6
    assert np.max(np.abs(r1.pk_cb / pcb - 1)) < 1e-6
    assert abs(r1.sigma8_0 / c.get_sigma8_0() - 1) < 1e-6
    assert r1.omega_m == pytest.approx(r0.omega_m, rel=0, abs=0)
    assert r1.pk_lin[3, 10] == pytest.approx(r0.pk_lin[3, 10] * np.exp(0.4), rel=1e-13)
    # r0 must not have been modified by the second call
    assert r0.pk_lin[0, 0] != r1.pk_lin[0, 0]


def test_ns_rescale_equals_fresh_camb():
    """n_s (+A_s) change: exact tilt (k/0.05)^(dn_s) -> fresh CAMB to <1e-6 incl. sigma8 (default key without n_s)."""
    be = LinearBackend(nthreads=4)
    base = dict(omch2=0.11, ombh2=0.0225, h=0.72)
    be.compute(**base, n_s=1.05, ln_1e10_A_s=3.2)
    r = be.compute(**base, n_s=0.88, ln_1e10_A_s=2.9)
    assert be.stats["camb_runs"] == 1
    p, c = fresh_camb(**base, n_s=0.88, ln_1e10_A_s=2.9, z=Z_GRID_SPEC)
    _, _, pk = c.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    assert np.max(np.abs(r.pk_lin / pk - 1)) < 1e-6
    assert abs(r.sigma8_0 / c.get_sigma8_0() - 1) < 1e-6
    # rescaled sigma8 consistent with the rescaled spectrum itself: A_s-only change afterwards
    r2 = be.compute(**base, n_s=0.88, ln_1e10_A_s=3.3)
    assert r2.sigma8_0 == pytest.approx(r.sigma8_0 * np.sqrt(np.exp(0.4)), rel=1e-12)
    # strict key of DESIGN.md: n_s in the key -> second n_s triggers CAMB
    be2 = LinearBackend(nthreads=4, key_on_ns=True)
    be2.compute(**base, n_s=1.05, ln_1e10_A_s=3.2)
    be2.compute(**base, n_s=1.05, ln_1e10_A_s=3.0)
    assert be2.stats["camb_runs"] == 1
    be2.compute(**base, n_s=0.88, ln_1e10_A_s=3.0)
    assert be2.stats["camb_runs"] == 2


@pytest.mark.timing
def test_rescale_is_fast(backend, lin_sanity):
    t = time.perf_counter()
    for i in range(20):
        backend.compute(**{**SANITY, "ln_1e10_A_s": 3.0 + 0.01 * i})
    assert (time.perf_counter() - t) / 20 < 0.02          # << 0.3 s CAMB run


def test_lru_eviction():
    be = LinearBackend(nthreads=4, cache_size=2, z_grid=Z_GRID_REDUCED)
    c = [dict(omch2=0.10 + 0.01 * i, ombh2=0.022, h=0.7, n_s=0.96, ln_1e10_A_s=3.0) for i in range(3)]
    for d in c:
        be.compute(**d)
    assert be.stats["camb_runs"] == 3 and len(be._cache) == 2
    be.compute(**c[2])
    assert be.stats["camb_runs"] == 3                     # still cached
    be.compute(**c[0])
    assert be.stats["camb_runs"] == 4                     # was evicted
    be.compute(**{**c[2], "mnu": 0.1})                    # mnu is part of the key
    assert be.stats["camb_runs"] == 5


# ------------------------------------------------------------------------------------------ distances
def test_distances_vs_scipy_quad(lin_sanity):
    """chi(z) = (c/H0) int dz/E vs scipy quad of 1/E(z), E = spline of the returned ez (independent)."""
    r = lin_sanity
    c_kms = 299792.458
    assert r.ez[0] == pytest.approx(1.0, abs=1e-14)
    E = CubicSpline(r.z, r.ez)
    for zz in (0.1, 0.5, 1.0, 1.9):
        ref = quad(lambda x: 1.0 / E(x), 0, zz, epsabs=0, epsrel=1e-12)[0] * c_kms / 100.0   # Mpc/h
        got = CubicSpline(r.z_chi, r.chi)(zz)
        assert got == pytest.approx(ref, rel=2e-7)
    ref6 = quad(lambda x: 1.0 / E(x), 0, 6.0, epsabs=0, epsrel=1e-12)[0] * c_kms / 100.0
    assert r.chi[-1] == pytest.approx(ref6, rel=1e-5)
    assert r.z_chi[0] == 0 and r.z_chi[-1] == 6 and r.z_chi.size == 6000
    assert np.all(np.diff(r.chi) > 0)
    # chi is in Mpc/h: low-z limit c z / 100
    assert r.chi[1] == pytest.approx(c_kms / 100.0 * Z_CHI[1], rel=2e-3)


def test_ez_vs_analytic_flat_lcdm(lin_sanity):
    """E(z)^2 = Om_m (1+z)^3 + Om_r (1+z)^4 + Om_L to the accuracy of treating nu as matter (spec s2.3)."""
    r = lin_sanity
    z = r.z
    a_m = r.omega_m
    # radiation: CMB photons + 3.044 massless-equivalent species, nu counted in omega_m at late times
    og = 2.4728e-5 * (2.7255 / 2.7255) ** 4 / r.h ** 2          # omega_gamma h^2 = 2.4728e-5 at T=2.7255 K
    Orad = og * (1 + 7 / 8 * (4 / 11) ** (4 / 3) * 3.044)
    E2 = a_m * (1 + z) ** 3 + Orad * (1 + z) ** 4 + (1 - a_m - Orad)
    # neutrino mass is only approximately 'matter' at z>~10; check z <= 3 to 5e-4
    sel = z <= 3
    assert np.max(np.abs(np.sqrt(E2[sel]) / r.ez[sel] - 1)) < 5e-4


# ------------------------------------------------------------------------------------------ cb spectrum
def test_pk_cb_identity():
    """pk_cb == ((wc d_c + wb d_b)/(wc+wb))^2 built from CAMB's delta_cdm / delta_baryon auto & cross spectra."""
    be = LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED)
    d = dict(omch2=0.12, ombh2=0.0222, h=0.7, n_s=0.97, ln_1e10_A_s=3.0)
    r = be.compute(**d)
    p, c = fresh_camb(**d, z=Z_GRID_REDUCED)
    g = lambda a, b: c.get_linear_matter_power_spectrum(var1=a, var2=b, hubble_units=True, k_hunit=True)[2]
    fc, fb = d["omch2"] / (d["omch2"] + d["ombh2"]), d["ombh2"] / (d["omch2"] + d["ombh2"])
    pcb = fc ** 2 * g("delta_cdm", "delta_cdm") + 2 * fc * fb * g("delta_cdm", "delta_baryon") \
        + fb ** 2 * g("delta_baryon", "delta_baryon")
    assert np.max(np.abs(r.pk_cb / pcb - 1)) < 1e-6


def test_pk_cb_vs_total(lin_sanity):
    """Massive-neutrino suppression: pk_cb/pk_lin -> 1 at low k, > 1 at high k (~1.0106 at k=30)."""
    r = lin_sanity
    ratio = r.pk_cb / r.pk_lin
    assert np.max(np.abs(ratio[:, 0] - 1)) < 1e-4
    assert 1.008 < ratio[0, -1] < 1.013                   # z=0, k=30 h/Mpc: measured 1.0106 (~0.75 f_nu)
    assert np.all(ratio[:, -1] > 1)
    assert np.all(ratio[0] > 1 - 1e-4)                    # cold spectrum never below total (beyond 1e-4 noise)
    # cold-matter sigma8 is larger by ~0.4 % (spec s2.4): approx via top-hat integral of the ratio
    R = 8.0
    k = r.k_h
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    integ = lambda pk: np.sqrt(trapezoid(k ** 3 * pk / (2 * np.pi ** 2) * W ** 2, np.log(k)))
    assert integ(r.pk_cb[0]) / integ(r.pk_lin[0]) - 1 == pytest.approx(0.004, abs=0.0015)
    assert integ(r.pk_lin[0]) == pytest.approx(r.sigma8_0, rel=1e-3)


def test_pk_physical_shape(lin_sanity):
    r = lin_sanity
    assert np.all(np.isfinite(r.pk_lin)) and np.all(r.pk_lin > 0)
    assert np.all(np.diff(r.z) > 0) and r.z[0] == 0 and r.z[-1] == 6
    # growth: P decreases with z at all k
    assert np.all(np.diff(r.pk_lin, axis=0) < 0)
    # large-scale growth factor D(z=1)/D(0) in a LCDM-like model ~ 0.6
    D = np.sqrt(r.pk_lin[np.searchsorted(r.z, 1.0), 0] / r.pk_lin[0, 0])
    assert 0.55 < D < 0.66
    # turnover near k_eq ~ 0.01-0.03 h/Mpc
    kpk = r.k_h[np.argmax(r.pk_lin[0])]
    assert 0.01 < kpk < 0.05
    # arrays are read-only shared views
    with pytest.raises(ValueError):
        r.k_h[0] = 1.0


# ------------------------------------------------------------------------------------------ grids / threads
def test_grids():
    assert Z_GRID_SPEC.size == 150 and Z_GRID_SPEC[0] == 0 and Z_GRID_SPEC[-1] == 6
    assert Z_GRID_SPEC[1] == pytest.approx(0.02) and Z_GRID_SPEC[99] == pytest.approx(1.98)
    assert Z_GRID_SPEC[100] == pytest.approx(2.0) and np.diff(Z_GRID_SPEC)[-1] == pytest.approx(4 / 49)
    assert Z_GRID_REDUCED.size == 41 and np.all(np.diff(Z_GRID_REDUCED) > 0)
    assert Z_GRID_REDUCED[0] == 0 and Z_GRID_REDUCED[-1] == 6
    assert 1 <= default_nthreads() <= 4
    with pytest.raises(ValueError):
        LinearBackend(z_grid=[0.1, 0.5])
    with pytest.raises(ValueError):
        LinearBackend(z_grid=[0.0, 0.5, 0.4])


def test_reduced_grid_matches_full_grid(lin_sanity):
    """The 41-node grid gives the same P(k,z) at shared redshifts, and cubic-in-z interpolation is accurate."""
    be = LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED)
    r = be.compute(**SANITY)
    assert r.pk_lin.shape == (41, 194)
    assert r.sigma8_0 == pytest.approx(lin_sanity.sigma8_0, rel=1e-9)
    idx_full = np.array([np.argmin(np.abs(lin_sanity.z - z)) for z in r.z])
    shared = np.abs(lin_sanity.z[idx_full] - r.z) < 1e-12
    assert shared.sum() >= 15
    assert np.max(np.abs(r.pk_lin[shared] / lin_sanity.pk_lin[idx_full[shared]] - 1)) < 1e-6
    # interpolating ln P in z from the reduced grid reproduces the full grid for z <= 3
    f = CubicSpline(r.z, np.log(r.pk_lin), axis=0)
    sel = lin_sanity.z <= 3
    err = np.max(np.abs(np.exp(f(lin_sanity.z[sel])) / lin_sanity.pk_lin[sel] - 1))
    assert err < 2e-4


_THREAD_COSMO = dict(omch2=0.13, ombh2=0.0221, h=0.68, n_s=0.99, ln_1e10_A_s=3.0)


@pytest.mark.slow
def test_thread_independence():
    """Results identical for 1 and 4 threads."""
    r1 = LinearBackend(nthreads=1).compute(**_THREAD_COSMO)
    r4 = LinearBackend(nthreads=4).compute(**_THREAD_COSMO)
    assert np.max(np.abs(r1.pk_lin / r4.pk_lin - 1)) < 1e-10


@pytest.mark.slow
@pytest.mark.timing
def test_thread_speed():
    """4 threads at least 1.5x faster than 1 (needs >= 4 free cores; skipped on small machines)."""
    b1 = LinearBackend(nthreads=1)
    t = time.perf_counter(); b1.compute(**_THREAD_COSMO); t1 = time.perf_counter() - t
    b4 = LinearBackend(nthreads=4)
    t = time.perf_counter(); b4.compute(**_THREAD_COSMO); t4 = time.perf_counter() - t
    assert t1 / t4 > 1.5


# ------------------------------------------------------------------------------------------ independent reviewer checks
def _sigma_R(k_h, pk, R=8.0):
    """sigma(R) by adaptive quadrature of the top-hat integral over a cubic-spline of ln P(ln k) (spec s2.4)."""
    lk = np.log(k_h)
    f = CubicSpline(lk, np.log(pk))

    def integrand(x):
        k = np.exp(x)
        y = k * R
        W = 3 * (np.sin(y) - y * np.cos(y)) / y ** 3
        return k ** 3 * np.exp(f(x)) / (2 * np.pi ** 2) * W ** 2
    return np.sqrt(quad(integrand, lk[0], lk[-1], limit=1000, epsrel=1e-10)[0])


def test_sigma8_brute_force_is_total_matter_in_h_units(lin_sanity):
    """sigma8_0 == top-hat integral (R = 8 Mpc/h) of pk_lin in (Mpc/h)^3 vs h/Mpc units (fixes h^3 and 2 pi^2
    factors) and it is the TOTAL-matter value: the cold-matter integral is 0.4-0.5 % larger (spec s2.4)."""
    r = lin_sanity
    s_tot, s_cb = _sigma_R(r.k_h, r.pk_lin[0]), _sigma_R(r.k_h, r.pk_cb[0])
    assert s_tot == pytest.approx(r.sigma8_0, rel=6e-4)          # coarse 194-node spline limits to ~3e-4
    assert abs(s_cb / r.sigma8_0 - 1) > 3.5e-3
    assert s_cb / s_tot - 1 == pytest.approx(0.0045, abs=0.001)
    # growth: sigma(8) at z=1 scales as D(1)/D(0) of the large-scale modes within the scale-dependence of growth
    i1 = np.searchsorted(r.z, 1.0)
    assert _sigma_R(r.k_h, r.pk_lin[i1]) / s_tot == pytest.approx(
        np.sqrt(r.pk_lin[i1, 0] / r.pk_lin[0, 0]), rel=0.02)


def test_z_axis_independent_single_redshift_run():
    """Row iz of pk_lin is P(k, z[iz]): compare with a CAMB run that has ONE redshift (no ordering ambiguity)."""
    d = dict(omch2=0.1181838, ombh2=0.02228089, h=0.679565, n_s=0.9679701, ln_1e10_A_s=3.07)
    r = LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED).compute(**d)
    for zt in (0.5, 2.0):
        p = camb.set_params(H0=100 * d["h"], ombh2=d["ombh2"], omch2=d["omch2"], ns=d["n_s"],
                            As=np.exp(d["ln_1e10_A_s"]) * 1e-10, mnu=0.06, tau=0.0)
        p.WantCls = False
        p.set_matter_power(redshifts=[zt], kmax=20.0)
        c = camb.get_results(p)
        kh, _, pk = c.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
        iz = int(np.argmin(np.abs(r.z - zt)))
        assert r.z[iz] == pytest.approx(zt, abs=1e-12)
        assert np.max(np.abs(pk[0] / r.pk_lin[iz] - 1)) < 1e-5
        assert np.max(np.abs(kh / r.k_h - 1)) < 1e-6


def test_background_independent_density_budget():
    """E(z) and chi(z) from an explicit density budget (photons, massless nu, EXACT Fermi-Dirac massive nu,
    Lambda from flatness) with scipy quad, independent of CAMB's h_of_z / comoving_radial_distance; 3e-5."""
    d = dict(omch2=0.1181838, ombh2=0.02228089, h=0.679565, n_s=0.9679701, ln_1e10_A_s=3.07)
    r = LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED).compute(**d)
    p = camb.set_params(H0=100 * d["h"], ombh2=d["ombh2"], omch2=d["omch2"], mnu=0.06)
    c = camb.get_results(p)
    og, onr = c.get_Omega("photon", 0), c.get_Omega("neutrino", 0)
    Ob = d["ombh2"] / d["h"] ** 2 + d["omch2"] / d["h"] ** 2
    assert Ob + r.omnuh2 / d["h"] ** 2 == pytest.approx(r.omega_m, rel=1e-12)
    onu1 = 7 / 8 * (4 / 11) ** (4 / 3) * og                       # one massless species today
    deg = p.nu_mass_degeneracies[0]
    y0 = 0.06 / (8.617333262e-5 * 2.7255 * (4 / 11) ** (1 / 3))   # m / (k_B T_nu)
    fd = lambda y: quad(lambda x: x * x * np.sqrt(x * x + y * y) / (np.exp(x) + 1), 0, 60, epsabs=0, epsrel=1e-12)[0]
    f0 = fd(0.0)
    onm = lambda a: deg * onu1 * fd(y0 * a) / f0 / a ** 4
    ode = 1 - Ob - og - onr - onm(1.0)

    def E(z):
        a = 1 / (1 + z)
        return np.sqrt(Ob / a ** 3 + (og + onr) / a ** 4 + onm(a) + ode)
    Ez = np.array([E(z) for z in r.z])
    assert np.max(np.abs(r.ez / Ez - 1)) < 3e-5
    cs = CubicSpline(r.z_chi, r.chi)
    for zz in (0.3, 1.0, 3.0, 6.0):
        ref = quad(lambda x: 299792.458 / 100.0 / E(x), 0, zz, epsabs=0, epsrel=1e-11)[0]     # Mpc/h
        assert cs(zz) == pytest.approx(ref, rel=3e-5)


@pytest.mark.parametrize("bad", [
    dict(ln_1e10_A_s=float("nan")), dict(ln_1e10_A_s=float("inf")), dict(ln_1e10_A_s=-900.0),
    dict(n_s=float("nan")), dict(n_s=float("inf")), dict(h=float("nan")), dict(h=0.0), dict(h=-0.7),
    dict(omch2=float("nan")), dict(omch2=-0.1), dict(omch2=0.0), dict(ombh2=float("nan")), dict(ombh2=0.0),
    dict(ombh2=-0.01), dict(mnu=float("nan")), dict(mnu=-1.0), dict(n_s=1e4), dict(n_s=-11.0)])
def test_invalid_inputs_raise_and_do_not_poison_cache(bad):
    """NaN/negative densities used to crash CAMB (heap corruption / segfault), NaN or negative mnu was silently
    treated as 0, and a NaN A_s on the FIRST call of a cosmology was stored as that cosmology's reference
    amplitude, so every later (valid) call returned NaN."""
    be = LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED)
    good = dict(omch2=0.12, ombh2=0.0222, h=0.7, n_s=0.97, ln_1e10_A_s=3.0)
    with pytest.raises(ValueError):
        be.compute(**{**good, **bad})
    assert len(be._cache) == 0 or be.stats["camb_runs"] == 0 or all(
        np.all(np.isfinite(e.pk_ref)) and np.isfinite(e.As_ref) for e in be._cache.values())
    r = be.compute(**good)                       # the same cosmology afterwards is healthy
    assert np.all(np.isfinite(r.pk_lin)) and np.isfinite(r.sigma8_0) and 0.5 < r.sigma8_0 < 1.2


def test_determinism_and_no_state_leak_between_calls():
    """Same inputs -> bit-identical outputs, independent of call history (A_s / n_s walks, other cosmologies)."""
    be = LinearBackend(nthreads=4, z_grid=Z_GRID_REDUCED)
    a = dict(omch2=0.12, ombh2=0.0222, h=0.7, n_s=0.97, ln_1e10_A_s=3.0)
    r1 = be.compute(**a)
    be.compute(**{**a, "n_s": 0.9, "ln_1e10_A_s": 2.5})
    be.compute(**{**a, "omch2": 0.13})
    r2 = be.compute(**a)
    assert np.array_equal(r1.pk_lin, r2.pk_lin) and np.array_equal(r1.pk_cb, r2.pk_cb)
    assert r1.sigma8_0 == r2.sigma8_0 and r1.S8 == r2.S8
    r2.pk_lin[:] = 0                              # caller scribbling must not corrupt the cache
    assert np.array_equal(r1.pk_lin, be.compute(**a).pk_lin)
