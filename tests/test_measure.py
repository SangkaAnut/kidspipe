"""Fast synthetic tests for kidspipe.measure (xi, paircount, xi2cosebis, official) and the chunk fold rule
of scripts/dr5_measure_xi_chunked.py.  No dependence on the 40M-row catalogue; tests that need the official
release skip cleanly when it is absent.  Every historical bug listed in runs/dr5_measure/RESUME.md and
CHUNKING.md has at least one test that fails if it is reintroduced (table: runs/dr5_measure/tests_measure/
TEST_MEASURE.md).  TreeCorr is run with <= 2 threads."""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

treecorr = pytest.importorskip("treecorr")

from kidspipe.measure import official, paircount, xi as xim_mod, xi2cosebis  # noqa: E402
from kidspipe.theory.cosebis import CosebisFilters, E_from_xi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = 12          # coarse log bins 2-300 arcmin for the synthetic tests
THREADS = 2
HAVE_RELEASE = (official.OFFICIAL / "KiDS_Legacy_xipm.fits").exists()
needs_release = pytest.mark.skipif(not HAVE_RELEASE, reason="official KiDS-Legacy release files not present")


# --------------------------------------------------------------------------------------------- synthetic fields
def lens_field(n, side, seed, ra0=150.0, dec0=0.0):
    """n points in a side x side deg patch, shear = sum of three tangential 'lenses' (sky convention:
    g = -A exp(2 i phi_c), phi_c measured from +RA towards +Dec).  Gives xi_+ >> 0 and xi_- clearly != xi_+."""
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(0, side, n)
    dec = dec0 + rng.uniform(0, side, n)
    g = np.zeros(n, complex)
    for fx, fy in [(0.2, 0.2), (0.7, 0.8), (0.5, 0.14)]:
        dx = (ra - (ra0 + fx * side)) * np.cos(np.deg2rad(dec))
        dy = dec - (dec0 + fy * side)
        g += -0.3 * (0.1 * side / (np.hypot(dx, dy) + 0.01 * side)) * np.exp(2j * np.arctan2(dy, dx))
    g = np.clip(g.real, -0.9, 0.9) + 1j * np.clip(g.imag, -0.9, 0.9)
    return dict(ra=ra, dec=dec, e1=g.real.copy(), e2=g.imag.copy(), w=rng.uniform(0.5, 1.5, n))


def tc_run(rec1, rec2=None, nbins=NB, flip_g2=True, ordered_auto=False):
    """TreeCorr through the production wrappers.  The catalogue's sky-frame shear needs flip_g2=True to match
    PairCounter's (RA,Dec)-frame phi (xi_+ is unaffected).  ordered_auto: process(cat, cat) (ordered pairs)."""
    c1 = xim_mod.make_treecorr_cat(rec1, flip_g2=flip_g2)
    if rec2 is None and not ordered_auto:
        gg = treecorr.GGCorrelation(xim_mod.treecorr_kwargs(nbins, 0.0, THREADS))
        gg.process(c1)
    else:
        c2 = c1 if rec2 is None else xim_mod.make_treecorr_cat(rec2, flip_g2=flip_g2)
        gg = xim_mod.run_pair(c1, c2, nbins=nbins, bin_slop=0.0, num_threads=THREADS)
    return xim_mod.corr_to_arrays(gg)


def pc_of(rec, nbins=NB):
    return paircount.PairCounter(rec["ra"], rec["dec"], rec["e1"], rec["e2"], rec["w"], nbins=nbins)


def brute(rec, rec2=None, nbins=NB, conj=False):
    """O(N^2) reference with full N x N matrices: haversine separation, flat (dRA cos dec, dDec) position angle,
    complex-shear products.  Auto: each unordered pair once; cross: every (i in rec, j in rec2).  conj=True builds
    the BUGGY xi_- (gA*conj(gB)) so tests can show the field discriminates."""
    A, B = rec, (rec if rec2 is None else rec2)
    ra1, de1 = np.deg2rad(A["ra"])[:, None], np.deg2rad(A["dec"])[:, None]
    ra2, de2 = np.deg2rad(B["ra"])[None, :], np.deg2rad(B["dec"])[None, :]
    h = np.sin((de2 - de1) / 2) ** 2 + np.cos(de1) * np.cos(de2) * np.sin((ra2 - ra1) / 2) ** 2
    th = np.rad2deg(2 * np.arcsin(np.sqrt(h))) * 60.0
    dra = (B["ra"][None, :] - A["ra"][:, None] + 180) % 360 - 180
    ddec = B["dec"][None, :] - A["dec"][:, None]
    phi = np.arctan2(ddec, dra * np.cos(0.5 * (de1 + de2)))
    gA = (A["e1"] + 1j * A["e2"])[:, None] * np.exp(-2j * phi)
    gB = (B["e1"] + 1j * B["e2"])[None, :] * np.exp(-2j * phi)
    gAr, gBr = (A["e1"] + 1j * A["e2"])[:, None], (B["e1"] + 1j * B["e2"])[None, :]
    wij = A["w"][:, None] * B["w"][None, :]
    keep = np.ones_like(th, bool)
    if rec2 is None:
        keep = np.triu(np.ones_like(th, bool), 1)
    edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), nbins + 1))
    prod_m = gA * (np.conj(gB) if conj else gB)
    prod_p = (gAr * np.conj(gBr)).real
    out = {k: np.zeros(nbins) for k in ("np", "w", "p", "m", "x")}
    for k in range(nbins):
        m = keep & (th >= edges[k]) & (th < edges[k + 1])
        out["np"][k] = m.sum()
        out["w"][k] = wij[m].sum()
        out["p"][k] = (wij * prod_p)[m].sum()
        out["m"][k] = (wij * prod_m.real)[m].sum()
        out["x"][k] = (wij * prod_m.imag)[m].sum()
    good = out["w"] > 0
    for k in ("p", "m", "x"):
        out[k] = np.where(good, out[k] / np.where(good, out["w"], 1), 0.0)
    return out


def is_integral(x):
    """Integrality test used for pair counts: floor(x)==x.  np.allclose is meaningless at 1e10 (see RESUME bug 4)."""
    x = np.asarray(x, float)
    return bool(np.all(np.floor(x) == x))


@pytest.fixture(scope="module")
def dense():
    """2500 points in a 5x5 deg patch; PairCounter and TreeCorr (unordered auto) results."""
    rec = lens_field(2500, 5.0, seed=1)
    a = tc_run(rec)
    xp, xm, xx, n, w = pc_of(rec).run()
    return dict(rec=rec, tc=a, pc=dict(xip=xp, xim=xm, xix=xx, npairs=n, weight=w))


@pytest.fixture(scope="module")
def small():
    """250 points in 2x2 deg, for brute-force comparisons."""
    return lens_field(250, 2.0, seed=7)


# --------------------------------------------------------------------------------------------- xi.py
def _bin_files(tmp_path, nbin=6):
    for k in range(nbin):          # file bin{k}.npz = tomographic bin k+1; contents identify it
        n = 5 + k
        np.savez(tmp_path / f"bin{k}.npz", ra=np.full(n, 10.0 * (k + 1)), dec=np.zeros(n),
                 e1=np.full(n, float(k + 1)), e2=np.zeros(n), w=np.ones(n))


@pytest.mark.parametrize("b", [1, 2, 3, 4, 5, 6])
def test_load_bin_is_one_based_not_off_by_one(tmp_path, b):
    """RESUME bug 1: tomographic bin b lives in bin{b-1}.npz (zero-based files).  Reading bin{b} returned b+1."""
    _bin_files(tmp_path)
    d = xim_mod.load_bin(b, work=tmp_path)
    assert np.all(d["e1"] == b), f"load_bin({b}) returned the content of tomographic bin {d['e1'][0]:.0f}"
    assert d["e1"].size == 5 + (b - 1)


def test_load_bin_out_of_range_raises(tmp_path):
    _bin_files(tmp_path)
    for bad in (0, 7):        # 0 -> bin-1.npz, 7 -> bin6.npz: neither exists; must raise, not wrap or shift
        with pytest.raises(FileNotFoundError):
            xim_mod.load_bin(bad, work=tmp_path)


def test_load_bin_default_work_dir_constant():
    from kidspipe import paths
    assert xim_mod.WORK == paths.dr5_work_dir()


def test_treecorr_kwargs_declare_arcmin_and_binning():
    """RESUME bug 3: without sep_units='arcmin' every bin is silently empty."""
    kw = xim_mod.treecorr_kwargs()
    assert kw["sep_units"] == "arcmin"
    assert (kw["nbins"], kw["min_sep"], kw["max_sep"], kw["bin_type"]) == (1000, 2.0, 300.0, "Log")
    assert kw["bin_slop"] == 0.05 and kw["num_threads"] == 16
    assert xim_mod.treecorr_kwargs(nbins=7, bin_slop=0.0, num_threads=3)["nbins"] == 7


def test_no_empty_bins_with_arcmin_units(dense):
    """RESUME bug 3 (behavioural): a dense patch must populate every 2-300' bin; omitting sep_units empties them."""
    n = dense["tc"]["npairs"]
    assert np.all(n > 0), n
    assert n.min() > 50
    # and meanr must lie inside the bin in arcmin (not radians / degrees)
    edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), NB + 1))
    mr = dense["tc"]["meanr"]
    assert np.all((mr >= edges[:-1]) & (mr <= edges[1:]))


def test_tc_without_sep_units_would_be_empty(dense):
    """Documents the trap: same call minus sep_units gives (near-)empty bins on this patch."""
    kw = xim_mod.treecorr_kwargs(NB, 0.0, THREADS)
    del kw["sep_units"]
    gg = treecorr.GGCorrelation(kw)
    gg.process(xim_mod.make_treecorr_cat(dense["rec"]))
    assert gg.npairs.sum() < 0.01 * dense["tc"]["npairs"].sum()


def test_r_nom_is_geometric_centre_of_1000_log_bins():
    """RESUME 'binning' row: r_nom = sqrt(e_k e_k+1) of 1000 log bins 2-300 arcmin."""
    e = np.exp(np.linspace(np.log(2.0), np.log(300.0), 1001))
    geo = np.sqrt(e[:-1] * e[1:])
    gg = treecorr.GGCorrelation(xim_mod.treecorr_kwargs(1000, 0.0, THREADS))
    assert gg.rnom.shape == (1000,)
    np.testing.assert_allclose(gg.rnom, geo, rtol=1e-12, atol=0)
    np.testing.assert_allclose(official.fine_bin_centres(), geo, rtol=1e-13, atol=0)
    arith = 0.5 * (e[:-1] + e[1:])
    assert np.max(np.abs(arith / geo - 1)) > 1e-6          # the arithmetic centre is a different, wrong thing
    assert abs(geo[0] / 2.0 - 1) > 1e-3 and geo[0] > 2.0 and geo[-1] < 300.0
    np.testing.assert_allclose(paircount._log_edges(2.0, 300.0, 1000), e, rtol=1e-13)
    np.testing.assert_allclose(xi2cosebis.bin_edges(geo), e, rtol=1e-12)    # edges recovered from the centres


def test_corr_to_arrays_schema(dense):
    a = dense["tc"]
    assert set(a) == {"r_nom", "meanr", "xip", "xim", "xip_im", "xim_im", "varxi", "npairs", "weight"}
    assert all(v.shape == (NB,) and v.dtype == float for v in a.values())


def test_make_treecorr_cat_roundtrip(small):
    c = xim_mod.make_treecorr_cat(small)
    assert c.nobj == small["ra"].size
    np.testing.assert_allclose(c.g1, small["e1"])
    c2 = xim_mod.make_treecorr_cat(small, flip_g2=True)
    np.testing.assert_allclose(c2.g2, -small["e2"])
    c3 = xim_mod.make_treecorr_cat(small, g1=small["e2"], g2=small["e1"])       # override arrays
    np.testing.assert_allclose(c3.g1, small["e2"])


# --------------------------------------------------------------------------------------------- paircount.py
def test_unit_vectors_and_edges():
    v = paircount.unit_vectors([0, 90, 0], [0, 0, 90])
    np.testing.assert_allclose(v, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-15)
    assert abs(paircount.ARCMIN_PER_RAD - 3437.7467707849396) < 1e-9


def test_paircounter_matches_bruteforce_auto(small):
    ref = brute(small)
    xp, xm, xx, n, w = pc_of(small).run()
    assert (n > 0).sum() >= 8
    np.testing.assert_array_equal(n, ref["np"])
    np.testing.assert_allclose(w, ref["w"], rtol=1e-12)
    np.testing.assert_allclose(xp, ref["p"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xm, ref["m"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xx, ref["x"], rtol=1e-9, atol=1e-12)


def test_paircounter_matches_bruteforce_cross(small):
    other = lens_field(200, 2.0, seed=11)
    ref = brute(small, other)
    xp, xm, xx, n, w = pc_of(small).run(other=pc_of(other))
    np.testing.assert_array_equal(n, ref["np"])
    np.testing.assert_allclose(xm, ref["m"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xp, ref["p"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xx, ref["x"], rtol=1e-9, atol=1e-12)


def test_xim_is_not_xip_phase_bug(small, dense):
    """RESUME bug 2: xi_- = Re[gA*gB e^{-4i phi}] (no conjugate).  With gA*conj(gB) the phases cancel and the
    estimator returns xi_+ (and xi_x -> 0).  The field must discriminate, and the counter must follow the right one."""
    ref_ok, ref_bug = brute(small), brute(small, conj=True)
    # the buggy variant IS xi_+ and the synthetic field makes xi_+ and xi_- clearly different
    np.testing.assert_allclose(ref_bug["m"], ref_bug["p"], rtol=1e-9, atol=1e-12)
    good = ref_ok["np"] > 0
    assert np.max(np.abs(ref_ok["m"] - ref_ok["p"])[good]) > 0.2 * np.max(np.abs(ref_ok["p"][good]))
    _, xm, xx, _, _ = pc_of(small).run()
    xp = pc_of(small).run()[0]
    assert np.max(np.abs(xm - xp)[good]) > 0.2 * np.max(np.abs(xp[good]))         # not recomputing xi_+
    assert np.max(np.abs(xm - ref_bug["m"])[good]) > 0.2 * np.max(np.abs(xp[good]))
    # and on the dense patch: xi_- agrees with TreeCorr at r > 0.999, xi_+ does not stand in for it
    p, a = dense["pc"], dense["tc"]
    assert np.corrcoef(p["xim"], a["xim"])[0, 1] > 0.999
    assert np.corrcoef(p["xim"], p["xip"])[0, 1] < 0.9 or np.max(np.abs(p["xim"] - p["xip"])) > 0.2 * np.max(np.abs(p["xip"]))


def test_paircounter_vs_treecorr_dense(dense):
    """A4: exact counts, xi_+ / xi_- / xi_x agree with TreeCorr (bin_slop=0) at r > 0.999."""
    p, a = dense["pc"], dense["tc"]
    np.testing.assert_allclose(p["npairs"], a["npairs"], rtol=5e-3)          # bin-edge ties from the chord/arc maths
    assert np.corrcoef(p["xip"], a["xip"])[0, 1] > 0.999
    assert np.corrcoef(p["xim"], a["xim"])[0, 1] > 0.999
    assert np.corrcoef(p["xix"], -a["xim_im"])[0, 1] > 0.99      # flip_g2 (frame handedness) flips the sign of xi_x
    np.testing.assert_allclose(p["xip"], a["xip"], atol=2e-3 * np.abs(a["xip"]).max())
    np.testing.assert_allclose(p["xim"], a["xim"], atol=2e-3 * np.abs(a["xip"]).max())
    assert np.max(np.abs(a["xim"] - a["xip"])) > 0.2 * np.max(np.abs(a["xip"]))      # discriminating field


def test_pair_counts_are_integers_floor_test(dense):
    """RESUME bug 4: integrality via floor(x)==x.  np.allclose cannot tell 1e10 from 1e10+0.4."""
    assert is_integral(dense["pc"]["npairs"]) and is_integral(dense["tc"]["npairs"])
    assert not is_integral(dense["tc"]["weight"])                                   # random weights are not integers
    big = np.array([3.0e10, 1.0e10 + 0.4])
    assert np.allclose(big, np.floor(big))            # allclose is blind ...
    assert not is_integral(big)                       # ... floor(x)==x is not
    assert np.all(dense["tc"]["npairs"] == np.round(dense["tc"]["npairs"]))


def test_ordered_vs_unordered_pair_counting(small):
    """Auto: PairCounter and process(cat) count each UNORDERED pair once; process(cat, cat) counts ORDERED pairs
    (factor 2, same weighted xi).  Brute force on 250 points."""
    ref = brute(small)
    unord = tc_run(small)
    ordd = tc_run(small, ordered_auto=True)
    n_pc = pc_of(small).run()[3]
    np.testing.assert_array_equal(n_pc, ref["np"])                      # PairCounter == brute force, exactly
    np.testing.assert_allclose(unord["npairs"], ref["np"], atol=3, rtol=1e-3)   # TreeCorr differs only on bin-edge ties
    np.testing.assert_array_equal(ordd["npairs"], 2 * unord["npairs"])   # ordered = exactly 2 x unordered
    np.testing.assert_allclose(ordd["weight"], 2 * unord["weight"], rtol=1e-12)
    assert ref["np"].sum() == small["ra"].size * (small["ra"].size - 1) / 2 - (brute_outside(small))
    np.testing.assert_allclose(ordd["xip"], unord["xip"], rtol=1e-10, atol=1e-14)


def brute_outside(rec):
    """Number of unordered pairs outside the 2-300' range (brute force)."""
    A = rec
    ra, de = np.deg2rad(A["ra"]), np.deg2rad(A["dec"])
    h = (np.sin((de[None, :] - de[:, None]) / 2) ** 2
         + np.cos(de)[:, None] * np.cos(de)[None, :] * np.sin((ra[None, :] - ra[:, None]) / 2) ** 2)
    th = np.rad2deg(2 * np.arcsin(np.sqrt(h))) * 60
    iu = np.triu_indices(ra.size, 1)
    t = th[iu]
    return int(((t < 2.0) | (t >= 300.0)).sum())


def test_cross_pairs_have_no_factor_two(small):
    """Cross pairs: TreeCorr counts every (i in A, j in B) once, = PairCounter(A).run(B) = brute force; no fold."""
    other = lens_field(200, 2.0, seed=11)
    ref = brute(small, other)
    a = tc_run(small, other)
    np.testing.assert_allclose(a["npairs"], ref["np"], atol=3, rtol=1e-3)       # not 2x: no fold for cross pairs
    assert abs(a["npairs"].sum() / ref["np"].sum() - 1) < 1e-3
    np.testing.assert_array_equal(pc_of(small).run(other=pc_of(other))[3], ref["np"])
    b = tc_run(other, small)                                   # symmetric in the counts
    np.testing.assert_array_equal(b["npairs"], a["npairs"])


def test_paircounter_chunking_independent(small):
    """The query chunk size is an implementation detail: counts must not change."""
    a = pc_of(small).run(chunk=17)
    b = pc_of(small).run(chunk=200_000)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-15)


def test_paircounter_ra_wraparound():
    """Points either side of RA=0/360 are neighbours (dRA wrapped to [-180,180))."""
    ra = np.array([359.9, 0.1]); dec = np.zeros(2)
    xp, xm, xx, n, w = paircount.PairCounter(ra, dec, [0.1, 0.2], [0.0, 0.3], [1.0, 2.0], nbins=12).run()
    assert n.sum() == 1 and abs(w.sum() - 2.0) < 1e-12
    assert abs(xp[n > 0][0] - (0.1 * 0.2 + 0.0 * 0.3)) < 1e-12


def test_paircounter_wraparound_field_xim_xix_match_bruteforce():
    """Added by the verifier: a mutant without the dRA wrap in the position angle survived the 2-point test above
    because xi_+ does not depend on phi.  A patch straddling RA=0/360 must give xi_-/xi_x equal to the (wrapping)
    brute force."""
    rec = lens_field(250, 2.0, seed=5, ra0=359.0)
    rec["ra"] = rec["ra"] % 360.0
    assert (rec["ra"] < 1.0).any() and (rec["ra"] > 359.0).any()
    ref = brute(rec)
    xp, xm, xx, n, w = pc_of(rec).run()
    good = ref["np"] > 0
    assert good.sum() >= 8
    np.testing.assert_array_equal(n, ref["np"])
    np.testing.assert_allclose(xp, ref["p"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xm, ref["m"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(xx, ref["x"], rtol=1e-9, atol=1e-12)
    assert np.max(np.abs(ref["m"] - ref["p"])[good]) > 0.2 * np.max(np.abs(ref["p"][good]))   # field discriminates


# --------------------------------------------------------------------------------------------- chunked driver
def _load_script():
    spec = importlib.util.spec_from_file_location("dr5_measure_xi_chunked", ROOT / "scripts" / "dr5_measure_xi_chunked.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return _load_script()


def test_fold_factor_and_chunk_pair_lists(drv):
    assert drv.fold_factor(1, 1, True) == 1.0
    assert drv.fold_factor(0, 3, True) == 2.0
    assert drv.fold_factor(3, 0, True) == 2.0
    assert all(drv.fold_factor(a, b, False) == 1.0 for a in range(4) for b in range(4))      # cross: never folded
    K = 5
    assert len(drv.chunk_pairs(K, True)) == K * (K + 1) // 2 and all(a <= b for a, b in drv.chunk_pairs(K, True))
    assert len(drv.chunk_pairs(K, False)) == K * K
    assert drv.NLETTER[1] == 7416371 and len(drv.PAIRS) == 21


def test_chunk_masks_partition_and_balance(drv):
    rng = np.random.default_rng(3)
    ra = rng.uniform(0, 360, 10_001)
    ra[:5] = ra.max()                    # include ties at the top edge
    m = drv.chunk_masks(ra, 7)
    tot = np.sum(m, axis=0)
    assert np.all(tot >= 1)             # every galaxy in a chunk (last chunk is closed)
    sizes = np.array([x.sum() for x in m])
    assert sizes.sum() >= ra.size and sizes.max() - sizes.min() <= 12     # equal-count
    assert drv.chunk_masks(ra, 7)[2].tolist() == m[2].tolist()              # deterministic -> reproducible ids


def _chunk_setup(drv, rec_a, rec_b, K, nb=10):
    cats_a = [xim_mod.make_treecorr_cat({k: rec_a[k][m] for k in ("ra", "dec", "e1", "e2", "w")})
              for m in drv.chunk_masks(rec_a["ra"], K)]
    cats_b = [xim_mod.make_treecorr_cat({k: rec_b[k][m] for k in ("ra", "dec", "e1", "e2", "w")})
              for m in drv.chunk_masks(rec_b["ra"], K)]
    return cats_a, cats_b


def _write_chunks(drv, outroot, b1, b2, cats1, cats2, K, nb, skip_fold=False):
    same = b1 == b2
    d = outroot / "chunks" / f"Bin{b1}_Bin{b2}"
    d.mkdir(parents=True)
    for (a, b) in drv.chunk_pairs(K, same):
        gg = xim_mod.run_pair(cats1[a], cats2[b], nbins=nb, bin_slop=0.0, num_threads=THREADS)
        arr = xim_mod.corr_to_arrays(gg)
        arr["fold"] = np.float64(1.0 if skip_fold else drv.fold_factor(a, b, same))
        np.savez(d / f"c{a:02d}_{b:02d}.npz", **arr)
    return d


@pytest.fixture(scope="module")
def chunk_case(drv, tmp_path_factory):
    """Bin1 (auto) and Bin1xBin2 (cross) on 600+700 lens-field points in a 3x3 deg patch, K=3 RA strips."""
    K, nb = 3, 10
    r1 = lens_field(600, 3.0, seed=21)
    r2 = lens_field(700, 3.0, seed=22)
    c1, c2 = _chunk_setup(drv, r1, r2, K)
    full1 = xim_mod.corr_to_arrays(xim_mod.run_pair(xim_mod.make_treecorr_cat(r1), xim_mod.make_treecorr_cat(r1),
                                                    nbins=nb, bin_slop=0.0, num_threads=THREADS))
    full12 = xim_mod.corr_to_arrays(xim_mod.run_pair(xim_mod.make_treecorr_cat(r1), xim_mod.make_treecorr_cat(r2),
                                                     nbins=nb, bin_slop=0.0, num_threads=THREADS))
    root = tmp_path_factory.mktemp("chunked")
    _write_chunks(drv, root, 1, 1, c1, c1, K, nb)
    _write_chunks(drv, root, 1, 2, c1, c2, K, nb)
    bad = tmp_path_factory.mktemp("chunked_nofold")
    _write_chunks(drv, bad, 1, 1, c1, c1, K, nb, skip_fold=True)
    return dict(K=K, nb=nb, full1=full1, full12=full12, root=root, bad=bad, c1=c1, r1=r1)


def _assemble(drv, root, K, monkeypatch):
    monkeypatch.setattr(drv, "OUTROOT", root)
    drv.do_assemble(argparse.Namespace(nchunks=K))
    return root / "assembled" / "ns"


def test_fold_rule_process_cat_cat(chunk_case, drv):
    """CHUNKING s1: process(cat,cat) == sum_c process(c,c) + 2 sum_{a<b} process(a,b), exactly (integers)."""
    K, cs, full = chunk_case["K"], chunk_case["c1"], chunk_case["full1"]
    n = np.zeros_like(full["npairs"]); w = np.zeros_like(n); n_nofold = np.zeros_like(n)
    for a in range(K):
        for b in range(a, K):
            g = xim_mod.corr_to_arrays(xim_mod.run_pair(cs[a], cs[b], nbins=chunk_case["nb"], bin_slop=0.0,
                                                        num_threads=THREADS))
            f = 1.0 if a == b else 2.0
            n += f * g["npairs"]; w += f * g["weight"]; n_nofold += g["npairs"]
    np.testing.assert_array_equal(n, full["npairs"])
    np.testing.assert_allclose(w, full["weight"], rtol=1e-7)      # TreeCorr sums weights in a different order (~1e-9)
    assert n_nofold.sum() < 0.99 * full["npairs"].sum()          # omitting the 2 loses a lot (3.3% on real data)


def test_assemble_auto_pair_is_exact(chunk_case, drv, monkeypatch):
    out = np.load(_assemble(drv, chunk_case["root"], chunk_case["K"], monkeypatch) / "Bin1_Bin1.npz")
    f = chunk_case["full1"]
    np.testing.assert_array_equal(out["npairs"], f["npairs"])
    assert is_integral(out["npairs"])
    np.testing.assert_allclose(out["weight"], f["weight"], rtol=1e-7)
    scale = np.abs(f["xip"]).max()
    np.testing.assert_allclose(out["xip"], f["xip"], rtol=1e-6, atol=1e-7 * scale)
    # xi_-, xi_x: single-shot TreeCorr is itself only ~1e-3 x scale exact for them (cell projection), chunk sums
    # agree with the independent counter to ~3e-4 x scale
    for k in ("xim", "xim_im"):
        np.testing.assert_allclose(out[k], f[k], rtol=0, atol=3e-3 * scale)


# Regression (found by this suite 2026-10-29, fixed): see TEST_MEASURE.md
def test_assembled_auto_xip_im_matches_single_shot(chunk_case, drv, monkeypatch):
    out = np.load(_assemble(drv, chunk_case["root"], chunk_case["K"], monkeypatch) / "Bin1_Bin1.npz")
    np.testing.assert_allclose(out["xip_im"], chunk_case["full1"]["xip_im"], rtol=0,
                               atol=1e-3 * np.abs(chunk_case["full1"]["xip"]).max())


def test_assemble_cross_pair_no_fold(chunk_case, drv, monkeypatch):
    out = np.load(_assemble(drv, chunk_case["root"], chunk_case["K"], monkeypatch) / "Bin1_Bin2.npz")
    f = chunk_case["full12"]
    np.testing.assert_array_equal(out["npairs"], f["npairs"])
    np.testing.assert_allclose(out["weight"], f["weight"], rtol=1e-7)
    np.testing.assert_allclose(out["xip"], f["xip"], rtol=1e-6, atol=1e-7 * np.abs(f["xip"]).max())
    np.testing.assert_allclose(out["xim"], f["xim"], rtol=0, atol=3e-3 * np.abs(f["xip"]).max())


def test_assemble_without_fold_is_detectably_wrong(chunk_case, drv, monkeypatch):
    """Guard on the guard: a fold of 1 everywhere (the obvious first implementation) must NOT reproduce the single shot."""
    out = np.load(_assemble(drv, chunk_case["bad"], chunk_case["K"], monkeypatch) / "Bin1_Bin1.npz")
    assert out["npairs"].sum() < 0.99 * chunk_case["full1"]["npairs"].sum()


def test_assemble_refuses_incomplete_pairs(chunk_case, drv, monkeypatch, tmp_path, capsys):
    import shutil
    src = chunk_case["root"] / "chunks" / "Bin1_Bin1"
    dst = tmp_path / "chunks" / "Bin1_Bin1"
    shutil.copytree(src, dst)
    sorted(dst.glob("c*.npz"))[-1].unlink()
    out = _assemble(drv, tmp_path, chunk_case["K"], monkeypatch)
    assert "INCOMPLETE" in capsys.readouterr().out
    assert not (out / "Bin1_Bin1.npz").exists()


def test_load_pair_cats_uses_one_based_bins(drv, tmp_path, monkeypatch):
    """The driver's (i,j)->(b1,b2)=(i+1,j+1) must reach load_bin as 1-based tomographic bins."""
    _bin_files(tmp_path)
    for k in range(6):
        d = dict(np.load(tmp_path / f"bin{k}.npz"))
        d["ra"] = np.linspace(10.0, 11.0, d["ra"].size)
        np.savez(tmp_path / f"bin{k}.npz", **d)
    orig = xim_mod.load_bin
    monkeypatch.setattr(xim_mod, "load_bin", lambda b, work=tmp_path: orig(b, work=tmp_path))
    ca, cb = drv.load_pair_cats(3, 4, 2, {})
    assert sum(c.nobj for c in ca) == 5 + 2 and sum(c.nobj for c in cb) == 5 + 3
    assert all(np.all(c.g1 == 3.0) for c in ca) and all(np.all(c.g1 == 4.0) for c in cb)


# --------------------------------------------------------------------------------------------- xi2cosebis.py
def _xi_from_cl(th_rad, sign_b=False):
    """xi_+ = int l dl/2pi C J0, xi_- = int l dl/2pi C J4 for a smooth E-mode C(l) (sign_b: B-mode spectrum,
    xi_- flips sign)."""
    from scipy.special import jv
    lnl = np.linspace(0.0, np.log(3e5), 3000)
    l = np.exp(lnl)
    C = 1e-9 * (l / 100) / (1 + (l / 100) ** 4) * np.exp(-(l / 5e4) ** 2)
    f = np.full(l.size, lnl[1] - lnl[0]); f[[0, -1]] *= 0.5
    f = f * l * l * C / (2 * np.pi)
    xp = jv(0, np.outer(th_rad, l)) @ f
    xm = jv(4, np.outer(th_rad, l)) @ f
    return xp, (-xm if sign_b else xm)


@pytest.fixture(scope="module")
def cos_case():
    F = CosebisFilters(0.5, 300.0, 6)                      # staged 0.5-300' tables, no mpmath
    nb = 300
    e = np.exp(np.linspace(np.log(0.5), np.log(300.0), nb + 1))
    th = np.sqrt(e[:-1] * e[1:])
    xp, xm = _xi_from_cl(th * xi2cosebis.ARCMIN)
    thf = np.exp(np.linspace(np.log(0.4), np.log(400.0), 500))
    xpf, xmf = _xi_from_cl(thf * xi2cosebis.ARCMIN)
    Eref, Bref = E_from_xi(thf * xi2cosebis.ARCMIN, xpf, xmf, F, 6)
    return dict(F=F, th=th, e=e, xp=xp, xm=xm, Eref=Eref, Bref=Bref)


def test_e_mode_xi_gives_b_n_null(cos_case):
    c = cos_case
    E, B = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, unit="rad")
    assert np.all(np.abs(E) > 1e-12)
    assert np.max(np.abs(B) / np.abs(E)) < 1e-3, (E, B)
    # a pure B-mode spectrum (xi_- sign flipped) swaps the roles: E ~ 0, B = the same numbers
    xpB, xmB = _xi_from_cl(c["th"] * xi2cosebis.ARCMIN, sign_b=True)
    E2, B2 = xi2cosebis.e_from_xi_binned(c["th"], xpB, xmB, c["F"], 6, unit="rad")
    assert np.max(np.abs(E2) / np.abs(B2)) < 1e-3
    np.testing.assert_allclose(B2, E, rtol=1e-3)


def test_binned_agrees_with_theory_cosebis_route(cos_case):
    """The measurement-side integrator (binned, radians) vs kidspipe.theory.cosebis.E_from_xi (spline + Gauss)."""
    c = cos_case
    E, B = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, scheme="edges", unit="rad")
    np.testing.assert_allclose(E, c["Eref"], rtol=2e-3)
    assert np.max(np.abs(B - c["Bref"])) < 1e-3 * np.max(np.abs(c["Eref"]))
    Ec, _ = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, scheme="centre", unit="rad")
    np.testing.assert_allclose(Ec, c["Eref"], rtol=2e-3)


def test_radians_vs_arcmin_units(cos_case):
    """unit='rad' = unit='arcmin' x ARCMIN^2 (theta dtheta in rad^2); using the arcmin number as if it were radians
    would be off by 1/ARCMIN^2 = 1.2e7."""
    c = cos_case
    Er, Br = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, unit="rad")
    Ea, Ba = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, unit="arcmin")
    np.testing.assert_allclose(Er, Ea * xi2cosebis.ARCMIN ** 2, rtol=1e-13)
    np.testing.assert_allclose(Br, Ba * xi2cosebis.ARCMIN ** 2, rtol=1e-12, atol=1e-30)
    assert abs(xi2cosebis.ARCMIN - np.pi / 10800.0) < 1e-18
    assert np.max(np.abs(Ea / c["Eref"] - 1)) > 1e6                  # arcmin units would not match the radian reference
    np.testing.assert_allclose(Er, c["Eref"], rtol=2e-3)


def test_edges_weights_exact_and_default(cos_case):
    c = cos_case
    w = xi2cosebis._weights(c["th"], xi2cosebis.bin_edges(c["th"]), "edges")
    assert abs(w.sum() / ((300.0 ** 2 - 0.5 ** 2) / 2) - 1) < 1e-12        # exact int theta dtheta over the range
    wc = xi2cosebis._weights(c["th"], xi2cosebis.bin_edges(c["th"]), "centre")
    with pytest.raises(ValueError):
        xi2cosebis._weights(c["th"], c["e"], "nonsense")
    d = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6)
    e = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, scheme="edges")
    np.testing.assert_array_equal(d[0], e[0]); np.testing.assert_array_equal(d[1], e[1])
    # coarse binning separates the schemes: they are genuinely different weights
    th = c["th"][::30]; xp = c["xp"][::30]; xm = c["xm"][::30]
    Eed = xi2cosebis.e_from_xi_binned(th, xp, xm, c["F"], 6, scheme="edges")[0]
    Ece = xi2cosebis.e_from_xi_binned(th, xp, xm, c["F"], 6, scheme="centre")[0]
    assert not np.allclose(Eed, Ece, rtol=1e-6)


def test_b_sign_convention_algebra():
    """E = (A_+ + A_-)/2, B = (A_+ - A_-)/2: xi_- = 0 gives E == B; xi_+ = 0 gives E == -B."""
    F = CosebisFilters(0.5, 300.0, 3)
    e = np.exp(np.linspace(np.log(0.5), np.log(300.0), 101)); th = np.sqrt(e[:-1] * e[1:])
    f = np.exp(-th / 40.0)
    E, B = xi2cosebis.e_from_xi_binned(th, f, 0 * f, F, 3)
    np.testing.assert_allclose(E, B, rtol=1e-12)
    E, B = xi2cosebis.e_from_xi_binned(th, 0 * f, f, F, 3)
    np.testing.assert_allclose(E, -B, rtol=1e-12)
    # leading dimensions are preserved
    E2, _ = xi2cosebis.e_from_xi_binned(th, np.stack([f, 2 * f]), np.stack([0 * f, 0 * f]), F, 3)
    assert E2.shape == (2, 3)
    np.testing.assert_allclose(E2[1], 2 * E2[0], rtol=1e-12)


def test_bin_edges_roundtrip_arbitrary_range():
    e = np.exp(np.linspace(np.log(1.7), np.log(55.0), 41))
    np.testing.assert_allclose(xi2cosebis.bin_edges(np.sqrt(e[:-1] * e[1:])), e, rtol=1e-12)


# Regression (found by this suite 2026-10-29, fixed): see TEST_MEASURE.md
def test_trapz_scheme_matches_edges(cos_case):
    c = cos_case
    Et = xi2cosebis.e_from_xi_binned(c["th"], c["xp"][None], c["xm"][None], c["F"], 6, scheme="trapz", unit="rad")[0][0]
    # trapz drops the outer half-bins (O(dln theta)): on 300 bins modes 1-3 agree to <1 %, modes 4-6 to <8 %.
    # The fixed bug (a missing factor theta) was a factor ~19.
    np.testing.assert_allclose(Et[:3], c["Eref"][:3], rtol=1e-2)
    np.testing.assert_allclose(Et, c["Eref"], rtol=0.1)
    E1 = xi2cosebis.e_from_xi_binned(c["th"], c["xp"], c["xm"], c["F"], 6, scheme="trapz", unit="rad")[0]
    np.testing.assert_allclose(E1, Et, rtol=1e-12)                  # 1-D input works (was an IndexError)


# --------------------------------------------------------------------------------------------- official.py
# Regression (found by this suite 2026-10-29, fixed): see TEST_MEASURE.md
def test_weighted_rebin_theta_2d_input():
    th = official.fine_bin_centres()
    _, thb, e = official.weighted_rebin(th, np.ones((2, 1000)), np.ones((2, 1000)), nang=9)
    assert np.all((thb > e[:-1]) & (thb < e[1:]))


def test_fine_bin_centres_default_and_custom():
    c = official.fine_bin_centres()
    assert c.shape == (1000,) and np.all(np.diff(c) > 0) and 2.0 < c[0] and c[-1] < 300.0
    c2 = official.fine_bin_centres(1.0, 100.0, 2)
    np.testing.assert_allclose(c2, [np.sqrt(1 * 10.0), np.sqrt(10.0 * 100.0)])
    assert len(official.PAIRS) == 21 and official.PAIRS[0] == (0, 0) and official.PAIRS[1] == (0, 1)
    assert official.PAIRS[5] == (0, 5) and official.PAIRS[6] == (1, 1) and official.PAIRS[-1] == (5, 5)


def test_cov_block_extractors_index_correctly():
    nang = 9
    n = 2 * 21 * nang
    cov = np.arange(n * n, dtype=float).reshape(n, n)
    p = official.cov_xip_block(cov, nang)
    m = official.cov_xim_block(cov, nang)
    assert p.shape == m.shape == (21 * nang, 21 * nang)
    np.testing.assert_array_equal(p, cov[:21 * nang, :21 * nang])
    np.testing.assert_array_equal(m, cov[21 * nang:, 21 * nang:])
    cb = np.arange(840 * 840, dtype=float).reshape(840, 840)
    e = official.cov_e_block(cb, 6)
    assert e.shape == (126, 126)
    assert e[0, 0] == cb[0, 0] and e[6, 6] == cb[20, 20] and e[7, 6] == cb[21, 20]      # pair-major, first 6 modes


def test_weighted_rebin_synthetic():
    th = official.fine_bin_centres()
    npw = np.ones((2, 1000))
    xi_c = np.vstack([np.full(1000, 3.0), np.full(1000, -1.0)])
    out, thb, e = official.weighted_rebin(th, xi_c, npw, nang=9)
    np.testing.assert_allclose(out, [[3.0] * 9, [-1.0] * 9], rtol=1e-13)
    assert out.shape == (2, 9)
    o1d, th1d, e = official.weighted_rebin(th, xi_c[0], npw[0], nang=9)        # 1-D input: theta is well defined
    np.testing.assert_allclose(o1d, 3.0, rtol=1e-13)
    assert np.all(np.diff(th1d) > 0) and np.all((th1d > e[:-1]) & (th1d < e[1:]))
    # weighting: doubling the weight of the upper half of every bin pulls the mean up
    ramp = np.vstack([th, th])
    w2 = np.ones((2, 1000)); w2[:, 500:] = 5.0
    o2, _, _ = official.weighted_rebin(th, ramp, w2, nang=9)
    o1, _, _ = official.weighted_rebin(th, ramp, npw, nang=9)
    assert np.all(o2 >= o1 - 1e-12)


def _write_fake_ascii(root, order):
    d = root / "cosmosis_xipm"
    d.mkdir(parents=True)
    keys = ["xip", "xim", "xip_im", "xim_im", "sigma_xip", "sigma_xim", "npairs", "weight", "npairs_weighted",
            "r_nom", "meanr", "meanlogr"]
    th = official.fine_bin_centres()
    for (i, j) in order:
        cols = [np.full(1000, 10.0 * i + j) if k == "xip" else th if k == "r_nom" else np.full(1000, 1.0)
                for k in keys]
        np.savetxt(d / f"XI_KiDS_Legacy_NScomb_nBins_6_Bin{i}_Bin{j}.ascii", np.column_stack(cols),
                   header=" ".join(keys), comments="# ")


def test_load_official_xi_ordering_synthetic(tmp_path, monkeypatch):
    _write_fake_ascii(tmp_path, [(i + 1, j + 1) for i, j in official.PAIRS])
    monkeypatch.setattr(official, "OFFICIAL", tmp_path)
    d = official.load_official_xi("NScomb")
    assert d["xip"].shape == (21, 1000)
    expect = [10.0 * (i + 1) + (j + 1) for i, j in official.PAIRS]
    np.testing.assert_array_equal(d["xip"][:, 0], expect)                  # row k <-> PAIRS[k] (Bin1_Bin1, 1_2, ...)
    assert d["pairs"] == official.PAIRS
    np.testing.assert_allclose(d["r_nom"][3], official.fine_bin_centres())


def test_load_official_xi_missing_files_raise(tmp_path, monkeypatch):
    _write_fake_ascii(tmp_path, [(i + 1, j + 1) for i, j in official.PAIRS][:-1])
    monkeypatch.setattr(official, "OFFICIAL", tmp_path)
    with pytest.raises(FileNotFoundError):
        official.load_official_xi("NScomb")


def _fake_fits(root, shuffle_seed=0):
    from astropy.io import fits
    root.mkdir(parents=True, exist_ok=True)
    P = official.PAIRS
    rng = np.random.default_rng(shuffle_seed)

    perms = {}

    def tab(name, nang, vals, with_ang):
        rows = [(i + 1, j + 1, a + 1, vals(k, a), 2.0 * (a + 1)) for k, (i, j) in enumerate(P) for a in range(nang)]
        perm = perms.setdefault(nang, rng.permutation(len(rows)))     # xiP and xiM share row order (as in the release)
        rows = [rows[x] for x in perm]
        cols = [fits.Column(name="BIN1", format="K", array=[r[0] for r in rows]),
                fits.Column(name="BIN2", format="K", array=[r[1] for r in rows]),
                fits.Column(name="ANGBIN", format="K", array=[r[2] for r in rows]),
                fits.Column(name="VALUE", format="D", array=[r[3] for r in rows])]
        if with_ang:
            cols.append(fits.Column(name="ANG", format="D", array=[r[4] for r in rows]))
        return fits.BinTableHDU.from_columns(cols, name=name)
    nz = fits.BinTableHDU.from_columns([fits.Column(name="Z", format="D", array=[0.1, 0.2])], name="NZ_SOURCE")
    nz.header["NGAL1"] = 1.5; nz.header["SIG_E1"] = 0.27
    nang = 9
    cov = rng.normal(size=(2 * 21 * nang,) * 2)
    h = fits.HDUList([fits.PrimaryHDU(), tab("xiP", nang, lambda k, a: 100.0 * k + a, True),
                      tab("xiM", nang, lambda k, a: -100.0 * k - a, True),
                      fits.ImageHDU(cov, name="COVMAT"), nz])
    h.writeto(root / "KiDS_Legacy_xipm.fits", overwrite=True)
    ne = 20
    cov2 = rng.normal(size=(2 * 21 * ne,) * 2)
    h2 = fits.HDUList([fits.PrimaryHDU(), tab("En", ne, lambda k, a: 100.0 * k + a, False),
                       tab("Bn", ne, lambda k, a: -100.0 * k - a, False), fits.ImageHDU(cov2, name="COVMAT"), nz])
    h2.writeto(root / "KiDS_Legacy_cosebis.fits", overwrite=True)
    return cov, cov2


def test_fits_loaders_reorder_pair_major_synthetic(tmp_path, monkeypatch):
    pytest.importorskip("astropy")
    cov, cov2 = _fake_fits(tmp_path)
    monkeypatch.setattr(official, "OFFICIAL", tmp_path)
    x = official.load_official_xipm_fits()
    exp = np.array([[100.0 * k + a for a in range(9)] for k in range(21)])
    np.testing.assert_array_equal(x["xiP"], exp)
    np.testing.assert_array_equal(x["xiM"], -exp)
    assert x["nang"] == 9 and x["theta"].shape == (9,)   # theta = ang[:nang] of the file order (not re-sorted): only the real file checks monotonicity
    assert x["cov"].shape == (378, 378) and np.allclose(x["cov"], x["cov"].T)
    np.testing.assert_array_equal(x["pair"], np.repeat(np.arange(21), 9).reshape(21, 9))
    np.testing.assert_array_equal(x["ang"], np.tile(np.arange(1, 10), (21, 1)))
    assert x["header"]["NGAL1"] == 1.5 and x["header"]["SIG_E1"] == 0.27
    c = official.load_official_cosebis()
    exp2 = np.array([[100.0 * k + a for a in range(20)] for k in range(21)])
    np.testing.assert_array_equal(c["En"], exp2)
    np.testing.assert_array_equal(c["Bn"], -exp2)
    assert c["cov"].shape == (840, 840) and np.allclose(c["cov"], c["cov"].T)


# real release files (skipped when absent) -----------------------------------------------------------------
@needs_release
def test_real_ascii_shapes_and_order():
    d = official.load_official_xi("NScomb")
    assert d["xip"].shape == (21, 1000) and d["pairs"] == official.PAIRS
    np.testing.assert_allclose(d["r_nom"][0], official.fine_bin_centres(), rtol=1e-5)
    assert np.all(d["r_nom"] == d["r_nom"][0])
    assert np.all(d["npairs_weighted"] > 0) and np.all(np.isfinite(d["xip"]))
    # auto pairs sit at the PAIRS indices with i == j
    auto = [k for k, (i, j) in enumerate(official.PAIRS) if i == j]
    assert auto == [0, 6, 11, 15, 18, 20]


@needs_release
def test_real_fits_shapes_and_order():
    x = official.load_official_xipm_fits()
    assert x["xiP"].shape == (21, 9) and x["xiM"].shape == (21, 9) and x["cov"].shape == (378, 378)
    assert np.all(np.diff(x["theta"]) > 0) and 0.5 < x["theta"][0] < 10 and x["theta"][-1] < 300
    np.testing.assert_array_equal(x["pair"], np.repeat(np.arange(21), 9).reshape(21, 9))
    assert np.linalg.eigvalsh(x["cov"]).min() > -1e-12 * np.abs(x["cov"]).max()
    assert any(k.startswith("NGAL") for k in x["header"])
    c = official.load_official_cosebis()
    assert c["En"].shape == (21, 20) and c["Bn"].shape == (21, 20) and c["cov"].shape == (840, 840)
    assert official.cov_e_block(c["cov"], 6).shape == (126, 126)


@needs_release
def test_real_ascii_consistent_with_fits_ordering():
    """Same pair index means the same tomographic pair in the ascii (fine) and the FITS (9-bin) loaders:
    re-binned ascii xi+ must track the FITS xi+ pair by pair (correlation over all 21x9 points, and per pair)."""
    d = official.load_official_xi("NScomb")
    x = official.load_official_xipm_fits()
    reb, _, _ = official.weighted_rebin(d["r_nom"][0], d["xip"], d["npairs_weighted"], nang=9)
    assert np.corrcoef(reb.ravel(), x["xiP"].ravel())[0, 1] > 0.99
    # a wrong pair mapping would break the pair-by-pair relation at least for pair (1,6) vs (1,1)
    assert np.corrcoef(reb[5], x["xiP"][5])[0, 1] > np.corrcoef(reb[5], x["xiP"][0])[0, 1]


# --------------------------------------------------------------------------------------------- angle_slop option
# TreeCorr 5 `angle_slop` (default 0.1 rad) bounds the projection-angle error of merged cell pairs; xi_- carries
# e^{-4 i phi}.  The option is plumbed through xi.py and the chunked driver; None (default) = TreeCorr's own default.
def test_treecorr_kwargs_angle_slop_default_is_absent_and_set_passes_through():
    assert "angle_slop" not in xim_mod.treecorr_kwargs()                      # default unchanged
    assert "angle_slop" not in xim_mod.treecorr_kwargs(angle_slop=None)
    assert xim_mod.treecorr_kwargs(angle_slop=0.0)["angle_slop"] == 0.0       # 0.0 is a real value, not "unset"
    assert xim_mod.treecorr_kwargs(angle_slop=0.02)["angle_slop"] == 0.02
    base = xim_mod.treecorr_kwargs()
    assert {k: v for k, v in xim_mod.treecorr_kwargs(angle_slop=0.5).items() if k != "angle_slop"} == base


def test_run_pair_passes_angle_slop_to_treecorr(monkeypatch, small):
    seen = []
    real = treecorr.GGCorrelation

    def spy(cfg, *a, **k):
        seen.append(dict(cfg))
        return real(cfg, *a, **k)

    monkeypatch.setattr(treecorr, "GGCorrelation", spy)
    c = xim_mod.make_treecorr_cat(small)
    xim_mod.run_pair(c, c, nbins=NB, bin_slop=0.0, num_threads=THREADS)
    xim_mod.run_pair(c, c, nbins=NB, bin_slop=0.0, num_threads=THREADS, angle_slop=0.0)
    xim_mod.run_pair(c, c, nbins=NB, bin_slop=1.0, num_threads=THREADS, angle_slop=0.02)
    assert "angle_slop" not in seen[0]
    assert seen[1]["angle_slop"] == 0.0 and seen[1]["bin_slop"] == 0.0
    assert seen[2]["angle_slop"] == 0.02 and seen[2]["bin_slop"] == 1.0


def test_run_pair_default_equals_explicit_treecorr_default_and_angle_slop_zero_is_exact(small):
    """Default behaviour unchanged (None == TreeCorr's 0.1) and angle_slop=0 reproduces brute force."""
    c = xim_mod.make_treecorr_cat(small)
    d0 = xim_mod.corr_to_arrays(xim_mod.run_pair(c, c, nbins=NB, bin_slop=0.0, num_threads=THREADS))
    d1 = xim_mod.corr_to_arrays(xim_mod.run_pair(c, c, nbins=NB, bin_slop=0.0, num_threads=THREADS, angle_slop=0.1))
    np.testing.assert_allclose(d0["xim"], d1["xim"], rtol=1e-12, atol=1e-15)     # threads sum in a different order
    np.testing.assert_allclose(d0["xip"], d1["xip"], rtol=1e-12, atol=1e-15)
    ex = xim_mod.corr_to_arrays(xim_mod.run_pair(c, c, nbins=NB, bin_slop=0.0, num_threads=THREADS, angle_slop=0.0))
    # TreeCorr's own brute-force mode is the exactness reference (spherical projection angles at each galaxy).  The
    # module-level brute() uses a flat mean-dec position angle: it differs from TreeCorr by ~1e-5 in xi+ on this field,
    # so it is only a loose sanity bound here (this test originally, wrongly, demanded rtol 1e-9 against it).
    kw = xim_mod.treecorr_kwargs(NB, 0.0, THREADS)
    kw["brute"] = True
    bf = treecorr.GGCorrelation(kw)
    bf.process(c, c)
    np.testing.assert_allclose(ex["xip"], bf.xip, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ex["xim"], bf.xim, rtol=1e-9, atol=1e-12)
    ref = brute(small, nbins=NB)
    np.testing.assert_allclose(ex["xip"], ref["p"], rtol=0, atol=5e-5)
    np.testing.assert_array_equal(ex["npairs"], bf.npairs)
    np.testing.assert_allclose(ex["npairs"], 2.0 * ref["np"], atol=6, rtol=1e-3)   # bin-edge pairs differ in separation formula


def test_slop_record_default_adds_nothing_and_set_is_recorded(drv):
    assert drv.slop_record(0.0, None) == {}                                   # legacy / default chunk files unchanged
    r = drv.slop_record(0.0, 0.0)
    assert float(r["angle_slop"]) == 0.0 and float(r["bin_slop"]) == 0.0      # angle_slop=0 IS recorded
    r = drv.slop_record(1.5, None)
    assert "angle_slop" not in r and float(r["bin_slop"]) == 1.5
    assert float(drv.slop_record(0.5, 0.02)["angle_slop"]) == 0.02


def _run_driver(drv, monkeypatch, root, chunk_case, extra, chunk="0-0"):
    cats = chunk_case["c1"]
    monkeypatch.setattr(drv, "OUTROOT", root)
    monkeypatch.setattr(drv, "load_pair_cats", lambda b1, b2, nch, cache: (cats, cats))
    monkeypatch.setattr(sys, "argv", ["drv", "--nchunks", str(chunk_case["K"]), "--pairs", "1-1", "--chunk", chunk,
                                      "--nbins", str(chunk_case["nb"]), "--threads", str(THREADS)] + extra)
    drv.main()


def test_driver_angle_slop_flag_recorded_in_chunk_and_manifest(chunk_case, drv, monkeypatch, tmp_path):
    import json
    _run_driver(drv, monkeypatch, tmp_path / "d", chunk_case, [])                       # default
    f = tmp_path / "d" / "chunks" / "Bin1_Bin1" / "c00_00.npz"
    z = np.load(f)
    assert "angle_slop" not in z.files and "bin_slop" not in z.files                    # default run: unchanged schema
    assert json.loads((tmp_path / "d" / "manifest.json").read_text())["angle_slop"] is None
    _run_driver(drv, monkeypatch, tmp_path / "s", chunk_case, ["--angle-slop", "0.0", "--bin-slop", "0.5"])
    z = np.load(tmp_path / "s" / "chunks" / "Bin1_Bin1" / "c00_00.npz")
    assert float(z["angle_slop"]) == 0.0 and float(z["bin_slop"]) == 0.5
    m = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert m["angle_slop"] == 0.0 and m["bin_slop"] == 0.5


def test_assemble_refuses_to_mix_slop_settings(chunk_case, drv, monkeypatch, tmp_path):
    import shutil
    K = chunk_case["K"]
    src = chunk_case["root"] / "chunks" / "Bin1_Bin1"
    for variant in ("angle", "bin"):
        dst = tmp_path / variant / "chunks" / "Bin1_Bin1"
        shutil.copytree(src, dst)
        files = sorted(dst.glob("c*.npz"))
        d = dict(np.load(files[0]))
        d.update({"angle_slop": np.float64(0.02), "bin_slop": np.float64(0.0)} if variant == "angle"
                 else {"bin_slop": np.float64(1.5)})
        np.savez(files[0], **d)                       # one file differs from the (legacy, key-less) rest
        with pytest.raises(RuntimeError, match="mix"):
            _assemble(drv, tmp_path / variant, K, monkeypatch)
        assert not (tmp_path / variant / "assembled" / "ns" / "Bin1_Bin1.npz").exists()
    # all files equal (all with angle_slop=0.02) assembles and records the setting; legacy key-less files still assemble
    dst = tmp_path / "same" / "chunks" / "Bin1_Bin1"
    shutil.copytree(src, dst)
    for f in sorted(dst.glob("c*.npz")):
        d = dict(np.load(f)); d.update(angle_slop=np.float64(0.02), bin_slop=np.float64(0.0)); np.savez(f, **d)
    out = np.load(_assemble(drv, tmp_path / "same", K, monkeypatch) / "Bin1_Bin1.npz")
    assert float(out["angle_slop"]) == 0.02
    out0 = np.load(_assemble(drv, chunk_case["root"], K, monkeypatch) / "Bin1_Bin1.npz")
    assert "angle_slop" not in out0.files


# --------------------------------------------------------------------------------------------- Prompt 09: mmap, pruning, field cache
def _load_extract_script():
    spec = importlib.util.spec_from_file_location("dr5_extract", ROOT / "scripts" / "dr5_extract.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_bin_mmap_matches_npz(tmp_path):
    """mmap=True reads bin{b-1}/*.npy read-only memmaps matching bin{b-1}.npz bit-for-bit."""
    ext = _load_extract_script()
    npz_dir = tmp_path / "npz"
    u_dir = tmp_path / "uncompressed"
    npz_dir.mkdir()
    rng = np.random.default_rng(42)
    for k in range(6):
        n = 12 + 3 * k
        np.savez(
            npz_dir / f"bin{k}.npz",
            ra=rng.uniform(0.0, 360.0, n),
            dec=rng.uniform(-35.0, 5.0, n),
            e1=rng.normal(0.0, 0.25, n),
            e2=rng.normal(0.0, 0.25, n),
            w=rng.uniform(0.2, 2.0, n),
            north=(rng.random(n) > 0.5),
        )
    ext.convert_npz_to_uncompressed(npz_dir, u_dir)

    for b in range(1, 7):
        d_npz = xim_mod.load_bin(b, work=npz_dir, mmap=False)
        d_mm = xim_mod.load_bin(b, work=u_dir, mmap=True)
        assert set(d_npz) == set(d_mm)
        for col in d_npz:
            assert isinstance(d_mm[col], np.memmap)
            assert not d_mm[col].flags.writeable
            np.testing.assert_array_equal(d_mm[col], d_npz[col])
        sub = xim_mod.load_bin(b, work=u_dir, mmap=True, columns=("ra", "dec"))
        assert set(sub) == {"ra", "dec"}
        np.testing.assert_array_equal(sub["ra"], d_npz["ra"])

    with pytest.raises(FileNotFoundError):
        xim_mod.load_bin(1, work=tmp_path / "absent", mmap=True)


def test_dr5_extract_uncompressed_cli_preserves_npz(tmp_path):
    """--from-npz --uncompressed-dir converts requested bins without altering existing bin*.npz files."""
    import hashlib
    import subprocess

    npz_dir = tmp_path / "npz"
    u_dir = tmp_path / "u"
    npz_dir.mkdir()
    _bin_files(npz_dir)

    before_digest = {
        p.name: (hashlib.md5(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(npz_dir.glob("bin*.npz"))
    }
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dr5_extract.py"),
            "--out", str(npz_dir),
            "--uncompressed-dir", str(u_dir),
            "--from-npz",
            "--bins", "1,6",
        ],
        check=True,
    )
    assert (u_dir / "bin0" / "ra.npy").is_file()
    assert (u_dir / "bin5" / "w.npy").is_file()
    for middle in range(1, 5):
        assert not (u_dir / f"bin{middle}").exists()

    for b in (1, 6):
        ref = xim_mod.load_bin(b, work=npz_dir, mmap=False)
        got = xim_mod.load_bin(b, work=u_dir, mmap=True)
        for col in ref:
            np.testing.assert_array_equal(got[col], ref[col])

    after_digest = {
        p.name: (hashlib.md5(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(npz_dir.glob("bin*.npz"))
    }
    assert after_digest == before_digest


def test_load_pair_cats_mmap_and_needed_subset(drv, tmp_path):
    """Streaming mmap load_pair_cats builds only needed chunks and gives bitwise-identical TreeCorr outputs."""
    ext = _load_extract_script()
    npz_dir = tmp_path / "npz"
    u_dir = tmp_path / "u"
    npz_dir.mkdir()
    r1 = lens_field(300, 4.0, seed=101, ra0=150.0)
    r2 = lens_field(320, 4.0, seed=102, ra0=150.0)
    np.savez(npz_dir / "bin0.npz", **r1)
    np.savez(npz_dir / "bin1.npz", **r2)
    ext.convert_npz_to_uncompressed(npz_dir, u_dir, bins=[0, 1])

    c1_full, c2_full = drv.load_pair_cats(1, 2, 3, {}, mmap=False, work=npz_dir)
    cache_mm = {}
    c1_sub, c2_sub = drv.load_pair_cats(1, 2, 3, cache_mm, mmap=True, work=u_dir, needed={1: {0}, 2: {2}})
    assert c1_sub[0] is not None and c1_sub[1] is None and c1_sub[2] is None
    assert c2_sub[2] is not None and c2_sub[0] is None and c2_sub[1] is None

    ref = xim_mod.corr_to_arrays(xim_mod.run_pair(c1_full[0], c2_full[2], nbins=NB, bin_slop=0.0, num_threads=1, angle_slop=0.0))
    got = xim_mod.corr_to_arrays(xim_mod.run_pair(c1_sub[0], c2_sub[2], nbins=NB, bin_slop=0.0, num_threads=1, angle_slop=0.0))
    np.testing.assert_array_equal(got["npairs"], ref["npairs"])
    for k in ("weight", "meanr", "xip", "xim", "varxi"):
        np.testing.assert_array_equal(got[k], ref[k])


def test_bounding_cap_pruning_and_manifest_assembly(drv, tmp_path, monkeypatch, capsys):
    """Conservative cap prune skips distant pairs, handles RA=0/360 wraparound, and assembles identically."""
    import json
    import shutil

    # Wraparound sanity: 359.9 deg and 0.1 deg are 0.2 deg apart (12 arcmin < 300'), MUST NOT skip!
    cap_a = drv.bounding_cap_rad(np.deg2rad([359.85, 359.95]), np.deg2rad([0.0, 0.0]))
    cap_b = drv.bounding_cap_rad(np.deg2rad([0.05, 0.15]), np.deg2rad([0.0, 0.0]))
    assert not drv.can_skip_pair(cap_a, cap_b)
    # Antipodal degenerate points still produce a valid cap, and empty caps are skipped
    cap_anti = drv.bounding_cap_rad(np.deg2rad([0.0, 180.0]), np.deg2rad([0.0, 0.0]))
    assert cap_anti["radius_rad"] >= np.pi / 2 - 1e-12
    assert drv.can_skip_pair(drv.bounding_cap_rad([], []), cap_b)

    ext = _load_extract_script()
    npz_dir = tmp_path / "npz"
    u_dir = tmp_path / "u"
    npz_dir.mkdir()
    patch1 = lens_field(240, 2.0, seed=111, ra0=10.0)
    patch2 = lens_field(240, 2.0, seed=112, ra0=35.0)
    rec = {k: np.concatenate([patch1[k], patch2[k]]) for k in ("ra", "dec", "e1", "e2", "w")}
    np.savez(npz_dir / "bin0.npz", **rec)
    ext.convert_npz_to_uncompressed(npz_dir, u_dir, bins=[0])

    # Run --prune --mmap and --no-prune --no-mmap via CLI main()
    out_pruned = tmp_path / "out_pruned"
    out_full = tmp_path / "out_full"
    orig_outroot = drv.OUTROOT
    try:
        monkeypatch.setattr(drv, "OUTROOT", out_full)
        monkeypatch.setattr(sys, "argv", [
            "drv", "--nchunks", "4", "--pairs", "1-1", "--nbins", str(NB),
            "--bin-slop", "0.0", "--angle-slop", "0.0", "--threads", "1",
            "--no-prune", "--no-mmap", "--work", str(npz_dir),
        ])
        drv.main()

        monkeypatch.setattr(drv, "OUTROOT", out_pruned)
        monkeypatch.setattr(sys, "argv", [
            "drv", "--nchunks", "4", "--pairs", "1-1", "--nbins", str(NB),
            "--bin-slop", "0.0", "--angle-slop", "0.0", "--threads", "1",
            "--prune", "--mmap", "--work", str(u_dir),
        ])
        drv.main()

        manifest = json.loads((out_pruned / "manifest.json").read_text())
        assert manifest["prune_zero_pairs"] is True
        assert manifest["mmap"] is True
        zero_list = [tuple(x) for x in manifest["analytically_zero_pairs"]["Bin1_Bin1"]]
        assert len(zero_list) >= 4
        assert manifest["chunk_pairs_analytically_zero"] == len(zero_list)
        for (a, b) in zero_list:
            assert not (out_pruned / "chunks" / "Bin1_Bin1" / f"c{a:02d}_{b:02d}.npz").exists()
            with np.load(out_full / "chunks" / "Bin1_Bin1" / f"c{a:02d}_{b:02d}.npz") as zf:
                assert float(np.sum(zf["npairs"])) == 0.0

        # --status reports cleanly on pruned tree
        monkeypatch.setattr(sys, "argv", ["drv", "--status", "--nchunks", "4"])
        drv.main()
        assert "Bin1_Bin1" in capsys.readouterr().out

        # Assemble both and confirm exact agreement
        monkeypatch.setattr(drv, "OUTROOT", out_full)
        drv.do_assemble(argparse.Namespace(nchunks=4))
        monkeypatch.setattr(drv, "OUTROOT", out_pruned)
        drv.do_assemble(argparse.Namespace(nchunks=4))

        a_full = np.load(out_full / "assembled" / "ns" / "Bin1_Bin1.npz")
        a_pruned = np.load(out_pruned / "assembled" / "ns" / "Bin1_Bin1.npz")
        np.testing.assert_array_equal(a_pruned["npairs"], a_full["npairs"])
        for k in ("weight", "meanr", "xip", "xim", "varxi"):
            np.testing.assert_allclose(a_pruned[k], a_full[k], rtol=1e-14, atol=0.0)

        # Removing one active file makes --assemble refuse with INCOMPLETE
        inc = tmp_path / "incomplete"
        shutil.copytree(out_pruned, inc)
        first_active = sorted((inc / "chunks" / "Bin1_Bin1").glob("c*.npz"))[0]
        first_active.unlink()
        (inc / "assembled" / "ns" / "Bin1_Bin1.npz").unlink()
        monkeypatch.setattr(drv, "OUTROOT", inc)
        capsys.readouterr()
        drv.do_assemble(argparse.Namespace(nchunks=4))
        assert "INCOMPLETE" in capsys.readouterr().out
        assert not (inc / "assembled" / "ns" / "Bin1_Bin1.npz").exists()
    finally:
        drv.OUTROOT = orig_outroot


def test_treecorr_gfield_cached_on_catalog(small):
    """TreeCorr 5.1.4 caches GField on Catalog across repeated run_pair calls with the same parameters."""
    other = lens_field(220, 2.0, seed=99)
    c0 = xim_mod.make_treecorr_cat(small, flip_g2=True)
    c1 = xim_mod.make_treecorr_cat(other, flip_g2=True)
    builds = []
    orig_init = treecorr.GField.__init__

    def spy_init(self, *args, **kwargs):
        builds.append(id(args[0]) if args else None)
        return orig_init(self, *args, **kwargs)

    treecorr.GField.__init__ = spy_init
    try:
        xim_mod.run_pair(c0, c0, nbins=NB, bin_slop=0.0, num_threads=THREADS, angle_slop=0.0)
        assert len(builds) == 1
        xim_mod.run_pair(c0, c1, nbins=NB, bin_slop=0.0, num_threads=THREADS, angle_slop=0.0)
        assert len(builds) == 2
        xim_mod.run_pair(c0, c1, nbins=NB, bin_slop=0.0, num_threads=THREADS, angle_slop=0.0)
        assert len(builds) == 2
    finally:
        treecorr.GField.__init__ = orig_init
