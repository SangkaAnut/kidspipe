"""Fast tests for scripts/dr5_measure_xi_unordered.py (synthetic catalogues, <= 2 TreeCorr threads; release-dependent
tests skip when the official files are absent)."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

treecorr = pytest.importorskip("treecorr")

from kidspipe.measure import official, xi as xim_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unord", ROOT / "scripts/dr5_measure_xi_unordered.py")
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
needs_release = pytest.mark.skipif(not (official.OFFICIAL / "KiDS_Legacy_xipm.fits").exists(), reason="official release absent")


def cat_rec(n, seed):
    rng = np.random.default_rng(seed)
    return dict(ra=150 + rng.uniform(0, 2, n), dec=rng.uniform(0, 2, n), e1=rng.normal(0, .25, n), e2=rng.normal(0, .25, n),
                w=rng.uniform(.5, 1.5, n))


def test_parse_pairs_and_index():
    assert u.parse_pairs("autos") == [(i, i) for i in range(1, 7)]
    assert len(u.parse_pairs("all")) == 21 and u.parse_pairs("3-1,2-2") == [(1, 3), (2, 2)]
    assert [u.pair_index(p) for p in u.parse_pairs("all")] == list(range(21))
    assert u.pair_index((1, 1)) == 0 and u.pair_index((6, 6)) == 20 and u.pair_index((1, 6)) == 5
    assert u.mfac((1, 6)) == pytest.approx((1 - 0.023) * 1.045)


def test_unordered_auto_is_half_of_ordered():
    kw = xim_mod.treecorr_kwargs(nbins=10, bin_slop=1.5, num_threads=2)
    cat = xim_mod.make_treecorr_cat(cat_rec(3000, 1))
    a = treecorr.GGCorrelation(kw); a.process(cat)
    o = treecorr.GGCorrelation(kw); o.process(cat, cat)
    assert np.allclose(o.npairs, 2 * a.npairs, rtol=0, atol=1e-9) and np.allclose(o.weight, 2 * a.weight)
    assert np.allclose(o.xip, a.xip, rtol=1e-10, atol=1e-14) and np.allclose(o.xim, a.xim, rtol=1e-10, atol=1e-14)


def test_run_matches_direct_treecorr_and_resumes(tmp_path, monkeypatch):
    recs = {b: cat_rec(2000, b) for b in (1, 2)}
    monkeypatch.setattr(xim_mod, "load_bin", lambda b, work=None: recs[b])
    monkeypatch.setattr(u, "OUTROOT", tmp_path)
    monkeypatch.setattr(u, "_load_chunked", lambda: type("D", (), {"slop_record": staticmethod(lambda b, a: {"bin_slop": np.float64(b)})}))
    ns = lambda **k: argparse.Namespace(**{**dict(pairs="1-1,1-2", bin_slop=1.5, threads=2, force=False, subsample=0, seed=0), **k})
    u.do_run(ns())
    out = tmp_path / "b1.5"
    kw = xim_mod.treecorr_kwargs(bin_slop=1.5, num_threads=2)
    c1, c2 = (xim_mod.make_treecorr_cat(recs[b]) for b in (1, 2))
    for name, args in (("Bin1_Bin1", (c1,)), ("Bin1_Bin2", (c1, c2))):
        g = treecorr.GGCorrelation(kw); g.process(*args)
        d = np.load(out / f"{name}.npz")
        assert np.array_equal(d["npairs"], g.npairs) and np.allclose(d["xip"], g.xip, rtol=0, atol=1e-12)
        assert str(d["pairing"]) == "unordered" and "treecorr_version" in d.files and "seconds" in d.files
    t = (out / "Bin1_Bin1.npz").stat().st_mtime_ns
    u.do_run(ns())
    assert (out / "Bin1_Bin1.npz").stat().st_mtime_ns == t           # resumable: existing pair skipped
    assert (out / "manifest.json").exists()


@needs_release
def test_block_selection_matches_fits_labels():
    from astropy.io import fits
    ctx = u.load_context()
    with fits.open(official.OFFICIAL / "KiDS_Legacy_xipm.fits") as h:
        t = h["xiP"].data; tm = h["xiM"].data
        cov = 0.5 * (np.array(h["COVMAT"].data) + np.array(h["COVMAT"].data).T)
    for p in [(1, 1), (2, 5), (3, 6), (6, 6)]:
        rows = np.where((t["BIN1"] == p[0]) & (t["BIN2"] == p[1]))[0]
        rows = rows[np.argsort(t["ANGBIN"][rows])]
        b = u.pair_blocks(ctx, p)
        assert np.allclose(b["val9"]["xip"], t["VALUE"][rows]) and np.allclose(b["val9"]["xim"], tm["VALUE"][rows])
        assert np.allclose(b["sig9"]["xip"], np.sqrt(np.diag(cov)[rows]))
        assert np.allclose(b["sig9"]["xim"], np.sqrt(np.diag(cov)[189 + rows]))


@needs_release
def test_eval_scale_invariant_and_ordered_normalisation():
    ctx = u.load_context(); p = (4, 4); K = u.pair_index(p)
    r = ctx["asc"]["r_nom"][K]
    nz = np.random.default_rng(3).normal(0, 1, (2, r.size))
    o = dict(r_nom=r, xip=ctx["asc"]["xip"][K] * 1.1 + 0.3 * ctx["asc"]["sigma_xip"][K] * nz[0],
             xim=ctx["asc"]["xim"][K] * 1.1 + 0.3 * ctx["asc"]["sigma_xim"][K] * nz[1],
             npairs=ctx["asc"]["npairs_weighted"][K].copy(), weight=ctx["asc"]["weight"][K].copy())
    o2 = {**o, "npairs": 2 * o["npairs"], "weight": 2 * o["weight"]}
    th = dict(o)
    a = u.evaluate_pair(p, o, False, ctx, th); b = u.evaluate_pair(p, o2, True, ctx, th)
    for k in ("xip", "xim"):
        assert a["A3"][k]["max"] == pytest.approx(b["A3"][k]["max"], abs=1e-12)
    assert a["B1"]["max_En_20"] == pytest.approx(b["B1"]["max_En_20"], abs=1e-12)
    assert a["vs_theirs"]["npairs_ratio_unordered"] == pytest.approx(1.0) == pytest.approx(b["vs_theirs"]["npairs_ratio_unordered"])
    assert a["A1"]["boxcar_over_white"] == b["A1"]["boxcar_over_white"] and a["A1"]["A_p_xip"] == pytest.approx(1.1, rel=0.05)


def test_truncated_pair_file_is_remeasured(tmp_path, monkeypatch):
    recs = {1: cat_rec(1500, 1)}
    monkeypatch.setattr(xim_mod, "load_bin", lambda b, work=None: recs[b])
    monkeypatch.setattr(u, "OUTROOT", tmp_path)
    monkeypatch.setattr(u, "_load_chunked", lambda: type("D", (), {"slop_record": staticmethod(lambda b, a: {"bin_slop": np.float64(b)})}))
    ns = argparse.Namespace(pairs="1-1,1-1", bin_slop=1.5, threads=2, force=False, subsample=0, seed=0)
    out = tmp_path / "b1.5"; out.mkdir()
    (out / "Bin1_Bin1.npz").write_bytes(b"PK\x03\x04truncated")
    u.do_run(ns)
    assert u.pair_file_ok(out / "Bin1_Bin1.npz") and not list(out.glob("*.tmp*"))
    import json
    m = json.loads((out / "manifest.json").read_text())
    assert "Bin1_Bin1" in m["pairs"] and m["pairs"]["Bin1_Bin1"]["nobj1"] == 1500 and m["calls"][0]["threads"] == 2
    assert u.parse_pairs("1-1,1-1") == [(1, 1)]


@needs_release
def test_theirs_bin6_pins_known_gate_values():
    d = u.THEIRS / "xipm_bin6_bin6.npz"
    if not d.exists():
        pytest.skip("their npz absent")
    ctx = u.load_context()
    R = u.evaluate_pair((6, 6), u.load_theirs((6, 6)), False, ctx)
    assert R["A3"]["xip"]["max"] == pytest.approx(0.0238, abs=1e-3) and R["A3"]["xim"]["max"] == pytest.approx(0.0438, abs=1e-3)
    assert R["B1"]["max_En_20"] == pytest.approx(0.043, abs=1e-3)
    assert max(R["A1"]["boxcar_over_white"]) == pytest.approx(0.453, abs=1e-3)
