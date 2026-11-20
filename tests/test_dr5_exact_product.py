"""Tests for the DR5 exact product (scripts/dr5_make_exact_product.py); skip when the product, its report, the assembled or chunk
files or the release are absent.  No gate is tested (PI decision 2026-11-02: the exact run is reported, not gated)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from kidspipe.measure import official

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / "runs/dr5_measure/product_exact/dr5_xipm_cosebis_exact.npz"
REPORT = ROOT / "runs/dr5_measure/product_exact/exact_vs_b1p5.json"
B15 = ROOT / "runs/dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz"
ASM = ROOT / "runs/dr5_measure/xi_chunked/assembled/ns"
CHUNKS = ROOT / "runs/dr5_measure/xi_chunked/chunks"
pytestmark = pytest.mark.skipif(not all(f.exists() for f in (PRODUCT, REPORT, B15, ASM))
                                or not (official.OFFICIAL / "KiDS_Legacy_xipm.fits").exists(),
                                reason="exact product, its report, assembled files or official release absent")
# the raw chunk files (346 MB) are not shipped: only the test that re-checks them needs them
needs_chunks = pytest.mark.skipif(not CHUNKS.exists(), reason="raw chunk files (runs/dr5_measure/xi_chunked/chunks) absent: not shipped")


@pytest.fixture(scope="module")
def mx():
    """scripts/dr5_make_exact_product.py, loaded only when the tests actually run (never at collection)."""
    spec = importlib.util.spec_from_file_location("mkexact", ROOT / "scripts" / "dr5_make_exact_product.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prod():
    return dict(np.load(PRODUCT))


@pytest.fixture(scope="module")
def b15():
    return dict(np.load(B15))


@pytest.fixture(scope="module")
def ctx(mx):
    return mx.u.load_context()


def copy(d):
    return {k: v.copy() for k, v in d.items()}


def test_settings_and_shapes(prod):
    assert prod["pair_ids"].tolist() == [[i + 1, j + 1] for i, j in official.PAIRS]
    assert float(prod["bin_slop"]) == 0.0 and float(prod["angle_slop"]) == 0.0 and str(prod["pairing"]) == "unordered"
    assert prod["xip"].shape == prod["varxi"].shape == prod["meanr"].shape == (21, 1000)
    assert prod["En"].shape == (21, 20) and prod["xip9"].shape == (21, 9)
    prov = json.loads(str(prod["provenance_json"]))
    assert prov["gate"].startswith("none") and prov["nchunks"] == 16


def test_validate_passes(prod, b15, ctx, mx):
    mx.validate_exact_product(prod, b15, ctx["fx"])


def test_mixups_are_caught(prod, b15, ctx, mx):
    bad = copy(prod)
    for k in ("xip_raw", "xim_raw", "xip", "xim", "npairs", "weight", "varxi", "xip9", "xim9"):    # swap two pairs, keep labels
        bad[k][[0, 5]] = bad[k][[5, 0]]
    with pytest.raises(AssertionError):
        mx.validate_exact_product(bad, b15, ctx["fx"])
    bad = copy(prod); bad["xip"][3] = bad["xip_raw"][3]                                              # m-correction missing
    with pytest.raises(AssertionError):
        mx.validate_exact_product(bad, b15, ctx["fx"])


def test_consistently_wrong_m_is_caught(prod, b15, ctx, mx):
    """m values shifted and every m-dependent array recomputed consistently: only the anchor to the pipeline's m catches it."""
    bad = copy(prod)
    bad["m_gold"] = bad["m_gold"] + 0.01
    mf = np.array([(1 + bad["m_gold"][i - 1]) * (1 + bad["m_gold"][j - 1]) for i, j in bad["pair_ids"]])
    bad["m_factor"] = mf
    for s in ("xip", "xim"):
        bad[s] = bad[s + "_raw"] / mf[:, None]
        bad[s + "9"] = np.array([mx.u.rebin9(bad["r_nom"][k], bad["npairs"][k], bad[s][k]) for k in range(21)])
    with pytest.raises(AssertionError, match="m_gold"):
        mx.validate_exact_product(bad, b15, ctx["fx"])


def test_auto_pairs_are_unordered(prod, b15, ctx, mx):
    """The driver's ordered auto counts must have been halved: totals match the unordered bin_slop 1.5 product to 1e-4."""
    r = prod["npairs"].sum(1) / b15["npairs"].sum(1)
    assert np.all(np.abs(r - 1) < 1e-4)
    bad = copy(prod)
    autos = [k for k, (i, j) in enumerate(prod["pair_ids"]) if i == j]
    bad["npairs"][autos] *= 2.0                                                                      # forgotten conversion
    with pytest.raises(AssertionError):
        mx.validate_exact_product(bad, b15, ctx["fx"])


def test_meanr_is_checked(prod, b15, ctx, mx):
    bad = copy(prod); bad["meanr"][0, 400] *= 1.1                                                   # outside its fine bin
    with pytest.raises(AssertionError, match="meanr"):
        mx.validate_exact_product(bad, b15, ctx["fx"])


def test_cosebis_are_the_integral_of_the_stored_xi(prod, ctx, mx):
    mx.validate_cosebis(prod, ctx["filt"])
    bad = copy(prod); bad["En"][7] *= 3.0
    with pytest.raises(AssertionError):
        mx.validate_cosebis(bad, ctx["filt"])


def test_product_matches_assembled_files(prod, mx):
    mx.validate_sources(prod, srcdir=ASM)
    bad = copy(prod); bad["varxi"][0, 400] *= 1.0001                                                # pair 1-1 is an auto pair
    with pytest.raises(AssertionError):
        mx.validate_sources(bad, srcdir=ASM)


@needs_chunks
def test_chunks_are_exact_and_complete(monkeypatch, mx):
    rows, meanr = mx.check_chunks()
    assert len(rows) == 21 and all(r["chunk_files"] == (136 if a == b else 256)
                                   for (a, b), r in zip(mx.PAIRS1, rows.values()))
    monkeypatch.setattr(mx.ch, "chunk_slops", lambda z: (1.5, 0.1))                                  # chunk files at bin_slop 1.5
    with pytest.raises(AssertionError, match="expected exact"):
        mx.check_chunks()


def _leaves(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            yield from _leaves(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _leaves(x, y, f"{path}[{i}]")
    else:
        yield path, a, b


def test_report_matches_product(prod, b15, ctx, mx):
    """The stored comparison (JSON and the markdown table) is what the current product gives (catches a stale report)."""
    rep = json.loads(REPORT.read_text())
    per, full, arrays, decomp, fw = mx.compare(prod, b15, ctx)
    new = {"per_pair": per, "fullvector": full, "decomposition": decomp, "filter_weighting": fw, "arrays": arrays,
           "edge": mx.edge_counts(per)}
    n = 0
    for path, a, b in _leaves(new, {k: rep[k] for k in new}):
        n += 1
        if isinstance(a, str):
            assert a == b, path
        else:
            assert np.isclose(a, b, rtol=1e-10, atol=1e-14), (path, a, b)
    assert n > 1000
    md = (REPORT.parent / "EXACT_VS_B1P5.md").read_text()
    for p in ("1-1", "3-3", "6-6"):
        assert mx.md_row(p, per[p]) in md, f"markdown row {p} is stale"
    assert rep["fullvector"]["Bnull_exact"]["pte_6m"] == pytest.approx(mx.bnull(prod["Bn"], ctx["cb"]["cov"])["pte_6m"], rel=1e-12)
