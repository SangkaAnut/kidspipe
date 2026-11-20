"""Fast tests for the DR5 production product (scripts/dr5_make_product.py); skip when the product or release is absent."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from kidspipe.measure import official

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / "runs/dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz"
pytestmark = pytest.mark.skipif(not PRODUCT.exists() or not (official.OFFICIAL / "KiDS_Legacy_xipm.fits").exists(),
                                reason="product or official release absent")
spec = importlib.util.spec_from_file_location("mkprod", ROOT / "scripts/dr5_make_product.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)


@pytest.fixture(scope="module")
def prod():
    return dict(np.load(PRODUCT))


@pytest.fixture(scope="module")
def fx():
    return official.load_official_xipm_fits()


def test_order_and_shapes(prod):
    assert prod["pair_ids"].tolist() == [[i + 1, j + 1] for i, j in official.PAIRS]
    assert prod["xip"].shape == prod["npairs"].shape == (21, 1000) and prod["En"].shape == (21, 20) and prod["xip9"].shape == (21, 9)
    assert [str(s) for s in prod["source_files"]] == [f"Bin{i}_Bin{j}.npz" for i, j in prod["pair_ids"]]
    assert len(set(str(s) for s in prod["source_sha256"])) == 21 and str(prod["treecorr_version"]) == "5.1.4" and float(prod["bin_slop"]) == 1.5


def test_m_correction_identity(prod):
    m = np.array([-0.023, -0.016, -0.011, 0.020, 0.030, 0.045])
    mf = np.array([(1 + m[i - 1]) * (1 + m[j - 1]) for i, j in prod["pair_ids"]])
    assert np.allclose(prod["m_factor"], mf, rtol=1e-14)
    assert np.allclose(prod["xip"] * mf[:, None], prod["xip_raw"], rtol=1e-13) and np.allclose(prod["xim"] * mf[:, None], prod["xim_raw"], rtol=1e-13)


def test_validate_passes_on_product(prod, fx):
    mk.validate_product(prod, fx)


def test_shuffled_pair_is_caught(prod, fx):
    bad = {k: v.copy() for k, v in prod.items()}
    for k in ("xip_raw", "xim_raw", "xip", "xim", "npairs", "weight"):       # swap two pairs' fine data, keep their labels
        bad[k][[0, 5]] = bad[k][[5, 0]]
    with pytest.raises(AssertionError):
        mk.validate_product(bad, fx)
    bad2 = {k: v.copy() for k, v in prod.items()}
    bad2["pair_ids"] = bad2["pair_ids"][::-1].copy()                          # wrong pair order
    with pytest.raises(AssertionError):
        mk.validate_product(bad2, fx)
    bad3 = {k: v.copy() for k, v in prod.items()}
    bad3["xip"][3] = bad3["xip_raw"][3]                                       # m-correction missing for one pair
    with pytest.raises(AssertionError):
        mk.validate_product(bad3, fx)


@pytest.fixture(scope="module")
def filt():
    return mk.u.load_context()["filt"]


def test_cosebis_are_the_integral_of_the_stored_xi(prod, filt):
    mk.validate_cosebis(prod, filt)
    for k, f in (("En", 3.0), ("Bn", 0.0)):                                   # a scaled or zeroed COSEBIs array must be caught
        bad = {x: v.copy() for x, v in prod.items()}
        bad[k][7] *= f
        with pytest.raises(AssertionError):
            mk.validate_cosebis(bad, filt)


def test_product_matches_its_source_files(prod):
    src = ROOT / "runs/dr5_measure/xi_unordered/b1.5"
    mk.validate_sources(prod, src)
    bad = {x: v.copy() for x, v in prod.items()}
    bad["xip_raw"][2, 500] *= 1.0001                                          # one altered fine bin
    with pytest.raises(AssertionError):
        mk.validate_sources(bad, src)
