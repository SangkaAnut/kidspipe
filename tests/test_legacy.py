"""KiDS-Legacy COSEBIs pipeline tests (fast; one CAMB run for the official-best-sample check)."""
import numpy as np
import pytest

from kidspipe.legacy import (LegacyCosebisLikelihood, LegacyCosebisModel, LegacyData, chain_samples, ia_amplitudes,
                             load_chain)
from kidspipe.legacy.data import PAIRS
from kidspipe.theory.cosebis import CosebisFilters


def test_ia_amplitudes():
    A = ia_amplitudes(5.0, 0.5, [13.5] * 6)
    assert np.allclose(A, 5.0 * np.array([0.158, 0.198, 0.206, 0.258, 0.207, 0.026]))
    A2 = ia_amplitudes(2.0, 1.0, [14.5] * 6, f_r=np.ones(6))
    assert np.allclose(A2, 20.0)


def test_data_layout():
    d = LegacyData.load()
    assert d.value.shape == (420,) and d.cov.shape == (840, 840) and d.nz.shape == (6, 119)
    like = LegacyCosebisLikelihood()
    assert like.n_data == 126 and len(PAIRS) == 21
    assert like.chi2(like.d) == pytest.approx(0.0, abs=1e-12)
    assert like.chi2(np.zeros(126)) > 500


def test_filters_2_300_conditions():
    f = CosebisFilters(2.0, 300.0, 6, source="mpmath")
    G = f.gram(6)
    tgt = (300.0 - 2.0) * np.eye(6)
    assert np.max(np.abs(G - tgt)) / 298.0 < 1e-9
    c2, c4, sc = f.constraints(6)
    assert np.max(np.abs(c2) / sc) < 1e-9 and np.max(np.abs(c4) / sc) < 1e-9


def test_official_best_sample_chi2():
    ch = load_chain()
    ok = np.isfinite(ch[:, 43]) & np.isfinite(ch[:, 22])
    row = np.where(ok)[0][np.argmax(ch[ok, 43])]
    s = chain_samples(ch, [row])[0]
    m = LegacyCosebisModel(cache_dir="runs/legacy/cache")
    like = LegacyCosebisLikelihood()
    c2 = like.chi2(m.mu(s))
    assert abs(c2 - s["chi2_off"]) < 0.5
    assert abs(m.last_derived["sigma_8"] / s["sigma_8_off"] - 1) < 3e-3
