"""Tests for kidspipe.theory.nz (spec s3): loading, normalisation, photo-z shift."""
import numpy as np
from kidspipe._compat import trapezoid
import pytest
from scipy.interpolate import interp1d

from kidspipe.theory.nz import (NZ, NZ_FILES, default_nz_dir, load_nz, normalise_nz,
                                shift_nz, shift_nz_batch)


@pytest.fixture(scope="module")
def nz():
    return load_nz()


def test_loading_nodes_and_norm(nz):
    assert nz.z.size == 121 and nz.n.shape == (5, 121)
    assert nz.z[0] == 0.0
    assert nz.z[1] == pytest.approx(0.025, abs=1e-14)
    assert nz.z[2] == pytest.approx(0.075, abs=1e-14)
    assert nz.z[-1] == pytest.approx(5.975, abs=1e-12)
    assert np.all(np.diff(nz.z) > 0)
    np.testing.assert_allclose(trapezoid(nz.n, nz.z, axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.all(nz.n[:, 0] == 0.0)
    assert np.all(nz.n >= 0)


def test_loading_matches_raw_files_and_renormalisation_size(nz):
    d = default_nz_dir()
    expected_trapz = [0.9945, 0.9981, 0.9985, 0.9994, 0.9998]  # VALIDATION_TARGETS: trapz of raw n
    for i, name in enumerate(NZ_FILES):
        raw = np.loadtxt(d / name)
        assert raw.shape == (120, 2)
        assert raw[0, 0] == 0.0 and raw[-1, 0] == pytest.approx(5.95)
        assert raw[:, 1].sum() * 0.05 == pytest.approx(1.0, abs=2e-6)  # raw sum*dz = 1
        zc = np.concatenate([[0.0], raw[:, 0] + 0.025])
        nr = np.concatenate([[0.0], raw[:, 1]])
        tz = trapezoid(nr, zc)
        # VALIDATION_TARGETS quotes trapz of the raw (unshifted) tabulation
        assert trapezoid(raw[:, 1], raw[:, 0]) == pytest.approx(expected_trapz[i], abs=1.5e-4)
        assert 0.99 < tz < 1.0  # the shift+prepend trapz is a 0.1-0.5 % renormalisation
        np.testing.assert_allclose(nz.z, zc, atol=1e-13)
        np.testing.assert_allclose(nz.n[i], nr / tz, rtol=1e-13, atol=0)


def test_mean_redshifts_increase(nz):
    m = nz.mean_z()
    assert np.all(np.diff(m) > 0)
    assert 0.2 < m[0] < 0.4 and 0.8 < m[-1] < 1.2


def test_normalise_nz_batch():
    z = np.linspace(0, 3, 50)
    n = np.random.default_rng(0).uniform(0.1, 2, (3, 50))
    out = normalise_nz(z, n)
    np.testing.assert_allclose(trapezoid(out, z, axis=1), 1.0, atol=1e-14)


def test_shift_zero_is_identity(nz):
    np.testing.assert_allclose(nz.shifted(np.zeros(5)), nz.n, rtol=0, atol=1e-13)
    np.testing.assert_allclose(shift_nz(nz.z, nz.n[2], 0.0), nz.n[2], atol=1e-13)


def test_shift_matches_scipy_interp1d_reference(nz):
    """Spec s3.2: interp1d(linear, bounds_error=False, fill_value=0)(z - dz) then trapz renormalisation."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        dz = rng.uniform(-0.1, 0.1, 5)
        ref = np.empty_like(nz.n)
        for i in range(5):
            f = interp1d(nz.z, nz.n[i], kind="linear", bounds_error=False, fill_value=0.0)
            r = f(nz.z - dz[i])
            ref[i] = r / trapezoid(r, nz.z)
        np.testing.assert_allclose(nz.shifted(dz), ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(shift_nz_batch(nz.z, nz.n, dz), ref, rtol=1e-12, atol=1e-14)
        for i in range(5):
            np.testing.assert_allclose(shift_nz(nz.z, nz.n[i], dz[i]), ref[i], rtol=1e-12, atol=1e-14)


def test_shift_normalised_nonnegative_and_direction(nz):
    dz = np.array([0.05, 0.03, -0.02, 0.1, -0.1])
    sh = nz.shifted(dz)
    np.testing.assert_allclose(trapezoid(sh, nz.z, axis=1), 1.0, atol=1e-12)
    assert np.all(sh >= 0)
    dmean = nz.mean_z(sh) - nz.mean_z()
    # positive dz moves to higher z; the mean moves by ~dz (small tail truncation)
    assert np.all(np.sign(dmean) == np.sign(dz))
    np.testing.assert_allclose(dmean, dz, atol=3e-3)


def test_shift_exact_grid_step():
    """On a uniform grid a shift by an integer number of nodes is an exact relabelling."""
    z = np.arange(0, 6.0, 0.05)
    n = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
    n /= trapezoid(n, z)
    out = shift_nz(z, n, 0.25, renormalise=False)
    np.testing.assert_allclose(out[5:], n[:-5], atol=1e-14)
    assert np.all(out[:5] == 0.0)
    # without truncation mass the renormalisation changes nothing
    out_r = shift_nz(z, n, 0.25)
    np.testing.assert_allclose(out_r, out, rtol=1e-6)


def test_zero_fill_and_truncation(nz):
    sh = shift_nz(nz.z, nz.n[0], 0.1, renormalise=False)
    assert np.all(sh[nz.z < 0.1] == 0.0)  # nothing below z = dz
    sh = shift_nz(nz.z, nz.n[4], -0.1, renormalise=False)
    assert np.all(sh[nz.z > nz.z[-1] - 0.1 + 1e-12] == 0.0)  # nothing above z_max + dz
    # mass pushed out of the table is redistributed by the renormalisation
    raw = shift_nz(nz.z, nz.n[0], 0.1, renormalise=False)
    ren = shift_nz(nz.z, nz.n[0], 0.1)
    np.testing.assert_allclose(ren, raw / trapezoid(raw, nz.z), rtol=1e-13)


def test_batch_scalar_broadcast_and_class(nz):
    a = nz.shifted(0.02)
    b = nz.shifted(np.full(5, 0.02))
    np.testing.assert_allclose(a, b)
    assert isinstance(nz, NZ) and nz.nbin == 5


def test_load_from_custom_dir_without_zero_row(tmp_path):
    z = np.arange(0, 3.0, 0.05)
    for i, name in enumerate(NZ_FILES):
        n = np.exp(-0.5 * ((z + 0.025 - 0.5 - 0.1 * i) / 0.2) ** 2)
        np.savetxt(tmp_path / name, np.c_[z, n])
    nz = load_nz(tmp_path)
    assert nz.z.size == z.size + 1 and nz.z[0] == 0.0
    np.testing.assert_allclose(trapezoid(nz.n, nz.z, axis=1), 1.0, atol=1e-13)


def test_shift_out_of_table_or_nan_raises_not_silent_nan(nz):
    """A shift that removes the whole distribution (or NaN) must raise, not return NaN rows silently."""
    for bad in (10.0, -10.0, np.nan):
        with pytest.raises(ValueError):
            nz.shifted(np.full(5, bad))
        with pytest.raises(ValueError):
            shift_nz(nz.z, nz.n[0], bad)
    with pytest.raises(ValueError):
        normalise_nz(nz.z, np.zeros((2, nz.z.size)))
    with pytest.raises(ValueError):
        nz.shifted(np.zeros(4))  # wrong number of bins


def test_negative_shift_fills_z0_node_by_interpolation(nz):
    """n'(0) = n(0 - dz) is nonzero for dz < 0 (linear interpolation, spec s3.2), then renormalised."""
    dz = -0.025
    sh = shift_nz(nz.z, nz.n[0], dz, renormalise=False)
    assert sh[0] == pytest.approx(nz.n[0, 1], rel=1e-14)  # z=0 -> z - dz = 0.025 node
    # node 1 (z=0.025) maps to z=0.05, the midpoint of the nodes 0.025 and 0.075
    assert sh[1] == pytest.approx(0.5 * (nz.n[0, 1] + nz.n[0, 2]), rel=1e-14)
    # beyond the first (0.025-wide) interval the grid is uniform with step 0.05: shift by half a step
    np.testing.assert_allclose(sh[2:-1], 0.5 * (nz.n[0, 2:-1] + nz.n[0, 3:]), rtol=1e-12, atol=1e-14)
    assert sh[-1] == 0.0 and trapezoid(shift_nz(nz.z, nz.n[0], dz), nz.z) == pytest.approx(1.0, abs=1e-13)


def test_kidspipe_data_env_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("KIDSPIPE_DATA", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        default_nz_dir()
