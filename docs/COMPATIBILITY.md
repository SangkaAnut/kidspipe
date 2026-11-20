# Supported versions and what changes between them

kidspipe was developed and validated with Python 3.12, numpy 1.26.4, scipy 1.13.1, numba 0.59.1, CAMB 1.5.8,
TreeCorr 5.1.4 and astropy 6.1.0 (the "workstation" environment). Google Colab now ships Python 3.13, numpy 2.4.6,
scipy 1.16.3, numba 0.64.0 and astropy 7.2.3, and `pip install camb` gives CAMB 2.0.5. Both environments are
supported:

| package | supported | tested |
|---|---|---|
| Python | 3.10 - 3.13 | 3.12.4 (workstation), 3.13.18 (Colab-like) |
| numpy | 1.26, 2.x | 1.26.4, 2.4.6 |
| numba | >= 0.59 (numpy 2.x needs >= 0.60) | 0.59.1, 0.64.0 |
| CAMB | 1.5.8, 2.x | 1.5.8, 2.0.5 |
| TreeCorr | >= 5.1 | 5.1.4 |

## Code changes made for numpy 2 and CAMB 2

* `np.trapz` no longer exists in numpy 2.4. All uses go through `kidspipe._compat.trapezoid`
  (`np.trapezoid` where available, else `np.trapz`).
* No CAMB call needed changing. The CAMB 2.0 release notes name no removed function among those kidspipe uses
  (`CAMBparams`, `set_params`, `set_cosmology`, `InitPower.set_params`, `set_matter_power`, `get_results`,
  `get_linear_matter_power_spectrum`, `get_matter_power_spectrum`, `get_sigma8_0`, `Halofit`, `config.ThreadNum`).

## Numerical differences, CAMB 1.5.8 (numpy 1.26.4) against CAMB 2.0.5 (numpy 2.4.6)

CAMB 2.0 changed several defaults:
* a higher internal accuracy target (`AccuracyTarget = 1`);
* the PRIMAT 2024 BBN helium table (YHe lower by about 2e-4);
* the `recfast_cosmorec` recombination fit;
* in 2.0.3, the mapping from `mnu` to Omega_nu h^2.

The same points were evaluated in both environments with `benchmarks/camb_version_compare.py`. Results are in
`runs/compat/points_camb1.5.8_numpy1.26.4.npz` and `runs/compat/points_camb2.0.5_numpy2.4.6.npz`.

**KV450 points:** 16 in total, namely the test, scan and start presets, the MAP, and 12 samples of the production chain.

| quantity | CAMB 2.0.5 minus CAMB 1.5.8 |
|---|---|
| chi2, exact CAMB linear spectrum (HMcode non-linear) | at most 0.0057 in absolute value (mean +0.0006) |
| chi2 at the MAP | 181.8742 -> 181.8759 |
| sigma8 | at most 9.7e-5 relative (mean +2.4e-5) |
| chi2 at the spec test cosmology (`kidspipe check`) | 196.104 -> 196.104 (anchor 196.06, tolerance 0.5) |
| sigma8 at the spec test cosmology | 0.709092 -> 0.709117 (anchor 0.709092) |
| amplitude scan, spec s4.2 | at most 0.01 per point; all within the 1.0 tolerance |
| HMcode against CAMB `mead` (Delta chi2) | at most 0.017 under both versions (unchanged) |

**KiDS-Legacy COSEBIs:** the 10 highest-posterior rows of the official chain.

| quantity | CAMB 2.0.5 minus CAMB 1.5.8 |
|---|---|
| chi2 of E_n 1-6 (126 points) | at most 0.0021 in absolute value |
| chi2 at the best-posterior row (step D) | 127.2336 -> 127.2336 (release 127.23) |
| sigma8 | -1.8e-4 relative (closer to the official chain's sigma8) |

**Linear-transfer emulator: one artefact per CAMB major version.** The emulator reproduces the CAMB of its training
set, so an artefact trained with one CAMB drifts from the exact spectrum of another. Two artefacts ship, with the same
design (box, Sobol seeds, 1435 training + 256 test runs, 64 + 10 PCA components), and the pipeline loads the one that
matches the installed CAMB major version (`kidspipe.paths.emulator_artefact`; `$KIDSPIPE_EMULATOR` overrides):

| artefact | trained with | held-out max / p99 / median error against dense CAMB (P_lin, k_h < 10, z <= 3) |
|---|---|---|
| `data/emulator/transfer_emulator_v1.npz` | CAMB 1.5.8 | 1.83e-3 / 1.08e-3 / 9.8e-5 |
| `data/emulator/transfer_emulator_v1_camb2.npz` | CAMB 2.0.5 | 1.62e-3 / 1.38e-3 / 1.6e-4; sigma8 within 1.3e-5 |

The acceptance target is 2e-3; both artefacts meet it. Validation and the training logs are in
`runs/emulator_train/validation.json` and `runs/emulator_train_camb2/{validation.json,train.log}`. Training the CAMB 2.0.5
artefact took 28 min with 20 workers.

Against exact CAMB at the 16 KV450 points, the largest |Delta ln L| of the emulator is:

| | CAMB 1.5.8 | CAMB 2.0.5 |
|---|---|---|
| CAMB-1.5.8 artefact | 5.8e-4 | 3.3e-3 (wrong pairing; P_lin up to 1.6 % off at the low-omch2 corner of the box) |
| CAMB-2.0.5 artefact | - | 5.7e-4 (Delta chi2 0.0011) |

(`runs/compat/points_camb2.0.5_numpy2.4.6_emulator-camb2.npz`.) A CAMB release after 2.0.5 still loads the CAMB-2
artefact and warns at load time if its exact version differs from the training stamp; retrain with
`scripts/train_emulator.py` (it writes the artefact for the installed major version) if that warning appears.

## Tests that depend on the CAMB version

The spec-anchor tests (`tests/test_linear.py::test_spec_sanity_point`, `::test_vs_spec_anchor_test_cosmology`,
`tests/test_nonlinear.py::test_spec_anchor_pnl`, `tests/test_e2e.py::test_derived_quantities_test_cosmology`) assert values
at tolerances down to 1e-9. The spec anchors were made with CAMB 1.x. Under CAMB >= 2 the same tests compare with values
measured by the same code under CAMB 2.0.5 (`data/validation/anchors_camb2.json`, `benchmarks/measure_camb2_anchors.py`,
`tests/_anchors.py`), **with unchanged tolerances**. The largest moves from the CAMB 1.5.8 values: P_lin(k = 0.1005) +2.1e-4,
P_nl(k = 1.003) +9.7e-5, omega_nu h^2 +5.1e-5, sigma8 and S8 +3.5e-5. A CAMB 2.x release that moves them again would make
these tests fail; re-measure with the benchmark script after checking the change.

## numba >= 0.60: bitwise determinism of the HMcode kernels

With numba 0.64, `fastmath=True` let LLVM reassociate sums inside the HMcode kernels differently depending on memory
alignment, so results changed by about 1 ulp with the thread count or the order of the redshifts. The kernels now use
fastmath without `reassoc`. They are again bitwise identical across thread counts, input layout and redshift order, on
numba 0.59 and 0.64. This changes P_nl by at most 2e-14 relative, with no measurable speed change.

## BLAS threads with pip's numpy (OpenBLAS)

pip's numpy ships OpenBLAS, which by default starts one thread per core even for small matrix products. On a 22-core
machine this made one emulator evaluation take about 22 ms instead of 3 ms (best of 30). Anaconda's numpy, which uses
MKL, and 2-8 vCPU Colab runtimes are not affected. The `kidspipe` CLI sets `OMP_NUM_THREADS=4` unless it is already
set, and OpenBLAS honours that; the tests do the same (`tests/conftest.py`). When calling the Python API on a many-core
machine, set `OMP_NUM_THREADS` (or `OPENBLAS_NUM_THREADS`) before importing numpy.

## Test suite
