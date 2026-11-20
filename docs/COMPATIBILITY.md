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

**Linear-transfer emulator.** The shipped artefact `data/emulator/transfer_emulator_v1.npz` was trained on CAMB 1.5.8.
Inside its box it does not call CAMB, so its output is the same under both versions. What changes is its error against
the exact spectrum of the CAMB that is installed:

| | CAMB 1.5.8 | CAMB 2.0.5 |
|---|---|---|
| max \|Delta ln L\| emulator against exact, 16 points | 5.8e-4 | 3.3e-3 (Delta chi2 0.0065) |

Under CAMB 2 the emulator reproduces CAMB 1.5.8 rather than the installed CAMB. The difference is far below anything
that moves a posterior (Delta chi2 < 0.01), but it exceeds the emulator's documented accuracy against its own
training CAMB (Delta ln L <= 1.1e-3). The package warns about this at load time (`RuntimeWarning: transfer emulator
... was trained with CAMB 1.5.8 but CAMB 2.0.5 is installed`). For an emulator matched to CAMB 2, retrain it with
`scripts/train_emulator.py`: about 17 core-hours to generate the training set, then the fit.

## Test suite

See the end of this file for the results of the full suite in both environments.
