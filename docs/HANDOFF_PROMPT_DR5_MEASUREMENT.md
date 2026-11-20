# Handoff prompt: measure KiDS-DR5 (KiDS-Legacy) shear two-point statistics and cross-check with the official release

> Paste everything below the line into the local agent. It is self-contained.

---

## 0. Who you are and how to work

You are continuing a scientific software project that a previous agent (Claude) built and validated. You have unlimited tokens but the machine is shared and modest: **22 CPU cores, 15 GB RAM, WSL2, data on `/mnt/d` (Windows drvfs: slow random I/O)**. Work carefully and verifiably:

- **Never claim a result you did not compute.** Every number in your report must come from a file you wrote under `runs/dr5_measure/` (JSON/NPZ), produced by a script you can name.
- After each step, **run the check listed for that step** and record pass/fail. If a check fails, stop and diagnose before continuing; do not loosen a tolerance to make it pass without writing down why.
- Look at every plot you make (open the PNG) and fix unreadable ones.
- Keep CPU use to at most 16 cores and memory below about 10 GB. **Never load the full catalogue into memory** (see section 3).
- Do not modify existing validated code paths in `src/kidspipe/` except through backward-compatible additions. Run the test suite at the end (`PYTHONPATH=src python3 -m pytest -q -m "not slow"`; it must still pass: 300 tests at handoff).

## 0.1 Python interpreter (important)

Your shell's `python3` may be a bare system Python (3.14, no numpy). **Always use the project's validated interpreter:**

```
PY=/home/anut/anaconda3/bin/python3      # Python 3.12.4, numpy 1.26.4, scipy 1.13.1, astropy 6.1.0, camb 1.5.8, numba 0.59.1, mpmath
$PY -m pip install --user treecorr       # install TreeCorr into THIS interpreter only
cd /mnt/d/cloning_kids_Kappa_pipeline && PYTHONPATH=src $PY -m pytest -q -m "not slow"   # baseline: 300 pass
```

Do not use the other conda envs (`hi_lensing`, `cov20_env`, `~/.claude-science/...`), and do not upgrade numpy: numba 0.59 needs numpy < 1.27 and the emulator artefact is tied to camb 1.5.8. Run the test suite first to confirm the baseline before you change anything.

## 1. Project state (what already exists and works)

Project root: `/mnt/d/cloning_kids_Kappa_pipeline` (note the capital K). It is not a git repo.

`kidspipe` is a clean-room Python cosmic-shear inference pipeline. It was rebuilt from the mathematics of the KiDS `kcap` pipeline and is validated:

| component | where | status |
|---|---|---|
| KV450 xi+/- theory + likelihood | `src/kidspipe/theory/*`, `likelihood/*` | chi2 = 196.10 vs spec anchor 196.06; posterior S8 = 0.7358 +- 0.0374 (published 0.737) |
| COSEBIs theory (filters T_+-n, W_n, E_n from C_ell and from xi) | `src/kidspipe/theory/cosebis.py`, `likelihood/cosebis.py` | T_+ matches the official tables to 1e-16; two E_n routes agree to 2e-4 sigma |
| KiDS-Legacy COSEBIs model (6 bins, HMcode-2020 + T_AGN, mass-dependent IA) | `src/kidspipe/legacy/` | reproduces the official chain chi2 sample by sample: mean diff -0.26, std 0.79, corr 0.972 over 3509 samples; best sample 127.23 vs 127.29; reweighted S8 0.8142 vs official 0.8128 |

Read these first: `README.md`, `docs/DESIGN.md`, `docs/COSEBIS.md`, `docs/KIDS_LEGACY_VALIDATION.md`, `docs/KIDS_LEGACY_REVIEW.md`, `docs/interface_notes/cosebis.md`, `docs/interface_notes/legacy.md`.

Useful existing functions (check the exact signatures in the source):

- `kidspipe.theory.cosebis.CosebisFilters`: T_+n(theta) and T_-n(theta) for any [theta_min, theta_max], with `source='mpmath'` to regenerate roots and norms. KiDS-Legacy uses **2.00-300.00 arcmin**.
- `kidspipe.theory.cosebis`: E_n and B_n from xi_+/-(theta) on a fine theta grid (`E_n = 1/2 int dtheta theta [T_+n xi_+ + T_-n xi_-]`, `B_n = 1/2 int ... [T_+n xi_+ - T_-n xi_-]`, theta in radians).
- `kidspipe.legacy.data`: loader for the official COSEBIs FITS file.

So far everything is **theory**. Nothing has yet been **measured** from the galaxy catalogue. That is your task.

## 2. Data (read-only; never modify these files)

Base: `/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/`

- `KiDS_Legacy_NS_unblind_final.fits`: the shear catalogue, **16.4 GB**, single binary table. Read `KiDS_Legacy_NS_unblind_final.readme.txt` first.
  - Relevant columns: `RAJ2000`, `DECJ2000` (deg), `PATCH` (North/South), `TOMOBIN` (tomographic bin index; find out which values mean "not selected"), `e1`, `e2` ("c-corrected and recalibrated Lensfit ellipticity, **no m correction**"), `weight` ("final source weight, including shear and gold weights"), `Z_B`.
  - Also `shear_weight_only`, `gold_weight_only` and the PSF columns, for null tests.
- `KiDs_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/` contains the **official measurements** (`README.txt` explains them):
  - `data/cosmosis_xipm/XI_KiDS_Legacy_NScomb_nBins_6_Bin{i}_Bin{j}.ascii`, 21 files for i <= j. TreeCorr output with columns `r_nom meanr meanlogr xip xim xip_im xim_im sigma_xip sigma_xim npairs weight npairs_weighted`. They have **1000 log bins between 2 and 300 arcmin** (North+South combined). **This is the most direct comparison target.**
  - `data/KiDS_Legacy_cosebis.fits`: `En` and `Bn` (6 bins, 21 pairs x 20 modes, theta 2-300'), `COVMAT` (840 x 840 = [En | Bn]), `NZ_SOURCE`.
  - `data/KiDS_Legacy_xipm.fits`: `xiP` and `xiM` (21 pairs x 9 bins, centres 2.64 ... 227.1'), `COVMAT` 378 x 378.
  - `data/KiDS_Legacy_bandpowers.fits`: `PeeE`, `PeeB` (21 x 8), `COVMAT` 336 x 336. Optional extra target.
  - `chains_and_config_files/*/KiDS_Legacy_*_pipeline.ini`: the official settings (theta ranges, n_max = 6 for COSEBIs, etc.).
  - Copies of the COSEBIs FITS file and config are also in `data/kids_legacy/` inside the project.

Papers, for definitions only (read via arXiv; do not copy anyone's code): Wright et al. 2025 (KiDS-Legacy cosmic shear, arXiv:2503.19441), Wright et al. 2025 (DR5), Reischke et al. 2025 (covariance). Schneider, Eifler & Krause 2010 (COSEBIs). Use the papers for the selection, the weights, the m-bias treatment and the theta binning conventions.

## 3. Engineering constraints and setup

1. **Install TreeCorr** into the user environment (`python3 -m pip install --user treecorr`) and record its version. TreeCorr is the reference estimator the official release used.
   - The **primary path is your own independent estimator**, used as a cross-check: a numba or KD-tree pair counter, for example with `scipy.spatial.cKDTree` on 3-D unit vectors with chord distances, plus tangential and cross shear projection.
   - Report both. Agreement between them is part of the validation.
2. **Catalogue reading.** The file is larger than RAM.
   - Read only the needed columns with `astropy.io.fits` and `memmap=True`, slicing in chunks of about 5e6 rows, or use `fitsio` if you install it.
   - Write one compact per-bin file **on the Linux file system** (for example `~/kids_dr5_work/bin{1..6}.npz` with ra, dec, e1, e2, w as float32/float64; not `/mnt/d`). Then work from those files.
   - Record row counts, the selection used and the sum of weights per bin. Check the effective number density against the `NZ_SOURCE` header keywords `NGAL_1..6` (arcmin^-2) and the ellipticity dispersion against `SIG_E_1..6` in the FITS files. They should agree to a few per cent if the selection and weights are right; this is a gate.
3. **Shear sign conventions.** KiDS e1/e2 versus RA/Dec orientation (TreeCorr `flip_g1/flip_g2`): determine the right convention. xi_+ is invariant under a sign flip of both components, but xi_- and the tangential/cross split are not. **Check against the official ascii files** (xip and xim), not by assumption.
4. **m-bias.** The catalogue `e1, e2` have no m correction. Find out from the paper and from the official files whether the released xi_+/- (and E_n) are m-corrected; the earlier project found that the Legacy likelihood does not divide by (1+m). Measure **uncorrected** first and compare. If you apply m, take the values from the paper and cite the table.
5. **Patches.** The official files are "NScomb", combining North and South. Measure N and S separately and combined (pairs across patches contribute nothing; a weighted combination of separate N and S counts equals the joint run). Report both.
6. Use 16 threads at most for TreeCorr (`num_threads=16`). Keep TreeCorr `bin_slop` small (0 or at most 0.05) for the 1000-bin run, and record the value.

## 4. Tasks

### Step A: auto- and cross-correlations xi_+/-(theta), 21 tomographic pairs
- Measure xi_+, xi_-, xi_x (the imaginary parts) and the pair counts and weights for all 21 pairs (i <= j, bins 1-6). Use the **same binning as the official ascii**: 1000 log bins over 2-300 arcmin (`bin_type='Log'`, check the edges against the `r_nom` column).
- **Check A1 (per pair):** the fractional difference of the summed weights (`weight` column) and of `npairs` versus the official values. Target better than 1e-3, given identical selection and weights.
- **Check A2:** xi_+ and xi_- versus official, per pair. Report:
  - (i) the median and max of |Delta xi| / sigma_xi (official `sigma_xip`) over the 1000 bins;
  - (ii) chi2 of the difference using the **binned** official covariance after re-binning to the 9 `xiP`/`xiM` bins (item A3).
  - Target: |Delta| much smaller than sigma, essentially identical if the inputs are identical.
- **Check A3:** re-bin your fine xi_+/- to the 9 official bins (find the official binning rule: weighted means over the fine bins, with the bin edges from the paper or the ini) and compare with `KiDS_Legacy_xipm.fits` `xiP`/`xiM` using its COVMAT. Report chi2(ours - official) / 378.
- **Check A4 (estimator cross-check):** your own pair counter versus TreeCorr on a sub-sample (for example 1 per cent of galaxies, or one patch, for two pairs). They must agree to numerical precision for the same binning.

### Step B: COSEBIs E_n, B_n from the measured xi_+/-
- Compute E_n and B_n, n = 1..20, over 2-300 arcmin, using `kidspipe.theory.cosebis` filters (mpmath roots for 2-300) and the fine-binned xi_+/-.
  - Use the official integration convention: the trapezoid in theta over the 1000 bins with `meanr`, or `r_nom`; determine which reproduces the official values better and document it.
- **Check B1 (integration step only):** feed the *official* ascii xi_+/- into your E_n/B_n code and compare with `KiDS_Legacy_cosebis.fits` En/Bn. This isolates the integration from the measurement. Report |Delta E_n| / sigma (sqrt of the COVMAT diagonal) per mode, and chi2 over modes 1-6 (126 points) and 1-20.
- **Check B2 (full measurement):** your measured xi_+/- to your E_n/B_n, against the official values. Same metrics.
- **Check B3:** B_n null test, chi2 of your B_n against zero with the official Bn covariance block (official data give 151.1 for 126 points, PTE 0.06, with modes 1-6; reproduce this from the official Bn as a sanity check, then report yours).

### Step C (optional, only if A and B pass): band powers and systematics
- Band powers PeeE/PeeB from xi_+/- if you can find the exact official definition in the paper. Compare with `KiDS_Legacy_bandpowers.fits`.
- PSF-leakage null test: cross-correlate the galaxy ellipticity with `PSF_e1/PSF_e2` (xi_{g,PSF}). Report its amplitude relative to xi_+.

### Step D: theory comparison with the existing model (light)
- Using `kidspipe.legacy.LegacyCosebisModel` at the official best-fit chain sample (`data/kids_legacy/official_cosebis/output_nautilus_cosebis_Fiducial.txt`, maximum `post`), compute chi2 of **your measured** E_n (modes 1-6, official covariance). Compare with 127.23 obtained with the official data vector.

### Step E: package the work
- New code goes into `src/kidspipe/measure/`, for example `catalogue.py` (chunked reader and selection), `xi.py` (TreeCorr wrapper plus the independent estimator), `cosebis_from_xi.py` (or reuse `theory/cosebis.py`), `compare.py`.
- Add tests in `tests/test_measure.py`. Make them **fast**: synthetic catalogues with a known input shear field, a constant shear giving xi_+ = |gamma|^2, rotation tests for xi_x = 0, and estimator agreement on a toy catalogue. They must not need the 16 GB file.
- Scripts go in `scripts/dr5_*.py`, outputs in `runs/dr5_measure/` (JSON, NPZ, logs) and plots in `plots/dr5_*.png`. Minimum set of plots:
  1. `dr5_01_xipm_vs_official.png`: 21 panels of theta*xi_+/- (ours and official) with the ratio or difference in sigma units.
  2. `dr5_02_npairs_weights_ratio.png`
  3. `dr5_03_En_Bn_vs_official.png`: 21 panels, modes 1-20, ours and official, with sigma error bars. Mark the n <= 6 analysis range.
  4. `dr5_04_integration_vs_measurement.png`: check B1 against check B2 residuals.
  5. `dr5_05_Bn_null.png`
  6. `dr5_06_estimator_crosscheck.png`: your estimator against TreeCorr.

## 5. Report (required): `docs/DR5_MEASUREMENT_REPORT.md`

Sections:
1. Summary: a pass/fail table of every check (A1-A4, B1-B3, C, D) with the key number and its target.
2. Data selection and weights: the exact cuts, galaxy counts and n_eff/sigma_e against the `NGAL_*` and `SIG_E_*` header values.
3. Conventions decided (sign flips, m-bias, theta bin weighting, integration rule), each with the **evidence**, meaning the numeric comparison that decided it.
4. Results with all plots embedded and captions.
5. Discrepancies and their likely causes, stated honestly. List what is unresolved.
6. Exact reproduce commands, run times, and software versions (TreeCorr, numpy, astropy).

## 6. Final deliverable: a cross-check prompt for Claude

Write `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md`. It is a self-contained prompt that the user will paste into Claude to **independently audit your work**. It must:
- State what you claim, as a numbered list of claims, each with the file that holds the number and the script that produced it.
- Ask Claude to **re-run** a cheap subset of the work independently. Examples: re-measure one auto pair and one cross pair on one patch with its own short script; recompute check B1; spot-check the catalogue selection counts; re-open your JSON files and confirm that the report numbers match them.
- Ask Claude to look for specific failure modes:
  - a wrong shear sign or flip;
  - double or missing m-correction;
  - wrong weights (`weight` against `shear_weight_only`);
  - wrong TOMOBIN mapping;
  - theta in arcmin against radians in the COSEBIs integral;
  - trapezoid against bin-averaged integration;
  - patch combination;
  - bin_slop effects;
  - tests that cannot fail;
  - numbers in the report that do not match the files.
- List the files Claude should read, in priority order. Tell it to use Sonnet 5.5 subagents at medium effort if it delegates (the user's standing preference), and to report findings ranked by severity.
- Keep it under about 150 lines; Claude has a limited token budget.

## 7. Definition of done
- [ ] The A1-A4 and B1-B3 checks are computed and reported with numbers (pass or fail, honestly).
- [ ] All plots are produced and viewed.
- [ ] `docs/DR5_MEASUREMENT_REPORT.md` is written.
- [ ] `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md` is written.
- [ ] `tests/test_measure.py` passes, and the full suite passes (300 or more tests).
- [ ] Nothing under `/mnt/d/HI-gamma-correlation/` was modified.
