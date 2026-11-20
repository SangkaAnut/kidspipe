# KiDS-Legacy COSEBIs validation against the official release

Our pipeline (`src/kidspipe/legacy`, notes in `docs/interface_notes/legacy.md`) is compared with the official KiDS-Legacy (KiDS DR5)
cosmic-shear COSEBIs release (Wright et al. 2025, A&A 703, A158): data vector + covariance (`data/kids_legacy/KiDS_Legacy_cosebis.fits`)
and the official nautilus chain (`official_cosebis/output_nautilus_cosebis_Fiducial.txt`, 18720 rows, 17704 with finite likelihood,
n_eff = 10593). Official ln L = post - prior = -chi2/2 exactly (verified, no constant). All numbers below come from
`runs/legacy_val/summary.json`, `chi2.npz`, `ia_scale_diag.json`, `timing_t{1,4}.json`.

## Verdict

* On the official best-posterior sample our chi2 = 127.23 vs official 127.29 (126 data points).
* On 3509 official chain samples (150 top-posterior + 3402 unique posterior-weighted draws) our chi2 reproduces the official chi2 with
  **mean difference -0.26, std 0.79, corr 0.972**; top-150: mean -0.13, std 0.40, max |d| 1.2. Worst single sample -6.66 (chi2_off 140).
* Importance-reweighting the official chain with exp(lnL_ours - lnL_off) moves S8 by only **+0.0014** (0.8128 -> 0.8142), sigma8 by -0.0065,
  Omega_m by +0.007; reweighting ESS fraction 69%. Cosmological conclusions are unchanged at <0.1 sigma(S8).
* The agreement is not perfect: a residual chi2 scatter of 0.8 and a small systematic negative offset remain (see Caveats). They are NOT
  yet explained; the CosmoPower-vs-CAMB hypothesis is untested and a correlation with log10 M_1 (an IA nuisance) was found.

## 1. Sample-by-sample chi2 (`plots/legacy_01_chi2_vs_official.png`)

| set | n | mean(ours-off) | std | max abs | 95% abs | frac |d|<0.5 | frac |d|<1 | corr |
|---|---|---|---|---|---|---|---|---|
| all | 3509 | -0.263 | 0.793 | 6.66 | 1.80 | 0.62 | 0.85 | 0.972 |
| top-150 posterior | 150 | -0.129 | 0.399 | 1.23 | 0.82 | 0.79 | 0.97 | 0.968 |
| weighted draws (unique) | 3402 | -0.267 | 0.802 | 6.66 | 1.83 | 0.61 | 0.85 | 0.971 |
Posterior-weighted (draw multiplicity): mean -0.261, std 0.794. official chi2 range 124.9 - 153.8. No failed evaluations.
Our CAMB sigma_8 vs chain: +4.9e-4 relative mean (std 3.7e-4), uncorrelated with the chi2 difference (corr -0.0006 on the first 650 rows).
Spearman correlation of the difference with chain parameters: log10 M_mean_1 -0.60 (M_2 similar), S8 -0.27, a -0.25, beta +0.22, h0 +0.13,
n_s +0.14, other < 0.12; with official chi2 -0.19. Mean difference by log10 M_1: [11,12) -0.24 (n=3434), [12,12.5) -1.9 (n=49).
Because log10 M_i are strongly prior-correlated with a, beta this is not a clean IA diagnosis.
Diagnostic `scripts/legacy_ia_scale_diag.py` (200 random draws, global IA-amplitude factor s): mean diff / std / slope per dex of log10 M_1:
s=0.7: -0.39/1.15/-3.8; 0.85: -0.43/1.01/-3.5; **1.0: -0.34/0.92/-3.0**; 1.15: -0.14/0.91/-2.1; 1.3: +0.19/1.03/-0.85; 1.6: +1.24/1.72/+2.7.
A ~20% larger IA amplitude would zero the mean offset but does not reduce the scatter (min std 0.91 at s ~ 1.1), so a simple global IA rescale is
not the cure; no change was made to the model.

## 2. Official best sample (chain row 17608) (`plots/legacy_02_En_data_vs_model.png`, `legacy_02b_residuals_Bmodes.png`)
* chi2_ours = 127.23, chi2_off = 127.29, chi2/126 = 1.010, PTE(126 dof) = 0.45 (parameters not subtracted). Whitened residuals: mean +0.11, std 1.00.
* Largest per-pair chi2 (6 points each): bins 2x6 19.5, 1x3 15.7, 2x2 14.3, 2x5 8.3, 4x4 8.2 (these are also the pairs where model visibly misses the data in
  the plot, i.e. a property of the official fit, not of our model).
* B_n null test (independent of the model): chi2(B_n, modes 1-6 vs 0, Bn covariance block) = **151.1 for 126 points, PTE 0.064**; all 20 modes
  459.3/420, PTE 0.090. Consistent with zero (mildly high). The En-Bn cross-covariance in COVMAT is exactly zero. For reference chi2(E_n vs 0) = 818.
  Note: our model is validated against E_n only; the B_n check validates the data/covariance reading, not the model.

## 3. Importance reweighting (`plots/legacy_03_S8_official_vs_ours.png`)
4000 draws with replacement from the official weights (3402 unique rows); weights = multiplicity x exp(lnL_ours - lnL_off). S8 is chain column `s_8_input`
(= sigma8 sqrt(Om/0.3) within 0.004), sigma8 col. 21, Omega_m col. 23 (the S_8 column of the file is empty). Note: the parameter *values* are the official
ones (emulator sigma8); only the weights change.

| quantity | official (full chain) | official (same 3402 draws, noise ref.) | reweighted | shift (bootstrap err) |
|---|---|---|---|---|
| S8 | 0.8128 +- 0.0187 | 0.8127 +- 0.0186 | 0.8142 +- 0.0186 | +0.0015 (0.0002) |
| Omega_m | 0.3311 +- 0.0557 | 0.3303 +- 0.0562 | 0.3382 +- 0.0582 | +0.0079 (0.0008) |
| sigma8 | 0.7842 +- 0.0785 | 0.7852 +- 0.0792 | 0.7777 +- 0.0794 | -0.0075 (0.0009) |
S8 median (16/84%): official 0.8135 +0.0174 -0.0191; reweighted 0.8149 +0.0177 -0.0192.
ESS: draws 2936 -> 2030 after reweighting (69%); ln L difference weighted mean +0.13, std 0.40; max/min ratio 55. Reweighting is well conditioned (no heavy tail).
Published (Wright et al. 2025, abstract; **verified via alphaXiv get_paper_content of arXiv:2503.19441**): fiducial COSEBIs S8 = 0.815 +0.016 -0.021. The official chain
gives 0.8135 +0.017 -0.019 (our reading), consistent. The same paper states CosmoPower vs direct CAMB changes cosmology by <= 0.03 sigma.
The shifts above are small but statistically significant relative to the bootstrap error (several sigma of the draw noise) - they reflect the systematic offset in section 1.

## 4. Filters and contributions
* `plots/legacy_04_filters.png`: T_+n, T_-n (theta 2-300 arcmin) and l W_n(l), n=1..6, from mpmath-regenerated roots/norms; Gram matrix int T_n T_m / (theta_max-theta_min)
  equals identity to 2.6e-15 off-diagonal.
* `plots/legacy_05_contributions.png`: GG (IA amplitude zero) vs IA part (total - GG) per pair at the best sample, a = 5.77, beta = 0.44,
  A_i = [0.118, 0.307, 0.440, 0.678, 0.657, 0.100]. IA is negative (GI-dominated) at all 126 points (negative fraction 1.0;
  a black triangle would mark any positive point); IA/total at mode 1 is -2% to -8% (6x6: -0.3%); range over all points -7.9% ... -0.08%. chi2 with IA switched off at this sample: 129.6 (vs 127.2),
  i.e. data discriminate IA only weakly.

## 5. Timing (`runs/legacy_val/timing_t4.json`, `timing_t1.json`)
Full E_n evaluation with a new cosmology (CAMB mead2020_feedback + Limber + operator): **1.31 s (4 threads)**, 3.53 s (1 thread; ~7-8 s per sample in the 10-worker
run due to memory contention). Nuisance-only change (cosmology cached): 28 ms (4 thr) / 75 ms (1 thr) means (distribution has outliers; std > mean, so quote as order of magnitude);
E_n operator application alone 5 microseconds. CAMB dominates.

## Conventions decided (details/evidence in docs/interface_notes/legacy.md)
CAMB 1.5.8 `mead2020_feedback` with log T_AGN, mnu 0.06 normal; chain `A_S` column = ln(1e10 A_s); n_i(z - bias_i) cubic interpolation of Z_MID nodes; A_i = a f_r,i (M_i/M_piv)^beta
with M_i = 10^log10_M_mean_i, M_piv = 10^13.5; covariance = En block of COVMAT (m-bias inside), no m division, no c-term; E_n integral over l in [1,1e5], theta 2-300 arcmin, modes 1-6.

## Caveats
1. Residual std 0.79 (max 6.7) and mean -0.26 in chi2 remain; attributed in the implementation notes to CosmoPower emulator error but **not proven**. The sigma8 mismatch does not correlate with it; correlation with log10 M_1/M_2 and S8 exists. A global IA factor does not fix it. Possible causes: IA/mass-dependence details, emulator error, ell-grid (50 nodes in the official project_2d; we use 100), growth-factor convention in bk_corrected NLA.
2. The official sigma8/Omega_m values in the chain come from the emulator; we do not recompute them for the reweighted plot.
3. Reweighting uses a subset (3402 of ~10^4 effective) of the chain; noise reference (same draws, official weights) is shown and the bootstrap error is quoted.
4. The B_n PTE of 0.06 is mildly low-ish; not a model issue.
5. Only the fiducial official run was available; no blinding/other-statistic comparisons.

## Reproduce
```
cd /mnt/d/cloning_kids_Kappa_pipeline; export PYTHONPATH=$PWD/src
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 scripts/legacy_validate.py eval --workers 10   # ~37 min, resumable, writes runs/legacy_val/chi2.npz
OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 python3 scripts/legacy_validate.py plots                # figures + runs/legacy_val/summary.json
OMP_NUM_THREADS=4 python3 scripts/legacy_validate.py timing; OMP_NUM_THREADS=1 python3 scripts/legacy_validate.py timing
OMP_NUM_THREADS=4 python3 scripts/legacy_ia_scale_diag.py 200                                  # optional IA-scale diagnostic
```
(`scripts/legacy_plots.py` holds the plotting/analysis code.) Selection: seed 7, 150 top-posterior rows + 4000 weighted draws.
