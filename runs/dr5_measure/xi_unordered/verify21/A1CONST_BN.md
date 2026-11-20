# A1CONST_BN verification (A1 one-constant A_p test, B_n null chi2, E_n vs official)

Scripts (scratch, not in repo): /tmp/claude-1000/-home-anut/8ecc82c7-22dd-490c-b045-63f93bce2233/scratchpad/v.py, v3.py, v4.py
Run as: cd /mnt/d/cloning_kids_Kappa_pipeline; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts /home/anut/anaconda3/bin/python3 <script>

## 1. A1 one-constant A_p test

Implementation read: a1check.a1b_noise (called by evaluate_a1_const with all 21 pairs). Per statistic (xi+, xi- separately):
A_p = sum(w t)/sum(w a) over 1000 fine bins (t = our RAW xi, a = official ascii, w = ascii npairs_weighted), lnA_p,
weighted mean lc with weights 1/sd_p^2, chi2 = sum((lnA_p - lc)/sd_p)^2, pte = chi2.sf(chi2, 21-1).
sd_p = fractional A_p error from the moving-block bootstrap (block 25, 600 reps, fine-bin index resampling) in
runs/dr5_measure/a1_salvage/a1_salvage.json (step2_amplitude.*_raw.A_pair_err_frac), made earlier from "their" curves
(which equal ours to 1e-11), so it is a fixed pre-computed input, not recomputed from this run.

Recomputed independently from the 21 pair npz files + ascii (v.py): xi+ chi2 12.7200, pte(dof 20) 0.8891, C 1.17322;
xi- chi2 19.3986, pte 0.4961, C 1.18208. Match claimed (0.889/1.173, 0.496/1.182). Using our own npairs as weights instead
of ascii weights changes A_p by <= 4.5e-4 relative. Pair ordering (pair_index) is monotone and consistent.

Fairness vs the gate text (a1check docstring (ii), RESUME): 21 pairs, one lnC fitted per statistic, dof 20, separate
xi+/xi-, pte > 0.01: all as described. The 42-curve count enters only criterion (i) (boxcar), which is a separate check.
Null = all pairs share one amplitude (ascii = blinded realisation, absolute scale ~0.85 NOT tested here; A3 covers it).

Caveats / sensitivity:
- Low power for a few pairs: stored sd_frac ranges 0.010-0.33 (xi+) and 0.008-0.89 (xi-). xi- Bin1_Bin1 (A=1.95, sd 0.89),
  Pairs with sd>0.1 (2 per statistic, e.g. xi+ sd 0.31, 0.33; xi- Bin1_Bin1 0.89, 0.16) are almost uninformative.
  Dropping the 2 pairs with sd>0.1 per statistic: xi+ chi2 12.59/18 dof pte 0.815; xi- chi2 19.00 pte 0.392. Still pass.
  Best-constrained pairs (sd ~1%) are sensitive to ~3-4% amplitude errors (delta chi2 ~ (x/sd)^2).
- xi+ chi2 12.7 on 20 dof is on the low side (pte 0.89), suggesting sd is, if anything, mildly conservative; a fresh bootstrap
  (3000 reps, v3.py) gives sd ratio to stored median 1.005 (xi+) / 1.012 (xi-), so the stored errors are reproducible.
  Diagonal chi2 with the fresh sd: xi+ 12.31 (pte 0.905), xi- 19.15 (pte 0.512).
- Cross-pair correlations are ignored by the gate (diagonal chi2). I estimated them with a shared-index block bootstrap
  (3000 reps, only captures correlation via common fine-bin resampling, NOT shared-galaxy noise): mean off-diagonal corr of
  lnA 0.006 / 0.002 (range -0.38..0.42 / -0.46..0.35). Full-covariance GLS chi2: xi+ 18.21 pte 0.574, xi- 17.00 pte 0.653.
  Both still pass > 0.01. Sign of the bias is not definite from this (mixed-sign correlations; pte moved down for xi+, up for xi-);
  true shared-galaxy correlations are not estimable here. Not a failure mode at the margin seen (ptes 0.5-0.9).
- sd comes from the same residuals being tested (self-calibrated), so the test checks pair-to-pair consistency relative to
  within-curve jitter only; a bootstrap that ignores curve-level coherence between neighbouring blocks could under-estimate sd,
  but the observed chi2/dof <= 1 argues against that.
Judgement: fair, correctly implemented gate; pass is robust to the variants I tried.

## 2. B_n null chi2 (v4.py; official B_n/cov read directly from KiDS_Legacy_cosebis.fits, cov [En|Bn], B block offset 420,
file ordering asserted; ours = e_from_xi_binned on m-corrected xi, nmodes 20, edges)

| pair | 6 modes ours / official (pte ours) | 20 modes ours / official (pte) | EVAL.md |
|---|---|---|---|
| Bin6_Bin6 | 19.11 / 19.24 (0.004) | 32.35 / 32.62 (0.040) | 19.1 (19.2) / 32.4 (32.6) ok |
| Bin3_Bin6 | 1.14 / 1.14 (0.980) | 28.69 / 28.69 (0.094) | 1.1 (1.1) / 28.7 (28.7) ok |
| Bin1_Bin1 | 7.89 / 7.89 (0.246) | 26.47 / 26.45 (0.151) | 7.9 / 26.5 (26.4) ok |
| Bin5_Bin6 | 1.76 / 1.73 (0.940) | 14.79 / 15.15 (0.788) | 1.8 (1.7) / 14.8 (15.2) ok |
| Bin2_Bin4 | 4.96 / 5.01 (0.549) | 16.24 / 16.31 (0.701) | 5.0 / 16.2 (16.3) ok |
| Bin4_Bin5 | 11.80 / 11.75 (0.067) | 29.93 / 29.89 (0.071) | 11.8 (11.7) / 29.9 (29.9) ok |

Note: Bin6_Bin6 6-mode chi2 19.2 has pte 0.004 in the OFFICIAL data too; ours reproduces it (not a measurement fault, but a
marginal-looking null value to be aware of; same as the earlier pilot 19.5/6).

## 3. E_n max|ours-official|/sigma (sigma = sqrt diag of official E cov), same run
Bin6_Bin6: 6 modes 0.041, 20 modes 0.043 (EVAL.md 0.041/0.043). Bin3_Bin6: 0.024/0.037 (0.024/0.037). Bin5_Bin6: 0.030/0.032 (0.030/0.032).
Also Bin1_Bin1 0.007/0.007, Bin2_Bin4 0.017/0.017, Bin4_Bin5 0.032/0.044: all match EVAL.md.

## Discrepancies
- None in numbers. Observations only: (a) A1 const test has near-zero power on 2 pairs per statistic (sd > 0.1); (b) cross-pair
  correlation ignored (effect inconclusive, tests still pass with full bootstrap covariance); (c) sd_frac is a pre-computed
  external input from the salvage step, not regenerated in this run; (d) Bin6_Bin6 B_n6 pte 0.004 is shared by the official data.

VERDICT: CONFIRMED_WITH_CAVEATS
