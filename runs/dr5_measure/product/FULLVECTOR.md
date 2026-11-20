# FULLVECTOR: spec full-vector checks on OUR bin_slop 1.5 xi (product option (a))

Script `scripts/dr5_make_product.py`; numbers in `fullvector_checks.json`. Our xi = verified TreeCorr 5.1.4, bin_slop 1.5, angle_slop 0.1, unordered, float64 positions. m-values (Wright+25 Table 1, as in `scripts/dr5_adopt_01.py`): [-0.023, -0.016, -0.011, 0.02, 0.03, 0.045]. The release used TreeCorr 4.2.3; agreement with 'theirs' below is code equivalence, not independent release validation.

## A2: fine-bin |Delta xi|/sigma_xi against the official ascii (1000 bins, official `sigma_xip/xim`)

**Amplitude handling.** The official fine ascii is m-uncorrected and has an unexplained scale (RESUME: A_p ~1.17 for ours/ascii, ascii amplitude ~0.85) and is a blinded realisation of the same sample, not a truth reference; the numbers are for information, no gate. Our RAW (m-uncorrected) xi is compared. Two treatments: `A=1` (no rescaling) and `A_p` (per-pair, per-statistic ratio-of-weighted-sums scale, the A1b convention, `ours - A_p * ascii`). The scale, not a shape difference, dominates the `A=1` numbers.

| statistic | treatment | overall median | overall max |
|---|---|---|---|
| xip | A=1 | 0.158 | 1.759 |
| xip | A_p | 0.149 | 1.980 |
| xim | A=1 | 0.151 | 2.532 |
| xim | A_p | 0.154 | 2.659 |

Per pair (median / max): xi+ A=1 | xi+ A_p | xi- A=1 | xi- A_p

| pair | A_p xi+ | A_p xi- | xi+ A=1 | xi+ A_p | xi- A=1 | xi- A_p |
|---|---|---|---|---|---|---|
| Bin1_Bin1 | 1.201 | 1.950 | 0.22 / 1.61 | 0.24 / 1.79 | 0.23 / 1.61 | 0.54 / 2.66 |
| Bin1_Bin2 | 1.161 | 1.191 | 0.16 / 1.21 | 0.16 / 1.20 | 0.16 / 1.35 | 0.15 / 1.35 |
| Bin1_Bin3 | 1.131 | 1.193 | 0.14 / 1.11 | 0.13 / 1.17 | 0.14 / 1.21 | 0.14 / 1.25 |
| Bin1_Bin4 | 1.316 | 1.178 | 0.15 / 1.18 | 0.17 / 1.28 | 0.14 / 1.14 | 0.14 / 1.23 |
| Bin1_Bin5 | 1.164 | 1.177 | 0.13 / 0.89 | 0.13 / 0.95 | 0.14 / 1.24 | 0.13 / 1.30 |
| Bin1_Bin6 | 1.198 | 1.211 | 0.16 / 1.11 | 0.16 / 1.11 | 0.15 / 1.08 | 0.16 / 1.32 |
| Bin2_Bin2 | 1.147 | 1.179 | 0.23 / 1.76 | 0.22 / 1.98 | 0.21 / 1.86 | 0.22 / 1.83 |
| Bin2_Bin3 | 1.161 | 1.205 | 0.14 / 1.17 | 0.14 / 1.18 | 0.14 / 1.15 | 0.13 / 1.34 |
| Bin2_Bin4 | 1.183 | 1.228 | 0.15 / 1.30 | 0.14 / 1.29 | 0.13 / 0.98 | 0.12 / 1.05 |
| Bin2_Bin5 | 1.150 | 1.199 | 0.14 / 1.06 | 0.13 / 1.06 | 0.13 / 0.92 | 0.12 / 1.04 |
| Bin2_Bin6 | 1.148 | 1.195 | 0.16 / 1.22 | 0.15 / 1.41 | 0.15 / 1.17 | 0.15 / 1.25 |
| Bin3_Bin3 | 1.176 | 1.240 | 0.18 / 1.26 | 0.17 / 1.43 | 0.18 / 1.96 | 0.17 / 2.16 |
| Bin3_Bin4 | 1.184 | 1.186 | 0.13 / 0.85 | 0.12 / 0.96 | 0.13 / 0.90 | 0.13 / 0.92 |
| Bin3_Bin5 | 1.141 | 1.215 | 0.12 / 1.02 | 0.10 / 1.13 | 0.12 / 1.12 | 0.13 / 1.10 |
| Bin3_Bin6 | 1.322 | 1.161 | 0.14 / 1.08 | 0.17 / 1.28 | 0.15 / 1.20 | 0.14 / 1.30 |
| Bin4_Bin4 | 1.180 | 1.207 | 0.19 / 1.24 | 0.18 / 1.32 | 0.18 / 1.58 | 0.19 / 1.67 |
| Bin4_Bin5 | 1.185 | 1.196 | 0.13 / 1.13 | 0.11 / 1.25 | 0.12 / 0.89 | 0.11 / 0.91 |
| Bin4_Bin6 | 1.188 | 1.168 | 0.15 / 0.99 | 0.14 / 1.01 | 0.15 / 1.48 | 0.14 / 1.58 |
| Bin5_Bin5 | 1.167 | 1.189 | 0.17 / 1.04 | 0.13 / 1.13 | 0.15 / 1.24 | 0.13 / 1.29 |
| Bin5_Bin6 | 1.183 | 1.165 | 0.14 / 1.11 | 0.13 / 1.05 | 0.14 / 0.94 | 0.14 / 0.97 |
| Bin6_Bin6 | 1.171 | 1.139 | 0.22 / 1.70 | 0.22 / 1.76 | 0.20 / 2.53 | 0.21 / 2.41 |

## A3 full vector (9-bin m-corrected xi vs `KiDS_Legacy_xipm.fits`, COVMAT 378x378)

- chi2(ours - official) = 0.0609 / 378 = 1.61e-04; max|d|/sigma = 0.0528.
- Per statistic (189 points each, covariance sub-block): xi+ chi2 0.0303 (/189 = 1.60e-04), max 0.0528; xi- chi2 0.0316, max 0.0438.
- GATE (adopted 2026-10-29): max|Delta|/sigma < 0.1: xi+ PASS, xi- PASS. chi2/dof < 1 against the official covariance is NOT a gate (RESUME bug 6: the covariance contains cosmic variance that cancels between measurements of the same galaxies, so chi2 is nearly blind to smooth errors, and 10 % amplitude error gives only ~12/378).

## B2: E_n and B_n difference vs official (official COVMAT 840x840 = [E_n | B_n], B block offset 420)

| modes | points | chi2 E_n | max/sigma E_n | chi2 B_n | max/sigma B_n |
|---|---|---|---|---|---|
| 6 | 126 | 0.0242 | 0.0408 | 0.0223 | 0.0437 |
| 20 | 420 | 0.0698 | 0.0455 | 0.0780 | 0.0602 |

E_n/B_n comparisons have no numeric gate in the spec; shown for information (as in the evaluator).

## B3: B_n null (chi2 vs zero, B_n covariance block)

| modes | points | official chi2 | official PTE (published) | ours chi2 | ours PTE |
|---|---|---|---|---|---|
| 6 | 126 | 151.058 | 0.0635 (0.0635) | 150.876 | 0.0648 |
| 20 | 420 | 459.328 | 0.0902 (0.0902) | 458.803 | 0.0930 |

Official reproduces the published PTEs 0.0635 / 0.0902: yes.

## B1 (integration check on official input), quoted, not recomputed

`b1_integration.json`, trial `meanr|edges|rad` (official ascii xi, no m-correction -> our E_n vs official): chi2 = 0.0977/126, 0.0537/420, max|dE|/sigma = 1.577. (The adopted integration test, their m-corrected xi, is in `adoption.json` B1.)

## Comparison with `adoption/adoption.json` (theirs)

**Latent bug in `scripts/dr5_adopt_01.py` B1 (found here, not edited):** its B_n-vs-official chi2 and max/sigma use the E_n covariance block (`cov_cb[np.ix_(idx, idx)]`, no +420 offset) for B_n; the same mistake its B3 section documents. adoption.json `B1_En_Bn_vs_official.Bn_6m/Bn_20m` are therefore not valid B_n numbers (0.0169/126, 0.0644/420, max 0.0299, 0.0520). With the correct B_n block our B2 is 0.0223/126, 0.0780/420, max 0.0437, 0.0602. The Bn rows in the table below are recomputed with the E_n block to compare like with like. The A3, E_n and B3 numbers in adoption.json are unaffected.

Max relative difference over 20 numbers: 2.2e-06. Entries above 1e-6 relative: ['A3 xip max_abs_over_sigma', 'A3 xim max_abs_over_sigma', 'B2/B1 En 6m chi2', 'B2/B1 En 20m max_abs_over_sigma', 'B2/B1 Bn 20m max_abs_over_sigma [ours recomputed with the En block, as adoption did]'].

| quantity | ours | theirs (adoption.json) | abs diff | rel diff |
|---|---|---|---|---|
| A3 xip chi2 | 0.0302764 | 0.0302764 | 2.97e-08 | 9.82e-07 |
| A3 xip max_abs_over_sigma | 0.0527751 | 0.0527751 | 9.93e-08 | 1.88e-06 |
| A3 xim chi2 | 0.031615 | 0.031615 | 1.40e-08 | 4.42e-07 |
| A3 xim max_abs_over_sigma | 0.0438488 | 0.0438489 | -6.75e-08 | -1.54e-06 |
| B2/B1 En 6m chi2 | 0.0241942 | 0.0241941 | 4.91e-08 | 2.03e-06 |
| B2/B1 En 6m max_abs_over_sigma | 0.0407834 | 0.0407834 | -2.03e-09 | -4.97e-08 |
| B2/B1 Bn 6m chi2 [ours recomputed with the En block, as adoption did] | 0.0168766 | 0.0168766 | 4.19e-09 | 2.48e-07 |
| B2/B1 Bn 6m max_abs_over_sigma [ours recomputed with the En block, as adoption did] | 0.0298652 | 0.0298652 | -1.46e-08 | -4.89e-07 |
| B3 6m chi2_official | 151.058 | 151.058 | 0.00e+00 | 0.00e+00 |
| B3 6m pte_official | 0.063517 | 0.063517 | 0.00e+00 | 0.00e+00 |
| B3 6m chi2_ours | 150.876 | 150.876 | -2.65e-06 | -1.75e-08 |
| B3 6m pte_ours | 0.0647959 | 0.0647959 | 1.87e-08 | 2.89e-07 |
| B2/B1 En 20m chi2 | 0.0698267 | 0.0698266 | 4.12e-08 | 5.90e-07 |
| B2/B1 En 20m max_abs_over_sigma | 0.0455189 | 0.0455189 | -7.35e-08 | -1.62e-06 |
| B2/B1 Bn 20m chi2 [ours recomputed with the En block, as adoption did] | 0.0643906 | 0.0643906 | 3.59e-09 | 5.58e-08 |
| B2/B1 Bn 20m max_abs_over_sigma [ours recomputed with the En block, as adoption did] | 0.0520473 | 0.0520474 | -1.14e-07 | -2.20e-06 |
| B3 20m chi2_official | 459.328 | 459.328 | -5.68e-14 | -1.24e-16 |
| B3 20m pte_official | 0.0901892 | 0.0901892 | 3.33e-16 | 3.69e-15 |
| B3 20m chi2_ours | 458.803 | 458.803 | -2.38e-06 | -5.19e-09 |
| B3 20m pte_ours | 0.0929955 | 0.0929954 | 1.29e-08 | 1.38e-07 |

Direct joint-378 A3 chi2 (adoption.json has per-statistic blocks only): ours 0.060852, theirs (recomputed here from their xi, same code) 0.060852; max |m-corrected fine xi ours - theirs| = 3.87e-11.

Ours = theirs here is code equivalence (both TreeCorr 5.1.4, bin_slop 1.5, same extract), not independent release validation (release: TreeCorr 4.2.3). Only A3 is amplitude-sensitive (~1 %: xi*1.01 fails 2/21 pairs). Exact (bin_slop 0) was not run for all pairs.

## Step D (optional): model chi2 of our E_n, modes 1-6

`LegacyCosebisModel` at official best-posterior chain row 17608: chi2 official data vector 127.234 (quoted 127.23), chi2 our E_n 127.516 (126 points, official En covariance block). Difference +0.283. Runtime 16 s.
