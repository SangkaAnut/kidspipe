# ours (b1.5) vs theirs: why they differ although settings and npairs agree

Scripts (all new, scripts/): verify_slop15_autos_oursvstheirs_a.py (a), _c.py + _c_an.py (c), _npairs.py (per-bin npairs). python3 anaconda, PYTHONPATH=src, 6 threads, subsample 200k (9-12 s per run).

## Verdict
(i) INPUT difference, and only one: our work-file ~/kids_dr5_work/bin<b-1>.npz stores ra/dec as float32; theirs (pipeline/00_extracts bin<b>.npz) stores ra/dec as float64. Everything else is bit-identical. NOT (ii): row permutation changes xi by exactly 0.000 sigma, and the float32 effect is the same at bin_slop 0 as at 1.5. The difference is a coherent-in-structure but zero-mean "pair migration between adjacent fine bins" effect, ~0.1 sigma per fine bin, ~0.01-0.04 sigma after 9-bin rebinning.

## (a) Difference per fine bin, in shape-noise sigma = sqrt(varxip) (same for ours/theirs: ratio 1.0000), all 6 autos
Raw (m-uncorrected) xi_ours - xi_theirs, divided by theirs sigma_xip / sigma_xim (ours varxi gives identical numbers, ratio 1.0000):
| pair | xi+ mean / rms / max (sigma) | xi- mean / rms / max | lag-1 autocorr xi+ / xi- | 9-bin rms (official sigma, m-corrected) xi+ / xi- | max 9-bin |d|/sig_off xi+ / xi- |
|---|---|---|---|---|---|
| B1 | 0.000 / 0.116 / 0.41 | -0.000 / 0.119 / 0.41 | -0.48 / -0.50 | 0.010 / 0.010 | 0.02 / 0.02 |
| B2 | 0.000 / 0.116 / 0.42 | 0.000 / 0.111 / 0.50 | -0.54 / -0.54 | 0.013 / 0.011 | 0.02 / 0.02 |
| B3 | -0.000 / 0.108 / 0.42 | 0.000 / 0.113 / 0.55 | -0.51 / -0.51 | 0.006 / 0.007 | 0.02 / 0.01 |
| B4 | 0.000 / 0.106 / 0.49 | 0.000 / 0.099 / 0.52 | -0.51 / -0.43 | 0.006 / 0.006 | 0.01 / 0.02 |
| B5 | 0.000 / 0.109 / 0.39 | 0.000 / 0.109 / 0.44 | -0.50 / -0.51 | 0.008 / 0.010 | 0.01 / 0.02 |
| B6 | -0.000 / 0.143 / 0.56 | -0.000 / 0.150 / 0.67 | -0.52 / -0.47 | 0.016 / 0.012 | 0.04 / 0.02 |
Not a single fine bin exceeds 0.7 sigma; 0 of 1000 bins > 2 sigma in all 12 curves. Mean = 0 to 3 decimals, sign-positive fraction 0.49-0.51.
Character: zero-mean and not a coherent offset/tilt (band means in 2-6, 6-20, 20-60, 60-300' all |mean|<0.001 sigma; 9-bin deltas change sign across bins). It is NOT white noise either: adjacent fine bins are anticorrelated (lag-1 -0.43..-0.54, lag-5 ~0), so block-means of 10 bins scatter only 0.3-0.4x the white expectation. That anticorrelation is what pair migration across a bin edge produces (a pair moves from bin k to k+1). It averages out on rebinning (fine 0.11 -> 9-bin 0.006-0.016) but NOT in the COSEBIs high modes, which weight fine bins with alternating structure; hence E_n20 max 0.120 (B6) ours vs 0.043 theirs.
Direct evidence: per-bin npairs differ in 99.8-100% of the 1000 bins (rms relative 5e-5 overall, 1.3e-4 at 2-6'; lag-1 -0.34..-0.52), while the totals agree to 1e-5..1e-3 (sum dn = +27k, +87k, +38k, -95k, +72k, +102k out of ~1e11), i.e. the "npairs ratio 1.0000" in EVAL.md hides the per-bin migration. Their weight/npairs sums match: weight sums of the catalogues are identical (below).
The EVAL.md "max rel diff 8e+02..2.8e+04" is a bad metric: xi crosses zero, |a-b|/|b| blows up at the sign changes.

## (b) Inputs, bin 6 and all bins (theirs 00_extracts/outputs/bin<b>.npz vs ours ~/kids_dr5_work/bin<b-1>.npz)
- galaxy count equal, bins 1-6: 7416371, 7359911, 6799681, 6880432, 6477538, 5960461.
- row ORDER identical (e1, e2, weight compared row by row, max abs difference exactly 0.0 in all 6 bins).
- e1, e2: float32 in both, identical, same sign, no flip (theirs e1_sign=e2_sign=+1; ours flip_g1/g2 False), no m or c correction before TreeCorr in either (theirs corrects m afterwards: xi/mfac; ours raw, evaluator divides by mfac, so we compare raw to xip_raw).
- weight: float32 in both, identical; weight sums equal to all printed digits (bin6: 31501070.3933 both; bin1 82612776.949). Theirs used 'weight' column through w=; ours the same.
- ra, dec: theirs float64, ours float32 (cast to float64 by make_treecorr_cat after the information is lost). Max |ra diff| 1.53e-5 deg (0.055 arcsec), dec 1.9e-6 deg (0.007 arcsec), identical in all 6 bins. Fine bins are 0.5% wide in log r: 0.6 arcsec at 2', 1.5 arcmin at 300', so this shifts a few 1e-5 of the pairs across bin edges, mostly at small scales. TreeCorr call: the same kwargs (1000 Log bins 2-300 arcmin, bin_slop 1.5, default angle_slop), TreeCorr 5.1.4 on both sides (theirs npz treecorr_version = 5.1.4).
- npairs per bin: see (a).

## (c) Hypothesis test, bin 6 random subsample 200k (seed 12345, rows kept in file order), 6 threads, TreeCorr 5.1.4
Difference in units of shape-noise sigma (rms over 1000 fine bins; max; lag-1; mean). xi+ / xi-:
| comparison | bin_slop 1.5 | bin_slop 0 |
|---|---|---|
| row-permutation 1 vs original (f64) | rms 0.000, max 0.00 | 0.000 |
| permutation 1 vs permutation 2 (f64) | 0.000 | 0.000 |
| permutation 1 vs original (f32) | 0.000 | 0.000 |
| float32-rounded vs float64 positions (same order), the ours-vs-theirs analogue | xi+ 0.086 (max 0.68, lag1 -0.52) / xi- 0.073 (max 0.39, lag1 -0.42) | xi+ 0.085 (max 0.68, lag1 -0.52) / xi- 0.073 (max 0.39, lag1 -0.42) |
| bin_slop 1.5 vs 0 (f64), the approximation error itself | xi+ 0.274 (max 1.51, lag1 -0.44) / xi- 0.310 (max 1.48) | - |
(50k subsample: f32-vs-f64 rms 0.072/0.072, permutation 0.000, same pattern.) Pair counts: f32 vs f64 changes npairs in 968 of 1000 bins (rel. total 2e-5) at slop 1.5 and 967 at slop 0; permutation changes npairs in 0 bins.
Reading: (1) TreeCorr's kd-tree splits on the geometric median/centre, so the result is independent of input order to the last bit: input-ordering tree noise does not exist. (2) The float32 effect is identical with bin_slop 0 (no approximation), so it is not an approximation artefact; it is the pair-edge migration from moving the points. (3) Size: f32-vs-f64 0.08-0.09 sigma on 200k, consistent with the 0.10-0.15 seen on the full catalogues (ours-vs-theirs). 9-bin subsample deltas of f32-f64: up to 0.019 (bin 1) and <0.01 elsewhere in subsample shape-noise sigma. (4) The approximation error (slop 1.5 vs 0) is itself ~3x larger, 0.27-0.31 sigma fine-bin, up to 0.11 sigma 9-bin (shape-noise sigma, 200k); it is deterministic given the positions, and is shared by theirs and ours (same TreeCorr algorithm; official used 4.2.3 vs our 5.1.4 is a separate, untested difference).

## (d) Conclusion and consequences
Ours-vs-theirs is explained by (i), the float32 ra/dec in our work files; (ii) tree-structure noise is refuted (permutation exactly 0; f32 effect unchanged at slop 0); no (iii) needed: identical settings, counts, order, e, w, and the amplitude (weighted ratio) agree. Residual unexplained: none seen; I did not rerun the full catalogue (forbidden), so the full-catalogue prediction rests on the subsample scaling and on the pair-migration signature in the full npairs.
A3 margin: A3 uses |d|/sigma_official on 9 bins; ours-vs-theirs 9-bin differences are <= 0.041 sigma (B6 xi+ bin 1 +0.041, B6 xi- bin 9 -0.019, bin 6 +0.017). The B6 A3 rise 0.044 -> 0.063 is the last 9-bin of xi- (-0.044 -> -0.063): a 0.019 sigma shift, consistent with that bound. B6 xi+ also moves in bin 1: 0.005 -> 0.046 (0.041, its A3 max). So the gate (< 0.1) is still passed with margin 0.037 (B6 xi-) and 0.054 (xi+) in ours; the float32-positions effect can move A3 by up to ~0.04-0.05 sigma by itself, the same size as the headroom of 0.05-0.06 between theirs and the gate if the effects add adversely, so rerunning with float64 positions would be more defensible than relying on the margin. Gate margin is dominated by this plus the TreeCorr 4.2.3 vs 5.1.4 / bin_slop approximation (0.1 sigma level in 9 bins at the 200k shape-noise scale), not by any bug. Gates are PI-owned.
Also: E_n20 max 0.120 (B6) is driven by the same fine-bin noise (anticorrelated bin-to-bin), still far from any visible sign problem; B_n null chi2 unaffected (19.2 vs 19.2 official).

## Which ours-vs-theirs metric the evaluator should report
Replace max_rel_diff by: (1) fine-bin z = (xi_ours - xi_theirs)/sqrt(varxi) per bin: report rms, max|z|, mean and lag-1 autocorrelation for xi+ and xi- (expect rms ~0.1, mean ~0, no bin > 1); (2) the 9-bin delta over official sigma (m-corrected) max|Delta9|/sigma, directly comparable to A3; (3) per-bin npairs relative difference rms, with a flag if sum differs by >1e-3. Avoid ratios of xi values.
