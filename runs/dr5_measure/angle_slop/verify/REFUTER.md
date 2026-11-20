# REFUTER record: attempt to refute C4 (official release = bin_slop~1.5 approximation, ours is exact)
Only existing json/md used; no TreeCorr runs; no repo files edited.

## Numbers computed here (from exact_bin6_eval_b0_a0.json and fullpair_vs_exact.json, 20 modes)
- corr(exact-official E_n, exact-slop15 E_n) = 0.979; regression slope 1.038; residual rms 0.034 vs rms(exact-official) 0.326 -> R^2 = 0.989.
- Residual (official-minus-slop15, up to sign) = slop15-official curve = -0.025,-0.055,-0.068,-0.071,-0.05 ... +0.034 (n=13) ... -0.048 (n=20): max 0.071, changes sign at n~11, not monotone. The monotone -0.05..-0.48 drift is almost entirely the exact-minus-slop15 curve (+0.028..+0.437 in fullpair_vs_exact.json). Identity check: (ex-off)-(ex-s15) equals (s15-off) to 1e-15 (so the "explanation" is by construction the statement "slop1.5 through our code reproduces official", which is the empirical content).
- theirs (external TreeCorr 5.1.4, bin_slop 1.5, unordered) vs official E_n: max 0.043, no drift.
- A3 xi- 9-bin: ex-off last bin -0.156; slop15-off -0.063; theirs -0.044. Per-bin (ex - s15) equals fullpair xim9 to 4 digits (-0.093 in the 150-300' bin). So ~60 % of the failing bin is removed by slop1.5; a residual 0.04-0.06 sigma remains (also in theirs: 0.044), i.e. of the size of tree-structure/noise differences at fixed bin_slop.

## New evidence found (not in ANGLE_SLOP.md)
ANGLE_SLOP.md says the official bin_slop is "NOT established". The project's own files cite it: /mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/code/config.py:35-37 ("Wright+25 Sect. 2.5.1 and footnote 1: TreeCorr v4.2.3, bin_slop=1.5 ... XI = dict(min_sep=2, max_sep=300, nbins=1000, bin_slop=1.5)"), REPORT.md:37, and runs/dr5_measure/PRIOR_VALIDATION_01.md:101 ("Their production run used bin_slop=1.5"). This is SECOND-HAND (an agent-written report quoting the paper); I could not open the paper itself (not found on disk), and VERIFY.md:54 says no bin_slop in the release folder. If the paper footnote is as quoted, C4's premise is documented, not just inferred.

## Alternatives tested
(i) 9-bin rebin convention / outer edge / weighting: RULED OUT as cause. Same rebin code for ours, slop15, theirs. weighted_rebin with official npairs_weighted gives the same (0.1563 vs 0.1563). Slop1.5 and theirs pass through the identical rebin and are 0.06/0.04, so a convention error would show there too. Residual: rebin is of fine bins, not a direct coarse run (not independently checked by me beyond that equivalence).
(ii) Official E_n from different range/integration (trapz truncation 0.2-2 %): RULED OUT as cause of the drift. E_n for ours/slop15/theirs all computed by the same xi2cosebis code against the same official E_n; theirs and slop15 sit at <=0.07 with no monotone drift. Any integration difference would be common to all three. (Residual common-mode up to 0.07 at n<=5 is not excluded; it is the size of slop15-official.)
(iii) weights / m-correction / A_p: RULED OUT as the cause of the ours-vs-slop15 gap (A_p equal to 4 digits in ours vs slop15: 1.1669 vs 1.1699 is the 0.26 % xi shift itself; same weights, same catalogue). A common-mode m or weight error would affect slop15 too, which agrees with official.
(iv) catalogue/selection: npairs ours/theirs = 2.0000102 rules out a different galaxy set to 1e-5 (pair counts are catalogue+weight-position sensitive), but does NOT rule out a weight-value or e1/e2 difference; those are common to slop15 (which matches official) so this is covered by the same argument as (iii).
(v) TreeCorr 4.2.3 (official, per config.py) vs 5.1.4: RULED OUT as a major factor: theirs is 5.1.4 at slop 1.5 and agrees with official (0.044 / 0.043). Cannot exclude a small 4.x vs 5.x tree-structure effect of order the 0.04-0.06 remaining.
(vi) ordered vs unordered fold bookkeeping: RULED OUT as the cause: slop15 uses the identical chunked/ordered/fold path and matches official; npairs ours = 2x theirs to 1e-5 and the re-assembly reproduces to 0. Only a bug that exists solely at bin_slop 0 remains possible (survives, but brute_check shows (0,0) = TreeCorr brute force to 5e-14 on patches).
(vii) Is a smooth one-signed E_n drift what a bin_slop approximation produces? PARTLY. It requires a coherent amplitude shift rather than pair-reshuffling noise; the 20 E_n modes are strongly correlated so any large-scale amplitude shift looks smooth and one-signed. Evidence for coherence: xi- shift +0.69 +- 0.29 % (jackknife t 2.4) and +1.54 +- 0.68 % (4/4 20 %-subsamples) per PILOT_BIN6.md/VERIFY.md; signal-only synthetic bias 0.052 sigma (ANGLE_SLOP). Coherence is suggestive (t~2.3-2.4, one realisation), not proven; the mechanism is unproven. Note the drift is statistically tiny: chi2(E_n ours vs official) = 0.78/20 using the official covariance.

## What survives (not refuted, not proven)
1. Mimicry is not uniqueness: slop 1.5 matching official does not show the official release is slop 1.5 unless the paper footnote is confirmed; some other approximate-tree setting (e.g. 4.2.3 default angle behaviour, patches) could also match. Only 1.5 was tried at full pair.
2. Remaining 0.04-0.06 sigma in the 150-300' xi- bin (theirs also 0.044) is unexplained but small.
3. A bug present only at bin_slop=0 is excluded only by brute-check on patches and pair-count agreement, not by an independent estimator on real full data.
4. C4's "not an error in our measurement" holds for the TreeCorr-level definition of the estimator; an independent numpy brute estimator on a subsample vs the exact TreeCorr (0,0) was not run here.

## Verdict
C4 survives refutation. Strongest support: slop1.5 through our code removes 98.9 % of the E_n variance (corr 0.979) and 60 % of the failing xi- bin; theirs reproduces official at 0.04; conventions are common to all three runs. The official bin_slop=1.5 appears in project files (config.py:35-37) citing Wright+25 footnote 1 (second-hand; verify against the paper).

## Single discriminating test
Read the actual Wright+25 Sect. 2.5.1 footnote 1 (bin_slop value, TreeCorr 4.2.3). If it says 1.5, C4 is settled. Computational alternative: full Bin6_Bin6 at bin_slop 1.0 (or 0.75): C4 predicts E_n drift to shrink toward ~half and A3 xi- distance to official to rise relative to 1.5 if official is 1.5 (a non-monotone or flat response would signal something else).
