# RECOMPUTE: independent recompute of the slop-1.5 unordered autos

Script: `scripts/verify_slop15_autos_recompute.py` (own code; imports only `kidspipe.theory.cosebis.CosebisFilters`, astropy, numpy, scipy; does not import the measure script or kidspipe a1check/official/xi2cosebis). Run with 6 threads, no TreeCorr run.

Method: official 9-bin xi, E_n, B_n and covariances read straight from the FITS; rows selected with `BIN1==BIN2==p` and sorted by `ANGBIN` (asserted 9 xi rows / 20 En rows per pair; xiM and Bn row order identical to xiP / En; covariance 378 = [xiP|xiM], 840 = [En|Bn] indexed by the same rows, so it follows file order). Ours: raw xi / (1+m_p)^2 (m = -0.023,-0.016,-0.011,0.020,0.030,0.045), npairs-weighted rebin on 9 log bins 2-300' by r_nom, sigma = sqrt(diag cov). E_n/B_n: 0.5*arcmin^2 * sum over fine bins of (exact edge-area weight) * T+-(theta_k) xi+-, filters tmin=2,tmax=300, mpmath. A1: official ascii of the SAME pair (file `..._NScomb_nBins_6_Bin{p}_Bin{p}.ascii`, r_nom asserted equal to ours), A_p = sum(g*ours)/sum(g*ascii) with g=npairs_weighted, z=(ours-A_p*ascii)/sigma_ascii, 20-bin boxcar std / (std(z)/sqrt20). A1 computed for all 6 pairs.

## Numbers (mine vs EVAL.md)
| pair | A3 xi+/xi- (mine) | EVAL | E_n max 6m/20m (mine) | EVAL | B_n chi2 6 (off) / 20 (off) mine | EVAL | A1 boxcar/white xi+ / xi- mine (max) | EVAL max |
|---|---|---|---|---|---|---|---|---|
| 1-1 | 0.0163/0.0183 | 0.0163/0.0183 | 0.0185/0.0213 | 0.018/0.021 | 7.96 (7.89) / 26.87 (26.45) | 8.0 (7.9)/26.9 (26.4) | 0.351 / 0.793 (0.793) | 0.793 |
| 2-2 | 0.0251/0.0195 | 0.0251/0.0195 | 0.0183/0.0276 | 0.018/0.028 | 5.75 (5.73) / 17.87 (18.05) | 5.8 (5.7)/17.9 (18.1) | 0.295 / 0.279 | 0.295 |
| 3-3 | 0.0172/0.0100 | 0.0172/0.0100 | 0.0098/0.0185 | 0.010/0.018 | 7.71 (7.72) / 28.66 (28.62) | 7.7 (7.7)/28.7 (28.6) | 0.315 / 0.348 | 0.348 |
| 4-4 | 0.0332/0.0377 | 0.0332/0.0377 | 0.0328/0.0440 | 0.033/0.044 | 3.26 (3.30) / 15.86 (15.75) | 3.3 (3.3)/15.9 (15.7) | 0.338 / 0.326 | 0.338 |
| 5-5 | 0.0401/0.0225 | 0.0401/0.0225 | 0.0229/0.0376 | 0.023/0.038 | 6.39 (6.61) / 17.94 (18.13) | 6.4 (6.6)/17.9 (18.1) | 0.347 / 0.321 | 0.347 |
| 6-6 | 0.0458/0.0633 | 0.0458/0.0633 | 0.0798/0.1201 | 0.080/0.120 | 19.23 (19.24) / 32.35 (32.62) | 19.2 (19.2)/32.4 (32.6) | 0.446 / 0.437 | 0.446 |

Largest difference vs EVAL.md at the EVAL print precision: A3 0.0000 (4 d.p.); E_n 0.0005 (rounding of the 3 d.p. table, all within 0.0005); B_n chi2 <= 0.05 (EVAL prints 1 d.p.); A1 0.0000 (3 d.p.). **No difference > 0.003. Every EVAL.md number reproduced.** The official block selected by BIN1/BIN2 is verified (assertions above; also matches pair-major loader).

Sensitivity (not an EVAL discrepancy): assigning fine bins by meanr instead of r_nom changes A3 by < 1e-4; weighting the rebin by pair-weight instead of npairs changes A3 by up to 0.0024 (B3 xi+ 0.0172 -> 0.0196, xi- B1 0.0183 -> 0.0193). All still < 0.1 (largest 0.0633).

B_n null: chi2 6 modes matches official per pair (pte ours: B6 = 0.0038 but official B6 chi2 19.24 is the same, so inherited from the official data, not introduced by the measurement). E_n: B6 E_n20 max 0.120 sigma (theirs 0.043) and B6 E_n 6-mode 0.080; if there is any E_n gate near 0.1 this is the only pair above it. Flag for PI (gate thresholds are PI-owned).

## npz integrity / provenance
- All arrays finite (0 NaN in xip, xim, npairs), min npairs/bin 6.5e5-9.1e5, 1000 bins; r_nom equals official ascii r_nom (rtol 1e-5).
- Recorded scalars in every npz: pairing=unordered, bin_slop=1.5, treecorr 5.1.4, threads 20, nobj1=nobj2. The npz does not store angle_slop; manifest.json records "TreeCorr default (None)". Checked myself: TreeCorr 5.1.4 `GGCorrelation(min_sep=2,max_sep=300,nbins=1000,sep_units='arcmin')` has default bin_slop = 1.0 and angle_slop (`b`-related `angle_slop` attribute) = 0.1; ours explicit bin_slop 1.5 gives b=0.00752 vs default b=0.00501. So bin_slop 1.5 is NOT the library default for this binning (it is the official-style setting claimed); "angle_slop default" = 0.1. I could not verify bin_slop/angle_slop from the output alone (npz only stores values it was told); settings come from the runner and the manifest.
- nobj per bin vs catalogue `/home/anut/kids_dr5_work/bin{p-1}.npz` (len of w): 7416371, 7359911, 6799681, 6880432, 6477538, 5960461, exactly equal to npz nobj1/nobj2 and manifest. Catalogue e1,e2,w contain no NaN; weighted mean e1,e2 ~ 0 (<5e-4 rounded), consistent with unflipped.
- Weight sums: catalogue sum(w) = 82.61M, 73.19M, 65.20M, 58.30M, 50.81M, 31.50M (bins 1-6). Pair-weight sum / npairs = 124.21, 98.87, 91.96, 71.88, 61.76, 28.11 vs (mean w)^2 = 124.08, 98.88, 91.96, 71.80, 61.54, 27.93: consistent within 0.7 %, so the weight column is the product of catalogue weights and unordered counting is self-consistent.
- Cross-check to official ascii (same pair): official/ours sum(weight) = 1.994, 1.995, 2.003, 1.998, 2.001, 2.005, i.e. official counts ordered pairs (x2), ours unordered; official/ours npairs = 1.414, 1.421, 1.414, 1.419, 1.426, 1.434 (about sqrt 2, not 2; official npairs column is not simply a pair count - cf. OPEN_ISSUE_pair_counts.md; it does not affect the gates, which are scale invariant). Raw amplitude ours/ascii (npairs_weighted) = 1.205, 1.147, 1.178, 1.179, 1.167, 1.170 (the ascii is m-uncorrected and ~0.85 amplitude, consistent).
- Not verifiable here: sign/flip of e1,e2 (npz holds catalogue values; the A1 shape agreement and A3 sign agreement with official xi+/xi- support unflipped) and that the whole-catalogue run really used angle_slop default (only the manifest says so).

## About "ours-vs-theirs max rel diff" (8e+02 ... 2.8e+04)
The metric is max |a-b|/max(|b|,1e-30) over 1000 fine bins; xi crosses zero (xi+ ~ 1e-7 at large theta, xi- changes sign), so a near-zero denominator dominates; it is not a meaningful agreement measure. Quick own check of ours vs theirs on fine bins (npairs ratio 1.0000 per EVAL) was not recomputed beyond the table. Better metric: max |a-b|/sigma_ascii (already in eval.json as max_abs_diff_over_asc_sigma) or amp-ratio.
