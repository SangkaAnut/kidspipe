# Exact run vs bin_slop 1.5 product (all 21 pairs)

Product: `runs/dr5_measure/product_exact/dr5_xipm_cosebis_exact.npz` (bin_slop 0, angle_slop 0, unordered convention). Reference: `runs/dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz`.
PI decision (2026-11-02): the exact run is not gated; the differences below are reported, with no pass/fail.
All differences are m-corrected and in units of the release's own sigma (diagonal of the official covariance): 9-bin xi
(npairs-weighted rebin, each run with its own npairs) and COSEBIs modes 1-20.

## Per pair

The last two columns set the exact-minus-b1.5 xi difference to zero in the outermost 5 fine bins (theta 292.6-300
arcmin) before computing the COSEBIs difference; the bin_slop 1.5 values are kept there. They show how much of the COSEBIs
difference comes from those bins; they are not COSEBIs of a truncated xi.

| pair | xi+ ex-b1.5 | xi- ex-b1.5 | xi+ ex-off | xi- ex-off | xi+ b1.5-off | xi- b1.5-off | E n<=6 | E n<=20 | B n<=6 | B n<=20 | E n<=20, edge diff zeroed | B n<=20, edge diff zeroed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 0.076 | 0.088 | 0.073 | 0.078 | 0.010 | 0.010 | 0.075 | 0.260 | 0.156 | 0.530 | 0.105 | 0.169 |
| 1-2 | 0.071 | 0.103 | 0.070 | 0.101 | 0.010 | 0.010 | 0.151 | 0.240 | 0.103 | 0.181 | 0.113 | 0.078 |
| 1-3 | 0.121 | 0.094 | 0.119 | 0.099 | 0.009 | 0.007 | 0.043 | 0.175 | 0.033 | 0.071 | 0.220 | 0.109 |
| 1-4 | 0.061 | 0.081 | 0.055 | 0.077 | 0.021 | 0.017 | 0.047 | 0.126 | 0.311 | 0.579 | 0.056 | 0.099 |
| 1-5 | 0.104 | 0.125 | 0.107 | 0.133 | 0.028 | 0.031 | 0.088 | 0.245 | 0.016 | 0.112 | 0.088 | 0.133 |
| 1-6 | 0.044 | 0.148 | 0.063 | 0.174 | 0.026 | 0.026 | 0.134 | 0.157 | 0.303 | 0.468 | 0.138 | 0.180 |
| 2-2 | 0.077 | 0.066 | 0.083 | 0.069 | 0.007 | 0.010 | 0.053 | 0.191 | 0.099 | 0.328 | 0.116 | 0.387 |
| 2-3 | 0.141 | 0.146 | 0.151 | 0.148 | 0.010 | 0.008 | 0.058 | 0.125 | 0.105 | 0.205 | 0.063 | 0.086 |
| 2-4 | 0.075 | 0.113 | 0.081 | 0.117 | 0.014 | 0.012 | 0.056 | 0.431 | 0.039 | 0.143 | 0.139 | 0.115 |
| 2-5 | 0.059 | 0.113 | 0.057 | 0.125 | 0.023 | 0.012 | 0.191 | 0.301 | 0.103 | 0.188 | 0.211 | 0.069 |
| 2-6 | 0.044 | 0.108 | 0.040 | 0.102 | 0.013 | 0.014 | 0.102 | 0.238 | 0.251 | 0.284 | 0.101 | 0.136 |
| 3-3 | 0.084 | 0.125 | 0.076 | 0.129 | 0.008 | 0.014 | 0.306 | 0.605 | 0.065 | 0.144 | 0.112 | 0.076 |
| 3-4 | 0.079 | 0.094 | 0.073 | 0.103 | 0.014 | 0.015 | 0.047 | 0.194 | 0.095 | 0.425 | 0.113 | 0.268 |
| 3-5 | 0.056 | 0.057 | 0.068 | 0.091 | 0.024 | 0.034 | 0.083 | 0.116 | 0.016 | 0.098 | 0.174 | 0.139 |
| 3-6 | 0.054 | 0.040 | 0.054 | 0.051 | 0.022 | 0.018 | 0.083 | 0.273 | 0.015 | 0.179 | 0.062 | 0.053 |
| 4-4 | 0.063 | 0.087 | 0.059 | 0.097 | 0.030 | 0.031 | 0.064 | 0.064 | 0.256 | 0.369 | 0.083 | 0.134 |
| 4-5 | 0.049 | 0.101 | 0.077 | 0.135 | 0.042 | 0.035 | 0.256 | 0.576 | 0.042 | 0.163 | 0.090 | 0.056 |
| 4-6 | 0.088 | 0.114 | 0.064 | 0.107 | 0.036 | 0.018 | 0.069 | 0.102 | 0.120 | 0.292 | 0.026 | 0.120 |
| 5-5 | 0.070 | 0.069 | 0.072 | 0.075 | 0.030 | 0.031 | 0.022 | 0.103 | 0.021 | 0.328 | 0.035 | 0.063 |
| 5-6 | 0.044 | 0.113 | 0.070 | 0.121 | 0.053 | 0.031 | 0.056 | 0.151 | 0.090 | 0.090 | 0.050 | 0.073 |
| 6-6 | 0.072 | 0.114 | 0.078 | 0.158 | 0.024 | 0.044 | 0.216 | 0.459 | 0.175 | 0.347 | 0.071 | 0.112 |

Ranges over 21 pairs: xi+ exact-b1.5 0.044 to 0.141, xi- 0.040 to 0.148; exact-official xi+ 0.040 to 0.151, xi- 0.051 to 0.174; b1.5-official xi+ 0.007 to 0.053, xi- 0.007 to 0.044; E_n n<=6 0.022 to 0.306, n<=20 0.064 to 0.605; B_n n<=6 0.015 to 0.311, n<=20 0.071 to 0.579; with the edge-bin difference zeroed E_n n<=20 0.026 to 0.220, B_n n<=20 0.053 to 0.387.
Fine bins: median |exact-b1.5|/sqrt(varxi) 0.246 to 0.324 (xi+), 0.260 to 0.343 (xi-); mean signed -0.0011 to +0.0007 (xi+), -0.0014 to +0.0008 (xi-). Total pair counts exact/b1.5 1.0000041 to 1.0000057. max |meanr / r_nom - 1| = 3.2e-04 (fine-bin half-width in ln theta: 2.5e-03).

## Full vector (official covariance)

chi2 of a difference uses the release covariance only as a yardstick: the exact-minus-b1.5 difference is not a draw from it,
so chi2/dof << 1 is expected; it says how large the shift is relative to the statistical errors.

| difference | xi+ (189) | xi- (189) | joint (378) | E 1-6 (126) | E 1-20 (420) | B 1-6 (126) | B 1-20 (420) |
|---|---|---|---|---|---|---|---|
| exact - b1.5 | 0.484 | 0.568 | 1.063 | 0.561 | 5.054 | 0.522 | 4.740 |
| exact - official | 0.532 | 0.611 | 1.153 | 0.627 | 5.250 | 0.583 | 4.907 |
| b1.5 - official | 0.030 | 0.032 | 0.061 | 0.024 | 0.070 | 0.022 | 0.078 |

B_n null test (chi2 of B_n with the official B covariance; not a gate here):

| B_n of | modes 1-6: chi2 (126 points) | PTE | modes 1-20: chi2 (420 points) | PTE |
|---|---|---|---|---|
| exact | 152.00 | 0.0572 | 457.39 | 0.1009 |
| bin_slop 1.5 product | 150.88 | 0.0648 | 458.80 | 0.0930 |
| official release | 151.06 | 0.0635 | 459.33 | 0.0902 |

(The official row reproduces the published PTEs 0.0635 / 0.0902; that checks the covariance indexing.)

## Model chi2 of E_n modes 1-6 (step D)

`LegacyCosebisModel` at the official best-posterior chain row 17608, official E_n covariance, 126 points: official data 127.234 (chain quotes 127.29), bin_slop 1.5 product 127.516 (+0.283), exact 127.709 (+0.476).

## Where the COSEBIs difference comes from (angular decomposition, all 21 pairs)

COSEBIs are linear in xi, so the exact-minus-b1.5 difference splits exactly into contributions from angular ranges
(the code asserts the parts add up). Per pair, max over modes 1-20 (and 1-6) of |contribution| / sigma:

| range | E n<=6 | E n<=20 | B n<=6 | B n<=20 |
|---|---|---|---|---|
| 2-100 arcmin | 0.005 to 0.032 | 0.007 to 0.050 | 0.005 to 0.046 | 0.007 to 0.046 |
| 100 arcmin to the edge bins | 0.014 to 0.190 | 0.029 to 0.231 | 0.011 to 0.370 | 0.051 to 0.370 |
| outermost 5 fine bins (292.6-300 arcmin) | 0.007 to 0.344 | 0.070 to 0.547 | 0.018 to 0.485 | 0.023 to 0.625 |

Filter weighting: per unit change of xi in one fine bin, the mean |response| of a mode to the outermost 5 bins is 17 to 374 times its median response over all 1000 bins, over modes 1-20 and both xi+ and xi- (the same for B_n, 17 to 374: E_n and B_n use the same two filters and differ only in sign).

## Reading

- COSEBIs differences come from large angles: below 100 arcmin the difference contributes at most 0.050 sigma to any mode in any pair; 100 arcmin to the edge bins up to 0.23 (E_n) and 0.37 (B_n); the outermost 5 fine bins up to 0.55 (E_n) and 0.63 (B_n).
- xi: exact differs from bin_slop 1.5 by at most 0.148 sigma per 9-bin point (joint chi2 of the difference 1.06 over 378 points). The bin_slop 1.5 product is much closer to the release (joint 0.061), which used bin_slop 1.5, so the exact-vs-official
  difference is essentially this exact-vs-b1.5 shift.
- Averaged over all 1000 fine bins the signed exact-minus-b1.5 difference is zero (mean z above), i.e. no overall shift of xi.
  That does not exclude local differences: see the next point.
- COSEBIs: zeroing the exact-minus-b1.5 difference in the outermost 5 fine bins at least halves max|dE_n| (modes 1-20) in 13 of 21 pairs and max|dB_n| in 13. Five largest, full -> edge difference zeroed: E_n 3-3 0.61->0.11, 4-5 0.58->0.09, 6-6 0.46->0.07, 2-4 0.43->0.14, 2-5 0.30->0.21; B_n 1-4 0.58->0.10, 1-1 0.53->0.17, 1-6 0.47->0.18, 3-4 0.43->0.27, 4-4 0.37->0.13. It raises the value in 3 (E) and 4 (B) pairs, where the edge part partly cancels the rest: E_n 1-3 0.17->0.22, 3-5 0.12->0.17, 4-4 0.06->0.08; B_n 1-3 0.07->0.11, 1-5 0.11->0.13, 2-2 0.33->0.39, 3-5 0.10->0.14. For modes 1-6 it raises max|dE_n| in 7 pairs (1-3 0.04->0.15, 2-2 0.05->0.09, 2-4 0.06->0.13, 2-5 0.19->0.19, 3-4 0.05->0.07, 4-4 0.06->0.07, 5-5 0.02->0.03).
  The sign of the largest E_n difference varies between pairs (12 positive, 9 negative).
  The large weight of the edge bins in the COSEBI filters (above) is why a small re-binning difference there shows up in E_n/B_n.
- The B_n null of the exact measurement is listed for information; no gate applies.

Arrays for plots (per pair, 9-bin and per-mode differences in sigma, with and without the edge-bin part) are in `exact_vs_b1p5.json`.
