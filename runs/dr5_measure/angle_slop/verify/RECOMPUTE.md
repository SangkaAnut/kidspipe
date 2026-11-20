# RECOMPUTE (verifier "recompute"): independent recomputation of C2 and C3

Script: scripts/verify_angle_slop_recompute.py (own parsing of the FITS, own rebin, own E_n/B_n sums; only CosebisFilters from src/kidspipe/theory/cosebis.py reused). Numbers: recompute_numbers.json. 2 threads.

## Setup checks
- xipm FITS: COVMAT STRT_0=0 NAME_0=xiP, STRT_1=189 NAME_1=xiM (header-confirmed). Bin6_Bin6 rows 180..188 in both xiP and xiM; cosebis FITS COVMAT STRT_0=0 En, STRT_1=420 Bn, Bin6_Bin6 rows 400..419. Right diagonal block used (I selected rows by BIN1==BIN2==6, not by pair index). Official ANG (2.64..227.1 arcmin) are arcmin; our rebin uses 9 log edges 2-300.
- m-correction: all three of ours divided by (1+0.045)^2 = 1.092025 (m_6=0.045, scripts/dr5_adopt_01.py:46); applied identically. Raw (no m) is not used. Total npairs exact = pilot = 1.6136e12; slop15 1.61359e12.
- Units: E_n = 0.5*(ARCMIN^2)*sum_k w_k(Tp xi+ + Tm xi-), w_k=(b_{k+1}^2-b_k^2)/2 in arcmin^2, result in rad^2; matches official E_n scale (3.4e-9). sigma = sqrt(diag) of En block (2.7e-10 .. 7e-10), consistent units.

## Kernel / integrator cross-checks
- Independent Simpson E_1 (20001-pt log grid, interpolated exact xi): 3.42687e-9 vs my fine-bin sum 3.42688e-9 (3e-6 relative). Uses same T filters though.
- Real independent check of the filters: official-release-independent "THEIRS" m-corrected xi (HI-gamma xipm_bin6_bin6.npz) through the same code gives E_n - official = 0.0408 sigma (6 modes), 0.0431 (20); A3 xi+ 0.0238, xi- 0.0438. Reproduces the adoption claim; so filters+units+covariance indexing are right to ~0.04 sigma.
- My attempt at an independent T_- from the Schneider-type tail integral did NOT match (my quick formula/convention was wrong; all sign variants disagree), so T_- was NOT independently verified by quadrature; only validated end-to-end via the THEIRS->official comparison above. Flagged as a gap.

## Results (max |Delta|/sigma_official, bins weighted by our npairs = adoption convention, dr5_adopt_01.py:104 uses t["npairs"])
| set | xi+ 9 | xi- 9 | E 6 modes | E 20 modes | B null chi2/6 | B chi2/20 |
|---|---|---|---|---|---|---|
| exact (0,0) | 0.0748 | 0.1563 | 0.2380 | 0.4844 | 19.465 | 32.234 |
| pilot (0,0.1) | 0.0748 | 0.1568 | 0.2375 | 0.4841 | 19.470 | 32.243 |
| slop15 (1.5,0.1) | 0.0456 | 0.0888 | 0.0710 | 0.0710 | 19.144 | 31.744 |
Exact xi- per bin: -0.013 -0.050 -0.040 0.018 0.018 0.082 -0.005 0.003 -0.156 (150-300' bin). Exact xi+ max at bin 5 (-0.0748). E_n exact: monotone drift -0.053 (n=1) to -0.484 (n=20). chi2(E-Eoff, 6 modes, official cov) exact 0.081, slop15 0.011.
Weighting sensitivity (exact / pilot / slop15 xi+, xi-): npairs: .0748/.1563, .0748/.1568, .0456/.0888; our `weight` column: .0741/.1560, .0741/.1565, .0468/.0958; official ascii npairs_weighted: .0743/.1563, .0743/.1568, .0455/.0926. So slop15 xi- ranges 0.089-0.096 by weighting (spread 0.007 > 0.005 tolerance, but the adoption convention npairs gives exactly the claimed 0.0888). Verdict on A3 unaffected: exact xi- FAIL (0.156 > 0.1) under all weightings; slop15 PASS (<0.1) under all weightings.
(The official fine-bin ascii is a blinded realisation, so its npairs_weighted is only a weight.)

## C3: exact vs slop15 (our code, same data), sigma_official
xi+ 0.0511, xi- 0.0929, E6 0.1944, E20 0.4367 (monotone dE 0.028 -> 0.437). pilot-exact: 0.00003 / 0.00058 / 0.0004 / 0.0010.

## Comparison with claimed values (flag > 0.005)
C2: xi+ 0.0748 ok, xi- 0.1563 ok, E6 0.238 ok, E20 0.484 ok, B chi2 19.47/6 ok; pilot 0.1568/0.2375/0.4841 ok (to 6e-4). No discrepancy.
C3: 0.051/0.093/0.194/0.437 ok; slop15 vs official 0.0456/0.0888/0.071 ok (npairs weighting). No discrepancy > 0.005 with the stated convention.
Caveat: the "slop15 vs official 0.071 for E6 and E20" is one number because the max over 20 modes is at n=4 (0.071); the slop15 dE changes sign across n.

## Not checked / caveats
- C1 (brute force), C4 interpretation, C5, C6 not addressed here. C4 is only suggestive from these numbers: slop15 is closer to official than exact on xi- (0.089 vs 0.156) and E_n (0.071 vs 0.484) but official COVMAT sigma is a model covariance and the official xi is from the same galaxies, so a systematic of 0.1-0.5 sigma is a legit, but not unique, explanation (e.g. different m/weights/shear-bias treatment would also give a smooth E_n drift; the exact-vs-official E_n drift is monotone, similar to the slop15-exact drift, which supports C4 but does not prove which bin_slop the release used).
- m_6=0.045 taken from the repo, not independently verified against the release header.
