# AUDIT_NUMBERS: number audit of the DR5 report and the cross-check prompt

Date 2026-11-02. Scope: `docs/DR5_MEASUREMENT_REPORT.md` and `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md`. Read-only except this file. Documents were NOT fixed.
My scratch scripts (not in the repo): `/tmp/claude-1000/-home-anut/8ecc82c7-22dd-490c-b045-63f93bce2233/scratchpad/audit_chi2.py` and `audit_inline.py`.
Python `/home/anut/anaconda3/bin/python3`, `PYTHONDONTWRITEBYTECODE=1`. No TreeCorr run, nothing launched.

## 1. Verdict

No headline number is wrong. Every quoted chi2, max sigma, PTE, count and timing that I recomputed or read from its file matched. There are 6 minor wording/label problems in the report, 2 stale or wrong statements in the prompt, and 1 consistency note about the "additional part". All cited paths exist; all 7 embedded plots exist.

## 2. Independent recomputation (from the product npz and the FITS COVMATs, no pipeline code imported)

Inputs: `product/dr5_xipm_cosebis_binslop1p5.npz`, `KiDS_Legacy_xipm.fits` (COVMAT 378x378, rows xiP then xiM, FITS row order mapped via BIN1/BIN2/ANGBIN), `KiDS_Legacy_cosebis.fits` (COVMAT 840x840 = [E_n 420 | B_n 420]).
I redid the m-correction from `xip_raw` with m = [-0.023,-0.016,-0.011,0.02,0.03,0.045] and the 9-bin npairs-weighted rebin myself (it equals the stored `xip9/xim9` exactly, diff 0.0).

| # | quantity | my value | stated (report / FULLVECTOR) | match |
|---|---|---|---|---|
| 1 | A3 joint chi2, 378 pts | 0.060852 | 0.0609 | yes |
| 1b | A3 max abs d/sigma; xi+ ; xi- | 0.052775 (Bin5_Bin6) ; 0.052775 ; 0.043849 (Bin6_Bin6) | 0.0528 ; 0.0528 ; 0.0438 | yes |
| 1c | A3 xi+ chi2 / xi- chi2 (189 pts each) | 0.030276 / 0.031615 | 0.0303 / 0.0316 | yes |
| 2 | E_n chi2 6 modes (126) ; max | 0.024194 ; 0.040783 | 0.0242 ; 0.0408 | yes |
| 2b | E_n chi2 20 modes (420) ; max | 0.069827 ; 0.045519 | 0.0698 ; 0.0455 | yes |
| 3 | B_n vs official, block offset 420, 6 modes ; max | 0.022317 ; 0.043673 | 0.0223 ; 0.0437 | yes |
| 3b | B_n vs official, 20 modes ; max | 0.078003 ; 0.060213 | 0.0780 ; 0.0602 | yes |
| 3c | same with the E_n block (the adoption.json bug) 6 / 20 modes | 0.016877 / 0.064391, max 0.029865 / 0.052047 | 0.0169 / 0.0644, 0.0299 / 0.0520 | yes |
| 4 | B3 null, official 126 / 420 pts | chi2 151.058 (PTE 0.063517) / 459.328 (0.090189) | 151.058 (0.0635) / 459.328 (0.0902) | yes |
| 4b | B3 null, ours 126 / 420 pts | 150.876 (0.064796) / 458.803 (0.092995) | 150.876 (0.0648) / 458.803 (0.0930) | yes |
| 5 | A3 amplitude sensitivity, ours x f: failing pairs of 21 at 0.1 | f=1.005: 0 ; 1.01: 2 ; 1.02: 13 ; 1.05: 21 | same | yes |
| 6 | A3 with no m-correction, per-pair max | 0.028 to 0.920; 14 of 21 pairs fail 0.1; Bin1_Bin1 0.124 | "0.12 to 0.92 (fails)", "Bin1_Bin1 needs the correction" | see problem R3 |

So at least seven full-vector chi2 values (A3 joint, A3 xi+, A3 xi-, E_n 126, E_n 420, B_n 126, B_n 420, B3 x4) were recomputed independently and agree to the digits quoted.

Other recomputed items:
- Product integrity: for all 21 pairs sha256 and size of `b1.5/Bin{I}_Bin{J}.npz` equal `source_sha256/source_sizes` in the product; `xip_raw`, `npairs` equal the pair files bitwise; manifest `npairs`/`weight` sums equal the file sums (rel 1e-9); every pair file has `radec_dtype` float64 (crosses store 'float64,float64'), TreeCorr 5.1.4, bin_slop 1.5, angle_slop 0.1, unordered, finite xi. Product `r_nom` equals the official ascii `r_nom` to relative 4.6e-13 (report: 5e-13).
- Inline numbers (report 5.9, none stored in a file), all reproduced from the official ascii and our pair files: whole-pair weight ratio autos 0.49888 to 0.50141, crosses 0.99816 to 1.00257; npairs ratio autos 0.6974 to 0.7074, crosses 1.3988 to 1.4148; fine-bin ranges 0.4987 to 1.0029 (weight) and 0.695 to 1.416 (npairs), as in Figure 2 text; xi_x: max abs 6.37e-4, rms/sqrt(varxi) per pair 0.972 to 1.039, max 4.035.
- a4_recheck.json: counts ratio 0.999961 to 1.0000197 (theta <= 60 arcmin, 68 of 100 bins); xi+ max 0.0300 (Bin6-6) / 0.0309 (Bin1-6); xi- with g2 flip 0.0472 / 0.0517, without 3.970 / 3.208; bin_slop 1.5 vs 0 per-bin npairs differ by up to 0.69% (Bin6-6) / 0.50% (Bin1-6) (report: "about 0.6%"). All match.
- manifest: per-pair seconds, autos 683, 673, 626, 641, 602, 564; crosses 1170 to 1356; total 22,748 s = 6.319 h; 20 threads in all 21. Match. `run_f64_cross.log`: peak RSS 4.40 to 5.02 GB, elapsed 1175 to 1341 s for the 11 crosses (RESUME "1175-1340"). Match.
- b1_integration.json trial table (0.0977/1.577 meanr|edges|rad; 0.0978/1.578 centre; 0.0981/1.582 r_nom; 4.84/13.56 trapz; En scale ratio 0.907; chi2 0.0537 at 420). Match except R1 below.
- paper_gate.json n_eff, ratios (0.9964 to 0.9999), rows; extract_info.json N/S counts; sigma_e.json C.9 divide-by-(1+m) ratios 1.00003, 0.99991, 0.99954, 0.99972, 0.99940, 0.99933 (report 1.0000, 0.9999, 0.9995, 0.9997, 0.9994, 0.9993). Match.
- eval.json: A1 boxcar max 0.787 (Bin1_Bin1, xi-), xi+ max 0.606, pte 0.8891 / 0.4961, C 1.1732 / 1.1821; A_p xi+ 1.131 to 1.322, xi- 1.139 to 1.240 (Bin1_Bin1 1.950); B_n null chi2 difference to official at most 0.358 (Bin5_Bin6, 20 modes) ("within about 0.4"). Match. EVAL.md per-pair A3 values quoted in report 4.6 match.
- exact_bin6_eval_b0_a0.json: xi+ 0.0748, xi- 0.1563, E_n 0.238 (6) / 0.484 (20), B_n null 19.47 vs slop-1.5 19.14; slop 1.5 xi- 0.0888, E_n 0.071; theirs 0.0238 / 0.0438; new_vs_pilot 0.0006 sigma. ANGLE_SLOP.md headline 3: 0.051 / 0.093 / 0.194 / 0.437; R^2 0.989 in `angle_slop/verify/VERIFY.md`. Match.
- fullvector_checks.json step D: row 17608, 127.234, 127.516 (+0.283), chain's own 127.287. Match (read, not recomputed).
- Papers (alphaXiv): Wright+25 arXiv:2503.19441 Table 1 Ngold, n_eff, sigma_e, m_gold all match the report and the code list M; total 40 894 394; effective area 967.4 deg2; footnote 1 on page 6 (Sect. 2.5.1): "We use Treecorr version 4.2.3, with binslop = 1.5". arXiv:2503.19439 (DR5 paper) Table 14 / Eqs. 29-39, 41 409 179 sources, 967.4 deg2, 1014.0 deg2 (Table 12: 1014.031). Match.
- Tests: `pytest tests/test_dr5_product.py tests/test_measure.py tests/test_measure_unordered.py`: 68 passed in 33.9 s (report: 68 passed in 34 s). `pytest_after.log` tail "1 failed, 287 passed, 12 deselected" (report) matches.
- Versions: Python 3.12.4, TreeCorr 5.1.4, numpy 1.26.4, astropy 6.1.0, scipy 1.13.1, fitsio 1.4.2 (numba 0.59.1). 22 cores. Match.
- Paths: every file, script, directory and log named in both documents exists (checked by script plus manual list, including `~/kids_dr5_work/*`, `threshold_calibration/`, `pytest_after.log`, the ini file with `column_npair_name = npairs_weighted` at line 57, the "theirs" `fiducial/` directory). Plots present: `plots/dr5_01_xipm_vs_official.png`, `dr5_02_npairs_weights_ratio.png`, `dr5_03_En_vs_official.png`, `dr5_03b_Bn_vs_official.png`, `dr5_04_integration_vs_measurement.png`, `dr5_05_Bn_null.png`, `dr5_06_estimator_crosscheck.png`.

## 3. Problems (documents not fixed)

### Report `docs/DR5_MEASUREMENT_REPORT.md`

| id | severity | where | what is written | correct value |
|---|---|---|---|---|
| R1 | minor | 3.6 table, last row | "any, arcmin instead of radians: 7e14 and larger / 1e8 and larger" | In `b1_integration.json` the arcmin trials are: chi2(126) 7.4e14 (edges, centre; meanr and r_nom) but 3.7e13 for trapz; max abs(dE)/sigma 1.6e8 (edges, centre) but 1.0e7 for trapz. Correct range: chi2(126) 3.7e13 to 7.4e14, max 1.0e7 to 1.6e8. |
| R2 | minor | 1.2, A3 row ("chi2/378 of only 0.5, 3.0, 12, 27"); same wording in RESUME bug 6 and FULLVECTOR.md ("~12/378") | the numbers are called chi2/378 | They are the TOTAL chi2 for 378 points. I get 0.505, 2.99, 11.98, 27.02 for scale 1.02, 1.05, 1.10, 1.15 on our xi, so chi2/dof is 0.0013, 0.0079, 0.032, 0.071. (Numbers agree with `A1_SALVAGE.md`, which uses the same ambiguous label.) |
| R3 | minor | 3.2 ("no m-correction gives A3 residuals of 0.12 to 0.92 sigma (fails)") | reads as if all pairs fail | 14 of 21 pairs fail 0.1 without m-correction (failing range 0.124 to 0.920); the other 7 pass (0.028 to 0.089). Bin1_Bin1 does fail (0.124). Source (`VERIFY.md` row) has the same shorthand. |
| R4 | minor | 5.4 "xi- of all bin_slop 1.5 variants sits 0.04 to 0.06 sigma" | all variants | The report's own 5.6 and `exact_bin6_eval_b0_a0.json` (`slop15`) give 0.0888 for the float32-era chunked/ordered slop-1.5 Bin6_Bin6 xi-. Listed variants are 0.063 (float32 unordered), 0.0438 (theirs), 0.0438 (ours); the 0.0888 variant is outside 0.04 to 0.06. |
| R5 | minor | 1.2, A1 threshold row "Null simulations (4000, same-sample jitter model, A_p refit)" | 4000 | `threshold_calibration/standardised/STANDARDISED.md`: the refit null uses 12000+6000+4000 sims, 99th percentile quoted from 6000 sims; 4000 is the white-noise sim in the older post-adoption note (`A1_SALVAGE.md`). 99th pct 1.03 and false-fail 9.2% (1.0) are right. |
| R6 | minor | 2, sigma_e bullet "Residual 0.06 to 0.07% in bins 4 to 6" | bins 4 to 6 | Residual from sigma_e.json: bin 4 0.03%, bin 5 0.06%, bin 6 0.07%. RESUME says "bins 5-6", which is right. |
| R7 | minor | 3.5 "The 1.406 to 1.415 ratio ... Ratio of whole-pair sums: crosses 1.399 to 1.415" | two different ranges for the same quantity | Whole-pair npairs ratio for crosses is 1.3988 to 1.4148 (my recompute); 1.406 is the Bin2_Bin2 (auto) number from `PAPER_SELECTION.md` in another convention. State one range and say which pair types. |
| R8 | note | 1.1 table and 5.10 on "additional part" | scope "not specified here" | The task brief says to treat it as unspecified, and the report does. But `runs/dr5_measure/RESUME.md` (header and "Not yet done") says the additional part is the ~45 h exact full run and gives its gate (comparison with the bin_slop 1.5 product, no pass/fail). The report and RESUME therefore differ in what they say; the PI's own wording was not seen by me. No number is affected. |

### Prompt `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md`

| id | severity | where | what is written | correct value |
|---|---|---|---|---|
| P1 | minor | R7 ("`weight == shear_weight_only` for only 69-84% of rows") | 69-84% | `~/kids_dr5_work/extra/extra_info.json` `weight_eq_shearw_frac` per bin: 0.692, 0.814, 0.845, 0.913, 0.964, 0.557. Range is 56% to 96%. |
| P2 | minor | R5 "EVAL.md title says 'auto-pair' but covers 21 pairs" | stale | `b1.5/EVAL.md` now begins "Unordered pair evaluation (21 pairs), source=ours, bin_slop=1.5". The "auto-pair" title was in the pre-fix version (`verify21/VERIFY.md` discrepancy 1). `run_f64_all.log` autos + 4 crosses and `git_head` null are still true. |
| P3 | minor | "Report (being written in parallel, may not exist yet)" and "Files to read" item 3 "may be absent" | stale | The report exists (41,939 bytes). |

All other numeric claims in the prompt (claims 1 to 17 table, R4 sensitivity 1.01 -> 2/21 and 1.02 -> 13/21, A3 values, B1/B2/B3 numbers, row 17608, a4 numbers, failure-mode numbers 0.093-0.156, 0.44-0.48, 0.10-0.15, 9%/1.03, 4.6 GB) matched their files.

## 4. Stated plainly: what I did not check

- The E_n/B_n arrays in the product were compared to the FITS, but I did NOT regenerate them from the fine xi with the integrator (R3 of the prompt was not redone). So "product En/Bn equal what the integrator gives" rests on the producer script, not on me.
- Step D (model chi2 127.234 / 127.516) read from `fullvector_checks.json`, not rerun.
- I did not view the plots; I only confirmed the files exist. Figure captions with plot-derived numbers (e.g. Figure 4 rms 0.387, 0.603, 0.0151, 0.0156; "rms about 1.2 sigma at n = 1") were not recomputed.
- Not recomputed, only read from md/json: wrong-pair-row sensitivity (1.6 to 20 sigma), swap xi+/xi- (3 to 7 sigma), ours-vs-theirs 3.87e-11 and npairs bitwise 21/21 (I have not loaded the "theirs" files), the GGL sign numbers (S/N +8.6, +15.6, amplitude 2.15e-3), `convention.json` scores (read: 0.1635, -0.0476, matches), A1 null-sim percentiles, the 150k-subsample xi- flip (0.036 vs 0.021), bootstrap ratios 1.005/1.012, full-covariance pte 0.574/0.653, the angle_slop patch table, the claim that angle_slop was added in TreeCorr 5.0, the 4.2.3 release being not installed, float32 position error 0.055 arcsec and 0.10-0.15 sigma rms, extract timing 432/451 s (log lines read: match), the "1 failed" old test log content.
- I did not read the catalogue FITS (TOMOBIN mapping), did not check the extract against the catalogue, and did not mutate tests.
- A_p, A2 per-pair tables in FULLVECTOR.md were checked against `fullvector_checks.json` by spot reading and by the overall medians/maxima in the report (1.759/1.980/2.532/2.659 and medians 0.158/0.149/0.151/0.154: all equal the json), not cell by cell.
- Citation volume/page strings (A&A 703, A158) were not verified; arXiv IDs, titles, Table 1, Table 12, Table 14 and the TreeCorr footnote were.
