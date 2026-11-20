# Prompt: adversarial cross-check of the KiDS-DR5 (KiDS-Legacy) shear measurement

You are a fresh Claude instance with no context. Audit, do not confirm. Your job is to find errors, overstated claims and
checks that cannot fail in the work below. Report only what you computed or read. Say plainly what you could not check.

## Hard rules
- Project root: `/mnt/d/cloning_kids_Kappa_pipeline` (all paths below are relative to it unless absolute).
- `/mnt/d/HI-gamma-correlation/` is READ-ONLY (official data, catalogue). Run python touching it with `PYTHONDONTWRITEBYTECODE=1`.
- Python: `/home/anut/anaconda3/bin/python3` with `PYTHONPATH=src` (add `scripts` if you import them). Never `-I`. Never pip install.
- Machine: 16 GB RAM, 22 cores. NO full-catalogue TreeCorr, no job over ~10 min, one process per pair (a cross pair needs ~4.6 GB).
- Do not launch any `*.sh` under `runs/` (never `runs/dr5_measure/angle_slop/queue4.sh`). Do not edit gates, thresholds, existing xi products or eval files.
  Write your scratch scripts and outputs to a new directory (e.g. `/tmp/crosscheck/`), not into `runs/`.
- If you delegate, use Sonnet 5.5 subagents at medium effort only (standing user preference). Give each a narrow, checkable task and re-verify what it returns.

## What is being claimed
Product (PI decision (a), 2026-11-02): TreeCorr 5.1.4, bin_slop 1.5, angle_slop default 0.1, unordered pairs, float64 positions, catalogue
`weight`, e1/e2 as stored (no flip), 21 tomographic pairs, 1000 log bins 2-300 arcmin. An exact (bin_slop 0) run of all pairs was NOT done;
the ~45 h exact full run is planned later, not started, and out of scope for this audit. The release used TreeCorr 4.2.3.
Pair files: `runs/dr5_measure/xi_unordered/b1.5/Bin{I}_Bin{J}.npz` (r_nom, xip, xim RAW = m-uncorrected, npairs, weight).
Product: `runs/dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz`. Report: `docs/DR5_MEASUREMENT_REPORT.md`.

| # | Claim | Holding file | Producing script |
|---|---|---|---|
| 1 | Per-bin row counts = Wright+25 Table 1 Ngold exactly: 7416371, 7359911, 6799681, 6880432, 6477538, 5960461 (total 40,894,394, no bad TOMOBIN) | `~/kids_dr5_work/extract_info.json`, `runs/dr5_measure/paper_gate.json` | `scripts/dr5_extract.py`, `scripts/dr5_area.py` |
| 2 | Extract is float64 for ra/dec/e1/e2/w and equals the earlier pipeline's float64 extract bitwise (ra/dec) | `runs/dr5_measure/xi_unordered/extract_check_f64.json` | `scripts/dr5_extract_check_f64.py` |
| 3 | sigma_e (Joachimi+21 C.9, w^2, divided by 1+m) / header SIG_E = 1.0000, 0.9999, 0.9995, 0.9997, 0.9994, 0.9993 | `runs/dr5_measure/sigma_e/sigma_e.json` | `scripts/dr5_sigma_e.py` |
| 4 | Shear convention: catalogue e1,e2 as is, no flip (GGL gamma_t > 0, S/N +8.6; PairCounter needs (e1,-e2)) | `runs/dr5_measure/sign_convention/sign.json` | `scripts/dr5_sign_convention.py`, `scripts/dr5_sign_verify.py` |
| 5 | All 21 pairs: A3 (9-bin xi vs official FITS, m-corrected) max 0.0528 sigma (Bin5_Bin6 xi+), xi- max 0.0438; gate 0.1, 21/21 pass | `runs/dr5_measure/xi_unordered/b1.5/eval.json`, `EVAL.md` | `scripts/dr5_measure_xi_unordered.py` (`eval`) |
| 6 | A1 (boxcar/white <= 1.03, max 0.787 Bin1_Bin1; const-A_p pte 0.889 xi+ / 0.496 xi-) passes. Blind to common amplitude and xi- sign | same `eval.json` (`A1_const`) | same, `kidspipe/measure/a1check.py` |
| 7 | Ours = theirs (earlier pipeline's xi): npairs bitwise 21/21, xi <= 3.9e-11. Code equivalence only, same TreeCorr 5.1.4 and extract | `runs/dr5_measure/xi_unordered/verify21/VERIFY.md`, `eval.json` (`vs_theirs`) | `scripts/dr5_measure_xi_unordered.py`, `scripts/dr5_adopt_01.py` |
| 8 | B1 (official fine ascii -> our integrator -> official E_n): chi2/126 = 0.098, chi2/420 = 0.054, max 1.58 sigma, trial `meanr\|edges\|rad`. Not amplitude-free: ascii has ~0.907 En scale | `runs/dr5_measure/b1_integration.json` | `scripts/dr5_b1_integration.py` |
| 9 | B2 (OUR xi -> E_n vs official, official E_n cov): chi2 0.0242/126, 0.0698/420, max 0.0455 sigma (E_n); B_n 0.0223/126, 0.0780/420 (uses B block offset +420). Ungated | `runs/dr5_measure/product/fullvector_checks.json` | `scripts/dr5_make_product.py make` |
| 10 | B3 B_n null: ours chi2 150.876 / PTE 0.0648 (126 pts), 458.803 / 0.0930 (420); official 151.058 / 0.0635, 459.328 / 0.0902 (published) | `fullvector_checks.json` | `scripts/dr5_make_product.py make` |
| 11 | A3 full 378-vector: joint chi2 0.0609/378, max 0.0528 sigma; xi+ 0.0528, xi- 0.0438 | `fullvector_checks.json`, `product/FULLVECTOR.md` | `scripts/dr5_make_product.py make` |
| 12 | Step D: best-posterior chain row 17608, chi2(126) = 127.234 with official E_n (spec 127.23), 127.516 with ours | `fullvector_checks.json` (`D`) | `scripts/dr5_make_product.py stepd` |
| 13 | Bug in `scripts/dr5_adopt_01.py`: its B1 B_n-vs-official numbers use the E_n cov block (no +420), so those entries in `adoption/adoption.json` are invalid; A3, E_n, B3 entries unaffected. Not edited | `fullvector_checks.json` (`vs_adoption`) | `scripts/dr5_make_product.py` |
| 14 | Estimator: independent `paircount` vs TreeCorr on a ~1 deg disc: counts match (ratio 0.99996-1.00002), xi+ max 0.03 sigma, xi- matches only with g2 flipped (0.05 vs 3-4 sigma). Subsample, not full catalogue; a frame/convention difference of the counter | `runs/dr5_measure/product/a4_recheck.json`, `a4_recheck.log` | `scripts/dr5_make_plots.py a4` |
| 15 | Seven report plots, viewed: `plots/dr5_01` to `dr5_06` plus `dr5_03b` | `plots/` | `scripts/dr5_make_plots.py plots` |
| 16 | Pre-product evidence: exact Bin6_Bin6 (float32 era) fails A3 xi- 0.156 sigma; bin_slop 1.5 explains 98.9% of the E_n gap; angle_slop does not | `runs/dr5_measure/angle_slop/ANGLE_SLOP.md`, `exact_bin6_eval_b0_a0.json` | `scripts/dr5_angle_slop_*.py`, `scripts/dr5_pilot_bin6_exact.py` |
| 17 | Official `npairs` column is a normalised non-integer column, so a raw-count A1 cannot pass; use `npairs_weighted` | `runs/dr5_measure/PAPER_SELECTION.md` | `scripts/dr5_diagnose_ratio.py` |

## Cheap re-run subset (write your OWN short scripts; do not import the evaluator for the independent parts)
Use `kidspipe.measure.xi.load_bin(b)` (1-based bin; file `bin{b-1}.npz`) only to read arrays; it has keys ra, dec, e1, e2, w, north.
Official data: `/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/data/`
(`KiDS_Legacy_xipm.fits`, `KiDS_Legacy_cosebis.fits`, `cosmosis_xipm/*.ascii`).
- R1 Catalogue counts (claim 1): without reading the whole catalogue into RAM twice, `np.load` each `bin{0..5}.npz` and compare `len(ra)` to Table 1 Ngold above;
  check dtype float64; check `sum(w)`. Optionally read only `TOMOBIN` of `/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_NS_unblind_final.fits` with fitsio in chunks and `np.bincount`
  it (read-only, one column) to confirm the mapping TOMOBIN k -> `bin{k-1}.npz`.
- R2 Subsample auto + cross pair at bin_slop 1.5 (claims 5, 7): take a disc of ~0.5 deg radius (about 5e4 galaxies per bin; pick Bin6 and Bin1)
  with ra/dec cuts. Run a fresh `treecorr.GGCorrelation(nbins=1000, bin_type="Log", min_sep=2, max_sep=300, sep_units="arcmin", bin_slop=1.5, num_threads=8)`
  with `treecorr.Catalog(ra, dec, ra_units="deg", dec_units="deg", g1=e1, g2=e2, w=w)`: auto `process(cat6)`, cross `process(cat1, cat6)`.
  Seconds only. Compare with `kidspipe.measure.xi.run_pair` output, and check the unordered auto convention (auto npairs = half of the ordered count).
  Compare also the independent counter `kidspipe.measure.paircount` for xi+ (and xi- with e2 sign handled as in `a4_recheck.log`).
  The full-pair `Bin6_Bin6.npz` cannot be re-measured; instead assert `npairs.sum()`, `weight.sum()` against `manifest.json` and finiteness.
- R3 Recompute B1 (claim 8) yourself: read the 21 official ascii files, rebuild the COSEBIs integral with theta in radians and
  bin-edge weights (0.5*(e_{k+1}^2 - e_k^2)), filters from `kidspipe.theory.cosebis.CosebisFilters(2.0, 300.0, source="mpmath")`, compare with `KiDS_Legacy_cosebis.fits` En
  using the 420x420 E block of COVMAT (B block starts at 420). Confirm 0.098/126 and max 1.58 sigma, and that arcmin instead of radians gives a huge chi2.
- R4 Recompute A3 (claim 5) for 3 pairs including Bin1_Bin1, from `b1.5/*.npz` and `KiDS_Legacy_xipm.fits` without importing `dr5_measure_xi_unordered`:
  divide raw xi by (1+m_i)(1+m_j) with m = [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045], rebin to 9 log bins by npairs weights, divide by sqrt(diag COVMAT).
  Then deliberately apply no m-correction, a swapped pair, flipped xi- sign, and xi x1.02 and confirm the stated sensitivity (xi x1.01 fails 2/21, x1.02 13/21).
- R5 JSON vs report (claim list above and the report): reopen `eval.json`, `fullvector_checks.json`, `b1_integration.json`, `a4_recheck.json`, `paper_gate.json`
  and diff every number quoted in `docs/DR5_MEASUREMENT_REPORT.md` (if present), `product/FULLVECTOR.md`, `b1.5/EVAL.md`, `runs/dr5_measure/RESUME.md`. Flag stale statements
  (`run_f64_all.log` holds only autos + 4 crosses; manifest `git_head` is null).
- R6 Run `PYTHONPATH=src /home/anut/anaconda3/bin/python3 -m pytest tests/test_dr5_product.py tests/test_measure.py tests/test_measure_unordered.py -q` (fast, no 16 GB file). The last full-suite result is `runs/dr5_measure/product/pytest_full.log`.
- R7 Weights: from `~/kids_dr5_work/extra/` (columns `weight`, `shear_weight_only`, `gold_weight_only`; `extra_info.json` shows `weight == shear_weight_only` for only 56-96% of rows, per bin)
  confirm the product used `weight` (the shipped final shear+gold weight, per the catalogue readme and `PAPER_SELECTION.md`), and quantify how much a subsample xi+ changes if `shear_weight_only` is used.

## Failure modes to hunt (for each: say how you tested it and the numeric outcome)
1. Shear sign or flip: e2 sign (xi- only), a global flip, `flip_g1/flip_g2` in `make_treecorr_cat`. Note the open caveat: sign validated by GGL on a subsample and by A3, not on the full catalogue; A1 cannot see an xi- sign flip.
2. m-correction doubled or missing: `xip_raw` vs `xip` in the product (`/ (1+m_i)(1+m_j)`); official fine ascii is m-uncorrected, FITS 9-bin is corrected. Check each comparison uses the matching one, and that Bin1_Bin1 (m = -0.023) needs it.
3. Weights: `weight` vs `shear_weight_only`; 9-bin rebin weights (own npairs vs `npairs_weighted`); `sigma_e` using w^2.
4. TOMOBIN mapping: `bin{b-1}.npz` off-by-one (fixed bug 1: this once measured bin b+1); pair-to-row mapping in FITS vs file order (pair-major, `(1,1),(1,2),...,(6,6)`); wrong mappings must fail A3 by many sigma.
5. theta units: arcmin vs radians in the COSEBIs integral (`unit="rad"` multiplies by arcmin^2 in radians); TreeCorr `sep_units='arcmin'` missing silently empties every bin.
6. Integration rule: trapezoid vs bin-averaged ("edges") weights; the `trapz` scheme had a missing-factor-theta bug (fixed, not on the B1 path) and keeps a ~2% O(dln theta) end error at n=4.
7. Patch combination: whole-catalogue unordered runs (no patches) vs the chunked ordered driver (fold factor, auto-pair Im xi+ fix, `varxi` of ordered auto is half). Confirm no chunked/ordered file leaked into the product.
8. bin_slop effects: product is 1.5 (as the release), not exact; the exact Bin6_Bin6 differs (xi- 0.093-0.156 sigma, E_n up to 0.44-0.48 sigma). Check nothing claims the product is "exact" or that bin_slop 1.5 is unbiased (suggestive only, t~2.3). angle_slop default 0.1 vs 0 changes xi- at coarse binning only.
9. Float32 positions: earlier xi used float32 ra/dec (rms 0.10-0.15 sigma in fine bins). Confirm every product input has `radec_dtype float64` in `manifest.json` and that no `b1.5_f32` file is mixed in. The gates cannot distinguish float32 from float64; only ours-vs-theirs does.
10. Tests that cannot fail: read `tests/test_dr5_product.py`, `tests/test_measure_unordered.py` (`test_theirs_bin6_pins_known_gate_values`), `tests/test_measure.py`. For each test, mutate the code or the input (swap pair, drop m, change a constant) and see whether it fails. Report tests that still pass.
11. Report numbers not matching files (R5). Also check wording: "ours = theirs" must be described as code equivalence, not release validation; "A3 resolves ~1%" is the only amplitude sensitivity; E_n/B_n comparisons are ungated (no numeric target in the spec).
12. Post-hoc thresholds: A3 < 0.1 and A1 boxcar <= 1.03 were set after seeing results. The justification is a same-sample null simulation (99th pct of the A1 max is 1.03; 1.0 false-failed ~9%). Re-read `runs/dr5_measure/a1_salvage/A1_SALVAGE.md` and `kidspipe/measure/a1check.py`: is the null sim representative, is the 1% size honest, were thresholds chosen to pass?
13. Product-vs-prior consistency: `adoption/adoption.json` bug (claim 13); also m_gold is the project's Table 1 list, not read from the release.
14. Unsupported physical claims: A2 (ascii amplitude ~1.13-1.32, unexplained) is information only; the ascii is not a truth reference; the estimator-check frame difference (claim 14) is a convention difference, not shown to be correct or incorrect for the product.

## Files to read, in priority order
1. `runs/dr5_measure/RESUME.md` (state, decisions, bugs 1-16)
2. `docs/HANDOFF_PROMPT_DR5_MEASUREMENT.md` sections 4-7 (the spec)
3. `docs/DR5_MEASUREMENT_REPORT.md` (the claims as written)
4. `runs/dr5_measure/xi_unordered/verify21/VERIFY.md`, `b1.5/EVAL.md`, `b1.5/eval.json`, `b1.5/manifest.json`
5. `scripts/dr5_measure_xi_unordered.py` (evaluator), `scripts/dr5_make_product.py` (product + full vector), `src/kidspipe/measure/{xi,xi2cosebis,official,a1check,paircount}.py`
6. `runs/dr5_measure/product/{README.md,FULLVECTOR.md,fullvector_checks.json,a4_recheck.json}`
7. `scripts/dr5_extract.py`, `~/kids_dr5_work/extract_info.json`, `runs/dr5_measure/{paper_gate.json,PAPER_SELECTION.md,b1_integration.json}`
8. `scripts/dr5_adopt_01.py`, `runs/dr5_measure/adoption/ADOPTION_01.md`, `tests/test_dr5_product.py`, `tests/test_measure_unordered.py`, `tests/test_measure.py`
9. `runs/dr5_measure/angle_slop/ANGLE_SLOP.md`, `runs/dr5_measure/sign_convention/SIGN_CONVENTION.md`, `runs/dr5_measure/sigma_e/SIGMA_E.md`, `plots/dr5_0*.png` (look at them)

## Output
Write findings ranked by severity: BLOCKING (a headline number or claim is wrong), MAJOR (a check cannot fail, a convention is unvalidated, an overstatement),
MINOR (label, provenance, wording). For each: claim number, what you ran (command or script path), the numbers you got vs the stated ones, and confidence.
End with a list of what you did NOT check and why. Do not soften: "no issue found" is only valid with the test that could have found it.
