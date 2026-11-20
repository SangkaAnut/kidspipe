# DR5 measurement — resume note

Updated this session. **The open question is CLOSED: there is no missing selection.**
Nothing long is running. **PI decision 2026-11-02: option (a).** The product is the verified bin_slop 1.5 run (`xi_unordered/b1.5/`, 21 pairs, float64 positions); the exact (0,0) run is NOT part of this step. The PI wants the ~45 h exact full run as an additional part later (stated 2026-11-02); not started, it waits for the PI's go-ahead.

## Headline

The 1.406 pair-count "excess" is **not** a footprint or selection difference. Our file *is* the
published KiDS-Legacy gold sample: the six `Ngold` values in Table 1 of Wright et al. 2025
(A&A 703, A158; arXiv:2503.19441) are digit-for-digit our row counts, and `sum(weight)` matches the
official ascii to 0.3–0.4% on an auto pair, a cross pair and a second auto pair. The 1.406 factor
lives in the **official `npairs` column**, which is not a raw integer pair count — it is continuous
(mean |frac−0.5| = 0.250, uniformly distributed) while ours are exact integers. A1 as specified
compares a raw count to a normalised column and therefore **cannot** pass.

Full evidence and citations: `runs/dr5_measure/PAPER_SELECTION.md`.

## Where the code is
- `src/kidspipe/measure/` (new package, additive — validated theory/likelihood untouched)
  - `official.py`    loaders for the official ascii xi+/-, xipm FITS, COSEBIs FITS, cov blocks
  - `xi.py`          TreeCorr wrapper (Log 1000 bins, 2-300 arcmin) + `load_bin`
  - `paircount.py`   independent cKDTree pair counter
  - `xi2cosebis.py`  binned xi -> E_n/B_n integrator
- `scripts/dr5_*.py` extraction, B1, A4, convention, cut scan, area, diagnostics
- Data cache: `~/kids_dr5_work/bin{0..5}.npz` (tomo bins 1-6, 40,894,394 rows total)
  and `~/kids_dr5_work/extra/` (12 extra columns, separate tree)

## Checks: status
| check | status | evidence |
|---|---|---|
| selection gate | **PASS** | No cut needed: our per-bin counts equal the paper's `Ngold` exactly; n_eff reproduces the paper's `neff,gold` to 0.0–0.4% using the paper's own 967.4 deg2. `runs/dr5_measure/paper_gate.json` |
| binning | PASS | r_nom = geometric centre of 1000 log bins, agreement 5e-13 |
| B1 integration | PASS | chi2/126 = 0.098 (radians, edges weighting); `runs/dr5_measure/b1_integration.json` |
| A4 estimator | PASS | pair counts exact 0.500000 (ordered vs unordered); xi_- r = 0.99999; `runs/dr5_measure/a4_estimator.json` |
| A1 = A1b-noise | **PASS (adopted 2026-10-29, PI sign-off)** | `kidspipe.measure.a1check.a1b_noise`, run by `scripts/dr5_adopt_01.py` (`adoption/adoption.json -> A1_A1b_noise`). Their RAW xi vs the official fine ascii, one scale per pair: max boxcar/white 0.61 (xi+) / 0.79 (xi-) <= 1.03 (threshold moved from 1.0, PI 2026-10-29); one-constant pte 0.89 / 0.50 > 0.01; C = 1.173 / 1.182. The ascii is a **blinded realisation of the same sample** (m-uncorrected, amplitude x~0.85). Retracted: "different sample", "shape problem", "8 sigma internally inconsistent" (official-cov chi2 ascii-vs-FITS is 0.077/0.070). Old corr/npairs-ratio A1 and the name A1a are dropped. CAVEAT: the threshold (1.0 or 1.03) rejects a WHITE residual (white max over 42 curves: median 1.20, 99th pct 1.33); valid for same-galaxy comparisons only. Same-sample null WITH the A_p refit: 99th pct of the max is 1.03, so 1.0 false-failed ~9 % of correct measurements; **1.03 adopted (1 % size, PI 2026-10-29)**; real 0.79 has null pte 0.82 (0.81 retracted). Per-curve standardised option is experimental, not a gate (fails end-to-end size): see `a1_salvage/A1_SALVAGE.md` threshold follow-up. |
| sigma_e gate | **PASS (2026-10-29, verified)** | Joachimi+21 Eq. C.9 per component, `sigma_e = sqrt(sum w^2 |e|^2 / (2 sum w^2)) / (1+m_gold)`, catalogue `weight`, Wright+25 Table-1 m_gold: ratio to `SIG_E_i` (NZ_SOURCE header of KiDS_Legacy_xipm.fits) 1.0000, 0.9999, 0.9995, 0.9997, 0.9994, 0.9993. The old mismatch = wrong estimator (sum w e^2/sum w) and (1+m) multiplied instead of divided. Residual 0.06-0.07 % in bins 5-6 not explained by Table-1 rounding (immaterial, ~0.1 of the m error). `sigma_e/SIGMA_E.md`, `sigma_e/VERIFY.md`, `scripts/dr5_sigma_e.py` |
| A2/A3/B2/B3 | **PASS (from their xi, through our code)** | `scripts/dr5_adopt_01.py` → `adoption/ADOPTION_01.md`. A3 xi vs 9-bin FITS: **gate = max|Δ|/σ < 0.1** (adopted 2026-10-29; chi2/dof<1 vs the official cov is not a gate): max 0.053σ (xi+) / 0.044σ (xi-) PASS; chi2/dof=0.0002 for information. B1 their xi → *our* COSEBIs: En 6m 0.024/126, 20m 0.070/420, max 0.046σ. B3 B_n null reproduces published 0.0635/0.0902 exactly. No 49 h run needed for these. |
| pilot Bin6_Bin6 (OUR xi) | **PARTIAL (2026-10-30, verified)** | `pilot_bin6/PILOT_BIN6.md`, `VERIFY.md`. Pair counting/fold/assembly exact on a real 6M auto pair (npairs = 2 x theirs to 1e-5). A1 PASS (boxcar/white 0.42/0.46 <= 1.03). B_n null reproduces official (chi2 19.5/6 vs 19.2). **A3 FAILS on xi- (0.157 sigma > 0.1, largest bin; xi+ 0.075)** and E_n drift to 0.48 sigma (20 modes); the same code at bin_slop 1.5 passes (0.089). **Not angle_slop:** the full exact run at (bin_slop 0, angle_slop 0) (6999 s) gives A3 xi- 0.1563 / xi+ 0.0748, E_n20 0.484, B_n chi2 19.47/6, i.e. identical to the pilot (max 9-bin difference 6e-4 sigma; `angle_slop/exact_bin6_eval_b0_a0.json`). Our exact xi differs from the official one. The official release used **TreeCorr 4.2.3, bin_slop = 1.5** (Wright+25 Sect. 2.5.1 footnote 1, read by the verifier); slop 1.5 through our code explains 98.9 % of the exact-vs-official E_n gap (R^2 0.989) and 60 % of the failing xi- bin. Whether slop 1.5 is *biased* (vs merely different) is suggestive only (xi- +1.5 %, t~2.3). `angle_slop/ANGLE_SLOP.md`, `angle_slop/verify/VERIFY.md` |
| shear sign convention | **PASS (2026-10-30, verified)** | `sign_convention/SIGN_CONVENTION.md`, `VERIFY.md`. TreeCorr with catalogue e1,e2 as is (no flip_g1/flip_g2). GGL bin1 lenses x bin6 sources: gamma_t > 0, S/N +8.6 (and +15.6 in the independent re-run); flipped variants give ~0 or negative; gamma_x null. Re xi+/- cannot fix the sign (only the parity class). PairCounter needs (e1, -e2). No flip statement exists in the release |
| bin_slop 1.5, all 21 pairs (OUR xi, official style, float64 positions) | **PASS (2026-11-01, verified with caveats)** | `xi_unordered/b1.5/EVAL.md` (autos-only snapshot `EVAL_autos.md`), `xi_unordered/verify21/VERIFY.md`. Logs: `run_f64_all.log` (6 autos + Bin1_Bin2..Bin1_Bin5, then OOM kill) and `run_f64_cross.log` (other 11 crosses, one process per pair, 4.4-5.0 GB peak, 1175-1340 s each). All 21 PASS A3 (max 0.053 sigma, Bin5_Bin6 xi+; next 0.044 Bin6_Bin6 xi-) and A1 boxcar (max 0.787, Bin1_Bin1); A1 const-A_p over 21 pairs / 42 curves pte 0.889 / 0.496 (full-covariance bootstrap variant 0.574 / 0.653) -> **A1 PASS**. E_n20 max 0.046 sigma; B_n null chi2 within ~0.4 of official for every pair (both ungated). Independent recomputation reproduces A3, boxcar, const-A_p, B_n, E_n exactly; wrong pair->row mappings give 1.6-20 sigma, no m-correction 0.12-0.92 sigma. **Ours = theirs:** npairs bitwise 21/21, xi to <= 3.8e-11 (their e1/e2/w float32), confirming float32 positions were the only difference at these settings. Caveats: (1) ours = theirs is code equivalence (theirs is also TreeCorr 5.1.4, bin_slop 1.5, same extract; release used 4.2.3), so release agreement rests on A3 alone; (2) only A3 sees a common amplitude error, resolution ~1 % (xi*1.01 fails 2/21, xi*1.02 13/21, xi*1.005 passes); A1 boxcar and const-A_p are amplitude-blind and xi- sign-blind; (3) the gates cannot tell float32 from float64 positions (float32 run also passed); (4) A1 const uses the precomputed per-pair sd_frac from a1_salvage.json (re-bootstrap agrees to ~1 %); 2 pairs per statistic have sd 0.16-0.89 and carry almost no power; (5) manifest git_head is null (no git repo). Float32 run superseded, kept in `xi_unordered/b1.5_f32/` |
| C, D | not started | optional / gated on A,B |

## Decisions settled this session (with evidence, do not re-litigate)
1. **No selection cut to apply.** The paper's Eq. 29–39 cuts use columns that do not exist in the
   released catalogue (`MASK`, `GAAP_Flag_*`, `contamination_radius`, `fitclass`, `R`), because they
   were applied upstream. Table 14 of 2503.19439 documents them; Table 1 of 2503.19441 proves the
   result. Handoff step 2 ("implement the selection") is void.
2. **Area 1014.0 deg2 is the imaging footprint, not the shear footprint.** Fiducial mask
   `WCS+THELI+AW+NIR` bits 1–14 = 1014.031 deg2 (Table 12). The cosmic-shear analysis then applies an
   astrometric mask (delta_r_int > 0.6", delta_r_exp in [0.20, 0.25]) giving **967.4 deg2 and
   41,409,179 sources**, then `0.1 < zB <= 2.0` giving 40,950,607. The handoff's gate target
   "near 1014 deg2" is wrong by construction; 967.4 is the right number.
3. **`NGAL_i` in the FITS header are the 967.4 deg2 densities rescaled to the 1014.03 deg2 mosaic
   area.** `NGAL_i * 1014.03/967.4` reproduces the paper's `neff,gold` to 3 decimals in all six bins.
4. **Use `npairs_weighted`, not `npairs`, for any pair-count comparison** —
   `KiDS_Legacy_xipm_pipeline.ini:57` sets `column_npair_name = npairs_weighted`.
5. The 1.394->1.440 "drift with separation" in the old OPEN_ISSUE note was an artefact of 20-bin
   rebinning a stored probe. On 1000 bins the ratio is flat: ~1.405–1.415 across 2'–300'.

## Cost warning for whoever resumes
**Estimated (extrapolated from one chunk-pair): ~49 h on one 16-thread process** (verifier best estimate for (0,0): ~45 h at 16 thr, range 37-60; ~38 h at 20 thr), consistent with the original 30-50 h handoff range.
Evidence: Bin1_Bin1 chunk 0-0 = 1.300e11 pairs in 538.8 s = 2.41e8 pairs/s; ordered pairs total
n(n-1) for autos and n_i*n_j for crosses = 9.763e14; fraction inside 300 arcmin = 0.04397
(calibrated from a full 10% run: 2.4183e12 ordered pairs / n1(n1-1)); => 4.29e13 binned pairs.
NOTE: an earlier revision of this note said 85 h — that was a 2x bookkeeping error from dividing by
n^2/2 (unordered) and applying it to ordered counts. The 49 h figure is the corrected one.
`bin_slop` <= 0.05 does not change cost; bin_slop >= 0.5 does (0.5: x0.57, 1: x0.36, 1.5: x0.15-0.23) but changes xi-/E_n (`angle_slop/ANGLE_SLOP.md`). angle_slop = 0 disables the bin_slop saving (0.5:0 == 0:0). Coarse bins do not help.
**Use the chunked driver** — see `CHUNKING.md`.

## Bugs already found and fixed (do not reintroduce)
1. `load_bin()` off-by-one: read tomo bin b+1. Fixed; this is why the headline ratio was
   previously quoted as 1.417 against the wrong official pair (correct value 1.406).
2. `paircount.py` xi_- used `gA * conj(gB)`, whose phases cancel -> recomputed xi_+.
   Now `gA * gB` (net e^{-4i phi}); xi_- then matches TreeCorr at r = 0.99999.
3. TreeCorr needs `sep_units='arcmin'` or every bin is silently empty.
4. `np.allclose` on values ~1e10 is meaningless for testing integrality; compare `floor(x)==x`.
5. A1 salvage: a least-squares slope `sum(t a/s^2)/sum(a^2/s^2)` between two noisy fine-bin curves is diluted
   (gave A~1.03 instead of 1.17); use a ratio of weighted sums. Correlation 0.98 is not a "shape problem".
6. The official-covariance chi2 contains cosmic variance, which cancels between two measurements of the same galaxies, so it
   is nearly blind to smooth amplitude/shape errors (15 % offset -> chi2/dof 0.07; 10 % scale error on A3 -> total chi2 12 over 378 points, chi2/dof 0.032).
   Never use chi2/dof<1 against the official cov as an identity gate; use max|d|/sigma or noise-level statistics.
7. The '3.34 / 8 sigma' ascii-vs-FITS inconsistency quoted in PROMPT_A1_SALVAGE.md does not reproduce (0.0765/0.0699, max 1.19 sigma).
8. A boxcar-rms/white-noise ratio is ~1 for a WHITE residual and its max over 42 curves is ~1.2 (99th pct 1.33); a
   threshold of 1.0 only passes anti-correlated (same-galaxy) residuals. State which regime a noise gate assumes.
9. sigma_e: Eq. C.9 uses w^2 (not w) and DIVIDES by (1+m); the catalogue e1,e2 are m-uncorrected.
10. `xi2cosebis` scheme='trapz' integrated theta T xi in ln(theta) (missing a factor theta, E ~19x small; 1-D input crashed).
    Fixed 2026-10-29; not on the B1 path ('edges'). trapz keeps an O(dln theta) end-truncation error (2 % at n=4, 1000 bins).
11. `official.weighted_rebin` theta for 2-D input divided an all-rows sum by one row's weights (xi was right). Fixed.
12. Chunk assembly of AUTO pairs: Im xi_+ cancels over ordered pairs but the 2x fold of a<b chunk-pairs doubles it, so
    assembled `xip_im` was spurious. Fixed: zero for auto pairs (exact). xip, xim, xim_im were always correct.
    Regression tests for 9-12 and for bugs 1-4 + the fold factor: `tests/test_measure.py` (mutation-checked, `tests_measure/`).
13. **bin_slop=0 with default angle_slop is not brute-force exact for xi- in TreeCorr 5 (immaterial at production binning, see correction).** `angle_slop` (default 0.1 rad) approximates the projection angle;
    xi- carries e^{-4i phi}. 80 deg2 bin-6 patch: angle_slop 0.1 vs 0 changes xi- by up to 0.14 sigma (~2.4 % at 60-300'),
    xi+ by 0; with 0.1, an RA rotation changes xi- by 0.11 sigma, with 0 it is invariant. The 14x cost seen there was
    an artefact of 20 WIDE bins: at the production 1000 bins, (bin_slop 0, angle_slop 0) costs the same as (0, default)
    (patch A, 690k x 690k: 538.5 s vs 540.7 s) and equals brute force to 1e-14 (`angle_slop/brute_check.json`).
    `pilot_bin6/angle_slop/`. Pilot, theirs and slop1.5 used the default; the full-pair (0,0) run exists (`pilot_bin6/b0_a0_chunks/`).
    **Correction 2026-10-30:** on the 80 deg2 / 20-bin test angle_slop mattered, but at the production 1000 bins it does
    not: full Bin6_Bin6 (0,0) vs (0,default) differ by <= 6e-4 sigma. angle_slop does NOT explain the pilot's A3 xi-
    failure (`angle_slop/ANGLE_SLOP.md`); bin_slop does (full pair, 1.5 vs exact: xi- 0.093, E_n20 0.437 sigma).
14. Assembled `varxi` of ORDERED auto pairs is half the true variance (use 2*varxi). sum(weight)/official npairs_weighted
    = 72.08 with a ~3 % drift in theta (unexplained; irrelevant to xi).

15. **Catalogue positions stored as float32** (`scripts/dr5_extract.py:100`, `ra=r.astype(np.float32)`): max 0.055 arcsec
    position error. Moves every fine bin by rms 0.10-0.15 sigma (zero-mean, adjacent bins anticorrelated), the same at
    bin_slop 0 and 1.5; 9-bin up to 0.04 official sigma (Bin6 A3 xi- 0.063 ours vs 0.044 theirs with float64). Affects ALL
    our xi so far (pilot, exact, slop15, autos). e1/e2/w float32 are identical to theirs. Fix = float64 ra/dec + re-extract
    FIXED 2026-11-01 (PI-approved): `dr5_extract.py` keeps float64 for all columns; float32 files kept in
    `~/kids_dr5_work/f32_backup/`; integrity check `scripts/dr5_extract_check_f64.py` ALL_OK (ra/dec bitwise = the earlier
    pipeline's float64 extract). `xi_unordered/run_f64_all.sh` (6 autos then 15 crosses at bin_slop 1.5, log
    `xi_unordered/run_f64_all.log`): float64 autos now equal theirs (npairs bitwise, xi to 4e-11), confirming float32
    positions were the ONLY ours-vs-theirs difference; crosses: the single-process run was OOM-killed on Bin1_Bin6 (14.6 GB RSS on a 16 GB box; a fresh cross pair needs ~4.6 GB, so ~2.5 GB accumulated per pair). Bin1_Bin2..Bin1_Bin5 were saved intact (npairs bitwise = theirs). Resumed with `xi_unordered/run_f64_cross.sh` (one process per pair, peak RSS logged; log `run_f64_cross.log`). Use one process per pair for any multi-pair run on this machine. Old float32 autos in `xi_unordered/b1.5_f32/`. Chunked pilot/exact chunk files are
    float32-based and must be recomputed for any exact production run. All 21 float64 pairs done and verified 2026-11-01 (see the gate row).
16. **`scripts/dr5_adopt_01.py` B1 B_n-vs-official uses the E_n covariance block** (no +420 offset, same mistake its B3 section documents). The B_n numbers in that section of `adoption/adoption.json` (0.0169/126, 0.0644/420) are invalid; correct values on ours: 0.0223/126, 0.0780/420, max 0.0437 and 0.0602 (`product/FULLVECTOR.md`). A3, E_n and B3 entries are unaffected. Script and json NOT edited yet.
17. **`scripts/dr5_measure_xi_chunked.py` `do_assemble` copies `meanr` from the FIRST chunk file** (it sets `meanr` once when the accumulator is created and never accumulates it). The assembled files' `meanr` is therefore chunk c00_00's (off by up to 2.3e-5 relative from the full-pair value). Found 2026-11-11 by the review of the exact product; no measured number used it (COSEBIs use `r_nom`). The exact product recomputes `meanr` from the chunk files (weighted mean; agrees with the bin_slop 1.5 run's meanr to 1.2e-6). Driver NOT edited (fix: accumulate fold*w*meanr and divide by the summed weight).

See also skill `kids-shear-two-point-measurement` for the full trap list.

## Not yet done
- **Production: DONE by PI decision (a), 2026-11-02.** Product = `xi_unordered/b1.5/` at (bin_slop 1.5, angle_slop default 0.1). The chunked driver (`CHUNKING.md`) and its exact chunk files stay available, but those files are float32-based and must be recomputed before any exact use. Never launch `angle_slop/queue4.sh`.
- **Exact full run (PI's additional part), split into 3 sessions (PI, 2026-11-09).** Estimate 40-55 h total at 20 threads (cost model ~38 h; Bin6-anchored ~55 h; no exact cross pair timed before). Sessions balanced to 1/3 each by the bin_slop 1.5 timings: S1 = 5-5 5-6 1-1 1-3 1-5 2-4 2-6; S2 = 1-2 2-5 3-3 3-5 4-4 4-5 4-6; S3 = 1-4 1-6 2-2 2-3 3-4 3-6 6-6 (rebalance S2/S3 from S1 timings). **Session 1 STARTED 2026-11-09** (PI go-ahead): `xi_chunked/session1.sh`, log `session1.log`, bin_slop 0, angle_slop 0, 16 chunks, 20 thr, one process per pair; `touch xi_chunked/STOP` stops it cleanly after the current pair; killing it loses at most one chunk-pair. The float32 chunk files were moved to `xi_chunked/f32_archive_20261030/` before the start. Prerequisites as checked:
  1. DONE 2026-11-09: the float32-era chunk files (Bin6_Bin6 136, Bin1_Bin1 1), `assembled/` and `manifest.json` moved to `xi_chunked/f32_archive_20261030/`; the driver's `load_bin` reads the float64 extract (checked).
  2. Memory: the driver caches loaded bins across pairs in one process, so each pair runs in its own process (session script); peak RSS logged per pair.
  3. Gates, DECIDED by the PI 2026-11-02: A3 at 0.1 does NOT apply to the exact run, because the release itself used bin_slop 1.5 (the float32 exact Bin6_Bin6 gave A3 xi- 0.156 sigma, mostly from bin_slop). The exact run is judged by comparison with the bin_slop 1.5 product: report the difference per pair, no pass/fail. A3 at 0.1 stays the gate for the bin_slop 1.5 validation/product.
  4. **Session 1 DONE 2026-11-10 00:29** (13.65 h, 7/7 pairs, exit 0, peak RSS 4.9 GB). 1552/4656 chunk-pairs done. Per-pair 1.56-2.21 h. Interim check in `xi_chunked/SESSION1_CHECK.md`: integrity clean; exact-minus-b1.5 at 9 bins 0.044-0.125 sigma_official, E_n n<=20 up to 0.43, B_n n<=20 up to 0.53 (1-1), same scale as the Bin6_Bin6 pilot. Measured model predicts S2 13.5 h, S3 13.4 h (no reassignment needed; total ~40.5 h). **S2 and S3 NOT started: wait for the PI's go-ahead.** For S2, copy `session1.sh` to `session2.sh` with the S2 pair list and SESSION2 labels.
  6. **Session 2 STARTED 2026-11-10 00:44** (PI go-ahead; `xi_chunked/session2.sh`, log `session2.log`). Done before a WSL restart at ~08:50: 1-2 (7950 s), 2-5 (6687 s), 3-3 (5772 s), 3-5 (6728 s), all exit 0; 4-4 killed at 58/136 chunk-pairs (rows 0-3 complete; all 962 session-2 files load cleanly with exact settings, no partial file). **Resumed 08:56** with `session2_resume.sh` (4-4 4-5 4-6), log `session2_resume.log`; finished chunk-pairs are skipped. **Session 2 DONE 2026-11-10 13:37:56** (compute 12.78 h; 14/21 pairs, 3104/4656 chunk-pairs). Check in `xi_chunked/SESSION2_CHECK.md`: integrity clean (incl. the split 4-4); xi exact-minus-b1.5 0.049-0.125; E_n n<=20 up to 0.61 (3-3), 0.58 (4-5), driven by the outermost 5 fine bins (290-300'), where the COSEBI filters amplify bin_slop 1.5 re-binning noise. S3 predicted 13.0 h.
  7. **Session 3 DONE 2026-11-11 08:48:42** (started 2026-11-10 19:55:18; 12.89 h; `session3.log`, `xi_chunked/SESSION3_CHECK.md`). **Exact run COMPLETE: 4656/4656 chunk-pairs, 39.3 h of computing.** Assembled 08:51 (`xi_chunked/assembled/ns/`, raw ordered convention for autos).
  8. **Exact product DONE 2026-11-11:** `product_exact/dr5_xipm_cosebis_exact.npz` (+ README, `EXACT_VS_B1P5.md`, `exact_vs_b1p5.json`), builder `scripts/dr5_make_exact_product.py` (autos converted to unordered; meanr recomputed from chunks, bug 17; validated before publishing, no gate; step D; 21-pair angular decomposition of the COSEBIs difference), tests `tests/test_dr5_exact_product.py` (10; adversarial review 2026-11-11, 11 findings fixed; full suite 391 passed, 0 failed, `product_exact/pytest_full_20261111.log`), figure `plots/dr5_exact_vs_b1p5.png` (`scripts/dr5_plot_exact_vs_b1p5.py`); value plots added 2026-11-19: `plots/dr5_08_xipm_exact_vs_official.png`, `dr5_09_En_exact.png`, `dr5_09b_Bn_exact.png` (`scripts/dr5_plot_exact_values.py`); report section 5.11 of `docs/DR5_MEASUREMENT_REPORT.md`. Every comparison number recomputed independently from the chunk files (agreement <= 2e-14). Headline: exact-b1.5 9-bin xi+ 0.044-0.141, xi- 0.040-0.148; joint chi2 1.06/378; E_n 1-20 up to 0.61 (mostly the outermost 5 fine bins); step D chi2 E_n 1-6: official 127.234, b1.5 127.516, exact 127.709. Not gated (PI 2026-11-02).
  9. **MPI support DONE 2026-11-20** (user request; mpi4py 4.1.2 installed with approval, Open MPI 5.0.10; target = this laptop). `src/kidspipe/mpi.py` (mpi4py.futures; rank 0 runs, ranks 1..N-1 work, N >= 3 for the CLI); `--mpi` on the CLI samplers (map/mcmc/nested/bench), `scripts/dr5_measure_xi_chunked.py` (one tomographic pair at a time, heaviest chunk-pairs first, workers keep only the current pair's bins, atomic chunk writes `.cAA_BB.npz.part` then rename; new `--outroot`), `scripts/legacy_validate.py eval`, `scripts/run_kv450_reweight.py`. Guards: mpirun without `--mpi` refused; `--status/--assemble` with `--mpi` refused. Equivalence checked: MAP (seed 7, 2 workers) and nautilus (nlive 60, 400 calls, seed 3) bit-identical to the process pool; DR5 Bin6_Bin6 6 chunk-pairs recomputed under `mpirun -n 3` (scratch `--outroot`) match the stored exact files (npairs exact, xi/weight <= 3e-10 rel = thread summation order; xip_im of the auto pair is round-off ~1e-18 in both). Adversarial review (sonnet high): 1 confirmed bug fixed (a root-side error waited for the whole task queue: now queued tasks are cancelled; old 20 s vs new < 10 s in `tests/test_mpi.py`), plus worker-initializer tracebacks printed and reweight variants not accumulated. No speed-up on one machine (MAP 229 vs 238 s, nautilus 380 vs 375 s): MPI is for multi-node use. On 16 GB use <= 2 DR5 MPI workers. Usage: `docs/USAGE.md` "## MPI"; DR5 check evidence `xi_chunked/mpi_check_20261120/MPI_CHECK.md`; report section 5.12. Full suite 400 passed, 0 failed. Backups of the three edited scripts in the 2026-11-20 session scratchpad (`*.bak_20261120`).
  5. Assembly caveat: exact auto pairs are ORDERED counts (npairs, weight 2x; varxi 0.5x the unordered values); xi unaffected. Halve/double for autos before any use of absolute npairs/weight/varxi (see `xi_chunked/SESSION1_CHECK.md`).
- Test failures of 2026-11-02 FIXED 2026-11-09 (PI approved): `test_hankel.py` timing now a relative check (fast >= 10x FFT, one BLAS thread, best of 5 blocks; ratio >= 34 over 21 runs incl. 2x oversubscribed); `test_emulator.py` sigma8 compared at rtol 1e-12 (CAMB nthreads=4 leaves MKL at 4 threads in the main thread, so worker-thread sigma8 differs by ~2e-14). Mutation checks: FFT fallback fails the timing test; shared last_derived fails the thread test 3/3. Full suite 381 passed, 0 failed (`product/pytest_full_20261109.log`). Originals backed up in the session scratchpad `test_backup_20261109/`.

## Done 2026-11-02 (spec deliverables)
- Product and full-vector checks: `product/` (`dr5_xipm_cosebis_binslop1p5.npz`, `README.md`, `FULLVECTOR.md`); scripts `dr5_make_product.py`, tests `tests/test_dr5_product.py`.
- Plots `plots/dr5_01`..`dr5_06`, `dr5_03b`, `dr5_autos_b1p5` (scripts `dr5_make_plots.py`, `dr5_plot_autos.py`).
- `docs/DR5_MEASUREMENT_REPORT.md`, `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md`. Audits `product/AUDIT_NUMBERS.md`, `product/AUDIT_REFEREE.md` describe the draft; findings are corrected in the documents. Step C, N/S patches and xi_x in the product remain not done.
