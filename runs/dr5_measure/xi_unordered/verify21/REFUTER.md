# REFUTER record: DR5 xi+- at bin_slop 1.5, 21 pairs

Scope: read scripts/dr5_measure_xi_unordered.py, a1check.py, RESUME.md, both logs, manifest.json, eval.json, EVAL.md, the "theirs" npz
(read-only). Nothing existing modified. Scratch scripts: /tmp/claude-1000/-home-anut/8ecc82c7-22dd-490c-b045-63f93bce2233/scratchpad/pert.py,
pert2.py (run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts /home/anut/anaconda3/bin/python3 <file>`). No TreeCorr on full catalogues.

## 1. Circularity
- Gates A1 (official ascii), A3 (official KiDS_Legacy_xipm.fits + its covariance), B_n/E_n (official COSEBIs FITS + cov) all compare OURS to
  the OFFICIAL release. Nothing there is derived from ours or from theirs. NOT circular.
- Caveat on independence of thresholds: A3 < 0.1 sigma and A1 1.03 were adopted/moved after their xi was seen (RESUME: A3 adopted
  2026-10-29, A1 moved 1.0 -> 1.03 with a null-simulation justification). Baseline margin: A3 max 0.053 vs 0.1.
- m_gold used in A3 is the project's list M (Wright+25 Table 1), not read from the release FITS.
- The "ours vs theirs" comparison is NOT an independent check against the release. "theirs" (HI-gamma-correlation/.../fiducial/*.npz) is
  TreeCorr **5.1.4** (npz field treecorr_version), bin_slop 1.5, same settings, same project, same catalogue extract (RESUME: ra/dec bitwise equal
  to the earlier pipeline's float64 extract). So ours==theirs is a code/data-reproducibility check only; it adds nothing to agreement with
  the release, and the "vs_theirs" gates pass trivially (theirs is pushed through the same evaluate_pair; identical numbers by construction).
  The official release used TreeCorr 4.2.3; neither ours nor theirs tests that difference, and no gate can see it (all gate numbers are
  <= 0.053 sigma anyway).
- Same-extract blind spot: any error in the shared extraction (weights, column choice, m handling) is common to ours and theirs and only
  visible through the official comparison.

## 2. Gate power (perturb the loaded 21 pair arrays in memory, push through M.evaluate_pair / M.evaluate_a1_const)
Command: pert.py, pert2.py. Baseline reproduces the claim exactly: A1 box 21/21 (max 0.787), A3 21/21 (max 0.053), E_n20 max 0.046,
const pte 0.889 / 0.496.

| perturbation (all 21 pairs unless noted) | A1 boxcar pass | A3 pass (max) | A1 const-A_p pte xi+/xi- (pass?) |
|---|---|---|---|
| xi x 1.002 | 21/21 | 21/21 (0.051) | 0.889/0.496 pass |
| xi x 1.005 | 21/21 | 21/21 (0.076) | pass |
| xi x 1.01 | 21/21 | 19/21 (0.125) | pass |
| xi x 1.02 | 21/21 | 8/21 (0.223) | 0.889/0.496 pass |
| xi x 1.05 | 21/21 | 0/21 (0.517) | pass |
| xi x 1.10 | 21/21 | 0/21 (1.04) | pass |
| xi+ only x1.05 / xi- only x0.90 | 21/21 | 0/21 (0.515 / 0.952) | pass |
| tilt 1+0.05 ln(r/24.5) | 21/21 | 0/21 (1.04) | pass |
| xim -> -xim | 21/21 | 0/21 (19.6) | pass (A1 blind to xim sign!) |
| swap xip<->xim | 6/21 | 0/21 | FAIL |
| xi = 0 | 0/21 (nan) | 0/21 | FAIL |
| white noise 1x / 3x / 10x ascii sigma per fine bin | 14 / 11 / 9 of 21 | 0/21 (4.9 / 13 / 49) | FAIL (pte ~0) |
| white noise 0.1x / 0.3x | 21/21 | 0/21 (0.35 / 1.02) | pass |
| only Bin6_Bin6 x1.30 | 21/21 | 20/21 | FAIL (6e-5 / 1e-31) |
| only Bin3_Bin4 x1.10 | 21/21 | 20/21 | FAIL (0.029 / 1.2e-4) |

Findings:
- A1 per-curve boxcar test is amplitude-blind (one scale per pair absorbs it) and sign-of-xim blind; the const-A_p test catches only
  pair-to-pair inconsistency, not a common scale (xi x1.10 on all pairs passes A1 completely; C moves 1.17 -> 1.29 with no gate on C).
  A1 alone would pass a 10 % amplitude error, an xi- sign flip, and 0.3x-sigma white noise. It is a weak gate; the "10 % random subsample"
  case (noise >> ascii sigma) is caught by A1-const and by A3, but not by A1-boxcar alone (9/21 pass).
- A3 is the only strong, non-circular gate: it detects a common amplitude error of ~1 % (fails 2 of 21 at 1 %, 13 of 21 at 2 %), white
  noise, tilt, sign. It does not detect <= 0.5 % amplitude errors. The all-21 A3 claim is genuine but its resolution is ~1 %.
- E_n/B_n have no pass/fail threshold in the code; "B_n null chi2 matches official" is a printed comparison (max |diff| 0.4 in chi2, e.g.
  Bin5_Bin6 14.8 vs 15.2; several 20-mode chi2 are 28-35 for 20 dof, official identical), i.e. ours inherits official's own values.
- The superseded float32-position autos (b1.5_f32, evaluated with the same code) also PASS everything: A1 max 0.793, A3 max 0.063, E_n20 0.120
  (only E_n is larger, ungated). So the gates cannot discriminate f32 from f64 positions; only the ours-vs-theirs comparison can.

## 3. e2 sign
Subsample test (150k random Bin6 galaxies, 5.1e8 pairs, bin_slop 1.5, TreeCorr 5.1.4, 8 threads; command in scratch, inline python):
flipping e2 (or e1) leaves xi+ unchanged (max diff 2.7e-5, tree-approximation level) but changes xi- by up to 0.036 (> the xi- scale 0.021
on this noisy subsample). So xi- (hence A3 on xim) IS sensitive to the e2 sign; my prior expectation that Re xi+- cannot see it was wrong for xi-.
Not quantified on the full catalogue (not allowed); on full data the effect equals the real xi- change and an in-memory xim sign flip fails A3
at 19.6 sigma, so a real flip would very likely fail A3, but this was not measured directly. xi+ cannot see it, A1 is blind to xi-
sign. Sign convention rests on the separate GGL gamma_t evidence, not on this run.

## 4. Mixed float32/float64 products, OOM, resume
- ~/kids_dr5_work/bin0..5.npz: all columns float64 (checked bin0), mtime 2026-11-01 08:11:08-08:11:11, before every output (first npz 08:23).
  f32 files are in f32_backup/ and b1.5_f32/ (separate dirs).
- manifest.json (13 calls, all TreeCorr 5.1.4, 20 threads) lists radec_dtype float64 for every one of 21 pairs; every npz carries
  radec_dtype=float64, bin_slop 1.5, angle_slop_effective 0.1, treecorr_version 5.1.4.
- Log: run_f64_all.log process 2 (started 09:15) skipped the 6 autos, wrote Bin1_Bin2..Bin1_Bin5 (npz mtimes 09:37-10:42, consistent with
  1285-1356 s each), then the kernel OOM-killed it on Bin1_Bin6. Writes are atomic (tmp + os.replace) and gated by pair_file_ok on resume;
  Bin1_Bin6 was re-measured in a fresh process (10:44, 1235 s, RSS 4.7 GB) and the other 10 crosses in per-pair processes. No pair was
  skipped, partially resumed or reused from the f32 run. No mixing found.
- Ours vs theirs, all 21 pairs, computed here: npairs array_equal True for all 21; max |dxip| 1.3e-11..3.3e-11, max |dxim| 1.2e-11..3.8e-11;
  weight rel diff <= 7e-14. (f32 vs f64 ours on autos differ by 3.6e-5..8.2e-5 in xip and npairs not equal, confirming the f32 effect is real.)

## 5. Labels
- EVAL.md header (also the title string in do_eval) says "Unordered auto-pair evaluation" but the file contains all 21 pairs (autos + 15 crosses)
  and the A1 const test over 21 pairs. Misdescription (cosmetic, code line "# Unordered auto-pair evaluation, source=...").
- EVAL.md A1 column reads "PASS (boxcar only)": correct, the const test is on a separate line. The run_f64_all.log name is stale (it only
  holds 4 crosses; the rest are in run_f64_cross.log). The RESUME row for autos still says "15 cross pairs running"; stale.
- eval.json: complete True, missing [], A1_const verdict PASS 0.889 / 0.496: consistent with the claim.

## 6. Is "float32 positions were the only ours-vs-theirs difference" established?
For the shared extract, settings and TreeCorr version: yes. After the fix, all 21 pairs have bitwise-equal npairs and xi equal to ~3e-11
(float64 rounding of e1/e2/w float32-origin values is identical); before, f32 positions moved xi by 6e-5 and npairs. No residual difference to
explain. Limits: (a) it proves equivalence of two pipelines on one common extract, not of the extract to the release; (b) both use TreeCorr 5.1.4,
the release 4.2.3; (c) "ONLY" refers to tested settings (bin_slop 1.5, default angle_slop); and the chunked/exact products remain f32 as noted.

## Discrepancies
1. EVAL.md/eval title says "auto-pair evaluation" for a 21-pair file.
2. "Reproduces the official release" rests on A3 (resolution ~1 %), A1 const, and ungated E_n/B_n; A1-boxcar is amplitude/xi- sign blind; gates
   pass the f32 version too. Claim is true as stated (gates pass) but the gates' power is limited.
3. ours==theirs is not an independent cross-check (theirs = TreeCorr 5.1.4, same extract); do not count it as evidence of release agreement.
4. B_n "matches official" has no numeric criterion in code (differences up to 0.4 chi2).
5. Threshold history (A3 0.1, A1 1.03 set after seeing results) and project-supplied m_gold weaken independence slightly.
6. Stale labels: run_f64_all.log name, RESUME "15 cross pairs running".

VERDICT: CONFIRMED_WITH_CAVEATS
