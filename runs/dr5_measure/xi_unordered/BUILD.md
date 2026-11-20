# Unordered auto-pair measurement: build and regression (script `scripts/dr5_measure_xi_unordered.py`)

Subcommands: `run` (official-style unordered TreeCorr, whole catalogue, bin_slop default 1.5, resumable, manifest.json) and
`eval` (A1, A3, B1, B_n null, ours-vs-theirs; writes eval.json + EVAL.md; `--source theirs` previews from their xi).
Evaluation functions: `evaluate_pair(pair, ours, ordered, ctx, theirs)`, scale invariant in npairs/weight.
m-correction: (1+m_i)(1+m_j), m from `scripts/dr5_adopt_01.py` (Wright+25 Table 1). FITS row order verified pair-major (asserted in `load_context`).

## (1)+(2) Bin6_Bin6 reproduction (evaluator, ordered=True for assembled files)
| input | A1 boxcar/white xi+ / xi- | A3 xi+ / xi- | E_n max (6m / 20m) | B_n null chi2 6 / 20 |
|---|---|---|---|---|
| slop15 assembled (ordered) | 0.4072 / 0.4441 | 0.0456 / 0.0888 | 0.071 / 0.071 | 19.14 / 31.74 |
| theirs | 0.4517 / 0.4533 | 0.0238 / 0.0438 | 0.041 / 0.043 | 19.1 / 32.4 |
| b0_a0 assembled (ordered) | 0.4209 / 0.4566 | 0.0748 / 0.1563 | 0.238 / 0.484 | 19.47 / 32.23 |
All agree with the known numbers to < 0.001 (identical to 1e-12 vs exact_bin6_eval_b0_a0.json).
slop15 vs theirs: npairs ratio (ordered/2) 0.9999998, weighted amplitude ratio 0.9993 (xi+) / 0.9984 (xi-).

## (3) Theirs through the evaluator, all 6 autos (`theirs_eval/EVAL.md`)
| pair | A1 max boxcar/white | A3 xi+ / xi- | E_n max 6m / 20m | B_n chi2 6 (off) / 20 (off) |
|---|---|---|---|---|
| 1-1 | 0.787 | 0.0095 / 0.0097 | 0.007 / 0.007 | 7.9 (7.9) / 26.5 (26.4) |
| 2-2 | 0.306 | 0.0070 / 0.0105 | 0.009 / 0.009 | 5.7 (5.7) / 18.1 (18.1) |
| 3-3 | 0.358 | 0.0079 / 0.0140 | 0.007 / 0.015 | 7.7 (7.7) / 28.7 (28.6) |
| 4-4 | 0.322 | 0.0302 / 0.0306 | 0.020 / 0.034 | 3.3 (3.3) / 15.9 (15.7) |
| 5-5 | 0.354 | 0.0300 / 0.0315 | 0.023 / 0.046 | 6.4 (6.6) / 17.8 (18.1) |
| 6-6 | 0.453 | 0.0238 / 0.0438 | 0.041 / 0.043 | 19.1 (19.2) / 32.4 (32.6) |
All PASS (A1 <= 1.03, A3 < 0.1).

## (4) Runner smoke, 50k-galaxy random subsample (`b1.5_sub50000/`, --subsample, seed per bin), 8 threads
Compared with a direct `treecorr.GGCorrelation(kwargs).process(cat)`: Bin6_Bin6 npairs exactly equal, max|dxip|,|dxim| = 2.4e-17;
cross Bin1_Bin6 (process(cat1, cat2)) npairs exactly equal, max|dxi| = 1.9e-17 (thread summation order, not bitwise; < 1e-12).
Runtime 0.3 s / 0.6 s. Output carries seconds, treecorr_version, pairing="unordered", bin_slop, slop_record.

## (5) Tests
`tests/test_measure_unordered.py` (5 tests: pair parsing/index, unordered = ordered/2, runner vs direct TreeCorr + resume + manifest,
block selection vs FITS BIN1/BIN2 labels, eval scale invariance/ordered normalisation) + `tests/test_measure.py`: 62 passed.

## Review fixes
Each finding reproduced by reading/probing, then fixed in `scripts/dr5_measure_xi_unordered.py` (src and existing scripts untouched).
- MAJOR non-atomic npz / resume trusts any file (real): npz now written to `<name>.tmp.npz` then `os.replace`; resume validates the file (`pair_file_ok`: loads, required keys non-empty) and re-measures an unreadable one. Test `test_truncated_pair_file_is_remeasured`.
- MAJOR manifest only at end / thin provenance (real): `manifest.json` rewritten atomically after every pair. Per pair: seconds, npairs, weight, threads, treecorr version, nobj1/2, catalogue file/size/mtime. Top level: argv/threads/treecorr/start/cpu_count appended per call (`calls` list, no longer overwritten), git HEAD (null here: repo is not under git), effective angle_slop (TreeCorr default prints None = library default). A finished pair missing from the manifest is rebuilt from its npz on resume. Startup prints threads/cpu_count. Recommend the lead also pipes to `| tee run.log`.
- MAJOR eval silently skips missing pairs (real): WARNING per missing pair; verdict printed as INCOMPLETE if any requested pair is missing; eval.json carries requested/missing/complete; EVAL.md gets an INCOMPLETE line; exit non-zero with no files written when no pair is found (the output dir is no longer created before this check, so a wrong `--subsample` / `--bin-slop` cannot clobber or litter anything).
- MINOR A1 overall is a partial-set verdict (real): label now "A1 overall <verdict> (partial set: N of 21 pairs; gate calibrated on 21 pairs / 42 curves)". Thresholds unchanged.
- MINOR per-pair A1 column (real): renamed "A1 boxcar ... (boxcar only)", const-A_p test stated as reported below the table.
- MINOR mutation test gap (real): `m/mfac` -> `m*mfac` now fails `test_theirs_bin6_pins_known_gate_values` (A3 0.0238/0.0438, E_n20 0.043, A1 0.453, abs 1e-3; release-gated). Confirmed by mutating a scratch edit then restoring.
- MINOR duplicates / threads (real, cheap): `parse_pairs` dedupes. Default `--threads 20` kept (production default per brief; cpu_count printed at start).
- MINOR json default=float (latent): replaced by `_jdefault` handling np.bool_ and arrays.
- Kwargs-confirmation finding: not a defect, no action.
- Noted, not changed: `load_ours` infers ordered only from a `pairing`/`fold` key; the pilot slop15 assembled file has neither, so `eval` auto-detect reports ordered=False for it (affects only the npairs-vs-theirs ratio; gates are scale invariant). Callers of `evaluate_pair` pass `ordered=True` explicitly, as in regression (1).

Re-run: (1) slop15 assembled via evaluator: A1 0.407/0.444, A3 0.0456/0.0888, E_n20 0.071 (unchanged). (3) theirs all 6 autos: identical table to above, all PASS; const-A_p pte 0.933/0.130. (4) 50k subsample runner, Bin6_Bin6 + Bin1_Bin6 (8 threads): 0.3 s / 0.6 s, manifest populated, resume skips; direct-TreeCorr equality is covered by the test. (5) tests: 64 passed (7 new-file + 57).
