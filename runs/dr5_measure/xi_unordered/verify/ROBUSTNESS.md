# ROBUSTNESS check of the slop 1.5 unordered auto run (b1.5)

Scripts: `scripts/verify_slop15_autos_robustness.py` (independent A_p / boxcar / A3 / const-test recomputation, reads npz + official ascii/FITS directly),
`scripts/verify_slop15_autos_memory.py` (RSS/time on <=300k subsamples, 6 threads). Nothing in src/, existing scripts, RESUME.md or run outputs was touched.
Tests (e): `tests/test_measure_unordered.py` + `tests/test_measure.py`: **64 passed, 0 skipped** (33 s).

## (a) Timings: no proof of contention, but Bin5/Bin6 are genuinely anomalous (UNEXPLAINED)
| pair | N gal | s (ours, 20 thr) | npairs | ns/pair | theirs s | ours/theirs |
|---|---|---|---|---|---|---|
| 1-1 | 7.42M | 692 | 1.209e12 | 0.573 | 696 | 0.99 |
| 2-2 | 7.36M | 700 | 1.183e12 | 0.592 | 678 | 1.03 |
| 3-3 | 6.80M | 632 | 1.013e12 | 0.624 | 660 | 0.96 |
| 4-4 | 6.88M | 712 | 1.048e12 | 0.679 | 679 | 1.05 |
| 5-5 | 6.48M | 850 | 0.938e12 | 0.906 | 625 | 1.36 |
| 6-6 | 5.96M | 986 | 0.807e12 | 1.222 | 611 | 1.61 |
- Time per pair rises monotonically with run position (0.57 -> 1.22 ns/pair, x2.1) while work falls. In theirs the time falls with N as expected. So the work itself does not explain it.
- Contention evidence: load average at 18:52:50 (2 min after the run ended) was 2.4 / 12.98 / 17.32. With 20 threads running 76 min, 15-min avg after ~2.3 min of idle = 20*exp(-2.3/15) = 17.2, i.e. the load of the last 15 min (the whole of Bin6) is explained by our own 20 threads alone; no extra runnable load. No swap used (4 GB swap, 0 used), memory 11 GB free, PSI cpu/mem ~0. dmesg shows no OOM. Cannot exclude: WSL/host (Windows-side) CPU load or throttling, which load average inside WSL does not show; nor 20 of 22 vCPUs being shared with the host.
- Not a process-state leak in TreeCorr: 3 consecutive `process()` calls in one process (300k auto, 6 thr) took 19.0 / 16.5 / 15.3 s (no growth); RSS constant. Subsample Bin6 vs Bin1 vs Bin5 at 300k: 15.3-19.0 s vs 13.3-14.5 s vs 14.3 s (Bin6 maybe ~10-20 % slower per pair, +-20 % run-to-run noise); nowhere near the 2.1x. So the data content (clustering/tree) at most explains a small part.
- Verdict: results are unaffected (npairs/weight match theirs 1.0000000x and gates pass; timing cannot change numbers), but treat run time as variable. Suggest for the cross run: record per-pair `seconds` plus `uptime`/`nproc` load, and put a Bin5/Bin6 pair first to see whether the slowdown follows the bin (data) or elapsed time (host).

## (b) eval.json vs EVAL.md and hand-check
- All 6 table rows of EVAL.md re-generated from eval.json (A1 boxcar max, A3, E_n max, chi2 E_n): 0 mismatches. pass flags all True, `complete: true`, no missing.
- Hand recomputation from the raw npz and the official products (not through the runner's helpers): official ascii `XI_KiDS_Legacy_NScomb_nBins_6_Bin{k}_Bin{k}.ascii` for k=2,5 (also 1,3,4,6), A_p = sum(w o)/sum(w asc) with w = ascii npairs_weighted, boxcar/white from z=(o - A asc)/sigma_asc; A3 from `KiDS_Legacy_xipm.fits` extensions xiP/xiM rows selected by BIN1==BIN2==k (rows 54.. for pair 2, 162.. for pair 5; ANGBIN 9 rows each), sigma from sqrt(diag COVMAT) at pair-major offsets (xip K*9, xim 189+K*9), own-npairs 9-bin rebin and /(1+m)^2.
  - Bin2: A_p 1.147467 / 1.180727, boxcar 0.2953 / 0.2787, A3 0.0251 / 0.0195 -- identical to eval.json to all printed digits.
  - Bin5: A_p 1.167064 / 1.189120, boxcar 0.3468 / 0.3209, A3 0.0401 / 0.0225 -- identical.
  - Bins 1,3,4,6 also identical (A3 0.0163/0.0183, 0.0172/0.0100, 0.0332/0.0377, 0.0458/0.0633). Correct rows used.
- "ours-vs-theirs max rel diff" (8e2..2.8e4) is a bad metric, confirmed: max |a-b|/max(|b|,1e-30) per fine bin blows up where theirs xi crosses zero. The meaningful numbers already in eval.json: max |ours-theirs| / sigma_ascii per fine bin = 0.31-0.52 (xi+ 0.31-0.44, xi- 0.33-0.52, largest Bin6 xi- 0.519), npairs-weighted amplitude ratio ours/theirs 0.998-1.014 (Bin1 xi- 1.014 is the extreme; the others within +-0.3 %), npairs ratio 1+-1.3e-7. Recommend replacing the column with these two.

## (c) A1 one-constant test over 6 pairs
- Code path: `evaluate_a1_const` -> `a1check.a1b_noise` with sd_frac = a1_salvage.json step2_amplitude `{xip,xim}_raw.A_pair_err_frac` indexed by `pair_index` (PAIRS order, autos idx 0,6,11,15,18,20 -- I used the same list). Statistic: chi2 = sum((ln A_p - ln C)/sd)^2 with ln C the inverse-variance weighted mean, dof = n-1 = 5, `chi2.sf`. Exactly as the docstring designs (design text says 21 pairs/dof 20; dof here is correctly 5).
- I reproduced independently: xi+ chi2 1.289, pte 0.936, C 1.1686; xi- chi2 9.000, pte 0.1091, C 1.1817 (identical to eval.json).
- C vs A_p: xi+ A_p = 1.205, 1.147, 1.178, 1.179, 1.167, 1.170 (theirs per salvage 1.201, 1.147, 1.176, 1.180, 1.167, 1.171); weights 1/sd^2 (sd 6.3 %, 2.2 %, 31 %, 1.5 %, 1.2 %, 4.0 %) so C is set by Bins 4,5 (and 2,6): 1.1686 is consistent. xi- A_p = 1.978, 1.181, 1.239, 1.206, 1.189, 1.137 with sd 89 %, 4.7 %, 16 %, 1.3 %, 1.3 %, 1.6 %: C = 1.1817 sits on Bins 4,5. Both close to the 21-pair salvage global C 1.1728 / 1.1830.
- xi- weakness: chi2 contributions 0.33, 0.00, 0.08, 2.36, 0.24, **5.98**; Bin6 xi- A_p 1.137 is 3.8 % below C (2.4 sigma) and drives 2/3 of chi2. Theirs does the same (A_p 1.139, pte 0.130 per BUILD.md), so it is a property of the shared sample/official ascii, not of our measurement. Bin1 xi- A_p = 1.98 (theirs 1.95): xi- of bin 1 is ~0 so the ratio is ill-defined; it is harmless only because sd=0.89.
- Partial-set caveat: stated in both the code label and EVAL.md ("PASS (partial set: 6 of 21 pairs; gate calibrated on 21 pairs / 42 curves)"). Not stated, and should be: with 12 curves instead of 42 the max-boxcar criterion is easier (null 99th pct 1.03 was for the max over 42), and the const test has dof 5 with only ~4 informative pairs. The result is "no contradiction so far" and not a gate pass; the PI gate verdict belongs to the 21-pair set. Margin to thresholds (0.79 vs 1.03; pte 0.109 vs 0.01) is nonetheless comfortable.

## (d) Cross pairs: memory and time
Memory (subsamples, 6 thr, `verify_slop15_autos_memory.py`; RSS after process() minus the 76 MB interpreter baseline):
| N per bin | cross 1x2 | auto |
|---|---|---|
| 100k | +73 MB (peak 238 incl. loading full bin file) | +40 MB |
| 200k | +141 MB | |
| 300k | +209 MB (cats 62 MB, process +147) | +111 MB (cats 42 MB) |
Linear (100k->200k->300k: 0.68 KB per N cross; 0.37 KB per N auto). Extrapolated to full N=7.4M: **cross ~5.1 GB** (cats ~1.5 GB + trees/work ~3.6 GB), auto ~2.7 GB; plus the transient of `load_bin` for the second bin (~0.3-0.6 GB) -> peak ~5.5-6 GB for Bin1 x Bin2 (largest). Machine: 15 GB total, 11-12 GB available, swap 0 used. Fits with ~2x headroom; I could not read the auto run's real peak RSS (not logged); the extrapolation assumes tree nodes ~ linear in N (holds across 100-300k) and subsample density being lower than the full one (node count is bounded by 2N, so the full-density tree should not exceed this by much, but this is not measured). 20 threads add only small per-thread bin buffers. Suggest logging peak RSS (resource.getrusage) in the manifest.
- Runner cross path = `gg.process(cats[a], cats[b])`; cats of both bins are held simultaneously and freed after each pair (`del gg, cats, arr`); smoke tests (BUILD.md, test) cover it. Cross work ratio at 300k: 35.4 s vs 19.0 s auto (npairs 3.94e9 vs 2.04e9, ratio 1.93), consistent with theirs cross ~2.0x auto (1393 vs 696 s).
Time: theirs 15 cross pairs 1176-1393 s, sum 19,780 s (5.5 h). Ours/theirs on autos was ~1.0 for the 4 first pairs, 1.36-1.61 for the last two. Estimate: optimistic 5.5 h (ratio 1.0); pessimistic, if the Bin5/Bin6 slowdown applies to the 9 cross pairs involving bin 5 or 6 (their sum 11,647 s) at x1.5 and the other 6 (8,133 s) at 1.0: ~7.1 h. Range **5.5-7.1 h** for 15 crosses (+1.3 h already spent on autos). If the unexplained slowdown is host-related and gets worse (x2.1 seen at the end), up to ~9 h.

## Problems / recommendations
1. Unexplained Bin5/Bin6 slowdown (x1.4/x1.6 vs theirs, ns/pair x2.1 between first and last pair); not external runnable load inside WSL, not a TreeCorr state leak, small at most from data. Log load/time for crosses.
2. Replace the "max rel diff" column by max |d|/sigma_ascii and the amplitude ratio.
3. State that A1 PASS here is a 6-pair partial verdict with weaker power (12 curves, dof 5); xi- const pte 0.109 is driven by Bin6 (also in theirs).
4. Log peak RSS and load in manifest for the cross run.
