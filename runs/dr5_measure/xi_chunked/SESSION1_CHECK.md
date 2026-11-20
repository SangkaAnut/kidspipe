# Exact run, session 1: timings and interim check (2026-11-10)

Session 1 ran 2026-11-09 10:49:51 to 2026-11-10 00:29:04 (13.65 h): pairs 5-5 5-6 1-1 1-3 1-5 2-4 2-6, bin_slop 0, angle_slop 0,
16 chunks, 20 threads, float64 positions, one process per pair (`session1.sh`, `session1.log`). All exit 0; no warning or traceback in the log.

## Timings

| pair | s | h | peak RSS GB | ordered pairs binned | pairs/s |
|---|---|---|---|---|---|
| 5-5 | 5614 | 1.56 | 2.26 | 1.877e12 | 3.34e8 |
| 5-6 | 5955 | 1.65 | 4.27 | 1.739e12 | 2.92e8 |
| 1-1 | 7640 | 2.12 | 2.57 | 2.418e12 | 3.16e8 |
| 1-3 | 7780 | 2.16 | 4.87 | 2.211e12 | 2.84e8 |
| 1-5 | 7464 | 2.07 | 4.76 | 2.126e12 | 2.85e8 |
| 2-4 | 7973 | 2.21 | 4.87 | 2.222e12 | 2.79e8 |
| 2-6 | 6726 | 1.87 | 4.56 | 1.941e12 | 2.89e8 |

Model: time = ordered pairs / rate, rate fitted separately for autos (3.25e8/s) and crosses (2.86e8/s); pair counts from the
bin_slop 1.5 product (x2 for autos). Leave-one-out error on session 1: -5.4 % to +5.7 %.
Prediction: S2 (1-2 2-5 3-3 3-5 4-4 4-5 4-6) 13.5 h (13.2-13.8); S3 (1-4 1-6 2-2 2-3 3-4 3-6 6-6) 13.4 h (13.1-13.7).
The sessions are already balanced; no reassignment. Total exact run ~40.5 h.

## Interim check of the 7 pairs (analysis only; nothing in `chunks/` changed)

Integrity (independent audit agent): 1552 files, names equal to `chunk_pairs(16, same)`; every file bin_slop 0, angle_slop 0;
fold 2 exactly for auto a != b; r_nom bitwise equal across files and to the product; no NaN/inf where weight > 0; mtimes inside
each pair's log window (no stale files); the other 14 pair directories absent.

Assembly: `do_assemble` (run with OUTROOT redirected to scratch) equals an independent fold-weighted sum to 0.0 relative.

Exact minus bin_slop 1.5 product, m-corrected, in official sigma (diagonal of the release covariance). 9-bin max for xi;
max over modes for COSEBIs. Computed twice independently (main session and agents); the two agree to the quoted digits.

| pair | xi+ ex-b1.5 | xi- ex-b1.5 | xi+ ex-off | xi- ex-off | xi+ b1.5-off | xi- b1.5-off | E n<=6 | E n<=20 | B n<=6 | B n<=20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5-5 | 0.070 | 0.069 | 0.072 | 0.075 | 0.030 | 0.032 | 0.022 | 0.103 | 0.021 | 0.328 |
| 5-6 | 0.044 | 0.113 | 0.070 | 0.121 | 0.053 | 0.031 | 0.056 | 0.151 | 0.090 | 0.090 |
| 1-1 | 0.076 | 0.088 | 0.073 | 0.078 | 0.010 | 0.010 | 0.075 | 0.260 | 0.156 | 0.530 |
| 1-3 | 0.121 | 0.094 | 0.119 | 0.099 | 0.009 | 0.007 | 0.043 | 0.175 | 0.033 | 0.071 |
| 1-5 | 0.104 | 0.125 | 0.107 | 0.133 | 0.028 | 0.031 | 0.088 | 0.245 | 0.016 | 0.112 |
| 2-4 | 0.075 | 0.113 | 0.081 | 0.118 | 0.014 | 0.012 | 0.056 | 0.431 | 0.039 | 0.143 |
| 2-6 | 0.044 | 0.108 | 0.040 | 0.102 | 0.013 | 0.014 | 0.102 | 0.238 | 0.251 | 0.284 |

Reference (earlier float32 exact Bin6_Bin6): exact vs b1.5 xi+ 0.05, xi- 0.09, E n<=6 0.19, n<=20 0.44; exact vs official
xi+ 0.075, xi- 0.156. The 7 pairs are on the same scale; none stands out. The exact-vs-official difference is set by the
exact-vs-b1.5 shift (b1.5 vs official is 0.007-0.053). Per the PI decision this is reported, not gated.
E_n differences grow with n overall but not strictly monotonically. B_n differences reach 0.53 sigma (1-1, n<=20); the 6-6
reference did not quote B_n.

## Bookkeeping caveat for assembly (not a measurement error)

The chunked driver correlates a catalogue with itself, so auto pairs count ORDERED pairs: assembled `npairs` and `weight` are
2.00001x the unordered bin_slop 1.5 values and `varxi` is exactly 0.5x (each pair enters twice, fully correlated). xi+/xi- are
unaffected (both orders land in the same bin at bin_slop 0; weights cancel in the ratio); crosses agree at 1.000. Before the
exact auto pairs are used for anything that reads npairs, weight or varxi as absolute numbers (e.g. a noise check), halve
npairs/weight and double varxi for autos, or document the convention. Not changed in the driver (no chunk file depends on it).

Scripts and raw outputs: session scratchpad `s1check/` (mine/, audit/, xi/, cosebis/); summary numbers copied to
`session1_check_xi9.json`, `session1_check_cosebis.json`.
