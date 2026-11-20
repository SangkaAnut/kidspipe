# southpatch verification (new Southern patch, Bin6xBin6)

Script: scripts/verify_angle_slop_southpatch.py (run / analyze / extra). Outputs: verify/southpatch/{S_Bin6_Bin6_b*_a*.npz, analysis.json, extra.json, run.log}.
Patch: ra 10-24, dec -36..-27 (KiDS-S), 571,938 Bin6 galaxies (study: A 690k, B 782k, S 101k). Production binning, run_pair(c, c) (ordered auto), 8 threads.
Conversion = study's own (`st.metrics`, `st.fine_scale`): per-fine-bin patch deviation x (external full-pair TreeCorr sigma / patch sigma*sqrt2), rebinned to 9 bins, / m-factor / official sigma. My own E_n recomputation via cosebis_E agrees (En_own_max20 == E20).

## Results (max |dev|/sigma_official, deviation from (0,0) on the same patch)
| setting | s (8 thr) | xi+ | xi- | E6 | E20 | note |
|---|---:|---:|---:|---:|---:|---|
| 0:0 | 756 | ref | ref | | | |
| 0:0.1 | 761 | 0.000 | 0.001 | 0.001 | 0.001 | max fine-bin abs dxi- 1.3e-7 |
| 0.5:0 | 827 | 1e-11 | 1e-11 | 3e-12 | 3e-12 | identical to (0,0): npairs identical, max abs dxi 3e-17 |
| 1.5:0.1 | 231 | 0.065 | 0.102 | 0.132 | 0.242 | |

Runtimes: the 1.5:0.1 run overlapped heavy load (load avg 14-27 from other verifiers); the other three ran at load ~8 (mostly mine). Ratio 0:0 : 0:0.1 : 1.5:0.1 = 1 : 1.01 : 0.31 (study patch A 1 : 1.00 : 0.23; mine is contended so the 1.5:0.1 ratio is not reliable).

## Findings
1. C1 cost/equality part reproduced: (0,0.1) = (0,0) to 1e-7 in xi-, cost identical (761 vs 756 s). Brute-force equality not re-tested by me.
2. (0.5, 0) is NOT a viable cheaper setting. angle_slop=0 makes bin_slop irrelevant: result identical to exact, cost the same (827 s vs 756 s, +9%, within contention noise). The study's own S-patch grid already shows this (grid/S_..._b0.5_a0 33.9 s, b1_a0 33.7, b1.5_a0 33.9 vs b0_a0 33.5), but ANGLE_SLOP.md never states it. Deviation is trivially 0 (<0.1), but there is no saving. Any saving needs angle_slop>0.
3. (1.5,0.1) on this patch is OUTSIDE the study's patch range: E6 0.132 (study patches A 0.034, B 0.039, S 0.078; table worst 0.078), E20 0.242 (0.108/0.094/0.191), xi- 0.102 (A 0.038, B 0.081, S 0.034; table worst 0.106 incl. pair 1-6), xi+ 0.065 (0.058/0.053/0.041). Full pair: 0.051/0.093/0.194/0.437. Full/south ratio: xi+ 0.8, xi- 0.9, E6 1.5, E20 1.8 (vs 2-2.5x for study patches). C5 (patch forecasts are not upper bounds for E_n) is supported; the south patch is the largest patch seen yet still under-predicts E_n by 1.5-1.8x.
4. E_n structure: south En(1.5:0.1) is smooth and one-signed: 0 sign changes, lag-1 autocorrelation 0.98, rising 0.021 (n=1) -> 0.218 (n=10) -> 0.242 (n=14), then plateau ~0.2. Full pair: 0.028 -> 0.44. South/full ratio per mode 0.75..0.46, profile correlation 0.93, same sign. Study patch S also has corr 0.95 with full. But study patches A and B have the OPPOSITE sign in the first modes (first-6 ratio to full -0.1..-0.4; B overall corr 0.08). So the deviation is coherent across n (smooth, low-dimensional) but amplitude AND SIGN vary between patches: not white noise in n, not a fixed bias either. With 5 realisations (A, B, S, South, full) one cannot say whether the full-pair 0.44 is typical.
5. xi 9-bin: patch pattern correlates weakly with full pair (xi+ corr 0.26, xi- 0.61); rms south/full 1.35 and 1.20, i.e. xi deviations are consistent with the conversion. Largest south xi- terms are bins 7 (-0.102) and 8 (+0.097) (60-300'), adjacent bins of opposite sign; full pair has -0.071 and +0.093 in the same bins.
6. Soundness of conversion (my judgement): consistent for noise-like deviations (deviation ~ shape noise/sqrt(Npairs)). South scale factor 0.335 at small scales to 0.313 at 150-300' (area-based sqrt(109/1347)=0.28, plausible). Caveats: (a) it divides by the OFFICIAL sigma (includes cosmic variance) while scaling with the shape-noise-only external sigma; I computed shape-noise-only/official 9-bin sigma = 1.1-1.2 at small scales falling to 0.5 (xi+) / 0.76 (xi-) at 150-300'; fine for conversion but not a like-for-like ratio. (b) any signal-proportional component does not shrink with area; the study checks this only on a synthetic field. (c) E_n are linear functionals of the fine-bin deviation vector; scaling each fine bin by an independent-noise ratio is valid only if fine-bin deviations are independent noise, but they are smooth/anticorrelated between neighbours, which may explain the systematic under-prediction. Not verified further.
7. 0.5:0 viability: accuracy yes (exactly 0), cost saving none, so not a useful candidate. The meaningful unmeasured mid-cost candidate is 0.5:0.1 (not run here; study A 0.082/0.254 E6/E20 vs S 0.037/0.04 shows a 6x patch-to-patch spread).

## Not checked
No south signal-only run; no full-survey hours extrapolation from my contended timings; B_n: south 1.5:0.1 max |Bn|/sigma 0.14 (scaled, noise-like).
