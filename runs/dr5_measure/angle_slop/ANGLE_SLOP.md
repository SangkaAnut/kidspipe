# TreeCorr (bin_slop, angle_slop) study for the DR5 production run

Setting names are `bin_slop:angle_slop`; `0.1` is TreeCorr 5.1.4's default angle_slop (what the driver uses when `--angle-slop` is not given).
Production binning (1000 log bins 2-300', `ordered` process(cat,cat)), real KiDS-Legacy DR5 Bin6 / Bin1 catalogues, patches A (ra 165-185, 690k Bin6), B (ra 195-215, 782k, disjoint), S (24 deg2).
Code: `scripts/dr5_angle_slop_study.py` (run/brute/analyze), `scripts/dr5_angle_slop_report.py`, `scripts/dr5_angle_slop_fullpair.py`. Data: `grid/*.npz`, `study_grid.json`, `study.json`, `fullpair_vs_exact.json`. Plots: `plots/dr5_angle_slop_bias_vs_hours.png`, `plots/dr5_angle_slop_bias_per_bin.png`.

## Headline

1. **angle_slop is irrelevant at production binning.** (0,0) and (0,0.1) cost the same (patch A, 690k x 690k, 16 threads, uncontended: 538.5 s vs 540.7 s) and on the FULL Bin6_Bin6 pair differ by 0.0006 sigma_official (xi-), 0.00003 (xi+), 0.001 (E_n, 20 modes) (`exact_bin6_eval_b0_a0.json` -> `new_vs_pilot`, `fullpair_vs_exact.json`). (0,0) equals TreeCorr brute force to 5e-14 (`brute_check.json`); (0,0.1) differs from brute by up to 0.0035 sigma per fine bin.
2. **This contradicts bug 13's attribution.** The A3 xi- failure of the pilot (0.157 sigma) is NOT caused by angle_slop: the exact (0,0) full-pair measurement gives A3 xi- 0.1563 (pilot 0.1568), xi+ 0.0748, E_n max 0.484 (20 modes) / 0.238 (6). The failure is the difference between an exact measurement and the official/external ones, which carry the bin_slop approximation (slop 1.5 through our code: 0.089 / E_n 0.071; external: 0.044 / 0.043). What bin_slop the official release used is NOT established here.
3. **bin_slop > 0 is where the cost saving and the error come from.** Measured on the full Bin6_Bin6 pair, (1.5, 0.1) vs exact: xi+ 0.051, xi- 0.093, E_n 0.194 (6 modes), 0.437 (20 modes) sigma_official; the E_n deviation is a smooth monotonic drift (0.03 at n=1 to 0.44 at n=20).
4. **The patch-based forecasts under-predict that measured realisation by ~2-2.5x for E_n** (1.5:0.1: patch worst 0.078 / 0.191 vs measured 0.194 / 0.437; xi- patch 0.106 worst (pair 1-6) vs 0.093 measured). So the patch columns below are not upper bounds. Only settings with a full-pair measurement (last-but-one column) are established; the rest are indicative.

## Decision table

Hours = 49 h x runtime ratio vs `0:0.1` (patch A uncontended ratios; `*` = small-patch S ratio only). Bias columns: worst over patches/pairs of the "noise-like" deviation (setting minus exact on a real patch, rescaled bin by bin to the official full-survey sigma) and the coherent "signal" bias on a noiseless synthetic field (xi- signal-only column), in official sigma. Worst includes pairs 6-6 (A,B,S) and 1-6 (A); several settings were run on A only (B / 1-6 were limited to 0:0, 1.5:0.1, 1:0.1, 1.5:0.003 for B and 0:0, 0:0.1, 1:0.1, 1.5:0.1 for 1-6). "stable A/B" = xi- 9-bin noise-like agrees between A and B within a factor 2 (+0.02); the E_n columns are NOT stable between patches (e.g. 1.5:0.1: A 0.108, B 0.094, S 0.191, 1-6 0.037).

| setting | hours | xi+ 9-bin | xi- 9-bin | E_n (6) | E_n (20) | signal-only xi- | stable A/B | MEASURED full pair (xi+ / xi- / E6 / E20) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0:0 | 48.8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | yes | not measured | exact reference (= brute force to 1e-14) |
| 0:0.1 | 49.0 | 0.000 | 0.002 | 0.002 | 0.002 | 0.000 | n/a | 0.000 / 0.001 / 0.000 / 0.001 | negligible (<=0.03) |
| 0.5:0.1 | 27.8 | 0.026 | 0.021 | 0.082 | 0.254 | 0.014 | n/a | not measured | too large (>0.1) |
| 1:0.1 | 17.7 | 0.068 | 0.070 | 0.092 | 0.260 | 0.034 | yes | not measured | too large (>0.1) |
| 1:0.003 | 29.3* | 0.022 | 0.024 | 0.017 | 0.032 | 0.005 | n/a | not measured | acceptable (<=0.1) |
| 1.5:0.1 | 11.2 | 0.058 | 0.106 | 0.078 | 0.191 | 0.052 | yes | 0.051 / 0.093 / 0.194 / 0.437 | too large (>0.1) |
| 1.5:0.01 | 11.2 | 0.058 | 0.038 | 0.034 | 0.108 | n/a | n/a | not measured | too large (>0.1) |
| 1.5:0.005 | 17.4 | 0.066 | 0.056 | 0.102 | 0.272 | n/a | n/a | not measured | too large (>0.1) |
| 1.5:0.003 | 26.2 | 0.026 | 0.035 | 0.019 | 0.152 | 0.009 | yes | not measured | too large (>0.1) |
| 1.5:0.002 | 32.7 | 0.011 | 0.033 | 0.049 | 0.099 | n/a | n/a | not measured | acceptable (<=0.1) |
| 1.5:0.001 | 41.6* | 0.009 | 0.021 | 0.022 | 0.039 | 0.000 | n/a | not measured | acceptable (<=0.1) |

Notes on the table: 1.5:0.01, 0.02, 0.05 and 0.1 are identical in results and cost (angle_slop >= 0.01 does not bind at bin_slop 1.5); 1.5:0.003/0.005 are non-monotonic in E_n (0.152/0.272 on A vs 0.057 on B for 1.5:0.003), i.e. patch noise, so angle_slop in 0.001-0.1 cannot be ranked with these data. Settings such as 1:0.003 / 1.5:0.001 / 1.5:0.002 look "acceptable" on patches but cost 29-42 h* (hours for 1:0.003 and 1.5:0.001 from the small patch only) and have no full-pair measurement; their saving over exact is 15-40 % at an unmeasured full-pair E_n error.

## Cross pair 1-6 (patch A) and stability
Pair 1-6 (923k x 691k), exact 1124 s vs default 1141 s (contended, 6 threads; same cost conclusion). 1.5:0.1: xi+ 0.057, xi- 0.106, E6 0.011, E20 0.037; 1:0.1: 0.046, 0.070, 0.057, 0.093. Patch B (6-6): 1.5:0.1 xi- 0.081, E20 0.094; 1:0.1 xi- 0.024, E20 0.186. Signal-field bias (SIGA, coherent) for 1.5:0.1 peaks at 0.052 sigma in xi- (the 4-10' bins) and <0.001 in xi+.

## Timings and contention
Hours use only the uncontended patch-A / patch-S timings recorded earlier (16 threads). Runs made in this session (patch B, SIGA, pair 1-6) used 6 threads alongside a 14-thread exact job and are CONTENDED (e.g. B exact 2243 s, SIGA exact 1871 s); their seconds are in the npz but not used for hours. Chunk-pair timings (queue4) were not done. The 49 h base is the CHUNKING.md estimate for all 21 pairs at (0, default); applying a Bin6_Bin6-patch ratio to the whole production assumes the ratio is the same for cross pairs (1-6: 228/1141 = 0.20 for 1.5:def vs 0.23 for 6-6 on A, contended, so roughly yes).
For scale: the full exact Bin6_Bin6 pair took 6999 s (1.9 h, 14 threads, contended) (`exact_bin6_run.log`).

## Full-pair exact result
Source `exact_bin6_eval_b0_a0.json` (exists; written by the detached job, EXIT=0). Exact (0,0) vs official, 9-bin max |Delta|/sigma (A3 convention): xi+ 0.0748 (PASS), xi- 0.1563 (FAIL, 150-300' bin); E_n 6 modes 0.238, 20 modes 0.484; A1 boxcar/white 0.421 / 0.457 (PASS); B_n null chi2 19.47/6 (official 19.24). Same numbers for the (0,default) pilot: 0.0748 / 0.1568 / 0.484 / 0.238. For slop 1.5 through our code: 0.0456 / 0.0888 / 0.071 / 0.071. External (theirs): 0.0238 / 0.0438 / 0.043 / 0.041. Exact vs slop-1.5 on the same data: see headline item 3.

## Recommendation
Use **(bin_slop 0, angle_slop 0)** for production: ~49 h (same as the default), exact to 1e-14 vs brute force, and no approximation error left in xi+, xi- or E_n. It is the only setting whose accuracy is established, and the exact measurement is the reference for any later comparison. If 49 h is too long, the next option is a setting with a full-pair check first (e.g. a full Bin6_Bin6 run at 0.5:0.1) because the patch forecasts under-predict E_n. [Verifier correction: 0.5:0 is identical to 0:0 and not cheaper; angle_slop=0 disables the bin_slop saving, so any mid-cost candidate needs angle_slop > 0. Hours: 49 h is an extrapolation, best estimate ~45 h at 16 threads (37-60); 1.5:0.1 is 8-15 h. Official bin_slop 1.5 IS documented: Wright+25 Sect. 2.5.1 footnote 1. See `verify/VERIFY.md`.] 1.5:0.1 (11 h) is only appropriate if the goal is to mimic the official approximation; its measured E_n error vs exact is 0.44 sigma at 20 modes.

## Not established
- Which TreeCorr settings the official release used (the agreement of slop 1.5 with official is empirical).
- Full-pair accuracy of any setting other than (0,0.1) and (1.5,0.1); mid-cost settings (0.5, 1.0, 1.5 with angle_slop 0.003) are patch-forecast only.
- Why the real E_n deviation of 1.5:0.1 exceeds the patch forecast (one survey realisation vs patch noise, or deviation not scaling as shape noise).
- Chunk-pair timings; cross pairs other than 1-6; patches in the South.
