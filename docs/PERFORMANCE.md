# kidspipe performance: cost per move, accuracy per step, expected sampling cost

Everything here was measured with this code (CAMB 1.5.8, numba 0.59.1, numpy 1.26, Python 3.12) on a 22-logical-core WSL2 machine that is **shared with other jobs**.
Section 1 says which numbers are trustworthy and why. Section 8 (speed-up versus the original kcap/CosmoSIS pipeline) is an **estimate**: kcap itself was never built or run.
"External" marks numbers that are not from this repository.

Contents: 1 provenance and noise, 2 per-move cost, 3 stage breakdown, 4 threads, 5 accuracy cost of each step, 6 emulator verification,
7 expected cost of full runs, 8 speed-up versus kcap (estimate), 9 start-up and memory, 10 caveats, 11 reproduce.

## 1. Provenance of the numbers, and the machine-load problem

| file (in `runs/`) | what | when | status |
|---|---|---|---|
| `bench_t4.json`, `bench_t1.json` | `benchmarks/bench_pipeline.py`, 4 / 1 threads, 29 timed moves per type (7 for config i), no accuracy part | 2026-10-25 22:07 / 22:09 | **used for the headline tables** |
| `bench_thread_scan_ii.json`, `bench_thread_scan_iii.json` (+ `bench_threads_*`) | same script, 1, 2, 4, 8, 16 threads | 2026-10-25 22:16-22:19 | used for section 4; reproduces `bench_t1/t4` to 2-5 % |
| `bench_pipeline.json` | same script with the accuracy part (20 points) | 2026-10-25 21:33 | **loaded** (see below); only its chi2 columns are used (section 5) |
| `bench_default/bench.json` | `python -m kidspipe bench` (config ii, one parameter at a time) | 2026-10-25 19:17 | moderately loaded (slow move 772 ms instead of ~550 ms) |
| `perf_doc/*.json` | runs made for this document (2026-10-26): accuracy decomposition, emulator verification, stage breakdown, start-up, two re-checks of the timing tables | today | stage breakdown was taken **under load** (load average 22-24); start-up at load 9-16; see 3 and 9 |

**Why `bench_pipeline.json` is not the headline.** Its numbers are 1.9-2.1x slower than `bench_t4.json` for configs ii-iv (ii slow block 1156 ms vs 582 ms; iii 72 vs 35 ms)
and internally inconsistent (config ii, which does *less* work than i, is slower than i for a slow move: 1104 vs 991 ms); it was taken while other jobs were running.
The earlier summary figures quoted for it (i: 2.5 s, ii: 1.15 s, iii: 72 ms) are therefore load-inflated. `bench_t4/t1` and the thread scan are three independent runs that agree with each other
(1-thread ii slow block: 1932 ms in `bench_t1`, 1960 ms in the scan; iii: 76.3 vs 77.5 ms) and with the quiet-machine CAMB timing in `docs/interface_notes/linear_background.md`
(`camb.get_results`, 150 z, 4 threads: 0.53 s; 1 thread: 2.04 s). I could not verify the load at 22:07-22:19 (it was not logged); "quiet" is an inference from that consistency.

**I could not re-measure on a quiet machine today.** The load average was 21-25 for the whole session (a 14-core science job). As a calibration I re-ran configs ii-iv at 1 thread
(`perf_doc/bench_recheck_t1.json`): everything was 2.7-3.6x slower than `bench_t1` (ii slow block 5813 vs 1932 ms; iii slow block 213 vs 76 ms; iii medium move 228 vs 64 ms; fast move 3.6 vs 1.3 ms).
The slow-down is not uniform (CAMB 2.7x, numba HMcode 3.6x), so scaling loaded numbers back is not exact. Use ratios between configurations from the quiet set only.
A second re-check at 4 threads (`perf_doc/bench_recheck_t4_load9.json`, load average 9 rising to 22 during the run; ii slow block 1.70 s vs 0.58 s, iii slow block 106 ms vs 35 ms, iii fast move 12 ms vs 1.2 ms, i slow block 1.63 s vs 1.10 s) was 1.5-10x slower and very noisy: not usable, kept only as evidence that the machine was not quiet.
Thread settings matter too: numba ignores `OMP_NUM_THREADS` and starts one thread per core unless `NUMBA_NUM_THREADS` is set (the CLI now sets it, `docs/REVIEW.md` F3; Python API users must set it). The benchmark script sets `NUMBA_NUM_THREADS` to `--threads`, and `run_bench_all.sh` set it explicitly, so the headline tables are not affected; results (not timings) are bit-identical for any thread count.

Timing convention: wall clock of `CosmicShearModel.mu` (the likelihood itself costs 0.2 ms with the generic Cholesky path, 0.03 ms in `kidspipe bench`). Every timed move goes to a **new** point (no cache can serve it);
the median over moves is quoted; the first move of each series is dropped; JIT is warm (a warm-up model is built first). "Cold" = first call of a freshly built model object (all caches empty, JIT warm).

## 2. Cost per move

Move types of `benchmarks/bench_pipeline.py`: *fast* = A_IA and dz1..5 move (only C_ell/Hankel/likelihood rerun); *medium* = ln_1e10_A_s and A_baryon move (exact P_lin rescale, new P_nl, Limber power look-up);
*n_s* = tilt of P_lin, new P_nl; *slow single* = one of omch2, ombh2, h (+1 % random-walk step); *slow block* = omch2, ombh2, h and n_s together (what a blocked Metropolis step does). Medians in ms.

Configurations: (i) `kcap` fidelity (FFT Hankel + node-trapezoid binning) with CAMB 'mead' non-linear: the faithful reference path; (ii) `fast` exact-binning operator + stand-alone numba HMcode + CAMB transfer;
(iii) = ii with the transfer **emulator** (`CosmicShearModel(linear_backend='emulator')`); (iv) = iii on the 41-node z grid (`z_grid='reduced'`); (v) = ii on the 41-node grid (no emulator).

**4 threads per process** (`bench_t4.json`; 22:07, treated as quiet):

| config | cold (min / median) | fast (A_IA, dz) | medium (lnAs, A_baryon) | n_s | slow single | slow block |
|---|---|---|---|---|---|---|
| i   kcap + CAMB mead | 1026 / 1059 | 1.8 | 480 | 517 | 1033 | 1096 |
| ii  fast + HMcode + CAMB | 554 / 560 | 1.1 | 25.9 | 27.2 | 579 | 582 |
| iii ii + emulator | 32.8 / 34.0 | 1.2 | 25.8 | 25.8 | 30.5 | 35.5 |
| iv  iii + 41 z | 20.4 / 21.5 | 1.1 | 11.9 | 14.0 | 16.7 | 23.4 |
| v   ii + 41 z (CAMB) | 537 / 543 | 1.1 | 12.2 | 13.1 | 553 | 554 |

**1 thread per process** (`bench_t1.json`; the relevant case for many parallel chains / nested workers):

| config | cold (min / median) | fast | medium | n_s | slow single | slow block |
|---|---|---|---|---|---|---|
| i   | 2852 / 2862 | 2.3 | 965 | 985 | 2901 | 2863 |
| ii  | 1903 / 1933 | 1.3 | 65.1 | 66.1 | 1928 | 1932 |
| iii | 73.3 / 76.0 | 1.4 | 63.7 | 65.7 | 71.1 | 76.3 |
| iv  | 37.4 / 39.2 | 1.4 | 27.5 | 27.5 | 33.4 | 39.0 |
| v   | 1846 / 1858 | 1.5 | 27.1 | 26.9 | 1866 | 1859 |

p90 of the slow block is within 10 % of the median for ii-iv (e.g. iii: 37.9 vs 35.5 ms at 4 threads); n = 29 per cell, 7 for config i, one machine state, so quote 2 significant digits at most.
For comparison the **load-inflated** `bench_pipeline.json` (4 threads): i 1022 cold / 2479 slow block (median of 7, mean 1958), ii 1128 / 1156, iii 90 / 72, iv 83 / 45; its fast-move column (2.6-3.9 ms, 12 ms for v) is also 2-10x too high.

Reading the tables:

* In the faithful path (i) every A_s, n_s or A_baryon move costs ~0.5 s because CAMB's own HMcode (150 z) is re-run: 480 ms. The stand-alone numba HMcode does the same job in ~20-26 ms (4 threads).
* The CAMB transfer evolution itself (~0.54 s at 4 threads, 1.9 s at 1 thread) is untouched in i and ii. Only the emulator removes it: iii's slow block is 16x cheaper than ii (35.5 vs 582 ms at 4 threads; 25x at 1 thread).
* With the emulator the cost is no longer dominated by the slow block: a *medium* move (26 ms) costs 73 % of a slow block (35.5 ms). P_nl (HMcode) is what is left (section 3).
* Reducing the z grid from 150 to 41 nodes halves the P_nl + Limber stage (medium 26 -> 12 ms at 4 threads) but changes chi2 by up to 0.07 (section 5). `fast` moves are unchanged (1.1 ms).
* The fast block (A_IA, dz): 1.1-1.2 ms with the fast operator vs 1.8 ms for the kcap FFT path.

## 3. Stage breakdown

`CosmicShearModel.timing()` splits a call into `linear` (CAMB transfer or emulator + background), `pnl` (HMcode or CAMB 'mead'), `limber_power` (P(k,z) -> Q(ell, chi) look-up table on the 2000-node grid and n(z) grid),
`limber_cl` (kernels + 15 C_ell), `hankel` (C_ell -> binned xi+/-); the likelihood is separate.

**Measured today under load (load average 22-24, 4 threads, median of 5 / 15 / 60 moves; `perf_doc/stages_loaded.json`). Absolute values are 2.5-3.6x the quiet ones: use the shares.**
Quiet-machine totals for the same moves are in section 2.

| config / move | linear | pnl | limber_power | limber_cl | hankel | sum (ms, loaded) | likelihood |
|---|---|---|---|---|---|---|---|
| i slow block | 1296 (43 %) | 1707 (56 %) | 25 (1 %) | 7 (0.2 %) | 1.5 | 3037 | 0.22 |
| i medium | 0.2 | 1579 (99 %) | 14 (1 %) | 6 | 1.7 | 1601 | 0.20 |
| ii slow block | 1406 (91 %) | 109 (7 %) | 19 (1 %) | 6 | 0.6 | 1541 | 0.18 |
| ii medium | 0.2 | 62.5 (75 %) | 16.4 (20 %) | 3.6 (4 %) | 0.15 | 82.9 | 0.19 |
| iii slow block | 11.4 (9 %) | 92.9 (75 %) | 15.8 (13 %) | 4.0 (3 %) | 0.13 | 124 | 0.18 |
| iii medium | 0.4 | 71.7 (77 %) | 17.0 (18 %) | 3.3 (4 %) | 0.16 | 92.6 | 0.21 |
| iv slow block | 16.8 (18 %) | 48.2 (52 %) | 20.2 (22 %) | 6.8 (7 %) | 0.3 | 92.3 | 0.19 |
| iv medium | 0.3 | 28.2 (45 %) | 23.7 (38 %) | 7.2 (11 %) | 3.7 (?) | 63.0 | 0.21 |
| any, fast move (ii) | | | | 3.4 (95 %) | 0.2 (5 %) | 3.6 | 0.21 |

The loaded run is noisy (e.g. the 3.7 ms Hankel in `iv medium` and 3.4 ms in the iv fast move are artefacts: the same stage costs 0.1-0.2 ms in ii and iii and the quiet fast move is 1.1 ms in total).
The least loaded stage record available is `bench_default/bench.json` (config ii, 4 CAMB threads, 19:17, slow move 772 ms; taken before the numba-thread fix, so numba probably used all 22 threads): for single-parameter medium moves linear 0.1-0.7 ms, **pnl 7-14 ms, limber_power 2.4-4.4 ms, limber_cl 1.4-2.0 ms, hankel 0.1 ms**
(total 11-21 ms); fast moves: limber_cl 0.5 ms (A_IA, delta_c, A_c) or 1.3 ms (dz_i, which rebuild the shifted n(z)) and hankel 0.05 ms; slow moves: linear 747-770 ms of 772-789.

Conclusions that hold in all three records: (1) the CAMB transfer evolution is >90 % of any slow move without the emulator; (2) with the emulator the linear stage is ~10-12 ms
(1 thread, difference of `bench_t1` iii slow block 76.3 and medium 63.7 ms; the emulator's own `predict` is 2-6 ms, the rest is the CAMB background for chi(z) and E(z), the exp of two 150 x 199 arrays and the sigma8 quadrature);
(3) HMcode is 75 % of a medium or emulated slow move, the Limber look-up 13-20 %, C_ell 3-5 %, Hankel and likelihood together < 1 %.
The next acceleration target is therefore HMcode (or the 41-node grid), not the linear stage.

## 4. Thread scaling (`bench_thread_scan_ii/iii.json`, 11 / 24 moves per cell, same session)

CAMB threads are set per call through `camb.config.ThreadNum`; numba threads (`NUMBA_NUM_THREADS`) follow the same setting in the benchmark. Medians in ms.

| threads | ii slow block | speed-up | ii medium | iii slow block | iii medium | iii fast |
|---|---|---|---|---|---|---|
| 1  | 1960 | 1.0 | 65.5 | 77.5 | 65.4 | 1.5 |
| 2  | 1008 | 1.9 | 36.6 | 46.5 | 36.3 | 1.3 |
| 4  | 557  | 3.5 | 21.4 | 34.8 | 20.8 | 1.3 |
| 8  | 392  | 5.0 | 20.4 | 33.1 | 16.2 | 1.5 |
| 16 | 317  | 6.2 | 14.8 | 45.6 | 12.7 | 1.7 |

* CAMB scales sub-linearly: 97 % efficiency at 2 threads, 88 % at 4, 62 % at 8, 39 % at 16. Throughput per core is best with **one thread per chain**.
  CAMB's default is to use all cores unless pinned (and the kcap study did not record any pinning); the independent implementation measured the CAMB step at 3.7 s with 22 threads vs 1.00 s with 4 (`MASTER_MATH_SPEC.md` s15.5, `KCAP_OVERVIEW.md` s7 item 1): oversubscribing threads is a 3.7x loss by itself.
* The emulated slow block saturates at 4-8 threads (34.8 -> 33.1 ms) and gets **worse** at 16 (45.6 ms): the non-threaded linear stage (~10 ms) plus per-call thread start-up overhead; the medium (P_nl-dominated) move still speeds up 5x at 16 threads.
  I did not profile the 16-thread degradation; this explanation is a guess. Use 2-4 threads per process with the emulator, 1 when running many chains.
* The README's measured pool throughput (8 single-thread workers, config ii, loaded machine: 2.83 new cosmologies/s vs 4.1/s ideal from the quiet 1-thread cost) suggests ~70 % parallel efficiency across processes
  (memory bandwidth and SMT sharing); a 20-worker emulator run was not measured.
* Rule of thumb for the tables above: if you run N parallel processes on this machine, expect each to be up to 1.5x slower than the numbers in section 2 (and ~3x under the load seen today).

## 5. Accuracy cost of each step

Same 20 points for every configuration (`bench_pipeline.json`'s seed-11 Sobol design: omch2 in [0.04, 0.30], ombh2, h inside the prior box, n_s in [0.8, 1.2], S8 ~ U(0.65, 0.85), A_baryon, A_IA, c-term and dz from their priors);
chi2(i) ranges over 184.6-372.7 (10 points < 200, 16 < 230, one at 299 and one at 373). The comparison of every cumulative step uses chi2 of configuration i (the faithful `kcap` path with CAMB 'mead') as the reference.
Two further configurations (kcap + HMcode, fast + CAMB mead) were run to isolate individual steps (`perf_doc/accuracy_decomp.py`, `.json`; `bench_pipeline.py` configs `vi`, `vii`).
Entries are max |Delta chi2| / rms / mean Delta chi2 (new minus old), first over all 20 points, then over the 16 points with chi2 < 230.

| step | what changes | all 20 | chi2 < 230 (16 pts) | nature |
|---|---|---|---|---|
| **1** | kcap -> `fast` exact bin-averaged Bessel operator (with CAMB 'mead' held fixed: vii - i) | 1.16 / 0.37 / -0.24 | 0.35 / 0.17 / -0.13 | **numerical choice**, not an error: the exact integral over each theta bin versus the spec's node-trapezoid |
| 2 | CAMB 'mead' -> stand-alone HMcode (kcap binning: vi - i) | 0.22 / 0.069 / -0.040 | 0.12 / 0.039 / -0.021 | true approximation difference of the re-implementation |
| 2' | same, with `fast` binning (ii - vii) | 0.21 / 0.068 / -0.039 | 0.12 / 0.039 / -0.021 | (the HMcode difference is the same with either binning: rms 0.069 vs 0.068) |
| 3 | transfer emulator (iii - ii) | 0.0073 / 0.0022 / -0.0004 | 0.0018 / 0.0007 / -0.0001 | true approximation error; same with kcap binning: 0.019 / 0.0059 / +0.0022 |
| 4 | 150 -> 41 z nodes (iv - iii; the same with CAMB, v - ii) | 0.073 / 0.021 / +0.012 | 0.021 / 0.008 / +0.005 | true approximation error (0.073 in both); with kcap + CAMB 'mead' it is 0.35 / 0.10 / +0.06 (0.096 for chi2 < 230) |
| cumulative | i -> iii (fast + HMcode + emulator) | 1.39 / 0.44 / -0.28 | 0.41 / 0.20 / -0.15 | |
| cumulative | i -> iv (+ 41 z) | 1.31 / 0.42 / -0.27 | 0.40 / 0.20 / -0.14 | |

Reading:

* Of the cumulative difference between the fast+emulator path and the faithful path, ~85-90 % is step 1, the binning convention, which grows with the residual (0.18 max over the 10 points with chi2 < 200, 0.35 for chi2 < 230, 1.2 at chi2 = 373).
  It is **not** an error of the approximations: `kcap` fidelity reproduces the spec anchors to <= 0.08 in chi2, the `fast` path to <= 0.56 (README); use `--fidelity kcap` / `fidelity='kcap'` where the original convention matters.
  The tolerance written in `DESIGN.md` (|Delta chi2| < 0.1 for fast vs faithful) is **not met on this set even for chi2 < 200** (max 0.21 over those 10 points, 0.18 of it from the binning alone); the README quotes 0.03-0.04 at the points it checked at chi2 189-196. Within chi2 < 230 the maximum is 0.41.
* The true approximation errors are small: HMcode vs CAMB 'mead' <= 0.12 near the data (mean -0.02), the emulator <= 0.002 near the data and <= 0.007 over the whole range, the 41-node z grid <= 0.02 near the data (but 0.07 at chi2 ~ 370, and a positive mean shift of 0.005-0.012: it is not free).
  `MASTER_MATH_SPEC.md` s15.3 reports 188.668 vs 188.668 for the 41-node grid at a single point; across these 20 points it is not that small.
* 20 points is a small sample, concentrated at S8 0.65-0.85 where chi2 ranges widely; the per-step statistics are indicative, not a bound. The independent emulator check with 30 new points is in section 6.

## 6. Emulator verification (independent run, today)

The emulator (`kidspipe.theory.emulator`, artefact `data/emulator/transfer_emulator_v1.npz`, 9.9 MB) emulates the linear total- and cold-matter spectrum at the reference primordial spectrum on a fixed (z, k) grid
(169 master z nodes, 199 k nodes in [5e-5, 21] /Mpc) as a function of (omch2, ombh2, h) only: PCA (64 + 8 components) + polynomial/Gaussian-process regression, trained on 1435 CAMB runs (Sobol design + 27-point lattice + 384 extra at low omch2) with a dense k sampling of 40 nodes per ln k.
A_s and n_s are exact rescales; chi(z), E(z), Omega_m are exact (CAMB background only); sigma8 is a quadrature of the emulated spectrum. Box: omch2 [0.03, 0.34], ombh2 [0.019, 0.026], h [0.64, 0.82], m_nu = 0.06 eV; outside it the request is passed to exact CAMB.
All numbers below were produced by `perf_doc/verify_emulator.py` (seed 20261026, new points not used in any earlier validation; log `perf_doc/verify_emulator.log`).

**(a) Pipeline, 30 new random in-box points** (Sobol in the box, n_s in [0.8, 1.2], S8 ~ U(0.65, 0.85), other parameters from their priors; chi2 range 185.9-303.7; both models `fast` + HMcode + spec z grid):

| quantity (emulator minus exact CAMB) | value |
|---|---|
| max abs Delta chi2 / rms / mean | **0.0040** / 0.0016 / -0.0008 |
| max relative difference of any element of mu (195 values) | 1.4e-4 (median over points of the per-point max: 4.0e-5) |
| max abs difference of mu in units of the sqrt of the covariance diagonal | 2.7e-4 |
| fallbacks used / emulator evaluations | 0 / 30 |
| (reference) CAMB with dense k sampling (40 nodes per ln k, kmax 21) minus default-sampling CAMB | max abs Delta chi2 0.013, rms 0.0048; max mu difference 5.2e-4 |

The earlier cached validation (`validate_emulator_pipeline*.json`: 50 points, max 0.0077 spec grid, 0.012 on the 41-node grid) is consistent. In this pipeline test the emulator is closer to the production CAMB
(0.004) than a denser CAMB run is (0.013): the cost of CAMB's own k sampling in chi2 is about three times the emulator error.

**(b) Out-of-box requests fall back to exact CAMB, with identical numbers.** Five points (omch2 = 0.45 and 0.02; ombh2 = 0.030 and 0.015; h = 0.90 and 0.60) through `CosmicShearModel(linear_backend='emulator')` and through the CAMB model:
`mu` is **bitwise identical** (`np.array_equal`) in all five, the counter `linear.stats['fallback']` increases by exactly 1 per point, `emu_evals` stays 0, and an in-box control point increments `emu_evals` and not `fallback`.
The box is closed: a point exactly on the upper corner is emulated; a point 1e-9 (relative) above a face falls back. (Tests added: `test_box_edge_is_inclusive_and_just_outside_is_refused`, `test_model_level_fallback_is_bitwise_exact_camb_and_counted`.)

**(c) Error of the emulated linear P(k, z) versus k and z** (28 points: 8 corners pulled in by 1 % of the log range, 6 rim points with one coordinate on the 1 % rim and the others random, 6 random, plus the 8 exact corners which are training points).
Relative error |P_CAMB / P_emu - 1| of P_lin (the cold spectrum P_cb gives the same maxima: 1.30e-3 vs dense truth), restricted to k_h < 10 and z <= 3, CAMB values spline-interpolated in ln k onto the emulator nodes.
Statistics are over the 20 out-of-sample points (including the training corners changes nothing: 1.30e-3 vs dense truth, 6.7e-3 vs default sampling).

| truth | max | worst-point p99 | worst-point median | share of (k, z) values below 5e-4 (worst point) | below 2e-3 (worst point) |
|---|---|---|---|---|---|
| **dense-sampling CAMB** (40 nodes per ln k, the training truth) | **1.30e-3** | 3.1e-4 | 1.5e-5 | 99.5 % | 100 % |
| default-sampling CAMB (what the production `LinearBackend` delivers) | 6.5e-3 | 5.2e-3 | 2.1e-5 | 93.3 % | 97.9 % |

Maximum error by k_h band (z-independent to 1 %: every z band 0.0 ... 3 has the same maximum, 1.3e-3 / 6.5e-3):

| k_h [h/Mpc] | 1e-4 - 0.01 | 0.01 - 0.1 | 0.1 - 1 | 1 - 3 | 3 - 10 |
|---|---|---|---|---|---|
| vs dense truth | 3.6e-5 | **1.3e-3** | 1.6e-4 | 3.8e-5 | 1.9e-5 |
| vs default-sampling CAMB | 3.1e-4 | 1.3e-3 | **6.5e-3** | 1.2e-4 | 2.0e-5 |

Acceptance targets (<= 2e-3, goal <= 5e-4 for k_h < 10, z <= 3):

* **vs the dense-sampling truth the emulator meets the 2e-3 target (max 1.30e-3, at z ~ 0, k_h ~ 0.058) and does not meet the 5e-4 goal as a maximum**; 99.5 % of all values at every test point are below 5e-4, the p99 is below 3.1e-4,
  and the training notes (`runs/emulator_train/validation.json`) give a max of 2.3e-4 on their 256 held-out points once a narrow "CAMB glitch window" is excluded (1.8e-3 including it). I did not check whether my 1.3e-3 maximum lies inside that window; my maximum is consistent with their 1.8e-3.
* **vs CAMB at its default sampling the maximum is 6.5e-3, above the 2e-3 target**, only at k_h = 0.45-0.60, only for omch2 near the lower box edge together with ombh2 near the upper edge (2 of the 20 out-of-sample points, the two inset corners with low omch2 and high ombh2, plus the two exact training corners there), and is **an artefact of CAMB's own k nodes** (BAO aliasing when CAMB's native 190-ish nodes are interpolated):
  at the same points the emulator agrees with the dense truth to 5e-4 and 1.0e-3. The consequence for the likelihood is negligible (section 6a: 0.004 in chi2 at worst). If a reviewer insists on the literal reading "versus the production CAMB output", the target is not met at that corner.
* Error structure: negligible at high k (3-10 h/Mpc: 2e-5); the dense-truth maximum sits at k_h ~ 0.01-0.1; no visible z dependence (the five z bands have the same maximum to 0.5 %).

## 7. Expected cost of full runs

**Blocked-Metropolis cycle** (default schedule of `kidspipe mcmc`: 1 joint update of the six cosmological parameters, `--rep-medium 2` updates of (n_s, ln_1e10_A_s, A_baryon), `--rep-fast 6` updates of the eight fast parameters = 9 likelihood evaluations).
Cost = slow block + 2 x medium + 6 x fast (e.g. iii at 4 threads: 35.5 + 2 x 25.8 + 6 x 1.2 = 94.1 ms), from section 2 (`python benchmarks/bench_pipeline.py --mcmc runs/bench_t4.json` prints this). Per chain; 4 chains run in parallel.

| config | 4 threads: ms / cycle | ms / eval | 10^4 cycles | 2.5 x 10^4 cycles | 1 thread: ms / cycle | 10^4 cycles | 2.5 x 10^4 cycles |
|---|---|---|---|---|---|---|---|
| i   | 2067 | 230 | 5.7 h | 14.4 h | 4806 | 13.4 h | 33.4 h |
| ii  | 641  | 71  | 1.8 h | 4.5 h  | 2070 | 5.8 h  | 14.4 h |
| iii | 94.1 | 10.5 | 15.7 min | 39.2 min | 212 | 35.4 min | 88.5 min |
| iv  | 53.8 | 6.0 | 9.0 min  | 22.4 min | 102.6 | 17.1 min | 42.8 min |

How many cycles are needed is **not** established here. The two existing runs (1300 cycles x 4 chains incl. burn-in) were not converged for omch2, ombh2, h, sigma8 (ESS 40-100, R-hat up to 1.5; README).
If ESS grows linearly with length, ESS ~1000 in the worst direction needs roughly 10-25x more cycles, i.e. 10^4 - 2.5 x 10^4 per chain: this is an extrapolation, not a measurement; the columns above bracket it.

Consistency check of the model: the joint-block run took 39 min for 4 x 1300 cycles = 1.8 s/cycle per chain with 4 chains x 4 CAMB threads on the loaded machine (load 25-50). The quiet-machine cost for config ii is 0.641 s/cycle,
a factor 2.8 lower, in line with the 2.7-3.6x load slow-down measured in section 1 (a consistency check, not an independent validation: the load during those chains was not logged, and those chains ran before the numba-thread fix, i.e. 4 x 22 numba threads on top of the load). So the same run would take ~14 min quiet in ii and ~2 min with the emulator (iii).

With the emulator, a medium update (26 ms) costs almost as much as a slow one (35.5 ms), so the reason for the blocked schedule (cheap medium/fast updates after one expensive slow one) is largely gone for the medium block:
a schedule with `--rep-medium 0` or 1 costs 43 or 69 ms/cycle instead of 94 ms (iii, 4 threads). Whether that mixes better per second was not tested.

**Nested sampling and MAP.** Every nested-sampling point is a new cosmology: the cost per call is the slow-block column (1 thread): i 2.9 s, ii 1.9 s, iii 76 ms, iv 39 ms.
With 20 single-thread workers (ideal, no contention): 10^6 calls = 40 h (i), 27 h (ii), **1.1 h (iii)**, 0.5 h (iv); with the 70 % cross-process efficiency suggested in section 4, 57 h, 38 h, 1.5 h, 0.8 h.
**Caveat that matters:** the prior of omch2 is [0.01, 0.99] but the emulator box is [0.03, 0.34]: only 31.6 % of the prior volume is inside the box, so the early (prior-wide) phase of a nested run falls back to exact CAMB for ~2/3 of its points (1.9 s each at 1 thread) before the live points concentrate in the box.
All 4000 + 2800 samples of the two existing chains lie inside the box (omch2 0.05-0.245), so MCMC started near the posterior is unaffected. A nested run with the emulator has not been done.
The finite-difference MAP gradient (3 CAMB runs, ~1.7 s quiet at 4 threads) becomes ~3 x 35 ms.

## 8. Speed-up versus the original kcap/CosmoSIS design: ESTIMATE ONLY

kcap was never built or run in this project (`KCAP_OVERVIEW.md` s6.3: the clone does not build with a current toolchain; no timings exist in the repository). Everything in this section is reasoning from operation counts and from the timings of this code,
and the stages marked "est." could be off by a large factor. What kcap does per likelihood call (`KCAP_OVERVIEW.md` s2.1, s6.4):

1. python CAMB in `transfer` mode with HMcode-2016 non-linear power on **150 z** and ~195 k (+ distances on 6000 z and unused outputs). Measured with CAMB 1.5.8 by the independent implementation and by this project's config i: **1.0 s with 4 threads, 2.9 s with 1, 3.7 s with CAMB's default 22 threads** (also `bench_t4` i: 1.03 s slow single, 1.10 s slow block, 0.48 s of it the 'mead' re-run). This is real CAMB work in compiled Fortran; kidspipe's config i spends the same time. Nothing is gained here without a different method.
2. `extrapolate_power`, `load_nz`, `source_photoz_bias`, `linear_alignment`: small.
3. Limber projection in C/GSL: kernels W, N from 1024-point Gauss-Legendre (5 x 1024 x 1024 integrand evaluations), then **55 spectra (15 GG + 25 GI + 15 II) x 400 ell by adaptive QAG at relative tolerance 1e-3** (est. 0.8-3 s: `KCAP_OVERVIEW.md` item 4 gives "5-20x" over the 0.16 s of the vectorised numpy equivalent; not measured). The non-smooth QAG result is also noise in C_ell.
4. nicaea FFTLog `cl2xi`: 15 spectra, one 1024-point transform each, ~30k complex-Gamma evaluations per call, plus a python node-trapezoid binning loop (28.7 ms measured in the spec study) and `keep_theory_vector` RNG draw (5-17 ms): est. 50-100 ms.
5. CosmoSIS datablock: lower-casing of keys and copies of P(k,z) (150 x ~400) several times per call (est. tens of ms), and **no fast/slow caching in the stock sampler setup**: every nuisance-parameter change reruns all of 1-4.

Estimated kcap cost of one likelihood call: **>= 1.1 s** (CAMB step at 4 threads + the cheap stages, a floor), **~2-4 s likely** (4 threads pinned + QAG projection), ~5 s with CAMB's default threading. The `KV450` sampler settings of kcap
(140k calls for a quick MultiNest, 350k-1M for production; 40 MPI ranks for 2 days requested, `KCAP_OVERVIEW.md` s2.3) imply 1e6 x ~3 s / 40 = 21 h, which is the right order of magnitude for the requested allocation, a weak plausibility check of the per-call estimate.

Per-call speed-up of kidspipe (quiet-machine medians, 4 threads, section 2) against that estimate:

| move | kidspipe | vs kcap floor (1.1 s) | vs kcap likely (~3 s) | what the gain consists of |
|---|---|---|---|---|
| nuisance only (A_IA, delta_c, A_c, dz) | 1.1-1.2 ms | >= 900x | ~2500x | caching: stage-2 (P_nl, Limber tables) kept, linear operator for Hankel/binning; exact, no approximation |
| A_s, n_s, A_baryon | 26 ms | >= 40x | ~110x | numba HMcode (26 ms vs CAMB's 480 ms) + exact P_lin rescale (no new transfer function) |
| omch2, ombh2 or h, exact CAMB (config ii) | 580 ms | ~1.9x | ~5x | HMcode in numba and faster projection only; the same CAMB transfer evolution |
| omch2, ombh2 or h, emulator (config iii) | 31-36 ms | >= 31x | ~85x | approximation: emulator, 0.004 in chi2 (section 6) |
| ... and 41-node z grid (config iv) | 17-23 ms | >= 47x | ~130x | + 0.02-0.07 in chi2 (section 5) |

Of the whole-run gain, caching (rows 1-2) and the emulator (row 4) dominate; the exact-CAMB path (row 3) is only a modest improvement. The speed-up is therefore **mostly an algorithmic/caching effect plus one controlled approximation, not faster physics code**,
and it is largest for samplers that make many nuisance-only moves (blocked MCMC) and smallest for nested sampling with exact CAMB.
Rough end-to-end figure (estimate): 1e6 kcap calls at ~3 s are ~800 core-hours; 4 chains x 2.5 x 10^4 blocked cycles with the emulator at 1 thread are 10^5 cycles x 0.21 s = 5.9 core-hours. About 140x less CPU, for a different algorithm (MCMC without evidence instead of MultiNest), so it is a cost comparison, not an equal-output comparison.
Not measured: any science-equivalence of an emulator-based chain with a CAMB-based one (only per-point chi2 agreement is shown).

## 9. Start-up and memory (measured today at load average 9-16, partly 22; `perf_doc/startup.py`, 4 threads)

| item | value |
|---|---|
| numba JIT, **empty cache** (first `mu` of a fresh process: HMcode, Limber, Hankel kernels) | 15.5 s (`NUMBA_NUM_THREADS=4`, load 9-16); 31.0 s (22 numba threads, load 22). The project README's quiet-machine figure is ~10 s; not re-measured quiet |
| same, JIT cache on disk (default location `src/kidspipe/**/__pycache__`) | first `mu`: 0.46-0.53 s (emulator) / 1.4 s (exact CAMB: includes the first CAMB run); second, new-cosmology `mu`: 0.05 s (emulator) / 0.94 s (CAMB) |
| import `kidspipe` + camb + numba | 0.8-0.9 s (1.0-1.4 s at load 22) |
| load the emulator artefact (`TransferEmulator()`, 9.9 MB npz) | 0.18-0.20 s (0.21-0.42 s at load 22); +21 MB RSS |
| build the model object (n(z) tables, Hankel operator) | 0.32-0.52 s |
| whole process, ready and 3 evaluations done (cached JIT, emulator) | 2.4-2.5 s (3.2-4.1 s at load 22) |
| resident memory after import / after emulator load / after a few evaluations | 149 MB / 170 MB / 249 MB (peak 274-295 MB); CAMB-only model: 222 MB. A 20 000-call run stays flat at 270-275 MB (`docs/REVIEW.md`, hunt_mem) |
| per-process memory in an MCMC pool (observed with `top`, 8 workers) | ~310 MB each |
| disk: emulator artefact / training data (`runs/emulator_train`, not needed at run time) | 9.9 MB / 500 MB |

Memory scales with processes (20 workers ~ 6 GB of the machine's 15.7 GB); the emulator adds ~20 MB per process.
The numba cache is per source file and per CPU: a different machine, a changed source file or `NUMBA_CACHE_DIR` pointing at an empty directory brings back the ~10-30 s compile; with many workers started at once on a cold cache each compiles separately.
The emulator artefact is stamped with the CAMB version (1.5.8), box, seeds and sizes (`meta_json`); `tests/test_emulator.py::test_artefact_stamps` checks them.

## 10. Caveats, honestly

* **Not measured on a quiet machine today** (section 1); the headline numbers are from yesterday's `bench_t4/t1`, whose quietness is inferred. Headline ratios (iii vs ii: 16x slow block, 25x at one thread) rest on those.
* **The emulator is opt-in and API-only.** `CosmicShearModel(linear_backend='emulator')` or `api.build_posterior(linear_backend='emulator')` works (checked today: chi2 188.73 at the `scan` preset, 0 fallbacks); the CLI (`_model_kw` in `cli.py`) has no switch for it, so `kidspipe mcmc/nested/map` always use exact CAMB.
* **Box coverage.** Outside omch2 [0.03, 0.34], ombh2 [0.019, 0.026], h [0.64, 0.82] or m_nu != 0.06 the cost silently goes back to ~0.55-1.9 s per new cosmology. `linear.stats['fallback']` counts it; check it after every run. ombh2 and h boxes equal the prior ranges; omch2 covers 31.6 % of its prior.
* The emulator does not emulate the CAMB 'mead' non-linear spectrum: `pnl_backend='camb'` with the emulator triggers a lazy exact CAMB run (correct, but not fast).
* The independent review (`docs/REVIEW.md` s4) probed the box faces: at the 8 corners and 6 face centres the ln P error is up to 1.13e-3 (my 28-point P check: 1.30e-3 max, same order) and |Delta chi2| up to 0.046, but only at chi2 = 2977 (1.5e-5 relative); at chi2 = 230-250 corners |Delta chi2| <= 0.004. The 0.004 / 0.0077 pipeline numbers hold for data-compatible amplitudes (S8 0.65-0.85).
* The emulator error versus the production CAMB output is dominated by CAMB's own k-node aliasing at the low-omch2/high-ombh2 corner (section 6c); it is a 0.004 effect in chi2.
* The 41-node z grid is a different approximation from the emulator: up to 0.07 in chi2 on these 20 points, with a positive mean shift. Decide per use case.
* The `fast` exact-binning difference is a deliberate numerical choice (section 5, step 1); it must not be counted as emulator or HMcode error, and it is not an error against the spec anchors beyond 0.56.
* `tests/test_hankel.py::test_timing_fast_operator_and_fft` is a wall-clock test and fails on a loaded machine; `tests/test_emulator.py::test_evaluation_time_below_10ms` asserts on the best of 30 evaluations for the same reason.
* Wall-clock numbers depend strongly on co-running jobs (2.7-3.6x slow-down seen); the CPU time of a single-thread process is not a cure (the slower core speed under SMT/memory contention shows up in CPU time as well).
* The kcap comparison in section 8 is an estimate; do not quote its ratios as measured.

## 11. Reproduce

```bash
cd <repo>; export PYTHONPATH=$PWD/src OPENBLAS_NUM_THREADS=1
# timing tables (run only on a quiet machine: check `uptime`)
OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 python3 benchmarks/bench_pipeline.py --threads 4 --moves 30 --no-acc --out runs/bench_t4.json
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 benchmarks/bench_pipeline.py --threads 1 --moves 30 --no-acc --out runs/bench_t1.json
python3 benchmarks/bench_pipeline.py --threads-scan 1 2 4 8 16 --moves 25 --cfg-scan iii        # thread scaling (writes runs/bench_thread_scan_iii.json)
python3 benchmarks/bench_pipeline.py --mcmc runs/bench_t4.json                                   # expected-MCMC accounting
# accuracy decomposition (section 5): all seven configurations on the same points, no timing
python3 benchmarks/bench_pipeline.py --configs i ii iii iv v vi vii --acc-only --n-acc 20 --out runs/acc_decomp.json
# independent emulator check (section 6) and the stage breakdown / start-up
python3 runs/perf_doc/verify_emulator.py ; python3 runs/perf_doc/stages.py ; python3 runs/perf_doc/startup.py emulator
python3 benchmarks/validate_emulator_pipeline.py --n 50 --seed 777                              # the cached 50-point validation
PYTHONPATH=src python3 -m pytest -q tests/test_emulator.py -m "not slow"                          # emulator unit tests incl. the held-out checks
```
