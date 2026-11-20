# Independent whole-package review of `kidspipe`

Reviewer: independent agent (did not write the code); machine shared with ~20 other jobs (load average 20-25 throughout,
so all wall times below are pessimistic); <= 4 cores used. Clean-room rule respected (no `reference/`, `docs/study/`,
`docs/spec_checks/*.py` read). All scratch scripts and logs are in `runs/review/` (names given below).
Python 3.12.4, numpy 1.26.4, scipy 1.13.1, numba 0.59.1, CAMB 1.5.8.

## 0. Verdict

* **No defect was found that changes the likelihood of the default configuration by more than ~0.1 in chi2.** An end-to-end
  re-derivation written from `MASTER_MATH_SPEC.md` only (own CAMB driver, own Limber quadrature, own Hankel quadrature,
  own data/covariance handling) reproduces the package to 0.006 in chi2 at the spec test cosmology and 0.07 at an off-centre
  point with every nuisance parameter switched on (section 2).
* Four real (non-numerical) defects were found and fixed (section 7): n_s listed as a slow parameter (approved cleanup),
  a thread race on `CosmicShearModel.last_derived`, numba ignoring `OMP_NUM_THREADS` (CPU oversubscription), and no
  CAMB-version guard on the emulator artefact. Several documentation claims are stale or unbacked (section 5).
* One scientific question cannot be settled from the repository: **whether the shipped data vector is already m-corrected**
  (section 6). The effect is +-3 in chi2 and +-0.2 % in S8, i.e. small compared with the 5 % statistical error on S8,
  but it is a systematic that should be closed from the original data release notes.
* The full test-suite result is in section 1.

Findings are ranked in section 8 (severity, evidence, fix).

## 1. Test suite, clean copy, packaging

**Full suite including the `slow` tests** (`pytest -q`, 2 threads, machine load 20-25; logs `runs/review/full_suite2.log`, `full_suite3.log`):

* run 1 (tree after F1, before F2-F4): **269 passed**, 1 warning (scipy `IntegrationWarning` in a quadrature inside `test_hankel.py:299`), 24 min.
* run 2 (tree after all fixes; 277 tests collected because the emulator owner added tests meanwhile): **276 passed, 1 failed** in 23 min. The failure was my own new
  regression test `test_shared_model_threads_last_derived_is_thread_local`, which compared `mu` bitwise: with `OMP_NUM_THREADS=2` multi-threaded OpenBLAS splits a matmul differently under thread
  contention (1 ulp). Relaxed to `rtol=1e-10` for `mu` (the sigma8 comparison stays exact, which is what the test is about); the mutant without the fix is still killed (73 wrong values), and
  `tests/test_emulator.py` now passes completely (23 passed, 48 s).
* Timing tests are load-sensitive: in an earlier `-x` run (load 20-25) `test_emulator.py::test_evaluation_time_below_10ms` failed; it passed in runs 1 and 2. `test_hankel.py::test_timing_fast_operator_and_fft` passed in both runs.
* The two `slow` CLI nested-sampler smoke tests dominate the wall time (444 s nautilus, 385 s dynesty on the loaded machine).
* Fast subset (`-m "not slow"`) is 259+ tests (README/task statement: 256 before the later additions).

**Clean copy** (`runs/cleancopy`, made with `tar --exclude=./runs --exclude=./reference`, 14 MB, PYTHONPATH install,
no editable install):

* `python -m kidspipe check` (log `runs/review/cleancopy_check1.log` for the tree as received, `cleancopy_check3.log` for the final tree = the copy now in `runs/cleancopy`): ALL CHECKS PASSED in both; chi2 = 196.104 (anchor 196.06),
  sigma8/Omega_m/S8 = 0.709092/0.250296/0.647692 (identical to the anchors). Cold start 36-44 s (numba JIT of a fresh checkout,
  loaded machine) against "~10 s" in the README. `check --scan --camb` also passes (`cleancopy_check2.log`: scan max |diff| 0.56, backend diff -0.013).
* Data paths: resolved relative to the source tree (`Path(__file__).parents[3]/data`) in the copy; `KIDSPIPE_DATA=<other root>`
  is honoured by `likelihood.data` and `theory.nz`; a non-existent `KIDSPIPE_DATA` raises a clear `FileNotFoundError`
  (not a silent fallback). **The emulator artefact path ignores `KIDSPIPE_DATA`** (only `KIDSPIPE_EMULATOR`).
* JIT cache in a read-only tree (copy of `src/` + `data/` with `chmod -R a-w`, no `__pycache__`): works; numba falls back to
  its per-user cache (`~/.cache/numba`), cold start 31 s, nothing is written into the tree, correct chi2 (196.106 with the reduced grid).
* Packaging: `pip wheel .` (no isolation) produces a wheel that contains **no data files** (27 files, all `.py`). Only an
  editable/source-tree install works out of the box (README says `pip install -e .`, which is right); a regular
  `pip install .` would need `KIDSPIPE_DATA` *and* `KIDSPIPE_EMULATOR`.

## 2. Independent end-to-end re-derivation (task 2)

Script `runs/review/indep.py` (no `kidspipe` import for the physics; ~350 lines). Implementation choices that differ from the package
on purpose: P_nl straight from CAMB `Halofit('mead', HMCode_A_baryon, HMCode_eta_baryon=0.98-0.12A)` on its native grid;
spec s2.6 extension re-implemented with `np.polyfit`; ln P bicubic `RectBivariateSpline` in (z, ln k) (package: linear in z,
4-point Lagrange in ln k); distances from `H(z)` of CAMB with `scipy.quad`/dense `CubicSpline` (checked to 5e-11); W_i from the
exact piece-wise-linear cumulative integrals with inner `quad`; C_ell by Gauss-Legendre per n(z) interval with the k = 500 h/Mpc
cut inserted exactly; xi+/- by **direct quadrature of (1/2pi) int l C J_mu(l theta) dl** (Gauss-Legendre panels of width pi/4 in
l theta up to l = 5e6, J0/J4 from `scipy.special.jv`) at 6 Gauss nodes per theta bin with the theta-weighted exact bin average;
data/cov/c-term/m-bias/cuts/chi2 re-read from the files with numpy (`np.linalg.solve`).
(The first version of my Hankel quadrature used panels of width pi with 10 nodes and was *not converged* at the 1-2 % level for xi+;
the package FFT and operator routes were right and my quadrature wrong - found by changing panel width, kept here as an example of why
two independent paths were needed. Converged result below.)

Test cosmology = spec s4.1 (omch2 0.1, ombh2 0.022, h 0.7, n_s 0.96, ln As 3.0, A_baryon 2, no nuisance);
off-centre = (0.14, 0.0235, 0.68, 1.02, 3.1, A_baryon 2.7, A_IA 1.3, delta_c 3e-4, A_c 1.2, dz = (0.02, -0.015, 0.01, -0.008, 0.006)).

| quantity | independent | package |
|---|---|---|
| sigma8, Omega_m, S8 (test) | 0.7090923, 0.2502956, 0.6476924 | 0.709092, 0.250296, 0.647692 (api / `check`) |
| Limber C_ell, pair (3,1) at ell = 10 / 300 / 3000 (test): nested `scipy.quad` vs my vectorised GL | rel diff 2.4e-5 / 6e-8 / 1.6e-6 (off-centre: -1.9e-5 / -6e-8 / 1e-6) | (the ell = 10 difference is attributed to the step of the integrand at k = 500 h/Mpc inside the first quad interval; not investigated further) |
| package Limber vs mine, all pairs, 10 < ell < 1e4 | max 1.1e-4 | |
| same, ell < 10 / ell > 1e4 | 2e-3 / 1e-2 (largest at ell ~ 5e5) | |
| **chi2, test cosmology** | **196.0970** | 196.0910 (camb backend, fast), 196.1043 (hmcode, fast), 196.0618 (camb, kcap) |
| **chi2, off-centre** | **350.7577** | 350.6878 (camb, fast), 350.4400 (hmcode, fast), 351.8341 (camb, kcap) |
| max rel. deviation of the 195-vector, test / off (camb backend, fast) | 4.3e-3 / 2.6e-3 (xi+ bin 0, largest ell) | median 9e-6 / 1.5e-5 |
| chi2 of the difference vector (package - mine) | 0.0002 / 0.0002 | |

So the default (`fast`) path with the CAMB non-linear backend agrees with an independent implementation to 6e-3 in chi2 at chi2 = 196
and 7e-2 at chi2 = 351 (the larger number is the residual times the difference vector; the difference vector itself has chi2 = 2e-4).
Differences that are *physics/modelling choices, not bugs*: stand-alone HMcode vs CAMB 0.013 (test) / 0.25 (off-centre, sigma8 = 0.93); `kcap` fidelity
vs exact bin average 0.03 / 1.15. Also reproduced independently: README amplitude scan (fast: 192.42/188.73/199.22/239.13, max |diff| 0.560;
kcap: 192.40/188.77/199.40/239.61, max |diff| 0.081; `runs/review/scan_check.py`).

Sensitivity of my own chi2 to the systematics switches at the two points: no m-correction 193.11 / 344.87, no c-term 197.18 / 350.66.

## 3. Hunts (task 3)

| hunt | script / log | result |
|---|---|---|
| **stale caches, random interleaving**: 90 distinct points (5 random prior points, box-edge points at omch2 = 0.34 +- 1e-13/1e-9, h = 0.64, ombh2 = 0.026, a point far outside the emulator box, each with n_s-only, A_s-only, A_baryon-only and fast-parameter variants), 300 evaluations in random order through ONE model (cache_size 3) vs a fresh model per point | `hunt_cache.py` | CAMB backend: max rel. deviation 2.9e-12 (0 of 300 above 1e-9). Emulator backend (83 emulator evaluations, 44 fallbacks, i.e. many emulator <-> CAMB transitions at the box edge): max rel. deviation 1.3e-15. **No stale-cache bug.** (The 1e-12 level is the documented amplification of the 7e-16 rescale difference by HMcode.) |
| **thread safety**: one shared model, 4 threads x 60 calls, 12 parameter vectors, `sys.setswitchinterval(1e-6)` | `hunt_threads.py` | `mu` correct in 240/240 calls (bitwise with 1 BLAS thread; ~1e-16 with 2) for CAMB and emulator backends, no exception. **But `model.last_derived` was a single shared attribute: wrong in 15/240 calls (6 %)** -> `DerivedRecorder` would store another thread's sigma8/Omega_m. Fixed (section 7); regression test `test_shared_model_threads_last_derived_is_thread_local` (mutation-checked: 61 wrong values without the fix). Remaining thread-unsafe pieces (documented below, not changed): `Posterior` counters/memo, `LinearBackend.stats`, `camb.config.ThreadNum` is a process global written before every CAMB call (two backends with different `nthreads` in two threads race), and `_sigma8_via_camb`/`_apply_state` run `calc_power_spectra` outside the backend lock (CAMB itself is not documented as re-entrant). The supported parallelism is processes (CLI uses `spawn`). |
| **fork safety**: parent evaluates the model (CAMB OpenMP + numba/TBB threads initialised), then `multiprocessing` `fork` child evaluates (a) a cached point and (b) a *new* cosmology (new CAMB run + HMcode `prange`) | `hunt_fork.py`, `hunt_fork2.py` | no hang, no crash; child result identical to the parent's (rel. diff 0.0 for the new cosmology; 1 ulp for a cached point, from multi-threaded BLAS summation order). Emulator artefact is loaded per process (10 MB npz) and also works after fork. Note this is one configuration (numba threading layer = tbb); `spawn` is what the CLI uses and what `test_e2e` exercises. |
| **prior corners**: all 64 vertices of the 6-d cosmological box (omch2, ombh2, h, n_s, ln As, A_baryon), each single parameter at lo/hi, all 32 sign patterns of dz = +-0.1 | `hunt_corners.py/.json` | 124/124 cases give finite ln L, no exception (`Posterior.last_error` never set); sigma8 spans 0.056 ... 6.0, Omega_m 0.044 ... 2.48 (omch2 = 0.99, h = 0.64 gives Omega_Lambda < 0 and is evaluated without complaint), chi2 from 266 to 2.2e8; <= 6.2 s per case. |
| **HMcode in extreme cosmologies** vs CAMB `mead` (32 vertices of omch2, ombh2, h, ln As x A_baryon, n_s = 1, 41-node grid) | `hunt_hm_extreme.py` | no NaN, no non-positive P_nl anywhere. Largest abs(P_hm/P_camb - 1) (z <= 2, k < 10) is <= 4.1e-3 (6.0e-3 over the whole grid; largest at Omega_m = 2.5, sigma8 = 3.8) at 28 of 32 vertices, ~3e-3 at the sigma8 ~ 0.3 vertices with Omega_m = 0.07. The 4 outliers (0.50-0.55, both h, both A_baryon) are the omch2 = 0.01, ombh2 = 0.026, ln As = 5 corner (baryon fraction 0.71): single k nodes sitting on a deep acoustic null where P_lin(k = 0.036) = 5e-4 against 20 at the neighbours (both codes fill the null with 0.07-0.1 of 1-halo power); the median abs(dP/P) there is 4e-5. Physical irrelevance (chi2 ~ 240 + at sigma8 = 0.3), not a bug. |
| **memory over 20 000 calls** (emulator backend, 2 % slow / 14 % medium / 84 % fast moves, `Posterior` + `DerivedRecorder`) | `hunt_mem.py` | RSS 226 -> 270 MB after the first 2500 calls (caches filling) -> 275 MB at 20 000: flat. Caches are bounded (stage-2 4, dz-shift cache 64, CAMB 8, emulator 8); `DerivedRecorder.table` grows by one entry per new cosmology (3047 entries in the run; capped at 2e6 entries which is O(1 GB) per process - note for very long chains). |
| **determinism**: `mu` hashes and a short fixed-seed blocked-Metropolis chain (15 + 10 cycles) | `hunt_repro.py` | identical SHA-256 of the `mu` vectors and of the chain (samples + ln P) for `NUMBA_NUM_THREADS` = 1 and 22 (emulator backend). CAMB backend with 1 vs 4 CAMB threads: `mu` hashes identical for the three test cosmologies; the fixed-seed chain has an identical accept/reject pattern and samples that agree to 7e-15 (relative), but `ln P` differs by up to 9e-9 (CAMB OpenMP summation order amplified by HMcode), so the SHA of the chain differs: runs are reproducible for a given seed and thread count, and to ~1e-10 across CAMB thread counts. The tiny CLI chains in `test_e2e` check seed determinism at the CLI level. |
| **numerical convergence knobs** at both cosmologies (CAMB backend, hmcode, fast) | `grid_sens.py` | spec 150-node vs reduced 41-node z grid: dchi2 = -0.002 (test) / +0.051 (off-centre, chi2 = 350); `n_nodes` 2000 -> 5000: +0.0001 / +0.0003; `n_ell` 400 -> 800: 0.0000 / 0.0000. Numerically converged; the reduced grid (used by the 41-node benchmark) is safe near the data (<= 0.05). |

## 4. Emulator (task 4)

* **Box check**: `TransferEmulator.contains` is inclusive (`lo <= v <= hi`) on (omch2, ombh2, h) = [0.03, 0.34] x [0.019, 0.026] x [0.64, 0.82]
  and requires `|mnu - 0.06| < 1e-12`; everything else (and `fallback=False` -> `ValueError`) goes to exact CAMB. Fallback output is
  *bit-identical* to `LinearBackend` (tested by `test_fallback_outside_box_and_mnu`, and by the interleaving hunt above). Crossing the
  box edge changes the spectrum by the emulator error (<= 1.2e-3 in P, 1e-3 in sigma8, see below), i.e. a discontinuity of that size in ln L of
  order 0.01-0.05 at worst; harmless for sampling, but the posterior is then formally defined by two different codes.
* **Claims vs held-out data**. Artefact metadata: n_train = 1435 (+ n_train_aug = 384 per the metadata), 256 held-out points, held-out max error of the
  chosen `polygp2` regressor 1.45e-3 in ln P_tot (2.3e-5 in the cold/total ratio). Pipeline validation (`runs/validate_emulator_pipeline.json`,
  50 Sobol points *inside* the box, n_s in [0.8, 1.2], amplitude chosen so S8 in [0.65, 0.85], chi2 184-335): max |dchi2| = 0.0077 -> this holds
  for **data-compatible amplitudes in the interior** only. My additional stress test (`hunt_emu_edge.py`): all 8 corners and 6 face centres of the box
  vs exact CAMB at the same (n_s = 0.97, ln As set for sigma8 = 0.75): ln P error (5 redshifts, k in 1e-4 ... 21/Mpc) 6e-5 ... 1.13e-3 (max at the two
  (omch2 lo, ombh2 hi) corners), sigma8 error 3.6e-5 ... 1.06e-3, chi2 differences between -0.019 and +0.046 (the +0.046 is at chi2 = 2977, i.e.
  1.5e-5 relative; at chi2 = 230-250 corners |dchi2| <= 0.004). Consistent with the stated 1.45e-3; no sign of edge blow-up.
* **Extrapolation guards**: none are needed inside the box (inputs are mapped to log-coordinates in [-1, 1]); outside the box the call never reaches the GP/polynomial.
  Not guarded: `n_s`/`A_s` (exact rescale, no emulation), `z` grids other than the two production grids (cubic spline in z of the emulated ln P; `ValueError` beyond z = 6).
* **Artefact versioning**: the npz stamps `camb_version = 1.5.8`, box, seeds, n_train/n_test, creation time; the installed CAMB is 1.5.8. Before this review
  the stamp was only checked in a unit test (and only up to the *major* version, "1" == "1"); the loader itself never compared it. Added a runtime
  `RuntimeWarning` when the installed CAMB differs (section 7). There is still no checksum of the artefact, and the CAMB *settings* that define the training target
  (high_precision, kmax = 20/Mpc padded to 21, `DoLateRadTruncation`, `WantCls = False`) are only recorded implicitly through `kmax_train`.
* **Documentation gaps**: `docs/PERFORMANCE.md` is cited by `emulator.py` and `linear.py` but does not exist; README said "No emulator is included" (fixed); USAGE/DESIGN do not mention
  the emulator and the CLI has no `--linear-backend` option (Python API only; `scripts/run_kv450_exact.py` uses it).

## 5. Documentation, dead code, API consistency (task 5)

| claim / item | status |
|---|---|
| README "No emulator is included" | **wrong** (emulator + artefact ship in the repo); corrected. |
| README/USAGE "OMP_NUM_THREADS ... CAMB/numba threads of one process" | **wrong for numba**: numba reads `NUMBA_NUM_THREADS` (default = all cores; measured 22 threads, TBB layer, with `OMP_NUM_THREADS=4`). Fixed in the CLI, USAGE corrected; the Python API still needs `NUMBA_NUM_THREADS` set by the user. Results are bit-identical for any thread count (hash test above). |
| README "`kidspipe check` ~7 s (first run adds ~10 s)" | measured 36 s for the first model evaluation of a fresh checkout on a loaded machine (JIT); not verifiable on a quiet machine here. |
| README validation table: 196.104 / 196.091 / 196.075, scan 192.42 ... 239.13, kcap 192.40 ... 239.61 | reproduced (196.1043, 196.0910; scan numbers exactly; 196.075 reproduced as hmcode+kcap = 196.0751; camb+kcap = 196.0618). |
| README "hmcode vs CAMB ... max abs diff in chi2 0.09" | true near the best fit (8 points); grows to 0.25 at the off-centre point (chi2 = 351, sigma8 = 0.93). |
| README / DESIGN model-vector contract | DESIGN.md says `CosmicShearModel.mu` "includes" delta_c / A_c; the implementation (and `interface_notes/params_data_likelihood.md`) returns the *pure* theory and the likelihood adds them. DESIGN.md also still describes `Posterior(model, likelihood)`, the n_s-keyed stage-1 cache, the `cl_shear(...)` signature and `docs/INTERFACE_NOTES.md` (now `docs/interface_notes/*.md`). All deviations are recorded in the interface notes; in DESIGN.md only the `mu` line was corrected in this review (F5), the other items are still stale. |
| `docs/interface_notes/cli_api_e2e.md` "n_s: SLOW vs MEDIUM" | resolved (section 7). |
| dead code | public-but-unused helpers only (no unreachable branches found): `ChainResult.all_samples`, `ParamBlocks.as_dict/from_indices`, `Posterior.n_evals/random_start`, `params.dz_from_theta`; used only by tests: `condition_number`, `null_chi2`, `chi2_explicit_inverse`, `equal_weight`, `load_flinc_theory`, `HankelBinner.xi_theta`, ... Harmless. |
| dependencies | `pyproject.toml` lists unpinned `numpy scipy numba camb`; `np.trapz` (2 uses) is deprecated in numpy 2.x, numba 0.59 caps numpy < 1.27, and CAMB 1.5.8 is required by the emulator artefact and the N_eff = 3.044 anchors. Pinning (`numpy<2`, `camb==1.5.8`) would match the tested environment. |
| every CLI option in README/USAGE exists in `cli.py` | yes (checked by script); 15 options are defined but undocumented (`--thin`, `--no-hessian`, `--f-live`, `--dynamic`, ...). |

## 6. m-bias, delta_c / A_c, pair ordering and cuts (task 6)

### 6.1 m-bias: is the shipped data vector already m-corrected?

What the package does: divides the data by (1+m_a)(1+m_b) with the fixed m = (-0.0128, -0.0104, -0.0114, 0.0072, 0.0061) (spec s9) and leaves the covariance untouched
(`GaussianLikelihood(apply_m=True)`, `--no-mcorr` switches it off).

Evidence available **inside the repository**:

1. Provenance is lost: the file is `data/kv450/xipm_data_vector.dat`, a renamed copy; it has no header/README. The spec (s9, s16 item 2, D28) says the original
   file name contains `mcor` ("m-corrected") and the covariance is named `..._mcorr`, which *suggests* the vector is already corrected - but that is a statement
   about names, not about the numbers.
2. A pure data test is weak: the effect of (1+m_a)(1+m_b) is <= 2.6 % on any pair and the per-pair S/N is modest, and the correction is nearly
   degenerate with an amplitude change (which is a free parameter) and with photo-z shifts.
   I profiled chi2 over a continuous exponent gamma in d_gamma = d_shipped / [(1+m_a)(1+m_b)]^gamma (gamma = 1: the spec's treatment, shipped file uncorrected;
   gamma = 0: shipped file already corrected, no further division), minimising over the nuisance and cosmological parameters with the emulator
   backend (`mbias_gamma.py`, `mbias_gamma2.py`; L-BFGS-B, so these are approximate optima):
   * all 14 parameters free, each gamma started from the other's optimum as well: **chi2(gamma=0) = 180.04 (ln posterior objective 180.73) vs chi2(gamma=1) = 183.89 (184.57): Delta = +3.8**
     in favour of *not* dividing again (spec s15.3, amplitude-only: +3.2). S8 at the two optima was not separated reliably by my optimiser (the optima found coincide to 4 digits; the spec's +-0.2 % stands).
   * an earlier, more restricted profile (ombh2 = 0.0222, h = 0.70 fixed, 12 free parameters, 3 random starts) gave chi2 = 176.0 (gamma = -1), 181.3 (0), 187.5 (1), 187.9 (2), i.e. a difference
     of 6.2 between gamma = 1 and 0 and a monotonic improvement towards *negative* gamma; the gamma = 1 -> 2 step is not convex, so at least one of these is a poor local optimum, and the 6.2 is
     superseded by the 3.8 above.
   Reading: the data mildly prefer the correction *not* to be applied a second time (Delta chi2 ~ 3.8 for 1 degree of freedom, ~1.9 sigma), but chi2 keeps falling towards unphysical gamma < 0, so the
   (1+m_a)(1+m_b) pattern (a ~2.3 % tilt between the low-z and high-z pairs) is partly degenerate with other redshift-dependent effects (photo-z, IA, baryon feedback) and the test is **not** a clean
   measurement of the shipped file's calibration state. It is, however, consistent with the file name suggesting an already corrected vector.
   (External, not in the repo: the published KV450 maximum-likelihood chi2 is 180.6; the emulator-based 14-parameter optimum above is 180.04 for gamma = 0 and 183.89 for gamma = 1 - suggestive, not conclusive, since the published analysis used a different pipeline.)
3. Covariance cannot discriminate: the shot-noise relation Var(xi+_ab)^2 ~ Var_aa Var_bb / 4 (checked by the earlier reviewer) is invariant under a common
   (1+m_a)^-2 (1+m_b)^-2 rescaling of the covariance.
4. The fixed-theory sensitivity (my independent chi2): applying the correction changes chi2 by +2.99 (196.10 vs 193.11) at the spec test cosmology and by
   +5.89 at the off-centre point (this point is far from the best fit, so the number is not meaningful as a model comparison).
   The spec's own amplitude-only experiment (s15.3): no correction S8 +0.2 %, chi2 -3.2; applied twice S8 -0.2 %, chi2 +3.5.

**Conclusion**: undecidable from repo data alone; the sensitivity is small (abs(Delta S8) ~ 0.2 %, abs(Delta chi2) ~ 3-4 vs 180 at the best fit). It should be settled from the
original KV450 data release notes (external; I recall that the public `mcor` vectors were corrected before release but did not verify this) and then either keep `apply_m=True` or set
`apply_m=False` for the default likelihood. Leaving it as is matches the spec (which says kcap divides the data).

### 6.2 delta_c^2 sign and ownership

* Implemented: `mu+ = B[xi+] + delta_c^2 + A_c^2 c+`, `mu- = B[xi-] + A_c^2 c-` with `PLUS_MASK` selecting the xi+ entries (7 of 13 per pair) - matches spec s9 ("montepython" branch).
* **Sign**: delta_c enters only as delta_c^2, so the sign is irrelevant (`delta_c = -6e-4` and `+6e-4` give identical chi2 = 196.0307, `hunt_corners.json`).
* **Squared vs linear is settled by the priors themselves**: with the prior N(0, 2e-4) a *linear* offset of 1 sigma (2e-4) added to xi+ (data values 1e-6 ... 1e-5) would change chi2 by +9e4
  (4e-4: +3.6e5; 6e-4: +8e5); the squared form changes chi2 by +0.005 / +0.065 / +0.31. Only the squared form is consistent with a prior of this width (physically a constant ellipticity
  offset c contributes c^2 to xi+). delta_c is therefore essentially unconstrained by the data (prior-dominated); A_c: chi2(0.62) = +0.49, chi2(1.4) = -0.72 relative to A_c = 1; c-term switched off: +0.80 (spec 0.8).
* **Ownership**: `CosmicShearModel.mu` returns the pure theory; `GaussianLikelihood` adds delta_c^2 and A_c^2 c; `Posterior` forwards `theta[A_c]`, `theta[delta_c]` (auto-detected
  keyword arguments, asserted by `api.build_posterior`). Risk: calling `GaussianLikelihood.log_like(model.mu(theta))` by hand silently uses A_c = 1, delta_c = 0 (defaults). All
  in-repo callers pass them explicitly or go through `Posterior` (grep of `src/`, `benchmarks/`, `scripts/`); `cli.py:749` passes `1.0, 0.0` deliberately in a timing benchmark.
* The c-term file columns (TreeCorr: R_nom, meanR, meanlogR, xip, xim, ...) are columns 3, 4 = xip, xim; meanR (0.78628, 1.59981, ...) agrees with the data theta column (0.78613, 1.5996, ...) to 2e-4, confirming that c and data use the same bins.

### 6.3 Pair ordering and cuts, independently

* Ordering: my script maps pair (j, i), j >= i, to data column `1 + i(2n-i+1)/2 + (j-i)` and to covariance blocks `[xi+(9), xi-(9)]` per pair directly from the spec text. Result agrees with
  the package (section 2), but that only proves both read the spec identically. Empirical checks of the *data*: (i) at the amplitude optimum (chi2 = 188.64) swapping the data blocks of any two
  pairs (105 transpositions) raises chi2 in 100 cases, by 0.2 to 322 (median +34, 84 of the 105 by more than 10); the 5 that lower it (by 0.4-3.1: pairs (1,1)<->(2,2), (1,0)<->(2,2), (1,0)<->(2,0), (3,0)<->(4,0),
  (1,0)<->(4,0)) all exchange pairs with similar signal and are at the level of the noise; (ii) rolling the xi+ theta bins by one position: chi2 = 562 (test cosmology) and 705 (off-centre);
  (iii) per-pair chi2 contributions at the test cosmology are 7.5 ... 21.6 for 13 points each (no pair block stands out; sum 196.1). (`ordering_check.py`, `ordering_check2.py`.)
* Cuts: xi+ bins 0-6 (theta <= 72.4'), xi- bins 3-8 (theta >= 4.2'), same for every pair, applied to data, theory, c-term and as row/column deletion to the covariance before inversion -> 15 x 13 = 195 (spec s10.3).
  Bin edges exp(linspace(ln 0.5, ln 300, 10)); the c-term file's R_nom column (0.7134, 1.4521, 2.9558) is exactly the geometric centres of those edges.

## 7. Fixes applied (each re-read before editing; behaviour of existing results unchanged)

| # | file | change | evidence / test |
|---|---|---|---|
| F1 | `inference/posterior.py`, `api.py` (docstring), `docs/interface_notes/cli_api_e2e.md` | `SLOW_PARAMS = (omch2, ombh2, h)`, `MEDIUM_PARAMS = (n_s, ln_1e10_A_s, A_baryon)` - identical to `api.BLOCKS` (approved cleanup; n_s is an exact rescale of the cached transfer function). Only the move classification of default-constructed `Posterior`s changes; `api.build_posterior` and the CLI already passed `api.BLOCKS`. | new test `test_inference_blocks_match_api_blocks` |
| F2 | `theory/model.py` | `last_derived` is a thread-local property; stage-2 / n(z)-shift caches are guarded by an `RLock`. No numerical change. | 15/240 wrong values -> 0/240; `test_shared_model_threads_last_derived_is_thread_local` (mutation-checked) |
| F3 | `cli.py`, `docs/USAGE.md` | `NUMBA_NUM_THREADS` is set from `OMP_NUM_THREADS` / `--threads` / `--worker-threads` (unless the user set it), because numba otherwise starts 22 threads per process. Results are bit-identical (SHA-256 of `mu` and of a fixed-seed chain for 1 and 22 threads). Processes that are already running are not affected. | `hunt_repro.py` |
| F4 | `theory/emulator.py`, `tests/test_emulator.py` | `RuntimeWarning` at load if the installed CAMB version differs from the artefact's stamp. | `test_camb_version_mismatch_warns` |
| F5 | `README.md`, `docs/USAGE.md`, `docs/DESIGN.md` | removed "No emulator is included"; corrected the numba thread statement; DESIGN.md `mu` now says "pure theory". | - |

## 8. Findings ranked

Severity scale: HIGH = could change a scientific conclusion or silently corrupt results; MEDIUM = wrong output/behaviour in a supported-but-rare use, or a materially wrong claim; LOW = hygiene.

| # | sev | finding | evidence | status |
|---|---|---|---|---|
| 1 | HIGH (open, scientific) | Whether the shipped data vector is already m-corrected is not decidable from the repo; the likelihood divides it by (1+m_a)(1+m_b) | section 6.1: profile over the exponent gamma: gamma = 0 (no second division) preferred by Delta chi2 = 3.8 (spec: 3.2), abs(Delta S8) ~ 0.2 % (spec); degenerate with other trends | **open** - needs external confirmation; no code change |
| 2 | MEDIUM | `CosmicShearModel.last_derived` shared between threads -> wrong sigma8/Omega_m/S8 in `DerivedRecorder` (and wrong derived columns) if a model is ever used from several threads | 15/240 wrong; mutation-checked test | fixed (F2). Other state (`Posterior`, `LinearBackend.stats`, `camb.config.ThreadNum`, `calc_power_spectra` outside the lock) is still not thread-safe: use processes |
| 3 | MEDIUM | numba ignores `OMP_NUM_THREADS`; every process starts 22 numba threads, contradicting the docs and the CPU budget on a shared machine | `numba.config.NUMBA_NUM_THREADS = 22` with `OMP_NUM_THREADS=4` | fixed for the CLI (F3); API users must set `NUMBA_NUM_THREADS` |
| 4 | MEDIUM | stale / unbacked documentation: README denied the emulator; `docs/PERFORMANCE.md` cited but absent; emulator not in USAGE/DESIGN/CLI; DESIGN.md model-vector contract contradicts the implementation; README timing "~7 s check" | section 5 | README/USAGE fixed (F5); the rest open |
| 5 | MEDIUM | emulator artefact only version-checked in a unit test (major version); no runtime guard, no checksum | section 4 | warning added (F4); checksum open |
| 6 | MEDIUM | non-editable install is broken: wheel contains no `data/` (incl. the emulator), emulator path ignores `KIDSPIPE_DATA` | section 1 | open (documented; README requires `pip install -e .`) |
| 7 | LOW | `SLOW_PARAMS` listed n_s | - | fixed (F1) |
| 8 | LOW | validation claim of the emulator (abs(dchi2) <= 0.0077) is for data-compatible amplitudes in the box interior; at corners/faces the worst is 0.046 (at chi2 = 2977) | section 4 | documented here |
| 9 | LOW | HMcode vs CAMB difference grows with sigma8 (0.013 at chi2 196, 0.25 at chi2 351); README quotes the near-best-fit figure | section 2 | documented here |
| 10 | LOW | unpinned dependencies (`np.trapz`, numpy 2), unused public helpers, `DerivedRecorder` cap 2e6 entries (~1 GB) | section 5 | open |
| 11 | LOW | `GaussianLikelihood.log_like(mu)` with default `A_c = 1, delta_c = 0` is a silent trap for external callers | section 6.2 | documented |
| 12 | INFO | no numerical, caching, corner-case or fork problem found (sections 2-3); independent chi2 agrees to 0.006 / 0.07 | | |
