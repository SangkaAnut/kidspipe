# kcap (KiDS Cosmology Analysis Pipeline): overview, SWOT and efficiency opportunities

Audience: a cosmologist who has never opened the repo. Source of truth: `reference/kcap` (git HEAD `36b40c55394caf26021408e71ff74538bfa3d51e`, 2022-06-13, shallow clone, 1777 files, ~186 MB). Evidence: the seven verified study reports `docs/study/01..07` (cited `[01]`..`[07]`), source re-reads, and measurements made while writing these three documents (marked **(M)**; independent numpy implementation in `docs/spec_checks/`, python CAMB 1.5.8, WSL2, 22 logical cores). Companion documents: `MASTER_MATH_SPEC.md` (re-implementation spec) and `VALIDATION_TARGETS.md` (reference files and numbers).

---------------------------------------------------------------------------
## 1. What kcap is, in one page

kcap is the inference pipeline behind the KiDS cosmic-shear and 3x2pt papers (KV450: Hildebrandt+20; KiDS-1000: Joachimi+21 methodology, Asgari+21 cosmic shear, Heymans+21 3x2pt, Troester+21 extended models; see `README.md`). It is **not a stand-alone program**: it is

1. a set of **CosmoSIS modules** (a Bitbucket fork of the CosmoSIS standard library: python-CAMB wrapper, Limber projection in C/GSL, nicaea Hankel transform in C/FFTW, `cosebis/` C++ for COSEBIs and band powers, P_gm approximation, BOSS and Planck interfaces) plus ~15 small python modules in `utils/` and `modules/` (A_s, S8, photo-z shifts, scale cuts, Gaussian likelihoods);
2. an **orchestration layer** of `.ini` files (`runs/config/*.ini`) and a 1387-line config generator `utils/run_kcap.py` that assembles every KiDS-1000 run variant (EE, EE_nE, EE_nE_w, xipm, cosebis, extended cosmologies) from one master dictionary;
3. batch scripts (`runs/…`), slurm helpers and chain post-processing (`utils/process_chains.py`, PJ-HPD credible intervals);
4. dead weight: a deprecated python package `kcap/`, MontePython data files, two old Fortran CAMB copies, CLASS, Planck likelihood code, 20 MB of COSEBIs tables (repo-level analysis in [07 s6.2]).

Physical task: given sampled cosmological and nuisance parameters, predict the cosmic-shear data vector (xi+/xi-, COSEBIs or band powers; plus galaxy-galaxy lensing and BOSS clustering for 3x2pt), and return a Gaussian log-likelihood `ln L = -chi2/2` with a constant covariance. Sampling (MultiNest, PolyChord, emcee, metropolis, maxlike) is delegated to CosmoSIS.

Data model: a CosmoSIS **datablock** = dictionary of (section, key) -> value, lower-cased on every access; modules run in the order of `[pipeline] modules`; any non-zero status gives ln L = -inf [01 s1.2].

---------------------------------------------------------------------------
## 2. Fiducial KV450 cosmic-shear run (`runs/config/KV450_fiducial.ini`)

### 2.1 Module chain and data flow

```
sample_ln_As -> one_parameter_hmcode -> camb -> extrapolate_power -> load_nz -> source_photoz_bias
 -> linear_alignment -> projection -> add_intrinsic -> cl2xi -> derived_parameters -> mini_2pt_like
```

| # | module (implementation) | job | reads -> writes |
|---|---|---|---|
| 1 | `sample_ln_As` (`utils/sample_ln_As.py`) | A_s = exp(ln10As) 1e-10 | `ln_1e10_A_s` -> `A_s` |
| 2 | `one_parameter_hmcode` | eta0 = 0.98 - 0.12 A_baryon | `halo_model_parameters/A` -> `eta0` |
| 3 | `camb` (python CAMB, `mode=transfer`) | background, linear + HMcode-2016 non-linear P(k,z) on 150 z x ~195 k, distances on 6000 z, sigma8, S8 | cosmology -> `matter_power_lin/nl`, `distances`, `omega_m` |
| 4 | `extrapolate_power` | log-log linear extension of P to k_h = 500 (199 extra nodes) | `matter_power_*` (in place) |
| 5 | `load_nz` | read 5 DIR n(z) histograms, shift to bin centres, prepend z=0, trapz-normalise | files -> `nz_KV450_5bin/*` |
| 6 | `source_photoz_bias` | n_i(z) -> n_i(z - dz_i), renormalise | `nofz_shifts/bias_i` |
| 7 | `linear_alignment` (`bk_corrected` NLA) | P_II = F^2 P_nl, P_GI = F P_nl, F = -A C1rho Omega_m / D(z) | `matter_power_nl`, `omega_m`, `A_IA` -> `intrinsic_power`, `matter_intrinsic_power` |
| 8 | `projection` (`project_2d`, C/GSL) | Limber: W, N kernels (1024-pt GL), 15 GG + 25 GI + 15 II spectra on 400 log ell in [0.1, 5e5] | -> `shear_cl`, `shear_cl_gi`, `shear_cl_ii` |
| 9 | `add_intrinsic` | C = GG + II + GI + IG, in place | `shear_cl` |
| 10 | `cl2xi` (nicaea FFTLog) | xi+/- on a fixed 1024-point log theta grid | -> `shear_xi_plus/minus` (radians) |
| 11 | `derived_parameters` | S8 | output only |
| 12 | `mini_2pt_like` | bin theory (node trapezoid, 9 bins 0.5'-300'), add delta_c^2 and A_c^2 c-term, divide data by (1+m_i)(1+m_j), cut to 195 points, chi2 with the analytic 195x195 covariance | -> `likelihoods/mini_2pt_like_like` |

Detailed formulas, grids and defaults: `MASTER_MATH_SPEC.md` s2-s11. Sections `[consistency]`, `[sigma8_rescale]`, `[cosebis]` of the ini are never run [01 s1.2].

### 2.2 Parameters and priors (14 varied)

| parameter | range [lo, start, hi] | prior |
|---|---|---|
| omch2 | 0.01, 0.10, 0.99 | flat |
| ombh2 | 0.019, 0.022, 0.026 | flat |
| h | 0.64, 0.70, 0.82 | flat |
| n_s | 0.7, 1.09, 1.3 | flat |
| ln(1e10 A_s) | 1.7, 3.09, 5.0 | flat |
| A_baryon | 2.0, 3.0, 3.13 | flat |
| A_IA | -6, 0, 6 | flat (Gaussian is commented out) |
| delta_c | +-6e-4 | N(0, 2e-4) truncated |
| A_c | 0.62, 1.13, 1.4 | N(1.01, 0.13) truncated |
| dz_1..5 | +-0.1 | N(0; 0.039, 0.023, 0.026, 0.012, 0.011), independent |

Fixed: Omega_k=0, w=-1, w_a=0, m_nu = 0.06 eV (1 massive state); m = (-0.0128, -0.0104, -0.0114, 0.0072, 0.0061) applied to the data. Truncated-Gaussian log-priors are normalised, so evidences are absolute [01 s2.1].

### 2.3 Data and size
5 tomographic bins, 15 pairs, 9 theta bins in [0.5', 300'] (log), 270 raw points; cuts drop xi+ bins 8-9 and xi- bins 1-3 -> **195 points**. Analytic covariance, no Hartlap, no log-determinant. Typical sampling cost quoted in the ini: 140k likelihood calls for a "quick" 250-live-point MultiNest, 350k for 500 live points, 1M for the "mega-run"; the shipped slurm script requests 40 MPI tasks for 2 days [01 s6].

---------------------------------------------------------------------------
## 3. KiDS-1000 3x2pt run (`utils/run_kcap.py`, committed example `runs/3x2pt/data_iterated_cov/.../multinest_blindC_EE_nE_w`)

Effective chain [06 s1.2]:
`sample_S8 -> sigma8toAs -> correlated_dz_priors -> one_parameter_hmcode -> camb -> wedges -> approximate_P_gm -> extrapolate_power -> load_nz_fits -> source_photoz_bias -> linear_alignment -> magnification_alphas -> projection -> add_intrinsic -> add_magnification -> bandpower_shear_e -> bandpower_ggl -> scale_cuts -> BOSS_like -> 2x2pt_like`.

* Probes: cosmic-shear E-mode **band powers** (8 log bands, l in [100,1500], 5 bins, 120 points), GGL band powers (2 BOSS lens bins x 5 source bins; 22 points kept after cuts), BOSS DR12 xi wedges (2 z-bins x 3 wedges, 168 points). "Clustering" is the BOSS redshift-space xi, not angular w(theta).
* Differences from KV450: sampled parameter S_8 (A_s derived by a **second CAMB call** `sigma8toAs`), fast shear+IA kernel (15 instead of 55 Limber integrals), ell 1..3e4, cubic photo-z interpolation, n(z) from FITS (`load_nz_fits`), photo-z shifts decorrelated by Cholesky of `SOM_cov_multiplied.asc`, P_gm approximation `b1 P_nl + (b2 N_b2 - gamma2 N_g2 - gamma3 N_g3) P_lin^2` with 3 bias parameters per lens bin, magnification (alpha = 1.8, 2.62), no Hankel transform (band powers computed in C++), no m/c nuisance in the likelihood [06 s2].
* Parameter count: EE 12 varied, EE_nE_w 20 varied (5 cosmology + A_baryon + A_IA + 5 p_i + 8 bias) [01 s2.5].
* Extensions by ini edit: Planck 2018 (clik) and CMB lensing, w, mnu, curvature (**inconsistent in the projection**, see W3), f(R) via external ReACT/HMx, HMcode-2020 feedback [06 s2.10].
* Data actually in the clone: the blind-C KiDS-1000 band-power FITS (142-point KiDS vector after cuts), BOSS measurements and covariances, `SOM_cov_multiplied.asc`. Not in the clone: the BOSS `wedges` model, other blinds, `deltaz.asc`, xi/COSEBIs real data [06 s1.1, s8].

---------------------------------------------------------------------------
## 4. Orchestration, samplers, post-processing  [01]

* Run directories are self-contained (data copied, `command.sh`, git-status snapshots). Run types are set operations on one master config (`choose_pipeline`).
* Samplers: MultiNest (fiducial 250 live points, efficiency 0.8, tolerance 0.1 for the "quick" run; production 500-1000 live points, efficiency 0.3, tolerance 0.01-0.5), emcee (72 walkers), metropolis (diagonal 1 %-of-range proposal, no tuning - a demonstration setting), maxlike (unit-cube Nelder-Mead; Powell/L-BFGS variants only in the unseen fork), dypolychord (fork only).
* Post-processing: getdist wrappers; PJ-HPD (projected joint highest posterior density; `stat_tools.py`) is the KiDS-1000 headline interval; burn-in 30 % for MCMC; O(N^2) implementation.

---------------------------------------------------------------------------
## 5. Strengths

1. **Clear module contract**: each physical step has a (section, key) interface; swapping Halofit/HMcode, IA, likelihood is an ini edit [01 s5, 07 s5].
2. **Reproducible run management**: self-contained run directories with provenance; run types encoded as data (`run_kcap.py`) [01 s5].
3. **Published, community-checked physics**: used for KV450 and KiDS-1000; independent cross-checks against CCL and a CLASS-based MontePython port exist in the tree [07 s5].
4. **Smart parametrisations**: ln A_s or S_8 sampling, one-parameter baryon model (eta0 = 0.98-0.12 A), Cholesky-decorrelated n(z)-shift priors (unit-Gaussian sampling in p), exact truncated-normal priors and inverse-CDF hypercube transforms for nested sampling [01 s5].
5. **Algorithmic good ideas**: fast shear+IA kernel (exact for NLA); log-FFT Hankel transform (one 1024-pt FFT per spectrum, verified against direct quadrature to ~1e-8..4e-6 [04]); band powers / COSEBIs as linear estimators with E/B separation; analytic top-hat g functions for band powers; correct Hartlap treatment for the BOSS mock covariance.
6. **Careful statistics**: PJ-HPD intervals; likelihood inverts the cut covariance (not a slice of the inverse); Cholesky solve in the MontePython port.
7. **Complete KV450 data package** shipped: data vector, analytic covariance, DIR n(z) and covariances, c-term file, test FITS.
8. **Wide scope**: shear, GGL, magnification, IA-GGL, BOSS, Planck, extended cosmologies, MPI.

---------------------------------------------------------------------------
## 6. Weaknesses

### 6.1 Correctness / reproducibility
* **W1 Silent option typos**: `halofit_verion = mead` in 5 shipped inis (`KV450_fiducial.ini:124`, `BOSS_KV450.ini`, 3 Flinc inis) is ignored; CAMB's default applies, and in CAMB 1.5.8 that is HMcode-2020, which **ignores** the baryon parameters. Measured effect: S8 +0.3 %, A_baryon becomes a prior-only parameter; A_baryon 3.13 -> 2.0 alone moves S8 by +4.6 % **(M)** [01 Q1, 02 4.1, 07 Q1].
* **W2 Unpinned environment**: no CAMB version recorded (N_eff 3.044 vs 3.046, BBN YHe, HMcode, k-grid), `conda_env.yaml` unpinned and lacks camb/getdist [02 s8, 07 s6.3].
* **W3 Curvature not implemented**: `D_M` is the radial distance, `K` is never passed to the Limber code or written by any module; non-flat chains use flat geometry (3 % error in chi at Omega_k=0.4) [06 Q1, 03 s1.2].
* **W4 Definitions**: sampled `S_8_input` uses Omega_cb, `sigma8toAs` is a separate low-precision CAMB call without the neutrino hierarchy of the main run: +0.07 % in sigma8 at 0.06 eV, 3.5 % at 1 eV, 16 % at 3 eV; verified that 'normal' with m_nu=1 eV gives 3 massive states while the default gives 1 **(M)** [06 Q2].
* **W5 Numerical noise**: adaptive Limber QAG at 1e-3 relative gives a C_l that is not smooth in the parameters (harmful for gradients, Fisher, HMC, Nelder-Mead) [03 Q17].
* **W6 Inconsistencies**: `delta_c` is squared in one likelihood branch and linear in another; A_c is squared in python and linear in C++; two C_l interpolators (log-log linear vs natural cubic) in one stack; `bin_xi_*` get `theta_max = theta_min`; `add_intrinsic` GG copy is a no-op **(source-verified)**; `cut_cross` is always true if present; dozens of further quirks are catalogued in [01]-[06].
* **W7 Possible double m-correction** (file named `mcor`, plus ini correction); effect +-0.2 % in S8 **(M)**; no m uncertainty marginalisation [05 Q6].
* **W8 Validity**: P_gm fit extrapolated far outside any documented range, no clipping of negative P_gm, no cosmology dependence beyond (n_s, omch2/h) [06 Q6-Q8]; P_nl extended by a 3-point power law to 500 h/Mpc [02 Q4.10]; xi used below the Hankel module's documented 1' validity [05 Q9].

### 6.2 Testing
No CI, no regression numbers, no assertions in the CCL / MontePython comparisons (plots with +-5..10 % axes); the only strict tests are the CAMB-interface rtol 1e-5 test and the HMcode test (DM-only) [07 s1.2]. The Flinc xi-vector is the only stored theory vector for the full chain.

### 6.3 Maintainability and portability (why a fresh clone fails)
Ranked hypotheses for the failed clone attempt, from [07 s6.3] plus checks:
1. **CMake >= 4 refuses the project**: `cmake_minimum_required(VERSION 3.2)` with `cmake_policy(SET CMP0002 OLD)` ("Compatibility with CMake < 3.5 has been removed", reproduced in [07]); `build.py` runs bare `cmake ..`.
2. **Wrong CosmoSIS**: README and `build.py` require the Bitbucket fork `tilmantroester/cosmosis@kcap`; the version check (>= 0.0.10) passes with upstream CosmoSIS 3.x, so the fork is never installed and the include path (`cosmosis/datablock/c_datablock.h`) mismatches (reproduced with gcc: fatal error) [07 6.3.1].
3. **Python 3.12 / NumPy >= 1.24**: `import pkg_resources`, `distutils`, `np.int` at `limber.py:8`, `unicode` at `project_2d.py:459`, `mini_2pt_like.setup` raises on NumPy >= 1.25 with `cut_xi_plus != []` [05 Q13].
4. **Over-ambitious build** (old Fortran CAMB, clik, JLA, ReACT/HMx, 123 MB of library) with FTP/HTTP downloads of GSL 2.4, FFTW 3.3.7, CFITSIO; missing `fftw3.h`; Fortran flags for gfortran >= 10 not set.
5. drvfs (`/mnt/d`) is slow/fragile for autotools builds; build in the Linux file system.
6. No LICENSE at the repo root (all rights reserved by default) - legal reason for the clean-room route.
Maintenance: last commit 2022-06-13; `kcap/` and `montepython/` are deprecated; hard-coded personal paths and e-mail; generated scripts reference data that is not in the repo [07 s6.2, 01 Q17, Q22].

### 6.4 Performance structure
Per likelihood call kcap runs CAMB (transfer + HMcode on 150 z), optionally a second CAMB, kernels (5 x 1024 x 1024 integrand evaluations), 55 (KV450) or ~45 (3x2pt) x 400 adaptive Limber integrals, 15 FFT Hankel transforms with ~30k complex-Gamma evaluations, a 30-spline binning loop and an RNG draw; the datablock copies P(k,z) several times. No timing exists in the repo [01 s6, 07 s7].

---------------------------------------------------------------------------
## 7. Opportunities: ranked efficiency improvements

Baseline measurements **(M)** on the independent implementation (4 CAMB threads): CAMB transfer + HMcode 1.00 s (85 %), kernels + Limber 0.16 s, Hankel + binning < 0.01 s, chi2 < 1 ms. kcap itself was never built here; its per-stage costs below the CAMB step are estimates from operation counts in the reports [03 s7, 04 s6.4, 07 s7].

| rank | opportunity | where | evidence | expected gain | accuracy risk / validation |
|---|---|---|---|---|---|
| 1 | **Pin CAMB's OpenMP threads (4-8) / one thread per MPI rank** | CAMB step | HMcode 4.1-4.5 s at 22 threads vs 0.5 s at 4 [02 s6]; whole CAMB step 3.7 s (22 thr) vs 1.00 s (4) vs 2.86 s (1) **(M)** | **3.7x on the dominant stage, zero code** | none |
| 2 | **Compute HMcode on ~40 z nodes and spline in z** (log P vs z) | CAMB step | 41 z nodes: 1.00 -> 0.65 s, chi2 188.668 vs 188.668 **(M)**; HMcode cost linear in nz [02 s7.2] | 1.5x now; up to 3x on HMcode part | validated to < 1e-3 in chi2 here; re-validate per model |
| 3 | **Fast/slow factorisation with a cached (ell, chi) P-grid**: 8 of 14 parameters (A_IA, delta_c, A_c, dz_1..5) never touch CAMB; `Q(l,chi)=P_nl((l+1/2)/chi, z(chi))` depends only on cosmology; nuisance moves are an einsum | sampler + projection | 8/14 fast [01 s7.3, 07 s7.1]; contraction cost 0.16 s -> ~ms by design | 100-1000x for a fast-only step; 2-5x effective on a PolyChord / oversampled MCMC | exact (algebraic); CosmoSIS stock `fast_slow` re-runs all modules after the first fast one, so restructuring is required [07] |
| 4 | **Replace QAG + 1024^2 kernels by cumulative-sum kernels and a dense fixed quadrature on a shared grid; fold GG+GI+II into one combined kernel (15 instead of 55 integrals)** | projection | kernels + Limber for all pairs 0.16 s, error 1.5e-4 (2000 nodes) vs 12000-node reference **(M)**; kcap's kernel stage is 5 x 1.05e6 integrand evaluations [07 s2.7] | est. 5-20x vs kcap (not measured, kcap not built); makes C_l smooth in parameters | converged to < 3e-4 with 2000 nodes; check against kcap-faithful mode |
| 5 | **Hankel + binning as one precomputed matrix** (cache FFT kernel; B @ xi, or bin-averaged Bessel operator) | cl2xi + likelihood | FFT stage already 5 ms **(M)**; kcap's python binning loop 28.7 ms vs 28 us matrix [05 s7.1]; 30,780 `cdgamma` calls per call avoided [07] | removes ~30-60 ms per call | exact (linear map); exact-edge variant changes chi2 by 0.04 **(M)** |
| 6 | **Likelihood housekeeping**: `keep_theory_vector=F` while sampling (SVD-based RNG draw, 5-17 ms), Cholesky once, drop dead spline, drop prints | likelihood | [05 Q11] | 5-30 ms per call | none |
| 7 | **Trim the grids**: 400 -> ~120 ell nodes (Delta chi2 = 0.01 **(M)**), kmax 20 -> 10 /Mpc (0.015 **(M)**), drop unused CAMB outputs (D_L, MU, rs_DV, F_AP, get_BAO, derived thermal parameters; background_nz 6000 -> ~500), drop the second CAMB call (2 %, but a correctness fix) | CAMB wrapper / projection | [02 s7.6], [01 s7.1] | 10-20 % | per-case convergence tests; CAMB warns for HMcode with kmax<20 and z>4 |
| 8 | **Sampler**: replace MultiNest's 1.4e5-1e6 evaluations by a tuned MCMC (MAP + Hessian proposal, 10^4-10^5 evaluations) or HMC/NUTS in decorrelated parameters (needs smooth C_l, item 4); L-BFGS instead of Nelder-Mead for the MAP | sampler | [01 s7.7] | 10-100x fewer evaluations | evidence is lost with MCMC; use nested sampling only when needed |
| 9 | **Emulators** for P_nl(k,z) (CosmoPower-style with A_baryon as input) or for the 195-vector | CAMB step | [02 s7.9] | 100-1000x per evaluation | 0.5-1 % emulator error must be << 0.1 sigma of the data vector; keep CAMB as oracle |
| 10 | **Linear-map COSEBIs / band powers** (precomputed Q matrices; vectorised Bessel tables; cache tables as `.npy` keyed by all inputs) | cosebis/ | [04 s7.1-7.2] | est. 50-100x on a 0.3-1 s step; first-run table build from hours to ~1 min | verify with `Benjamins_Bandpower` file and T tables |
| 11 | **Post-processing**: PJ-HPD O(N^2) -> O(N log N) | stat_tools | [01 s7.10] | 100-1000x for N = 1e5 | exact |
| 12 | **Threaded, shared-memory evaluation** (no datablock copies; vectorised batch evaluation for emcee/PolyChord) | architecture | [01 s7.8] | 1.2-2x | none |

Combined plausible outcome: cold evaluation 3.7 s (default threads) -> ~0.5-0.6 s (items 1, 2, 7 with CAMB retained), nuisance-only moves in milliseconds (item 3), emulator-based evaluations ~10 ms (item 9). Items 1-3, 5, 6 are exact; 4 and 7 need convergence tests; 8-9 change the statistical procedure.

---------------------------------------------------------------------------
## 8. SWOT summary

| | Helpful | Harmful |
|---|---|---|
| **Internal** | **Strengths:** modular datablock contract; provenance-rich runs; peer-reviewed physics; fast shear+IA kernel; log-FFT Hankel; linear-estimator COSEBIs/BP; complete KV450 data package; wide probe coverage | **Weaknesses:** silent option typos (halofit); unpinned CAMB; curvature unsupported; S8 proxy mismatch; QAG noise; delta_c/A_c convention clashes; unit/ordering traps; no regression numbers or CI; datablock copies; HMcode dominates cost; possible double m-correction |
| **External** | **Opportunities:** clean-room Python pipeline (this project) with exact fast paths; threading discipline (3.7x); fast/slow caching (100x for nuisance steps); smooth C_l enabling gradient samplers; emulators; reuse of the published constraints (S8 = 0.737, chi2 = 180.6) as acceptance tests | **Threats:** build rot (CMake 4, NumPy 2, Python 3.12, FTP URLs, Bitbucket fork availability); no licence; CAMB drift changes results silently (mead2020); the BOSS module and several data products are absent so 3x2pt cannot be reproduced from the repo; last maintenance 2022 |

---------------------------------------------------------------------------
## 9. Recommended path for the user's rebuild (consistent with the three documents)

1. Build a clean-room python pipeline from `MASTER_MATH_SPEC.md`: CAMB (pinned) -> optional own HMcode -> (E) kernels/Limber -> FFT Hankel -> binning matrix -> chi2. The executable witness `docs/spec_checks/kv450_spec_crosscheck.py` shows the architecture works (chi2 188.7, 1.2 s per cold evaluation including CAMB).
2. Provide two numerics modes: `kcap-faithful` (node-trapezoid binning, slope +1/-3 extrapolation, k<=500 cut, etc.) and `exact` (exact bin edges, ln P interpolation, smooth quadrature), and test both against `VALIDATION_TARGETS.md` section 6.
3. Run threading discipline and fast/slow caching from day one; sample with a covariance-tuned MCMC first, nested sampling only for evidence.
4. Before trusting any number, record CAMB version and `halofit_version` in every output header.

---------------------------------------------------------------------------
## Appendix A. Minor findings worth knowing
* `KV450_fiducial.ini` `kmax=20` is in 1/Mpc and CAMB pads by 1.05, so k_h,max = 21/h (30.0 at h=0.7; 25.6-32.8 over the prior) [02 3.3].
* MultiNest ini values differ between files (0.8/0.1, 0.3/0.5, 0.3/0.01); `--sampler-config` booleans are strings (any non-empty string is true, `max_posterior ''` -> F) [01 Q16].
* The `burn=0.3` emcee option is never read; emcee chains are walker-interleaved and burned by 30 % of rows [01 Q15].
* `D_M` radial vs transverse; `omega_lambda` includes radiation; sigma8/S8 are taken at the first z-grid node, equal to z=0 only if `zmin=0` [02 s4].

## Appendix B. Reconciliation log: contradictions between the study reports and how they were resolved

| # | contradiction | resolution (evidence) |
|---|---|---|
| R1 | [07 s2.8] says `add_intrinsic` saves the GG term in `shear_cl_gg`; [01] and [03 Q6] say it does not | **Source**: `add_intrinsic.py:52` sets `shear_shear_gg = sec_names['shear_shear']` (`shear_cl`), lines 81 and 86 copy `shear_cl` onto itself -> no `shear_cl_gg` is created. [01]/[03] right, [07 s2.8] wrong. No effect on ln L. |
| R2 | CAMB k-grid size: 194 [02, 01], 197 [01 s3.2, 06], 215 or 197 [07] and k_h,max 28.6 vs 30.0 | **Run**: nk=194 at (omch2, ns) = (0.10, 1.09) and 197 at (0.12, 0.97); k_h,max = 1.05 kmax/h = 30.0 at h=0.7; 215 nodes is `WantCls=True` for the test_pofk cosmology (h=0.6796, k_max 30.9); the number depends on cosmology and `WantCls`. The 28.6 figure was a mis-statement later corrected in [01]. |
| R3 | [02]: 'normal' hierarchy at mnu=0.06 is inert / 2 eigenstates at 0.3 eV; [06 Q2]: 'normal' = three massive eigenstates | **Run**: 'normal' with mnu=1 eV -> 3 massive neutrinos (2 eigenstates: degeneracies 2.029 + 1.015); default hierarchy -> 1 eigenstate (1.015). Both reports correct at their own mnu; the sigma8toAs (default) vs main (normal) mismatch grows with mnu and is irrelevant at 0.06 eV. |
| R4 | [01 s7.1], [06 s7.5]: removing the `sigma8toAs` call saves 20-40 % | Corrected in the reports after measurement (2 % of CAMB time, 0.07-0.15 s vs ~4.6 s); it is a correctness fix, not a speed-up. |
| R5 | Number of fast parameters 9 [early 01] vs 8 [07] | `halo_model_parameters/A` is read inside `camb_interface.py:225-233` -> slow; 8 fast: A_IA, delta_c, A_c, dz_1..5. |
| R6 | [03 s1.1/Q5] versus [07 s7.3 (original)]: n(z) support ends at z~2 | DIR files are non-zero up to z=4.45-4.65; the lensing kernel covers 92 % of chi range; corrected in [07]. |
| R7 | `delta_c` squared or not: [01 Q9], [05 Q4], [07 Q5] | **Source** `mini_2pt_like.py:187` (linear, `xi_pm-bin-theta`) vs `:226` (squared, `montepython`); the fiducial ini uses `montepython`. |
| R8 | Stored `pk_mead.txt` provenance: eta=0.603 [script/values.ini] vs eta0=0.74 [07] | **Run**: CAMB 1.5.8 with A=2.0, eta=0.74 reproduces the array to 0.82 % max / 0.30 % rms (k>=1.1e-4, <30 h/Mpc); eta=0.603 deviates by up to 20 %. [07] right. |
| R9 | Hankel accuracy statement "1e-7 relative" [04 original] | Re-run: 8e-9 (mu=0, 1'), 6e-7 (mu=4, 1'), 4e-6 (mu=4, 10'); corrected in [04]. |
| R10 | `SOM_cov_multiplied.asc`: KV450 copy vs KiDS-1000 copy numerically different | Both exist (max entry difference 2.1e-5, sqrt(diag) 0.0096.. vs 0.0106..); KV450 fiducial uses neither (it uses `DIR_cov` widths). |
| R11 | HMcode/halofit default behind the typo: "mead" assumed in older text | Depends on CAMB version; on 1.5.8 the default is mead2020 (verified by instantiation); intended HMcode-2016 requires the key `halofit_version = mead`. |
| R12 | Which KiDS data are in the clone | [06] corrected [05/07 early claims]: the blind-C band-power FITS and BOSS files are present; xi/COSEBIs real data and `deltaz.asc` are not. |
