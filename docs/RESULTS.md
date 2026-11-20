# KV450 cosmic-shear analysis with `kidspipe`: results

All numbers below come from the runs in `runs/kv450_final/` (scripts: `scripts/run_kv450_*.py`). Configuration of every run unless stated
otherwise: transfer-function **emulator** (`linear_backend='emulator'`, box omch2 [0.03, 0.34], ombh2 [0.019, 0.026], h [0.64, 0.82]),
**HMcode-2016** non-linear spectrum, **`fast`** Hankel fidelity, **spec z grid** (150 nodes), m-bias correction on, c-term on, flat
prior box and truncated-Gaussian nuisance priors of `kidspipe.params`. 195 data points, 14 parameters (181 degrees of freedom).
Numbers marked *(external)* are published KV450 values that are not in the repository (as quoted in the task / `docs/VALIDATION_TARGETS.md`);
they were not independently checked.

Short version:

* The likelihood is extremely flat. 36 optimisations from diverse starts all end at chi2 = 181.9 - 182.9 (single plateau, no second mode);
  the best point (chi2 = 181.874, MAP of the posterior) sits on **four prior bounds** (ombh2 = 0.019, h = 0.82, A_baryon = 2, ln_1e10_A_s = 4.92 vs bound 5).
  chi2 changes by only 1.3 over the whole ln_1e10_A_s range 2 ... 5.
* The posterior is converged (14 chains, 272,840 post-burn samples, max split R-hat 1.017, min ESS 1626) and independently confirmed by nautilus nested sampling
  (all 17 marginal means within 0.04 sigma of the MCMC). S8 = 0.7358 +- 0.0374.
* It agrees with the published KV450 values (S8 = 0.737 +0.040/-0.036, Omega_m = 0.256, A_IA = 0.98 +- 0.69, A_baryon = 2.48) within a few percent of
  the statistical error. Sigma8, Omega_m, omch2, ombh2, h, ln_1e10_A_s are not constrained individually: their marginals are shaped by the prior box (projection / prior-volume effects, section 2.5).
* The emulator is not a limiting factor: exact CAMB changes ln L by <= 1.1e-3 on 200 posterior samples, and the posterior has no mass outside the emulator box.

## 1. Global maximum a posteriori (MAP)

**Procedure.** (a) *Explore*: 36 starts (`scan`, `start`, `test` presets, the boundary solution of the earlier run, and 32 random draws:
cosmology uniform over the emulator box for 3/4 of them and uniform over the whole cosmological prior box for 1/4, nuisance parameters from their priors),
bounded L-BFGS-B (200 iterations x 2 rounds, finite-difference gradients) on the **41-node z grid** (cheaper; chi2 differs from the spec grid by ~1e-3, section 4)
- 13 processes, 1735 s, 241,401 posterior evaluations. (b) *Polish* of the best candidates on the **spec z grid** (L-BFGS-B, 6 candidates; then L-BFGS-B + Powell, 3 candidates, 1733 s).
(c) *Exact check*: the polished points re-evaluated with exact CAMB (`linear_backend='camb'`, everything else identical).

**Explore stage (36 starts, `runs/kv450_final/map_explore.json`).** Best chi2 181.890, worst 182.864, median 182.112; 7 starts within 0.1 and 26 within 0.3
of the best, 34 within 0.7. All 36 end in the same valley (S8 = 0.769 - 0.783, Omega_m = 0.110 - 0.195, sigma8 1.0 - 1.3), with the optimiser stopping at different places
along it (omch2 0.046 - 0.069, ombh2 0.0192 - 0.0259, h 0.68 - 0.82, ln_1e10_A_s 4.1 - 5.0): the chi2 differences along the valley are below 1, so L-BFGS-B stops on the plateau.
No second basin was found (best-to-worst spread in ln P: 0.53).

**Best fit** (`runs/kv450_final/map_starts.json`, `map_best.json`; polished with Powell, spec grid; three independent polishes agree to 0.003 in chi2):

| quantity | value |
|---|---|
| chi2 of the data (195 points) at the MAP | **181.874** (ln L = -90.937; ln posterior = -63.969) |
| chi2 + Gaussian-prior chi2 of the nuisance parameters (0.356) | 182.230 |
| exact CAMB at the same point | 181.8742 (emulator 181.8741; differences <= 2e-4 for all 6 + 3 candidates checked) |
| external published ML chi2 (195 pts, 14 params) | 180.6 *(external)* |
| sigma8, Omega_m, S8 | 1.294, 0.110, 0.783 |

Parameters at the best fit and distance to the prior bounds (fraction of the full prior range):

| omch2 | ombh2 | h | n_s | ln_1e10_A_s | A_baryon | A_IA | delta_c | A_c | dz1 | dz2 | dz3 | dz4 | dz5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0542 | **0.01900** | **0.8200** | 1.143 | 4.923 | **2.000** | 0.750 | 0 | 1.023 | -0.0020 | 0.0116 | -0.0074 | -0.0010 | 0.0002 |
| 4.5 % from lower | **0 (lower bound)** | **0 (upper bound)** | 74 % | 2.3 % from upper | **0 (lower bound)** | 56 % | 50 % | 52 % | | | | | |

**The optimum lies on the prior bounds** (ombh2, h, A_baryon exactly, ln_1e10_A_s within 2.3 % of the range; the chain of three Powell polishes also moved
ln_1e10_A_s between 4.91 and 5.00 with chi2 changing by 0.003). It is inside the emulator box (omch2 = 0.054; ombh2 and h are on the box edge, which is closed and coincides with the prior bound), so
no exact-CAMB optimisation was needed. The earlier single-start result (chi2 = 181.85, ombh2 = 0.0204, ln post = -64.008) is a different point on the same plateau; its ln posterior
(-64.008) is lower than the present one (-63.969), its data chi2 happens to be 0.02 lower because the nuisance priors weigh differently.

**chi2 split at the best fit** (exact CAMB; `chi2_split.json`): xi+ 101.85 (105 points), xi- 80.02 (90 points); null chi2 (zero model) 269.30. By tomographic pair (13 points each):

| pair | 1x1 | 1x2 | 1x3 | 1x4 | 1x5 | 2x2 | 2x3 | 2x4 | 2x5 | 3x3 | 3x4 | 3x5 | 4x4 | 4x5 | 5x5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chi2 | 17.27 | 8.35 | 11.16 | 12.13 | 10.71 | 21.52 | 7.13 | 18.07 | 10.16 | 17.92 | 8.15 | 14.91 | 13.78 | 6.61 | 4.03 |

Nuisance-prior chi2 at the MAP: A_c 0.011, dz1 0.003, dz2 0.253, dz3 0.082, dz4 0.007, dz5 0.0002, delta_c 0 (total 0.356). The pair split sums to 181.87 (the cross-pair covariance is distributed with r_i (C^-1 r)_i).

**About the published 180.6.** Our MAP is 1.27 above it. Two further numbers help to place it, but the repository does not say which quantity the published number is:
(i) the lowest chi2 among the 272,840 posterior samples is 180.38; (ii) dropping the nuisance priors (maximum *likelihood* with flat priors inside the bounds, `run_kv450_ml.py`, 6 starts all converge to the same point)
gives chi2 = **177.33**, but at an unphysical point (dz5 = -0.1, dz4 = -0.089, A_c = 1.4, A_IA = 2.8) whose Gaussian-prior chi2 is 162.1, so it is not comparable. A chi2 of 180.6 lies between the
posterior-maximum value and the unconstrained maximum; we cannot reproduce or exclude it without knowing the definition. We do not claim to match it better than "within ~1 in chi2, with a plateau this flat".

**Profiles (how flat the degeneracy is)** (`profile.json`; each point re-optimised on the 41-node grid, chi2 re-evaluated on the spec grid; nuisance priors included in the optimised function, chi2 = data part):

| ln_1e10_A_s | 2.0 | 2.4 | 2.8 | 3.2 | 3.6 | 4.0 | 4.4 | 4.8 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|
| chi2 | 183.16 | 182.93 | 182.68 | 182.44 | 182.22 | 182.04 | 181.93 | 181.875 | 181.877 |
| S8 | 0.741 | 0.754 | 0.765 | 0.772 | 0.778 | 0.781 | 0.783 | 0.783 | 0.783 |
| Omega_m | 0.330 | 0.277 | 0.235 | 0.201 | 0.173 | 0.150 | 0.130 | 0.114 | 0.107 |

Over ln_1e10_A_s = 2 ... 5 (a factor 20 in A_s, sigma8 from 0.71 to 1.31, Omega_m from 0.33 to 0.11) chi2 changes by **1.3**. Profile in S8 (penalty method, sigma 0.002):

| S8 | 0.55 | 0.60 | 0.65 | 0.70 | 0.725 | 0.75 | 0.775 | 0.80 | 0.825 | 0.85 | 0.875 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chi2 | 207.27 | 196.68 | 188.42 | 183.67 | 182.74 | 182.31 | 181.96 | 182.01 | 183.01 | 185.00 | 187.93 | 191.95 |

Interpolating, the profile-likelihood interval at Delta chi2 = 1 above the MAP is S8 = 0.783 (+0.039, -0.063) (0.720 - 0.822). The data constrain S8, not (sigma8, Omega_m) separately.

## 2. Converged posterior

### 2.1 Sampler and runs

Blocked adaptive Metropolis (`scripts/run_kv450_mcmc.py`, own driver around `kidspipe.inference.mcmc.blocked_metropolis`, run with the emulator, one core per chain):
one cycle = 1 joint update of (omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon) + 1 update of (n_s, ln_1e10_A_s, A_baryon) + 8 updates of the eight fast parameters.
*Pilot*: 6 adaptive chains (1500 burn + 1500 cycles, started at 6 explore-stage optima spread in Omega_m) -> pooled covariance + block scales (the MAP Hessian is not used: the MAP is on boundaries).
*Production*: fixed-covariance Gaussian proposals (pilot covariance, conditional covariance per block); 600 discarded tuning cycles per chain (Robbins-Monro on the block scales, then frozen),
so the stored chain is an exact Markov chain. Each chain starts at the end of a pilot chain.

| run | chains | cycles/chain | wall | joint-block acceptance |
|---|---|---|---|---|
| `prod1` (seed 21) | 6 | 15,000 - 16,600 | 80 min | 0.20 - 0.45 |
| `prod2` (seed 22) | 8 | 22,400 - 25,000 | 110 min | 0.21 - 0.49 |

Total 287,200 cycles; the first 5 % of each chain is discarded -> **272,840 samples** used. About 0.28 s per cycle per chain (a new cosmology costs ~0.2 s single-threaded, dominated by HMcode;
the emulator itself is ~25 ms). Chains 6 and 7 of `prod2` re-use the start points of chains 0 and 1 (only 6 pilot chains existed); they differ in seed and tuning.

### 2.2 Convergence (`combined_prod1_prod2/analysis.json`, `analysis_prod12.txt`)

Split R-hat and tau on the 14 chains truncated to 14,250 steps each; ESS = (all post-burn samples) / tau.

* **max split R-hat = 1.017 (h), 1.017 (sigma8), 1.016 (ln_1e10_A_s), 1.015 (Omega_m), 1.013 (omch2)**, all others <= 1.006; rank-normalised R-hat maximum 1.018. Target < 1.02: met.
* **min ESS = 1626 (ln_1e10_A_s)**, 1700 (sigma8), 1717 (omch2), 1771 (h), 1827 (Omega_m), 2455 (ombh2), 4979 (n_s), 14,136 (A_baryon), 16,513 (S8), >= 44,000 for the other nuisance parameters. Target > 1000: met.
  Integrated autocorrelation time 111 - 168 cycles for the degenerate directions, 16.5 cycles for S8.
* The 14 chain means of S8 lie in 0.7339 - 0.7383 (spread 0.0013 = 3.5 % of the posterior sigma). `prod1` alone (6 chains) already gave R-hat <= 1.016, ESS >= 527.
* Caveat: the chains started from points spread along the plateau but not from a deliberately over-dispersed ensemble; the independent nested-sampling run (section 3) supports the result.

### 2.3 Emulator box and verification

* Posterior samples (272,840): omch2 in [0.0394, 0.2736], ombh2 and h cover their full prior (the emulator box edge coincides with the prior bound for these two). **Samples outside the box: 0**,
  i.e. the posterior mass outside the emulator box is < ~2e-3 at 95 % confidence given ESS ~ 1700 (0 of 272,840 counted). The upper side (omch2 > 0.34, 66 % of the prior volume) is 4.5 sigma above the posterior mean (largest sampled omch2 0.274). During pilot/exploration a handful of proposals
  (3 - 10 per chain) outside omch2 < 0.03 were served by exact CAMB automatically (`stats['fallback']`); none are in the production chains. No retraining was needed.
* **Exact-CAMB re-evaluation of 200 random thinned samples** (`exact_check_prod1_prod2.json`; same HMcode + fast + spec grid, only the linear stage differs): Delta ln L = ln L_emu - ln L_exact: mean +7e-5, sd 2.1e-4, max |Delta ln L| 1.1e-3 (all 200 in-box).
  At the 9 MAP candidates |Delta chi2| <= 2e-4. Compared with the posterior width this is negligible, so the emulator posterior is the exact posterior to well within the Monte-Carlo error.

### 2.4 Marginal posteriors (mean, std, median, equal-tail 68 %, 1-D HPD 68 %, projected joint-HPD 68 % of the 14-D posterior, PJ-HPD over (omch2 ... dz5, sigma8, Omega_m, S8))

| param | mean | std | median | 68 % (16, 84) | 1-D HPD 68 % | PJ-HPD 68 % | R-hat | ESS |
|---|---|---|---|---|---|---|---|---|
| omch2 | 0.1176 | 0.0495 | 0.1089 | 0.0655, 0.1733 | 0.0475, 0.1417 | 0.0394, 0.2736 | 1.013 | 1717 |
| ombh2 | 0.02236 | 0.00201 | 0.02230 | 0.02002, 0.02475 | 0.01902, 0.02358 | 0.019, 0.026 | 1.006 | 2455 |
| h | 0.7449 | 0.0498 | 0.7508 | 0.6858, 0.8001 | 0.7203, 0.82 | 0.64, 0.82 | 1.017 | 1771 |
| n_s | 1.025 | 0.141 | 1.028 | 0.871, 1.178 | 0.880, 1.186 | 0.70, 1.30 | 1.005 | 4979 |
| ln_1e10_A_s | 3.180 | 0.933 | 3.119 | 2.108, 4.280 | 1.7, 3.701 | 1.7, 5.0 | 1.016 | 1626 |
| A_baryon | 2.507 | 0.320 | 2.479 | 2.145, 2.889 | 2.0, 2.682 | 2.0, 3.13 | 1.001 | 14136 |
| A_IA | 0.924 | 0.733 | 0.938 | 0.215, 1.633 | 0.251, 1.667 | -1.882, 3.445 | 1.001 | 44115 |
| delta_c (1e-4) | 0.0028 | 1.98 | 0.0047 | -1.99, 2.00 | -1.97, 2.02 | -5.85, 5.89 | 1.000 | 78339 |
| A_c | 1.025 | 0.129 | 1.025 | 0.895, 1.154 | 0.895, 1.154 | 0.636, 1.399 | 1.000 | 78284 |
| dz1 | 0.0060 | 0.0360 | 0.0067 | -0.0306, 0.0426 | -0.0297, 0.0434 | -0.0998, 0.1000 | 1.000 | 66335 |
| dz2 | 0.0101 | 0.0209 | 0.0102 | -0.0105, 0.0309 | -0.0100, 0.0313 | -0.0531, 0.0721 | 1.000 | 74948 |
| dz3 | -0.0110 | 0.0215 | -0.0110 | -0.0323, 0.0102 | -0.0325, 0.0100 | -0.0786, 0.0549 | 1.000 | 70250 |
| dz4 | -0.0016 | 0.0118 | -0.0016 | -0.0133, 0.0102 | -0.0134, 0.0100 | -0.0367, 0.0356 | 1.000 | 75947 |
| dz5 | 0.0008 | 0.0109 | 0.0008 | -0.0100, 0.0116 | -0.0099, 0.0117 | -0.0319, 0.0353 | 1.000 | 77002 |
| **sigma8** | 0.840 | 0.168 | 0.826 | 0.654, 1.030 | 0.599, 0.957 | 0.505, 1.354 | 1.017 | 1700 |
| **Omega_m** | 0.2557 | 0.0929 | 0.2405 | 0.158, 0.360 | 0.124, 0.305 | 0.105, 0.593 | 1.015 | 1827 |
| **S8** | **0.7358** | **0.0374** | 0.7369 | 0.6988, 0.7728 | 0.7009, 0.7746 | 0.607, 0.871 | 1.002 | 16513 |

S8 further: 95 % equal-tail 0.6595 - 0.8061, 95 % HPD 0.6615 - 0.8078, histogram mode 0.7405. The PJ-HPD of S8 (0.607 - 0.871) is much wider than the marginal interval because the projection of the 68 %
joint-HPD region of a 14-D posterior with a 4-parameter prior-dominated degeneracy includes the whole valley; it is not comparable with the marginal intervals. (The PJ-HPD of the remaining parameters just reproduces the prior
ranges for ombh2, h, ln_1e10_A_s, n_s, A_baryon.) chi2 over the posterior: mean 186.92, min 180.38 (a sample, not the MAP), sd 2.74; mean(chi2) - chi2_MAP = 5.05.

### 2.5 Comparison with the published KV450 values *(external)*

| quantity | this work (mean +- std; 68 % equal-tail) | published *(external)* | difference |
|---|---|---|---|
| S8 | 0.7358 +- 0.0374 (-0.0370, +0.0370) | 0.737 +0.040 -0.036 | -0.0012 (0.03 sigma) |
| Omega_m | 0.256 +- 0.093 | ~0.256 | 0.000 |
| A_IA | 0.92 +- 0.73 | 0.98 +- 0.69 | -0.06 (0.08 sigma) |
| A_baryon | 2.507 +- 0.320 | 2.48 | +0.03 |
| h, ln_1e10_A_s, n_s, A_c (README table, external) | 0.745, 3.18, 1.025, 1.025 | 0.745, 3.158, 1.021, 1.022 | all within 0.03 sigma |

**They agree within the statistical error** for S8, Omega_m, A_IA, A_baryon (and the others quoted in the README). Caveats, stated plainly:

* The agreement of the *means* of weakly constrained parameters (Omega_m, h, ln_1e10_A_s) is a statement about the shape of the prior box: with flat priors on ln_1e10_A_s in [1.7, 5], omch2 in [0.01, 0.99],
  h in [0.64, 0.82], ombh2 in [0.019, 0.026], the marginal posteriors of these parameters are mostly prior-shaped and agree with another implementation of the same prior by construction rather than by data content.
  The likelihood alone prefers the corner (omch2 = 0.054, h = 0.82, ombh2 = 0.019, ln_1e10_A_s ~ 5, sigma8 = 1.3, Omega_m = 0.11, S8 = 0.78) that carries a tiny prior volume.
* **Projection / prior-volume effect on S8**: the posterior marginal mean S8 = 0.736 is 0.047 (1.3 sigma) *below* the best-fit / profile-likelihood S8 = 0.783 (profile Delta chi2 = 1 interval 0.720 - 0.822), because the prior volume at fixed S8 shrinks towards
  the high-sigma8 / low-Omega_m corner. The published number (S8 = 0.737) is also a marginal posterior; a comparison of *best fits* would be unreliable (plateau, section 1).
* sigma8 (0.84 +- 0.17), Omega_m, h, ombh2, omch2, ln_1e10_A_s are not independently constrained; their quoted intervals mainly reflect the prior box, and sigma8-Omega_m are anticorrelated at -0.93.
* The nuisance parameters are prior-dominated: the dz_i, A_c, delta_c posteriors reproduce their priors to within 0-17 % in width (std/sigma_prior: dz1 0.92, dz2 0.91, dz3 0.83, dz4 0.98, dz5 0.99, A_c 0.99, delta_c 0.99) and 0.5 sigma_prior in the means (dz2 +0.0101 = +0.44 sigma_prior, dz3 -0.0110 = -0.42 sigma_prior).
  A_IA is detected only at ~1.3 sigma (0.92 +- 0.73); A_baryon is pushed against its lower bound (HPD 2.0 - 2.68).
* Correlation of S8 with the sampled parameters is weak: ln_1e10_A_s 0.34, omch2 -0.20, A_baryon -0.19, A_IA 0.18.

## 3. Nested-sampling cross-check (nautilus, full 14-D problem)

`scripts/run_kv450_nested.py`: nautilus (neural-network-boosted importance nested sampling), 8 worker processes, emulator. The prior is **truncated to the emulator box** only for omch2 (flat on [0.03, 0.34]; ombh2 and h boxes coincide with the prior);
`log Z (box)` is therefore the evidence of the box-truncated prior. Because the MCMC shows no mass outside the box, `log Z (full prior) = log Z (box) + ln(0.31/0.98) = log Z (box) - 1.151` (flat omch2 prior), up to the (undetectable) mass at omch2 < 0.03 or > 0.34.

| run | z grid | n_live | likelihood calls | ESS | log Z (box) | log Z (full prior) | S8 (mean +- std) | wall |
|---|---|---|---|---|---|---|---|---|
| `nautilus1` | 41 nodes | 400 | 9,464 | 3,094 | -98.851 | -100.002 | 0.7318 +- 0.0414 | 249 s |
| `nautilus2` | 41 nodes | 2000 | 41,184 | 16,882 | -98.497 | -99.648 | 0.7363 +- 0.0375 | 1012 s |
| `nautilus3` | **150 nodes (spec)** | 2000 | 41,080 | 16,638 | **-98.510** | **-99.661** | 0.7354 +- 0.0376 | 1436 s |
| MCMC (section 2) | spec | - | 287,200 cycles | 16,513 (S8) | - | - | 0.7358 +- 0.0374 | - |

* `nautilus1` (n_live = 400) is under-converged (S8 sigma 10 % too wide, log Z 0.35 low) and is kept only to show the sensitivity; use `nautilus2/3`.
* `nautilus3` vs MCMC: the difference of the posterior means in units of the MCMC std is <= 0.036 for all 17 quantities (e.g. omch2 -0.029, h 0.007, ln_1e10_A_s 0.017, S8 -0.012, sigma8 0.017, Omega_m -0.030);
  the ratio of standard deviations is 0.973 - 1.010. (`nautilus2`: <= 0.026, 0.977 - 1.016.) Two independent algorithms and two z grids give the same posterior.
* The two n_live = 2000 evidences agree to 0.013; nautilus gives no error bar, the n_live-dependence (0.35 between n_live = 400 and 2000) suggests an uncertainty of order 0.1 or less on log Z for n_live = 2000, **not quantified**. Evidences are only meaningful with the stated priors.
  (The full-prior value is an extrapolation using the MCMC result that the mass outside the box is negligible.)

## 4. Systematic sensitivity

MAP column: L-BFGS-B from the best MAP (spec grid unless stated), `scripts/run_kv450_sens.py`. S8 columns: for the three first variants (parameter treatment) and `no m-bias correction`, three short blocked-Metropolis chains (4000 production cycles
after 1000 burn, same sampler, started at the pilot mean, `sens/*_chains.npz`); for the numerics variants, importance re-weighting of 500 random posterior samples (`run_kv450_reweight.py`: w = exp(ln L_variant - ln L_base)), which is accurate when the
chi2 changes by <~ 0.2 (the weights are essentially uniform: effective sample size 499.8 - 500). The reference is the production run: MAP chi2 181.889 (L-BFGS level; 181.874 after Powell), S8 = 0.7358 +- 0.0374 (Monte-Carlo error 0.0003).

| variant | best chi2 | Delta chi2 vs baseline | S8 mean +- std | Delta S8 (MC error) | notes |
|---|---|---|---|---|---|
| baseline (free A_baryon, A_IA, m-corr. on, hmcode, fast, spec) | 181.889 | 0 | 0.7358 +- 0.0374 | 0 | production |
| A_baryon fixed 2.0 | 181.889 | 0.000 | 0.7455 +- 0.0401 | **+0.0097** (0.0014) | the MAP already has A_baryon = 2 |
| A_baryon fixed 3.13 | 182.844 | +0.955 | 0.7217 +- 0.0346 | **-0.0141** (0.0007) | MAP S8 0.7275 |
| A_IA fixed 0 | 182.590 | +0.701 | 0.7271 +- 0.0357 | **-0.0087** (0.0012) | MAP S8 0.770 |
| m-bias correction off | 177.890 | -3.999 (not comparable: the data vector is changed) | 0.7416 +- 0.0369 | **+0.0058** (0.0011) | |
| CAMB 'mead' non-linear backend (exact CAMB linear) | 181.878 (40-iteration polish from the MAP; 181.874 with hmcode) | +0.004 | 0.7378 vs 0.7379 (same 500 draws) | -0.0001 | max Delta chi2 over the 500 samples 0.17 |
| `kcap` fidelity (FFT Hankel + node-trapezoid) | 181.897 | +0.008 | 0.7374 vs 0.7379 (same 500 draws) | -0.0005 | mean Delta chi2 +0.017, max 0.17; sd of S8 0.0362 |
| 41-node z grid | 181.890 | +0.001 | 0.7379 vs 0.7379 (same 500 draws) | -0.00001 | mean Delta chi2 +0.0006, max 0.006 |

Reading: the numerical choices (non-linear backend, Hankel fidelity, z grid) change S8 by < 0.001, i.e. < 3 % of the posterior sigma. The physical modelling choices shift S8 by up to 0.014 (0.38 sigma):
fixing A_baryon at its upper end (3.13, weak baryon feedback) lowers S8 by 0.014 and costs Delta chi2 = 0.96, fixing A_baryon at 2.0 raises it by 0.010 (no chi2 cost), fixing A_IA = 0 lowers it by 0.009 (Delta chi2 0.70),
and switching off the m-bias correction raises it by 0.006. None of them exceeds half a sigma. The chi2 values show that A_baryon and A_IA are not constrained by the data beyond ~1 in chi2.
The MAP-based S8 values (0.728 - 0.788) vary more than the posterior means because they lie on the plateau; they should not be used for S8.

## 5. What looked wrong or fragile (root causes)

1. **`python -m kidspipe` cannot use the emulator**: `cli.py::_model_kw` does not pass `linear_backend`, so `map`/`mcmc`/`nested` always run exact CAMB. All production runs here use `api.build_model(linear_backend='emulator')` from `scripts/run_kv450_common.py`.
   The CLI was not changed (not owned by this task).
2. **Thread oversubscription**: `theory/hmcode.py` and `theory/limber.py` use `numba` `prange`; `OMP_NUM_THREADS=1` does not limit numba. Many single-core worker processes therefore each started a 22-thread pool and
   ran far below full speed (observed: the first MAP batch produced no result in 12 min while the load average rose to 76, to which other jobs on the workstation contributed; after setting `NUMBA_NUM_THREADS=1` a start took 4 - 20 min). The drivers set `NUMBA_NUM_THREADS=1` (in `run_kv450_common.py`) before importing numpy/numba. The pipeline
   should do this itself when used with process pools.
3. **HMcode is the cost, not the emulator**: single-threaded a new cosmology costs ~0.2 s (HMcode ~0.19 s on 150 z nodes), the emulator ~25 ms. This is why the MAP exploration used the 41-node grid and why fewer medium moves per cycle were used.
4. **The MAP is not a point estimate**: chi2 differs by < 1 along a 4-parameter valley that reaches the prior bounds; small changes in start, optimiser or grid move ombh2 (0.019 - 0.026), h (0.68 - 0.82), ln_1e10_A_s (4.1 - 5.0).
   Best fits and Hessians are not meaningful here; use posterior summaries.
5. **chi2 definitions**: the lowest chi2 depends strongly on whether the Gaussian nuisance priors are included (181.87 with, 177.33 without, at an unphysical nuisance configuration). The published 180.6 *(external)* cannot be mapped onto either without its definition.
6. Five chains (prod1 #0; prod2 #0, 5, 6, 7) have joint-block acceptance 0.41 - 0.49 (block scale tuned small by the short 600-cycle tuning phase); this costs efficiency, not correctness (fixed proposal after tuning).

## 6. Reproduce

Everything runs from the repository root of a clone (scripts add `src/` to the path themselves; outputs go to `runs/kv450_final/`, or to `$KIDSPIPE_RUNS/kv450_final/` when that variable is set); one core per worker process. Wall times are for the shared development workstation (22 cores, WSL2); scale `--workers` to your machine (a free Colab runtime has 2 vCPUs, so these steps need hours there; `docs/COLAB.md` and `notebooks/01_kv450_inference.ipynb` give Colab-sized settings). The reference results of this analysis (`runs/kv450_final/`: MAP, chains, summaries) are shipped with the repository, so the later steps can be re-analysed without re-running the earlier ones.

```bash
# run from the repository root of your clone
# 1. global MAP
python3 scripts/run_kv450_map.py --stage explore --starts 36 --workers 13     # 36 starts, 41-node grid, ~29 min -> map_explore.json
python3 scripts/run_kv450_map.py --stage polish --top 6 --workers 6 --polish none && cp runs/kv450_final/map_starts.json runs/kv450_final/map_polish_lbfgs.json   # ~2 min
python3 scripts/run_kv450_map.py --stage polish --top 3 --workers 3 --polish powell    # ~29 min -> map_starts.json (spec grid)
python3 scripts/run_kv450_mapbest.py                                   # map_best.json, starts.json
python3 scripts/run_kv450_exact.py map                                 # exact-CAMB check of the polished candidates
python3 scripts/run_kv450_chi2split.py                                 # chi2 split at the MAP (exact CAMB)
python3 scripts/run_kv450_profile.py --workers 12                      # lnAs and S8 profiles -> profile.json
python3 scripts/run_kv450_ml.py --workers 6                            # maximum-likelihood (flat nuisance priors) -> ml.json
# 2. posterior
python3 scripts/run_kv450_mcmc.py pilot --tag pilot1 --chains 6 --burn 1500 --cycles 1500 --starts runs/kv450_final/starts.json
python3 scripts/run_kv450_mcmc.py prod --tag prod1 --pilot pilot1 --chains 6 --minutes 80  --tune 600 --seed 21
python3 scripts/run_kv450_mcmc.py prod --tag prod2 --pilot pilot1 --chains 8 --minutes 110 --tune 600 --seed 22
python3 scripts/run_kv450_analyze.py prod1,prod2 --burn-frac 0.05      # R-hat, ESS, marginals -> combined_prod1_prod2/analysis.json
python3 scripts/run_kv450_exact.py chain prod1,prod2 200               # exact-CAMB re-evaluation of 200 samples
python3 scripts/run_kv450_report.py                                    # tables of section 2.4 and 3
# 3. nested sampling
python3 scripts/run_kv450_nested.py --workers 8 --z-grid reduced --n-live 2000 --n-eff 8000 --tag nautilus2 --seed 12
python3 scripts/run_kv450_nested.py --workers 8 --z-grid spec    --n-live 2000 --n-eff 8000 --tag nautilus3 --seed 13
# 4. sensitivity
python3 scripts/run_kv450_sens.py --variants abar2,abar313,aia0,nomcorr,baseline,reduced,kcap --workers 3 --cycles 4000 --burn 1000 --chains 3 --map-file map_best.json
python3 scripts/run_kv450_sens.py --variants camb_mead --no-chain --workers 1 --map-file map_best.json
python3 scripts/run_kv450_reweight.py --tags prod1,prod2 --variants kcap,reduced,camb_mead --n 500 --workers 6
```

Notes: the first four sensitivity variants started from the L-BFGS-level MAP (chi2 181.889) that `map_best.json` held at the time; `map_best.json` was afterwards replaced by the Powell-polished point (181.874), which only matters for the `camb_mead` MAP run.
`run_kv450_sens.py` overwrites `sens/sens_summary.json` on every call (the copy `sens/sens_summary_main.json` holds the main run). Runs are deterministic given the seeds up to the usual floating-point and multi-process ordering effects
(the samplers draw from per-chain generators; the MAP and nautilus stages use fixed seeds).

Files in `runs/kv450_final/`: `map_explore.json`, `map_starts.json`, `map_best.json`, `map_exact_check*.json`, `chi2_split.json`, `profile.json`, `ml.json`, `pilot1/`, `prod1/`, `prod2/` (per-chain `prod_c*.npz`: samples, log_post, derived = sigma8, Omega_m, S8),
`combined_prod1_prod2/analysis.json`, `exact_check_prod1_prod2.json`, `nautilus{1,2,3}/` (`nested.npz`, `nested_summary.json`), `sens/`, `reweight.json`, and the `*.log` files.
