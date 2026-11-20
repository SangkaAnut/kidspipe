# COSEBIs validation (KV450 E_n, 15 pairs x up to 20 modes)

Scope: independent validation of `src/kidspipe/theory/cosebis.py` and `src/kidspipe/likelihood/cosebis.py` (spec s14.1), run on the
emulator model. Nothing from `reference/`, `docs/study/` or `docs/spec_checks/` was used. No source file under `src/` was edited by the validation.
Numbers: `runs/cosebis/validation_numbers_main.json`, `runs/cosebis/validation_numbers_post_n5.json`. Script: `scripts/cosebis_validate.py`.

## Summary
* The implementation reproduces its own consistency checks and the staged tables. Using the xi+/- best fit (`runs/kv450_final/map_best.json`):
  chi2 = 71.15 for 75 points (5 modes, p = 0.60 with dof = N) and 293.5 for 300 points (20 modes, p = 0.60). These are NOT a COSEBIs fit
  (parameters were fitted to xi+/-; same data), so they show consistency, not an independent goodness of fit.
* COSEBIs-only S8 = 0.7296 +- 0.0361 (MCMC, 5 modes) and 0.7293 +- 0.0343 (importance reweighting of the xi+/- chain), versus xi+/- 0.7356 +- 0.0374
  (0.7358 in `analysis_prod12.txt`). Shift -0.006 = -0.16 sigma_xi. As expected for the same data, close.
* **The COSEBIs MCMC is NOT converged to the requested standard**: S8 split R-hat = 1.054 (target < 1.05), sigma8 1.30, Omega_m 1.33 (see below).
  The S8 number from it is approximate; the reweighting is the cross-check and agrees.
* B-mode check against data is impossible: the FITS file has only the `En` table (no B_n). B_n theory is the zero placeholder.

## What was validated (all independently recomputed here)
| item | result |
|---|---|
| T_+n vs tabulated Tplus (0.5-100), n=1,2,5,10,20 | <= 5.5e-16 of peak |
| T_-n vs tabulated Tminus | 4e-9 (n=1), 2.9e-8 (2), 4e-7 (5), 2.8e-6 (10), 1.4e-5 (20) of peak; ours vs mpmath quadrature (n=2,5, 3 points each): 4e-14 abs. The table is the less accurate side |
| mpmath roots/norms (n<=10) vs tables | roots 1.7e-8..5e-8 (8-digit table), norms 0 rel |
| orthonormality (0.5-300, 20 modes) | 2.8e-7 relative; first 5 modes 6.4e-8 |
| E_n from C_ell vs from xi(theta), 15 pairs x 20 modes | max 2.0e-4 sigma_data; modes 1-5 max 1.8e-4 relative to the entry |
| l_max convergence (n<=5) | 1e4: 0.012 sigma; 3e4: 9e-4; 6e4: 2e-4; kcap 1.15e5: 2e-4 (plateau = FFT/spline floor) |
| pure-E B_n leakage / E_n, by mode | 3e-9 (n=1), 3e-7 (5), 4e-5 (9), up to ~4e-4 for n 11-12 and larger above (8-digit roots) |
| COVMAT | symmetric, positive definite (min eig 2.3e-22); condition 7.2e3 (5 modes), 2.9e4 (20); max off-diagonal correlation (5 modes) 0.91 |
| NZ_SOURCE (file) vs pipeline n(z) | mean z differs by <= 1.0e-3 (bin1 0.3944 vs 0.3954; bin5 0.9976 vs 0.9977); max profile diff 2.7e-3 of peak (histogram midpoints vs interpolation) |
| FITS row order | pair-major, mode fastest = `data.PAIRS` order (checked at load) |

chi2 at the xi+/- best fit (null = chi2 of zero model):

| modes | N | m-corrected | raw | null (corr / raw) | chi2 diag-only (corr) |
|---|---|---|---|---|---|
| 5 | 75 | 71.15 (p 0.60) | 69.87 (p 0.65) | 206.5 / 207.3 | 67.4 |
| 20 | 300 | 293.5 (p 0.60) | 287.7 (p 0.69) | 428.9 / 425.1 | 296.0 |

At the xi+/- posterior-mean theta: 81.3 (5 modes), 304.5 (20 modes). Over 200 xi+/- posterior draws: 5 modes 67.6 / 74.7 / 95.3 (min/median/max), 20 modes 290.0 / 297.0 / 316.5.
Whitened 20-mode residuals at the best fit: mean -0.05, std 0.99, max |r| 2.99. Normalised diagonal residual rms 0.95 (n<=5), 0.99 (all 20), max 2.4.
p-values use dof = N (no parameters subtracted; with 14 mostly prior-dominated parameters the true dof is not well defined). The xi+/- fit itself has chi2 181.9 on 195 points.
Visible structure: bins 2-4 and 2-5 data lie above the model at n=1..5 by about 1-2 sigma (correlated), seen in `cosebis_04c_residuals.png`.

The default `apply_m=True` divides data by (1+m_a)(1+m_b). Whether the shipped E_n are already m-corrected is not documented; raw data is slightly preferred
(Delta chi2 = 1.3 at 5 modes, 5.8 at 20) but the model was fitted with m-corrected xi+/-, so this cannot decide it.

## COSEBIs-only posterior (plot 06)
Method: `runs/cosebis/mcmc.py`, kidspipe `blocked_metropolis`, emulator model, 5 modes, m-corrected data, proposal covariance and starts from the xi+/- posterior
(prod1+prod2). 10 chains, 1500 burn-in + 500 recorded cycles each (chain 4 has 1000; the first 500 are used), 14 parameters sampled (delta_c, A_c are unconstrained by COSEBIs and just sample their priors).

* **Truncated to the emulator box** (omch2 in [0.03, 0.34], ombh2, h): proposals outside get ln P = -inf. A first run with CAMB fallback outside the box was far too slow
  (no chain finished in > 100 min) and was discarded. The xi+/- posterior has 0 mass outside the box and omch2 in [0.04, 0.26], but I did not
  verify that the COSEBIs-only posterior has none outside; 241-610 of the proposals per chain were rejected by the box edge, so the truncation may matter for the Omega_m-sigma8 tails.
* Effective compute was ~2 cores (10 workers got ~18 % CPU each), so the chains are short: 500 cycles per chain at S8 tau ~ 16 cycles gives only a few tens of independent samples per chain.
* Result: S8 = 0.7296 +- 0.0361 (68 %: 0.694 - 0.766), sigma8 0.793 +- 0.154, Omega_m 0.277 +- 0.088. Chain means of S8: 0.713 - 0.739. Split R-hat: S8 1.054 (fails < 1.05), sigma8 1.30, Omega_m 1.33. Parameters
  other than S8 are not converged; only S8 is quoted. Statistical error on the S8 mean is roughly 0.036/sqrt(few hundred ESS) ~ 0.002-0.003.
* Importance reweighting of 4023 thinned xi+/- chain samples with w = L_COSEBIs/L_xi (computed with this pipeline, same priors): ESS 1137, S8 = 0.7293 +- 0.0343; ln w has std 1.23 and
  corr(ln L_cos, ln L_xi) = 0.82 (the data are strongly correlated). Reweighting is limited to the xi+/- support (no new regions), so it is fair for close datasets only.
* Maximum-posterior sample: chi2_COSEBIs = 69.5 (5 modes); 50 thinned chain samples have min/median chi2 68.3 / 73.0. Chain max ln P = -8.07 (includes the normalised prior).
* The COSEBIs posterior is NOT a replication of the published KV450 COSEBIs analysis (different n_modes handling, no B-modes, priors per this project).

External, unverified (from memory, not checked here): Asgari et al. 2019 / Hildebrandt et al. 2020 quote KV450 S8 of order 0.74 for xi+/- and roughly 0.72 for COSEBIs (5 modes)
with errors ~0.04. I did not check these against the papers; treat as unverified.

## Plots (`plots/`)
* `cosebis_01_filters.png` T_+n, T_-n (n=1..5, 0.5-300 arcmin); table overlay (0.5-100); residual vs table (n=1,2,5,10,20). Top-panel y ranges clip the small-theta peaks.
* `cosebis_02_Wn.png` W_n(l) (l<1500), integrand l^2 W_n C_33/2pi, cumulative E_n(<l), l_max convergence. The integrand has converged by l ~ 1e4 (oscillating tail beyond).
* `cosebis_03_routes.png` E_n(C_l) vs E_n(xi): relative difference n<=5 per pair, difference/sigma for all modes, E_n values.
* `cosebis_04_data_vs_model.png` (n<=5), `cosebis_04b_data_vs_model_20modes.png`: data with sqrt(diag COVMAT) errors, xi+/- best fit, posterior-mean-theta model, 68 % posterior band.
  The posterior-mean-theta curve is above the best fit because the mean of a banana-shaped posterior is not a representative point.
* `cosebis_04c_residuals.png` normalised residual maps (best fit, posterior mean) and whitened-residual histogram.
* `cosebis_05_covariance.png` correlation matrix (75 and 300 points) and its spectrum.
* `cosebis_06_posterior.png` S8 marginals (xi+/-, COSEBIs MCMC, reweighted), sigma8-Omega_m scatter, S8 traces.
* `cosebis_07_chi2_samples.png` chi2 over xi+/- posterior draws.

## Caveats
* Not independent: filters, tables and likelihood come from the same implementation; I re-derived the T_- check with mpmath only at 6 points and re-ran the implementer's numbers (all reproduced).
* Tabulated roots have 8 digits: modes n > ~10 are limited to ~1e-4..1e-3 (B_n leakage), irrelevant for n<=5 and tiny against 20-mode errors.
* The emulator is an approximation (|Delta chi2| <= 0.004 on 30 test points for xi+/-; not re-tested for COSEBIs). `tests/test_e2e.py` was not re-run after the `model.cl` hook.
* MCMC short and truncated as described; S8 R-hat 1.054.
* The 20-mode chi2 of 293 is dominated by noise modes; modes 1-5 carry the signal.
* The `data/cosebis/cache/` operator cache holds npz files: a regenerable cache (see `data/README.md`); delete for a clean tree, it is rebuilt on first use (10-20 s per table).

## Reproduce
```bash
# from the repository root of your clone (after `pip install -e '.[all]'`, or with `export PYTHONPATH=$PWD/src`)
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python3 scripts/cosebis_validate.py main      # plots 01-05, 07, numbers (~1.5 min)
COSEBIS_WALL=1500 python3 runs/cosebis/mcmc.py 5 10 10000 1500 n5   # MCMC, ~30 min on ~2 effective cores
python3 scripts/cosebis_validate.py post      # plot 06 + reweighting (reweighting ~8 min, cached in runs/cosebis/reweight_n5.npz)
pytest -q tests/test_cosebis.py
```
