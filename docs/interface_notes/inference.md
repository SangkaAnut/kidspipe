# inference module - interface notes

Files: `src/kidspipe/inference/{__init__,posterior,maxlike,mcmc,nested,chains}.py`, `tests/test_inference.py`.

## Interface (as implemented)
* `Posterior(model_mu, likelihood, params_module=None, *, names, bounds, log_prior, prior_transform, start, blocks, memo=True, like_kwargs=None)`
  - `model_mu`: callable `theta->mu` or object with `.mu`; `likelihood`: callable or object with `.log_like(mu)`.
  - `params_module=None` imports `kidspipe.params` (needs `PARAM_NAMES`, `BOUNDS`, `log_prior`, `prior_transform`, optional `START`).
  - methods `log_prior`, `log_like` (bounds-checked, no prior), `log_post`, `eval -> (lnP, lnL)`, `prior_transform`,
    `log_post_batch`, `timing_report()`, `reset_stats()`, `random_start(rng)`; attributes `stats`, `blocks`, `bounds`, `names`.
  - `stats.by_kind_n/t` classify each *model* evaluation by the most expensive block that changed with respect to the
    previous model evaluation (slow/medium/fast/same). With a model cache of >=2 slow-stage entries a fast move after a
    rejected slow move is classified "slow" here (pessimistic) although the model serves it from cache.
* Blocks: `FAST_PARAMS`, `MEDIUM_PARAMS`, `SLOW_PARAMS` (name tuples), `ParamBlocks.from_names(names)` -> index arrays.
* `find_map(logpost, bounds, x0, n_start, ...) -> MapResult`; `numerical_hessian(logpost, x, bounds) -> HessianResult` (`.cov`, `.hess`);
  `map_and_covariance`.
* Samplers: `adaptive_metropolis(logpost, x0, n_steps, n_burn, cov0)`, `blocked_metropolis(logpost, x0, schedule, n_cycles, n_burn_cycles, cov0)`
  with `schedule = [(idx_block, repeats), ...]`, `ensemble_sampler(logpost, p0, n_steps, vectorized=False)`.
  Diagnostics `split_rhat`, `integrated_autocorr_time`, `effective_sample_size` (axis 0 = steps, optional axis 1 = chains).
* `run_nautilus(loglike, prior_transform, ndim, ...)`, `run_dynesty(..., dynamic=False)` -> `NestedResult` (samples, normalised weights, log_like, log_z, log_z_err).
  `make_nested_functions(post)` gives `(post.log_like, post.prior_transform, post.ndim)`.
* chains: `Chain` container, `marginal_stats`, `hpd_interval_1d`, `pjhpd_extents(samples, log_post, weights, levels)` (O(N log N)),
  `s8_column`/`append_s8`, `remove_burn_in`/`estimate_burn_in`, `save_chain`/`load_chain` (npz + txt), `summary_table`, `correlation_table`.

## Deviations / open questions
1. DESIGN.md says `Posterior(model, likelihood)`; the 3rd positional argument `params_module` was added as requested and it is duck-typed (a `SimpleParams` class is provided for toys).
2. The PJ-HPD needs the posterior density at every sample (`log_post = ln prior + ln L`). For nested samples the caller must add `ln prior` if the prior is not flat
   (the prior_transform of `kidspipe.params` has truncated-Gaussian priors for delta_c, A_c, dz); `NestedResult.log_like` alone is not enough in that case.
3. The Posterior's `log_like` returns -inf outside the parameter bounds (CAMB must never see out-of-range values).
4. Derived columns (sigma8, omega_m) must come from the model (e.g. blobs from `CosmicShearModel`); `chains.append_s8` expects them as named columns. Not provided by the Posterior (model_mu returns only mu).
5. Blocked sampler efficiency relies on the model caching >= 2 slow-stage results (LinearBackend `cache_size>=2`, default 8) and the P_nl/C_ell stage caches
   keyed on the corresponding parameter subsets; if the model keeps only the last point, a rejected slow move forces recomputation of the slow stage on the next block move.
6. nautilus returns no evidence error (`log_z_err = nan`). Nautilus calls the prior transform on batches when `vectorized=True`; the wrapper loops over rows unless `prior_vectorized=True`.
7. Hessian step control targets a quadratic change of -ln P in [0.02, 0.5]; for the real model (CAMB/numerical noise ~1e-6 in chi2) this should be well above noise but should be re-checked
   when the real model is plugged in (`HessianResult.steps`, `n_repaired`).

## Independent review changes (inference)
8. **A_c / delta_c were silently dropped (fixed).** `GaussianLikelihood.log_like(mu, A_c=1, delta_c=0)` takes the PURE theory vector and adds
   `delta_c^2` / `A_c^2 c` itself (see `params_data_likelihood.md` item 1), but `Posterior` called `likelihood(mu)` only, so both
   parameters were fixed at 1 and 0 in ln L and sampled from their priors alone. `Posterior` now forwards `theta[A_c]`, `theta[delta_c]` as keywords to
   the likelihood whenever the likelihood callable has explicit arguments of those names and they are parameter names (`Posterior.like_kwargs`;
   override with `like_kwargs={"kw": "param_name"}`, disable with `{}`). Consequence: `CosmicShearModel.mu` must return pure theory (no c-term,
   no delta_c^2), exactly as `params_data_likelihood.md` already requires; a model that adds them itself AND a likelihood with these keywords
   would double count.
9. `Posterior.in_bounds/log_prior/log_like/log_post` raise `ValueError` for theta of the wrong length (a length-1 theta used to broadcast against all bounds silently).
10. `blocked_metropolis` raises `ValueError` if the schedule does not cover every coordinate (they would never move); `allow_partial=True` keeps the old behaviour.
11. `numerical_hessian` raises `ValueError` (instead of failing deep inside `eigh`/`cholesky` or returning NaN) if any stencil value is non-finite after step control.
12. `NestedResult.log_post(log_prior)` = ln L + ln pi at the samples (input for `pjhpd_extents` with non-flat priors, deviation 2 above). `save_chain` rejects
    parameter names with whitespace (the text format could not be read back). `make_nested_functions` and `safe_loglike` are exported from the package.
13. `split_rhat`/ESS axis convention made explicit: a 2-D array is ONE chain `(steps, params)`; use 3-D `(steps, chains, params)` for several chains.
14. Cost note for the real model: on a synthetic 14-d problem with the real bounds the multi-start MAP (4 starts, L-BFGS-B + Nelder-Mead rounds) used 3e4-7e4 posterior evaluations
    (about 2-6 s at 30 us/eval), i.e. the MAP step is NOT cheap once ln P costs milliseconds: budget `n_start`, `n_rounds`, `polish` accordingly.

Tests added: forwarding of A_c/delta_c (toy and real `GaussianLikelihood`), theta-length check, schedule coverage, Hessian failure, 2-d banana marginal by quadrature
and a 1-d quartic CDF by quadrature for all three samplers (mutation-checked: a tempered acceptance ratio or a wrong stretch-move exponent makes them fail),
uniform-box boundary handling, `NestedResult.log_post`. `check_moments` tolerances are now derived from the effective sample size (previously std 15 % and correlations 0.25 absolute).
