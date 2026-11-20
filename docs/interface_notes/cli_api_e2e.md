# Interface notes: api.py, cli.py, tests/test_e2e.py

## api.py
* `build_posterior(model=None, likelihood=None, **model_kw)` returns an `inference.Posterior` with extra attributes `model`, `likelihood`,
  `recorder`. It asserts `post.like_kwargs == {'A_c': 'A_c', 'delta_c': 'delta_c'}` (the wiring that `Posterior` auto-detects, see
  `inference.md` item 8) so that a future change cannot silently fix A_c = 1, delta_c = 0.
* **Blocks differ from `inference.SLOW_PARAMS`.** `api.BLOCKS` = slow (omch2, ombh2, h), medium (n_s, ln_1e10_A_s, A_baryon), fast (A_IA, delta_c, A_c, dz1..5).
  `inference.SLOW_PARAMS` also lists n_s, but the linear backend keys its cache on (omch2, ombh2, h) and handles n_s as an exact
  rescale (bench: n_s move 19 ms vs ~780 ms for omch2/ombh2/h). The CLI schedules and the Posterior timing classes use `api.BLOCKS`.
  **Resolved in the independent review (docs/REVIEW.md): `inference.SLOW_PARAMS` = (omch2, ombh2, h), `MEDIUM_PARAMS` = (n_s, ln_1e10_A_s, A_baryon); they now equal `api.BLOCKS` (test `test_inference_blocks_match_api_blocks`).**
* **Derived quantities** (`sigma8`, `omega_m`, `S8`) depend only on the first five parameters. `DerivedRecorder` wraps `model.mu` and stores them
  per 5-tuple the first time a cosmology is evaluated (`model.last_derived`), so chains need no post-processing run; `lookup()` falls back to
  `LinearBackend.compute` for unseen points (CAMB cost if the transfer function is not cached). The table is capped at 2e6 entries.
* Presets: `test` (spec s4.1, A_baryon = 2), `scan` (s4.2 base at ln_1e10_A_s = 2.8), `start` (`params.START`). Theta files: JSON dict / 14-list / `map.json`, or `name = value` text.

## cli.py
* Parallelism is by spawned worker processes (own model + CAMB cache each). The env var `OMP_NUM_THREADS` of the parent is set to `--worker-threads`
  when a pool is created, because ProcessPoolExecutor spawns children lazily and they inherit it; the CAMB thread count is also passed explicitly.
* `_EvalPool` (nautilus): `map` ignores the callable nautilus passes and evaluates ln L (finite, -1e300 on failure) in the workers while recording the derived
  quantities; it is passed as `pool=(pool, None)` because nautilus would otherwise also route its bound-fitting `map` calls through it.
  dynesty uses `_ExecPool` with module-level `_w_loglike` and `params.prior_transform` (bound methods of `Posterior` would pickle CAMB state).
* `map`: `find_map` is generic and uses a finite-difference gradient (3 CAMB runs per gradient). The CLI therefore first optimises the 11 cheap parameters
  with omch2/ombh2/h frozen (`staged`, default on), then runs the full-space `find_map` from that point. One start = one job; `--workers` parallelises starts.
* `mcmc` rejects the MAP Hessian as proposal covariance when `at_boundary` is set for any parameter (`--force-hessian` overrides).
* Exit codes: `check` 1 on failure; `loglike` 2 if the point has `ln post = -inf`.

## Change made outside my files
* `inference/nested.py::safe_loglike` returned a local closure, which cannot be pickled, so `run_dynesty(..., pool=<process pool>)` failed (and, with a
  ProcessPoolExecutor, hung on shutdown). It now returns a picklable `_SafeLoglike` instance (same behaviour). Found by running `kidspipe nested --sampler dynesty --workers N`.

## Observations about other modules
* (resolved) `inference.SLOW_PARAMS` used to contain `n_s`; now aligned with `api.BLOCKS`.
* `CosmicShearModel.mu` recomputes the C_ell projection for pure likelihood-side parameters (`A_c`, `delta_c` are handled in the likelihood but the
  model still re-runs the Limber projection when only they change: 0.7 ms instead of ~0.03 ms). Harmless.
* Rescale noise: `LinearBackend` rescaling of A_s / n_s is exact to 7e-16 in P_lin, but HMcode's iterative non-linear-scale solve turns that into up to
  1.4e-11 relative differences in the final mu (measured in `tests/test_e2e.py::test_staged_cache_matches_fresh_model`).

## tests/test_e2e.py
Tolerances chosen from measurements: chi2 at the test cosmology 0.5 (observed 0.04), amplitude scan 1.0 (observed 0.56 with `fast`), hmcode vs camb 0.3
(observed 0.09), fast vs kcap 0.1 for chi2 < 198 (observed 0.03; the difference grows to 0.5 at chi2 ~ 240 and is only bounded by 1 there),
staged-cache 1e-12 for unchanged-cosmology points and 1e-10 for points reached through the n_s / A_s rescale (observed 6e-16 in the test, 1.4e-11 in an exploratory run). "Random prior draws" are replaced by draws scattered around the
best fit for the chi2-difference tests (whole-prior draws have chi2 of 1e3 to 1e6, where absolute chi2 differences between numerical variants are not meaningful).
