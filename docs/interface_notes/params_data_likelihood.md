# Interface notes: params / likelihood (data.py, gauss.py)

1. **Who adds delta_c and A_c?** DESIGN.md is ambiguous (model.mu "includes them" vs.
   `GaussianLikelihood.chi2(mu, A_c, delta_c)`). Implemented: `GaussianLikelihood.chi2/log_like/chi2_batch(mu, A_c=1, delta_c=0)`
   take the PURE binned theory 195-vector and add `delta_c^2` (xi+ entries only) and `A_c^2 * c-term`
   (spec s9) themselves. `CosmicShearModel.mu` should therefore return pure theory (no systematics);
   otherwise they would be double counted. `GaussianLikelihood.add_systematics` is public if the model wants it.
2. **m-bias** is applied inside `GaussianLikelihood` by dividing the data (`apply_m=True`), fixed `params.M_BIAS`.
   `KV450Data.d` is the raw (uncorrected) cut data; `GaussianLikelihood.d` is the corrected one.
3. **Theory ordering helper**: `kidspipe.likelihood.data.assemble_vector(xip[...,15,9], xim[...,15,9]) -> (...,195)`
   with pair index in `PAIRS = [(j,i) for i in range(5) for j in range(i,5)]` order, i.e. pair p is
   `C[j, i]` with j>=i. 195-layout: per pair `[xi+ bins 0..6, xi- bins 3..8]` (pair-major, interleaved).
   This is NOT the block layout `[xi+ (15x7) | xi- (15x6)]` of the Flinc file; use `from_pm_blocks`/`to_pm_blocks`.
4. **Data paths**: `data/kv450/{xipm_data_vector.dat,cov_analytic.txt,c_term_treecorr.out}` (lower-case dir `kv450`);
   override the root with `$KIDSPIPE_DATA`.
5. `params.log_prior(theta, normalized=True)`; extra helpers `prior_cdf`, `from_dict`, `eta0`, `A_s`, `dz_from_theta`, `FIXED`.
6. Empirical validation of the pair ordering: the Flinc theory vector (block layout -> `from_pm_blocks`)
   gives chi2 = 209.6 against the KV450 data/cov (a different cosmology, so only O(dof) is expected),
   whereas the unconverted layout gives 2.2e5 and a reversed pair order 627.
7. **Review changes (independent review)**:
   * `prior_transform`: NaN in `u` raises `ValueError`; finite `u` outside [0,1] is clipped (previously flat parameters were
     clipped but Gaussian ones returned NaN). `log_prior`, `prior_cdf` and `prior_transform` raise `ValueError` for a wrong last-axis length.
   * `GaussianLikelihood.chi2/chi2_batch`: non-finite `A_c`/`delta_c` now give `chi2 = inf` (`log_like = -inf`), never NaN
     (previously NaN propagated, and `log_like_batch` returned NaN).
   * `GaussianLikelihood.d` and the Cholesky factor are read-only arrays; `KV450Data.m_factor` validates `m.shape == (5,)`.
   * Independent verification (python loops from the raw files, no package code): data, cut covariance, c-term vector and m-factor
     are bit-identical; null chi2 = 269.304 (m-corrected, no c-term), 267.177 (no m), 267.700 (m + c-term). Covariance ordering is
     confirmed independently of the data by the shot-noise relation Var(xi+_ab)^2 ~ Var_aa Var_bb / 4 (ratios 0.28-0.32 for all
     10 cross pairs). Anchor-file xi (4 pairs) vs KV450 data incl. cross-covariances: chi2 = 55.5 for 52 points.
   * Timing (measured, 1 core): `chi2` 0.015 ms per call; `chi2_batch` 1000 vectors 4.9 ms.
