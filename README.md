# kidspipe

A clean-room Python re-implementation of the KiDS KV450 cosmic-shear (xi+/-) inference pipeline:
14 parameters (5 cosmological, baryon-feedback amplitude, intrinsic alignment, c-term nuisance, 5 photo-z shifts) ->
non-linear matter power spectrum -> Limber shear power spectra for 5 tomographic bins -> Hankel transform and angular
binning -> Gaussian likelihood against the 195-point KV450 data vector, plus a MAP finder, MCMC and nested samplers.

The mathematics comes from `docs/MASTER_MATH_SPEC.md` ("spec sN"); the module layout from `docs/DESIGN.md`. The
implementation was written without reading the original KiDS/kcap source code (clean room); numerical anchors used for validation are
the spec's own self-consistency numbers plus a few published values flagged as *external* below.

Pure Python (numpy, scipy, numba) + CAMB for the linear transfer function. No CosmoSIS, no MPI.

## Install

```bash
git clone <repo> && cd <repo>
pip install -e .                 # numpy scipy numba camb
pip install dynesty nautilus-sampler pytest   # optional: nested sampling, tests
python3 -m kidspipe check        # self-test, ~7 s (first run on a fresh checkout adds numba compilation: ~10 s quiet; 15 s measured at moderate load, 31 s on a loaded machine with 22 numba threads)
```

Tested here with Python 3.12, numpy 1.26, scipy 1.13, numba 0.59, camb 1.5.8, dynesty 2.1.4, nautilus 1.0.5. The data files needed are in `data/`.
Without installing, use `export PYTHONPATH=$PWD/src`.

## Quick start

```bash
python3 -m kidspipe loglike                        # chi2 / ln L / ln post / S8 at the spec test cosmology
python3 -m kidspipe map --starts 2 --maxiter 40 --rounds 1 --polish none    # MAP + Hessian -> runs/map/map.json
python3 -m kidspipe mcmc --chains 4 --workers 4 --worker-threads 4 --burn 300 --cycles 700   # -> runs/mcmc/
python3 -m kidspipe nested --sampler nautilus --workers 20                    # -> runs/nested/
python3 -m kidspipe bench                          # cost per parameter / block / stage
```

```python
from kidspipe import api
post = api.build_posterior()            # CosmicShearModel + GaussianLikelihood + params + blocks
th = api.preset("test")                 # spec test cosmology
print(-2 * post.log_like(th))           # 196.10
print(api.derived_for(post, th))        # {'sigma8': 0.70909, 'omega_m': 0.25030, 'S8': 0.64769}
```

Full command reference: `docs/USAGE.md`. Interface decisions and deviations from the design document: `docs/interface_notes/`.

## Architecture

```
params.py              names, bounds, priors (flat + truncated Gaussian), prior_transform
theory/linear.py       CAMB wrapper, LRU cache on (omch2, ombh2, h); n_s and A_s are exact rescales -> sigma8, Omega_m, chi(z)
theory/emulator.py     optional transfer-function emulator for (omch2, ombh2, h), exact-CAMB fallback outside its box (linear_backend="emulator")
theory/hmcode.py       stand-alone numba HMcode-2016 P_nl(k, z) (backend "hmcode"); CAMB 'mead' is backend "camb"
theory/nonlinear.py    backend dispatcher + high-k extension (spec s2.6)
theory/nz.py           KV450 n(z) and photo-z shifts
theory/limber.py       lensing kernels, NLA intrinsic alignments, Limber C_ell (15 pairs)
theory/hankel.py       C_ell -> xi+/- and theta binning; 'fast' = exact bin-averaged linear operator, 'kcap' = FFT + node-trapezoid
theory/model.py        CosmicShearModel: theta -> pure 195-vector, three cached stages (below)
likelihood/            data vector, cut covariance, c-term, m-bias; Cholesky chi2; adds delta_c^2 and A_c^2 c itself
inference/             Posterior (blocks, stats), find_map + Hessian, blocked Metropolis / ensemble, nautilus / dynesty, chains
api.py                 build_model / build_likelihood / build_posterior, presets, derived quantities (sigma8, Omega_m, S8)
cli.py, __main__.py    python -m kidspipe {check,loglike,map,mcmc,nested,bench}
```

The model re-uses work according to which parameter moved:

| block | parameters | what is recomputed | cost per move, exact CAMB* | with transfer emulator* |
|---|---|---|---|---|
| slow | omch2, ombh2, h (+ n_s, A_s, A_baryon in a joint move) | CAMB transfer function (+ everything below) | 0.58 s | 35 ms |
| medium | n_s, ln_1e10_A_s, A_baryon | exact rescale of cached P_lin, new HMcode P_nl, Limber power look-up | 26 ms | 26 ms |
| fast | A_IA, A_c, delta_c, dz1..dz5 | C_ell projection, Hankel/binning (linear operator), likelihood | 1.1 ms | 1.2 ms |

\* Median wall-clock of `CosmicShearModel.mu` at a new point, `fast` fidelity + stand-alone HMcode, 150-node z grid, 4 CAMB/numba threads, from `benchmarks/bench_pipeline.py`
(`runs/bench_t4.json`, 2026-10-25 22:07; "quiet machine" is inferred from agreement with two further runs, the load was not logged). With one thread per process: slow 1.93 s (emulator: 76 ms), medium 65 ms (emulator: 64 ms), fast 1.3 ms.
The likelihood alone costs 0.03-0.2 ms. The faithful `kcap` path (FFT Hankel + node-trapezoid binning + CAMB 'mead' non-linear, `--fidelity kcap --backend camb`) costs 1.1 s per slow move and ~0.5 s for every A_s / n_s / A_baryon move (CAMB re-runs its HMcode).
Earlier figures (slow 0.78 s, medium 11-21 ms, fast 0.7-1.7 ms from `kidspipe bench`; 1.15 s / 72 ms for exact / emulated slow moves in the first emulator summary) were taken on a loaded machine (load average 20-30) and are 1.4-2x too high.
The machine was loaded again when this section was last checked (load 22): the same 1-thread benchmark ran 2.7-3.6x slower than the quiet set, so re-measure before quoting absolute times.
Details, stage breakdown, thread scaling, accuracy cost of each step: **`docs/PERFORMANCE.md`**.

The emulator (`CosmicShearModel(linear_backend="emulator")` or `api.build_posterior(linear_backend="emulator")`; **not exposed by the CLI**) replaces the CAMB run by a PCA + polynomial/Gaussian-process
regression trained on 1435 CAMB runs inside the box omch2 [0.03, 0.34], ombh2 [0.019, 0.026], h [0.64, 0.82] (m_nu = 0.06 eV); outside the box it falls back to exact CAMB with bit-identical results (counter `model.linear.stats["fallback"]`).
Re-verified on 30 fresh points: max |Delta chi2| vs exact CAMB 0.004 (max relative difference of mu 1.4e-4); P_lin error vs dense-sampling CAMB <= 1.3e-3 for k_h < 10, z <= 3
(but 6.5e-3 against CAMB's own default k sampling at the low-omch2 / high-ombh2 corner around k_h ~ 0.5, an artefact of CAMB's nodes). The KV450 omch2 prior is [0.01, 0.99], so only 32 % of its volume is inside the box.
The `fast` exact bin-averaging differs from the `kcap` binning by up to 0.35 in chi2 for chi2 < 230 (1.2 at chi2 ~ 370); that is a numerical choice, not emulator error (the emulator contributes <= 0.007 over the same 20 points).

`Posterior` maps any failure to `-inf`: out-of-box or non-finite parameters never reach CAMB, and exceptions or non-finite values
from the model/likelihood become `ln L = -inf`.

## Validation status

What was measured (all with this code, `tests/test_e2e.py` and the CLI; "spec" = `docs/MASTER_MATH_SPEC.md` s4 / s15):

| check | result |
|---|---|
| chi2 at the spec test cosmology (A_baryon = 2) | 196.104 (hmcode), 196.091 (camb backend), 196.075 (`kcap` fidelity); spec anchor 196.06 |
| sigma8, Omega_m, S8 at that point | 0.709092, 0.250296, 0.647692 (spec: 0.709092, 0.250296, 0.647692) |
| amplitude scan, ln_1e10_A_s = 2.6 / 2.8 / 3.0 / 3.2 | 192.42 / 188.73 / 199.22 / 239.13 (spec 192.39 / 188.77 / 199.43 / 239.69); max difference 0.56 |
| same scan with `kcap` fidelity | 192.40 / 188.77 / 199.40 / 239.61 (agrees with the spec to <= 0.08, so the differences above come from the exact bin-average of the `fast` path, see below) |
| hmcode vs CAMB backend, 8 points near the best fit (chi2 189-236) | max abs diff in chi2 0.09 (at chi2 = 236), typically 0.01 |
| fast vs kcap fidelity | median element difference of mu 0.2 %, max 0.7 % (largest xi+ angular bin); |delta chi2| = 0.03-0.04 at chi2 189-196, 0.21 at 204, 0.38 at 224, 0.48-0.53 at 239-249, 0.61 at 259 |
| staged-cache correctness (15 prior draws from the central 50 % box; 15 first visits + 45 interleaved revisits through one model with cache_size = 4, then 15 checks against freshly built models) | max relative deviation of mu: 0 (bitwise) for revisits and fast-parameter changes, 6e-16 after an n_s / ln_1e10_A_s / A_baryon change; an exploratory run with larger medium-block changes saw up to 1.4e-11 (rescaled P_lin differs from a fresh CAMB run by 7e-16, HMcode's iterative solves amplify it) |
| A_c and delta_c reach the likelihood | yes (a former bug in the Posterior silently fixed them at 1 and 0; guarded by a test and by `build_posterior`) |
| single-start MAP (`map --starts 1 --maxiter 40 --rounds 1 --polish none`: 2558 evaluations, 275 s, + Hessian 544 evaluations, 66 s, on the loaded machine) | chi2 = 181.85, S8 = 0.786, sigma8 = 1.30, Omega_m = 0.109 (at h = 0.82, ln_1e10_A_s = 5, A_baryon = 2: prior boundaries). *External* published maximum-likelihood chi2: 180.6 (Hildebrandt et al. 2020, Table 6, as quoted in `docs/VALIDATION_TARGETS.md`) |

MCMC status: see the section "Sampling runs" below.

Interpretation of the fast/kcap difference: the spec's reference numbers come from the node-trapezoid binning of the original code ("kcap"
fidelity, which reproduces them to <= 0.08 in chi2). The default `fast` path integrates the Bessel kernel exactly over each theta bin, which
is the more accurate evaluation of the same definition and differs from the reference by up to 0.7 % in mu. Near the best fit this is a
chi2 shift below 0.1; it grows with the residual and reaches ~0.5 at chi2 ~ 240. Use `--fidelity kcap` to reproduce the original binning.

## Sampling runs

Two 4-chain blocked-Metropolis runs of the full 14-parameter KV450 posterior (`python -m kidspipe mcmc --chains 4 --workers 4 --worker-threads 4 --burn 300 ...`,
start at the `scan` preset, default proposal covariance, hmcode backend, `fast` fidelity; outputs in `runs/kv450_blocked*/`):

| run | cycles/chain (+burn) | wall | acceptance | worst split R-hat | S8 (mean +- std; 68 % limits) | S8 ESS / R-hat |
|---|---|---|---|---|---|---|
| separate slow block, seed 1 | 700 (+300) | 28 min | 0.23-0.26 | 1.70 (sigma8, omega_m, omch2, ln_1e10_A_s) | 0.728 +- 0.037; 0.691-0.766 | 280 / 1.04 |
| joint slow+medium block (default), seed 2 | 1000 (+300) | 39 min | 0.20-0.27 | 1.50 (ombh2), 1.20 (h), 1.12 (sigma8) | 0.734 +- 0.037; 0.699-0.770 | 527 / 1.005 |

Reading: these are **not converged chains** for the weakly constrained directions (the combination sigma8-Omega_m-h-ombh2 along the
degeneracy; ESS 40-100 for omch2, ombh2, h, ln_1e10_A_s, sigma8, Omega_m in the second run). The well-constrained combinations
(S8, nuisance parameters, A_baryon, n_s) have R-hat < 1.01 and ESS of 290-1050 in the second run. Proper marginals of omch2, ombh2, h, sigma8, Omega_m would need several times more cycles
(hours on this machine). Moving the six cosmological parameters in one joint proposal (same cost per cycle) reduced R-1 of omch2, ln_1e10_A_s and sigma8 from 0.65-0.69 to 0.06-0.12,
but ombh2 and h are still unconverged (R-hat 1.50, 1.20); the two runs also differ in seed and length, so this is indicative only.

Sanity comparison with published numbers (*external*, as quoted in `docs/VALIDATION_TARGETS.md` for Hildebrandt et al. 2020, Table 4 posterior means, not independently checked; the
published analysis used a different sampler and the paper's exact configuration, so only rough agreement is expected): S8 = 0.737 (+0.040/-0.036); this clone, second run: 0.734 +- 0.037.
Also published / this run (second run mean): n_s 1.021 / 1.024, A_baryon 2.484 / 2.51, A_c 1.022 / 1.025, A_IA 0.981 / 0.91, h 0.745 / 0.733, ln_1e10_A_s 3.158 / 3.08, Omega_m 0.256 / 0.266.
No claim of reproducing the published contours is made.

Cost per move measured inside the chains (load average 25-50 from other jobs, 4 CAMB threads per chain, 4 chains at once): slow 0.83-1.3 s,
medium 38-58 ms, fast 5-13 ms per posterior evaluation; about 1.8 s per cycle (1 slow + 2 medium + 6 fast), 2.8x the 0.64 s per cycle predicted for a quiet machine in `docs/PERFORMANCE.md`.
Per-cycle cost is dominated by the one expensive move (with the emulator the predicted cycle is 94 ms, 55 % of it the two medium updates).

Nested sampling: `kidspipe nested` was run only at smoke-test size (nautilus, 40 live points, 150 calls, 6 workers: finished in 2 min 55 s, 204 weighted samples
written with derived columns; dynesty with a process pool: 30 live points, 120 calls, 6 workers, 3 min 14 s, after a pickling fix in `inference/nested.py::safe_loglike`). No evidence or posterior from a converged nested run is claimed;
since every nested-sampling point is a new cosmology (1.9 s per evaluation per single-threaded worker on a quiet machine, 2.8 s on the loaded one; 2.83 evaluations/s measured with 8 workers on the loaded machine), a converged
14-parameter run (typically 10^5 - 10^6 calls) would take roughly 3-30 h on 20 cores with exact CAMB (estimate from the quiet 1-thread cost, ideal scaling); with the emulator ~0.1-1 h
plus a CAMB-priced early phase, because only 32 % of the omch2 prior lies inside the emulator box (not run).

## Honest limitations

* **CAMB is the bottleneck of the default (exact) path**: any change of omch2, ombh2 or h needs a new Boltzmann run (0.55 s with 4 threads on a quiet machine, 1.9 s with one
  thread; 0.8-2.8 s on the loaded shared machine). A blocked chain therefore does about one expensive move per cycle; a converged 14-parameter chain takes hours on this machine.
  Nested sampling treats every point as a new cosmology (no cache reuse). An optional transfer-function emulator for
  (omch2, ombh2, h) exists (`CosmicShearModel(linear_backend='emulator')`, `src/kidspipe/theory/emulator.py`, artefact `data/emulator/transfer_emulator_v1.npz`;
  exact-CAMB fallback outside its box; Python API only, not wired into the `kidspipe` CLI; validation in `runs/validate_emulator_pipeline*.json`, `docs/REVIEW.md` and `docs/PERFORMANCE.md` section 6).
  It is an approximation (|Delta chi2| 0.004 at worst on 30 fresh points), covers 32 % of the omch2 prior, and an emulator-based chain has not been compared with a CAMB-based chain in its posterior.
* **MAP**: the finite-difference gradient needs 3 CAMB runs per gradient evaluation; `map` first optimises the cheap 11 parameters with
  omch2/ombh2/h frozen. The single-start result above sits on three prior boundaries (a long sigma8-Omega_m degeneracy is open towards the box edge), so
  the Hessian there is one-sided and is not used as proposal covariance by `mcmc` (it says so). One start does not prove a global maximum; use `--starts`.
* The KV450 posterior is wide and non-Gaussian (Omega_m, sigma8, h, ln A_s hit the prior bounds); chain convergence must be judged with several chains (the CLI prints split R-hat and ESS).
* The fixed `m` shear biases, the fixed cosmology settings (flat, w = -1, one massive neutrino of 0.06 eV) and the HMcode-2016 baryon feedback
  model follow the spec; other non-linear models (HMcode-2020, Takahashi) are not implemented.
* The `fast` path differs from the original binning by up to 0.7 % in mu (see above); the spec anchors are matched to <= 0.56 in chi2 with `fast` and <= 0.08 with `kcap`.
* Hessian step control, boundary handling and the PJ-HPD on non-flat priors are described in `docs/interface_notes/inference.md`.
* Absolute timings above were taken on a machine shared with other jobs; relative costs are reliable, absolute ones are not (observed load slow-down 2.7-3.6x; `docs/PERFORMANCE.md` section 1).
  The speed-up versus the original kcap/CosmoSIS pipeline in `docs/PERFORMANCE.md` section 8 is an estimate: kcap itself was never run.
* Only KV450 cosmic shear (xi+/-) is implemented: no COSEBIs/band powers, no KiDS-1000, no 3x2pt, no Hartlap/ln|C| terms (the spec's likelihood has none).

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q -m "not slow"
```

`tests/test_e2e.py` holds the spec-anchor, cache-correctness, fidelity/backend, robustness and CLI tests; the nested-sampler CLI smoke tests,
the ensemble and the multi-process CLI tests are marked `slow`.

## COSEBIs

COSEBIs (E_n, 5 or 20 modes) are implemented and validated against the staged KV450 E_n data; see [docs/COSEBIS.md](docs/COSEBIS.md)
(numbers, plots `plots/cosebis_*.png`, caveats, reproduce commands). The COSEBIs-only S8 chain is short (split R-hat 1.054) and not a full production run.

## KiDS-Legacy COSEBIs validation
Comparison of the `kidspipe.legacy` COSEBIs pipeline with the official KiDS-Legacy release (chi2 on 3509 chain samples, importance reweighting of S8,
filters, timing): see [docs/KIDS_LEGACY_VALIDATION.md](docs/KIDS_LEGACY_VALIDATION.md); figures `plots/legacy_*.png`; script `scripts/legacy_validate.py`.
