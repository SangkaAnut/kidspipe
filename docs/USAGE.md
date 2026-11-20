# kidspipe usage

All commands are run from the repository root (a clone of this repository, or the Colab checkout) with the package installed or on the path.
Install with `pip install -e '.[all]'` (see the README for the extras); the `kidspipe` command is then equivalent to `python3 -m kidspipe`.
Input and output locations are controlled by environment variables (`KIDSPIPE_DATA`, `KIDSPIPE_RUNS`, ...; table in the README, defined in `src/kidspipe/paths.py`).

```bash
export PYTHONPATH=$PWD/src          # or: pip install -e .  (then PYTHONPATH is not needed)
export OMP_NUM_THREADS=4            # CAMB threads of one process (the CLI sets 4 if you do not)
export NUMBA_NUM_THREADS=4          # numba (HMcode/Limber) threads; numba ignores OMP_NUM_THREADS and uses ALL cores by default
                                    # (the `kidspipe` CLI sets it from OMP_NUM_THREADS / --threads / --worker-threads; the Python API does not)
python3 -m kidspipe --help
python3 -m kidspipe <command> --help
```

Every command takes the model options below, a deterministic `--seed` (where randomness is involved) and
`--out DIR` (default `runs/<name>/`, i.e. `$KIDSPIPE_RUNS/<name>/` when `KIDSPIPE_RUNS` is set; `--name` defaults to the command name).

| model option | meaning |
|---|---|
| `--backend hmcode\|camb` | non-linear P(k): stand-alone numba HMcode-2016 (default) or CAMB `mead` |
| `--fidelity fast\|kcap` | `fast`: exact bin-averaged Hankel operator (default). `kcap`: FFT Hankel + node-trapezoid binning (the spec's "faithful" path) |
| `--z-grid spec\|reduced` | 150-node or 41-node P(k,z) grid |
| `--n-ell`, `--n-nodes` | ell nodes (400) and Limber integration nodes (2000) |
| `--threads N` | CAMB threads in this process (default min(4, ncpu)) |
| `--no-mcorr`, `--no-cterm` | switch off the m-bias correction / the c-term (sensitivity studies only) |

## Parameter vector

14 varied parameters in this order (`kidspipe.params.PARAM_NAMES`):
`omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon, A_IA, delta_c, A_c, dz1..dz5`. Fixed: flat, w = -1,
m_nu = 0.06 eV (one massive neutrino). Bounds and priors: `kidspipe/params.py` (spec s1).

A point is given by `--preset` (`test`, `scan`, `start`), `--theta-file FILE` and `--set NAME=VALUE`
(increasing priority). `FILE` is JSON (`{"omch2": 0.12, ...}`, a list of 14 numbers, or a `kidspipe map` result)
or ini-like text (`name = value`, `#` and `;` comments); names not given are taken from the preset.

* `test`: omch2=0.1, ombh2=0.022, h=0.7, n_s=0.96, ln_1e10_A_s=3.0, A_baryon=2, no IA / shifts, A_c=1, delta_c=0
  (the point of the spec's chi2 = 196.06 anchor).
* `scan`: omch2=0.12, ombh2=0.0222, h=0.7, n_s=0.97, ln_1e10_A_s=2.8, A_baryon=3.13 (spec amplitude scan, near its minimum).
* `start`: the KV450 sampler start values (`params.START`).

## Commands

### `check`: self-test

```bash
python3 -m kidspipe check [--scan] [--camb] [--tol 0.5]
```

Builds the model (first call compiles/loads the numba kernels: ~1 s with the on-disk cache, ~10 s on a fresh
checkout), then checks chi2 at the spec test cosmology against 196.06 (default tolerance 0.5), the derived
sigma8/Omega_m/S8 against the spec anchors, that `A_c` and `delta_c` really reach the likelihood, that
out-of-range/NaN parameters give `-inf`, and prints timings. `--scan` adds the four amplitude-scan anchors of the spec
(tolerance 1.0), `--camb` compares with the other non-linear backend. Exit status 0 = all passed.

### `loglike`: one point

```bash
python3 -m kidspipe loglike                                  # test cosmology
python3 -m kidspipe loglike --preset scan --set A_IA=1 --json-stdout
python3 -m kidspipe loglike --theta-file mypoint.json --out runs/mypoint --mu
```

Prints chi2 (195 data points), ln L, ln prior (normalised), ln posterior, and sigma8, Omega_m, S8. `--mu` also writes the
pure binned theory vector (195) to `mu.txt` (before delta_c / A_c terms and m-bias, which the likelihood handles).
Exit status 2 if the point is outside the prior box.

### `map`: maximum a posteriori + Hessian

```bash
python3 -m kidspipe map --starts 4 --workers 4 --worker-threads 4 --name map
```

Multi-start bounded L-BFGS-B with a derivative-free polish (`--polish none|nelder-mead|powell|both`), followed by a
finite-difference Hessian (covariance, 1-sigma errors) at the best optimum. Starts: the `scan` preset, the KV450 start,
then random prior draws. Writes `map.json` (parameters, chi2, ln post, derived, Hessian covariance, per-start results).
`--workers N` runs one start per worker process.

Cost warning: the gradient is a finite difference and three of the 14 parameters (omch2, ombh2, h) each need a new CAMB transfer
function (about 0.8 s with 4 threads, ~2.8 s with 1 thread on the development workstation, 22 cores under WSL2), so one gradient costs ~2.5-8 s. Budget
`--maxiter`, `--rounds` and `--starts` accordingly.

### `mcmc`: chains

```bash
# 4 parallel blocked-Metropolis chains started at the MAP, proposal covariance from the MAP Hessian
python3 -m kidspipe mcmc --map runs/map/map.json --chains 4 --workers 4 --worker-threads 4 \
        --burn 300 --cycles 2000 --name kv450_chains
# affine-invariant ensemble sampler over a pool
python3 -m kidspipe mcmc --method ensemble --walkers 36 --steps 300 --workers 18 --name kv450_ens
```

*Blocked Metropolis.* One cycle = one joint update of the six parameters (omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon) [one new CAMB
transfer function: ~0.8 s on the loaded, shared development workstation], then `--rep-medium` (2) updates of (n_s, ln_1e10_A_s, A_baryon)
[~17 ms: exact rescale of the cached transfer function + new P_nl], then `--rep-fast` (6) updates of the eight fast parameters
(A_IA, delta_c, A_c, dz1..5) [~1.4 ms]. `--separate-slow` moves (omch2, ombh2, h) alone in the first block instead (worse for the
omch2 - A_s degeneracy, see README). Proposal covariances are the conditional covariances of a learned full covariance (adaptation during
burn-in only, so the production chain is a valid Markov chain). Burn-in cycles are not written to `chain.npz`. `--map` uses the MAP as
start point and its Hessian covariance as the prior proposal covariance unless the MAP lies on a prior boundary (then a default
2 %-of-box covariance is used; `--force-hessian` overrides).

*Outputs* (`runs/<name>/`): `chain.npz` + `chain.txt` (all chains concatenated, columns = 14 parameters, `sigma8`,
`omega_m`, `S8`; `log_post` = ln prior + ln L per sample; `meta` gives chain lengths), `chain_c<i>.npz/.txt` per chain,
`summary_blocked.txt` (mean, std, median, 68 % limits, marginal HPD, projected joint HPD), `mcmc_summary.json` (acceptance per block,
split R-hat and ESS per parameter across chains, evaluation counts, ms per call by move type). Load with
`kidspipe.inference.chains.load_chain("runs/<name>/chain")`.

Derived quantities are recorded when each cosmology is first evaluated, so they cost nothing extra.

### `nested`: nautilus / dynesty

```bash
python3 -m kidspipe nested --sampler nautilus --nlive 500 --n-eff 2000 --workers 20 --name nested
python3 -m kidspipe nested --sampler dynesty --nlive 500 --dlogz 0.5 --workers 20
```

Writes a weighted chain (`chain.npz/.txt`, with `weights`, `log_like`, `log_post`), `summary.txt` and `nested_summary.json`
(ln Z; nautilus gives no evidence error). Every nested-sampling point is a new cosmology, so cost per likelihood call is that of a CAMB
transfer function (see the bench numbers in the README). Use `--workers`, `--max-calls`, `--timeout` (nautilus) to bound a run;
`--checkpoint` (nautilus) writes `nautilus.hdf5` and resumes from it.
`log_z` uses the normalised priors of `params.py` (truncated Gaussians for delta_c, A_c, dz).

### `bench`: stage timings

```bash
python3 -m kidspipe bench [--workers 8 --worker-threads 1]
```

Moves one parameter at a time (every point new, differing from the previous one only in that parameter) and reports the mean
cost per move for each of the 14 parameters, split into the model stages (`linear` = CAMB transfer, `pnl` = HMcode,
`limber_power`, `limber_cl`, `hankel`), plus the likelihood-only cost and per-block means (slow / medium / fast). With
`--workers` it also measures the throughput of the worker pool on new cosmologies.

## MPI

Every command with `--workers` (`map`, `mcmc`, `nested`, `bench`) also runs over MPI ranks instead of local worker processes
(`src/kidspipe/mpi.py`, mpi4py.futures; tested with mpi4py 4.1.2 and Open MPI 5.0.10). Rank 0 runs the command; ranks
1..N-1 are the workers, each with its own model (built once by the same initialiser as the process pool) and
`--worker-threads` threads (numba and BLAS are limited on each worker). Tasks are handed out dynamically. N must be >= 3.

```bash
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
mpirun -n 9 -x PYTHONPATH -x OMP_NUM_THREADS python3 -m kidspipe nested --sampler nautilus --mpi --worker-threads 1
mpirun -n 5 -x PYTHONPATH -x OMP_NUM_THREADS python3 -m kidspipe map --starts 4 --mpi        # 4 workers
```

* `--workers` is ignored under `--mpi`; the worker count is N-1. Results are identical to the process pool with the same
  number of workers and seed (MAP, nautilus checked bit for bit).
* On one machine MPI is not faster than `--workers`; it is for runs across several nodes (and for clusters where MPI is the
  only way to use more than one node). Keep N x `--worker-threads` within the cores.
* Starting a program with `mpirun -n >1` but without `--mpi` is refused (every rank would run the whole job).
* If a task or the root fails, queued tasks are cancelled and the error is raised once the running tasks finish; a
  failing worker initialiser prints its traceback with the rank.

Batch scripts with `--mpi` (the `mpi` extra needs an MPI library, e.g. `apt install libopenmpi-dev openmpi-bin`; MPI is optional and not needed in Colab) (same model: rank 0 schedules and writes, ranks 1..N-1 evaluate):

```bash
mpirun -n 11 -x PYTHONPATH python3 scripts/legacy_validate.py eval --mpi        # Legacy chain samples, 10 workers
mpirun -n 7  -x PYTHONPATH python3 scripts/run_kv450_reweight.py --mpi --variants kcap,reduced
mpirun -n 3  -x PYTHONPATH python3 scripts/dr5_measure_xi_chunked.py --all --mpi --nchunks 16 --threads 10 --bin-slop 0 --angle-slop 0
```

The DR5 driver hands out the chunk-pairs of one tomographic pair at a time (heaviest first); each worker keeps only the
current pair's catalogues, about 2.5 GB (auto pair) to 5 GB (cross pair), so on a 16 GB machine use at most 2 workers.
Chunk files are written under a temporary name and renamed, so an interrupted run leaves no partial file; finished
chunk-pairs are skipped on restart as before. `--status` and `--assemble` run without mpirun.

## Python API

```python
from kidspipe import api
post = api.build_posterior()                       # Posterior with .model, .likelihood, .recorder
th = api.preset("test")                            # 14-vector
post.log_post(th), post.log_like(th)               # -inf outside the prior box, never raises for bad values
api.log_post_with_derived(post, th)                # (ln post, ln L, {'sigma8','omega_m','S8'})
api.derived_columns(post, samples)                 # (n, 3) for chain post-processing (cached; CAMB fallback)
post.model.mu(th)                                  # pure (195,) theory vector

# samplers and chain utilities
from kidspipe.inference import find_map, blocked_metropolis, load_chain, summary_table
```

`build_posterior(model=None, likelihood=None, **model_kw)` also accepts your own `CosmicShearModel` / `GaussianLikelihood`.

## Tests

```bash
pytest -q -m "not slow"                                     # unit + e2e (fast); pytest finds src/ through pyproject.toml
pytest -q tests/test_e2e.py                                  # incl. the slow nested-sampler smoke tests
```

Tests marked `timing` assert wall-clock speed and run only with `KIDSPIPE_TIMING_TESTS=1` or on machines with at least 16 cores.

## DR5 measurement chain

The DR5 scripts (`scripts/dr5_*.py`) read the public catalogue from `$KIDSPIPE_DR5_CATALOGUE` (default `~/kids_dr5_work/KiDS_Legacy_NS_unblind_final.fits`; download URL in
`data/README.md`), keep the per-bin extract in `$KIDSPIPE_DR5_WORK` and write to `$KIDSPIPE_RUNS/dr5_measure/`. `scripts/dr5_chain.py` runs the core chain step by step
(`scripts/README.md`); the recorded command sequence and run times are in `docs/DR5_MEASUREMENT_REPORT.md` section 6. Use one process per tomographic pair (memory, see that section).
