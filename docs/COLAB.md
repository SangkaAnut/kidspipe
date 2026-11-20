# Running kidspipe in Google Colab

This repository is meant to be **cloned into a Colab runtime and run end to end**: the KV450 inference, the COSEBIs and KiDS-Legacy validation, and the full DR5
measurement chain from the public 16.4 GB catalogue. A Colab session is temporary (about 12 h on the free tier, up to about 24 h on Pro; the local disk is wiped at the
end), so everything is built around **resumable stages whose results live on Google Drive**.

Everything below is generated from the code in `notebooks/` (generator: `notebooks/build_notebooks.py`), `scripts/colab/setup_env.py` (per-session setup) and
`scripts/dr5_chain.py` (the DR5 orchestrator).

| notebook | purpose | typical time (free 2 vCPU / Pro 8 vCPU, ESTIMATES) |
|---|---|---|
| `notebooks/00_setup_and_tests.ipynb` | clone, install, environment, `kidspipe check`, `pytest -m 'not slow'` | tests 30-60 min / 10-20 min |
| `notebooks/01_kv450_inference.ipynb` | quick CLI demos; the full `run_kv450_*` chain as resumable stages (`quick` or `documented` profile) | demos: minutes; `quick` chain 2-3 h; `documented` about 24 h / about 9 h |
| `notebooks/02_cosebis_and_legacy.ipynb` | COSEBIs validation, COSEBIs MCMC, re-weighting, KiDS-Legacy `eval` and plots | about 5 h / 1.5 h |
| `notebooks/03_dr5_measurement.ipynb` | download, extract, `bin_slop` 1.5 measurement, product, plots, then the exact run | `measure` about 63 h / 16 h; `exact` about 390 h / 98 h |

All times are **estimates** scaled from the timings recorded on the 22-core development workstation (WSL2): linear scaling in cores, and a Colab vCPU as fast as one of its cores. That is
probably optimistic for a shared Colab host; `scripts/dr5_chain.py status` re-calibrates its DR5 estimates from the first pair measured on your runtime.

## 1. One-time preparation

### 1.1 Access to the private repository: a fine-grained token in Colab Secrets

1. GitHub, Settings, Developer settings, Personal access tokens, **Fine-grained tokens**, Generate new token.
   * Resource owner: the account or organisation that owns the repository.
   * Repository access: **Only select repositories**, and select this repository only.
   * Permissions: **Contents: Read-only** (Metadata read-only is added automatically). Nothing else.
   * Expiration: as short as is convenient (a month, say). Copy the token once.
2. In Colab, left sidebar, key icon (**Secrets**), Add new secret: name **`GH_TOKEN`**, value the token. Switch **Notebook access** on for each notebook that clones (the switch is per notebook).
3. In the setup cell of each notebook set `GITHUB_REPO = "owner/name"`.

The setup cell reads the token with `google.colab.userdata.get("GH_TOKEN")` and passes it to `git` as an HTTP header (`git -c http.extraheader=...`), so the token is not stored in
`.git/config`, is not part of the remote URL and is masked in any error message. **Never paste the token into a cell or a file.** If a notebook output ever shows it, revoke it on GitHub.

Without GitHub access: put a zip of the repository at `MyDrive/kidspipe/kidspipe_repo.zip`; the setup cell unpacks it to `/content/kidspipe`.

### 1.2 Google Drive

The setup cell mounts Drive (`drive.mount('/content/drive')`, you approve once per session) and creates `MyDrive/kidspipe/` with `runs/`, `dr5_work/`, `plots/` and `numba_cache/`.
The free Drive quota (15 GB) is shared with Gmail and Photos. The pipeline needs about 2.2 GB for the whole DR5 chain, less for everything else (see section 4).

## 2. Every session: the setup cell

Each notebook starts with the same form cell ("1. Session setup"). **Run it first in every session.** It

1. mounts Drive;
2. clones the repository to `/content/kidspipe` (or updates an existing clone to the latest commit of the branch);
3. calls `scripts/colab/setup_env.py`, which creates the Drive directories, sets the environment variables below **in the notebook process** (so `!python ...` cells inherit them), pins
   the source mtimes (numba cache, section 7), runs `pip install -e '.[all]'` only if kidspipe or an extra is missing, and prints versions, cores, RAM and disk;
4. sets `WORKERS = os.cpu_count()`, used for every `--workers` / `--threads` in the notebooks.

### Environment variables (defined in `src/kidspipe/paths.py`)

| variable | set by the Colab setup to | meaning |
|---|---|---|
| `KIDSPIPE_RUNS` | `/content/drive/MyDrive/kidspipe/runs` | every run output (persistent). The shipped reference results in the clone's `runs/` stay read-only fallbacks (`kidspipe.paths.find_result`) |
| `KIDSPIPE_DR5_WORK` | `/content/drive/MyDrive/kidspipe/dr5_work` | per-bin extract of the DR5 catalogue, `bin0.npz` .. `bin5.npz`, 1.6 GB |
| `KIDSPIPE_DR5_CATALOGUE` | `/content/dr5/KiDS_Legacy_NS_unblind_final.fits` | the 16.4 GB catalogue, **local disk only** |
| `KIDSPIPE_PLOTS` | `/content/drive/MyDrive/kidspipe/plots` | figures written by the scripts (the clone's `plots/` keeps the shipped references) |
| `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` | `os.cpu_count()` | CAMB / numba threads. Worker-process steps override them with 1 per process in the command |
| `NUMBA_CACHE_DIR` | `/content/drive/MyDrive/kidspipe/numba_cache` | compiled numba functions |
| `KIDSPIPE_ROOT`, `KIDSPIPE_DATA`, `KIDSPIPE_OFFICIAL`, `KIDSPIPE_EMULATOR` | (defaults) | repository root, shipped input data, official KiDS-Legacy release files, emulator artefact |

**Pitfall.** If `KIDSPIPE_RUNS` is not set (for example you open a terminal and skip the setup cell), the runs directory is the clone's `runs/`, which already holds the shipped reference results.
`dr5_chain.py status` would then show every pair as done. Always run the setup cell, or `eval "$(python scripts/colab/setup_env.py --print-env)"` in a shell.

## 3. Which notebooks, in which order

1. **00** once per runtime to see that the machine works (self-test, tests).
2. **01** Part A (quick demos) to get a feel for the speeds. Part B with `PROFILE = "quick"` (a toy-size run of the whole chain, results in `runs_quick`) or `"documented"` (the sizes of
   `docs/RESULTS.md` section 6, scaled so every chain keeps its CPU time).
3. **02** reads the KV450 results from your Drive runs if you ran 01, else from the shipped ones; it can be run on its own.
4. **03** the DR5 chain. Independent of 01 and 02.

Stage markers (`<runs>/stages/<name>.done`) make 01 and 02 resumable: after a lost session re-run the setup cell and the same cell; finished stages are skipped. Inside a stage:

* MAP stages, nautilus nested sampling (checkpoint) and `legacy_validate.py eval` (skips finished rows) resume by themselves;
* `run_kv450_mcmc.py prod` does **not** resume (it writes `prod_c*.npz` checkpoints but a restart tunes and samples again), so it is one uninterrupted block of about 4 h (prod1) or 7 h
  (prod2) on 2 vCPU, 1.3 h and 1.8 h on 8 vCPU. Start it only with that much session left. A production run can be split into independent runs with different `--tag`/`--seed`, but
  the downstream scripts of the documented chain expect the tags `prod1` and `prod2`.

## 4. Resources

| | free | Pro / high-RAM |
|---|---|---|
| vCPU, RAM | 2, 12.7 GB | about 8, about 51 GB |
| session | about 12 h | up to about 24 h |
| local disk | about 100 GB | about 100 GB |

* **RAM.** KV450, COSEBIs and Legacy steps are light (about 310 MB per sampler worker). The DR5 chain runs **one tomographic pair per process**: a cross pair peaks at 4.4-5.0 GB RSS, an auto
  pair at 2.3-2.6 GB (recorded). A process that held many pairs was OOM-killed on the development machine; `dr5_chain.py` never does that and never runs two pairs at once, so 12.7 GB
  suffice. More RAM does not speed anything up.
* **Local disk.** The catalogue: 7.2 GB gz while downloading + 16.4 GB fits (delete the gz afterwards: `--delete-gz`).
* **Drive.** Per-bin npz 1.6 GB; `b1.5` pair files 1.7 MB; exact chunks 346 MB (4656 files of about 75 kB); products a few MB; KV450/COSEBIs/Legacy outputs tens of MB; numba cache a few
  tens of MB. Total about 2.2 GB for the DR5 chain. The 7.2 GB gz only if you choose to keep it on Drive (not recommended).
* **Drive I/O.** The Drive FUSE mount is slow for many small files and for repeated large reads. Keep the catalogue on local disk (the extraction reads all 16.4 GB). The per-bin npz
  (300 MB per bin) are read once at the start of every pair process; `--work-cache /content/dr5_work_local` (a notebook option) copies them from Drive to the local disk once per session.
  Chunk and pair files are written once, atomically.

## 5. The DR5 chain: `scripts/dr5_chain.py`

```
python scripts/dr5_chain.py status                      # what is done / left, estimated hours at this thread count
python scripts/dr5_chain.py download [--delete-gz]      # curl -C - (resumable), gz -> fits on local disk, NAXIS2 = 40,894,394 verified
python scripts/dr5_chain.py extract                     # scripts/dr5_extract.py -> bin0..5.npz in $KIDSPIPE_DR5_WORK
python scripts/dr5_chain.py measure --max-hours 11      # bin_slop 1.5; one process per pair; skips finished pairs
python scripts/dr5_chain.py eval && python scripts/dr5_chain.py product && python scripts/dr5_chain.py plots
python scripts/dr5_chain.py exact --max-hours 11        # bin_slop 0, angle_slop 0, 16 chunks; chunk-level resumable
python scripts/dr5_chain.py assemble && python scripts/dr5_chain.py exact-product && python scripts/dr5_chain.py plots
python scripts/dr5_chain.py compare                     # your results vs the reference results shipped in the repository
python scripts/dr5_chain.py all --max-hours 11 [--no-exact]   # everything in order; run again in the next session
```

Common options: `--threads N` (default `os.cpu_count()`), `--max-hours H`, `--pairs 1-2,3-3|autos|cross|all`, `--order heavy|light|index`, `--force`, `--dry-run`, `--runs-dir`,
`--work-dir`, `--catalogue`, `--work-cache DIR`. `--dry-run` prints every command, the pair order and the estimated times without running or writing anything.

How it behaves:

* **Idempotent; the files are the truth.** A b1.5 pair is done when its npz loads and has `xip, xim, npairs, weight, r_nom`; an exact pair is done when all 136 (auto) or 256 (cross) chunk
  files exist. Pair files and chunks are written atomically by the scripts, so a kill never leaves a half file.
* **One subprocess per tomographic pair**, started with `sys.executable` and the environment above, `OMP_NUM_THREADS` = `--threads`. The science settings are those of the called scripts
  (`dr5_measure_xi_unordered.py run --bin-slop 1.5`; `dr5_measure_xi_chunked.py --all --nchunks 16 --bin-slop 0 --angle-slop 0`); the chain does not change them.
* **Budget (`--max-hours`).** For `measure` no pair is started whose estimated time exceeds the remaining budget (the estimate is the recorded time at 20 threads x 20 / threads x the measured
  machine factor); heavy pairs first, then shorter ones fill the end. A started b1.5 pair always finishes (use `--hard-deadline` to kill it at the deadline instead; that loses the pair). For
  `exact` the running pair is stopped at the deadline (SIGTERM, then SIGKILL after 60 s; stale `.part` files are removed) and **at most the chunk-pair in progress is lost**; `--no-kill`
  only starts pairs whose estimate fits. A chunk-pair near the diagonal is about 9 min at 20 threads, so about 22 min at 8 vCPU and about 1.5 h at 2 vCPU: that is the worst case lost
  at a deadline.
* **Exit status**: 0 complete, **75 not finished (budget): run the same command in a new session**, 1 a step failed, 130 interrupted.
* **State**: `<runs>/dr5_measure/chain_state.json` (sessions, timings, per-pair wall time and machine factors). To stop cleanly between pairs create `<runs>/dr5_measure/chain_STOP`.
* **Inputs that cannot be regenerated.** `product` needs `adoption/adoption.json` and `a1_salvage/a1_salvage.json`, small results of the earlier pipeline. The chain copies them from the repository's
  shipped `runs/` into `$KIDSPIPE_RUNS` when they are missing there.
* **Detached runs.** `scripts/colab/setup_env.launch(cmd, log)` starts a command in its own session with output appended to a log on Drive, `tail(log)` follows it. A cell that blocks for
  hours is lost when the browser tab is closed; a detached job keeps running while the runtime lives (Colab can still reclaim an idle or over-age runtime).

### Download details

`download` fetches `https://kids.strw.leidenuniv.nl/DR5/data_files/KiDS_Legacy_NS_unblind_final.fits.gz` (`kidspipe.paths.CATALOGUE_URL`) with `curl -C -` (retrying and resuming; the size
is compared with the server's `Content-Length`), checks free disk (about 24 GB needed), decompresses with `pigz -dc` (else `gzip -dc`) into `...fits.part`, verifies the header
(`NAXIS2 = 40,894,394` rows via fitsio or astropy, and that the file is at least `NAXIS1 x NAXIS2` bytes) and only then renames it. A truncated gz makes the decompression fail instead of
producing a short file. **Acknowledgement:** the catalogue is a KiDS-Legacy data product; acknowledge KiDS and cite Wright et al. 2024 (A&A 686, A170) and the KiDS-Legacy papers listed at
<https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php> (the repository `README.md` has the full text).

### Session plan

| runtime | plan |
|---|---|
| free, 2 vCPU | Session 1: `download` + `extract` (0.5-1.5 h). Sessions 2-7: `measure --max-hours 11` (about 63 h in total; use `status` for the live estimate). Then `eval`, `product`, `plots`. The exact run (about 390 h) is not practical: compare with the exact product shipped in `runs/dr5_measure/product_exact` instead. |
| Pro, 8 vCPU | Session 1: download + extract + `measure` (budget 22-23 h; about 16 h in total). Then `eval`, `product`, `plots`. Exact run: `exact --max-hours 22` repeated, about 98 h in total (about 5 sessions). Then `assemble`, `exact-product`, `plots`, `compare`. |

## 6. Comparing with the shipped reference results

The repository contains the reference results of every chain (KV450 chains, COSEBIs, Legacy validation, the DR5 b1.5 pair files, the assembled exact pairs and the products). Your own results are written
to `$KIDSPIPE_RUNS` and never overwrite them. Notebook 01 prints your S8 and MAP chi2 next to the shipped ones; notebook 02 shows the Legacy summaries; `dr5_chain.py compare` compares every
pair file and product with the reference (bit-identical, or the maximum difference relative to the largest reference value of each array). Differences at the level of floating-point rounding are
expected between different TreeCorr, numpy or BLAS builds; Monte-Carlo steps (chains) agree only statistically. The agreement criteria and gates of the measurement are those of
`docs/DR5_MEASUREMENT_REPORT.md`; nothing here redefines them.

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| first calls take 10-40 s, every session | numba compiles on a cold cache. The setup cell puts the cache on Drive and pins the mtime of `src/**/*.py` to the commit time, because numba keys its cache on the source file's mtime and a fresh clone would otherwise invalidate it (measured: 11.5 s cold, 1.7 s with the pinned-mtime cache after a re-clone of the same commit). A new commit invalidates it, by design. The clone must stay at `/content/kidspipe` (the cache index also records the path). |
| `pip install` changes numpy or breaks an import | Colab ships numpy 2.x; kidspipe supports numpy 1.26 and 2.x. Do not `pip install` other numpy versions in the same session. If an import fails after the install, Runtime, Restart session, then the setup cell. |
| a DR5 pair is killed, exit -9 / 137, or the runtime says "Your session crashed" | out of memory. A cross pair needs about 5 GB. Restart the runtime (frees RAM), run the setup cell and the same command: finished pairs are skipped. Do not start a second heavy process (another notebook, a test run) alongside. |
| the session ended in the middle of a stage | nothing is lost except the pair or chunk-pair in progress (atomic writes). New session, setup cell, same cell/command. |
| Drive is slow or the mount hangs | many small files on Drive are slow. Keep the catalogue local (`/content/dr5`), use `--work-cache`, avoid `pytest` output or caches on Drive (`-p no:cacheprovider`). If the mount fails, Runtime, Disconnect and delete runtime, then reconnect and re-mount. The setup cell warns and falls back to the local disk (not persistent) if Drive is missing. |
| `Drive quota exceeded` | the free 15 GB is shared with Gmail and Photos. Delete the gz (`--delete-gz`) if you kept it on Drive. |
| `git clone failed` | wrong `GITHUB_REPO`, the token lacks Contents: read for this repository, it expired, or **Notebook access** is off for this notebook. Re-create the token (section 1.1). |
| `download` stops or `curl` reports an error | run `download` again; it resumes from the partial gz. If the decompression fails, the gz is truncated: delete it and download again. |
| `status` shows everything done in a fresh runtime | `KIDSPIPE_RUNS` points at the shipped `runs/`: run the setup cell. |
| the estimated times are far off | the machine factor in `status` is measured from the pairs you ran. Free-tier vCPUs can be 1.5-3x slower than the workstation cores; a noisy host varies by session. |
| a test fails on the free runtime | `pytest -m 'not slow'` skips the `timing` tests on fewer than 8 cores (override: `KIDSPIPE_TIMING_TESTS=1`); no tolerance is relaxed for slow machines. Report anything else. |
| mpi4py / `mpirun` | not needed and not installed by `.[all]`; the Colab notebooks use worker processes (`--workers`). MPI gives no speedup on one machine. |

## 8. Colab usage notes

Colab's limits (session length, idle timeout, available hardware) change and are not guaranteed; the numbers above are typical. Keep the tab open for foreground cells, or use the detached
runs of section 5. Use Colab for the work it is intended for (interactive notebooks and your own research runs): a multi-day DR5 reproduction occupies a runtime for most of the time
and may be better run on a workstation or cluster, where the same `scripts/dr5_chain.py` runs unchanged (set the variables of section 2, or none).
