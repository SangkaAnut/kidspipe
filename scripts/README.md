# scripts/

Every script runs from any working directory: the repository root is found from the script's own location, and every
input / output location goes through `kidspipe.paths` (see the table below). Scripts do not need `PYTHONPATH`; they put
`<root>/src` on `sys.path` themselves (a `pip install -e .` works as well).

Run order within each chain is the order of the tables. Costs are the wall times recorded in `docs/` and `runs/*/*.log` on the
development workstation (22 cores, WSL2) unless marked *est.* (scaled from those, assuming linear scaling in cores; Colab vCPUs
may be slower or faster). Core-hours = wall x processes.

## Paths and environment

| variable | meaning | default |
|---|---|---|
| `KIDSPIPE_RUNS` | where results are WRITTEN (`<runs>`) | `<root>/runs` |
| `KIDSPIPE_DATA` | shipped input data | `<root>/data` |
| `KIDSPIPE_OFFICIAL` | official KiDS-Legacy release files (`KiDS_Legacy_xipm.fits`, `KiDS_Legacy_cosebis.fits`, `cosmosis_xipm/`) | `<data>/kids_legacy` |
| `KIDSPIPE_DR5_WORK` | per-bin DR5 extract `bin0..5.npz` + `extract_info.json` (about 1.6 GB) | `~/kids_dr5_work` |
| `KIDSPIPE_DR5_CATALOGUE` | the uncompressed 16.4 GB DR5 shear catalogue | `<dr5_work>/KiDS_Legacy_NS_unblind_final.fits` |
| `KIDSPIPE_EMULATOR` | linear-transfer emulator artefact | `<data>/emulator/transfer_emulator_v1.npz` |
| `KIDSPIPE_PLOTS` | where figures go (scripts only) | `<root>/plots` |
| `KIDSPIPE_THEIRS` | the earlier pipeline's xi directory (optional, external; see "Investigation scripts") | `/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial` |

Result files used as INPUTS (MAP, production chains, `adoption.json`, `b1_integration.json`, products, legacy chi2 tables, ...)
are found with `kidspipe.paths.find_result(rel)`: `<runs>/<rel>` first, then the copy shipped in `<root>/runs/<rel>`. A Colab
session with `KIDSPIPE_RUNS` on Google Drive therefore starts from the shipped reference results and writes only its own new
ones to Drive. A stage that resumes its own work (`legacy_validate.py eval`, the MCMC and chunk drivers) looks only at `<runs>`,
so it never mistakes a shipped result for progress of your run.

Worker / thread defaults are `os.cpu_count()` everywhere (`--workers`, `--threads`); explicit values still work. Each MCMC / nested
worker needs about 0.3 GB; one TreeCorr process of the DR5 chain needs 2.3 GB (auto pair) to 5 GB (cross pair), so on a 12.7 GB
machine run ONE pair per process and never more than one MPI worker.

KiDS data acknowledgement: results derived from the KiDS-Legacy / DR5 data must acknowledge KiDS and cite Wright et al. 2024
(A&A 686, A170) and the KiDS-Legacy papers, see <https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php>.

---------------------------------------------------------------------------------------------------------------------------------

# CORE scripts

## 1. KV450 xi+/- chain (outputs in `<runs>/kv450_final/`; shared helpers in `run_kv450_common.py`, not a CLI)

| # | script and typical call | inputs | outputs | cost |
|---|---|---|---|---|
| 1 | `run_kv450_map.py --stage explore --starts 36 --workers N` | `data/` (emulator) | `map_explore.json` | 1735 s on 13 procs = 6.3 core-h |
| 2 | `run_kv450_map.py --stage polish --top 6 --workers 6 --polish none` (then `cp map_starts.json map_polish_lbfgs.json`); `--top 3 --polish powell` | `map_explore.json` | `map_starts.json` | ~2 min; ~1.5 core-h |
| 3 | `run_kv450_mapbest.py` | `map_starts.json`, `map_explore.json` | `map_best.json`, `starts.json` | seconds |
| 4 | `run_kv450_exact.py map` | `map_starts.json` | `map_exact_check_powell.json` | exact CAMB, a few min |
| 5 | `run_kv450_chi2split.py [map_best.json]` | `map_best.json` | `chi2_split.json` | seconds |
| 6 | `run_kv450_profile.py --workers N` | `map_best.json` | `profile.json` | est. 1-2 core-h |
| 7 | `run_kv450_mcmc.py pilot --tag pilot1 --chains 6 --burn 1500 --cycles 1500 --starts <runs>/kv450_final/starts.json` | `starts.json` | `pilot1/pilot.npz` + `pilot_c*.npz` | 940-1145 s per chain = 1.8 core-h |
| 8 | `run_kv450_mcmc.py prod --tag prod1 --pilot pilot1 --chains 6 --minutes 80 --tune 600 --seed 21` (and `prod2`, 8 chains, 110 min, seed 22) | `pilot1/pilot.npz` + `pilot_c*.npz` | `prod1/`, `prod2/` (`prod_c*.npz`) | 8 + 14.7 core-h. NOT resumable (checkpoints every 300 s but a restart re-tunes): on Colab use `--minutes` batches with different `--tag`/`--seed` (each chain is an independent valid chain) |
| 9 | `run_kv450_analyze.py prod1,prod2 --burn-frac 0.05` | prod chains | `combined_prod1_prod2/analysis.json` | minutes |
| 10 | `run_kv450_exact.py chain prod1,prod2 200` | prod chains | `exact_check_prod1_prod2.json` | 200 exact CAMB evals, ~10 min |
| 11 | `run_kv450_ml.py --workers N` | `map_best.json` + prod chains (so run after 8) | `ml.json` | ~1 core-h |
| 12 | `run_kv450_nested.py --workers N --z-grid reduced --n-live 2000 --n-eff 8000 --tag nautilus2 --seed 12`; `--z-grid spec --tag nautilus3 --seed 13` | `data/` | `nautilus2/`, `nautilus3/` (resumable hdf5) | 2.2 and 3.2 core-h |
| 13 | `run_kv450_sens.py --variants abar2,abar313,aia0,nomcorr,baseline,reduced,kcap --workers 3 --cycles 4000 --burn 1000 --chains 3`, then `--variants camb_mead --no-chain --workers 1` | `map_best.json`, `pilot1/pilot.npz` | `sens/` (`sens_summary.json` is overwritten per call: keep a copy) | about 2000 s per variant and chain |
| 14 | `run_kv450_reweight.py --tags prod1,prod2 --variants kcap,reduced,camb_mead --n 500 --workers N` | prod chains | `reweight.json` | minutes to ~1 core-h |
| 15 | `run_kv450_report.py` | `analysis.json`, prod chains, `nautilus*/nested*` (missing tags are reported and skipped) | markdown tables on stdout | seconds |
| 16 | `make_plots.py` | MAP, prod chains, profile, benchmark JSON (shipped copies are found) | `plots/01..08*.png` | about 1 min |

Total of steps 1-14: about 33 core-h (16 h wall on 2 vCPU, 4 h on 8 vCPU *est.*); the two production runs alone are 22.7 core-h.
`--chains` options are statistical settings, not parallelism, and keep their defaults.

## 2. COSEBIs (xi+/- best fit to E_n)

| # | script and call | inputs | outputs | cost |
|---|---|---|---|---|
| 1 | `cosebis_validate.py main` | `map_best.json`, prod chains, `data/cosebis/` | `plots/cosebis_01..05,07,08`, `<runs>/cosebis/validation_numbers_main.json` | ~1.5-2 min |
| 2 | `COSEBIS_WALL=1500 cosebis_mcmc.py 5 10 10000 1500 n5` (args: modes, chains, cycles, burn, tag; one process per chain) | prod chains, emulator | `<runs>/cosebis/chain_n5_c*.npz` | ~30 min on 10 procs = ~5 core-h. Moved here from `runs/cosebis/mcmc.py` |
| 3 | `cosebis_validate.py post` | `chain_n5_c*.npz`, prod chains | `plots/cosebis_06_posterior.png`, `validation_numbers_post_n5.json`, `reweight_n5.npz` (cache) | ~8 min (cached afterwards) |

## 3. KiDS-Legacy COSEBIs validation against the official chain

| # | script and call | inputs | outputs | cost |
|---|---|---|---|---|
| 1 | `legacy_validate.py eval --workers N` (single-threaded workers: set `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`; or `mpirun -n N+1 ... eval --mpi`) | official chain in `data/kids_legacy/official_cosebis/` | `<runs>/legacy_val/chi2.npz`, `selection.npz` (resumable), operator cache `<runs>/legacy/cache` | ~37 min on 22 cores; *est.* 1.8-2 h on 2 vCPU, 35-45 min on 8 vCPU |
| 2 | `legacy_validate.py plots` (calls `legacy_plots.py`) | `chi2.npz`, `selection.npz` (yours, else the shipped copy: no eval needed) | `plots/legacy_01..05*.png`, `summary.json` | minutes |
| 3 | `legacy_validate.py timing` (once with `OMP_NUM_THREADS=4`, once with 1) | official chain | `timing_t{4,1}.json` | seconds |
| 4 | `legacy_ia_scale_diag.py [n=200]` (optional) | `chi2.npz`, `selection.npz` | `ia_scale_diag.json` | minutes |

## 4. DR5 measurement chain (needs the public catalogue; see `docs/DR5_MEASUREMENT_REPORT.md`)

`dr5_chain.py` is the end-to-end runner (maintained separately) and calls the scripts below with exactly these arguments.
Persistent storage (`KIDSPIPE_RUNS`, `KIDSPIPE_DR5_WORK` on Drive) is mandatory on Colab: steps 2 and 8 resume per pair /
per chunk-pair across sessions.

| # | script and call | inputs | outputs | cost |
|---|---|---|---|---|
| 1 | `dr5_extract.py [--catalogue P] [--out D] [--chunk N] [--force]` | uncompressed catalogue (16.4 GB, local disk; download URL printed when missing; a `.gz` is refused) | `bin0..5.npz`, `extract_info.json` (atomic, skipped when complete) | 432 s with the file on local disk; RAM about 2-3 GB (est.); I/O bound |
| 2 | `dr5_measure_xi_unordered.py run --pairs I-J --bin-slop 1.5 --threads N` (one process per pair; `--pairs autos` / `all` exist) | `bin*.npz` | `<runs>/dr5_measure/xi_unordered/b1.5/Bin{I}_Bin{J}.npz`, `manifest.json` (resumable per pair) | total 6.3 h at 20 threads (auto 564-683 s, cross 1170-1356 s per pair); *est.* 16 h at 8 threads, 63 h at 2; RAM 4.4-5.0 GB per cross pair |
| 3 | `dr5_measure_xi_unordered.py eval --pairs all --bin-slop 1.5` | pair files, release, optionally `a1_salvage.json` and the earlier pipeline's xi | `b1.5/eval.json`, `EVAL.md` | minutes |
| 4 | `dr5_b1_integration.py` | release only | `<runs>/dr5_measure/b1_integration.json` | 3 s |
| 5 | `dr5_make_product.py make` | pair files, `adoption.json` (shipped), `b1_integration.json` | `<runs>/dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz`, `README.md`, `fullvector_checks.json`, `FULLVECTOR.md` | seconds |
| 6 | `dr5_make_product.py stepd` | product, legacy chain, `<runs>/legacy/cache` | merges step D into `fullvector_checks.json` | 16 s |
| 7 | `dr5_make_plots.py a4` then `dr5_make_plots.py plots` | `bin5.npz`, `bin0.npz` (a4); product, release (plots) | `product/a4_recheck.json` (268 s at 16 threads); `plots/dr5_01..06*.png` (about 15 s). Plot 2 omits its two ours-vs-theirs panels when the earlier pipeline's xi is absent; plot 6 is skipped without `a4_recheck.json` | |
| 8 | `dr5_measure_xi_chunked.py --all --pairs I-J --nchunks 16 --threads N --bin-slop 0 --angle-slop 0` (one process per pair), `--status`, then `--assemble --nchunks 16` | `bin*.npz` | `<runs>/dr5_measure/xi_chunked/chunks/` (4656 chunk-pairs, 346 MB, atomic and resumable), `assembled/ns/` | 39.3 h at 20 threads (3 sessions; 1.56-2.21 h per pair); *est.* 98 h at 8 threads, 390 h at 2 (infeasible); RAM 2.3-2.6 GB (auto), 4.3-5.0 GB (cross); `--mpi` gives no speedup on one machine |
| 9 | `dr5_make_exact_product.py` | chunk files and assembled pairs, the bin_slop 1.5 product, release | `<runs>/dr5_measure/product_exact/` (`dr5_xipm_cosebis_exact.npz`, `README.md`, `EXACT_VS_B1P5.md`, `exact_vs_b1p5.json`) | 53 s |
| 10 | `dr5_plot_exact_vs_b1p5.py`, `dr5_plot_exact_values.py` | `product_exact/` products (yours, else shipped) | `plots/dr5_exact_vs_b1p5.png`, `dr5_08_*.png`, `dr5_09*_*.png` | seconds each |

Optional / report helpers: `dr5_plot_report_figs.py` (figures for `docs/PIPELINE_ARCHITECTURE_COMPARISON.tex` into `docs/figs/`,
or `$KIDSPIPE_PLOTS/figs`), `dr5_mpi_chunk_compare.py NEW_DIR REF_DIR [--out f.json]` (compares chunk files of two runs, e.g. MPI vs serial).

Interface used by the DR5 runner (unchanged by the portability pass): all `--threads` defaults moved from fixed numbers to
`os.cpu_count()`; `dr5_extract.py` gained `--catalogue/--out/--chunk/--force`; nothing else in the call syntax changed.

## 5. Emulator retraining (optional: the shipped `data/emulator/transfer_emulator_v1.npz` is enough to run everything)

| step | call | outputs | cost |
|---|---|---|---|
| 1 | `train_emulator.py generate --workers N` | `<runs>/emulator_train/{train,train_aug,test}.npz` (about 520 MB), `kgrid.npy`, `zgrid.npy` | ~17 core-h (1051 + 384 + 256 CAMB runs) |
| 2 | `train_emulator.py fit` (joblib LOO-GP, `os.cpu_count()` jobs) | the artefact (`$KIDSPIPE_EMULATOR`, default the SHIPPED file, which is overwritten: set the variable to keep it) | minutes |
| 3 | `train_emulator.py validate` | `<runs>/emulator_train/validation.json` | minutes |

---------------------------------------------------------------------------------------------------------------------------------

# INVESTIGATION / one-off scripts (not part of any run order; do not run on Colab)

They record how decisions in `docs/DR5_MEASUREMENT_REPORT.md` and `runs/dr5_measure/*.md` were reached. A mechanical portability
pass was made (repository root from `__file__`, work/release locations from `kidspipe.paths`, case typo `KiDs_Legacy...` fixed
by using `paths.official_dir()`); they were not restructured and keep writing under `<root>/runs/dr5_measure/...`. Scripts
marked **EXT** read the earlier pipeline's outputs under `/mnt/d/HI-gamma-correlation/pipeline/...` (external, not in the
repository; the script header says so) and cannot run without that directory. **EXTRA** needs `dr5_extract_extra.py` output.

| script | what it was for | external inputs |
|---|---|---|
| `dr5_a1_gate.py` | redefined A1: sample-identity and weight gates against the official products | extract |
| `dr5_a1_salvage.py` | A1 salvage; its `a1_salvage.json` is an input of the evaluator (shipped) | **EXT** |
| `dr5_a1_salvage_verify.py` | independent verification of the A1 salvage | **EXT** |
| `a1_threshold_calibration/` (`acf, calib, power, standardised, stdcommon, verify_standardised, verify_analyse`) | A1b threshold calibration and standardised-criterion study | **EXT** |
| `dr5_adopt_01.py` | re-derives the adopted earlier-pipeline numbers (`adoption.json`, shipped) through this project's code | **EXT** |
| `dr5_a4_estimator.py` | A4: independent pair counter vs TreeCorr on a catalogue with a real shear field | none |
| `dr5_debug_paircount.py` | debugging the independent pair counter against TreeCorr | none |
| `dr5_amplitude_check.py` | xi+ amplitude of a 10 % subsample vs the official FITS | extract |
| `dr5_area.py` | converged sky-area measurement, area-vs-density test | extract |
| `dr5_cut_scan.py`, `dr5_diagnose_ratio.py`, `dr5_fix_ratio.py`, `dr5_probe_coarse.py` | hunt for the 1.417x pair-count excess (selection cuts, duplicates, probes) | extract, **EXTRA** |
| `dr5_extract_extra.py` | diagnostic extraction of 12 extra catalogue columns (973 s, 3.4 GB, writes `<dr5_work>/extra/`) | catalogue |
| `dr5_extract_check_f64.py` | integrity check of the float64 extract against float32 and the earlier extract | extract, **EXT** |
| `dr5_convention.py`, `dr5_pair_convention_test.py` | shear sign and pair-count (ordered vs unordered) conventions | extract |
| `dr5_sign_convention.py`, `dr5_sign_verify.py` | shear sign convention (parity, pilot, galaxy-galaxy lensing) and its verification | extract, **EXT** |
| `dr5_sigma_e.py`, `dr5_sigma_e_verify.py` | sigma_e gate against the release header, and its verification | extract, **EXTRA** |
| `dr5_binslop_test.py` | is bin_slop 1.5 usable (speed, accuracy) on a subsample | extract |
| `dr5_angle_slop_study.py`, `dr5_angle_slop_report.py`, `dr5_angle_slop_fullpair.py` | (bin_slop, angle_slop) accuracy / cost study | extract, **EXT** (study) |
| `dr5_chunk_diagnose.py`, `dr5_chunk_exactness.py` (superseded verdict), `dr5_chunk_verify.py`, `dr5_chunk_proof.py` | why naive chunking lost 3.3 %, and proofs that folded chunk summation is exact | extract |
| `dr5_measure_xi.py` | earlier all-pairs measurement driver (ordered pairs; superseded by `dr5_measure_xi_chunked.py` / `_unordered.py`) | extract |
| `dr5_pilot_bin6.py`, `dr5_pilot_bin6_exact.py`, `dr5_pilot_bin6_slop.py`, `dr5_pilot_verify.py`, `dr5_pilot_verify_slop.py` | Bin6_Bin6 pilot at several (bin_slop, angle_slop) and its verification | extract, **EXT** (not `_slop`, `_verify_slop`) |
| `dr5_plot_autos.py` | auto pairs at bin_slop 1.5 vs a float32 run (needs the obsolete `b1.5_f32/`) | obsolete float32 run |
| `verify_angle_slop_exactness.py`, `_exactness_driver.py`, `_recompute.py`, `_southpatch.py` | independent verification of the angle_slop study | extract, **EXT** (`_recompute`) |
| `verify_slop15_autos_memory.py`, `_recompute.py`, `_robustness.py`, `_oursvstheirs_{a,c,c_an,npairs}.py` | independent verification of the bin_slop 1.5 auto pairs (memory scaling, recomputation, ours vs the earlier pipeline) | extract, **EXT** (`oursvstheirs_*`) |
