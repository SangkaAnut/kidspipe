#!/usr/bin/env python3
"""Generate the Colab notebooks (plain json, no nbformat dependency).

    python notebooks/build_notebooks.py          # writes notebooks/0?_*.ipynb next to this file

The notebooks are generated so that the shared setup cell and the runtime tables exist exactly once (here). Edit this file, not the
.ipynb files, then re-run it. Every notebook is nbformat 4.5: cells with cell_type / id / source / metadata (+ outputs, execution_count
for code cells), no stored outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------------------------------------ building blocks
def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


class NB:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "id": f"{self.prefix}-{len(self.cells):02d}", "metadata": {}, "source": _lines(text)})

    def code(self, text: str, form: bool = False) -> None:
        meta = {"cellView": "form"} if form else {}
        self.cells.append({"cell_type": "code", "id": f"{self.prefix}-{len(self.cells):02d}", "metadata": meta, "outputs": [], "execution_count": None,
                           "source": _lines(text)})

    def dump(self, name: str) -> dict:
        nb = {
            "cells": self.cells,
            "metadata": {
                "colab": {"provenance": [], "name": name, "toc_visible": True},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return nb


# ------------------------------------------------------------------------------------------------ shared pieces
SECRETS_MD = """
### One-time preparation: access to the private repository

The repository is private, so the notebook needs a GitHub **fine-grained personal access token**. The token is stored in **Colab Secrets**
and is never written into a cell, a file or the output.

1. GitHub, Settings, Developer settings, Personal access tokens, **Fine-grained tokens**, Generate new token. Resource owner: the account that owns the
   repository. Repository access: **Only select repositories**, choose this repository. Permissions: **Contents: Read-only** (Metadata is added
   automatically). Pick a short expiry. Copy the token once.
2. In Colab, left sidebar, the key icon (**Secrets**), Add new secret: name `GH_TOKEN`, value = the token, switch **Notebook access** on for this notebook
   (the switch is per notebook).
3. Edit `GITHUB_REPO` in the setup cell below (`owner/name`).

No GitHub access? Put a zip of the repository at `MyDrive/kidspipe/kidspipe_repo.zip`; the setup cell unpacks it instead of cloning.
"""

RUNTIME_MD = """
### Which runtime?

| | free Colab | Colab Pro / high-RAM |
|---|---|---|
| vCPU | 2 | about 8 |
| RAM | 12.7 GB | about 51 GB |
| session length | about 12 h (idle and background limits apply) | up to about 24 h |
| local disk | about 100 GB (wiped at the end of the session) | the same |
| Google Drive | persistent, shared 15 GB quota (free) | persistent |

The notebooks use `os.cpu_count()` for threads and worker processes. All times in these notebooks are **estimates** scaled from timings recorded on a
22-core development workstation (WSL2): linear scaling in cores is assumed and a Colab vCPU is assumed to be as fast as one of its cores, which is probably
optimistic. Treat them as lower bounds on a shared, noisy Colab host.
"""

BOOTSTRAP = r'''
#@title 1. Session setup (run this first, in EVERY session) { display-mode: "form" }
#@markdown Private repository to clone (`owner/name`); the token comes from the Colab Secret `GH_TOKEN`.
GITHUB_REPO = "OWNER/REPO"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
#@markdown Persistent outputs live on Drive; the 16 GB catalogue lives on the local disk (never on Drive).
DRIVE_ROOT = "/content/drive/MyDrive/kidspipe"  #@param {type:"string"}
CATALOGUE_DIR = "/content/dr5"  #@param {type:"string"}
REPO_DIR = "/content/kidspipe"

import base64, importlib, os, pathlib, shutil, subprocess, sys

def _git(args, token=None, cwd=None):
    """git, with the token passed as an HTTP header (not stored in .git/config); the token is masked in anything returned."""
    cmd, secret = ["git"], []
    if token:
        b64 = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraheader=Authorization: Basic {b64}"]
        secret = [token, b64]
    r = subprocess.run(cmd + args, cwd=cwd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    for s in secret:
        out = out.replace(s, "***")
    return r.returncode, out.strip()

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab: Drive is not mounted; DRIVE_ROOT must be a writable directory.")

token = None
try:
    from google.colab import userdata
    token = userdata.get("GH_TOKEN")
except Exception as e:                                   # secret missing, or notebook access not granted
    print(f"No GH_TOKEN secret available ({type(e).__name__}); using an existing clone or the Drive zip if there is one.")

repo = pathlib.Path(REPO_DIR)
url = f"https://github.com/{GITHUB_REPO}.git"
if (repo / "pyproject.toml").exists():
    rc, out = _git(["fetch", "--depth", "1", "origin", BRANCH], token, cwd=repo)
    if rc == 0:
        _git(["reset", "--hard", "FETCH_HEAD"], cwd=repo)
        print("repository updated to the latest", BRANCH)
    else:
        print("could not update (using the existing clone):", out[-300:])
elif token or not pathlib.Path(DRIVE_ROOT, "kidspipe_repo.zip").exists():
    if GITHUB_REPO.startswith("OWNER/"):
        raise SystemExit("Edit GITHUB_REPO (owner/name) in this cell, and add the GH_TOKEN secret (see the markdown above).")
    rc, out = _git(["clone", "--depth", "1", "--branch", BRANCH, url, REPO_DIR], token)
    if rc != 0:
        raise SystemExit("git clone failed (wrong GITHUB_REPO, or the token lacks Contents: read access?):\n" + out[-500:])
    print("cloned", GITHUB_REPO)
else:
    tmp = pathlib.Path("/content/_unzip"); shutil.rmtree(tmp, ignore_errors=True)
    shutil.unpack_archive(str(pathlib.Path(DRIVE_ROOT, "kidspipe_repo.zip")), str(tmp))
    top = [p for p in tmp.iterdir() if p.is_dir()]
    shutil.move(str(top[0]) if len(top) == 1 and not (tmp / "pyproject.toml").exists() else str(tmp), REPO_DIR)
    print("unpacked the repository from the Drive zip")

sys.path.insert(0, f"{REPO_DIR}/scripts/colab")
import setup_env; importlib.reload(setup_env)
INFO = setup_env.setup(REPO_DIR, DRIVE_ROOT, CATALOGUE_DIR)      # env vars, pip install -e '.[all]' if needed, versions
os.chdir(REPO_DIR)
WORKERS = os.cpu_count() or 1
'''


def setup_cells(nb: NB, title_extra: str = "") -> None:
    nb.md("## Session setup\n\nA Colab runtime is fresh in every session: **run the next cell first, every time**. It mounts Drive, clones or updates the repository, "
          "installs the package if needed and sets the environment variables (`KIDSPIPE_RUNS`, `KIDSPIPE_DR5_WORK`, `KIDSPIPE_DR5_CATALOGUE`, `KIDSPIPE_PLOTS`, thread counts, "
          "numba cache on Drive) described in `docs/COLAB.md`." + title_extra)
    nb.code(BOOTSTRAP, form=True)


ACK_MD = """
**Acknowledgement.** The DR5 / KiDS-Legacy data are KiDS products. Acknowledge KiDS and cite Wright et al. 2024 (A&A 686, A170) and the KiDS-Legacy papers
(<https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php>); the repository's `README.md` has the full text. The repository is private; do not share the data or results
outside the terms of the KiDS release.
"""


# ================================================================================================ 00
def nb00() -> NB:
    nb = NB("nb00")
    nb.md("""
# 00. Setup and tests

Clone the private repository into the Colab runtime, mount Google Drive, install `kidspipe`, set the environment, and run the self-test and the fast test suite.
Run this notebook once to check that a runtime works; the other notebooks repeat the same setup cell (a session starts from scratch every time).

| notebook | what it does |
|---|---|
| **00_setup_and_tests** | this one |
| 01_kv450_inference | KV450 xi+/- likelihood, MAP, MCMC, nested sampling: quick demos and the full chain |
| 02_cosebis_and_legacy | COSEBIs validation and the KiDS-Legacy comparison |
| 03_dr5_measurement | the full DR5 measurement chain from the public catalogue (resumable over many sessions) |
""" + ACK_MD)
    nb.md(RUNTIME_MD)
    nb.md(SECRETS_MD)
    setup_cells(nb)
    nb.md("""
### What is where

* `/content/kidspipe`: the repository (local disk, recreated every session).
* `KIDSPIPE_RUNS = MyDrive/kidspipe/runs`: everything the pipeline writes (persistent). The reference results that ship in the repository (`/content/kidspipe/runs`) are read as
  fallback inputs (`kidspipe.paths.find_result`) but never overwritten.
* `KIDSPIPE_DR5_WORK = MyDrive/kidspipe/dr5_work`: the six per-bin npz of the DR5 extract (1.6 GB).
* `KIDSPIPE_DR5_CATALOGUE = /content/dr5/KiDS_Legacy_NS_unblind_final.fits`: the 16.4 GB catalogue, on local disk only.
* `NUMBA_CACHE_DIR = MyDrive/kidspipe/numba_cache`: compiled numba functions. The setup cell pins the source mtimes to the commit time so the cache stays valid across
  fresh clones of the same commit (first call of a fresh cache: about 10-30 s of JIT compilation; with a valid cache about 2 s).
""")
    nb.code("""
import os
for k in ("KIDSPIPE_ROOT", "KIDSPIPE_RUNS", "KIDSPIPE_DR5_WORK", "KIDSPIPE_DR5_CATALOGUE", "KIDSPIPE_PLOTS", "NUMBA_CACHE_DIR", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    print(f"{k:24s} {os.environ.get(k)}")
from kidspipe import paths
print("\\npaths.runs_dir()        ", paths.runs_dir())
print("paths.dr5_catalogue()   ", paths.dr5_catalogue())
""")
    nb.md("### Self-test\n\n`kidspipe check` evaluates chi2 at the spec test cosmology against the stored anchor (tolerance 0.5) and times the model stages. About 10-40 s including numba compilation.")
    nb.code("!python -m kidspipe check")
    nb.md("""
### Fast test suite

`pytest -m 'not slow'`. The suite checks the theory chain, the likelihood, the samplers, COSEBIs, the KiDS-Legacy pipeline and the DR5 measurement code (the DR5 tests that need
data skip when it is absent). **Runtime estimate: roughly 10-20 min on 8 vCPU, 30-60 min on the free 2-vCPU runtime** (the full suite took 10-12 min on the 22-core development
workstation, and it uses worker processes in several tests). Wall-clock assertions carry the `timing` marker and are skipped on machines with fewer than 8 cores unless
`KIDSPIPE_TIMING_TESTS=1`; no tolerance is relaxed for slow machines. Use `-x` to stop at the first failure.
""")
    nb.code("!python -m pytest -q -m 'not slow' -p no:cacheprovider --durations=10")
    nb.md("Optional: the tests marked `slow` (validation against high-accuracy references; long on 2 vCPU, hours possible).")
    nb.code("""
RUN_SLOW = False   #@param {type:"boolean"}
if RUN_SLOW:
    !python -m pytest -q -m 'slow' -p no:cacheprovider --durations=10
""")
    nb.md("### Record the environment on Drive (provenance of every later result)")
    nb.code("""
import pathlib
runs = pathlib.Path(os.environ["KIDSPIPE_RUNS"])
!pip freeze > {runs}/pip_freeze.txt
!git -C {REPO_DIR} log -1 --format='%H %cd %s' > {runs}/commit.txt
print((runs / "commit.txt").read_text()); print("saved", runs / "pip_freeze.txt", "and", runs / "session_info.json")
""")
    nb.md("""
### Next

* `01_kv450_inference.ipynb`: quick demos first (minutes), then the full chain.
* `02_cosebis_and_legacy.ipynb`: COSEBIs validation and the KiDS-Legacy comparison (the shipped KV450 chains are used when you have not re-run the KV450 chain).
* `03_dr5_measurement.ipynb`: the DR5 measurement from the public catalogue. `!python scripts/dr5_chain.py status` shows at any time what is done and what is left.
""")
    nb.code("!python scripts/dr5_chain.py status --threads {WORKERS}")
    return nb


# ================================================================================================ 01
KV450_TABLE = """
### Expected wall times of the full chain (ESTIMATES)

Core-hours are from the logs on the development workstation; wall = core-hours / min(processes, vCPU). Assumes a Colab vCPU as fast as a workstation core.

| step (script) | core-h | processes | wall, 2 vCPU | wall, 8 vCPU |
|---|---|---|---|---|
| MAP explore (`run_kv450_map.py --stage explore --starts 36`) | 6.3 | 13 | 3.2 h | 0.8 h |
| MAP polish, L-BFGS (`--top 6 --polish none`) | 0.1 | 6 | 5 min | 2 min |
| MAP polish, Powell (`--top 3`) | 1.5 | 3 | 0.75 h | 0.5 h |
| `run_kv450_mapbest`, `_exact map`, `_chi2split` | small | 1 | 10 min | 10 min |
| profile (`run_kv450_profile.py`) | 1-2 (not logged) | 12 | 0.5-1 h | 0.2 h |
| pilot chains (`run_kv450_mcmc.py pilot`) | 1.8 | 6 | 0.9 h | 0.3 h |
| production 1 (6 chains x 80 min of CPU) | 8 | 6 | 4 h | 1.3 h |
| production 2 (8 chains x 110 min of CPU) | 14.7 | 8 | 7.3 h | 1.8 h |
| analyze, exact-CAMB check of 200 samples, maximum likelihood | about 1 | 1-6 | 0.5 h | 0.3 h |
| nested sampling (`nautilus2`, `nautilus3`) | 2.2 + 3.2 | 8 | 1.1 + 1.6 h | 0.3 + 0.4 h |
| sensitivity variants (`run_kv450_sens.py`) | about 7 | 3 | 3.3 h | 2.2 h |
| re-weighting (`run_kv450_reweight.py`) | 0.5-1 | 6 | 0.5 h | 0.2 h |
| **total** | about 45 | | **about 24 h (3 sessions)** | **about 9 h** |

The `quick` profile below runs the same stages at toy size (a few chains of 10 minutes): about 2-3 h on 2 vCPU, to check that the whole chain works. Its results go to
`MyDrive/kidspipe/runs_quick` and are not the published numbers.

**Resumability.** Each stage is skipped when its marker `<runs>/stages/<name>.done` exists, so after a lost session you re-run the setup cell and the same stage cell. Inside a
stage: MAP stages and nested sampling (nautilus checkpoint) resume; **`run_kv450_mcmc.py prod` does not** (it writes `prod_c*.npz` checkpoints every 5 minutes but a restart tunes
and samples again). A production stage is therefore one uninterrupted block of about 4 h (prod1) or 7 h (prod2) on 2 vCPU: start it with at least that much session left.
"""


def nb01() -> NB:
    nb = NB("nb01")
    nb.md("""
# 01. KV450 inference

KV450 xi+/- likelihood (14 parameters), MAP, MCMC and nested sampling, with the emulated linear transfer function (`data/emulator`). Part A: quick CLI demos sized for the
free 2-vCPU runtime (minutes). Part B: the full chain `scripts/run_kv450_*.py` in the order of `docs/RESULTS.md` section 6, as resumable stages.
The reference results of this analysis are shipped in the repository's `runs/kv450_final/`, so notebook 02 and the plots work without running Part B.
""" + ACK_MD)
    nb.md(RUNTIME_MD)
    nb.md(SECRETS_MD)
    setup_cells(nb)
    nb.md("""
## Part A. Quick demos (free runtime)

Outputs go to `$KIDSPIPE_RUNS/<name>/` on Drive. RAM is not an issue (about 310 MB per sampler worker).
""")
    nb.code("""
import os, json, pathlib
RUNS = pathlib.Path(os.environ["KIDSPIPE_RUNS"])
def show(path, keys):
    d = json.load(open(path)); print({k: d[k] for k in keys if k in d})
""")
    nb.md("**A1. Likelihood at the spec test cosmology** (chi2, ln L, S8). Seconds.")
    nb.code("!python -m kidspipe loglike --name colab_loglike --json")
    nb.md("**A2. MAP search, small** (2 starts, 40 iterations, one round; about 10-25 min on 2 vCPU, estimate; the documented size of this demo took 275 s + 66 s Hessian on 22 loaded cores).")
    nb.code("""
!python -m kidspipe map --starts 2 --maxiter 40 --rounds 1 --polish none --workers {min(WORKERS, 2)} --worker-threads 1 --name colab_map
show(RUNS / "colab_map" / "map.json", ["chi2"])
""")
    nb.md("**A3. Blocked-Metropolis MCMC, tiny** (2 chains, 100 burn-in + 200 cycles; minutes). Not converged, only a pipeline check.")
    nb.code("""
!python -m kidspipe mcmc --map {RUNS}/colab_map/map.json --chains 2 --workers {min(WORKERS, 2)} --worker-threads 1 --burn 100 --cycles 200 --name colab_mcmc
""")
    nb.md("**A4. Nested sampling smoke test** (nautilus, 50 live points, at most 300 likelihood calls; a few minutes).")
    nb.code("""
!python -m kidspipe nested --sampler nautilus --nlive 50 --n-eff 100 --max-calls 300 --workers {WORKERS} --name colab_nested
""")
    nb.md("**A5. Timing per parameter block on this machine** (about a minute).")
    nb.code("!python -m kidspipe bench --workers 1 --n-slow 3 --n-medium 10 --n-fast 100 --name colab_bench")
    nb.md("""
## Part B. The full KV450 chain
""" + KV450_TABLE)
    nb.code("""
#@title Profile and helpers { display-mode: "form" }
PROFILE = "quick"  #@param ["quick", "documented"]
import os, pathlib
from setup_env import stage
W = WORKERS
w = lambda n: max(1, min(n, W))                      # worker processes for a step that can use n at most
if PROFILE == "quick":
    os.environ["KIDSPIPE_RUNS"] = f"{DRIVE_ROOT}/runs_quick"     # keep toy results apart from the real ones
else:
    os.environ["KIDSPIPE_RUNS"] = f"{DRIVE_ROOT}/runs"
RUNS = pathlib.Path(os.environ["KIDSPIPE_RUNS"]); OUT = RUNS / "kv450_final"; OUT.mkdir(parents=True, exist_ok=True)
S = "python3 scripts"
mins = lambda m, chains: round(m * chains / w(chains))          # --minutes is WALL time: keep the CPU time per chain of the documented run
print(f"profile {PROFILE}: {W} vCPU, outputs in {OUT}")
""", form=True)
    nb.md("### B1. Global MAP (resumable stages)")
    nb.code("""
if PROFILE == "documented":
    stage("map_explore",  f"{S}/run_kv450_map.py --stage explore --starts 36 --workers {w(13)}")
    stage("map_polish_lbfgs", f"{S}/run_kv450_map.py --stage polish --top 6 --workers {w(6)} --polish none && cp {OUT}/map_starts.json {OUT}/map_polish_lbfgs.json")
    stage("map_polish_powell", f"{S}/run_kv450_map.py --stage polish --top 3 --workers {w(3)} --polish powell")
else:
    stage("map_explore",  f"{S}/run_kv450_map.py --stage explore --starts 6 --workers {w(6)}")
    stage("map_polish_lbfgs", f"{S}/run_kv450_map.py --stage polish --top 2 --workers {w(2)} --polish none && cp {OUT}/map_starts.json {OUT}/map_polish_lbfgs.json")
    stage("map_polish_powell", f"{S}/run_kv450_map.py --stage polish --top 1 --workers 1 --polish powell")
stage("map_best",  f"{S}/run_kv450_mapbest.py")
stage("map_exact", f"{S}/run_kv450_exact.py map")
stage("chi2_split", f"{S}/run_kv450_chi2split.py")
if PROFILE == "documented":
    stage("profile", f"{S}/run_kv450_profile.py --workers {w(12)}")
""")
    nb.md("""
### B2. Posterior: pilot, two production runs, analysis

Not resumable inside `prod` (see above). `--minutes` is wall time per chain: the cell scales it by chains / processes so every chain keeps the CPU time of the documented run.
Check `Runtime, Change runtime type` first if you want the 8-vCPU machine.
""")
    nb.code("""
if PROFILE == "documented":
    pilot = "--chains 6 --burn 1500 --cycles 1500"; prod1 = f"--chains 6 --minutes {mins(80, 6)} --tune 600 --seed 21"; prod2 = f"--chains 8 --minutes {mins(110, 8)} --tune 600 --seed 22"
else:
    pilot = "--chains 2 --burn 300 --cycles 300"; prod1 = "--chains 2 --minutes 10 --tune 200 --seed 21"; prod2 = "--chains 2 --minutes 10 --tune 200 --seed 22"
stage("pilot", f"{S}/run_kv450_mcmc.py pilot --tag pilot1 {pilot} --starts {OUT}/starts.json")
stage("prod1", f"{S}/run_kv450_mcmc.py prod --tag prod1 --pilot pilot1 {prod1}")
stage("prod2", f"{S}/run_kv450_mcmc.py prod --tag prod2 --pilot pilot1 {prod2}")
stage("analyze", f"{S}/run_kv450_analyze.py prod1,prod2 --burn-frac 0.05")
stage("exact_chain", f"{S}/run_kv450_exact.py chain prod1,prod2 200")
if PROFILE == "documented":
    stage("ml", f"{S}/run_kv450_ml.py --workers {w(6)}")                # needs the production chains (despite its position in RESULTS.md)
""")
    nb.md("### B3. Nested sampling (nautilus; checkpointed, so an interrupted stage resumes)")
    nb.code("""
if PROFILE == "documented":
    stage("nautilus2", f"{S}/run_kv450_nested.py --workers {w(8)} --z-grid reduced --n-live 2000 --n-eff 8000 --tag nautilus2 --seed 12")
    stage("nautilus3", f"{S}/run_kv450_nested.py --workers {w(8)} --z-grid spec --n-live 2000 --n-eff 8000 --tag nautilus3 --seed 13")
else:
    stage("nautilus3", f"{S}/run_kv450_nested.py --workers {w(8)} --z-grid reduced --n-live 200 --n-eff 500 --tag nautilus3 --seed 13")
stage("report", f"{S}/run_kv450_report.py > {OUT}/report.txt && tail -n 40 {OUT}/report.txt")   # tables of docs/RESULTS.md sections 2.4 and 3
""")
    nb.md("### B4. Systematic sensitivity and re-weighting")
    nb.code("""
if PROFILE == "documented":
    stage("sens_main", f"{S}/run_kv450_sens.py --variants abar2,abar313,aia0,nomcorr,baseline,reduced,kcap --workers {w(3)} --cycles 4000 --burn 1000 --chains 3 --map-file map_best.json "
                       f"&& cp {OUT}/sens/sens_summary.json {OUT}/sens/sens_summary_main.json")     # run_kv450_sens.py overwrites sens_summary.json on every call
    stage("sens_camb", f"{S}/run_kv450_sens.py --variants camb_mead --no-chain --workers 1 --map-file map_best.json")
    stage("reweight", f"{S}/run_kv450_reweight.py --tags prod1,prod2 --variants kcap,reduced,camb_mead --n 500 --workers {w(6)}")
else:
    stage("sens_main", f"{S}/run_kv450_sens.py --variants baseline,reduced --workers {w(2)} --cycles 600 --burn 200 --chains 2 --map-file map_best.json")
    stage("reweight", f"{S}/run_kv450_reweight.py --tags prod1,prod2 --variants kcap,reduced --n 100 --workers {w(2)}")
""")
    nb.md("### B5. Compare with the reference results shipped in the repository\n\nThe documented run reproduces the shipped numbers up to Monte-Carlo noise (different chains); the `quick` profile will not match closely.")
    nb.code("""
import json
def s8(p):
    d = json.load(open(p))["stats"]["S8"]; return f"S8 = {d['mean']:.4f} +- {d['std']:.4f}"
mine = OUT / "combined_prod1_prod2" / "analysis.json"
ref = pathlib.Path(REPO_DIR) / "runs" / "kv450_final" / "combined_prod1_prod2" / "analysis.json"
print("yours   :", s8(mine) if mine.exists() else "(not produced yet)")
print("shipped :", s8(ref))
print("MAP chi2 yours  :", json.load(open(OUT / "map_best.json"))["chi2"] if (OUT / "map_best.json").exists() else "-")
print("MAP chi2 shipped:", json.load(open(pathlib.Path(REPO_DIR) / "runs/kv450_final/map_best.json"))["chi2"])
""")
    nb.md("""
### Notes

* If a stage fails or the session dies, fix the cause, re-run the setup cell, then the same stage cell: finished stages are skipped, `force` is available as `stage(name, cmd, force=True)`.
* `$KIDSPIPE_RUNS` on Drive is read first by every later step, the shipped `runs/` second: a step you did not re-run falls back to the shipped reference result. In `documented`
  mode run the steps in order; in an empty `runs_quick` the missing earlier steps come from the shipped results.
* Free tier: keep the browser tab open; long cells are cut when the runtime is reclaimed (idle or maximum duration). Stage markers make that harmless except inside `prod`.
""")
    return nb


# ================================================================================================ 02
def nb02() -> NB:
    nb = NB("nb02")
    nb.md("""
# 02. COSEBIs and the KiDS-Legacy validation

* **COSEBIs validation** (`docs/COSEBIS.md`): E_n modes from the KV450 xi+/- best fit, the COSEBIs-only MCMC and the re-weighting of the xi+/- chain.
* **KiDS-Legacy validation** (`docs/KIDS_LEGACY_VALIDATION.md`): the `kidspipe.legacy` COSEBIs model against the official KiDS-Legacy chain (3509 chain rows, seed 7).

Both read the KV450 results from `$KIDSPIPE_RUNS/kv450_final` when you produced them in notebook 01, otherwise from the copy shipped in the repository, so this notebook runs on its own.
""" + ACK_MD)
    nb.md(RUNTIME_MD)
    nb.md(SECRETS_MD)
    setup_cells(nb)
    nb.md("""
### Expected wall times (ESTIMATES, scaled from the 22-core workstation)

| step | core-h | wall, 2 vCPU | wall, 8 vCPU |
|---|---|---|---|
| `cosebis_validate.py main` (plots 01-05, 07; numbers) | 0.1 | 5-10 min | 2 min |
| `cosebis_mcmc.py 5 10 ...` (10 chains, 5 modes) | 5 | 2.1 h | 0.5 h |
| `cosebis_validate.py post` (re-weighting, plot 06) | about 1 | 0.5 h | 10 min |
| `pytest tests/test_cosebis.py` | | 5 min | 2 min |
| `legacy_validate.py eval` (3509 samples, resumable) | 3.5 s/sample/core: about 3.4 | 1.8-2 h | 35-45 min |
| `legacy_validate.py plots`, `timing` | small | 5 min | 2 min |
""")
    nb.code("""
import os, pathlib, shutil
from setup_env import stage, sh
W = WORKERS
RUNS = pathlib.Path(os.environ["KIDSPIPE_RUNS"]); PLOTS = pathlib.Path(os.environ["KIDSPIPE_PLOTS"])
S = "python3 scripts"
T4 = f"OMP_NUM_THREADS={min(W, 4)} NUMBA_NUM_THREADS={min(W, 4)}"     # the validation scripts: a few threads
T1 = "OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1"                          # worker-process steps: one thread per process
def show(*names):
    from IPython.display import Image, display
    for n in names:
        p = PLOTS / n
        print(p) if p.exists() else print("missing:", p)
        if p.exists(): display(Image(filename=str(p)))
""")
    nb.md("## Part A. COSEBIs")
    nb.code("stage('cosebis_main', f'{T4} {S}/cosebis_validate.py main')\nshow('cosebis_03_routes.png', 'cosebis_04_data_vs_model.png')")
    nb.md("""
The COSEBIs-only MCMC: one process per chain, `COSEBIS_WALL` = wall-clock limit **per chain** in seconds. The documented run is 10 chains x 1500 s on a machine with about 2 effective
cores per chain-group; the cell scales the limit by chains / min(chains, vCPU) so that each chain gets the same CPU time. Writes `<runs>/cosebis/chain_n5_c*.npz`.
""")
    nb.code("""
NCH = 10
wall = round(1500 * NCH / min(NCH, W))
stage('cosebis_mcmc', f'COSEBIS_WALL={wall} {S}/cosebis_mcmc.py 5 {NCH} 10000 1500 n5')
stage('cosebis_post', f'{T4} {S}/cosebis_validate.py post')
show('cosebis_06_posterior.png')
""")
    nb.code("!python -m pytest -q -p no:cacheprovider tests/test_cosebis.py")
    nb.md("""
## Part B. KiDS-Legacy validation

`eval` evaluates the COSEBIs likelihood at 3509 samples of the official chain with one process per worker (one thread each) and **resumes** from `<runs>/legacy_val/chi2.npz`: if the
session ends, run the cell again. The first call builds the W_n operator tables (10-20 s each) into `<runs>/legacy/cache`.
""")
    nb.code("""
rc = sh(f'{T1} {S}/legacy_validate.py eval --workers {W}')
print('exit status', rc, '(re-run this cell to resume if it did not finish)')
""")
    nb.code("""
if rc == 0:
    stage('legacy_plots', f'{T4} {S}/legacy_validate.py plots')
    stage('legacy_timing4', f'OMP_NUM_THREADS={min(W, 4)} {S}/legacy_validate.py timing')
    stage('legacy_timing1', f'OMP_NUM_THREADS=1 {S}/legacy_validate.py timing')
    show('legacy_01_chi2_vs_official.png', 'legacy_03_S8_official_vs_ours.png')
""")
    nb.md("Optional IA-scale diagnostic (reads `legacy_val/chi2.npz` and `selection.npz`; must run from the repository root):")
    nb.code("""
RUN_DIAG = False   #@param {type:"boolean"}
if RUN_DIAG:
    sh(f'OMP_NUM_THREADS={min(W, 4)} python3 scripts/legacy_ia_scale_diag.py 200', cwd=REPO_DIR)
""")
    nb.md("### Compare with the shipped reference results")
    nb.code("""
import json
mine, ref = RUNS / "legacy_val" / "summary.json", pathlib.Path(REPO_DIR) / "runs" / "legacy_val" / "summary.json"
for name, p in (("yours", mine), ("shipped", ref)):
    print(name, ":", (json.dumps(json.load(open(p)))[:400] + " ...") if p.exists() else "(none yet)")
""")
    nb.md("### Keep the figures\n\nThe scripts write to `$KIDSPIPE_PLOTS` (on Drive). The repository's own `plots/` keeps the shipped reference figures untouched.")
    nb.code("print(sorted(p.name for p in PLOTS.glob('*.png')))")
    return nb


# ================================================================================================ 03
def nb03() -> NB:
    nb = NB("nb03")
    nb.md("""
# 03. DR5 measurement: the full chain from the public catalogue

Everything from the 16.4 GB KiDS-Legacy (DR5) shear catalogue to the xi+/- and COSEBIs products, as `scripts/dr5_chain.py` stages that **resume across sessions**:

| stage | what | output |
|---|---|---|
| `download` | 7.2 GB gz from the KiDS server (resumable), decompressed to the 16.4 GB fits on the **local** disk | `/content/dr5/KiDS_Legacy_NS_unblind_final.fits` |
| `extract` | one pass over the catalogue, six per-bin npz | `dr5_work/bin0..5.npz` (1.6 GB, Drive) |
| `measure` | TreeCorr `bin_slop` 1.5, one process per tomographic pair (21 pairs) | `runs/dr5_measure/xi_unordered/b1.5/Bin*_Bin*.npz` |
| `eval`, `product` | comparison with the official release, xi+/- and COSEBIs product, full-vector checks, step D | `runs/dr5_measure/product/` |
| `plots` | the `dr5_*` figures | `$KIDSPIPE_PLOTS` |
| `exact` | exact run (`bin_slop` 0, `angle_slop` 0, 16 chunks), 4656 chunk-pairs, chunk-level resumable | `runs/dr5_measure/xi_chunked/chunks/` (346 MB) |
| `assemble`, `exact-product` | assemble the chunks, exact product and its comparison with the `bin_slop` 1.5 product | `runs/dr5_measure/product_exact/` |

The science settings are those of the scripts (the chain only decides which pair runs when). The shipped repository already contains the reference results of this chain
(`runs/dr5_measure/`); your run goes to `$KIDSPIPE_RUNS` on Drive and `compare` shows the differences.
""" + ACK_MD)
    nb.md(RUNTIME_MD)
    nb.md("""
### Cost and sessions (ESTIMATES)

| | free, 2 vCPU | Pro, 8 vCPU |
|---|---|---|
| download + decompress + extract | 0.5-1.5 h (network and disk bound) | the same |
| `measure` (21 pairs, `bin_slop` 1.5) | about 63 h = 6 sessions of 11 h (cross pair 3.3-3.8 h, auto pair 1.6-1.9 h) | about 16 h = 2 sessions of 11 h or 1 of 24 h (cross 50-57 min, auto 24-28 min) |
| `eval`, `product`, `plots` (a4 re-check included) | about 30-60 min | about 15 min |
| `exact` (4656 chunk-pairs) | about 390 h = 36 sessions: **not practical**, use the shipped exact product as the reference | about 98 h = 9 sessions of 11 h (a chunk-pair near the diagonal about 22 min) |
| `assemble`, `exact-product` | minutes | minutes |

Recorded on the 22-core workstation with 20 threads: `measure` 6.32 h (autos 564-683 s, crosses 1170-1356 s), `exact` 39.3 h in three sessions (1.30-2.31 h per pair).
The estimates scale these by 1/threads. After the first pair finishes on your machine, `dr5_chain.py status` re-calibrates them from the measured time.

**Memory.** One tomographic pair per process (a process that held many pairs was killed by the OOM killer): a cross pair peaks at about 5 GB RSS, an auto pair at about 2.6 GB.
Fine on the free 12.7 GB runtime; nothing is gained from more RAM because pairs run one after the other, each with all threads.

**Drive space.** Per-bin npz 1.6 GB; `b1.5` pairs 1.7 MB; exact chunks 346 MB; products a few MB; the 7.2 GB gz only if you keep it on Drive (the notebook keeps it local and can delete it,
`--delete-gz`). Total about 2.2 GB of the free 15 GB Drive quota. **Never keep the 16.4 GB fits on Drive**: the FUSE mount is slow and the extraction reads all 16.4 GB.

**Sessions.** Every command is idempotent; the truth is read from the files. `--max-hours H` is the budget of one invocation: no pair is started whose estimated time exceeds what is left; the exact run
is stopped at the deadline (it loses at most the chunk-pair in progress). Exit status 75 means "not finished, run the same command in a new session". The state and timings are in
`MyDrive/kidspipe/runs/dr5_measure/chain_state.json`.
""")
    nb.md(SECRETS_MD)
    setup_cells(nb)
    nb.code("""
#@title Session parameters { display-mode: "form" }
MAX_HOURS = 11  #@param {type:"number"}
#@markdown free Colab: about 11 (the session ends at about 12 h); Pro: up to about 23
DELETE_GZ = True  #@param {type:"boolean"}
LOCAL_WORK_CACHE = True  #@param {type:"boolean"}
#@markdown copy the per-bin npz from Drive to the local disk once per session (faster than reading Drive in every pair process)
import os, pathlib
from setup_env import sh, launch, tail
W = WORKERS
CHAIN = f"python3 scripts/dr5_chain.py"
CACHE = " --work-cache /content/dr5_work_local" if LOCAL_WORK_CACHE else ""
COMMON = f"--threads {W}"
LOG = pathlib.Path(os.environ["KIDSPIPE_RUNS"]) / "dr5_measure" / "chain_logs" / "session.log"
def run(sub, extra=""):
    rc = sh(f"{CHAIN} {sub} {COMMON} {extra}")
    print(f"\\nexit status {rc}" + ("  -> not finished: run this cell again in a new session" if rc == 75 else "  (failed: see the output above)" if rc else "  (complete)"))
    return rc
""", form=True)
    nb.md("## What is done and what is left")
    nb.code("run('status')")
    nb.md("""
## Option A: one cell, re-run in every session

Runs every stage in order and stops cleanly when the budget `MAX_HOURS` is used up. Run it, wait, and when the session ends or the cell stops with status 75, open a new session,
run the setup cell and this cell again. Remove `--no-exact` to continue into the exact run (Pro / 8 vCPU).
""")
    nb.code("""
rc = run("all", f"--max-hours {MAX_HOURS}{CACHE} --no-exact" + (" --delete-gz" if DELETE_GZ else ""))
""")
    nb.md("""
## Option B: stage by stage

### B1. Session A: download, extract (about 0.5-1.5 h)
""")
    nb.code("""
run("download", "--delete-gz" if DELETE_GZ else "")
run("extract")
""")
    nb.md("""
### B2. Sessions B...: the `bin_slop` 1.5 measurement

Re-run this cell in every session until it reports "all requested pairs done". `--order heavy` (default) starts the long cross pairs first and fills the end of the budget with the shorter auto pairs.
""")
    nb.code("""
rc = run("measure", f"--max-hours {MAX_HOURS}{CACHE}")
""")
    nb.md("""
Alternative for long runs: start the stage **detached** so that closing the browser tab does not stop it (it keeps running while the runtime lives), then poll the log.
Stop it with `p.terminate()` (the chain terminates the running pair cleanly; finished pairs stay on disk).
""")
    nb.code("""
START_DETACHED = False   #@param {type:"boolean"}
if START_DETACHED:
    p = launch(f"{CHAIN} measure {COMMON} --max-hours {MAX_HOURS}{CACHE}", LOG)
""")
    nb.code("print(tail(LOG, 25))")
    nb.md("""
### B3. Evaluate, make the product, plots (needs all 21 pairs)

`product` runs the release-only integration check, builds the xi+/- and COSEBIs product with the full-vector checks, and the model chi2 (step D). `plots` runs the estimator re-check on bins 1 and 6
(`a4`, a few minutes) and writes the `dr5_01...06` figures to `$KIDSPIPE_PLOTS`.
""")
    nb.code("""
run("eval")
run("product")
run("plots")
""")
    nb.md("""
### B4. Compare with the shipped reference results

`compare` lists, for each pair file and each product, whether your arrays are bit-identical to the reference results shipped in the repository or how large the difference is
relative to the reference. Differences at the level of floating-point rounding are expected when TreeCorr, numpy or the BLAS differ from the development workstation; the agreement criteria of the
measurement are in `docs/DR5_MEASUREMENT_REPORT.md` and are not redefined here.
""")
    nb.code("""
run("compare")
from IPython.display import Image, display
for n in ("dr5_01_xipm_vs_official.png", "dr5_03_En_vs_official.png"):
    for label, d in (("yours", os.environ["KIDSPIPE_PLOTS"]), ("shipped", "/content/kidspipe/plots")):
        f = pathlib.Path(d) / n
        print(label, f, "(missing)" if not f.exists() else "")
        if f.exists(): display(Image(filename=str(f), width=700))
""")
    nb.md("""
### B5. The exact run (Pro / 8 vCPU; chunk-level resumable)

One process per tomographic pair, exact settings (`bin_slop` 0, `angle_slop` 0, 16 chunks). At the deadline the running pair is stopped; at most the chunk-pair in progress is lost (it is recomputed next time).
Use `--no-kill` to only start pairs whose estimate fits. Re-run until it reports "all requested pairs complete". `--pairs 1-1,2-3` selects pairs.
""")
    nb.code("""
rc = run("exact", f"--max-hours {MAX_HOURS}{CACHE}")
""")
    nb.code("""
if rc == 0:
    run("assemble")
    run("exact-product")
    run("plots")
    run("compare")
""")
    nb.md("""
### Archive the results

Everything persistent is already on Drive (`MyDrive/kidspipe/`). To take the products away:
""")
    nb.code("""
import shutil
arc = shutil.make_archive("/content/dr5_products", "zip", os.environ["KIDSPIPE_RUNS"], "dr5_measure/product") if (pathlib.Path(os.environ["KIDSPIPE_RUNS"]) / "dr5_measure/product").exists() else None
print(arc)
""")
    nb.md("""
### Troubleshooting

* **`status` shows everything "done" immediately.** `KIDSPIPE_RUNS` points at the repository's shipped `runs/`, not at Drive: run the setup cell (it sets the Drive path).
* **A pair was killed (exit -9 / 137).** Out of memory. Make sure nothing else holds RAM (restart the runtime), and that only one pair runs. The chain never runs two pairs at once.
* **Download stops.** Run `download` again; it resumes (`curl -C -`). If the decompression fails the gz is truncated: delete it and download again.
* **Slow `measure`.** Check `status` for the machine factor; Colab vCPUs can be 2x slower than the workstation cores. Check `LOCAL_WORK_CACHE`.
* **The session was reclaimed.** Nothing is lost except the pair (or chunk-pair) in progress: new session, setup cell, same stage cell.
""")
    return nb


# ------------------------------------------------------------------------------------------------ main
NOTEBOOKS = {"00_setup_and_tests.ipynb": nb00, "01_kv450_inference.ipynb": nb01, "02_cosebis_and_legacy.ipynb": nb02, "03_dr5_measurement.ipynb": nb03}


def main() -> int:
    for name, fn in NOTEBOOKS.items():
        nb = fn().dump(name)
        (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", HERE / name, f"({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
