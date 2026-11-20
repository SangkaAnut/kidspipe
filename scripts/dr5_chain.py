#!/usr/bin/env python3
"""Resumable orchestrator for the FULL KiDS-Legacy (DR5) measurement chain, built for sessions that die (Google Colab).

Every step is idempotent and the truth about "what is done" is read from the files on disk, so the same command can be
re-run after any interruption:

  python scripts/dr5_chain.py status                      what is done / left, estimated hours at this machine's thread count
  python scripts/dr5_chain.py download [--delete-gz]      7.2 GB gz from the KiDS server (resumable) -> 16.4 GB fits (local disk)
  python scripts/dr5_chain.py extract                     scripts/dr5_extract.py: the 6 per-bin npz (about 1.6 GB)
  python scripts/dr5_chain.py measure --max-hours 11      bin_slop 1.5 product, one process per tomographic pair
  python scripts/dr5_chain.py eval                        measure_xi_unordered.py eval (needs all 21 pairs)
  python scripts/dr5_chain.py product                     dr5_b1_integration.py, dr5_make_product.py make, ... stepd
  python scripts/dr5_chain.py exact --max-hours 11        exact run (bin_slop 0, angle_slop 0, 16 chunks), chunk-level resumable
  python scripts/dr5_chain.py assemble                    assemble the chunks into one npz per pair (needs all 4656 chunk-pairs)
  python scripts/dr5_chain.py exact-product               dr5_make_exact_product.py
  python scripts/dr5_chain.py plots                       a4 re-check + the dr5_* figures (+ the exact figures when the exact product exists)
  python scripts/dr5_chain.py compare                     compare what you produced with the reference results shipped in the repository
  python scripts/dr5_chain.py all --max-hours 11          all of the above in order, stops cleanly when the budget is used up

``--dry-run`` prints every command (and the planned order and estimated times) without running anything.

Exit status: 0 everything requested is complete; 75 stopped because the time budget is used up (run the same command again,
in a new session); 1 a step failed.

Settings (never changed here): the science settings are those of the scripts this file calls. bin_slop 1.5 for the product,
bin_slop 0 / angle_slop 0 / 16 chunks for the exact run. This file only decides WHICH pair runs WHEN.

Memory: every tomographic pair runs in its own process (a process that held many pairs was OOM-killed). Peak RSS recorded on the
development machine: b1.5 cross pairs 4.4-5.0 GB, auto pairs about 2.3-2.6 GB.

Time estimates scale the recorded timings (20 threads; b1.5: auto 564-683 s, cross 1170-1356 s; exact: 1.30-2.31 h per pair) with
1/threads. They are ESTIMATES: they assume linear thread scaling and that a Colab vCPU is as fast as a core of the 22-core
development machine (probably optimistic). After a pair has finished here, the measured wall time re-calibrates the later
estimates (``machine factor`` in the status output).

Locations (all overridable, see ``kidspipe.paths``): ``--runs-dir`` / $KIDSPIPE_RUNS, ``--work-dir`` / $KIDSPIPE_DR5_WORK,
``--catalogue`` / $KIDSPIPE_DR5_CATALOGUE. The chain state is written to ``<runs>/dr5_measure/chain_state.json``.
Put ``--catalogue`` on LOCAL disk (never on a Drive mount); ``--work-cache DIR`` copies the per-bin npz to a local directory before
measuring (faster than reading them from Drive in every pair process).

KiDS acknowledgement: the catalogue is a KiDS-Legacy data product. Acknowledge KiDS and cite Wright et al. 2024 (A&A 686, A170)
and the KiDS-Legacy papers listed at https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))

from kidspipe import paths  # noqa: E402

EXIT_BUDGET = 75                      # EX_TEMPFAIL: not finished, run again

# ------------------------------------------------------------------------------------------------ reference data
NBIN = 6
PAIRS = [(i, j) for i in range(1, NBIN + 1) for j in range(i, NBIN + 1)]           # 1-based, same order as official.PAIRS
NCHUNKS = 16
CATALOGUE_ROWS = 40_894_394
CATALOGUE_ROW_BYTES = 401
GZ_GB, FITS_GB = 7.2, 16.4                                                          # sizes quoted by the KiDS server / the readme
REF_THREADS = 20

#: bin_slop 1.5, seconds per pair at 20 threads (runs/dr5_measure/xi_unordered/b1.5/manifest.json)
REF_B15_S = {
    (1, 1): 683, (2, 2): 673, (3, 3): 626, (4, 4): 641, (5, 5): 602, (6, 6): 564,
    (1, 2): 1356, (1, 3): 1292, (1, 4): 1308, (1, 5): 1285, (1, 6): 1235, (2, 3): 1320, (2, 4): 1335, (2, 5): 1286,
    (2, 6): 1228, (3, 4): 1276, (3, 5): 1239, (3, 6): 1184, (4, 5): 1255, (4, 6): 1192, (5, 6): 1170}
#: exact run (bin_slop 0, angle_slop 0, 16 chunks), hours per pair at 20 threads (xi_chunked/SESSION{1,2,3}_CHECK.md)
REF_EXACT_H = {
    (5, 5): 1.56, (5, 6): 1.65, (1, 1): 2.12, (1, 3): 2.16, (1, 5): 2.07, (2, 4): 2.21, (2, 6): 1.87,
    (1, 2): 2.21, (2, 5): 1.86, (3, 3): 1.60, (3, 5): 1.87, (4, 4): 1.75, (4, 5): 1.81, (4, 6): 1.68,
    (1, 4): 2.31, (1, 6): 1.83, (2, 2): 1.88, (2, 3): 1.97, (3, 4): 1.86, (3, 6): 1.74, (6, 6): 1.30}
#: peak RSS (GB) recorded on the development machine (cross / auto), plus a margin: used for a warning only
RAM_NEED_GB = {"cross": 5.5, "auto": 3.0}
#: share of the work of one exact pair in each chunk-pair (a, b), averaged over the 21 pairs (from the recorded chunk files:
#: ordered pairs counted, diagonal chunk of an auto pair halved). Used only to turn "chunks done" into "work done" for estimates.
_W_AUTO = {(0,0):0.053, (0,1):0.0171, (0,15):0.0188, (1,1):0.0526, (1,2):0.0194, (2,2):0.0525, (2,3):0.0166, (3,3):0.0536,
    (3,4):0.0174, (4,4):0.0521, (4,5):0.0108, (5,5):0.0314, (5,6):0.00573, (6,6):0.0438, (6,7):0.012, (7,7):0.0471, (7,8):0.0149,
    (8,8):0.0509, (8,9):0.0131, (9,9):0.0505, (9,10):0.0161, (10,10):0.0535, (10,11):0.0152, (11,11):0.049, (11,12):0.0146,
    (12,12):0.0445, (12,13):0.00376, (13,13):0.0382, (13,14):0.0149, (14,14):0.0481, (14,15):0.0163, (15,15):0.0525}
_W_CROSS = {(0,0):0.053, (0,1):0.0101, (0,15):0.0095, (1,0):0.00735, (1,1):0.0518, (1,2):0.0147, (2,1):0.00634, (2,2):0.0498,
    (2,3):0.0155, (3,2):0.00465, (3,3):0.0492, (3,4):0.0185, (4,3):0.00436, (4,4):0.0461, (4,5):0.0155, (5,4):0.00207,
    (5,5):0.0254, (5,6):0.00784, (6,5):0.00126, (6,6):0.0406, (6,7):0.0115, (7,6):0.00308, (7,7):0.0454, (7,8):0.0115,
    (8,7):0.00495, (8,8):0.0497, (8,9):0.0105, (9,8):0.00407, (9,9):0.0489, (9,10):0.0138, (10,9):0.00443, (10,10):0.0513,
    (10,11):0.0132, (11,10):0.00401, (11,11):0.0474, (11,12):0.0111, (12,11):0.00459, (12,12):0.0439, (12,13):0.00206,
    (13,12):0.00183, (13,13):0.0379, (13,14):0.00634, (14,13):0.00901, (14,14):0.0479, (14,15):0.007, (15,0):0.00924,
    (15,14):0.00948, (15,15):0.0525}

#: small inputs of the product that cannot be regenerated on a fresh machine (they come from the earlier pipeline); copied from
#: the shipped runs/ into <runs>/ when missing there
SEED_FILES = ["dr5_measure/adoption/adoption.json", "dr5_measure/a1_salvage/a1_salvage.json"]

ACK = ("KiDS acknowledgement: the DR5 catalogue is a KiDS-Legacy data product. Acknowledge KiDS and cite Wright et al. 2024 "
       "(A&A 686, A170) and the KiDS-Legacy papers: https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php")


def pname(p) -> str:
    return f"Bin{p[0]}_Bin{p[1]}"


def kind(p) -> str:
    return "auto" if p[0] == p[1] else "cross"


def parse_pairs(spec: str | None) -> list[tuple[int, int]]:
    if not spec or spec == "all":
        return list(PAIRS)
    if spec == "autos":
        return [p for p in PAIRS if p[0] == p[1]]
    if spec == "cross":
        return [p for p in PAIRS if p[0] != p[1]]
    out = []
    for tok in spec.split(","):
        i, j = (int(x) for x in tok.strip().split("-"))
        p = (min(i, j), max(i, j))
        if p not in PAIRS:
            raise SystemExit(f"bad pair '{tok}': use I-J with 1 <= I <= J <= {NBIN}")
        out.append(p)
    return out


def hms(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = max(seconds, 0.0)
    h = seconds / 3600
    return f"{h:.2f} h" if h >= 1 else f"{seconds / 60:.0f} min"


# ------------------------------------------------------------------------------------------------ configuration
def _abs(p) -> Path:
    """Absolute path against the CALLER's cwd (children run with cwd=ROOT and would otherwise read a relative path differently)."""
    return Path(p).expanduser().absolute()


class Cfg:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.runs = _abs(a.runs_dir) if a.runs_dir else _abs(paths.runs_dir())
        self.work = _abs(a.work_dir) if a.work_dir else _abs(paths.dr5_work_dir())
        self.catalogue = _abs(a.catalogue) if a.catalogue else _abs(paths.dr5_catalogue())
        self.work_cache = _abs(a.work_cache) if a.work_cache else None
        self.threads = int(a.threads) if a.threads else (os.cpu_count() or 1)
        self.dry = bool(a.dry_run)
        self.nchunks = int(a.nchunks)
        self.meas = self.runs / "dr5_measure"
        self.b15 = self.meas / "xi_unordered" / "b1.5"
        self.chunks = self.meas / "xi_chunked"
        self.state_file = _abs(a.state) if a.state else self.meas / "chain_state.json"
        self.stop_file = self.meas / "chain_STOP"
        self.shipped = paths.shipped_runs_dir()
        self.t0 = time.time()
        self.virtual = 0.0                                # dry-run clock offset (seconds)
        budget_h = a.max_hours
        self.deadline = None if budget_h is None else self.t0 + budget_h * 3600 - a.margin_min * 60

    def now(self) -> float:
        return time.time() + self.virtual

    def remaining(self) -> float:
        return math.inf if self.deadline is None else self.deadline - self.now()

    @property
    def measure_work(self) -> Path:
        return self.work_cache or self.work

    def env(self, work: Path | None = None) -> dict:
        e = dict(os.environ)
        e["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
        e["PYTHONUNBUFFERED"] = "1"
        e["PYTHONDONTWRITEBYTECODE"] = e.get("PYTHONDONTWRITEBYTECODE", "1")
        e["KIDSPIPE_RUNS"] = str(self.runs)
        e["KIDSPIPE_DR5_WORK"] = str(work or self.work)
        e["KIDSPIPE_DR5_CATALOGUE"] = str(self.catalogue)
        e["OMP_NUM_THREADS"] = str(self.threads)
        e["NUMBA_NUM_THREADS"] = str(self.threads)
        return e


# ------------------------------------------------------------------------------------------------ state file
def load_state(cfg: Cfg) -> dict:
    try:
        return json.loads(cfg.state_file.read_text())
    except Exception:
        return {"version": 1, "sessions": [], "records": {"b15": {}, "exact": {}}, "steps": {}}


def save_state(cfg: Cfg, st: dict, **extra) -> None:
    if cfg.dry:
        return
    st.update(extra)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st["paths"] = {"runs": str(cfg.runs), "work": str(cfg.work), "catalogue": str(cfg.catalogue)}
    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.state_file.with_name(f".{cfg.state_file.name}.part")
    tmp.write_text(json.dumps(st, indent=1, default=str))
    os.replace(tmp, cfg.state_file)


def note_step(cfg: Cfg, st: dict, step: str, status: str, **kw) -> None:
    st.setdefault("steps", {})[step] = {"status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "threads": cfg.threads, **kw}
    save_state(cfg, st)


# ------------------------------------------------------------------------------------------------ disk state
def b15_file(cfg: Cfg, p) -> Path:
    return cfg.b15 / f"{pname(p)}.npz"


def b15_ok(f: Path) -> bool:
    """A complete pair file (same test as dr5_measure_xi_unordered.pair_file_ok)."""
    if not f.is_file():
        return False
    try:
        import numpy as np
        with np.load(f) as d:
            return all(np.asarray(d[k]).size > 0 for k in ("xip", "xim", "npairs", "weight", "r_nom"))
    except Exception:
        return False


def chunk_dir(cfg: Cfg, p) -> Path:
    return cfg.chunks / "chunks" / pname(p)


def chunk_total(p, nchunks: int) -> int:
    return nchunks * (nchunks + 1) // 2 if p[0] == p[1] else nchunks * nchunks


def chunks_done(cfg: Cfg, p) -> tuple[int, float]:
    """(number of finished chunk-pairs, fraction of the pair's work they represent)."""
    d = chunk_dir(cfg, p)
    if not d.is_dir():
        return 0, 0.0
    table = _W_AUTO if p[0] == p[1] else _W_CROSS
    tot = sum(table.values())
    n, w = 0, 0.0
    for f in d.glob("c*.npz"):
        n += 1
        try:
            a, b = (int(x) for x in f.stem[1:].split("_"))
        except ValueError:
            continue
        w += table.get((a, b), 0.0)
    total = chunk_total(p, cfg.nchunks)
    if n >= total:
        return n, 1.0
    return n, min(w / tot, 0.999) if cfg.nchunks == NCHUNKS else n / total


def bins_present(work: Path) -> list[int]:
    return [b for b in range(1, NBIN + 1) if (work / f"bin{b - 1}.npz").is_file()]


def free_gb(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return None


def fits_header(path: Path) -> dict | None:
    """NAXIS1 / NAXIS2 of the first table extension (header only; fitsio, else astropy)."""
    try:
        import fitsio
        h = fitsio.read_header(str(path), ext=1)
        return {"NAXIS1": int(h["NAXIS1"]), "NAXIS2": int(h["NAXIS2"])}
    except ImportError:
        pass
    except Exception:
        return None
    try:
        from astropy.io import fits
        h = fits.getheader(str(path), 1)
        return {"NAXIS1": int(h["NAXIS1"]), "NAXIS2": int(h["NAXIS2"])}
    except Exception:
        return None


def verify_catalogue(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    h = fits_header(path)
    if h is None:
        ok = size >= CATALOGUE_ROWS * CATALOGUE_ROW_BYTES
        return ok, f"{size / 1e9:.2f} GB (no fitsio/astropy to read the header; size {'consistent' if ok else 'TOO SMALL'})"
    if h["NAXIS2"] != CATALOGUE_ROWS:
        return False, f"NAXIS2 = {h['NAXIS2']:,}, expected {CATALOGUE_ROWS:,}"
    if size < h["NAXIS1"] * h["NAXIS2"]:
        return False, f"truncated: {size:,} bytes, table needs {h['NAXIS1'] * h['NAXIS2']:,}"
    return True, f"{h['NAXIS2']:,} rows x {h['NAXIS1']} B, {size / 1e9:.2f} GB"


# ------------------------------------------------------------------------------------------------ estimates
def machine_factor(st: dict, which: str) -> float:
    """Measured-over-reference time ratio at equal threads (median over runs recorded in the chain state)."""
    vals = [r["factor"] for r in st.get("records", {}).get(which, {}).values() if isinstance(r, dict) and r.get("factor")]
    if not vals:
        return 1.0
    vals = sorted(vals)
    return min(max(vals[len(vals) // 2], 0.2), 10.0)       # clamp: a fluke must not make the budget logic absurd


def est_b15_s(cfg: Cfg, st: dict, p) -> float:
    return REF_B15_S[p] * REF_THREADS / cfg.threads * machine_factor(st, "b15")


def est_exact_s(cfg: Cfg, st: dict, p, remaining_only: bool = True) -> float:
    full = REF_EXACT_H[p] * 3600 * REF_THREADS / cfg.threads * machine_factor(st, "exact")
    if not remaining_only:
        return full
    return full * (1.0 - chunks_done(cfg, p)[1])


# ------------------------------------------------------------------------------------------------ running commands
_CHILD: subprocess.Popen | None = None


def _terminate(proc: subprocess.Popen, grace: float = 60.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _on_sigterm(signum, frame):                          # Colab "interrupt" / runtime shutdown
    raise KeyboardInterrupt


def run_cmd(cfg: Cfg, argv: list[str], label: str, *, timeout: float | None = None, env: dict | None = None,
            est_s: float = 0.0) -> tuple[int, float, bool]:
    """Run one subprocess in its own process group. Returns (returncode, wall seconds, timed_out). On timeout the process group
    gets SIGTERM, then SIGKILL after 60 s."""
    global _CHILD
    shown = " ".join(str(x) for x in argv)
    if cfg.dry:
        extra = f"   # est {hms(est_s)}" if est_s else ""
        print(f"[dry-run] {label}: {shown}{extra}", flush=True)
        cfg.virtual += min(est_s, timeout) if timeout is not None else est_s
        return 0, est_s, False
    print(f"\n=== {label}  [{time.strftime('%H:%M:%S')}]\n$ {shown}", flush=True)
    t = time.time()
    proc = subprocess.Popen(argv, cwd=str(ROOT), env=env or cfg.env(), start_new_session=True)
    _CHILD = proc
    timed_out = False
    try:
        while True:
            try:
                rc = proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if timeout is not None and time.time() - t > timeout:
                    print(f"=== {label}: time budget reached, stopping the process (SIGTERM)", flush=True)
                    _terminate(proc)
                    rc, timed_out = proc.returncode, True
                    break
    except KeyboardInterrupt:
        print(f"=== {label}: interrupted, stopping the process", flush=True)
        _terminate(proc)
        raise
    finally:
        _CHILD = None
    return rc, time.time() - t, timed_out


def py(script: str, *args) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *[str(x) for x in args]]


def seed_inputs(cfg: Cfg) -> None:
    """Copy small shipped inputs that cannot be regenerated (earlier-pipeline results) into <runs> when it is not the shipped runs/."""
    if cfg.runs.resolve() == cfg.shipped.resolve() or getattr(cfg, "_seeded", False):
        return
    cfg._seeded = True
    for rel in SEED_FILES:
        dst, src = cfg.runs / rel, cfg.shipped / rel
        if dst.exists() or not src.exists():
            continue
        if cfg.dry:
            print(f"[dry-run] copy shipped input {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied shipped input {rel} -> {dst}")


def sync_work_cache(cfg: Cfg) -> None:
    """Copy the per-bin npz (and extract_info.json) from --work-dir to the local --work-cache when missing or of different size."""
    if cfg.work_cache is None:
        return
    names = [f"bin{b}.npz" for b in range(NBIN)] + ["extract_info.json"]
    for n in names:
        src, dst = cfg.work / n, cfg.work_cache / n
        if not src.is_file():
            continue
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        if cfg.dry:
            print(f"[dry-run] cache {src} -> {dst}")
            continue
        cfg.work_cache.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.part")
        t = time.time()
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        print(f"cached {n} ({dst.stat().st_size / 1e6:.0f} MB, {time.time() - t:.0f} s)", flush=True)


def ram_warning(kind_: str) -> None:
    avail = mem_available_gb()
    if avail is not None and avail < RAM_NEED_GB[kind_]:
        print(f"WARNING: {avail:.1f} GB RAM available, a {kind_} pair peaked at about {RAM_NEED_GB[kind_] - 0.5:.1f} GB on the development "
              f"machine: the process may be OOM-killed (close other work or use a high-RAM runtime).", flush=True)


# ------------------------------------------------------------------------------------------------ status
def cmd_status(cfg: Cfg, st: dict) -> int:
    a = cfg.a
    th = cfg.threads
    print(f"DR5 chain status  [{time.strftime('%Y-%m-%d %H:%M:%S')}]   threads={th} (cpu_count={os.cpu_count()})")
    print(f"  runs        {cfg.runs}   (outputs in {cfg.meas})")
    print(f"  work (bins) {cfg.work}" + (f"   cache: {cfg.work_cache}" if cfg.work_cache else ""))
    print(f"  catalogue   {cfg.catalogue}")
    avail = mem_available_gb()
    print(f"  free disk   runs {free_gb(cfg.runs):.0f} GB, work {free_gb(cfg.work):.0f} GB, catalogue {free_gb(cfg.catalogue.parent):.0f} GB"
          + (f";  RAM available {avail:.1f} GB" if avail is not None else ""))

    gz = cfg.catalogue.with_name(cfg.catalogue.name + ".gz")
    cat_ok, cat_msg = verify_catalogue(cfg.catalogue)
    gz_msg = f"{gz.stat().st_size / 1e9:.2f} GB" if gz.is_file() else "absent"
    print(f"\n[download] fits: {'OK' if cat_ok else 'missing/incomplete'} ({cat_msg}); gz: {gz_msg}")
    present = bins_present(cfg.work)
    info_ok = (cfg.work / "extract_info.json").is_file()
    extract_done = len(present) == NBIN and info_ok
    print(f"[extract]  per-bin npz: {len(present)}/{NBIN}" + ("  extract_info.json present" if info_ok else "  extract_info.json MISSING"))

    f15, fex = machine_factor(st, "b15"), machine_factor(st, "exact")
    print(f"\nestimate basis: recorded 20-thread timings x 20/{th}; machine factor b1.5 {f15:.2f}, exact {fex:.2f} "
          f"(1.00 = no measurement of this machine yet)\n")
    print(f"{'pair':<11}{'b1.5':>6}{'est':>9}   {'exact chunks':>14}{'work':>7}{'est left':>10}   {'assembled':>9}")
    t15 = tex = 0.0
    n15 = nex = nas = 0
    for p in PAIRS:
        d15 = b15_ok(b15_file(cfg, p))
        e15 = 0.0 if d15 else est_b15_s(cfg, st, p)
        nd, fr = chunks_done(cfg, p)
        tot = chunk_total(p, cfg.nchunks)
        eex = 0.0 if nd >= tot else est_exact_s(cfg, st, p)
        asm = (cfg.chunks / "assembled" / "ns" / f"{pname(p)}.npz").is_file()
        t15 += e15
        tex += eex
        n15 += d15
        nex += nd >= tot
        nas += asm
        print(f"{pname(p):<11}{'done' if d15 else '-':>6}{('' if d15 else hms(e15)):>9}   {f'{nd}/{tot}':>14}{fr * 100:6.0f}%"
              f"{('' if nd >= tot else hms(eex)):>10}   {'yes' if asm else '-':>9}")
    print(f"\nb1.5 pairs   {n15}/21 done, estimated {t15 / 3600:.1f} h left -> {math.ceil(t15 / 3600 / a.session_hours) if t15 else 0} session(s) of {a.session_hours:g} h")
    print(f"exact pairs  {nex}/21 complete, estimated {tex / 3600:.1f} h left -> {math.ceil(tex / 3600 / a.session_hours) if tex else 0} session(s) of {a.session_hours:g} h; assembled {nas}/21")
    print("\nproducts")
    prods = [("b1.5 eval", cfg.b15 / "eval.json"), ("b1 integration", cfg.meas / "b1_integration.json"),
             ("b1.5 product", cfg.meas / "product" / "dr5_xipm_cosebis_binslop1p5.npz"),
             ("a4 re-check", cfg.meas / "product" / "a4_recheck.json"),
             ("exact product", cfg.meas / "product_exact" / "dr5_xipm_cosebis_exact.npz")]
    for name, f in prods:
        print(f"  {name:<15}{'yes' if f.is_file() else '-':<5}{f}")
    if st.get("updated"):
        print(f"\nchain_state.json last updated {st['updated']}; {len(st.get('sessions', []))} session(s) recorded")
    if a.json:
        print(json.dumps({"b15_done": n15, "exact_done": nex, "assembled": nas, "b15_est_h": t15 / 3600, "exact_est_h": tex / 3600,
                          "catalogue_ok": cat_ok, "extract_done": extract_done}))
    return 0


# ------------------------------------------------------------------------------------------------ download / extract
def remote_size(url: str) -> int | None:
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "kidspipe-dr5-chain"})
        with urllib.request.urlopen(req, timeout=30) as r:
            v = r.headers.get("Content-Length")
            return int(v) if v else None
    except Exception:
        return None


def cmd_download(cfg: Cfg, st: dict) -> int:
    a = cfg.a
    if a.dir:
        fits = _abs(a.dir) / paths.CATALOGUE_NAME
        if fits.resolve() != cfg.catalogue.resolve():
            print(f"note: downloading to {fits}; set KIDSPIPE_DR5_CATALOGUE={fits} (or pass --catalogue) for the later steps.")
    else:
        fits = cfg.catalogue
    gz = fits.with_name(fits.name + ".gz")
    part = fits.with_name(fits.name + ".part")
    print(ACK)
    print(f"source {a.url}\ntarget {fits}  (gz {gz})")
    if "/drive/" in str(fits).lower() or "mydrive" in str(fits).lower():
        print("WARNING: the catalogue is on a Drive mount. Keep the 16 GB fits on LOCAL disk (e.g. /content/dr5): Drive FUSE is slow and "
              "TreeCorr/fitsio read it many times.")
    ok, msg = verify_catalogue(fits)
    if ok:
        print(f"catalogue already complete: {msg}")
        if a.delete_gz and gz.is_file():
            print(f"removing {gz}")
            if not cfg.dry:
                gz.unlink()
        return 0
    if fits.exists():
        print(f"existing {fits} is not valid ({msg}); it will be replaced by a fresh decompression")

    if not cfg.dry:
        fits.parent.mkdir(parents=True, exist_ok=True)
    need = (0 if gz.is_file() else GZ_GB) + FITS_GB + 1.0
    have = free_gb(fits.parent)
    print(f"disk: {have:.1f} GB free, about {need:.1f} GB needed ({'gz present' if gz.is_file() else 'gz 7.2 GB'} + fits 16.4 GB + margin)"
          + ("  (the gz can be deleted afterwards with --delete-gz)" if not a.delete_gz else ""))
    if have < need and not cfg.dry:
        print("ERROR: not enough free disk space", file=sys.stderr)
        return 1

    # 1. download, resumable
    want = remote_size(a.url)
    have_gz = gz.stat().st_size if gz.is_file() else 0
    if want is not None and have_gz == want:
        print(f"gz already complete ({have_gz / 1e9:.2f} GB)")
    else:
        if want is not None:
            print(f"remote size {want / 1e9:.2f} GB, local {have_gz / 1e9:.2f} GB: resuming")
        curl, wget = shutil.which("curl"), shutil.which("wget")
        if curl:
            cmd = [curl, "-L", "--fail", "-C", "-", "-sS", "-o", str(gz), a.url]
        elif wget:
            cmd = [wget, "-c", "-nv", "-O", str(gz), a.url]
        else:
            print("ERROR: neither curl nor wget found", file=sys.stderr)
            return 1
        if cfg.dry:
            run_cmd(cfg, cmd, "download gz (resumable)")
        else:
            for attempt in range(1, a.retries + 1):
                print(f"\n=== download gz, attempt {attempt}/{a.retries} (resumes from {gz.stat().st_size / 1e9 if gz.is_file() else 0:.2f} GB)\n$ {' '.join(cmd)}", flush=True)
                proc = subprocess.Popen(cmd)
                try:
                    while proc.poll() is None:
                        try:
                            proc.wait(timeout=60)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                        print(f"  gz {gz.stat().st_size / 1e9 if gz.is_file() else 0:6.2f} GB" + (f" of {want / 1e9:.2f}" if want else "")
                              + f"  [{time.strftime('%H:%M:%S')}]", flush=True)
                except KeyboardInterrupt:
                    proc.terminate()
                    raise
                size = gz.stat().st_size if gz.is_file() else 0
                if proc.returncode == 0 and (want is None or size == want):
                    break
                if want is None and size > 7.0e9:        # cannot tell the full size; the decompression below detects a truncated gz
                    break
                print(f"download not finished (rc={proc.returncode}, {size / 1e9:.2f} GB): retrying in 20 s", flush=True)
                time.sleep(20)
            else:
                print("ERROR: download did not complete; run the command again to resume", file=sys.stderr)
                return 1

    # 2. stream-decompress into .part, rename only when the decompressor exited 0 (a truncated gz makes it fail)
    dec = shutil.which("pigz")
    dcmd = [dec, "-dc", str(gz)] if dec else [shutil.which("gzip") or "gzip", "-dc", str(gz)]
    if cfg.dry:
        print(f"[dry-run] decompress: {' '.join(dcmd)} > {part}; verify NAXIS2 = {CATALOGUE_ROWS:,}; rename to {fits}")
        return 0
    print(f"\n=== decompress ({dcmd[0]}) -> {part}", flush=True)
    t = time.time()
    with open(part, "wb") as out:
        proc = subprocess.Popen(dcmd, stdout=out)
        try:
            while proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    pass
                print(f"  {part.stat().st_size / 1e9:6.2f} / {FITS_GB:.1f} GB  [{time.time() - t:.0f} s]", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            raise
    if proc.returncode != 0:
        print(f"ERROR: decompression failed (rc={proc.returncode}); the gz is probably truncated: delete {gz} and run again", file=sys.stderr)
        part.unlink(missing_ok=True)
        return 1
    ok, msg = verify_catalogue(part)
    if not ok:
        print(f"ERROR: decompressed file failed verification: {msg}", file=sys.stderr)
        return 1
    os.replace(part, fits)
    print(f"catalogue ready: {fits}  ({msg}; {time.time() - t:.0f} s)")
    if a.delete_gz:
        gz.unlink()
        print(f"removed {gz}")
    note_step(cfg, st, "download", "done", file=str(fits), rows=CATALOGUE_ROWS)
    return 0


def cmd_extract(cfg: Cfg, st: dict) -> int:
    present = bins_present(cfg.work)
    if len(present) == NBIN and (cfg.work / "extract_info.json").is_file() and not cfg.a.force:
        print(f"extract: all {NBIN} per-bin files already in {cfg.work} (use --force to redo)")
        return 0
    if not cfg.dry:
        ok, msg = verify_catalogue(cfg.catalogue)
        if not ok:
            print(f"ERROR: catalogue not usable ({msg}). Run `dr5_chain.py download` first (or set --catalogue).", file=sys.stderr)
            return 1
        cfg.work.mkdir(parents=True, exist_ok=True)
    argv = py("dr5_extract.py", "--catalogue", cfg.catalogue, "--out", cfg.work)
    if cfg.a.force:
        argv.append("--force")
    rc, wall, _ = run_cmd(cfg, argv, "extract per-bin npz (one pass over 16.4 GB; single thread, minutes)", est_s=600)
    if rc != 0:
        print("ERROR: extraction failed", file=sys.stderr)
        return 1
    note_step(cfg, st, "extract", "done", wall_s=wall)
    return 0


# ------------------------------------------------------------------------------------------------ measure (bin_slop 1.5)
def _order(pairs, key, how):
    if how == "index":
        return list(pairs)
    return sorted(pairs, key=key, reverse=(how == "heavy"))


def cmd_measure(cfg: Cfg, st: dict) -> int:
    a = cfg.a
    want = parse_pairs(a.pairs)
    pending = [p for p in want if not b15_ok(b15_file(cfg, p)) or a.force]
    print(f"measure: {len(want) - len(pending)}/{len(want)} pairs already done, {len(pending)} to do, threads={cfg.threads}"
          + ("" if cfg.deadline is None else f", budget {hms(cfg.remaining())}"))
    if not pending:
        return 0
    needed = sorted({b for p in pending for b in p})
    if not cfg.dry:
        miss = [b for b in needed if b not in bins_present(cfg.work)]
        if miss:
            print(f"ERROR: per-bin file(s) for bin(s) {miss} missing in {cfg.work}. Run `dr5_chain.py extract` first.", file=sys.stderr)
            return 1
    sync_work_cache(cfg)
    seed_inputs(cfg)
    env = cfg.env(cfg.measure_work)
    order = _order(pending, lambda p: REF_B15_S[p], a.order)
    failed, consecutive, ran = [], 0, 0
    tried: set = set()
    while True:
        if cfg.stop_file.exists():
            print(f"stop file {cfg.stop_file} present: stopping")
            break
        # next pair in the chosen order whose estimate (re-calibrated after every finished pair) fits the remaining budget
        p = next((q for q in order if q not in tried and est_b15_s(cfg, st, q) <= cfg.remaining()), None)
        if p is None:
            break
        tried.add(p)
        est = est_b15_s(cfg, st, p)
        ram_warning(kind(p))
        argv = py("dr5_measure_xi_unordered.py", "run", "--pairs", f"{p[0]}-{p[1]}", "--bin-slop", "1.5", "--threads", cfg.threads)
        if a.force:
            argv.append("--force")
        timeout = cfg.remaining() if a.hard_deadline else None
        rc, wall, timed_out = run_cmd(cfg, argv, f"b1.5 {pname(p)} (est {hms(est)})", timeout=timeout, env=env, est_s=est)
        if cfg.dry:
            ran += 1
            continue
        ok = rc == 0 and b15_ok(b15_file(cfg, p))
        rec = {"wall_s": wall, "threads": cfg.threads, "rc": rc, "complete": ok, "timed_out": timed_out, "end": time.strftime("%Y-%m-%d %H:%M:%S")}
        if ok:
            rec["factor"] = wall * cfg.threads / (REF_B15_S[p] * REF_THREADS)
            consecutive = 0
            ran += 1
        elif timed_out:                                   # --hard-deadline cut the pair: a clean stop (exit 75), not a failure
            print(f"{pname(p)}: stopped at the deadline (the pair is lost and will be redone in the next session)")
        else:
            failed.append(p)
            consecutive += 1
            hint = " (killed: probably out of memory)" if rc in (-9, 137) else ""
            print(f"FAILED {pname(p)}: rc={rc}{hint}", file=sys.stderr)
        st.setdefault("records", {}).setdefault("b15", {})[pname(p)] = rec
        save_state(cfg, st)
        if timed_out:
            break
        if consecutive >= a.max_failures:
            print(f"{consecutive} failures in a row: stopping", file=sys.stderr)
            break
    skipped_budget = [q for q in order if q not in tried]
    for q in (skipped_budget if cfg.deadline is not None and not failed else []):
        print(f"not started {pname(q)}: estimated {hms(est_b15_s(cfg, st, q))} > remaining budget {hms(cfg.remaining())}")
    if cfg.dry:
        print(f"[dry-run] measure: {ran} pair(s) fit the budget, {len(skipped_budget)} do not")
        return EXIT_BUDGET if skipped_budget else 0
    left = [p for p in want if not b15_ok(b15_file(cfg, p))]
    done_all = not left
    note_step(cfg, st, "measure", "done" if done_all else "partial", pairs_done=len(want) - len(left))
    if failed:
        return 1
    if left:
        print(f"\nmeasure: {len(left)} pair(s) left ({', '.join(pname(p) for p in left)}). Run the same command again in a new session.")
        return EXIT_BUDGET
    print("\nmeasure: all requested pairs done")
    return 0


def all_b15_done(cfg: Cfg) -> bool:
    return all(b15_ok(b15_file(cfg, p)) for p in PAIRS)


def cmd_eval(cfg: Cfg, st: dict) -> int:
    if not cfg.dry and not all_b15_done(cfg):
        miss = [pname(p) for p in PAIRS if not b15_ok(b15_file(cfg, p))]
        print(f"ERROR: eval needs all 21 b1.5 pairs; missing {len(miss)}: {', '.join(miss)}. Run `measure` first.", file=sys.stderr)
        return 1
    seed_inputs(cfg)
    rc, wall, _ = run_cmd(cfg, py("dr5_measure_xi_unordered.py", "eval", "--pairs", "all", "--bin-slop", "1.5", "--threads", cfg.threads),
                          "evaluate b1.5 pairs against the official release", est_s=300)
    if rc == 0:
        note_step(cfg, st, "eval", "done", wall_s=wall)
    return 0 if rc == 0 else 1


def cmd_product(cfg: Cfg, st: dict) -> int:
    if not cfg.dry and not all_b15_done(cfg):
        print("ERROR: the product needs all 21 b1.5 pairs. Run `measure` first.", file=sys.stderr)
        return 1
    seed_inputs(cfg)
    steps = [("b1 integration check (release only)", py("dr5_b1_integration.py")),
             ("make the b1.5 product + full-vector checks", py("dr5_make_product.py", "make")),
             ("step D (model chi2)", py("dr5_make_product.py", "stepd"))]
    for label, argv in steps:
        rc, _, _ = run_cmd(cfg, argv, label, est_s=60)
        if rc != 0:
            print(f"ERROR: step failed: {label}", file=sys.stderr)
            return 1
    note_step(cfg, st, "product", "done")
    return 0


# ------------------------------------------------------------------------------------------------ exact run
def exact_args(cfg: Cfg, *extra) -> list[str]:
    return ["--nchunks", cfg.nchunks, "--outroot", cfg.chunks, *extra]


def clean_part_files(cfg: Cfg, p) -> None:
    d = chunk_dir(cfg, p)
    if d.is_dir():
        for f in d.glob(".*.part"):
            f.unlink(missing_ok=True)


def cmd_exact(cfg: Cfg, st: dict) -> int:
    a = cfg.a
    want = parse_pairs(a.pairs)
    pending = [p for p in want if chunks_done(cfg, p)[0] < chunk_total(p, cfg.nchunks) or a.force]
    tot_est = sum(est_exact_s(cfg, st, p) for p in pending)
    v0 = cfg.virtual
    print(f"exact: {len(want) - len(pending)}/{len(want)} pairs complete, {len(pending)} to do, threads={cfg.threads}, "
          f"estimated {tot_est / 3600:.1f} h left" + ("" if cfg.deadline is None else f", budget {hms(cfg.remaining())}"))
    if not pending:
        return 0
    if not cfg.dry:
        miss = [b for b in sorted({b for p in pending for b in p}) if b not in bins_present(cfg.work)]
        if miss:
            print(f"ERROR: per-bin file(s) for bin(s) {miss} missing in {cfg.work}. Run `dr5_chain.py extract` first.", file=sys.stderr)
            return 1
    sync_work_cache(cfg)
    env = cfg.env(cfg.measure_work)
    order = _order(pending, lambda p: REF_EXACT_H[p], a.order)
    failed, consecutive, stopped = [], 0, False
    for p in order:
        if cfg.stop_file.exists():
            print(f"stop file {cfg.stop_file} present: stopping before {pname(p)}")
            stopped = True
            break
        est = est_exact_s(cfg, st, p)
        rem = cfg.remaining()
        if a.no_kill:                                          # finish whole pairs only
            if est > rem:
                print(f"skip {pname(p)}: estimated {hms(est)} > remaining budget {hms(rem)} (--no-kill)")
                continue
            timeout = None
        else:                                                  # chunk-level resume: work until the deadline, lose at most one chunk-pair
            if rem < a.min_start_min * 60:
                print(f"remaining budget {hms(rem)} < {a.min_start_min:g} min: not starting {pname(p)}")
                stopped = True
                break
            timeout = rem
        ram_warning(kind(p))
        nd0, fr0 = chunks_done(cfg, p)
        argv = py("dr5_measure_xi_chunked.py", "--all", "--pairs", f"{p[0]}-{p[1]}", *exact_args(cfg), "--threads", cfg.threads,
                  "--bin-slop", 0, "--angle-slop", 0)
        if a.force:
            argv.append("--force")
        rc, wall, timed_out = run_cmd(cfg, argv, f"exact {pname(p)}  ({nd0}/{chunk_total(p, cfg.nchunks)} chunk-pairs present, est {hms(est)} left)",
                                      timeout=timeout, env=env, est_s=est)
        if cfg.dry:
            continue
        clean_part_files(cfg, p)
        nd1, fr1 = chunks_done(cfg, p)
        complete = nd1 >= chunk_total(p, cfg.nchunks)
        rec = {"wall_s": wall, "threads": cfg.threads, "rc": rc, "timed_out": timed_out, "complete": complete,
               "chunks_before": nd0, "chunks_after": nd1, "end": time.strftime("%Y-%m-%d %H:%M:%S")}
        if (timed_out or rc == 0) and fr1 - fr0 > 0.05:
            rec["factor"] = wall * cfg.threads / (REF_EXACT_H[p] * 3600 * REF_THREADS * (fr1 - fr0))
        st.setdefault("records", {}).setdefault("exact", {})[pname(p)] = rec
        save_state(cfg, st)
        if timed_out:
            print(f"{pname(p)}: stopped at the deadline with {nd1}/{chunk_total(p, cfg.nchunks)} chunk-pairs (resumes from there)")
            stopped = True
            break
        if rc != 0 or not complete:
            failed.append(p)
            consecutive += 1
            hint = " (killed: probably out of memory)" if rc in (-9, 137) else ""
            print(f"FAILED {pname(p)}: rc={rc}{hint}", file=sys.stderr)
            if consecutive >= a.max_failures:
                print(f"{consecutive} failures in a row: stopping", file=sys.stderr)
                break
        else:
            consecutive = 0
    if cfg.dry:
        left_est = sum(est_exact_s(cfg, st, p) for p in order)
        print(f"[dry-run] exact: this budget covers {hms(cfg.virtual - v0)} of {hms(left_est)} estimated work")
        return EXIT_BUDGET if left_est > cfg.virtual - v0 + 1 else 0
    left = [p for p in want if chunks_done(cfg, p)[0] < chunk_total(p, cfg.nchunks)]
    note_step(cfg, st, "exact", "done" if not left else "partial", pairs_complete=len(want) - len(left))
    if failed:
        return 1
    if left:
        print(f"\nexact: {len(left)} pair(s) incomplete. Run the same command again in a new session.")
        return EXIT_BUDGET
    print("\nexact: all requested pairs complete")
    return 0


def cmd_assemble(cfg: Cfg, st: dict) -> int:
    if not cfg.dry:
        miss = [pname(p) for p in PAIRS if chunks_done(cfg, p)[0] < chunk_total(p, cfg.nchunks)]
        if miss:
            print(f"ERROR: {len(miss)} pair(s) still incomplete ({', '.join(miss)}); run `exact` first.", file=sys.stderr)
            return 1
    rc, wall, _ = run_cmd(cfg, py("dr5_measure_xi_chunked.py", "--assemble", *exact_args(cfg)), "assemble chunk-pairs", est_s=60)
    if rc == 0:
        note_step(cfg, st, "assemble", "done", wall_s=wall)
    return 0 if rc == 0 else 1


def cmd_exact_product(cfg: Cfg, st: dict) -> int:
    if not cfg.dry:
        miss = [pname(p) for p in PAIRS if not (cfg.chunks / "assembled" / "ns" / f"{pname(p)}.npz").is_file()]
        if miss:
            print(f"ERROR: assembled pairs missing ({len(miss)}); run `assemble` first.", file=sys.stderr)
            return 1
        if not (cfg.meas / "product" / "dr5_xipm_cosebis_binslop1p5.npz").is_file():
            print("ERROR: the exact product compares with the b1.5 product; run `product` first.", file=sys.stderr)
            return 1
    rc, wall, _ = run_cmd(cfg, py("dr5_make_exact_product.py"), "make the exact product", est_s=120)
    if rc == 0:
        note_step(cfg, st, "exact-product", "done", wall_s=wall)
    return 0 if rc == 0 else 1


def cmd_plots_exact(cfg: Cfg, st: dict) -> int:
    return cmd_plots(cfg, st, exact_only=True)


def cmd_plots(cfg: Cfg, st: dict, exact_only: bool = False) -> int:
    seed_inputs(cfg)
    prod = cfg.meas / "product" / "dr5_xipm_cosebis_binslop1p5.npz"
    steps = []
    if exact_only:
        pass
    elif prod.is_file() or cfg.dry:
        if cfg.a.skip_a4:
            pass
        else:
            sync_work_cache(cfg)
            steps.append(("a4: estimator re-check on bins 1 and 6 (about 5 min at 16 threads)", py("dr5_make_plots.py", "a4"), cfg.env(cfg.measure_work)))
        steps.append(("figures dr5_01 .. dr5_06", py("dr5_make_plots.py", "plots"), cfg.env()))
    else:
        print("plots: no b1.5 product yet (run `product`): skipping the b1.5 figures")
    if (cfg.meas / "product_exact" / "dr5_xipm_cosebis_exact.npz").is_file() or cfg.dry:
        steps.append(("exact vs b1.5 figure", py("dr5_plot_exact_vs_b1p5.py"), cfg.env()))
        steps.append(("exact values figures", py("dr5_plot_exact_values.py"), cfg.env()))
    for label, argv, env in steps:
        rc, _, _ = run_cmd(cfg, argv, label, env=env, est_s=120)
        if rc != 0:
            print(f"ERROR: {label} failed", file=sys.stderr)
            return 1
    if steps:
        note_step(cfg, st, "plots", "done")
    return 0


# ------------------------------------------------------------------------------------------------ compare with the shipped reference
#: provenance / metadata arrays of the pair and product files (wall time, thread count, object counts, settings): they differ on every
#: machine and say nothing about the physics, so they are not compared
_PROVENANCE = {"seconds", "threads", "nobj1", "nobj2", "source_sizes", "bin_slop", "angle_slop_effective"}


def _compare_npz(mine: Path, ref: Path) -> str:
    import numpy as np
    with np.load(mine) as a, np.load(ref) as b:
        worst, worst_key, identical, n = 0.0, "", True, 0
        for k in b.files:
            if k in _PROVENANCE:
                continue
            if k not in a.files:
                return f"key {k} missing"
            x, y = a[k], b[k]
            if x.shape != y.shape or x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
                continue
            n += 1
            if not np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else not np.array_equal(x, y):
                identical = False
                scale = float(np.nanmax(np.abs(y))) or 1.0
                r = float(np.nanmax(np.abs(np.asarray(x, float) - np.asarray(y, float)))) / scale
                if r > worst:
                    worst, worst_key = r, k
    return f"{n} arrays identical" if identical else f"max |diff|/max|ref| = {worst:.2e} ({worst_key}) over {n} arrays"


def cmd_compare(cfg: Cfg, st: dict) -> int:
    if cfg.runs.resolve() == cfg.shipped.resolve():
        print("runs dir is the shipped runs/: nothing to compare (set KIDSPIPE_RUNS to compare a new run).")
        return 0
    sets = [("b1.5 pairs", "dr5_measure/xi_unordered/b1.5", "Bin*.npz"), ("b1.5 product", "dr5_measure/product", "dr5_xipm_cosebis_binslop1p5.npz"),
            ("exact assembled", "dr5_measure/xi_chunked/assembled/ns", "Bin*.npz"), ("exact product", "dr5_measure/product_exact", "dr5_xipm_cosebis_exact.npz")]
    print("Compare your results (left) with the reference results shipped in the repository (right). Independent TreeCorr/numpy/BLAS versions\n"
          "give tiny differences; the recorded agreement tolerances are in docs/DR5_MEASUREMENT_REPORT.md and are not re-defined here.\n")
    for title, rel, pat in sets:
        mine_dir, ref_dir = cfg.runs / rel, cfg.shipped / rel
        files = sorted(ref_dir.glob(pat)) if ref_dir.is_dir() else []
        if not files:
            print(f"{title}: no shipped reference")
            continue
        print(f"{title}:")
        for f in files:
            m = mine_dir / f.name
            try:
                msg = _compare_npz(m, f) if m.is_file() else "not produced yet"
            except Exception as e:                  # unreadable (truncated) file
                msg = f"cannot compare ({type(e).__name__}: {str(e)[:60]})"
            print(f"  {f.name:<38}{msg}")
    return 0


# ------------------------------------------------------------------------------------------------ all
def cmd_all(cfg: Cfg, st: dict) -> int:
    a = cfg.a
    steps = [("download", cmd_download), ("extract", cmd_extract), ("measure", cmd_measure), ("eval", cmd_eval), ("product", cmd_product),
             ("plots", cmd_plots)]
    if not a.no_exact:
        steps += [("exact", cmd_exact), ("assemble", cmd_assemble), ("exact-product", cmd_exact_product), ("plots (exact)", cmd_plots_exact)]
    for name, fn in steps:
        print(f"\n######## {name}", flush=True)
        if name == "download" and extract_ready(cfg) and not a.force:
            print("download: not needed (all per-bin files are present)")
            continue
        rc = fn(cfg, st)
        if rc != 0:
            print(f"\nall: stopped at '{name}' (exit {rc})" + (": time budget used up, run the same command in a new session" if rc == EXIT_BUDGET else ""))
            return rc
        if cfg.deadline is not None and cfg.remaining() <= 0:
            print("\nall: time budget used up, run the same command in a new session")
            return EXIT_BUDGET
    print("\n[dry-run] end of plan" if cfg.dry else "\nall: complete")
    return 0


def extract_ready(cfg: Cfg) -> bool:
    return len(bins_present(cfg.work)) == NBIN and (cfg.work / "extract_info.json").is_file()


COMMANDS = {"status": cmd_status, "download": cmd_download, "extract": cmd_extract, "measure": cmd_measure, "eval": cmd_eval,
            "product": cmd_product, "exact": cmd_exact, "assemble": cmd_assemble, "exact-product": cmd_exact_product, "plots": cmd_plots,
            "compare": cmd_compare, "all": cmd_all}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring (python -c 'import ast,sys;print(ast.get_docstring(ast.parse(open(\"scripts/dr5_chain.py\").read())))')"
                                        " and docs/COLAB.md.")
    ap.add_argument("cmd", choices=list(COMMANDS))
    g = ap.add_argument_group("locations")
    g.add_argument("--runs-dir", default=None, help="outputs (default $KIDSPIPE_RUNS or <repo>/runs)")
    g.add_argument("--work-dir", default=None, help="per-bin npz (default $KIDSPIPE_DR5_WORK or ~/kids_dr5_work)")
    g.add_argument("--catalogue", default=None, help="the 16.4 GB fits on LOCAL disk (default $KIDSPIPE_DR5_CATALOGUE or <work>/KiDS_Legacy_NS_unblind_final.fits)")
    g.add_argument("--work-cache", default=None, help="local directory: the per-bin npz are copied there before measuring (use when --work-dir is on Drive)")
    g.add_argument("--state", default=None, help="chain state file (default <runs>/dr5_measure/chain_state.json)")
    g = ap.add_argument_group("running")
    g.add_argument("--threads", type=int, default=None, help="threads per process (default os.cpu_count())")
    g.add_argument("--max-hours", type=float, default=None, help="time budget of this invocation; no pair is started whose estimate exceeds what is left")
    g.add_argument("--margin-min", type=float, default=5.0, help="minutes kept free at the end of the budget (default 5)")
    g.add_argument("--pairs", default="all", help="'all', 'autos', 'cross' or a list like 1-1,2-3")
    g.add_argument("--order", choices=["heavy", "light", "index"], default="heavy", help="pair order (default heaviest first)")
    g.add_argument("--force", action="store_true", help="redo finished work")
    g.add_argument("--dry-run", action="store_true", help="print the commands and the planned order, run nothing")
    g.add_argument("--max-failures", type=int, default=2, help="stop after this many failed pairs in a row")
    g = ap.add_argument_group("measure / exact")
    g.add_argument("--hard-deadline", action="store_true", help="measure: also kill a b1.5 pair at the deadline (loses that pair; default: only the estimate gates starts)")
    g.add_argument("--no-kill", action="store_true", help="exact: only start pairs whose estimate fits, never stop a pair at the deadline")
    g.add_argument("--min-start-min", type=float, default=15.0, help="exact: do not start a pair with less budget than this (default 15 min)")
    g.add_argument("--nchunks", type=int, default=NCHUNKS, help=argparse.SUPPRESS)
    g = ap.add_argument_group("download / status / plots / all")
    g.add_argument("--url", default=paths.CATALOGUE_URL)
    g.add_argument("--dir", default=None, help="download directory (default: the directory of --catalogue)")
    g.add_argument("--delete-gz", action="store_true", help="remove the 7.2 GB gz after a verified decompression")
    g.add_argument("--retries", type=int, default=20, help="download attempts (each one resumes)")
    g.add_argument("--session-hours", type=float, default=11.0, help="status: session length used for 'sessions needed'")
    g.add_argument("--json", action="store_true", help="status: also print a one-line JSON summary")
    g.add_argument("--skip-a4", action="store_true", help="plots: skip the estimator re-check (needs bins 1 and 6, about 5 min)")
    g.add_argument("--no-exact", action="store_true", help="all: stop after the b1.5 product and its plots")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = Cfg(a)
    st = load_state(cfg)
    signal.signal(signal.SIGTERM, _on_sigterm)
    if cfg.dry and a.cmd != "status":
        print(f"[dry-run] nothing is executed or written. threads={cfg.threads}, runs={cfg.runs}, work={cfg.work}, catalogue={cfg.catalogue}")
    session = {"cmd": a.cmd, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "threads": cfg.threads, "max_hours": a.max_hours,
               "cpu_count": os.cpu_count(), "argv": sys.argv[1:]}
    if a.cmd not in ("status", "compare") and not cfg.dry:
        st.setdefault("sessions", []).append(session)
        st["sessions"] = st["sessions"][-200:]
        save_state(cfg, st)
    try:
        rc = COMMANDS[a.cmd](cfg, st)
    except KeyboardInterrupt:
        print("\ninterrupted: finished work is on disk; run the same command again to resume", file=sys.stderr)
        rc = 130
    if a.cmd not in ("status", "compare") and not cfg.dry:
        session.update(end=time.strftime("%Y-%m-%d %H:%M:%S"), exit=rc, wall_s=round(time.time() - cfg.t0))
        save_state(cfg, st)
    return rc


if __name__ == "__main__":
    sys.exit(main())
