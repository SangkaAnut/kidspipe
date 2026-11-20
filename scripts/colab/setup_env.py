"""Per-session environment setup for Google Colab (also usable on any Linux box). Imported by the first cell of every notebook.

A Colab runtime is fresh in every session (the local disk is wiped, Google Drive is not). ``setup()``:

1. creates the persistent directories on Drive and the local scratch directory for the catalogue,
2. sets the environment variables that ``kidspipe.paths`` reads, in THIS process (so ``!python ...`` cells inherit them),
3. makes the numba cache usable across sessions (cache on Drive; source mtimes pinned to the commit time, see ``stabilise_mtimes``),
4. ``pip install -e '.[all]'`` when the package or one of its extras is missing (skipped when already importable),
5. prints the versions, cores, RAM and disk, and writes ``<runs>/session_info.json`` (provenance of the commit and the versions).

Layout (defaults; ``DRIVE_ROOT`` = /content/drive/MyDrive/kidspipe):

==============================  =============================================================  ====================================
``KIDSPIPE_RUNS``               ``DRIVE_ROOT/runs``              all run outputs (persistent)  small (< 1 GB for the DR5 chain)
``KIDSPIPE_DR5_WORK``           ``DRIVE_ROOT/dr5_work``          per-bin npz (persistent)      1.6 GB
``KIDSPIPE_DR5_CATALOGUE``      ``/content/dr5/KiDS_Legacy_...fits``  local disk (NOT Drive)   16.4 GB (+ 7.2 GB gz while downloading)
``KIDSPIPE_PLOTS``              ``DRIVE_ROOT/plots``             figures (the repo's plots/ keeps the shipped references)
``NUMBA_CACHE_DIR``             ``DRIVE_ROOT/numba_cache``       compiled functions            a few 10 MB
==============================  =============================================================  ====================================

Command line (for ``%%bash`` cells): ``python scripts/colab/setup_env.py --print-env`` prints ``export`` lines.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/kidspipe"
DEFAULT_CATALOGUE_DIR = "/content/dr5"
CATALOGUE_NAME = "KiDS_Legacy_NS_unblind_final.fits"
# import name -> pip package, for the "is everything installed" test of the [all] extra
EXTRAS = {"treecorr": "treecorr", "fitsio": "fitsio", "dynesty": "dynesty", "nautilus": "nautilus-sampler", "h5py": "h5py",
          "matplotlib": "matplotlib", "joblib": "joblib", "pytest": "pytest"}


def environment(drive_root: str | os.PathLike = DEFAULT_DRIVE_ROOT, catalogue_dir: str | os.PathLike = DEFAULT_CATALOGUE_DIR,
                numba_cache: str = "drive") -> dict[str, str]:
    """The environment variables of a Colab session (``numba_cache``: 'drive', 'local' or 'off')."""
    drive_root, catalogue_dir = Path(drive_root), Path(catalogue_dir)
    n = str(os.cpu_count() or 1)
    env = {
        "KIDSPIPE_RUNS": str(drive_root / "runs"),
        "KIDSPIPE_DR5_WORK": str(drive_root / "dr5_work"),
        "KIDSPIPE_DR5_CATALOGUE": str(catalogue_dir / CATALOGUE_NAME),
        "KIDSPIPE_PLOTS": str(drive_root / "plots"),
        "OMP_NUM_THREADS": n,
        "NUMBA_NUM_THREADS": n,
        "PYTHONUNBUFFERED": "1",
        "HDF5_USE_FILE_LOCKING": "FALSE",     # h5py / nautilus checkpoints on the Drive FUSE mount fail with "unable to lock file" otherwise
    }
    if numba_cache == "drive":
        env["NUMBA_CACHE_DIR"] = str(drive_root / "numba_cache")
    elif numba_cache == "local":
        env["NUMBA_CACHE_DIR"] = "/content/numba_cache"
    return env


def stabilise_mtimes(repo: Path = REPO) -> int:
    """Set the mtime of every ``src/**/*.py`` to the time of the HEAD commit.

    numba's on-disk cache is keyed on the source file's (mtime, size). A fresh ``git clone`` gives every file a new mtime, which would
    invalidate the cache on Drive in every session; pinning the mtime to the commit time makes the cache valid for the same commit
    (and automatically stale after a new commit)."""
    try:
        ts = int(subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%ct"], capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        return 0
    n = 0
    for f in (repo / "src").rglob("*.py"):
        try:
            os.utime(f, (ts, ts))
            n += 1
        except OSError:
            pass
    return n


def _missing_extras() -> list[str]:
    return [pip for mod, pip in EXTRAS.items() if importlib.util.find_spec(mod) is None]


def _kidspipe_from_repo(repo: Path) -> bool:
    spec = importlib.util.find_spec("kidspipe")
    return bool(spec and spec.origin and Path(spec.origin).resolve().is_relative_to(repo.resolve()))


def ensure_installed(repo: Path = REPO, extras: str = "all", force: bool = False) -> None:
    """``pip install -e '.[all]'`` unless kidspipe (from this checkout) and every extra can already be imported."""
    missing = _missing_extras()
    if force or missing or not _kidspipe_from_repo(repo):
        print(f"pip install -e '.[{extras}]'  (missing: {', '.join(missing) or 'kidspipe from this checkout'}) ...", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[{extras}]"], capture_output=True, text=True)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-3000:])
            raise RuntimeError("pip install failed (see the output above)")
        importlib.invalidate_caches()
    src = str(repo / "src")
    if src not in sys.path:                      # editable installs can need a kernel restart to be seen; this makes it work at once
        sys.path.insert(0, src)


def versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for m in ("numpy", "scipy", "numba", "astropy", "camb", "treecorr", "fitsio", "mpmath", "matplotlib", "dynesty", "nautilus", "h5py", "pytest"):
        try:
            mod = importlib.import_module(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception as e:                    # noqa: BLE001
            out[m] = f"NOT AVAILABLE ({type(e).__name__})"
    return out


def machine() -> dict[str, float | int | str]:
    ram = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) / 1e6
    except Exception:
        pass
    local = shutil.disk_usage("/content" if Path("/content").exists() else "/")
    return {"cpu_count": os.cpu_count() or 1, "ram_gb": round(ram, 1), "local_disk_free_gb": round(local.free / 1e9, 1)}


def git_head(repo: Path = REPO) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return "?"


def setup(repo: str | os.PathLike = REPO, drive_root: str | os.PathLike = DEFAULT_DRIVE_ROOT, catalogue_dir: str | os.PathLike = DEFAULT_CATALOGUE_DIR,
          install: bool = True, numba_cache: str = "drive", quiet: bool = False) -> dict:
    """Prepare the session; returns a dict with the environment, versions and machine description."""
    repo = Path(repo).resolve()
    drive_root = Path(drive_root)
    if not drive_root.parent.exists():
        print(f"WARNING: {drive_root.parent} does not exist: is Google Drive mounted? Outputs go to the local disk and are LOST when the "
              f"session ends. Falling back to /content/kidspipe_data.")
        drive_root = Path("/content/kidspipe_data")
    env = environment(drive_root, catalogue_dir, numba_cache)
    for k in ("KIDSPIPE_RUNS", "KIDSPIPE_DR5_WORK", "KIDSPIPE_PLOTS"):
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    Path(env["KIDSPIPE_DR5_CATALOGUE"]).parent.mkdir(parents=True, exist_ok=True)
    if "NUMBA_CACHE_DIR" in env:
        Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    os.environ["KIDSPIPE_ROOT"] = str(repo)
    os.environ["PYTHONDONTWRITEBYTECODE"] = os.environ.get("PYTHONDONTWRITEBYTECODE", "1")
    npin = stabilise_mtimes(repo) if numba_cache != "off" else 0
    if install:
        ensure_installed(repo)
    ver, mach = versions(), machine()
    info = {"commit": git_head(repo), "env": env, "versions": ver, "machine": mach, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "mtimes_pinned": npin}
    try:
        (Path(env["KIDSPIPE_RUNS"]) / "session_info.json").write_text(json.dumps(info, indent=1))
    except OSError:
        pass
    if not quiet:
        print(f"repo {repo}  commit {info['commit']}")
        print(f"machine: {mach['cpu_count']} vCPU, {mach['ram_gb']} GB RAM, {mach['local_disk_free_gb']} GB free on the local disk")
        for k, v in env.items():
            print(f"  {k}={v}")
        print("versions: " + ", ".join(f"{k} {v}" for k, v in ver.items()))
        if mach["ram_gb"] and mach["ram_gb"] < 20:
            print("note: < 20 GB RAM (free tier): run ONE DR5 pair per process (scripts/dr5_chain.py does); a cross pair needs about 5 GB.")
    return info


# ------------------------------------------------------------------------------------------------ helpers for long jobs
def launch(cmd: str, log: str | os.PathLike) -> subprocess.Popen:
    """Start a shell command detached from the notebook cell, output appended to ``log`` (e.g. on Drive).

    A cell that blocks for hours is lost when the browser tab is closed; a detached job keeps running while the runtime lives.
    Follow it with ``tail(log)``; stop it with ``stop(p)`` (NOT ``p.terminate()``: that signals only the ``sh`` wrapper, not the chain;
    ``stop`` signals the whole process group, and the DR5 chain then stops its pair cleanly: finished work stays on disk)."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "ab", buffering=0)
    f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {cmd}\n".encode())
    return subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy(), cwd=str(REPO))


def stop(p: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    """Signal the whole process group of a job started with ``launch`` (or ``sh``): the shell, the chain and its children.
    ``launch`` makes the child a session leader, so its process group id is ``p.pid``."""
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass


def tail(log: str | os.PathLike, n: int = 30) -> str:
    try:
        lines = Path(log).read_text(errors="replace").splitlines()
    except OSError:
        return "(no log yet)"
    return "\n".join(lines[-n:])


def sh(cmd: str, cwd: str | os.PathLike | None = None) -> int:
    """Run a shell command, streaming its output into the notebook cell; returns the exit status (Colab does not always show the
    output of a plain ``subprocess.call``)."""
    p = subprocess.Popen(cmd, shell=True, cwd=str(cwd or REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=os.environ.copy(), start_new_session=True)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        stop(p)                      # the child has its own session: signal the whole group, not only the shell
        p.wait()
        raise
    return p.wait()


def stage(name: str, cmd: str, force: bool = False, cwd: str | os.PathLike | None = None) -> None:
    """Resumable pipeline stage: skip when ``<runs>/stages/<name>.done`` exists, run ``cmd`` otherwise and write the marker only when it
    exits with status 0. After a lost session, re-running the same cell skips the finished stages."""
    d = Path(os.environ.get("KIDSPIPE_RUNS", REPO / "runs")) / "stages"
    d.mkdir(parents=True, exist_ok=True)
    mark = d / f"{name}.done"
    if mark.exists() and not force:
        print(f"[skip] {name}: done ({mark.read_text().strip()})")
        return
    print(f"[run ] {name}: {cmd}", flush=True)
    t = time.time()
    rc = sh(cmd, cwd)
    if rc != 0:
        raise RuntimeError(f"stage '{name}' failed with exit status {rc}")
    mark.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {(time.time() - t) / 60:.1f} min")
    print(f"[done] {name} in {(time.time() - t) / 60:.1f} min", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print-env", action="store_true", help="print `export` lines for a shell")
    ap.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    ap.add_argument("--catalogue-dir", default=DEFAULT_CATALOGUE_DIR)
    ap.add_argument("--numba-cache", choices=["drive", "local", "off"], default="drive")
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args(argv)
    if a.print_env:
        for k, v in environment(a.drive_root, a.catalogue_dir, a.numba_cache).items():
            print(f"export {k}='{v}'")
        return 0
    setup(REPO, a.drive_root, a.catalogue_dir, install=not a.no_install, numba_cache=a.numba_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
