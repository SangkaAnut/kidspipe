"""Step A (chunked): measure xi+/xi-/xi_x for the 21 tomographic pairs in resumable spatial chunks.

WHY CHUNKING IS EXACT HERE
--------------------------
TreeCorr counts ORDERED pairs, so for a partition of the sky into chunks {c}:

    process(cat, cat)  ==  sum_c process(c_c, c_c)  +  2 * sum_{a<b} process(c_a, c_b)

Verified to npairs ratio 1.000000000 and sum(weight) ratio 1.000000000 for K=4 and K=8, on an
auto pair and on a cross pair (scripts/dr5_chunk_diagnose.py, scripts/dr5_chunk_verify.py; logs
in runs/dr5_measure/chunk_diagnose.log and chunk_verify.log).

NOTE the factor 2 on cross-chunk pairs. Omitting it loses 3.3% (measured) -- that is exactly the
mistake that makes naive chunking look inexact.

WHAT THIS BUYS
--------------
The 21-pair production run is ~30-50 h in one process. Here the unit of work is
(pair, chunk-a, chunk-b), each written to its own npz, so:
  * an interruption loses at most one chunk-pair, not the whole run;
  * work can be distributed across processes/machines by simply picking different --chunk ids;
  * progress is inspectable at any time with --status;
  * a chunk-pair that finishes is never recomputed (existence check on its npz).

xi+ is accumulated as sum(w*xi)/sum(w) across chunk-pairs, which is exactly how TreeCorr's own
finalize() combines patches.

Usage
-----
  # what is left, and the estimated cost
  python3 scripts/dr5_measure_xi_chunked.py --status --nchunks 16

  # run one chunk-pair (or a range) in this process
  python3 scripts/dr5_measure_xi_chunked.py --chunk 3 --nchunks 16 --threads 16

  # run everything not yet done, one chunk-pair at a time, in this process
  python3 scripts/dr5_measure_xi_chunked.py --all --nchunks 16 --threads 16

  # the same over MPI: rank 0 schedules chunk-pairs (one tomographic pair at a time), ranks 1..N-1 compute,
  # each with --threads threads; memory per worker ~2.5 GB (auto pair) to ~5 GB (cross pair)
  mpirun -n 3 python3 scripts/dr5_measure_xi_chunked.py --all --mpi --nchunks 16 --threads 10

  # assemble the finished chunk-pairs into one npz per tomographic pair
  python3 scripts/dr5_measure_xi_chunked.py --assemble --nchunks 16

Output tree: <runs>/dr5_measure/xi_chunked/   (<runs> = $KIDSPIPE_RUNS, default <root>/runs; the chunk files are resumable, so keep it on
persistent storage on Colab)
  chunks/Bin{i}_Bin{j}/c{a:02d}_{b:02d}.npz     per chunk-pair
  assembled/ns/Bin{i}_Bin{j}.npz                same schema as dr5_measure_xi.py
  manifest.json                                  parameters + provenance
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidspipe import paths  # noqa: E402
from kidspipe.measure import official, xi  # noqa: E402
from kidspipe.measure.official import PAIRS  # noqa: E402

_ORIG_LOAD_BIN = xi.load_bin

#: output tree: <runs>/dr5_measure/xi_chunked with <runs> = $KIDSPIPE_RUNS (default <root>/runs); the per-bin input extract is
#: read from $KIDSPIPE_DR5_WORK (kidspipe.measure.xi.load_bin).  --threads defaults to min(16, os.cpu_count()).
OUTROOT = paths.runs_dir() / "dr5_measure" / "xi_chunked"
NLETTER = {1: 7416371, 2: 7359911, 3: 6799681, 4: 6880432, 5: 6477538, 6: 5960461}

# Conservative spherical bounding-cap pre-pruning constants (matches unmerged Prompt 01
# ``src/kidspipe/measure/chunk_geometry.py`` verbatim; duplicated locally on ``main`` so this
# branch modifies only its assigned files — note for maintainer when merging 01 and 09).
TMAX_ARCMIN: float = float(xi.TMAX)
TMAX_RAD: float = float(np.deg2rad(TMAX_ARCMIN / 60.0))
SAFETY_REL: float = 1e-9
SAFETY_ABS: float = 1e-12
SKIP_RULE: str = "angdist(centre_a, centre_b) - r_a - r_b > TMAX_rad * (1 + 1e-9) + 1e-12"
CAPS_CACHE_VERSION: int = 1
_CHUNK_STEM_RE = re.compile(r"^c(\d+)_(\d+)$")


# ----------------------------------------------------------------------------- chunking
def chunk_edges(ra: np.ndarray, nchunks: int) -> np.ndarray:
    """Deterministic quantile RA strip edges for ``nchunks`` equal-count chunks."""
    nchunks = int(nchunks)
    if nchunks < 1:
        raise ValueError(f"nchunks must be >= 1, got {nchunks}")
    ra_arr = np.asarray(ra, np.float64)
    if ra_arr.size == 0:
        return np.zeros(nchunks + 1, dtype=np.float64)
    return np.quantile(ra_arr, np.linspace(0.0, 1.0, nchunks + 1))


def _chunk_mask_from_edges(ra: np.ndarray, edges: np.ndarray, k: int, nchunks: int) -> np.ndarray:
    lo, hi = edges[k], edges[k + 1]
    return (ra >= lo) & (ra < hi) if k < nchunks - 1 else (ra >= lo) & (ra <= hi)


def chunk_masks(ra: np.ndarray, nchunks: int) -> list[np.ndarray]:
    """Equal-count RA strips. Deterministic given the catalogue, so chunk ids are reproducible.

    Equal-count (not equal-area) strips: this guarantees balanced work per chunk, which is what
    matters for load distribution. Pairs between distant strips are computed all the same, so the
    decomposition is exact regardless of how the sky is cut.
    """
    nchunks = int(nchunks)
    ra_arr = np.asarray(ra, np.float64)
    if ra_arr.size == 0:
        return [np.zeros(0, dtype=bool) for _ in range(nchunks)]
    edges = chunk_edges(ra_arr, nchunks)
    return [_chunk_mask_from_edges(ra_arr, edges, k, nchunks) for k in range(nchunks)]


def chunk_pairs(nchunks: int, same_bin: bool) -> list[tuple[int, int]]:
    """Chunk-pair list. For an auto pair the fold factor 2 is applied at assembly time."""
    if same_bin:
        return [(a, b) for a in range(nchunks) for b in range(a, nchunks)]
    return [(a, b) for a in range(nchunks) for b in range(nchunks)]


def fold_factor(a: int, b: int, same_bin: bool) -> float:
    if same_bin and a != b:
        return 2.0
    return 1.0


# ----------------------------------------------------------------------------- bounding-cap pruning
def unit_vectors_rad(ra_rad: Any, dec_rad: Any) -> np.ndarray:
    """Convert ``(ra, dec)`` in radians to ``(N, 3)`` float64 Cartesian unit vectors."""
    ra = np.asarray(ra_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    cos_dec = np.cos(dec)
    return np.stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1)


def angdist(u: Any, v: Any) -> float | np.ndarray:
    """Kahan great-circle angular separation (radians) between unit vectors ``u`` and ``v``."""
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    d = 2.0 * np.arctan2(
        np.linalg.norm(u_arr - v_arr, axis=-1),
        np.linalg.norm(u_arr + v_arr, axis=-1),
    )
    if u_arr.ndim == 1 and v_arr.ndim == 1:
        return float(d)
    return np.asarray(d, dtype=np.float64)


def bounding_cap_rad(ra_rad: Any, dec_rad: Any) -> dict[str, Any]:
    """Compute a conservative spherical bounding cap for coordinates in radians."""
    ra_arr = np.atleast_1d(np.asarray(ra_rad, dtype=np.float64))
    dec_arr = np.atleast_1d(np.asarray(dec_rad, dtype=np.float64))
    if ra_arr.shape != dec_arr.shape:
        raise ValueError(f"ra and dec shape mismatch: {ra_arr.shape} vs {dec_arr.shape}")
    n = int(ra_arr.size)
    if n == 0:
        return {"centre": [1.0, 0.0, 0.0], "radius_rad": 0.0, "nobj": 0, "empty": True}

    u = unit_vectors_rad(ra_arr.ravel(), dec_arr.ravel())
    m = u.mean(axis=0)
    norm = float(np.linalg.norm(m))
    if norm > 1e-12:
        centre = m / norm
    else:
        centre = u[0] / np.linalg.norm(u[0])
    centre = centre / np.linalg.norm(centre)
    radius_rad = 0.0 if n == 1 else float(np.max(angdist(u, centre)))
    return {
        "centre": [float(c) for c in centre],
        "radius_rad": radius_rad,
        "nobj": n,
        "empty": False,
    }


def skip_threshold_rad(tmax_arcmin: float = TMAX_ARCMIN) -> float:
    """Maximum separation threshold in radians, inflated by relative and absolute safety margins."""
    return float(np.deg2rad(float(tmax_arcmin) / 60.0) * (1.0 + SAFETY_REL) + SAFETY_ABS)


def can_skip_pair(cap_a: dict[str, Any], cap_b: dict[str, Any], tmax_arcmin: float = TMAX_ARCMIN) -> bool:
    """Return True iff chunk-pair ``(cap_a, cap_b)`` is provably disjoint beyond ``tmax_arcmin``."""
    if cap_a.get("empty", False) or int(cap_a.get("nobj", 0)) == 0:
        return True
    if cap_b.get("empty", False) or int(cap_b.get("nobj", 0)) == 0:
        return True
    ca = np.asarray(cap_a["centre"], dtype=np.float64)
    cb = np.asarray(cap_b["centre"], dtype=np.float64)
    min_dist = float(angdist(ca, cb)) - float(cap_a["radius_rad"]) - float(cap_b["radius_rad"])
    return bool(min_dist > skip_threshold_rad(tmax_arcmin))


def partition_chunk_pairs(
    caps_a: list[dict[str, Any]],
    caps_b: list[dict[str, Any]],
    same_bin: bool,
    tmax_arcmin: float = TMAX_ARCMIN,
    candidates: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition candidate chunk-pairs into ``(active_pairs, skipped_pairs)`` in sorted order."""
    if candidates is None:
        if same_bin:
            if len(caps_a) != len(caps_b):
                raise ValueError("same_bin=True requires len(caps_a) == len(caps_b)")
            pairs = chunk_pairs(len(caps_a), True)
        else:
            pairs = [(a, b) for a in range(len(caps_a)) for b in range(len(caps_b))]
    else:
        pairs = sorted({(int(a), int(b)) for a, b in candidates})

    active: list[tuple[int, int]] = []
    skipped: list[tuple[int, int]] = []
    for a, b in pairs:
        if can_skip_pair(caps_a[a], caps_b[b], tmax_arcmin=tmax_arcmin):
            skipped.append((a, b))
        else:
            active.append((a, b))
    return active, skipped


def compute_caps_from_catalogs(cats: Any) -> list[dict[str, Any]]:
    """Compute bounding caps from Catalog-like objects with ``.ra, .dec`` (radians) and ``.nobj``."""
    caps: list[dict[str, Any]] = []
    for k, cat in enumerate(cats):
        if cat is None or int(cat.nobj) == 0:
            cap = bounding_cap_rad(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
        else:
            cap = bounding_cap_rad(cat.ra, cat.dec)
        cap["chunk"] = int(k)
        caps.append(cap)
    return caps


def _close_memmap(arr: Any) -> None:
    """Close underlying mmap object on a ``numpy.memmap`` array so FD/pages release promptly."""
    if isinstance(arr, np.memmap):
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            mm.close()


def _load_bin_columns(
    b: int,
    *,
    work: Path | None = None,
    mmap: bool = False,
    columns: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Call ``xi.load_bin`` while respecting older monkeypatched test doubles that lack kwargs."""
    if mmap:
        kw: dict[str, Any] = {"mmap": True}
        if work is not None:
            kw["work"] = work
        if columns is not None:
            kw["columns"] = columns
        return xi.load_bin(b, **kw)
    if xi.load_bin is _ORIG_LOAD_BIN:
        kw = {}
        if work is not None:
            kw["work"] = work
        if columns is not None:
            kw["columns"] = columns
        return xi.load_bin(b, **kw)
    if work is not None:
        return xi.load_bin(b, work=work)
    return xi.load_bin(b)


def compute_bin_caps(
    b: int,
    nchunks: int,
    *,
    work: Path | None = None,
    mmap: bool = False,
    edges: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], int, np.ndarray]:
    """Compute per-chunk bounding caps streaming only ``ra`` and ``dec`` (no shear arrays in RAM)."""
    ra_rec = _load_bin_columns(b, work=work, mmap=mmap, columns=("ra", "dec") if xi.load_bin is not _ORIG_LOAD_BIN else ("ra",))
    ra_arr = np.asarray(ra_rec["ra"], dtype=np.float64) if not mmap else ra_rec["ra"]
    nobj = int(ra_arr.size)
    if edges is None:
        edges = chunk_edges(ra_arr, nchunks)
    idx_by_k = [np.flatnonzero(_chunk_mask_from_edges(ra_arr, edges, k, nchunks)).astype(np.int32)
                for k in range(nchunks)]
    ra_chunks = [np.deg2rad(np.asarray(ra_arr[idx], dtype=np.float64)) for idx in idx_by_k]

    dec_in_rec = "dec" in ra_rec
    if dec_in_rec:
        dec_arr = ra_rec["dec"]
    else:
        _close_memmap(ra_rec["ra"])
        del ra_arr, ra_rec
        dec_rec = _load_bin_columns(b, work=work, mmap=mmap, columns=("dec",))
        dec_arr = dec_rec["dec"]

    caps: list[dict[str, Any]] = []
    for k in range(nchunks):
        dec_rad_k = np.deg2rad(np.asarray(dec_arr[idx_by_k[k]], dtype=np.float64))
        cap = bounding_cap_rad(ra_chunks[k], dec_rad_k)
        cap["chunk"] = int(k)
        caps.append(cap)
        ra_chunks[k] = None
        idx_by_k[k] = None

    if dec_in_rec:
        _close_memmap(ra_rec["ra"])
        _close_memmap(ra_rec["dec"])
        del ra_rec
    else:
        _close_memmap(dec_rec["dec"])
        del dec_rec
    del dec_arr, ra_chunks, idx_by_k
    gc.collect()
    return caps, nobj, edges


def load_caps_cache(cache_path: Path, nchunks: int) -> dict[int, dict[str, Any]]:
    """Load cached per-bin caps if valid for ``CAPS_CACHE_VERSION`` and ``nchunks``."""
    cache_path = Path(cache_path)
    nchunks = int(nchunks)
    if not cache_path.is_file():
        return {}
    try:
        data = json.loads(cache_path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get("version") != CAPS_CACHE_VERSION or data.get("nchunks") != nchunks:
        return {}
    bins = data.get("bins")
    if not isinstance(bins, dict):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in bins.items():
        if not isinstance(v, dict):
            return {}
        out[int(k)] = v
    return out


def save_caps_cache(cache_path: Path, nchunks: int, bins_caps: dict[int, dict[str, Any]]) -> None:
    """Merge ``bins_caps`` into ``cache_path`` atomically."""
    cache_path = Path(cache_path)
    nchunks = int(nchunks)
    merged = load_caps_cache(cache_path, nchunks)
    for k, v in bins_caps.items():
        merged[int(k)] = v
    payload = {
        "version": CAPS_CACHE_VERSION,
        "nchunks": nchunks,
        "chunking": "equal-count RA strips, quantile edges",
        "bins": {str(k): merged[k] for k in sorted(merged)},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f".{cache_path.name}.part")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, cache_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _resolve_work_key(work: Path | None) -> str:
    return str(Path(work if work is not None else xi.WORK).expanduser().resolve())


def get_bin_caps(
    b: int,
    nchunks: int,
    *,
    cache_path: Path,
    work: Path | None = None,
    mmap: bool = False,
    edges_cache: dict[int, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Return per-chunk bounding caps for bin ``b``, using ``chunk_caps.json`` when valid."""
    b = int(b)
    nchunks = int(nchunks)
    work_key = _resolve_work_key(work)
    cached = load_caps_cache(cache_path, nchunks)
    entry = cached.get(b)
    if isinstance(entry, dict) and isinstance(entry.get("caps"), list) and len(entry["caps"]) == nchunks:
        entry_work = entry.get("work")
        if entry_work is None or entry_work == work_key:
            if not mmap:
                return entry["caps"]
            ra_npy = Path(work if work is not None else xi.WORK) / f"bin{b - 1}" / "ra.npy"
            if ra_npy.is_file():
                mm = np.load(ra_npy, mmap_mode="r")
                nobj_ok = (entry.get("nobj") == int(mm.shape[0]))
                _close_memmap(mm)
                del mm
                if nobj_ok:
                    return entry["caps"]

    edges = edges_cache.get(b) if edges_cache is not None else None
    caps, nobj, edges = compute_bin_caps(b, nchunks, work=work, mmap=mmap, edges=edges)
    if edges_cache is not None:
        edges_cache[b] = edges
    save_caps_cache(cache_path, nchunks, {b: {"nobj": nobj, "work": work_key, "caps": caps}})
    return caps


def read_manifest_zero_pairs(outroot: Path, pair_name: str, nchunks: int) -> list[tuple[int, int]] | None:
    """Read ``analytically_zero_pairs[pair_name]`` from ``manifest.json`` if matching ``nchunks``."""
    mpath = Path(outroot) / "manifest.json"
    if not mpath.is_file():
        return None
    try:
        data = json.loads(mpath.read_text())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("nchunks") != int(nchunks):
        return None
    azp = data.get("analytically_zero_pairs")
    if not isinstance(azp, dict) or pair_name not in azp:
        return None
    return sorted({(int(ab[0]), int(ab[1])) for ab in azp[pair_name]})


# ----------------------------------------------------------------------------- slop bookkeeping
def slop_record(bin_slop: float, angle_slop: float | None) -> dict:
    """Scalars stored in a chunk npz so that --assemble can refuse to mix settings.

    Nothing is added for the legacy configuration (bin_slop 0, angle_slop default), so existing chunk files and
    default runs stay byte-compatible; 'bin_slop' is stored whenever it is non-zero or angle_slop is set, and
    'angle_slop' whenever it is set.  A missing key therefore means: bin_slop 0 / angle_slop = TreeCorr default."""
    rec: dict = {}
    if angle_slop is not None:
        rec["angle_slop"] = np.float64(angle_slop)
    if angle_slop is not None or float(bin_slop) != 0.0:
        rec["bin_slop"] = np.float64(bin_slop)
    return rec


def chunk_slops(npz) -> tuple[float, float | None]:
    """(bin_slop, angle_slop) of a chunk file; angle_slop None = TreeCorr default."""
    b = float(npz["bin_slop"]) if "bin_slop" in npz.files else 0.0
    a = float(npz["angle_slop"]) if "angle_slop" in npz.files else None
    return b, a


# ----------------------------------------------------------------------------- catalogue
def _evict_other_bins(cache: dict, keep_bins: tuple[int, ...]) -> None:
    evicted = False
    for kb in [k for k in list(cache) if k not in keep_bins]:
        for c in cache[kb]:
            if c is not None and hasattr(c, "clear_cache"):
                c.clear_cache()
        del cache[kb]
        evicted = True
    if evicted:
        gc.collect()


def _release_chunk(cache: dict, uses: dict[tuple[int, int], int], b: int, k: int) -> None:
    key = (b, k)
    if key not in uses:
        return
    uses[key] -= 1
    if uses[key] <= 0:
        lst = cache.get(b)
        if lst is not None and 0 <= k < len(lst) and lst[k] is not None:
            if hasattr(lst[k], "clear_cache"):
                lst[k].clear_cache()
            lst[k] = None


def load_pair_cats(
    b1: int,
    b2: int,
    nchunks: int,
    cache: dict,
    *,
    mmap: bool = False,
    work: Path | None = None,
    needed: dict[int, set[int]] | None = None,
    edges_cache: dict[int, np.ndarray] | None = None,
):
    """Load needed chunk Catalog objects for bin ``b1``/``b2``, streaming one column at a time."""
    for b in {b1, b2}:
        want = list(range(nchunks)) if needed is None else sorted(int(k) for k in needed.get(b, ()))
        if b not in cache or not isinstance(cache[b], list) or len(cache[b]) != nchunks:
            cache[b] = [None] * nchunks
        missing = [k for k in want if cache[b][k] is None]
        if not missing:
            continue

        if mmap or (xi.load_bin is _ORIG_LOAD_BIN):
            ra_rec = _load_bin_columns(b, work=work, mmap=mmap, columns=("ra",))
            ra_arr = ra_rec["ra"]
            n_total = int(ra_arr.size)
            edges = edges_cache.get(b) if edges_cache is not None else None
            if edges is None:
                edges = chunk_edges(ra_arr, nchunks)
                if edges_cache is not None:
                    edges_cache[b] = edges
            idx_by_k = {
                k: np.flatnonzero(_chunk_mask_from_edges(ra_arr, edges, k, nchunks)).astype(np.int32)
                for k in missing
            }
            subs = {
                k: {"ra": np.array(ra_arr[idx_by_k[k]], dtype=np.float64, copy=True)}
                for k in missing
            }
            _close_memmap(ra_arr)
            del ra_arr, ra_rec

            for col in ("dec", "e1", "e2", "w"):
                col_rec = _load_bin_columns(b, work=work, mmap=mmap, columns=(col,))
                col_arr = col_rec[col]
                for k in missing:
                    subs[k][col] = np.array(col_arr[idx_by_k[k]], dtype=np.float64, copy=True)
                _close_memmap(col_arr)
                del col_arr, col_rec

            del idx_by_k
            for k in missing:
                cache[b][k] = xi.make_treecorr_cat(subs.pop(k))
            del subs
            gc.collect()
        else:
            rec = _load_bin_columns(b, work=work, mmap=False)
            ra_arr = rec["ra"]
            n_total = int(ra_arr.size)
            edges = chunk_edges(ra_arr, nchunks)
            for k in missing:
                m = _chunk_mask_from_edges(ra_arr, edges, k, nchunks)
                sub = {kk: rec[kk][m] for kk in ("ra", "dec", "e1", "e2", "w")}
                cache[b][k] = xi.make_treecorr_cat(sub)
            del rec

        sizes = [int(cache[b][k].nobj) for k in missing if cache[b][k] is not None]
        mode = "mmap" if mmap else "npz"
        rng_txt = f"{min(sizes):,} .. {max(sizes):,}" if sizes else "0"
        print(f"  loaded bin{b} ({mode}, {len(missing)}/{nchunks} chunks): {n_total:,} galaxies "
              f"({rng_txt} per loaded chunk)", flush=True)
    return cache[b1], cache[b2]


_DEFAULT_LOAD_PAIR_CATS = load_pair_cats


def write_chunk_atomic(f: Path, arr: dict) -> None:
    """Write a chunk npz under a temporary name, then rename: an interrupted run never leaves a partial c*.npz
    (the temporary name '.cAA_BB.npz.part' does not match the 'c*.npz' glob of --assemble/--status)."""
    tmp = f.with_name("." + f.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arr)
    tmp.replace(f)


def compute_chunk(cats1, cats2, a: int, b: int, same: bool, args) -> dict:
    gg = xi.run_pair(cats1[a], cats2[b], nbins=args.nbins, bin_slop=args.bin_slop, num_threads=args.threads,
                     angle_slop=args.angle_slop)
    arr = xi.corr_to_arrays(gg)
    arr["fold"] = np.float64(fold_factor(a, b, same))
    arr.update(slop_record(args.bin_slop, args.angle_slop))
    return arr


def todo_pairs(args) -> list:
    todo = PAIRS
    if args.pairs:
        want = {tuple(int(x) - 1 for x in p.split("-")) for p in args.pairs.split(",")}
        todo = [(i, j) for i, j in PAIRS if (i, j) in want]
    return todo


def write_manifest(args, t0, n_done, n_skip, extra=None, zero_pairs_map=None) -> None:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    mpath = OUTROOT / "manifest.json"
    merged_zero: dict[str, list[list[int]]] = {}
    if mpath.is_file():
        try:
            prev = json.loads(mpath.read_text())
            if prev.get("nchunks") == args.nchunks and isinstance(prev.get("analytically_zero_pairs"), dict):
                for k, v in prev["analytically_zero_pairs"].items():
                    merged_zero[str(k)] = sorted([int(a), int(b)] for a, b in v)
        except Exception:
            pass
    if zero_pairs_map:
        for k, v in zero_pairs_map.items():
            merged_zero[str(k)] = sorted([int(a), int(b)] for a, b in v)
    payload = {
        "nchunks": args.nchunks, "nbins": args.nbins, "bin_slop": args.bin_slop,
        "angle_slop": args.angle_slop,   # null = TreeCorr default (0.1 rad in TreeCorr 5)
        "threads": args.threads,
        "mmap": bool(getattr(args, "mmap", False)),
        "prune_zero_pairs": bool(getattr(args, "prune", True)),
        "analytically_zero_rule": SKIP_RULE,
        "analytically_zero_tmax_arcmin": TMAX_ARCMIN,
        "analytically_zero_pairs": merged_zero,
        "chunk_pairs_analytically_zero": sum(len(v) for v in merged_zero.values()),
        "binning": "Log, 1000 bins, 2-300 arcmin, sep_units='arcmin'",
        "chunking": "equal-count RA strips, quantile edges",
        "fold_rule": "auto pairs: 2 * sum_{a<b} process(c_a,c_b); cross pairs: no fold",
        "exactness_check": "chunk_diagnose.log, chunk_verify.log (ratio 1.000000000)",
        "partial_total_seconds": time.time() - t0,
        "chunk_pairs_written": n_done, "chunk_pairs_skipped": n_skip, **(extra or {}),
    }
    tmp = mpath.with_name(".manifest.json.part")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, mpath)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _partition_pair_work(
    b1: int,
    b2: int,
    args,
    chunk_list,
    *,
    cats1=None,
    cats2=None,
    work: Path | None = None,
    edges_cache: dict[int, np.ndarray] | None = None,
):
    """Determine active/skipped chunk-pairs for ``(b1, b2)`` and filter requested ``work_cps``."""
    same = (b1 == b2)
    all_cps = chunk_pairs(args.nchunks, same)
    if getattr(args, "prune", True):
        if cats1 is not None and cats2 is not None:
            caps1 = compute_caps_from_catalogs(cats1)
            caps2 = caps1 if same else compute_caps_from_catalogs(cats2)
        else:
            cache_path = OUTROOT / "chunk_caps.json"
            mmap = bool(getattr(args, "mmap", False))
            caps1 = get_bin_caps(b1, args.nchunks, cache_path=cache_path, work=work,
                                 mmap=mmap, edges_cache=edges_cache)
            caps2 = caps1 if same else get_bin_caps(
                b2, args.nchunks, cache_path=cache_path, work=work,
                mmap=mmap, edges_cache=edges_cache,
            )
        active_all, zero_all = partition_chunk_pairs(caps1, caps2, same)
    else:
        active_all, zero_all = all_cps, []

    zero_set = set(zero_all)
    if chunk_list is None:
        work_cps = active_all
    else:
        work_cps = []
        for ab in chunk_list:
            if ab in zero_set:
                print(f"Bin{b1}_Bin{b2} c{ab[0]:02d}_{ab[1]:02d}: analytically zero (skipped)", flush=True)
            else:
                work_cps.append(ab)
    return active_all, zero_all, work_cps


# ----------------------------------------------------------------------------- MPI workers
_WCACHE: dict = {}


def _mpi_task(job):
    """One chunk-pair on a worker rank.  Keeps only the current pair's needed chunks in memory."""
    b1, b2, a, b, args = job
    _evict_other_bins(_WCACHE, (b1, b2))
    work = args.work.expanduser() if getattr(args, "work", None) is not None else None
    mmap = bool(getattr(args, "mmap", False))
    need = {b1: {a, b}} if b1 == b2 else {b1: {a}, b2: {b}}
    cats1, cats2 = load_pair_cats(b1, b2, args.nchunks, _WCACHE, mmap=mmap, work=work, needed=need)
    f = OUTROOT / "chunks" / f"Bin{b1}_Bin{b2}" / f"c{a:02d}_{b:02d}.npz"
    t1 = time.time()
    arr = compute_chunk(cats1, cats2, a, b, b1 == b2, args)
    write_chunk_atomic(f, arr)
    from kidspipe import mpi
    return (b1, b2, a, b, time.time() - t1, float(arr["npairs"].sum()), float(arr["weight"].sum()), mpi.rank())


def _mpi_init(outroot: str):
    global OUTROOT
    OUTROOT = Path(outroot)


def run_mpi(args, todo, chunk_list) -> None:
    """Rank 0 hands out active chunk-pairs of one tomographic pair at a time (heaviest first)."""
    from kidspipe import mpi
    nworkers = mpi.require_workers(1)
    work = args.work.expanduser() if getattr(args, "work", None) is not None else None
    with mpi.executor(_mpi_init, (str(OUTROOT),), threads=args.threads) as ex:
        if ex is None:
            return
        print(f"MPI: {nworkers} worker ranks x {args.threads} threads; rank 0 schedules (mmap={args.mmap}, "
              f"prune={args.prune}).", flush=True)
        pool = mpi.Pool(ex, nworkers)
        t0 = time.time()
        n_done = n_skip = 0
        edges_cache: dict[int, np.ndarray] = {}
        zero_pairs_map: dict[str, list[tuple[int, int]]] = {}
        for (i, j) in todo:
            b1, b2 = i + 1, j + 1
            pname = f"Bin{b1}_Bin{b2}"
            d = OUTROOT / "chunks" / pname
            d.mkdir(parents=True, exist_ok=True)
            _, zero_all, cps = _partition_pair_work(
                b1, b2, args, chunk_list, work=work, edges_cache=edges_cache,
            )
            zero_pairs_map[pname] = zero_all
            jobs = []
            for (a, b) in cps:
                if (d / f"c{a:02d}_{b:02d}.npz").exists() and not args.force:
                    n_skip += 1
                    continue
                jobs.append((b1, b2, a, b, args))
            jobs.sort(key=lambda jb: abs(jb[2] - jb[3]))          # near-diagonal chunk-pairs carry most pairs: start them first
            for (_, _, a, b, dt, npairs, w, rk) in pool.imap_unordered(_mpi_task, jobs):
                n_done += 1
                print(f"[{time.time()-t0:8.0f}s] {pname} c{a:02d}_{b:02d}: {dt:6.1f}s npairs={npairs:.3e} "
                      f"w={w:.3e} (rank {rk})", flush=True)
        write_manifest(args, t0, n_done, n_skip,
                       {"mpi_workers": nworkers, "threads_per_worker": args.threads},
                       zero_pairs_map=zero_pairs_map)
        print(f"\n{n_done} chunk-pairs written, {n_skip} already present, {time.time()-t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- main
def main() -> None:
    global OUTROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchunks", type=int, default=16)
    ap.add_argument("--pairs", default="", help="comma list i-j (1-based); default all 21")
    ap.add_argument("--chunk", default="", help="chunk-pair a-b to run (a,b chunk ids)")
    ap.add_argument("--all", action="store_true", help="run every chunk-pair not yet done")
    ap.add_argument("--assemble", action="store_true", help="assemble chunk-pairs into pair npz")
    ap.add_argument("--status", action="store_true", help="report progress and estimated cost")
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1), help="TreeCorr threads (default min(16, os.cpu_count()); per worker rank under --mpi)")
    ap.add_argument("--bin-slop", type=float, default=0.0)
    ap.add_argument("--angle-slop", type=float, default=None,
                    help="TreeCorr angle_slop (rad). Default None = TreeCorr default (0.1); 0 = exact xi_- "
                         "projection angles, equal to brute force (~14x slower on a 20 WIDE-bin test only; at the production "
                         "1000 bins it costs the same, 538.5 s vs 540.7 s on a 690k x 690k patch, and the default differs "
                         "from 0 by <= 6e-4 sigma on the full Bin6_Bin6 pair; see runs/dr5_measure/angle_slop/). "
                         "Recorded in manifest.json and in every chunk npz when set.")
    ap.add_argument("--nbins", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="recompute even if the npz exists")
    ap.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True,
                    help="skip chunk-pairs whose bounding caps are separated by > tmax (default: True)")
    ap.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=False,
                    help="memory-map uncompressed bin{k}/*.npy arrays instead of reading bin*.npz (default: False)")
    ap.add_argument("--work", type=Path, default=None,
                    help="override catalogue input directory containing bin*.npz or bin*/ (default $KIDSPIPE_DR5_WORK)")
    ap.add_argument("--mpi", action="store_true",
                    help="distribute chunk-pairs over MPI ranks: `mpirun -n N python scripts/dr5_measure_xi_chunked.py --all "
                         "--mpi --threads T ...`; rank 0 schedules, ranks 1..N-1 compute (N >= 2), each with T threads")
    ap.add_argument("--outroot", default=None, help="output tree (default <runs>/dr5_measure/xi_chunked); for tests")
    args = ap.parse_args()
    if args.outroot:
        OUTROOT = Path(args.outroot)
    from kidspipe import mpi
    if args.mpi and (args.status or args.assemble):
        raise SystemExit("--mpi is for computing chunk-pairs; run --status / --assemble without mpirun")
    if not args.mpi:
        mpi.refuse_unflagged_launch("--mpi")

    if args.status:
        return do_status(args)
    if args.assemble:
        return do_assemble(args)

    todo = todo_pairs(args)
    chunk_list = ([tuple(int(x) for x in args.chunk.split("-"))]
                  if args.chunk else None)
    work = args.work.expanduser() if args.work is not None else None

    OUTROOT.mkdir(parents=True, exist_ok=True)
    if args.mpi:
        return run_mpi(args, todo, chunk_list)
    t0 = time.time()
    cache: dict = {}
    edges_cache: dict[int, np.ndarray] = {}
    zero_pairs_map: dict[str, list[tuple[int, int]]] = {}
    patched_loader = (load_pair_cats is not _DEFAULT_LOAD_PAIR_CATS)
    n_done = n_skip = 0
    for (i, j) in todo:
        b1, b2 = i + 1, j + 1
        same = (b1 == b2)
        pname = f"Bin{b1}_Bin{b2}"
        d = OUTROOT / "chunks" / pname
        d.mkdir(parents=True, exist_ok=True)
        _evict_other_bins(cache, (b1, b2))

        if patched_loader:
            cats1, cats2 = load_pair_cats(b1, b2, args.nchunks, cache)
            _, zero_all, cps = _partition_pair_work(
                b1, b2, args, chunk_list, cats1=cats1, cats2=cats2,
            )
        else:
            _, zero_all, cps = _partition_pair_work(
                b1, b2, args, chunk_list, work=work, edges_cache=edges_cache,
            )
        zero_pairs_map[pname] = zero_all

        work_cps = []
        for (a, b) in cps:
            f = d / f"c{a:02d}_{b:02d}.npz"
            if f.exists() and not args.force:
                n_skip += 1
            else:
                work_cps.append((a, b))
        if not work_cps:
            continue

        work_cps.sort()
        uses: dict[tuple[int, int], int] = {}
        for (a, b) in work_cps:
            uses[(b1, a)] = uses.get((b1, a), 0) + 1
            if not (same and a == b):
                uses[(b2, b)] = uses.get((b2, b), 0) + 1

        for (a, b) in work_cps:
            if not patched_loader:
                need = {b1: {a, b}} if same else {b1: {a}, b2: {b}}
                cats1, cats2 = load_pair_cats(
                    b1, b2, args.nchunks, cache,
                    mmap=args.mmap, work=work, needed=need, edges_cache=edges_cache,
                )
            f = d / f"c{a:02d}_{b:02d}.npz"
            t1 = time.time()
            arr = compute_chunk(cats1, cats2, a, b, same, args)
            write_chunk_atomic(f, arr)
            n_done += 1
            print(f"[{time.time()-t0:8.0f}s] {pname} c{a:02d}_{b:02d}: "
                  f"{time.time()-t1:6.1f}s npairs={arr['npairs'].sum():.3e} "
                  f"w={arr['weight'].sum():.3e}", flush=True)
            del arr
            if not patched_loader:
                _release_chunk(cache, uses, b1, a)
                if not (same and a == b):
                    _release_chunk(cache, uses, b2, b)

    write_manifest(args, t0, n_done, n_skip, zero_pairs_map=zero_pairs_map)
    print(f"\n{n_done} chunk-pairs written, {n_skip} already present, {time.time()-t0:.0f}s",
          flush=True)


def do_status(args) -> None:
    prune = bool(getattr(args, "prune", True))
    caps_cache = load_caps_cache(OUTROOT / "chunk_caps.json", args.nchunks) if prune else {}
    print(f"{'pair':>12} {'done':>7} {'active':>7} {'skip':>7} {'total':>7} {'%':>6}")
    grand_done = grand_active = grand_zero = grand_tot = 0
    for (i, j) in PAIRS:
        b1, b2 = i + 1, j + 1
        same = (b1 == b2)
        pname = f"Bin{b1}_Bin{b2}"
        d = OUTROOT / "chunks" / pname
        tot = len(chunk_pairs(args.nchunks, same))
        done = len(list(d.glob("c*.npz"))) if d.exists() else 0
        zero_pairs = read_manifest_zero_pairs(OUTROOT, pname, args.nchunks)
        if zero_pairs is None and prune and b1 in caps_cache and b2 in caps_cache:
            c1 = caps_cache[b1].get("caps", [])
            c2 = caps_cache[b2].get("caps", [])
            if len(c1) == args.nchunks and len(c2) == args.nchunks:
                _, zero_pairs = partition_chunk_pairs(c1, c2, same)
        n_zero = len(zero_pairs) if zero_pairs is not None else 0
        n_active = tot - n_zero
        pct = 100.0 * done / max(1, n_active)
        grand_done += done
        grand_active += n_active
        grand_zero += n_zero
        grand_tot += tot
        print(f"{pname:<12} {done:7d} {n_active:7d} {n_zero:7d} {tot:7d} {pct:6.1f}")
    grand_pct = 100.0 * grand_done / max(1, grand_active)
    print(f"{'TOTAL':>12} {grand_done:7d} {grand_active:7d} {grand_zero:7d} {grand_tot:7d} {grand_pct:6.1f}")
    # Cost model, CALIBRATED on measurement.
    # Measured (chunked_smoke.log): Bin1_Bin1 chunk 0-0 = 1.300e11 pairs in 538.8 s = 2.41e8 pairs/s.
    # Measured (independent 10% run): full Bin1_Bin1 = 2.4183e12 ORDERED pairs inside 2-300 arcmin.
    # TreeCorr counts ordered pairs, so an auto pair has n(n-1) of them (NOT n^2/2 -- dividing by
    # n^2/2 and then applying the fraction to a sum of ordered counts double-counts by 2x; an
    # earlier version of this function made exactly that mistake and inflated the estimate to 85 h).
    n1 = NLETTER[1]
    frac_in = 2.4183e12 / (n1 * (n1 - 1))
    total_ordered = (sum(n * (n - 1) for n in NLETTER.values())
                     + sum(NLETTER[i + 1] * NLETTER[j + 1] for (i, j) in PAIRS if i != j))
    inside = frac_in * total_ordered
    print(f"\ntotal ORDERED pairs over 21 pairs      : {total_ordered:.3e}")
    print(f"calibrated fraction inside 300 arcmin  : {frac_in:.5f} "
          f"(= 2.4183e12 / n1(n1-1), measured)")
    print(f"pairs actually binned                  : {inside:.3e}")
    print(f"at the measured 2.41e8 pairs/s         : {inside/2.41e8/3600:.0f} h total")
    print(f"per chunk-pair (nchunks={args.nchunks}): {inside/grand_tot/2.41e8:.0f} s mean,")
    print("                                         but the distribution is very uneven:")
    print("                                         near-diagonal chunk-pairs ~540 s,")
    print("                                         distant chunk-pairs seconds.")


def do_assemble(args) -> None:
    outdir = OUTROOT / "assembled" / "ns"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"assembling with nchunks={args.nchunks} -> {outdir}")
    zero_counts: dict[str, int] = {}
    for (i, j) in PAIRS:
        b1, b2 = i + 1, j + 1
        same = (b1 == b2)
        pname = f"Bin{b1}_Bin{b2}"
        d = OUTROOT / "chunks" / pname
        all_cps = chunk_pairs(args.nchunks, same)
        zero_pairs = read_manifest_zero_pairs(OUTROOT, pname, args.nchunks)
        expected_active = sorted(set(all_cps) - set(zero_pairs or []))
        if zero_pairs:
            zero_counts[pname] = len(zero_pairs)

        present: dict[tuple[int, int], Path] = {}
        if d.exists():
            for f in sorted(d.glob("c*.npz")):
                m = _CHUNK_STEM_RE.fullmatch(f.stem)
                if m:
                    present[(int(m.group(1)), int(m.group(2)))] = f
        if not present and zero_pairs is None:
            if d.exists():
                print(f"{pname}: INCOMPLETE 0/{len(expected_active)} chunk-pairs, skipping")
            else:
                print(f"{pname}: no chunk directory, skipping")
            continue
        missing_active = [ab for ab in expected_active if ab not in present]
        if missing_active:
            print(f"{pname}: INCOMPLETE {len(present)}/{len(expected_active)} active chunk-pairs, skipping")
            continue

        if not expected_active:
            nbins = getattr(args, "nbins", None)
            if nbins is None and (OUTROOT / "manifest.json").is_file():
                try:
                    nbins = int(json.loads((OUTROOT / "manifest.json").read_text()).get("nbins", official.N_THETA_FINE))
                except Exception:
                    nbins = official.N_THETA_FINE
            if nbins is None:
                nbins = official.N_THETA_FINE
            rnom = np.asarray(official.fine_bin_centres(n=int(nbins)), float)
            zeros = np.zeros_like(rnom)
            nans = np.full_like(rnom, np.nan)
            out = {
                "r_nom": rnom, "meanr": rnom.copy(),
                "xip": nans.copy(), "xim": nans.copy(),
                "xip_im": zeros.copy() if same else nans.copy(),
                "xim_im": nans.copy(),
                "varxi": nans.copy(),
                "npairs": zeros.copy(), "weight": zeros.copy(),
            }
            np.savez(outdir / f"{pname}.npz", **out)
            print(f"{pname}: assembled from 0 active ({len(zero_pairs or [])} pruned) chunk-pairs  npairs=0.0000e+00 w=0.0000e+00")
            continue

        chunks_data: list[dict] = []
        for ab in expected_active:
            with np.load(present[ab]) as z:
                chunks_data.append({k: z[k] for k in z.files})

        slops = {chunk_slops_dict(c) for c in chunks_data}
        if len(slops) > 1:
            raise RuntimeError(f"{pname}: chunk files mix different (bin_slop, angle_slop) settings "
                               f"{sorted(slops, key=str)}; refusing to assemble")

        acc = None
        rnom = None
        meanr = None
        first_meanr = None
        last_chunk = chunks_data[-1]
        for a in chunks_data:
            fold = float(a["fold"])
            w = np.asarray(a["weight"], float)
            if acc is None:
                acc = {k: np.zeros_like(w)
                       for k in ("npairs", "weight", "wxip", "wxim", "wxip_im", "wxim_im",
                                 "varxip_num")}
                rnom = np.asarray(a["r_nom"], float)
                first_meanr = np.asarray(a["meanr"], float)
            if meanr is None and float(w.sum()) > 0.0:
                meanr = np.asarray(a["meanr"], float)
            acc["npairs"] += fold * np.asarray(a["npairs"], float)
            acc["weight"] += fold * w
            acc["wxip"] += fold * w * np.asarray(a["xip"], float)
            acc["wxim"] += fold * w * np.asarray(a["xim"], float)
            acc["wxip_im"] += fold * w * np.asarray(a["xip_im"], float)
            acc["wxim_im"] += fold * w * np.asarray(a["xim_im"], float)
            acc["varxip_num"] += fold * np.asarray(a["varxi"], float) * w ** 2
        if meanr is None:
            meanr = first_meanr
        wsafe = np.where(acc["weight"] > 0, acc["weight"], np.nan)
        out = {
            "r_nom": rnom, "meanr": meanr,
            "xip": acc["wxip"] / wsafe, "xim": acc["wxim"] / wsafe,
            # auto pairs: Im xi_+ cancels exactly over ordered pairs, but folding (2x for a<b) does not cancel it
            "xip_im": np.zeros_like(rnom) if same else acc["wxip_im"] / wsafe,
            "xim_im": acc["wxim_im"] / wsafe,
            "varxi": acc["varxip_num"] / wsafe ** 2,
            "npairs": acc["npairs"], "weight": acc["weight"],
        }
        if "angle_slop" in last_chunk:      # last file read; all files agree (checked above)
            out["angle_slop"] = np.float64(last_chunk["angle_slop"])
        np.savez(outdir / f"{pname}.npz", **out)
        zero_msg = f" (+{len(zero_pairs)} pruned)" if zero_pairs else ""
        print(f"{pname}: assembled from {len(expected_active)} chunk-pairs{zero_msg}  "
              f"npairs={out['npairs'].sum():.4e} w={out['weight'].sum():.4e}")
    idx_tmp = outdir / ".index.json.part"
    idx_tmp.write_text(json.dumps({
        "nchunks": args.nchunks,
        "source": str(OUTROOT / "chunks"),
        "analytically_zero_counts": zero_counts,
        "note": "assembled by fold-weighted summation; exact to 1e-9 (chunk_verify.log)",
    }, indent=2))
    os.replace(idx_tmp, outdir / "index.json")


def chunk_slops_dict(d: dict) -> tuple[float, float | None]:
    """(bin_slop, angle_slop) of a loaded chunk dictionary; angle_slop None = TreeCorr default."""
    b = float(d["bin_slop"]) if "bin_slop" in d else 0.0
    a = float(d["angle_slop"]) if "angle_slop" in d else None
    return b, a


if __name__ == "__main__":
    main()
