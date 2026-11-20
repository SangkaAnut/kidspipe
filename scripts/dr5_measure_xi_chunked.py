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
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidspipe import paths  # noqa: E402
from kidspipe.measure.official import PAIRS  # noqa: E402

#: output tree: <runs>/dr5_measure/xi_chunked with <runs> = $KIDSPIPE_RUNS (default <root>/runs); the per-bin input extract is
#: read from $KIDSPIPE_DR5_WORK (kidspipe.measure.xi.load_bin).  --threads defaults to min(16, os.cpu_count()).
OUTROOT = paths.runs_dir() / "dr5_measure" / "xi_chunked"
NLETTER = {1: 7416371, 2: 7359911, 3: 6799681, 4: 6880432, 5: 6477538, 6: 5960461}


# ----------------------------------------------------------------------------- chunking
def chunk_masks(ra: np.ndarray, nchunks: int) -> list[np.ndarray]:
    """Equal-count RA strips. Deterministic given the catalogue, so chunk ids are reproducible.

    Equal-count (not equal-area) strips: this guarantees balanced work per chunk, which is what
    matters for load distribution. Pairs between distant strips are computed all the same, so the
    decomposition is exact regardless of how the sky is cut.
    """
    ra = np.asarray(ra, np.float64)
    edges = np.quantile(ra, np.linspace(0.0, 1.0, nchunks + 1))
    masks = []
    for k in range(nchunks):
        lo, hi = edges[k], edges[k + 1]
        m = (ra >= lo) & (ra < hi) if k < nchunks - 1 else (ra >= lo) & (ra <= hi)
        masks.append(m)
    return masks


def chunk_pairs(nchunks: int, same_bin: bool) -> list[tuple[int, int]]:
    """Chunk-pair list. For an auto pair the fold factor 2 is applied at assembly time."""
    if same_bin:
        return [(a, b) for a in range(nchunks) for b in range(a, nchunks)]
    return [(a, b) for a in range(nchunks) for b in range(nchunks)]


def fold_factor(a: int, b: int, same_bin: bool) -> float:
    if same_bin and a != b:
        return 2.0
    return 1.0


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
def load_pair_cats(b1: int, b2: int, nchunks: int, cache: dict):
    """Load bin b1/b2 once, split into chunk Catalog objects, memoised per process."""
    from kidspipe.measure.xi import load_bin, make_treecorr_cat

    for b in ({b1, b2}):
        if b in cache:
            continue
        rec = load_bin(b)
        masks = chunk_masks(rec["ra"], nchunks)
        cats = []
        for k, m in enumerate(masks):
            sub = {kk: rec[kk][m] for kk in ("ra", "dec", "e1", "e2", "w")}
            cats.append(make_treecorr_cat(sub))
        cache[b] = cats
        print(f"  loaded bin{b}: {rec['ra'].size:,} galaxies in {nchunks} chunks "
              f"({cats[0].nobj:,} .. {cats[-1].nobj:,} per chunk)", flush=True)
        del rec
    return cache[b1], cache[b2]


def write_chunk_atomic(f: Path, arr: dict) -> None:
    """Write a chunk npz under a temporary name, then rename: an interrupted run never leaves a partial c*.npz
    (the temporary name '.cAA_BB.npz.part' does not match the 'c*.npz' glob of --assemble/--status)."""
    tmp = f.with_name("." + f.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arr)
    tmp.replace(f)


def compute_chunk(cats1, cats2, a: int, b: int, same: bool, args) -> dict:
    from kidspipe.measure.xi import run_pair, corr_to_arrays
    gg = run_pair(cats1[a], cats2[b], nbins=args.nbins, bin_slop=args.bin_slop, num_threads=args.threads,
                  angle_slop=args.angle_slop)
    arr = corr_to_arrays(gg)
    arr["fold"] = np.float64(fold_factor(a, b, same))
    arr.update(slop_record(args.bin_slop, args.angle_slop))
    return arr


def todo_pairs(args) -> list:
    todo = PAIRS
    if args.pairs:
        want = {tuple(int(x) - 1 for x in p.split("-")) for p in args.pairs.split(",")}
        todo = [(i, j) for i, j in PAIRS if (i, j) in want]
    return todo


def write_manifest(args, t0, n_done, n_skip, extra=None) -> None:
    (OUTROOT / "manifest.json").write_text(json.dumps({
        "nchunks": args.nchunks, "nbins": args.nbins, "bin_slop": args.bin_slop,
        "angle_slop": args.angle_slop,   # null = TreeCorr default (0.1 rad in TreeCorr 5)
        "threads": args.threads,
        "binning": "Log, 1000 bins, 2-300 arcmin, sep_units='arcmin'",
        "chunking": "equal-count RA strips, quantile edges",
        "fold_rule": "auto pairs: 2 * sum_{a<b} process(c_a,c_b); cross pairs: no fold",
        "exactness_check": "chunk_diagnose.log, chunk_verify.log (ratio 1.000000000)",
        "partial_total_seconds": time.time() - t0,
        "chunk_pairs_written": n_done, "chunk_pairs_skipped": n_skip, **(extra or {}),
    }, indent=2))


# ----------------------------------------------------------------------------- MPI workers
_WCACHE: dict = {}


def _mpi_task(job):
    """One chunk-pair on a worker rank.  Keeps only the current pair's bins in memory (an earlier single-process run
    that cached every bin was OOM-killed)."""
    b1, b2, a, b, args = job
    for k in [k for k in _WCACHE if k not in (b1, b2)]:
        del _WCACHE[k]
    cats1, cats2 = load_pair_cats(b1, b2, args.nchunks, _WCACHE)
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
    """Rank 0 hands out the chunk-pairs of one tomographic pair at a time (heaviest first) to the worker ranks."""
    from kidspipe import mpi
    nworkers = mpi.require_workers(1)
    with mpi.executor(_mpi_init, (str(OUTROOT),), threads=args.threads) as ex:
        if ex is None:
            return
        print(f"MPI: {nworkers} worker ranks x {args.threads} threads; rank 0 schedules. Peak memory per worker is about "
              f"2.5 GB (auto pair) to 5 GB (cross pair): keep workers x that below the free memory.", flush=True)
        pool = mpi.Pool(ex, nworkers)
        t0 = time.time()
        n_done = n_skip = 0
        for (i, j) in todo:
            b1, b2 = i + 1, j + 1
            same = b1 == b2
            d = OUTROOT / "chunks" / f"Bin{b1}_Bin{b2}"
            d.mkdir(parents=True, exist_ok=True)
            cps = chunk_pairs(args.nchunks, same) if chunk_list is None else chunk_list
            jobs = []
            for (a, b) in cps:
                if (d / f"c{a:02d}_{b:02d}.npz").exists() and not args.force:
                    n_skip += 1
                    continue
                jobs.append((b1, b2, a, b, args))
            jobs.sort(key=lambda jb: abs(jb[2] - jb[3]))          # near-diagonal chunk-pairs carry most pairs: start them first
            for (_, _, a, b, dt, npairs, w, rk) in pool.imap_unordered(_mpi_task, jobs):
                n_done += 1
                print(f"[{time.time()-t0:8.0f}s] Bin{b1}_Bin{b2} c{a:02d}_{b:02d}: {dt:6.1f}s npairs={npairs:.3e} "
                      f"w={w:.3e} (rank {rk})", flush=True)
        write_manifest(args, t0, n_done, n_skip, {"mpi_workers": nworkers, "threads_per_worker": args.threads})
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

    OUTROOT.mkdir(parents=True, exist_ok=True)
    if args.mpi:
        return run_mpi(args, todo, chunk_list)
    t0 = time.time()
    cache: dict = {}
    n_done = n_skip = 0
    for (i, j) in todo:
        b1, b2 = i + 1, j + 1
        same = (b1 == b2)
        d = OUTROOT / "chunks" / f"Bin{b1}_Bin{b2}"
        d.mkdir(parents=True, exist_ok=True)
        cps = chunk_pairs(args.nchunks, same) if chunk_list is None else chunk_list
        cats1, cats2 = load_pair_cats(b1, b2, args.nchunks, cache)
        for (a, b) in cps:
            f = d / f"c{a:02d}_{b:02d}.npz"
            if f.exists() and not args.force:
                n_skip += 1
                continue
            t1 = time.time()
            arr = compute_chunk(cats1, cats2, a, b, same, args)
            write_chunk_atomic(f, arr)
            n_done += 1
            print(f"[{time.time()-t0:8.0f}s] Bin{b1}_Bin{b2} c{a:02d}_{b:02d}: "
                  f"{time.time()-t1:6.1f}s npairs={arr['npairs'].sum():.3e} "
                  f"w={arr['weight'].sum():.3e}", flush=True)
            del arr

    write_manifest(args, t0, n_done, n_skip)
    print(f"\n{n_done} chunk-pairs written, {n_skip} already present, {time.time()-t0:.0f}s",
          flush=True)


def do_status(args) -> None:
    print(f"{'pair':>12} {'done':>7} {'total':>7} {'%':>6}")
    grand_done = grand_tot = 0
    for (i, j) in PAIRS:
        b1, b2 = i + 1, j + 1
        same = (b1 == b2)
        d = OUTROOT / "chunks" / f"Bin{b1}_Bin{b2}"
        tot = len(chunk_pairs(args.nchunks, same))
        done = len(list(d.glob("c*.npz"))) if d.exists() else 0
        grand_done += done
        grand_tot += tot
        print(f"Bin{b1}_Bin{b2:<6} {done:7d} {tot:7d} {100*done/tot:6.1f}")
    print(f"{'TOTAL':>12} {grand_done:7d} {grand_tot:7d} {100*grand_done/grand_tot:6.1f}")
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
    for (i, j) in PAIRS:
        b1, b2 = i + 1, j + 1
        same = (b1 == b2)
        d = OUTROOT / "chunks" / f"Bin{b1}_Bin{b2}"
        if not d.exists():
            print(f"Bin{b1}_Bin{b2}: no chunk directory, skipping")
            continue
        files = sorted(d.glob("c*.npz"))
        expect = len(chunk_pairs(args.nchunks, same))
        if len(files) < expect:
            print(f"Bin{b1}_Bin{b2}: INCOMPLETE {len(files)}/{expect} chunk-pairs, skipping")
            continue
        acc = None
        slops = {chunk_slops(np.load(f)) for f in files}
        if len(slops) > 1:
            raise RuntimeError(f"Bin{b1}_Bin{b2}: chunk files mix different (bin_slop, angle_slop) settings "
                               f"{sorted(slops, key=str)}; refusing to assemble")
        for f in files:
            a = np.load(f)
            fold = float(a["fold"])
            w = np.asarray(a["weight"], float)
            if acc is None:
                acc = {k: np.zeros_like(w)
                       for k in ("npairs", "weight", "wxip", "wxim", "wxip_im", "wxim_im",
                                 "varxip_num")}
                rnom = np.asarray(a["r_nom"], float)
                meanr = np.asarray(a["meanr"], float)
            acc["npairs"] += fold * np.asarray(a["npairs"], float)
            acc["weight"] += fold * w
            acc["wxip"] += fold * w * np.asarray(a["xip"], float)
            acc["wxim"] += fold * w * np.asarray(a["xim"], float)
            acc["wxip_im"] += fold * w * np.asarray(a["xip_im"], float)
            acc["wxim_im"] += fold * w * np.asarray(a["xim_im"], float)
            acc["varxip_num"] += fold * np.asarray(a["varxi"], float) * w ** 2
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
        if "angle_slop" in a.files:      # last file read; all files agree (checked above)
            out["angle_slop"] = np.float64(a["angle_slop"])
        np.savez(outdir / f"Bin{b1}_Bin{b2}.npz", **out)
        print(f"Bin{b1}_Bin{b2}: assembled from {len(files)} chunk-pairs  "
              f"npairs={out['npairs'].sum():.4e} w={out['weight'].sum():.4e}")
    (outdir / "index.json").write_text(json.dumps({
        "nchunks": args.nchunks,
        "source": str(OUTROOT / "chunks"),
        "note": "assembled by fold-weighted summation; exact to 1e-9 (chunk_verify.log)",
    }, indent=2))


if __name__ == "__main__":
    main()
