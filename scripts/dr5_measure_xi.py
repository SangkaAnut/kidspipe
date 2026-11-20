"""Step A: measure xi_+/xi_-/xi_x for all 21 tomographic pairs (1<=i<=j<=6) on the official
binning (Log, 1000 bins, 2-300 arcmin), plus pair counts and weights.

Resumable: each pair is written to its own npz as soon as it finishes, so an interrupted run
can be restarted without recomputing finished pairs.

Three patch modes, all into separate output trees:
  ns   : North and South galaxies combined (the official "NScomb" product)
  north: North only
  south: South only
Pairs across patches contribute nothing, so a weighted combination of the north and south
runs equals the ns run up to weighting -- that is checked in scripts/dr5_compare.py.

Usage:
  PYTHONPATH=src python3 scripts/dr5_measure_xi.py --mode ns --bin-slop 0.05 --threads 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidspipe.measure.official import PAIRS
from kidspipe.measure.xi import load_bin, make_treecorr_cat, run_pair, corr_to_arrays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dr5_measure" / "xi"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ns", choices=["ns", "north", "south"])
    ap.add_argument("--bin-slop", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--nbins", type=int, default=1000)
    ap.add_argument("--pairs", default="", help="comma list i-j (1-based) to restrict, e.g. 1-1,1-2")
    args = ap.parse_args()

    outdir = OUT / args.mode
    outdir.mkdir(parents=True, exist_ok=True)
    todo = PAIRS
    if args.pairs:
        want = {tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")}
        todo = [(i, j) for i, j in PAIRS if (i + 1, j + 1) in want]

    t0 = time.time()
    cats = {}
    for b in range(1, 7):
        if not any(b in (i + 1, j + 1) for i, j in todo):
            continue
        rec = load_bin(b)
        if args.mode != "ns":
            keep = rec["north"] if args.mode == "north" else ~rec["north"]
            rec = {k: (rec[k][keep] if k != "north" else rec[k][keep]) for k in rec}
        cats[b] = make_treecorr_cat(rec)
        print(f"[{time.time()-t0:7.0f}s] bin{b}: {cats[b].nobj:,} objects ({args.mode})", flush=True)
        del rec

    index = []
    for (i, j) in todo:
        b1, b2 = i + 1, j + 1
        f = outdir / f"Bin{b1}_Bin{b2}.npz"
        rec_meta = {"bin1": b1, "bin2": b2, "mode": args.mode, "bin_slop": args.bin_slop,
                    "nbins": args.nbins, "threads": args.threads}
        if f.exists():
            print(f"[{time.time()-t0:7.0f}s] Bin{b1}_Bin{b2}: already done, skipping", flush=True)
            index.append({**rec_meta, "file": f.name, "seconds": None, "cached": True})
            continue
        t1 = time.time()
        gg = run_pair(cats[b1], cats[b2], nbins=args.nbins, bin_slop=args.bin_slop,
                      num_threads=args.threads)
        a = corr_to_arrays(gg)
        dt = time.time() - t1
        np.savez(f, **a)
        print(f"[{time.time()-t0:7.0f}s] Bin{b1}_Bin{b2}: {dt:7.1f}s "
              f"npairs={a['npairs'].sum():.4e} xip[500]={a['xip'][500]:+.4e} "
              f"xim[500]={a['xim'][500]:+.4e}", flush=True)
        index.append({**rec_meta, "file": f.name, "seconds": dt, "cached": False,
                      "npairs_sum": float(a["npairs"].sum()),
                      "weight_sum": float(a["weight"].sum()),
                      "npairs_nonzero_bins": int((a["npairs"] > 0).sum())})
        del gg, a

    (outdir / "index.json").write_text(json.dumps({
        "mode": args.mode, "bin_slop": args.bin_slop, "nbins": args.nbins,
        "threads": args.threads, "pairs": index, "total_seconds": time.time() - t0,
        "binning": "Log, 1000 bins, 2-300 arcmin, sep_units='arcmin'",
    }, indent=2))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
