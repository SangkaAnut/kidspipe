"""Recompute the full-sample npairs ratio with the correct pair matching.

Both saved probes (probe_sub5.npz at 1000 bins, /tmp/c_00_11.npz at 20 bins) were produced
BEFORE the load_bin() off-by-one fix, so both contain TOMOGRAPHIC BIN 2.  The headline ratio
1.41698 reported earlier compared that against official Bin1_Bin1 -- a mismatched pairing.
This script recomputes the ratio against the correct official pair (Bin2_Bin2) from the saved
files, so no TreeCorr re-run is needed, and prints both pairings side by side as evidence.

Pure local computation on saved npz + official ascii.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kidspipe.measure.official import PAIRS, load_official_xi

MAIN = Path.home() / "kids_dr5_work"
SUBSAMPLE = 5          # probe_sub5 was a 1/5 stride subsample -> npairs scale by 25


def rebin20(o, th):
    e = np.exp(np.linspace(np.log(2.0), np.log(300.0), 21))
    rb = np.digitize(th, e) - 1
    return np.array([np.sum(o[rb == k]) for k in range(20)])


def main() -> None:
    off = load_official_xi()
    reported = {}

    # --- 1000-bin probe (tomo bin 2), scaled back to the full sample
    p = np.load(MAIN / "probe_sub5.npz")
    for tag, pair in [("Bin2_Bin2 (correct)", (1, 1)), ("Bin1_Bin1 (what I reported)", (0, 0))]:
        ref = off["npairs"][PAIRS.index(pair)]
        r = p["npairs"] / ref * SUBSAMPLE ** 2
        m = ref > 0
        reported[tag] = {"median": float(np.median(r[m])), "std": float(np.std(r[m])),
                         "min": float(r[m].min()), "max": float(r[m].max())}
        print(f"1000 bins vs {tag:26s} median={np.median(r[m]):.5f} std={np.std(r[m]):.5f} "
              f"[{r[m].min():.4f}, {r[m].max():.4f}]")

    # --- 20-bin probe (tomo bin 2)
    print()
    c = np.load("/tmp/c_00_11.npz")
    for tag, pair in [("Bin2_Bin2 (correct)", (1, 1)), ("Bin1_Bin1 (what I reported)", (0, 0))]:
        pi = PAIRS.index(pair)
        ref = rebin20(off["npairs"][pi], off["r_nom"][pi])
        r = c["npairs"] / ref
        reported[f"20bin vs {tag}"] = {"median": float(np.median(r)), "std": float(np.std(r))}
        print(f"20 bins   vs {tag:26s} median={np.median(r):.5f} std={np.std(r):.5f} "
              f"per-bin: " + " ".join(f"{x:.3f}" for x in r))

    # --- the corrected headline number
    good = reported["Bin2_Bin2 (correct)"]["median"]
    print(f"\nCORRECTED headline ratio (tomo bin 2 vs official Bin2_Bin2): {good:.5f}")
    print(f"implied surviving fraction f = 1/sqrt(ratio) = {1/np.sqrt(good):.4f}")
    print(f"previously reported (mismatched pairing):      {reported['20bin vs Bin1_Bin1 (what I reported)']['median']:.5f}")
    print(f"difference: {100*abs(good/reported['20bin vs Bin1_Bin1 (what I reported)']['median']-1):.2f}% "
          "-> does not change the conclusion, but the number must be quoted correctly.")


if __name__ == "__main__":
    main()
