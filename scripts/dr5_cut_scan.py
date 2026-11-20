"""Which selection cut explains the 1.417x pair-count excess?

For an auto-pair count npairs ~ (f N)^2/2, a measured npairs ratio of 1.417 means the official
sample keeps f = 1/sqrt(1.417) = 0.840 of our galaxies, uniformly in every bin.  This script
prints, for each candidate cut on the extra columns, the surviving fraction per tomographic
bin, so the cut that removes ~16% uniformly can be identified.

Also reports the pair-count-implied f and the n_eff/NGAL consistency for each candidate.
Pure local computation on ~/kids_dr5_work/extra (no catalogue access).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kidspipe.measure.official import load_official_cosebis

EXTRA = Path.home() / "kids_dr5_work" / "extra"
RATIO = 1.41698
F_TARGET = 1.0 / np.sqrt(RATIO)


def main() -> None:
    hdr = load_official_cosebis()["header"]
    data = {b: dict(np.load(EXTRA / f"bin{b-1}.npz")) for b in range(1, 7)}
    n0 = {b: data[b]["weight"].size for b in data}
    print(f"npairs ratio {RATIO:.5f} -> implied surviving fraction f = {F_TARGET:.4f}")
    print(f"target: each bin keeps {100*F_TARGET:.1f}% of {sum(n0.values()):,} galaxies\n")

    def frac(name, mask_fn):
        line = []
        for b in range(1, 7):
            m = mask_fn(data[b])
            line.append(m.mean())
        line = np.array(line)
        print(f"{name:38s} " + " ".join(f"{x:6.3f}" for x in line) +
              f"   spread={line.std():.4f}  dev_from_f={np.abs(line-F_TARGET).max():.3f}")
        return line

    print(f"{'cut':38s} " + " ".join(f"bin{b:<2d}" .rjust(6) for b in range(1, 7)))
    frac("all rows (reference)", lambda d: np.ones(d["weight"].size, bool))
    frac("SG2DPHOT != 1 (drop stars)", lambda d: d["SG2DPHOT"] != 1)
    for lo, hi in [(0.0, 1.2), (0.5, 1.2), (0.0, 1.0), (0.3, 1.2)]:
        frac(f"FLUX_RADIUS in ({lo},{hi})", lambda d, lo=lo, hi=hi: (d["FLUX_RADIUS"] > lo) & (d["FLUX_RADIUS"] < hi))
    for thr in [5.0, 10.0, 20.0]:
        frac(f"model_SNratio > {thr}", lambda d, t=thr: d["model_SNratio"] > t)
    frac("n_exposures_used >= 5", lambda d: d["n_exposures_used"] >= 5)
    frac("n_exposures_used == 5", lambda d: d["n_exposures_used"] == 5)
    for zl in [0.1, 0.2, 0.3]:
        frac(f"Z_B > {zl}", lambda d, z=zl: d["Z_B"] > z)
    frac("gold_weight_only > 0", lambda d: d["gold_weight_only"] > 0)
    frac("shear_weight_only > 0", lambda d: d["shear_weight_only"] > 0)
    for q in [0.1, 0.16, 0.2]:
        frac(f"weight > {100*q:.0f}th pctile", lambda d, q=q: d["weight"] > np.quantile(d["weight"], q))
    for mg in [24.0, 24.5, 25.0]:
        frac(f"MAG_AUTO < {mg}", lambda d, m=mg: d["MAG_AUTO"] < m)

    # n_eff-based cross-check for the reference and the star-rejected sample
    print("\nn_eff / NGAL implied area (deg2), reference vs star-rejected:")
    for tag, fn in [("all", lambda d: np.ones(d["weight"].size, bool)),
                    ("no stars", lambda d: d["SG2DPHOT"] != 1)]:
        areas = []
        for b in range(1, 7):
            d = data[b]
            m = fn(d)
            w = d["weight"][m].astype(np.float64)
            neff = w.sum() ** 2 / (w ** 2).sum()
            areas.append(neff / hdr[f"NGAL_{b}"] / 3600.0)
        print(f"  {tag:10s} " + " ".join(f"{a:8.2f}" for a in areas))

    out = {"npairs_ratio": RATIO, "implied_f": float(F_TARGET),
           "note": "see printed table; a cut matching f with small bin-to-bin spread is the "
                   "candidate for the official selection"}
    (Path(__file__).resolve().parents[1] / "runs" / "dr5_measure" / "cut_scan.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
