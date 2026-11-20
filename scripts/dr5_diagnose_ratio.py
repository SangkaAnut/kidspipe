"""Diagnose the 1.417x pair-count excess: is it duplicates, a broken column, or upstream thinning?

Tests:
 1. FLUX_RADIUS / other extra columns: are they populated at all (sentinel detection)?
 2. Exact duplicate sky positions (would inflate pairs at zero separation -> a spike in the
    smallest bins; the 20-bin probe showed a *uniform* ratio, so this should come out clean).
 3. Whether the excess is scale-independent: ratio of our npairs to the official in the
    smallest vs largest separations from the saved 20-bin probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kidspipe.measure.official import PAIRS, load_official_xi

EXTRA = Path.home() / "kids_dr5_work" / "extra"
MAIN = Path.home() / "kids_dr5_work"


def main() -> None:
    print("=== 1. column population (tomographic bin 1) ===")
    d = dict(np.load(EXTRA / "bin0.npz"))
    for c in d:
        a = d[c]
        print(f"  {c:26s} dtype={a.dtype} min={np.nanmin(a):>10.4g} max={np.nanmax(a):>10.4g} "
              f"n_unique={np.unique(a).size:>8d} nan={int(np.sum(~np.isfinite(a))):>8d} "
              f"frac_zero={np.mean(a == 0):.4f}")

    print("\n=== 2. duplicate sky positions (bin 1, full) ===")
    m = dict(np.load(MAIN / "bin0.npz"))
    key = (np.round(m["ra"].astype(np.float64), 5) * 1e6).astype(np.int64) * 10_000_000 + \
          np.round(m["dec"].astype(np.float64), 5).astype(np.int64)
    _, cnt = np.unique(key, return_counts=True)
    n = key.size
    print(f"  n={n:,}  unique positions={cnt.size:,}  duplicates={int((cnt > 1).sum()):,} "
          f"({100*(cnt > 1).sum()/n:.4f}% of positions, {100*(n-cnt.size)/n:.4f}% of rows)")
    print(f"  max multiplicity={cnt.max()}  rows in multi-occupancy groups={int(cnt[cnt>1].sum()):,}")

    print("\n=== 3. is the pair excess scale-independent? (20-bin probe, bin2 vs official Bin2_2) ===")
    off = load_official_xi()
    pi = PAIRS.index((1, 1))          # Bin2_Bin2, matching the probe catalogue
    probe = np.load(MAIN / "probe_sub5.npz")
    o = off["npairs"][pi]
    e = np.exp(np.linspace(np.log(2.0), np.log(300.0), 21))
    rb = np.digitize(off["r_nom"][pi], e) - 1
    onp = np.array([np.sum(o[rb == k]) for k in range(20)])
    pr = probe["npairs"] / onp          # 1/5 subsample -> ratio x25 of this
    print("  separation-dependent npairs ratio (scaled to full sample):")
    print("   " + " ".join(f"{x*25:5.3f}" for x in pr[:10]))
    print("   " + " ".join(f"{x*25:5.3f}" for x in pr[10:]))
    print(f"  mean={pr.mean()*25:.4f}  std={pr.std()*25:.4f}  "
          f"small-sep mean={pr[:5].mean()*25:.4f}  large-sep mean={pr[-5:].mean()*25:.4f}")


if __name__ == "__main__":
    main()
