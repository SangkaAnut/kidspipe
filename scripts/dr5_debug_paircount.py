# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Debug the pair counter against TreeCorr on a small, controlled catalogue.

The A4 run showed npairs disagreeing by up to 50% on an auto-pair and |xi_x| ~ 0.26, which is
not a convention difference -- a broken counter or a binning mismatch.  This isolates it:

 1. Foreground-only catalogue (no wrap at RA=0), so chord/angle binning is unambiguous.
 2. Compare npairs per bin, not just the max relative deviation.
 3. Check the chord-distance -> separation-angle conversion against TreeCorr's meanr.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidspipe.measure.paircount import PairCounter
from kidspipe.measure.xi import make_treecorr_cat, run_pair, corr_to_arrays

NBINS = 1000


def main() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    # compact field, no RA wrap, small enough that dec-plane approx is good
    ra = rng.uniform(10.0, 12.0, n)
    dec = rng.uniform(-1.0, 1.0, n)
    g1 = rng.normal(0, 0.3, n)
    g2 = rng.normal(0, 0.3, n)
    w = rng.uniform(0.5, 1.5, n)

    pc = PairCounter(ra, dec, g1, g2, w, min_sep=2.0, max_sep=300.0, nbins=NBINS)
    xip, xim, xix, npairs, wsum = pc.run()
    print(f"paircount: total npairs={npairs.sum():.1f}  nonzero bins={int((npairs>0).sum())}")

    cat = make_treecorr_cat(dict(ra=ra.astype(np.float32), dec=dec.astype(np.float32),
                                 e1=g1, e2=g2, w=w))
    gg = run_pair(cat, cat, nbins=NBINS, bin_slop=0.0)
    a = corr_to_arrays(gg)
    print(f"treecorr : total npairs={a['npairs'].sum():.1f}  nonzero bins={int((a['npairs']>0).sum())}")
    print(f"ratio total = {npairs.sum()/a['npairs'].sum():.6f}   (expect ~0.5 if TreeCorr "
          f"counts ordered pairs and I count unordered)")

    m = (npairs > 0) & (a["npairs"] > 0)
    print(f"\ncommon bins: {m.sum()}")
    print("first 5 common bins:")
    for k in np.flatnonzero(m)[:5]:
        print(f"  bin {k:4d} r_nom={a['r_nom'][k]:8.3f} npairs mine={npairs[k]:10.1f} "
              f"tc={a['npairs'][k]:10.1f} ratio={npairs[k]/a['npairs'][k]:.4f}")
    print("\nmid bins:")
    for k in np.flatnonzero(m)[len(np.flatnonzero(m)) // 2: len(np.flatnonzero(m)) // 2 + 5]:
        print(f"  bin {k:4d} r_nom={a['r_nom'][k]:8.3f} npairs mine={npairs[k]:10.1f} "
              f"tc={a['npairs'][k]:10.1f} ratio={npairs[k]/a['npairs'][k]:.4f}")
    print(f"\nnpairs ratio: median={np.median(npairs[m]/a['npairs'][m]):.6f} "
          f"std={np.std(npairs[m]/a['npairs'][m]):.6f}")
    print(f"xip  corr={np.corrcoef(xip[m], a['xip'][m])[0,1]:.6f} "
          f"maxrel={np.abs(xip[m]/a['xip'][m]-1).max():.3e}")
    print(f"xim  corr={np.corrcoef(xim[m], a['xim'][m])[0,1]:.6f} "
          f"maxrel={np.abs(xim[m]/a['xim'][m]-1).max():.3e}")
    print(f"xi_x mine: max|.|={np.abs(xix[m]).max():.3e}  (should be ~0 for random shears)")
    print(f"xi_x tc  : max|.|={np.abs(a['xim_im'][m]).max():.3e}")


if __name__ == "__main__":
    main()
