# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Check A4 done properly: independent pair counter vs TreeCorr on a catalogue with a REAL
shear field, so that xi_+, xi_- are non-zero and a frame error would show up as a wrong
amplitude or a wrong sign rather than as noise.

Two tests:
  1. pair counts: TreeCorr counts ordered pairs for an auto-correlation, this counter counts
     unordered -> expect an exact factor 0.5, with no scatter (this is the actual A4 claim
     about "the same binning").
  2. xi: a correlated shear field built from a smooth function of position (so xi != 0 and is
     smooth in theta), compared bin by bin for all four TreeCorr flip combinations.  The
     statistic is the correlation coefficient and the amplitude ratio, on bins where the
     signal is well above the noise (|xi| > 3 x the noise estimate from bin scatter).

The earlier A4 attempt used random uncorrelated shears, which made xi pure noise and produced
a meaningless comparison (and a spurious large xi_x).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidspipe.measure.paircount import PairCounter
from kidspipe.measure.xi import make_treecorr_cat, run_pair, corr_to_arrays

OUT = Path(__file__).resolve().parents[1] / "runs" / "dr5_measure"
NBINS = 200


def main() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    ra = rng.uniform(10.0, 12.0, n)
    dec = rng.uniform(-1.0, 1.0, n)
    # smooth correlated shear field: a plane wave in position => coherent, non-zero xi
    kx, ky = 2 * np.pi / 0.6, 2 * np.pi / 0.4
    g1 = 0.05 * np.cos(kx * ra + ky * dec)
    g2 = 0.05 * np.sin(kx * ra - ky * dec)
    w = np.ones(n)

    pc = PairCounter(ra, dec, g1, g2, w, min_sep=2.0, max_sep=300.0, nbins=NBINS)
    xip, xim, xix, npairs, wsum = pc.run()
    print(f"paircount: npairs={npairs.sum():.0f}  |xi_x|max={np.abs(xix[npairs>0]).max():.3e}")

    res = {}
    for flip1, flip2 in [(False, False), (False, True), (True, False), (True, True)]:
        cat = make_treecorr_cat(dict(ra=ra.astype(np.float32), dec=dec.astype(np.float32),
                                     e1=g1, e2=g2, w=w), flip_g1=flip1, flip_g2=flip2)
        gg = run_pair(cat, cat, nbins=NBINS, bin_slop=0.0)
        a = corr_to_arrays(gg)
        key = f"flip1={int(flip1)},flip2={int(flip2)}"
        m = (npairs > 0) & (a["npairs"] > 0)
        d = {"npairs_ratio_median": float(np.median(npairs[m] / a["npairs"][m])),
             "npairs_ratio_std": float(np.std(npairs[m] / a["npairs"][m]))}
        for comp in ("xip", "xim"):
            mine, theirs = locals()[comp][m] if False else (xip if comp == "xip" else xim)[m], a[comp][m]
            # signal-to-noise: use the scatter of xi over neighbouring bins as the noise proxy
            noise = np.median(np.abs(np.diff(theirs))) or 1.0
            good = np.abs(theirs) > 3 * noise
            d[f"{comp}_corr_all"] = float(np.corrcoef(mine, theirs)[0, 1])
            d[f"{comp}_ratio_all"] = float(np.median(mine / theirs))
            if good.sum() > 5:
                d[f"{comp}_corr_highsnr"] = float(np.corrcoef(mine[good], theirs[good])[0, 1])
                d[f"{comp}_ratio_highsnr"] = float(np.median(mine[good] / theirs[good]))
                d[f"{comp}_maxrel_highsnr"] = float(np.abs(mine[good] / theirs[good] - 1).max())
                d[f"{comp}_n_highsnr"] = int(good.sum())
        res[key] = d
        print(f"  {key}: npairs ratio={d['npairs_ratio_median']:.6f}+-{d['npairs_ratio_std']:.1e} | "
              f"xip r={d['xip_corr_all']:+.5f} ratio={d['xip_ratio_all']:+.4f} | "
              f"xim r={d['xim_corr_all']:+.5f} ratio={d['xim_ratio_all']:+.4f}"
              + (f" | xim high-SNR r={d['xim_corr_highsnr']:+.5f} ratio={d['xim_ratio_highsnr']:+.4f}"
                 if "xim_corr_highsnr" in d else ""))

    best = max(res, key=lambda k: res[k]["xip_corr_all"] + res[k]["xim_corr_all"])
    out = {"what": "check A4: independent pair counter vs TreeCorr, correlated shear field",
           "n_gal": n, "nbins": NBINS, "results": res, "best": best,
           "note": "TreeCorr npairs counts ordered pairs on auto-correlations; this counter "
                   "counts unordered, hence the exact 0.5"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a4_estimator.json").write_text(json.dumps(out, indent=2))
    print(f"\nbest by xi agreement: {best}")


if __name__ == "__main__":
    main()
