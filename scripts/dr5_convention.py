# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Decide the shear sign convention against the official ascii xi_+/xi_-, on a 1/20 subsample.

e1/e2 in KiDS are Lensfit ellipticities in the (RA, Dec) sky frame; TreeCorr expects the
(Dec, RA)-ish frame, so a flip of one component is the usual fix.  xi_+ is invariant under
flipping BOTH components, xi_- is not, so all four combinations are measured and the one
that reproduces the official xi_+ AND xi_- with a positive slope is adopted.  Nothing assumed.

Because the subsample changes the normalisation but not the shape, the diagnostic is the
correlation coefficient and the *sign* of the ratio to the official values.

Writes runs/dr5_measure/convention.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidspipe.measure.official import PAIRS, load_official_xi
from kidspipe.measure.xi import load_bin, make_treecorr_cat, run_pair, corr_to_arrays

OUT = Path(__file__).resolve().parents[1] / "runs" / "dr5_measure"
STRIDE = 20
THREADS = min(16, os.cpu_count() or 1)
PAIRS_TODO = [(1, 1), (1, 2)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    off = load_official_xi()
    recs = {}
    for b in (1, 2):
        r = load_bin(b)
        idx = np.arange(0, r["ra"].size, STRIDE)
        recs[b] = {k: r[k][idx] for k in r}
        print(f"bin{b}: {recs[b]['ra'].size:,} galaxies (stride {STRIDE})", flush=True)

    summary = {}
    for flip1, flip2 in [(False, False), (False, True), (True, False), (True, True)]:
        cats = {b: make_treecorr_cat(recs[b], flip_g1=flip1, flip_g2=flip2) for b in (1, 2)}
        key = f"flip1={int(flip1)},flip2={int(flip2)}"
        s = {}
        for (i, j) in PAIRS_TODO:
            gg = run_pair(cats[i], cats[j], num_threads=THREADS)
            a = corr_to_arrays(gg)
            pi = PAIRS.index((i - 1, j - 1))
            m = a["npairs"] > 0
            name = f"Bin{i}_Bin{j}"
            for comp in ("xip", "xim", "xip_im"):
                s[f"{name}_{comp}_corr"] = float(np.corrcoef(a[comp][m], off[comp][pi][m])[0, 1])
            # slope with the subsample normalisation removed: use the ratio's sign, and the
            # relative amplitude of xip vs xim (which is convention-sensitive)
            s[f"{name}_xip_over_xim_ours"] = float(np.median(a["xip"][m] / a["xim"][m]))
            s[f"{name}_xip_over_xim_off"] = float(np.median(off["xip"][pi][m] / off["xim"][pi][m]))
            s[f"{name}_npairs_sum"] = float(a["npairs"].sum())
            print(f"  {key} {name}: r_xip={s[f'{name}_xip_corr']:+.4f} "
                  f"r_xim={s[f'{name}_xim_corr']:+.4f} "
                  f"xip/xim ours={s[f'{name}_xip_over_xim_ours']:+.4f} "
                  f"off={s[f'{name}_xip_over_xim_off']:+.4f}", flush=True)
        s["score"] = float(sum(s[f"{n}_xim_corr"][0] if False else s[f"{n}_xim_corr"]
                               for n in [f"Bin{i}_Bin{j}" for i, j in PAIRS_TODO]))
        summary[key] = s

    best = max(summary, key=lambda k: summary[k]["score"])
    flips = {"flip_g1": "flip1=1" in best, "flip_g2": "flip2=1" in best}
    out = {
        "what": "shear sign convention decided against official xi+/- (check A1/A2 pre-step)",
        "stride": STRIDE, "threads": THREADS, "pairs": PAIRS_TODO,
        "binning": "Log, 1000 bins, 2-300 arcmin, sep_units='arcmin', bin_slop=0.05",
        "note": "xi_+ invariant under flipping both components; a single flip flips the sign of "
                "xi_-. r_xim is the decisive statistic because xi_- changes sign with a single flip.",
        "summary": summary,
        "adopted_key": best,
        "adopted_flips": flips,
        "runtime_s": time.time() - t0,
    }
    (OUT / "convention.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nadopted: {best} -> {flips}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
