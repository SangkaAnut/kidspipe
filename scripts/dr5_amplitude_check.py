# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Amplitude of xi_+ : our full-subsample measurement vs the official 9-bin xipm FITS.

Purpose: decide whether our xi_+ AMPLITUDE is right, independent of the npairs
normalisation question. If our sample were a different (larger-footprint) sample,
xi_+ would still be unbiased -- shear is a local quantity -- so a matching
amplitude is a genuine consistency check on the catalogue, not on npairs.
"""
import os

import numpy as np
import treecorr
from astropy.io import fits
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402

from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs

F = 0.10
d = load_bin(1)
N = d["ra"].size
rng = np.random.default_rng(2024)
idx = np.sort(rng.choice(N, int(N * F), replace=False))
rec = {k: d[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
print(f"bin1 {N:,} -> {idx.size:,} ({F:.0%})")

p = str(paths.official_dir() / "KiDS_Legacy_xipm.fits")
h = fits.open(p)
xp = h["xiP"].data
sel = (xp["BIN1"] == 1) & (xp["BIN2"] == 1)
ang = xp["ANG"][sel]
val = xp["VALUE"][sel]
cov = h["COVMAT"].data
sig = np.sqrt(np.diag(cov)[:189].reshape(21, 9)[0])
print("official centres :", ang.round(2))
print("official xiP     :", val.round(6))
print("official sigma   :", sig.round(6))

kw = treecorr_kwargs(nbins=9, bin_slop=0.0, num_threads=os.cpu_count() or 1)
cat = make_treecorr_cat(rec)
gg = treecorr.GGCorrelation(kw)
gg.process(cat, cat)
ours, oursig, rn = gg.xip, np.sqrt(gg.varxip), gg.rnom
print()
print("our rnom         :", rn.round(2))
print("our xiP          :", ours.round(6))
print("our sigma        :", oursig.round(6))
print()
print("ratio ours/off        :", (ours / val).round(4))
print("pull (ours-off)/sig_off:", ((ours - val) / sig).round(2))
print("pull / own noise       :", ((ours - val) / oursig).round(2))
print()
print("noise factor vs official = %.2f (expect ~%.2f for a %.0f%% subsample)"
      % (np.median(oursig / sig), 1 / np.sqrt(F), 100 * F))
np.savez(paths.runs_dir() / "dr5_measure" / "amplitude_check.npz",
         rn=rn, xip_ours=ours, xip_off=val, sig_off=sig, sig_ours=oursig)
print("wrote", paths.runs_dir() / "dr5_measure" / "amplitude_check.npz")
