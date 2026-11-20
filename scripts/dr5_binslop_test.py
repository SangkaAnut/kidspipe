"""Is bin_slop=1.5 usable? The validated sibling task (pipeline/01_lensing_validation) measured all 21
pairs with bin_slop=1.5 at 1.5-1.7e9 pairs/s and matched the official xi+/- to 0.051 sigma, with a
bin_slop 1.5 -> 0.75 convergence shift of only 0.096 sigma.

Our project has been using bin_slop=0/0.05 at 2.4e8 pairs/s -- 7x slower. If 1.5 is accurate on OUR
pipeline (same TreeCorr, same catalogue, same binning), the production run drops from ~49 h to a few
hours. This script measures the speed and the accuracy difference directly.

Method: full-density bin 1, one auto pair. Compare slop=0 (reference, already measured: 539 s per
1/16 chunk) against slop=1.5 and 0.75, on a 10% subsample for tractability, reporting
  (a) wall time and pairs/s,
  (b) npairs ratio vs the slop=0 reference (bin_slop must not change the pair count materially),
  (c) xi+ difference in units of the shape-noise sigma and in units of the OFFICIAL covariance sigma,
  (d) the rebinned 9-bin xi+ difference vs the official data vector.
"""
import time

import numpy as np
import treecorr

from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs
from kidspipe.measure.official import PAIRS, load_official_xi

F = 0.10
d = load_bin(1)
N = d["ra"].size
rng = np.random.default_rng(2024)
idx = np.sort(rng.choice(N, int(N * F), replace=False))
rec = {k: d[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
del d
print(f"bin1 {N:,} -> {idx.size:,} ({F:.0%}), 1000 log bins 2-300 arcmin", flush=True)

off = load_official_xi()
pi = PAIRS.index((0, 0))
o_xip = off["xip"][pi]
o_np = off["npairs"][pi]
rn = off["r_nom"][pi]
o_sig = off["sigma_xip"][pi]

cat = make_treecorr_cat(rec)
res = {}
for slop in (0.0, 0.05, 0.5, 0.75, 1.5, 3.0):
    gg = treecorr.GGCorrelation(treecorr_kwargs(1000, slop, 16))
    t = time.time()
    gg.process(cat, cat)
    dt = time.time() - t
    res[slop] = dict(t=dt, np=gg.npairs.copy(), xip=gg.xip.copy(),
                     var=gg.varxip.copy(), rnom=gg.rnom)
    print(f"  bin_slop={slop:4.2f}: {dt:7.1f} s   npairs={gg.npairs.sum():.5e}  "
          f"pairs/s={gg.npairs.sum()/dt:.3e}", flush=True)

ref = res[0.0]
print()
print(f"{'slop':>6} {'time':>8} {'speedup':>8} {'npairs ratio':>13} "
      f"{'max|dxip|/sig_own':>18} {'max|dxip|/sig_off':>18}")
for s, v in res.items():
    m = (v["np"] > 0) & (ref["np"] > 0)
    dnp = v["np"][m].sum() / ref["np"][m].sum()
    dx = v["xip"][m] - ref["xip"][m]
    pull_own = np.max(np.abs(dx) / np.sqrt(ref["var"][m]))
    pull_off = np.max(np.abs(dx) / o_sig[m])
    print(f"{s:6.2f} {v['t']:7.1f}s {ref['t']/v['t']:8.1f}x {dnp:13.8f} "
          f"{pull_own:18.4f} {pull_off:18.4f}")

# rebinned 9-bin comparison against the official data vector
edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
print()
print("Rebinned to 9 bins, max |ours - official| / sigma_official_diag:")
from astropy.io import fits
h = fits.open("/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/"
              "KiDS_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/"
              "data/KiDS_Legacy_xipm.fits")
xp9 = h["xiP"].data
sel = (xp9["BIN1"] == 1) & (xp9["BIN2"] == 1)
val9 = xp9["VALUE"][sel]
sig9 = np.sqrt(np.diag(h["COVMAT"].data)[:189]).reshape(21, 9)[0]
for s, v in res.items():
    coarse = np.zeros(9)
    for c in range(9):
        mm = (v["rnom"] >= edges[c]) & (v["rnom"] < (edges[c + 1] if c < 8 else 300.1))
        w = v["np"][mm]
        coarse[c] = np.sum(w * v["xip"][mm]) / np.sum(w) if w.sum() else np.nan
    # scale the subsample xi to full: xi is unbiased, so compare directly
    d9 = np.abs(coarse - val9) / sig9
    print(f"  slop={s:4.2f}: max={np.nanmax(d9):8.4f}  median={np.nanmedian(d9):8.4f}")

np.savez("runs/dr5_measure/binslop_test.npz",
         **{f"np_{k}": v["np"] for k, v in res.items()},
         **{f"xip_{k}": v["xip"] for k, v in res.items()},
         **{f"t_{k}": np.array(v["t"]) for k, v in res.items()},
         rnom=ref["rnom"])
print("\nwrote runs/dr5_measure/binslop_test.npz")
