# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Why did naive RA-strip chunking lose 3.3%? Compare three decompositions of the same pairs.

  (1) single shot                          gg.process(cat, cat)
  (2) naive strip sum, cross counted once  sum_{a<=b} process(chunk_a, chunk_b)
  (3) naive strip sum, cross counted twice sum_{a<=b} process(...) * (1 or 2)
  (4) TreeCorr native patches              Catalog(..., patch=id), one process(cat, cat)

(2) is expected to be low if auto-correlations count ordered pairs; (3) is expected to match
(1) exactly if the only issue is folding. (4) is the authoritative reference.
"""
import numpy as np
import treecorr
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402

from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs

F = 0.05
K = 4

d = load_bin(1)
N = d["ra"].size
rng = np.random.default_rng(5)
idx = np.sort(rng.choice(N, int(N * F), replace=False))
rec = {k: d[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
ra = rec["ra"].astype(np.float64)

# single
cat = make_treecorr_cat(rec)
gg = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
gg.process(cat, cat)
ref_np, ref_w = gg.npairs.copy(), gg.weight.copy()
print(f"(1) single            npairs={ref_np.sum():.6e}  sum(weight)={ref_w.sum():.6e}")

edges = np.quantile(ra, np.linspace(0, 1, K + 1))
masks = []
for k in range(K):
    if k < K - 1:
        masks.append((ra >= edges[k]) & (ra < edges[k + 1]))
    else:
        masks.append((ra >= edges[k]) & (ra <= edges[k + 1]))
print("chunk sizes:", [int(m.sum()) for m in masks], "total", int(sum(m.sum() for m in masks)))
subs = [{kk: rec[kk][m] for kk in ("ra", "dec", "e1", "e2", "w")} for m in masks]
cats = [make_treecorr_cat(s) for s in subs]

for label, fold in (("(2) cross once ", False), ("(3) cross twice", True)):
    acc_np = np.zeros_like(ref_np)
    acc_w = np.zeros_like(ref_w)
    for a in range(K):
        for b in range(a, K):
            g = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
            g.process(cats[a], cats[b])
            f = 2.0 if (fold and a != b) else 1.0
            acc_np += f * g.npairs
            acc_w += f * g.weight
    print(f"{label}         npairs={acc_np.sum():.6e}  sum(weight)={acc_w.sum():.6e}  "
          f"npairs ratio={acc_np.sum()/ref_np.sum():.9f}  w ratio={acc_w.sum()/ref_w.sum():.9f}")

# native patches: patch ids must be ints
patch = np.zeros(ra.size, dtype=int)
for k, m in enumerate(masks):
    patch[m] = k
catp = treecorr.Catalog(
    ra=rec["ra"].astype(np.float64), dec=rec["dec"].astype(np.float64),
    ra_units="deg", dec_units="deg",
    g1=rec["e1"].astype(np.float64), g2=rec["e2"].astype(np.float64),
    w=rec["w"].astype(np.float64), patch=patch,
)
gp = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
gp.process(catp, catp)
print(f"(4) native patches    npairs={gp.npairs.sum():.6e}  sum(weight)={gp.weight.sum():.6e}  "
      f"npairs ratio={gp.npairs.sum()/ref_np.sum():.9f}  w ratio={gp.weight.sum()/ref_w.sum():.9f}")
m = (ref_w > 0) & (gp.weight > 0)
print(f"    xi+ max |rel diff| vs single = {np.abs(gp.xip[m]/gg.xip[m]-1).max():.3e}")
print()
print("If (3) matches (1) exactly, naive chunking IS exact once the fold is handled.")
print("If (4) matches (1) exactly, TreeCorr patches are the safest chunking route.")
