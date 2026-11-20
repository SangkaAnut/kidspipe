# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Does gg.process(cat) count half as many pairs as gg.process(cat, cat)?

This is the difference between the two projects:
  pipeline/01_lensing_validation : gg.process(cats[i])            for i == j  (auto)
  this project (measure/xi.py)   : gg.process(cat1, cat2)         with cat2 = cat1

The validated task matched the official xi+/- to 0.051 sigma using the first form. Our npairs are
exactly 2x theirs on the same catalogue. This measures which convention is which, and whether the
xi values differ.
"""
import numpy as np
import treecorr
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402

from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs

print("treecorr", treecorr.__version__)

d = load_bin(1)
N = d["ra"].size
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(N, 20000, replace=False))
rec = {k: d[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
cat = make_treecorr_cat(rec)
print(f"using {rec['ra'].size:,} galaxies, 100 bins for speed")

res = {}
for label, fn in (("process(cat)", lambda g: g.process(cat)),
                  ("process(cat, cat)", lambda g: g.process(cat, cat))):
    g = treecorr.GGCorrelation(treecorr_kwargs(100, 0.0, 1))
    fn(g)
    res[label] = (g.npairs.copy(), g.xip.copy(), g.weight.copy())
    print(f"  {label:20s} npairs={g.npairs.sum():.6e}  weight={g.weight.sum():.6e}")

a, b = res["process(cat)"], res["process(cat, cat)"]
m = (a[0] > 0) & (b[0] > 0)
print()
print(f"npairs ratio  (cat)/(cat,cat) = {a[0][m].sum()/b[0][m].sum():.12f}")
print(f"weight ratio  (cat)/(cat,cat) = {a[2][m].sum()/b[2][m].sum():.12f}")
print(f"xip max |diff|                = {np.abs(a[1][m]-b[1][m]).max():.3e}")
print(f"xip corr                      = {np.corrcoef(a[1][m], b[1][m])[0,1]:.12f}")
print()
print("Interpretation: if npairs ratio == 0.5 and xip agrees to machine precision, then")
print("process(cat) counts UNORDERED pairs (each pair once) and the xi values are identical.")
print("That makes 'npairs' not comparable between the two projects, while xi is.")
