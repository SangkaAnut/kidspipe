# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Verify that folded chunk summation is exact for K=8, on an auto pair AND a cross pair.

The fold rule established by dr5_chunk_diagnose.py: TreeCorr counts ORDERED pairs, so
  sum_{a} process(c_a, c_a)  +  2 * sum_{a<b} process(c_a, c_b)  ==  process(cat, cat)
exactly (npairs ratio 1.000000000). This script checks that across more chunks, across two
tomographic bins, and quantifies the xi+ difference in sigma units rather than relative terms
(relative differences are meaningless where xi+ crosses zero).
"""
import numpy as np
import treecorr
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402

from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs

F = 0.05
K = 8


def split(rec, K):
    ra = rec["ra"].astype(np.float64)
    edges = np.quantile(ra, np.linspace(0, 1, K + 1))
    masks = []
    for k in range(K):
        masks.append((ra >= edges[k]) & (ra < edges[k + 1]) if k < K - 1
                     else (ra >= edges[k]) & (ra <= edges[k + 1]))
    return masks


sub = {}
for b in (1, 2):
    d = load_bin(b)
    n = d["ra"].size
    rng = np.random.default_rng(99 + b)
    idx = np.sort(rng.choice(n, int(n * F), replace=False))
    sub[b] = {k: d[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
    del d
    print(f"bin{b}: {idx.size:,}", flush=True)

for (i, j) in [(1, 1), (1, 2)]:
    ci = make_treecorr_cat(sub[i])
    cj = make_treecorr_cat(sub[j]) if j != i else ci
    g = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
    g.process(ci, cj)
    ref_np, ref_w, ref_xip, ref_var = g.npairs.copy(), g.weight.copy(), g.xip.copy(), g.varxip.copy()

    mi, mj = split(sub[i], K), split(sub[j], K)
    cai = [make_treecorr_cat({k: sub[i][k][m] for k in ("ra", "dec", "e1", "e2", "w")}) for m in mi]
    caj = (cai if j == i else
           [make_treecorr_cat({k: sub[j][k][m] for k in ("ra", "dec", "e1", "e2", "w")}) for m in mj])

    acc_np = np.zeros_like(ref_np)
    acc_w = np.zeros_like(ref_w)
    acc_wx = np.zeros_like(ref_xip)
    nproc = 0
    for a in range(K):
        for b in range(K):
            if j == i and b < a:
                continue  # symmetric: fold below
            gg = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
            gg.process(cai[a], caj[b])
            fold = 1.0 if (j != i or a == b) else 2.0
            acc_np += fold * gg.npairs
            acc_w += fold * gg.weight
            acc_wx += fold * gg.weight * gg.xip
            nproc += 1
    acc_xip = acc_wx / np.where(acc_w > 0, acc_w, np.nan)
    m = (ref_w > 0) & np.isfinite(acc_xip)
    pull = np.abs(acc_xip[m] - ref_xip[m]) / np.sqrt(ref_var[m])

    print(f"\nBin{i}_Bin{j}: {nproc} chunk-pair processes")
    print(f"  npairs ratio        = {acc_np.sum()/ref_np.sum():.12f}")
    print(f"  sum(weight) ratio   = {acc_w.sum()/ref_w.sum():.12f}")
    print(f"  max|npairs diff|    = {np.abs(acc_np-ref_np).max():.3e}")
    print(f"  xi+ corr            = {np.corrcoef(acc_xip[m], ref_xip[m])[0,1]:.12f}")
    print(f"  xi+ max pull (sigma) = {pull.max():.3e}   median {np.median(pull):.3e}")

print("\nIf npairs and weight ratios are 1.000000000 and max pull ~0, folded chunking is exact")
print("and can be used for the production run.")
