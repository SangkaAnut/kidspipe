"""Is spatial chunking EXACT? Compare patch-chunked accumulation against the single-shot run.

** CONCLUSION SUPERSEDED — READ THIS BEFORE USING THE OUTPUT **

This script's verdict line ("NOT exact -> chunking changes the answer") is MISLEADING and was
written before the cause was found. The 3.3% deficit it reports is entirely the auto-pair fold:
it sums cross-chunk pairs once, but TreeCorr counts ORDERED pairs, so they need a factor 2.

    process(cat,cat) == sum_c process(c_c,c_c) + 2 * sum_{a<b} process(c_a,c_b)

With the fold applied, chunking is exact to machine precision (npairs and sum(weight) ratios
1.000000000). See dr5_chunk_diagnose.py, dr5_chunk_verify.py and dr5_chunk_proof.py, and
runs/dr5_measure/CHUNKING.md. Kept because the 0.9672 number is the trap worth remembering.

--- original docstring ---
If chunking changes npairs or xip at the 1e-12 level, we can split the production run into
independent spatial chunks and sum them. If it changes them at the 1e-3 level, chunking is an
approximation and must not be used for A1/A2.

Method: bin 1, 5% subsample (cheap), 1000 log bins, 2-300 arcmin, bin_slop=0.
  single : gg.process(cat, cat)
  chunked: split the sky into K equal-area HEALPix-like RA strips, run every chunk-pair
           (k,k') with k<=k', and sum npairs / weight / xip-weightedly.
"""
import numpy as np
import treecorr

from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs

F = 0.05
K = 4  # chunks along RA

d = load_bin(1)
N = d["ra"].size
rng = np.random.default_rng(5)
idx = np.sort(rng.choice(N, int(N * F), replace=False))
rec = {k: d[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
print(f"bin1 {N:,} -> {idx.size:,}")

# --- single shot
cat = make_treecorr_cat(rec)
gg = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
gg.process(cat, cat)
ref_np = gg.npairs.copy()
ref_w = gg.weight.copy()
ref_xip = gg.xip.copy()
print(f"single: npairs={ref_np.sum():.6e}  sum(weight)={ref_w.sum():.6e}")

# --- chunked: RA strips
ra = rec["ra"].astype(np.float64)
# handle RA wrap: shift so the range is contiguous, then split by percentile of RA
order = np.argsort(ra)
edges = np.quantile(ra[order], np.linspace(0, 1, K + 1))
chunks = []
for k in range(K):
    if k < K - 1:
        m = (ra >= edges[k]) & (ra < edges[k + 1])
    else:
        m = (ra >= edges[k]) & (ra <= edges[k + 1])
    sub = {kk: rec[kk][m] for kk in ("ra", "dec", "e1", "e2", "w")}
    chunks.append(make_treecorr_cat(sub))
    print(f"  chunk {k}: {m.sum():,} galaxies")

acc_np = np.zeros_like(ref_np)
acc_w = np.zeros_like(ref_w)
acc_wxip = np.zeros_like(ref_xip)  # sum of w*xip
for a in range(K):
    for b in range(a, K):
        g = treecorr.GGCorrelation(treecorr_kwargs(1000, 0.0, 16))
        g.process(chunks[a], chunks[b])
        acc_np += g.npairs
        acc_w += g.weight
        acc_wxip += g.weight * g.xip
print(f"chunked: npairs={acc_np.sum():.6e}  sum(weight)={acc_w.sum():.6e}")

print()
print("=== agreement ===")
print(f"npairs  : max |diff| = {np.abs(acc_np - ref_np).max():.6e}   "
      f"total ratio = {acc_np.sum()/ref_np.sum():.12f}")
print(f"weight  : max |diff| = {np.abs(acc_w - ref_w).max():.6e}   "
      f"total ratio = {acc_w.sum()/ref_w.sum():.12f}")
acc_xip = acc_wxip / np.where(acc_w > 0, acc_w, np.nan)
m = (ref_w > 0) & np.isfinite(acc_xip) & (ref_xip != 0)
print(f"xip     : max |rel diff| = {np.abs(acc_xip[m]/ref_xip[m]-1).max():.6e}")
print(f"          corr = {np.corrcoef(acc_xip[m], ref_xip[m])[0,1]:.12f}")
print(f"          max |pull| in sigma units = "
      f"{np.abs((acc_xip[m]-ref_xip[m])/np.sqrt(gg.varxip[m])).max():.4f}")
print()
print("VERDICT:", "EXACT -> chunking is safe" if np.abs(acc_xip[m]/ref_xip[m]-1).max() < 1e-6
      else "NOT exact -> chunking changes the answer")
