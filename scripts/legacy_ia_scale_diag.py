"""Diagnostic: does a global rescaling of the IA amplitude remove the residual chi2 offset / log10 M_1 trend?
usage: python scripts/legacy_ia_scale_diag.py [n=200]   (reads runs/legacy_val/chi2.npz rows; writes runs/legacy_val/ia_scale_diag.json)"""
import json, sys
import numpy as np
from kidspipe.legacy import LegacyCosebisModel, LegacyCosebisLikelihood, load_chain, chain_samples
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ch = load_chain(); z = np.load("runs/legacy_val/chi2.npz")
rng = np.random.default_rng(5)
sel = np.load("runs/legacy_val/selection.npz")
rows = rng.choice(np.unique(sel["draws"]), n, replace=False)
S = chain_samples(ch, rows)
m = LegacyCosebisModel(nthreads=4, cache_dir="runs/legacy/cache", cache_size=n + 5); like = LegacyCosebisLikelihood()
scales = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
out = {}
for sc in scales:
    m.opt.n_amp_scale = sc
    c2 = np.array([like.chi2(m.mu(s)) for s in S]); d = c2 - np.array([s["chi2_off"] for s in S])
    lm1 = np.array([s["log10_M_mean_1"] for s in S])
    out[sc] = dict(mean=d.mean(), std=d.std(ddof=1), corr_with_logM1=float(np.corrcoef(lm1, d)[0, 1]),
                   slope_per_dex_logM1=float(np.polyfit(lm1, d, 1)[0]))
    print(sc, out[sc], flush=True)
json.dump({"n": n, "scales": out}, open("runs/legacy_val/ia_scale_diag.json", "w"), indent=1)
