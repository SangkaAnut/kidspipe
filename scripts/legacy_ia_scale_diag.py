"""Diagnostic: does a global rescaling of the IA amplitude remove the residual chi2 offset / log10 M_1 trend?
usage: python scripts/legacy_ia_scale_diag.py [n=200]   (reads <runs>/legacy_val/{chi2,selection}.npz, else the shipped copy; writes
<runs>/legacy_val/ia_scale_diag.json; <runs> = $KIDSPIPE_RUNS, default <root>/runs; runs from any directory)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths
from kidspipe.legacy import LegacyCosebisModel, LegacyCosebisLikelihood, load_chain, chain_samples
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = paths.runs_dir() / "legacy_val"; OUT.mkdir(parents=True, exist_ok=True)
ch = load_chain(); z = np.load(paths.find_result("legacy_val/chi2.npz"))
rng = np.random.default_rng(5)
sel = np.load(paths.find_result("legacy_val/selection.npz"))
rows = rng.choice(np.unique(sel["draws"]), n, replace=False)
S = chain_samples(ch, rows)
m = LegacyCosebisModel(nthreads=min(4, os.cpu_count() or 1), cache_dir=str(paths.runs_dir() / "legacy" / "cache"), cache_size=n + 5); like = LegacyCosebisLikelihood()
scales = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6]
out = {}
for sc in scales:
    m.opt.n_amp_scale = sc
    c2 = np.array([like.chi2(m.mu(s)) for s in S]); d = c2 - np.array([s["chi2_off"] for s in S])
    lm1 = np.array([s["log10_M_mean_1"] for s in S])
    out[sc] = dict(mean=d.mean(), std=d.std(ddof=1), corr_with_logM1=float(np.corrcoef(lm1, d)[0, 1]),
                   slope_per_dex_logM1=float(np.polyfit(lm1, d, 1)[0]))
    print(sc, out[sc], flush=True)
json.dump({"n": n, "scales": out}, open(OUT / "ia_scale_diag.json", "w"), indent=1)
