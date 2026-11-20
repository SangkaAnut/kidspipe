"""Verifier: bin_slop 0 vs 1.5 on random subsamples of bin6 (same galaxies, same pairs), plus chunk reproduction."""
import sys, time, json, numpy as np
from pathlib import Path
ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline"); sys.path.insert(0, str(ROOT/"src"))
from kidspipe.measure.xi import load_bin, make_treecorr_cat, run_pair, corr_to_arrays
rec = load_bin(6); N = rec["ra"].size; print("N", N, flush=True)
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [20261030, 777]
edges = np.exp(np.linspace(np.log(2), np.log(300), 10))
out = {}
for sd in seeds:
    rng = np.random.default_rng(sd); idx = np.sort(rng.choice(N, int(FRAC*N), replace=False))
    cat = make_treecorr_cat({kk: rec[kk][idx] for kk in ("ra", "dec", "e1", "e2", "w")})
    r = {}
    for slop in (0.0, 1.5):
        t = time.time(); a = corr_to_arrays(run_pair(cat, cat, bin_slop=slop, num_threads=8)); r[slop] = a
        print(f"seed {sd} slop {slop}: {time.time()-t:.0f}s npairs={a['npairs'].sum():.5e}", flush=True)
    a0, a1 = r[0.0], r[1.5]
    w = a0["weight"]; rn = a0["r_nom"]
    o = {"npairs_ratio": float(a1["npairs"].sum()/a0["npairs"].sum())}
    for key in ("xip", "xim"):
        o[key+"_total_shift_pct"] = float(100*((w*(a1[key]-a0[key])).sum()/(w*a0[key]).sum()))
        for lo, hi in ((2, 10), (10, 60), (60, 300.1)):
            m = (rn >= lo) & (rn < hi)
            o[f"{key}_{lo}_{hi}_pct"] = float(100*((w[m]*(a1[key][m]-a0[key][m])).sum()/(w[m]*a0[key][m]).sum()))
        # 9-bin shifts in units of the bin's own statistical error (from slop0 varxi, doubled for ordered)
        s9 = []
        for c in range(9):
            m = (rn >= edges[c]) & (rn < edges[c+1] if c < 8 else rn <= 300.01)
            d = (w[m]*(a1[key][m]-a0[key][m])).sum()/w[m].sum()
            err = np.sqrt(2*(a0["varxi"][m]*w[m]**2).sum())/w[m].sum()
            s9.append(float(d/err))
        o[key+"_9bin_shift_over_stat"] = s9
    out[sd] = o; print(json.dumps(o, indent=1), flush=True)
json.dump(out, open(ROOT/"runs/dr5_measure/pilot_bin6/verify_slop_subsample.json", "w"), indent=1)
