"""Peak-RSS / time scaling of the unordered runner path on random subsamples (<=300k). 6 threads. One config per call:
   python verify_slop15_autos_memory.py N cross|auto [repeat]"""
import sys, time, resource, numpy as np
sys.path.insert(0, "/mnt/d/cloning_kids_Kappa_pipeline/src")
import treecorr
from kidspipe.measure import xi as xim
from kidspipe.measure.xi import corr_to_arrays, make_treecorr_cat, treecorr_kwargs
N = int(sys.argv[1]); mode = sys.argv[2]; rep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
def cur():
    return int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) / 1024.0
b0 = cur()
bins = {"cross": (1, 2), "auto": (6, 6), "auto1": (1, 1), "auto5": (5, 5)}[mode]
cats = {}
for b in sorted(set(bins)):
    rec = xim.load_bin(b)
    sel = np.sort(np.random.default_rng(b).choice(rec["ra"].size, N, replace=False))
    rec = {k: rec[k][sel] for k in ("ra", "dec", "e1", "e2", "w")}
    cats[b] = make_treecorr_cat(rec); del rec
b1 = cur()
kw = treecorr_kwargs(bin_slop=1.5, num_threads=6)
for r in range(rep):
    gg = treecorr.GGCorrelation(kw); t = time.time()
    gg.process(cats[bins[0]]) if bins[0] == bins[1] else gg.process(cats[bins[0]], cats[bins[1]])
    dt = time.time() - t
    print(f"N={N} {mode} rep{r}: base {b0:.0f} MB, after cats {b1:.0f} MB, after process {cur():.0f} MB, peak {rss():.0f} MB, "
          f"time {dt:.1f}s npairs {gg.npairs.sum():.4e}", flush=True)
