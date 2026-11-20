"""Pilot Bin6_Bin6 re-measured with bin_slop=1.5 (their setting), otherwise OUR chunked-ordered settings.
Resumable; chunk files go to runs/dr5_measure/pilot_bin6/slop15_chunks/ (NOT xi_chunked/).  Assembly by
assemble() with the same fold-weighted sums as scripts/dr5_measure_xi_chunked.py::do_assemble."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline")
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("drv", ROOT / "scripts/dr5_measure_xi_chunked.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
from kidspipe.measure.xi import run_pair, corr_to_arrays  # noqa: E402

OUT = ROOT / "runs/dr5_measure/pilot_bin6/slop15_chunks"


def assemble(files):
    acc = None
    for f in files:
        a = np.load(f); fold = float(a["fold"]); w = np.asarray(a["weight"], float)
        if acc is None:
            acc = {k: np.zeros_like(w) for k in ("npairs", "weight", "wxip", "wxim", "wxim_im", "vn")}
            rnom = np.asarray(a["r_nom"], float); meanr = np.asarray(a["meanr"], float)
        acc["npairs"] += fold * a["npairs"]; acc["weight"] += fold * w
        acc["wxip"] += fold * w * a["xip"]; acc["wxim"] += fold * w * a["xim"]
        acc["wxim_im"] += fold * w * a["xim_im"]; acc["vn"] += fold * a["varxi"] * w ** 2
    ws = np.where(acc["weight"] > 0, acc["weight"], np.nan)
    return {"r_nom": rnom, "meanr": meanr, "xip": acc["wxip"] / ws, "xim": acc["wxim"] / ws,
            "xim_im": acc["wxim_im"] / ws, "varxi": acc["vn"] / ws ** 2,
            "npairs": acc["npairs"], "weight": acc["weight"]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cats1, cats2 = drv.load_pair_cats(6, 6, 16, {})
    t0 = time.time()
    for (a, b) in drv.chunk_pairs(16, True):
        f = OUT / f"c{a:02d}_{b:02d}.npz"
        if f.exists():
            continue
        t1 = time.time()
        arr = corr_to_arrays(run_pair(cats1[a], cats2[b], bin_slop=1.5, num_threads=8))
        arr["fold"] = np.float64(drv.fold_factor(a, b, True))
        np.savez(f, **arr)
        print(f"[{time.time()-t0:7.0f}s] c{a:02d}_{b:02d} {time.time()-t1:6.1f}s npairs={arr['npairs'].sum():.3e}", flush=True)
    np.savez(OUT.parent / "bin6_bin6_slop15_assembled.npz", **assemble(sorted(OUT.glob("c*.npz"))))
    print("done", time.time() - t0)
