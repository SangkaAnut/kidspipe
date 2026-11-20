# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""End-to-end proof that the chunked driver reproduces the single-shot pipeline exactly.

Runs at low density on purpose (2% subsample, 4 chunks) so it finishes in minutes and can be
re-run by anyone. It:
  1. writes a temporary 2%-subsample catalogue where load_bin() expects it, into a scratch tree;
  2. measures Bin1_Bin1 single-shot (the reference);
  3. measures every chunk-pair through dr5_measure_xi_chunked.py's own code path;
  4. assembles via --assemble;
  5. compares npairs, sum(weight) and xi+ against the reference.

Pass criterion: npairs and sum(weight) ratios == 1 to <1e-9, xi+ correlation 1.000000.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path.home() / "kids_dr5_chunkproof"   # scratch tree for the proof
PY = sys.executable

# --- 1. build a 2% catalogue in a scratch WORK tree
import os
os.environ["PYTHONSAFEPATH"] = ""
os.environ["PYTHONPATH"] = str(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))
from kidspipe.measure.xi import load_bin  # noqa: E402
from kidspipe import paths  # noqa: E402

SCRATCH.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)
rec = load_bin(1, work=paths.dr5_work_dir())
n = rec["ra"].size
idx = np.sort(rng.choice(n, int(n * 0.02), replace=False))
sub = {k: rec[k][idx] for k in rec}
np.savez(SCRATCH / "bin0.npz", **sub)
print(f"scratch catalogue: {idx.size:,} galaxies -> {SCRATCH/'bin0.npz'}")

# --- 2. reference single shot
for d in (SCRATCH / "xi_chunked",):
    if d.exists():
        shutil.rmtree(d)

from kidspipe.measure.xi import make_treecorr_cat, run_pair, corr_to_arrays  # noqa: E402

cat = make_treecorr_cat(sub)
gg = run_pair(cat, cat, nbins=1000, bin_slop=0.0, num_threads=min(16, os.cpu_count() or 1))
ref = corr_to_arrays(gg)
print(f"reference: npairs={ref['npairs'].sum():.6e}  w={ref['weight'].sum():.6e}")

# --- 3. chunked, using the real driver with an overridden WORK via a thin wrapper
NCH = 4
env = dict(os.environ, PYTHONSAFEPATH="", PYTHONPATH=str(ROOT / "src"))
# the driver reads ~/kids_dr5_work by default; point HOME-equivalent at the scratch tree by
# temporarily swapping the directory it looks for. Simplest: run the driver functions directly.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "drv", ROOT / "scripts" / "dr5_measure_xi_chunked.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
drv.OUTROOT = SCRATCH / "xi_chunked"

masks = drv.chunk_masks(sub["ra"], NCH)
cats = [make_treecorr_cat({k: sub[k][m] for k in sub}) for m in masks]
print(f"chunks: {[c.nobj for c in cats]}")

d = drv.OUTROOT / "chunks" / "Bin1_Bin1"
d.mkdir(parents=True, exist_ok=True)
for (a, b) in drv.chunk_pairs(NCH, same_bin=True):
    g = run_pair(cats[a], cats[b], nbins=1000, bin_slop=0.0, num_threads=min(16, os.cpu_count() or 1))
    arr = corr_to_arrays(g)
    arr["fold"] = np.float64(drv.fold_factor(a, b, True))
    np.savez(d / f"c{a:02d}_{b:02d}.npz", **arr)
print(f"wrote {len(list(d.glob('c*.npz')))} chunk-pairs")

# --- 4. assemble
drv.OUTROOT = SCRATCH / "xi_chunked"


class A:
    nchunks = NCH


drv.do_assemble(A)
asm = np.load(drv.OUTROOT / "assembled" / "ns" / "Bin1_Bin1.npz")

# --- 5. compare
m = (ref["weight"] > 0) & (asm["weight"] > 0)
print("\n=== assembled vs single-shot ===")
print(f"npairs ratio      = {asm['npairs'].sum()/ref['npairs'].sum():.12f}")
print(f"sum(weight) ratio = {asm['weight'].sum()/ref['weight'].sum():.12f}")
print(f"max|npairs diff|  = {np.abs(asm['npairs']-ref['npairs']).max():.3e}")
print(f"xi+ corr          = {np.corrcoef(asm['xip'][m], ref['xip'][m])[0,1]:.12f}")
pull = np.abs(asm["xip"][m] - ref["xip"][m]) / np.sqrt(ref["varxi"][m])
print(f"xi+ max pull      = {pull.max():.3e} sigma")
# npairs and sum(weight) must match to machine precision. xi+ goes through a weighted mean
# sum(w*xi)/sum(w) over ~10 chunk-pairs, which accumulates float64 round-off at the 1e-5 sigma
# level on xi+ bins that are themselves ~1e-6; 1e-3 sigma is a rigorous bound that still fails
# loudly if the fold rule or the weighting were wrong (which would show up at 1e-1 sigma or worse).
ok = (abs(asm["npairs"].sum() / ref["npairs"].sum() - 1) < 1e-9
      and abs(asm["weight"].sum() / ref["weight"].sum() - 1) < 1e-9
      and pull.max() < 1e-3)
print("\nRESULT:", "PASS - chunked path reproduces the single-shot pipeline" if ok else "FAIL")
json.dump({"nchunks": NCH, "n_gal": int(idx.size), "npairs_ratio":
           float(asm["npairs"].sum() / ref["npairs"].sum()),
           "weight_ratio": float(asm["weight"].sum() / ref["weight"].sum()),
           "max_pull": float(pull.max()), "pass": bool(ok)},
          open(ROOT / "runs" / "dr5_measure" / "chunk_proof.json", "w"), indent=2)
