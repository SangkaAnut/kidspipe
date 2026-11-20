"""Integrity check of the float64 re-extraction (RESUME bug 15) before any measurement uses it.

new  = ~/kids_dr5_work/bin<b-1>.npz          (float64, this re-extraction)
old  = ~/kids_dr5_work/f32_backup/bin<b-1>.npz (float32, what every earlier xi used)
ext  = earlier pipeline's float64 extract (READ-ONLY dir; ra/dec float64, e1/e2/weight float32)

Exit status 0 only if: same galaxies in the same order; ra/dec bitwise equal to ext; |new-old| ra/dec within float32
rounding; e1/e2/w equal to old after float32 rounding.  Writes runs/dr5_measure/xi_unordered/extract_check_f64.json.
"""
import json, sys
from pathlib import Path
import numpy as np

W = Path.home() / "kids_dr5_work"
EXT = Path("/mnt/d/HI-gamma-correlation/pipeline/00_extracts/outputs")
OUT = Path("/mnt/d/cloning_kids_Kappa_pipeline/runs/dr5_measure/xi_unordered/extract_check_f64.json")
rep, ok = {}, True
for b in range(1, 7):
    new, old, ext = np.load(W / f"bin{b-1}.npz"), np.load(W / "f32_backup" / f"bin{b-1}.npz"), np.load(EXT / f"bin{b}.npz")
    r = {"n_new": int(new["ra"].size), "n_old": int(old["ra"].size), "n_ext": int(ext["ra"].size),
         "dtypes_new": {k: str(new[k].dtype) for k in ("ra", "dec", "e1", "e2", "w")}}
    same_n = r["n_new"] == r["n_old"] == r["n_ext"]
    r["ra_dec_equal_ext"] = bool(same_n and np.array_equal(new["ra"], ext["ra"]) and np.array_equal(new["dec"], ext["dec"]))
    r["max_dra_vs_old_deg"] = float(np.abs(new["ra"] - old["ra"].astype(np.float64)).max()) if same_n else None
    r["max_ddec_vs_old_deg"] = float(np.abs(new["dec"] - old["dec"].astype(np.float64)).max()) if same_n else None
    r["radec_old_is_f32_rounding"] = bool(same_n and np.array_equal(new["ra"].astype(np.float32), old["ra"])
                                          and np.array_equal(new["dec"].astype(np.float32), old["dec"]))
    r["e_w_old_is_f32_rounding"] = bool(same_n and all(np.array_equal(new[k].astype(np.float32), old[k]) for k in ("e1", "e2", "w")))
    r["north_equal"] = bool(same_n and np.array_equal(new["north"], old["north"]))
    r["ok"] = bool(same_n and r["ra_dec_equal_ext"] and r["radec_old_is_f32_rounding"] and r["e_w_old_is_f32_rounding"]
                   and r["north_equal"] and r["dtypes_new"]["ra"] == "float64")
    ok &= r["ok"]; rep[f"bin{b}"] = r
    print(f"bin{b}: n={r['n_new']:,} ra/dec==ext {r['ra_dec_equal_ext']} f32-rounding(radec,e/w) "
          f"{r['radec_old_is_f32_rounding']},{r['e_w_old_is_f32_rounding']} max dra {r['max_dra_vs_old_deg']:.2e} -> {'OK' if r['ok'] else 'FAIL'}", flush=True)
rep["all_ok"] = bool(ok)
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(rep, indent=1))
print("ALL_OK" if ok else "CHECK_FAILED", flush=True)
sys.exit(0 if ok else 1)
