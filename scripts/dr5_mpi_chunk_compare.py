"""Compare chunk files of the chunked xi driver written in two runs (e.g. under MPI vs the original serial run).

    python scripts/dr5_mpi_chunk_compare.py NEW_DIR REF_DIR [--out compare.json]

For every c*.npz in NEW_DIR: same keys, non-float entries equal, and per float key the max relative and absolute
difference. Relative differences of keys whose reference values are pure round-off (|ref| max < 1e-15, e.g. the
imaginary parts of an auto pair) are reported but flagged `roundoff_only`.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(new: Path, ref: Path) -> dict:
    out = {}
    for f in sorted(new.glob("c*.npz")):
        a, b = np.load(f), np.load(ref / f.name)
        if sorted(a.files) != sorted(b.files):
            out[f.stem] = {"keys_equal": False}
            continue
        rec = {"keys_equal": True, "bit_identical": True, "keys": {}}
        for k in a.files:
            x, y = a[k], b[k]
            if x.dtype.kind not in "fc" or x.size == 0:
                eq = bool(np.array_equal(x, y))
                rec["keys"][k] = {"equal": eq}
                rec["bit_identical"] &= eq
                continue
            d = np.abs(x - y)
            rel = np.where(d > 0, d / np.maximum(np.abs(y), 1e-300), 0.0)
            rec["keys"][k] = {"max_rel": float(rel.max()), "max_abs": float(d.max()), "max_ref": float(np.abs(y).max()),
                              "roundoff_only": bool(np.abs(y).max() < 1e-15)}
            rec["bit_identical"] &= bool(np.array_equal(x, y))
        phys = [v["max_rel"] for v in rec["keys"].values() if "max_rel" in v and not v["roundoff_only"]]
        rec["max_rel_physical"] = max(phys) if phys else 0.0
        out[f.stem] = rec
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = compare(a.new, a.ref)
    for c, r in res.items():
        print(c, "keys differ" if not r["keys_equal"] else
              ("bit-identical" if r["bit_identical"] else f"max rel (physical keys) {r['max_rel_physical']:.2e}"))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
