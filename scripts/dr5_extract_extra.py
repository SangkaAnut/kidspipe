#!/usr/bin/env python3
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Diagnostic (read-only) extraction of extra catalogue columns, to pin down which selection
cut makes our pair counts match the official DR5 measurement.

Writes to ~/kids_dr5_work/extra/bin{0..5}.npz -- a SEPARATE tree, nothing existing is
overwritten -- and prints per-bin summary statistics for candidate cuts.

Why this is needed: our xi+/- shape matches the official ascii (r = 0.999 for xi_+), but our
pair counts are a constant 1.417x larger (per-bin spread 9.4e-4).  A constant factor with an
unchanged shape means our catalogue has ~sqrt(1.417) = 1.19x too many galaxies, i.e. the
official run applied a cut we have not.  This script records the columns needed to test the
candidates (star/galaxy classification, S/N, number of exposures, magnitude, Z_B).

Read-only with respect to the data release.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402

import numpy as np

CAT = paths.dr5_catalogue()
OUT = paths.dr5_work_dir() / "extra"
COLS = ["TOMOBIN", "Z_B", "SG2DPHOT", "FLUX_RADIUS", "model_SNratio", "pixel_SNratio",
        "n_exposures_used", "MAG_AUTO", "weight", "shear_weight_only", "gold_weight_only",
        "2D_measurement_variance"]
CHUNK = 5_000_000
NBIN = 6


def main() -> None:
    import fitsio

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = fitsio.FITS(str(CAT))
    nrows = f[1].get_nrows()
    print(f"rows={nrows:,}", flush=True)

    acc = {b: {c: [] for c in COLS} for b in range(1, NBIN + 1)}
    for start in range(0, nrows, CHUNK):
        stop = min(start + CHUNK, nrows)
        d = f[1].read(rows=np.arange(start, stop), columns=COLS)
        tomo = d["TOMOBIN"].astype(np.int16)
        for b in range(1, NBIN + 1):
            m = tomo == b
            if not m.any():
                continue
            for c in COLS:
                acc[b][c].append(d[c][m])
        print(f"  read {stop:,}/{nrows:,}  {time.time()-t0:.0f}s", flush=True)

    info = {"catalogue": str(CAT), "columns": COLS, "bins": {}}
    for b in range(1, NBIN + 1):
        cols = {c: np.concatenate(acc[b][c]) for c in COLS}
        np.savez(OUT / f"bin{b-1}.npz", **cols)
        n = cols["TOMOBIN"].size
        sg = cols["SG2DPHOT"]
        info["bins"][str(b)] = {
            "n": int(n),
            "frac_SG2DPHOT_eq1": float(np.mean(sg == 1)),
            "Z_B": {q: float(np.quantile(cols["Z_B"], q)) for q in (0.001, 0.01, 0.5, 0.99, 0.999)},
            "Z_B_nan": int(np.sum(~np.isfinite(cols["Z_B"]))),
            "FLUX_RADIUS": {q: float(np.quantile(cols["FLUX_RADIUS"], q)) for q in (0.01, 0.5, 0.99)},
            "model_SNratio": {q: float(np.quantile(cols["model_SNratio"], q)) for q in (0.01, 0.05, 0.5)},
            "n_exposures_used": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(cols["n_exposures_used"], return_counts=True))},
            "weight_gt0": int(np.sum(cols["weight"] > 0)),
            "shear_weight_gt0": int(np.sum(cols["shear_weight_only"] > 0)),
            "gold_weight_gt0": int(np.sum(cols["gold_weight_only"] > 0)),
            "weight_eq_shearw_frac": float(np.mean(np.isclose(cols["weight"], cols["shear_weight_only"]))),
        }
        print(f"bin{b}: n={n:,} SG2DPHOT==1 {info['bins'][str(b)]['frac_SG2DPHOT_eq1']:.5f} "
              f"weight==shearw {info['bins'][str(b)]['weight_eq_shearw_frac']:.3f}", flush=True)
        del cols

    (OUT / "extra_info.json").write_text(json.dumps(info, indent=2))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
