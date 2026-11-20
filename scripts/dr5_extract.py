#!/usr/bin/env python3
"""Step 0: extract the KiDS-Legacy shear catalogue into compact per-bin NPZ files.

Reads only the 7 columns we need from the 16.4 GB FITS binary table, in chunks,
and writes ONE npz per tomographic bin on the Linux filesystem (~/kids_dr5_work).

Also records the selection diagnostics needed for the n_eff / sigma_e gate.

Usage:
  PYTHONPATH=src /home/anut/anaconda3/bin/python3 scripts/dr5_extract.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

CAT = Path("/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_NS_unblind_final.fits")
OUT = Path.home() / "kids_dr5_work"
COLS = ["RAJ2000", "DECJ2000", "PATCH", "TOMOBIN", "e1", "e2", "weight"]
CHUNK = 5_000_000
NBIN = 6


def main() -> None:
    import fitsio

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = fitsio.FITS(str(CAT))
    nrows = f[1].get_nrows()
    print(f"rows={nrows:,}", flush=True)

    # accumulate per bin; 40.9M x ~20 B ~ 0.8 GB total -- fine
    ra = [[] for _ in range(NBIN)]
    dec = [[] for _ in range(NBIN)]
    e1 = [[] for _ in range(NBIN)]
    e2 = [[] for _ in range(NBIN)]
    w = [[] for _ in range(NBIN)]
    north = [[] for _ in range(NBIN)]  # bool: True == North

    tomo_hist = np.zeros(16, dtype=np.int64)
    n_bad_tomo = 0

    for start in range(0, nrows, CHUNK):
        stop = min(start + CHUNK, nrows)
        d = f[1].read(rows=np.arange(start, stop), columns=COLS)
        tomo = d["TOMOBIN"].astype(np.int16)
        tomo_hist += np.bincount(tomo[tomo >= 0], minlength=16)[:16]
        sel = (tomo >= 1) & (tomo <= NBIN)
        n_bad_tomo += int((~sel).sum())
        for b in range(1, NBIN + 1):
            m = tomo == b
            if not m.any():
                continue
            ra[b - 1].append(d["RAJ2000"][m])
            dec[b - 1].append(d["DECJ2000"][m])
            e1[b - 1].append(d["e1"][m])
            e2[b - 1].append(d["e2"][m])
            w[b - 1].append(d["weight"][m])
            patch = d["PATCH"][m]
            north[b - 1].append(np.char.startswith(patch.astype("U8"), "N"))
        print(f"  read {stop:,}/{nrows:,}  {time.time()-t0:.0f}s", flush=True)

    info = {
        "catalogue": str(CAT),
        "nrows_total": int(nrows),
        "nrows_used": int(tomo_hist[1:NBIN + 1].sum()),
        "nrows_rejected_bad_tomobin": int(n_bad_tomo),
        "tomobin_hist": {int(k): int(v) for k, v in enumerate(tomo_hist) if v},
        "selection": "all rows with 1 <= TOMOBIN <= 6; no further cuts "
                     "(e1,e2 are c-corrected/recalibrated, weight is the final "
                     "shear+gold weight as shipped)",
        "columns": COLS,
        "bins": {},
    }

    for b in range(NBIN):
        r = np.concatenate(ra[b]).astype(np.float64)
        dd = np.concatenate(dec[b]).astype(np.float64)
        # keep the catalogue's float64: float32 ra/dec (0.055" steps) moved fine-bin xi by ~0.1 sigma (RESUME bug 15)
        ee1 = np.concatenate(e1[b]).astype(np.float64)
        ee2 = np.concatenate(e2[b]).astype(np.float64)
        ww = np.concatenate(w[b]).astype(np.float64)
        nn = np.concatenate(north[b])

        good = np.isfinite(r) & np.isfinite(dd) & np.isfinite(ee1) & np.isfinite(ee2) & np.isfinite(ww) & (ww > 0)
        n_drop = int((~good).sum())
        r, dd, ee1, ee2, ww, nn = r[good], dd[good], ee1[good], ee2[good], ww[good], nn[good]

        sw = float(np.sum(ww, dtype=np.float64))
        sw2 = float(np.sum(ww.astype(np.float64) ** 2))
        # ellipticity dispersion: weighted rms of sqrt(e1^2+e2^2)/sqrt(2) per component
        e2sum = ee1.astype(np.float64) ** 2 + ee2.astype(np.float64) ** 2
        sig_e = float(np.sqrt(np.sum(ww.astype(np.float64) * e2sum) / sw / 2.0))

        np.savez(
            OUT / f"bin{b}.npz",
            ra=r, dec=dd,
            e1=ee1, e2=ee2, w=ww, north=nn,
        )
        info["bins"][str(b)] = {
            "n_gal": int(r.size),
            "n_dropped_nonfinite_or_nonpos_weight": n_drop,
            "n_north": int(nn.sum()),
            "n_south": int((~nn).sum()),
            "sum_w": sw,
            "sum_w2": sw2,
            "n_eff": sw ** 2 / sw2,
            "sigma_e": sig_e,
        }
        print(f"bin{b}: n={r.size:,} n_eff={sw**2/sw2:.3e} sigma_e={sig_e:.4f} "
              f"dropped={n_drop}", flush=True)
        del r, dd, ee1, ee2, ww, nn

    (OUT / "extract_info.json").write_text(json.dumps(info, indent=2))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
