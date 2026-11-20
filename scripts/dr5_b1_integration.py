"""Check B1: integrate the *official* xi+/- ascii into En/Bn and compare with the official
COSEBIs FITS.  This isolates the xi -> E_n integration from any measurement error and also
decides the integration convention (theta column, weight scheme, length unit).

Writes runs/dr5_measure/b1_integration.json
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidspipe.measure.official import (PAIRS, cov_e_block, load_official_cosebis, load_official_xi)
from kidspipe.theory.cosebis import CosebisFilters
from kidspipe.measure.xi2cosebis import e_from_xi_binned

OUT = Path(__file__).resolve().parents[1] / "runs" / "dr5_measure"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xi = load_official_xi()
    off = load_official_cosebis()
    filt = CosebisFilters(tmin=2.0, tmax=300.0, nmax=20, source="mpmath")
    sig20 = np.sqrt(np.diag(cov_e_block(off["cov"], 20)))
    sig6 = np.sqrt(np.diag(cov_e_block(off["cov"], 6)))

    trials = {}
    for col, scheme, unit in itertools.product(["meanr", "r_nom"], ["edges", "centre", "trapz"],
                                               ["arcmin", "rad"]):
        E, B = e_from_xi_binned(xi[col][0], xi["xip"], xi["xim"], filt, nmodes=20,
                                scheme=scheme, unit=unit)
        rE = (E - off["En"]).ravel()
        rB = (B - off["Bn"]).ravel()
        chi2_E = float(np.einsum("i,ij,j->", rE, np.linalg.inv(cov_e_block(off["cov"], 20)), rE))
        d = (E - off["En"]) / sig20.reshape(E.shape)
        r6 = (E[:, :6] - off["En"][:, :6]).ravel()
        chi2_6 = float(np.einsum("i,ij,j->", r6, np.linalg.inv(cov_e_block(off["cov"], 6)), r6))
        trials[f"{col}|{scheme}|{unit}"] = {
            "chi2_Emode_420_nomod": chi2_E / 420.0,
            "chi2_Emode_126_nomod": chi2_6 / 126.0,
            "max_abs_dE_over_sigma": float(np.abs(d).max()),
            "median_abs_dE_over_sigma": float(np.median(np.abs(d))),
            "B_vs_official_rms_over_sigma": float(np.sqrt(np.mean((rB / sig20) ** 2))),
            "En_rel_scale_ratio": float(np.sum(E * off["En"]) / np.sum(off["En"] ** 2)),
            "runtime_s": None,
        }
        print(f"{col:6s} {scheme:7s} {unit:6s}  chi2/420={chi2_E/420:10.3f} "
              f"chi2/126={chi2_6/126:9.3f}  max|dE|/sig={np.abs(d).max():9.3f} "
              f"rmsB/sig={np.sqrt(np.mean((rB/sig20)**2)):9.3f}", flush=True)

    best = min(trials, key=lambda k: trials[k]["chi2_Emode_126_nomod"])
    res = {
        "what": "check B1: official xi+/- ascii -> E_n/B_n, compared with official COSEBIs FITS",
        "note": "all trials share the same filters (mpmath roots, 2-300 arcmin, 20 modes); "
                "they differ in theta column, bin weight scheme and length unit",
        "trials": trials,
        "best_by_chi2_modes1to6": best,
        "best": trials[best],
        "official_En_modes1to6_head": off["En"][:3, :6].tolist(),
        "runtime_s": time.time() - t0,
    }
    (OUT / "b1_integration.json").write_text(json.dumps(res, indent=2))
    print(f"\nbest: {best}  chi2/126 = {trials[best]['chi2_Emode_126_nomod']:.4f}")
    print(f"wrote {OUT/'b1_integration.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
