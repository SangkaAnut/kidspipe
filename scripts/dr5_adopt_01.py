"""ADOPT the validated work from pipeline/01_lensing_validation, verified through THIS project's code.

We do not copy their JSON. We read their measured xi+/- (21 pairs, 1000 log bins), push it through
our own COSEBIs integrator (kidspipe.measure.xi2cosebis, settled at chi2/126 = 0.098 against the
official E_n), and compare to the official release ourselves. Their results are then used only as a
cross-check of ours.

What is verified here, all from their measured xi:
  A1  = A1b-noise (adopted 2026-10-29): RAW xi vs the official fine ascii (a blinded realisation of the
        same sample, m-uncorrected, x~0.85), one scale per pair; see kidspipe.measure.a1check
  A3  xi rebinned to 9 bins vs the xipm FITS; PASS iff max |Delta|/sigma < 0.1
  B1  their xi -> our E_n/B_n vs the official COSEBIs FITS
  B3  B_n null: chi2 of B_n against zero with the official Bn covariance block, per pair and all
  plus the m-correction question, re-measured from their raw (uncorrected) xi

Sources:
  their xi   : pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin{i}_bin{j}.npz
  official   : the KiDS-Legacy release data/ files
Outputs: runs/dr5_measure/adoption/  (json + npz + log)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline")
sys.path.insert(0, str(ROOT / "src"))

from kidspipe.measure.official import (PAIRS, load_official_cosebis, load_official_xi,  # noqa: E402
                                       cov_e_block)
from kidspipe.measure.a1check import A3_MAX_ABS_OVER_SIGMA, a1b_noise, a3_pass  # noqa: E402
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402
from kidspipe.theory.cosebis import CosebisFilters  # noqa: E402

THEIRS = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
OFFICIAL = Path("/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/"
                "KiDS_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/data")
OUT = ROOT / "runs" / "dr5_measure" / "adoption"
OUT.mkdir(parents=True, exist_ok=True)

# m_gold per bin, Wright+25 Table 1
M = [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045]

res = {}
off_xi = load_official_xi()
off_cb = load_official_cosebis()

# ---------------------------------------------------------------- load their 21 pairs
their = {}
for k, (i, j) in enumerate(PAIRS):
    f = THEIRS / f"xipm_bin{i+1}_bin{j+1}.npz"
    if not f.exists():
        print(f"MISSING {f}")
        continue
    d = np.load(f)
    their[k] = {kk: np.asarray(d[kk], float) for kk in
                ("r_nom", "meanr", "xip", "xim", "xip_raw", "xim_raw", "npairs", "weight",
                 "sigma_xip", "sigma_xim")}
    their[k]["m_factor"] = float(d["m_factor"])
print(f"loaded {len(their)}/21 of their pairs")

# ---------------------------------------------------------------- A1 = A1b-noise (adopted)
print("\n=== A1 (A1b-noise): their RAW xi vs official fine ascii, one scale per pair ===")
SALV = ROOT / "runs" / "dr5_measure" / "a1_salvage" / "a1_salvage.json"
if not SALV.exists():
    raise SystemExit(f"A1 needs the bootstrap A_p errors from {SALV}: run scripts/dr5_a1_salvage.py first")
_amp = json.loads(SALV.read_text())["step2_amplitude"]
sd_frac = {k: np.asarray(_amp[f"{k}_raw"]["A_pair_err_frac"], float) for k in ("xip", "xim")}
traw_p = np.array([their[k]["xip_raw"] for k in range(21)])
traw_m = np.array([their[k]["xim_raw"] for k in range(21)])
a1 = a1b_noise(traw_p, traw_m, off_xi, sd_frac)
res["A1_A1b_noise"] = a1
for key in ("xip", "xim"):
    d = a1[key]
    print(f"  {key}: max boxcar/white={d['max_boxcar_over_white']:.3f} (<= {a1['thresholds']['boxcar_over_white_max']})  "
          f"const chi2={d['const_chi2']:.2f}/20 pte={d['const_pte']:.3f} (> {a1['thresholds']['const_pte_min']})  C={d['C']:.4f}")
print(f"  -> A1 {'PASS' if a1['pass'] else 'FAIL'}")

# ---------------------------------------------------------------- A3 coarse 9-bin vs FITS
print("\n=== A3: their xi rebinned to 9 bins vs official xipm FITS (378 pts) ===")
h = fits.open(OFFICIAL / "KiDS_Legacy_xipm.fits")
cov_xi = 0.5 * (h["COVMAT"].data + h["COVMAT"].data.T)
edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))


def rebin9(r, w, y):
    out = np.zeros(9)
    for c in range(9):
        m = (r >= edges[c]) & (r < (edges[c + 1] if c < 8 else 300.01))
        out[c] = np.sum(w[m] * y[m]) / np.sum(w[m]) if w[m].sum() else np.nan
    return out


for ext, key in (("xiP", "xip"), ("xiM", "xim")):
    tab = h[ext].data
    val = np.array(tab["VALUE"], float).reshape(21, 9)
    dvec = np.zeros(21 * 9)
    for k in sorted(their):
        t = their[k]
        dvec[k * 9:(k + 1) * 9] = rebin9(t["r_nom"], t["npairs"], t[key]) - val[k]
    # full [xiP | xiM] covariance, output per statistic
    n = 189
    blk = 0 if ext == "xiP" else 1
    idx = np.arange(blk * n, (blk + 1) * n)
    dsub = dvec[~np.isnan(dvec)]
    Cs = cov_xi[np.ix_(idx, idx)]
    keep = ~np.isnan(dvec)
    Cs = Cs[np.ix_(keep, keep)]
    chi2 = float(dsub @ np.linalg.solve(Cs, dsub))
    sig = np.sqrt(np.diag(Cs))
    res[f"A3_{ext}"] = {"chi2": chi2, "dof": int(keep.sum()),
                        "chi2_per_dof": chi2 / keep.sum(),
                        "max_abs_over_sigma": float(np.max(np.abs(dsub / sig))),
                        "median_abs_over_sigma": float(np.median(np.abs(dsub / sig))),
                        "threshold_max_abs_over_sigma": A3_MAX_ABS_OVER_SIGMA,
                        "pass": a3_pass(float(np.max(np.abs(dsub / sig))))}
    print(f"  {ext}: chi2={chi2:.4f} / dof={int(keep.sum())} = {chi2/keep.sum():.4f}   "
          f"max|d|/sig={np.max(np.abs(dsub/sig)):.4f}  median={np.median(np.abs(dsub/sig)):.4f}  "
          f"-> {'PASS' if res[f'A3_{ext}']['pass'] else 'FAIL'} (max|d|/sig < {A3_MAX_ABS_OVER_SIGMA})")

# ---------------------------------------------------------------- B1: their xi -> our E_n/B_n
print("\n=== B1: their xi -> OUR COSEBIs integrator vs official COSEBIs ===")
filt = CosebisFilters(2.0, 300.0, source="mpmath", dps=80)
rnom = their[sorted(their)[0]]["r_nom"]
xip = np.zeros((21, rnom.size))
xim = np.zeros((21, rnom.size))
for k in sorted(their):
    xip[k] = their[k]["xip"]
    xim[k] = their[k]["xim"]
E, B = e_from_xi_binned(rnom, xip, xim, filt, nmodes=20, scheme="edges", unit="rad")
res["B1_En_Bn_vs_official"] = {}
cov_cb = off_cb["cov"]
for nm, ours, off in (("En", E, off_cb["En"]), ("Bn", B, off_cb["Bn"])):
    d = (ours - off).ravel()
    for nmode in (6, 20):
        idx = np.array([p * 20 + mm for p in range(21) for mm in range(nmode)])
        Cs = cov_cb[np.ix_(idx, idx)]
        chi2 = float(d[idx] @ np.linalg.solve(Cs, d[idx]))
        sig = np.sqrt(np.diag(Cs))
        res["B1_En_Bn_vs_official"][f"{nm}_{nmode}m"] = {
            "chi2": chi2, "dof": len(idx), "chi2_per_dof": chi2 / len(idx),
            "max_abs_over_sigma": float(np.max(np.abs(d[idx] / sig)))}
        print(f"  {nm} {nmode:2d} modes: chi2={chi2:8.3f}/{len(idx):3d} = {chi2/len(idx):.4f}  "
              f"max|d|/sig={np.max(np.abs(d[idx]/sig)):.4f}")

# ---------------------------------------------------------------- B3: B_n null
print("\n=== B3: B_n null test vs zero (official Bn covariance block) ===")
res["B3_Bn_null"] = {}
for nmode in (6, 20):
    idx = np.array([p * 20 + mm for p in range(21) for mm in range(nmode)])
    # COVMAT is [En (420) | Bn (420)]; the B_n null test must use the Bn block, offset by 21*20.
    # Using the En block silently gives pte=0.945 instead of the published 0.0635.
    Cs = cov_cb[np.ix_(idx + 420, idx + 420)]
    boff = off_cb["Bn"].ravel()[idx]
    chi2_off = float(boff @ np.linalg.solve(Cs, boff))
    bours = B.ravel()[idx]
    chi2_our = float(bours @ np.linalg.solve(Cs, bours))
    try:
        from scipy.stats import chi2 as chi2dist
        pte_off = float(chi2dist.sf(chi2_off, len(idx)))
        pte_our = float(chi2dist.sf(chi2_our, len(idx)))
    except Exception:
        pte_off = pte_our = float("nan")
    res["B3_Bn_null"][f"{nmode}m"] = {"chi2_official": chi2_off, "dof": len(idx),
                                      "pte_official": pte_off,
                                      "chi2_ours": chi2_our, "pte_ours": pte_our}
    print(f"  {nmode:2d} modes: official chi2={chi2_off:8.3f} pte={pte_off:.4f} | "
          f"ours chi2={chi2_our:8.3f} pte={pte_our:.4f}")

# ---------------------------------------------------------------- m-correction
print("\n=== m-correction: their m-corrected vs raw xi ===")
res["m_correction"] = {}
xip_raw = np.zeros_like(xip)
xim_raw = np.zeros_like(xim)
for k in sorted(their):
    xip_raw[k] = their[k]["xip_raw"]
    xim_raw[k] = their[k]["xim_raw"]
E_r, B_r = e_from_xi_binned(rnom, xip_raw, xim_raw, filt, nmodes=20, scheme="edges", unit="rad")
idx = np.array([p * 20 + mm for p in range(21) for mm in range(6)])
Cs = cov_cb[np.ix_(idx, idx)]
for label, arr in (("m-corrected", E), ("raw (no m)", E_r)):
    d = (arr - off_cb["En"]).ravel()[idx]
    c = float(d @ np.linalg.solve(Cs, d))
    res["m_correction"][label] = {"En_chi2_6m": c, "chi2_per_dof": c / len(idx)}
    print(f"  En {label:12s}: chi2={c:8.3f}/{len(idx)} = {c/len(idx):.4f}")

# ---------------------------------------------------------------- save
np.savez(OUT / "adopted_cosebis.npz", En=E, Bn=B, r_nom=rnom, xip=xip, xim=xim,
         En_official=off_cb["En"], Bn_official=off_cb["Bn"])
res["provenance"] = {
    "their_xi": str(THEIRS),
    "our_integrator": "kidspipe.measure.xi2cosebis.e_from_xi_binned, scheme=edges, unit=rad",
    "official": str(OFFICIAL),
    "n_pairs_loaded": len(their),
    "note": ("their npairs/weight are UNORDERED-pair counts (gg.process(cat)); do not compare "
             "raw pair counts across projects. xi values are convention-independent here."),
}
(OUT / "adoption.json").write_text(json.dumps(res, indent=2, default=float))
print(f"\nwrote {OUT/'adoption.json'}")
