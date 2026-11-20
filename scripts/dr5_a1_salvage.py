# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""A1 salvage: can 'their measured xi reproduces the official fine ascii' be made a meaningful check?

Everything is computed here from files; nothing is copied from earlier notes.  Outputs go to
runs/dr5_measure/a1_salvage/ (a1_salvage.json + PNGs).  Read-only on /mnt/d/HI-gamma-correlation.
Run:  cd /mnt/d/cloning_kids_Kappa_pipeline && PYTHONSAFEPATH= PYTHONPATH=src OMP_NUM_THREADS=4 \
      /home/anut/anaconda3/bin/python3 scripts/dr5_a1_salvage.py

Conventions (fixed, do not mix):
  their = prior project's m-corrected xi (1000 log bins);  their_raw = same without m-correction.
  asc   = official fine ascii (cosmosis_xipm);  FITS = KiDS_Legacy_xipm.fits (9 bins).
  A_p   = their / asc  (>1), per pair, ratio of npairs_weighted-weighted sums over all 1000 bins
          (a ratio of sums, NOT a least-squares slope: the slope is diluted by noise in asc).
  B_i   : A_p = B_i B_j.  s_i = 1/B_i = the ascii/their factor quoted by the prior project.
  Rebinning to 9 bins always uses kidspipe.measure.official.weighted_rebin with the ascii's
  npairs_weighted (== CosmoSIS bin_xi weighted_binning=1) unless stated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
from kidspipe.measure.official import (PAIRS, load_official_cosebis, load_official_xi,  # noqa: E402
                                       load_official_xipm_fits, weighted_rebin)
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402
from kidspipe.theory.cosebis import CosebisFilters  # noqa: E402

THEIRS = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
OUT = ROOT / "runs" / "dr5_measure" / "a1_salvage"
OUT.mkdir(parents=True, exist_ok=True)
M_GOLD = np.array([-0.023, -0.016, -0.011, 0.020, 0.030, 0.045])  # Wright+25 Table 1
PRIOR_S = np.array([0.893, 0.913, 0.917, 0.932, 0.949, 0.971])    # prior REPORT.md item 2 (ascii/FITS)
PROMPT_C = np.array([0.929, 0.953, 0.967, 0.968, 0.995, 1.012])   # PROMPT: log A = c_i+c_j+const
PROMPT_CONST = 0.913
RNG = np.random.default_rng(20261029)
res: dict = {}


def jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------ load
asc = load_official_xi()
fit = load_official_xipm_fits()
cov378 = fit["cov"]
th = asc["r_nom"][0]
w_asc = asc["npairs_weighted"]
T = {k: np.zeros((21, 1000)) for k in ("xip", "xim", "xip_raw", "xim_raw", "npairs", "weight")}
for k, (i, j) in enumerate(PAIRS):
    d = np.load(THEIRS / f"xipm_bin{i+1}_bin{j+1}.npz")
    assert np.max(np.abs(d["r_nom"] / th - 1)) < 1e-9
    for kk in T:
        T[kk][k] = d[kk]
pair_idx = np.array(PAIRS)
STAT = {"xip": (slice(0, 189), "xiP"), "xim": (slice(189, 378), "xiM")}


def r9(x, w=w_asc):
    return weighted_rebin(th, x, w, 9)[0]


def chi2_block(d, C):
    d = np.asarray(d).ravel()
    return float(d @ np.linalg.solve(C, d))


def block_stats(d, C, dof=None):
    d = np.asarray(d).ravel()
    n = d.size
    dof = n if dof is None else dof
    c = chi2_block(d, C)
    sig = np.sqrt(np.diag(C))
    return {"chi2": c, "dof": dof, "chi2_per_dof": c / dof, "pte": float(chi2dist.sf(c, dof)),
            "max_abs_over_sigma": float(np.max(np.abs(d / sig))),
            "diag_chi2_per_dof": float(np.sum((d / sig) ** 2) / n)}


# ===================================================================== STEP 1: reproduce
print("=== STEP 1: the three numbers ===")
s1 = {}
for key, (sl, ext) in STAT.items():
    C = cov378[sl, sl]
    F = fit[ext]
    a9 = r9(asc[key])
    t9 = r9(T[key])                         # same (ascii npairs_weighted) weights
    t9own = r9(T[key], T["npairs"])         # their own npairs weights (adopt_01 style)
    r = {
        "ascii_vs_FITS": {**block_stats(a9 - F, C), "median_ratio": float(np.median(a9 / F))},
        "their_vs_FITS_sameweights": {**block_stats(t9 - F, C), "median_ratio": float(np.median(t9 / F))},
        "their_vs_FITS_ownweights": block_stats(t9own - F, C),
        "their_vs_ascii_9bin": block_stats(t9 - a9, C),
    }
    s1[key] = r
    print(f"{ext}: ascii-vs-FITS chi2/dof={r['ascii_vs_FITS']['chi2_per_dof']:.4f} "
          f"(diag-only {r['ascii_vs_FITS']['diag_chi2_per_dof']:.3f}) max={r['ascii_vs_FITS']['max_abs_over_sigma']:.2f}sig "
          f"med ratio {r['ascii_vs_FITS']['median_ratio']:.3f} | their-vs-FITS "
          f"{r['their_vs_FITS_sameweights']['chi2_per_dof']:.5f} max {r['their_vs_FITS_sameweights']['max_abs_over_sigma']:.4f}")
a9_all = np.concatenate([r9(asc["xip"]).ravel(), r9(asc["xim"]).ravel()])
t9_all = np.concatenate([r9(T["xip"]).ravel(), r9(T["xim"]).ravel()])
F_all = np.concatenate([fit["xiP"].ravel(), fit["xiM"].ravel()])
s1["joint378"] = {"ascii_vs_FITS": block_stats(a9_all - F_all, cov378),
                  "their_vs_FITS": block_stats(t9_all - F_all, cov378)}
res["step1_reproduce"] = s1
print("joint378 ascii-vs-FITS:", s1["joint378"]["ascii_vs_FITS"]["chi2_per_dof"],
      " their-vs-FITS:", s1["joint378"]["their_vs_FITS"]["chi2_per_dof"])

# Why the full-covariance chi2 is tiny for a ~15 % offset: project the offset on the cov eigenmodes
for key, (sl, ext) in STAT.items():
    C = cov378[sl, sl]
    d = (r9(asc[key]) - fit[ext]).ravel()
    ev, U = np.linalg.eigh(C)
    proj = (U.T @ d) ** 2 / ev
    top = np.argsort(proj)[::-1][:5]
    res.setdefault("step1_offset_in_eigenmodes", {})[key] = {
        "chi2_total": float(proj.sum()),
        "top5_mode_chi2": proj[top], "top5_mode_rank_by_eigenvalue_desc": [int(len(ev) - 1 - t) for t in top],
        "chi2_in_top10_largest_variance_modes": float(proj[::-1][:10].sum()),
        "cov_condition_number": float(ev[-1] / ev[0])}

# --- their vs ascii at fine bins: per-pair scale, residual structure
def ratio_of_sums(t, a, g):
    return (g * t).sum(-1) / (g * a).sum(-1)


fine = {}
for key in ("xip", "xim"):
    A = ratio_of_sums(T[key], asc[key], w_asc)
    Araw = ratio_of_sums(T[key + "_raw"], asc[key], w_asc)
    sg = asc["sigma_" + key]
    z = (T[key] - A[:, None] * asc[key]) / sg
    zraw = (T[key + "_raw"] - Araw[:, None] * asc[key]) / sg
    lag1 = np.array([np.corrcoef(zz[:-1], zz[1:])[0, 1] for zz in z])
    lag5 = np.array([np.corrcoef(zz[:-5], zz[5:])[0, 1] for zz in z])
    # same residual after the 20-fine-bin boxcar average: does it average down like noise?
    zb = z.reshape(21, 50, 20).mean(-1)
    corr = np.array([np.corrcoef(T[key][k], asc[key][k])[0, 1] for k in range(21)])
    sigratio = np.array([np.median(np.load(THEIRS / f"xipm_bin{i+1}_bin{j+1}.npz")["sigma_xip"] / asc["sigma_xip"][k])
                         for k, (i, j) in enumerate(PAIRS)])
    fine[key] = {"A_their_over_ascii": A, "A_raw_over_ascii": Araw,
                 "rms_z_fine": z.std(1), "rms_z_fine_rawfit": zraw.std(1),
                 "lag1_autocorr_z": lag1, "lag5_autocorr_z": lag5,
                 "rms_z_after_20bin_boxcar": zb.std(1), "white_expectation_boxcar": 1 / np.sqrt(20),
                 "pearson_corr_fine": corr, "sigma_their_over_sigma_ascii_median": sigratio}
    print(f"{key}: corr {corr.min():.4f}/{np.median(corr):.4f}/{corr.max():.4f}  "
          f"rms z {z.std(1).mean():.3f}  lag1 {lag1.mean():.3f}  lag5 {lag5.mean():.3f}  "
          f"boxcar20 rms z {zb.std(1).mean():.3f} (white would be {z.std(1).mean()/np.sqrt(20):.3f})")
res["step1_fine_residual"] = fine

# ===================================================================== STEP 2: amplitude fit
print("\n=== STEP 2: amplitude hypothesis (1000 fine bins, moving-block bootstrap) ===")
BLK, NREP = 25, 600


def boot_idx():
    st = RNG.integers(0, 1000, size=40)
    return ((st[:, None] + np.arange(BLK)[None, :]) % 1000).ravel()[:1000]


# design for log A_p = b_i + b_j
X = np.zeros((21, 6))
for p, (i, j) in enumerate(PAIRS):
    X[p, i] += 1
    X[p, j] += 1


def sep_fit(logA, wgt):
    W = np.sqrt(wgt)
    b, *_ = np.linalg.lstsq(X * W[:, None], logA * W, rcond=None)
    return b


a2 = {}
for key in ("xip", "xim"):
    for tag, tk in (("corr", key), ("raw", key + "_raw")):
        A0 = ratio_of_sums(T[tk], asc[key], w_asc)
        reps = np.zeros((NREP, 21))
        for r in range(NREP):
            ix = boot_idx()
            reps[r] = ratio_of_sums(T[tk][:, ix], asc[key][:, ix], w_asc[:, ix])
        # positive-A replicates only (xim pair 1-1 can have a near-zero denominator)
        lreps = np.log(np.where(reps > 0, reps, np.nan))
        sdl = np.nanstd(lreps, axis=0)
        wgt = 1 / sdl ** 2
        b0 = sep_fit(np.log(A0), wgt)
        bb = np.array([sep_fit(np.where(np.isfinite(lreps[r]), lreps[r], np.log(A0)), wgt) for r in range(NREP)])
        resid = (np.log(A0) - X @ b0) / sdl
        chi2s = float(np.sum(resid ** 2))
        a2[f"{key}_{tag}"] = {
            "A_pair": A0, "A_pair_err_frac": sdl, "B_i": np.exp(b0), "B_i_err": np.exp(b0) * bb.std(0),
            "s_i_ascii_over_their": np.exp(-b0), "separable_chi2": chi2s, "separable_dof": 15,
            "separable_pte": float(chi2dist.sf(chi2s, 15)),
            "const_only_chi2": float(np.sum(((np.log(A0) - np.sum(np.log(A0) * wgt) / wgt.sum()) / sdl) ** 2)),
            "const_only_value": float(np.exp(np.sum(np.log(A0) * wgt) / wgt.sum())), "const_only_dof": 20,
            "boot_cov_logB": np.cov(bb.T)}
        print(f"{key:4s} {tag:4s}: B_i = {np.round(np.exp(b0), 4)}  +/- {np.round(np.exp(b0) * bb.std(0), 4)}  "
              f"sep chi2 {chi2s:.1f}/15  const-only chi2 {a2[f'{key}_{tag}']['const_only_chi2']:.1f}/20")
# ascii = k * (m-UNcorrected xi)?  Then A_raw = their_raw/ascii is the same constant for every pair and theta,
# and A_corr = A_raw/((1+m_i)(1+m_j)).  Test the one-parameter model per statistic (bootstrap errors).
def const_fit(A0, reps):
    lre = np.log(np.where(reps > 0, reps, np.nan))
    sd = np.nanstd(lre, axis=0)
    wg = 1 / sd ** 2
    lc = np.sum(np.log(A0) * wg) / wg.sum()
    lcr = np.array([np.sum(np.where(np.isfinite(l), l, np.log(A0)) * wg) / wg.sum() for l in lre])
    ch = float(np.sum(((np.log(A0) - lc) / sd) ** 2))
    return float(np.exp(lc)), float(np.exp(lc) * lcr.std()), ch


for key in ("xip", "xim"):
    A0 = ratio_of_sums(T[key + "_raw"], asc[key], w_asc)
    reps = np.array([ratio_of_sums(T[key + "_raw"][:, ix], asc[key][:, ix], w_asc[:, ix])
                     for ix in (boot_idx() for _ in range(NREP))])
    cval, cerr, ch = const_fit(A0, reps)
    a2[f"{key}_raw"]["global_const_model"] = {
        "description": "their_raw = C * ascii for ALL pairs and theta (i.e. ascii = (1/C) * m-uncorrected xi)",
        "C": cval, "C_err": cerr, "ascii_over_raw_k": 1 / cval, "chi2_over_21_pairs": ch, "dof": 20,
        "pte": float(chi2dist.sf(ch, 20))}
    print(f"{key}: their_raw = C*ascii, one global C={cval:.4f}+/-{cerr:.4f}  (ascii/raw={1/cval:.4f})  chi2={ch:.1f}/20 pte={chi2dist.sf(ch,20):.3f}")
    # theta dependence of C: 9 official theta ranges, pooled over pairs (block bootstrap inside each range)
    edg = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
    Cth, Cth_err = [], []
    for c in range(9):
        m_ = np.where((th >= edg[c]) & (th < edg[c + 1] if c < 8 else th <= edg[9]))[0]
        def pooled_T(ix):   # ratio of sums pooled over pairs: sum(g t)/sum(g a), g = npairs_weighted
            return (w_asc[:, ix] * T[key + "_raw"][:, ix]).sum() / (w_asc[:, ix] * asc[key][:, ix]).sum()
        c0 = pooled_T(m_)
        bs = []
        for _ in range(300):
            st = RNG.integers(0, m_.size, size=m_.size // 10 + 1)
            ix = m_[((st[:, None] + np.arange(10)[None, :]) % m_.size).ravel()[:m_.size]]
            bs.append(pooled_T(ix))
        Cth.append(c0); Cth_err.append(float(np.std(bs)))
    a2[f"{key}_raw"]["C_vs_theta_pooled"] = {"C": np.array(Cth), "err": np.array(Cth_err),
                                             "theta_lo": edg[:-1], "theta_hi": edg[1:]}
    print(f"   C(theta) pooled over pairs, 9 ranges: {np.round(Cth, 3)} +/- {np.round(Cth_err, 3)}")

# xi+ vs xi- equality of B_i
bp, bm = a2["xip_corr"], a2["xim_corr"]
dB = (np.log(bp["B_i"]) - np.log(bm["B_i"]))
# xi+ and xi- share the galaxies -> bootstrap jointly would be ideal; here use quadrature (conservative if
# errors positively correlated would be wrong sign; we state this in the report)
eB = np.sqrt((bp["B_i_err"] / bp["B_i"]) ** 2 + (bm["B_i_err"] / bm["B_i"]) ** 2)
a2["xip_vs_xim_logB_diff"] = {"diff": dB, "err_quadrature": eB, "pull": dB / eB, "chi2_6": float(np.sum((dB / eB) ** 2))}
print("xi+ vs xi- ln(B) pull:", np.round(dB / eB, 2), " chi2/6 =", np.sum((dB / eB) ** 2).round(2))

# --- 9-bin GLS fits with the OFFICIAL covariance (same instrument as the prompt's numbers)
a2["gls9"] = {}
for key, (sl, ext) in STAT.items():
    C = cov378[sl, sl]
    a9 = r9(asc[key]); t9 = r9(T[key]); F = fit[ext]
    for lab, data in (("FITS_over_ascii", F), ("their_over_ascii", t9)):
        Lc = np.linalg.cholesky(np.linalg.inv(C))
        mp = a9.ravel()
        pi = np.repeat(np.arange(21), 9)

        def f(lb, data=data):
            bb_ = np.exp(lb)
            return Lc.T @ (data.ravel() - bb_[pair_idx[pi, 0]] * bb_[pair_idx[pi, 1]] * mp)
        sol = least_squares(f, np.zeros(6), xtol=1e-12, ftol=1e-12)
        bb_ = np.exp(sol.x)
        err = bb_ * np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac)))
        # per-pair free scale (linear GLS) for the A1b chi2
        Xp = np.zeros((189, 21))
        Xp[np.arange(189), pi] = mp
        Ci = np.linalg.inv(C)
        Ap = np.linalg.solve(Xp.T @ Ci @ Xp, Xp.T @ Ci @ data.ravel())
        rr = data.ravel() - Xp @ Ap
        c_free = float(rr @ Ci @ rr)
        c_sep = float(np.sum(sol.fun ** 2))
        c_none = chi2_block(data - a9, C)
        a2["gls9"][f"{key}_{lab}"] = {"B_i": bb_, "B_i_err_cov_dominated": err, "s_i": 1 / bb_,
                                      "chi2_no_scale": c_none, "chi2_separable6": c_sep, "dof_sep": 189 - 6,
                                      "chi2_free21": c_free, "dof_free": 189 - 21, "A_pair": Ap,
                                      "pte_free21": float(chi2dist.sf(c_free, 189 - 21))}
        print(f"GLS9 {key} {lab:17s}: B_i={np.round(bb_,4)}  chi2 none={c_none:.1f}  sep6={c_sep:.1f}/183  free21={c_free:.1f}/168")
a2["reference_values"] = {
    "prior_project_s_i_ascii_over_FITS": PRIOR_S, "prior_project_B_i_equiv": 1 / PRIOR_S,
    "prompt_c_i": PROMPT_C, "prompt_const": PROMPT_CONST,
    "prompt_converted_s_i": PROMPT_C * np.sqrt(PROMPT_CONST),
    "prompt_converted_B_i": 1 / (PROMPT_C * np.sqrt(PROMPT_CONST)),
    "note": "prompt's log A=c_i+c_j+const with A=ascii/FITS => s_i=c_i*sqrt(const) when const is split symmetrically"}
res["step2_amplitude"] = a2

# ===================================================================== STEP 3: COSEBIs of the ascii
print("\n=== STEP 3: ascii and their-xi through OUR COSEBIs integrator ===")
cb = load_official_cosebis()
filt = CosebisFilters(2.0, 300.0, source="mpmath", dps=80)
NM = 20


def to_cb(xp, xm):
    return e_from_xi_binned(th, xp, xm, filt, nmodes=NM, scheme="edges", unit="rad")


def cb_chi2(x, off, ofs, nmode):
    idx = np.array([p * NM + m for p in range(21) for m in range(nmode)]) + ofs
    C = cb["cov"][np.ix_(idx, idx)]
    d = (x - off).ravel()[idx - ofs]
    c = float(d @ np.linalg.solve(C, d))
    return {"chi2": c, "dof": len(idx), "chi2_per_dof": c / len(idx), "pte": float(chi2dist.sf(c, len(idx))),
            "max_abs_over_sigma": float(np.max(np.abs(d / np.sqrt(np.diag(C)))))}


E_a, B_a = to_cb(asc["xip"], asc["xim"])
E_t, B_t = to_cb(T["xip"], T["xim"])
A_xp = ratio_of_sums(T["xip"], asc["xip"], w_asc)
A_xm = ratio_of_sums(T["xim"], asc["xim"], w_asc)
E_as, B_as = to_cb(A_xp[:, None] * asc["xip"], A_xm[:, None] * asc["xim"])   # ascii with a per-pair scale
s3 = {}
for lab, E, B in (("ascii", E_a, B_a), ("their", E_t, B_t), ("ascii_perpair_scaled_to_their", E_as, B_as)):
    s3[lab] = {f"En_{n}m": cb_chi2(E, cb["En"], 0, n) for n in (6, 20)}
    s3[lab].update({f"Bn_{n}m": cb_chi2(B, cb["Bn"], 420, n) for n in (6, 20)})
    print(f"{lab:30s} En6 {s3[lab]['En_6m']['chi2_per_dof']:.4f} (max {s3[lab]['En_6m']['max_abs_over_sigma']:.2f}s)  "
          f"En20 {s3[lab]['En_20m']['chi2_per_dof']:.4f} (max {s3[lab]['En_20m']['max_abs_over_sigma']:.2f}s)  "
          f"Bn20 {s3[lab]['Bn_20m']['chi2_per_dof']:.4f}")
res["step3_cosebis"] = s3
res["step3_cosebis"]["ratio_En_ascii_over_official_median"] = float(np.median(E_a[:, :6] / cb["En"][:, :6]))
res["step3_cosebis"]["ratio_En_their_over_official_median"] = float(np.median(E_t[:, :6] / cb["En"][:, :6]))

# which theta range carries the ascii-vs-their difference?  (linear swap test, 9 official theta bins)
edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
swap = []
idx6 = np.array([p * NM + m for p in range(21) for m in range(6)])
C6 = cb["cov"][np.ix_(idx6, idx6)]
dE_tot = (E_a - E_t).ravel()[idx6]
for c in range(9):
    msk = (th >= edges[c]) & (th < edges[c + 1] if c < 8 else th <= edges[9])
    dp = np.where(msk[None, :], asc["xip"] - T["xip"], 0.0)
    dm = np.where(msk[None, :], asc["xim"] - T["xim"], 0.0)
    Ed, _ = to_cb(dp, dm)
    d = Ed.ravel()[idx6]
    swap.append({"theta_lo": float(edges[c]), "theta_hi": float(edges[c + 1]),
                 "chi2_En6_of_this_range_alone": float(d @ np.linalg.solve(C6, d)),
                 "diag_sum_over_total": float(np.sum(d ** 2 / np.diag(C6)) / np.sum(dE_tot ** 2 / np.diag(C6)))})
res["step3_swap_by_theta"] = swap
res["step3_swap_total_chi2_En6_ascii_minus_their"] = float(dE_tot @ np.linalg.solve(C6, dE_tot))
print("swap test (En 6 modes chi2 of the ascii-their difference, by theta range):",
      [round(s["chi2_En6_of_this_range_alone"], 2) for s in swap], "total", round(res["step3_swap_total_chi2_En6_ascii_minus_their"], 2))

# chi2 partition by theta for ascii-vs-FITS (C^-1 partition; sums to the total)
part = {}
for key, (sl, ext) in STAT.items():
    C = cov378[sl, sl]
    d = (r9(asc[key]) - fit[ext]).ravel()
    contrib = (d * np.linalg.solve(C, d)).reshape(21, 9).sum(0)
    part[key] = {"per_theta_bin": contrib, "total": float(contrib.sum()),
                 "diag_per_theta_bin": (((d / np.sqrt(np.diag(C))) ** 2).reshape(21, 9)).sum(0)}
res["step3_chi2_partition_ascii_vs_FITS"] = part

# ===================================================================== STEP 4: A1b + sensitivity
print("\n=== STEP 4: A1b (shape-only, amplitude-marginalised) and its power ===")
a1b = {}


def a1b_chi2(data9_p, data9_m, model9_p, model9_m, joint=True):
    """Fit one scale per pair (shared by xi+ and xi-) with the official joint 378x378 covariance."""
    d = np.concatenate([data9_p.ravel(), data9_m.ravel()])
    m = np.concatenate([model9_p.ravel(), model9_m.ravel()])
    Xj = np.zeros((378, 21))
    pi = np.repeat(np.arange(21), 9)
    Xj[np.arange(189), pi] = model9_p.ravel()
    Xj[189 + np.arange(189), pi] = model9_m.ravel()
    Ci = np.linalg.inv(cov378)
    Ap = np.linalg.solve(Xj.T @ Ci @ Xj, Xj.T @ Ci @ d)
    r = d - Xj @ Ap
    c = float(r @ Ci @ r)
    return c, Ap


tp9, tm9 = r9(T["xip"]), r9(T["xim"])
ap9, am9 = r9(asc["xip"]), r9(asc["xim"])
c_a, Aa = a1b_chi2(tp9, tm9, ap9, am9)
c_f, Af = a1b_chi2(fit["xiP"], fit["xiM"], ap9, am9)
c_tf, At = a1b_chi2(tp9, tm9, fit["xiP"], fit["xiM"])     # same fit, FITS as the model (== A3 with a free scale)
dof = 378 - 21
a1b["their_vs_ascii_free21"] = {"chi2": c_a, "dof": dof, "pte": float(chi2dist.sf(c_a, dof)), "A_pair": Aa}
a1b["FITS_vs_ascii_free21"] = {"chi2": c_f, "dof": dof, "pte": float(chi2dist.sf(c_f, dof)), "A_pair": Af}
a1b["their_vs_FITS_free21"] = {"chi2": c_tf, "dof": dof, "pte": float(chi2dist.sf(c_tf, dof)), "A_pair": At}
print(f"A1b their-vs-ascii chi2={c_a:.2f}/{dof}; FITS-vs-ascii {c_f:.2f}/{dof}; their-vs-FITS {c_tf:.4f}/{dof}")
# power: inject a multiplicative tilt eps*ln(theta/theta_mid)/ln(150/2)... on their xi+ and xi-, refit
lt = np.log(fit["theta"] / np.sqrt(2 * 300))
lt = lt / np.abs(lt).max()
pw = []
for eps in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    tt = (1 + eps * lt)[None, :]
    c1, _ = a1b_chi2(tp9 * tt, tm9 * tt, ap9, am9)
    c2, _ = a1b_chi2(tp9 * tt, tm9 * tt, fit["xiP"], fit["xiM"])
    # the same injected shape error against the plain A3 (no amplitude freedom, official cov)
    c3 = chi2_block(np.concatenate([(tp9 * tt - fit["xiP"]).ravel(), (tm9 * tt - fit["xiM"]).ravel()]), cov378)
    pw.append({"tilt_eps": eps, "A1b_vs_ascii_chi2": c1, "A1b_vs_FITS_chi2": c2, "A3_no_scale_chi2": c3,
               "A1b_vs_ascii_pte": float(chi2dist.sf(c1, dof)), "A1b_vs_FITS_pte": float(chi2dist.sf(c2, dof))})
a1b["sensitivity_to_injected_tilt"] = pw
# power to a pure ~15 % amplitude error: A3 (no scale) with an injected global scale
amp = []
for f_ in (1.0, 1.02, 1.05, 1.1, 1.15):
    c3 = chi2_block(np.concatenate([(f_ * tp9 - fit["xiP"]).ravel(), (f_ * tm9 - fit["xiM"]).ravel()]), cov378)
    amp.append({"scale": f_, "A3_chi2_378": c3, "pte": float(chi2dist.sf(c3, 378))})
a1b["A3_power_to_global_amplitude_error"] = amp
print("tilt power:", [(p["tilt_eps"], round(p["A1b_vs_ascii_chi2"], 1)) for p in pw])
print("A3 amplitude power:", [(p["scale"], round(p["A3_chi2_378"], 1)) for p in amp])
res["step4_A1b"] = a1b

# fine-bin A1b-strict: what residual noise level does same-sample identity predict?  Use the ascii's own
# shape-noise sigma.  Statistic: rms of z = (their - A_p*ascii)/sigma_ascii after 20-bin boxcar, and
# the scatter of A_p about the model (global constant on the raw xi).
res["step4_notes"] = {"covariance_caveat": (
    "The official 378x378 covariance includes cosmic variance, which cancels exactly when two estimators "
    "use the same galaxies. chi2 with it is therefore far less sensitive than the real noise of the "
    "difference; see step1_offset_in_eigenmodes and the sensitivity tables.")}

# ---- A1b-noise: the version of A1b with teeth.  their RAW xi vs ascii, one scale per pair (ratio of sums),
# then (i) no low-frequency excess in the residual, (ii) the per-pair scales are one global constant.
#   (i)  z = (their_raw - A_p ascii)/sigma_ascii on 1000 fine bins; boxcar-average 20 bins.  If the residual is
#        white the boxcar rms is rms(z)/sqrt(20).  Any shape mismatch that is coherent over >1 fine bin
#        (tilt, wrong weighting, wrong sample) adds variance above that.  PASS: boxcar_rms/white <= 1 on all 42 curves.
#   (ii) chi2 of ln A_p about one global constant, using the bootstrap errors measured on the real data
#        (dof 20); PASS: pte > 0.01 separately for xi+ and xi-.
from kidspipe.measure.a1check import A1B_THRESH, a1b_noise as _a1b_noise  # noqa: E402  (adopted gate, shared)
SD_FRAC = {k: a2[f"{k}_raw"]["A_pair_err_frac"] for k in ("xip", "xim")}


def a1b_noise(tp_raw, tm_raw):
    out = _a1b_noise(tp_raw, tm_raw, asc, SD_FRAC)
    out.pop("thresholds", None)
    return out


raw_p, raw_m = T["xip_raw"], T["xim_raw"]
real = a1b_noise(raw_p, raw_m)
print("A1b-noise on real data:", json.dumps(real, default=jd))
inj = {"real": real, "tilt": [], "extra_noise": [], "pair_scale_scatter": [], "wrong_m_applied": None}
ltf = np.log(th / np.sqrt(2 * 300)); ltf = ltf / np.abs(ltf).max()
for eps in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    r = a1b_noise(raw_p * (1 + eps * ltf), raw_m * (1 + eps * ltf))
    inj["tilt"].append({"eps": eps, **{k: r[k] for k in ("pass",)}, "max_ratio": max(r["xip"]["max_boxcar_over_white"], r["xim"]["max_boxcar_over_white"]),
                        "min_const_pte": min(r["xip"]["const_pte"], r["xim"]["const_pte"])})
for f_ in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    r = a1b_noise(raw_p + f_ * asc["sigma_xip"] * RNG.standard_normal((21, 1000)),
                  raw_m + f_ * asc["sigma_xim"] * RNG.standard_normal((21, 1000)))
    inj["extra_noise"].append({"f_sigma_fine": f_, "pass": r["pass"],
                               "max_ratio": max(r["xip"]["max_boxcar_over_white"], r["xim"]["max_boxcar_over_white"]),
                               "min_const_pte": min(r["xip"]["const_pte"], r["xim"]["const_pte"])})
for sc in (0.01, 0.02, 0.03, 0.05, 0.1):
    fac = np.exp(sc * RNG.standard_normal(21))[:, None]
    r = a1b_noise(raw_p * fac, raw_m * fac)
    inj["pair_scale_scatter"].append({"sigma_lnA": sc, "pass": r["pass"],
                                      "min_const_pte": min(r["xip"]["const_pte"], r["xim"]["const_pte"])})
r = a1b_noise(T["xip"], T["xim"])   # feed the m-CORRECTED xi instead of raw: constant should fail
inj["wrong_m_applied"] = {"pass": r["pass"], "xip_const_pte": r["xip"]["const_pte"], "xim_const_pte": r["xim"]["const_pte"]}
res["step4_A1b_noise"] = {"thresholds": A1B_THRESH, "real": real, "power_injections": inj,
                          "why_these_thresholds": "see docstring comment above a1b_noise in the script"}
print("injection tilt:", [(d["eps"], d["pass"], round(d["max_ratio"], 2)) for d in inj["tilt"]])
print("injection extra noise:", [(d["f_sigma_fine"], d["pass"], round(d["max_ratio"], 2)) for d in inj["extra_noise"]])
print("injection pair scatter:", [(d["sigma_lnA"], d["pass"], d["min_const_pte"]) for d in inj["pair_scale_scatter"]])
print("m-corrected instead of raw:", inj["wrong_m_applied"])

(OUT / "a1_salvage.json").write_text(json.dumps(res, indent=1, default=jd))
print("wrote", OUT / "a1_salvage.json")

# ===================================================================== plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lab = [f"{i+1}-{j+1}" for i, j in PAIRS]
th9 = fit["theta"]
fig, axs = plt.subplots(3, 7, figsize=(24, 9), sharex=True, sharey=True)
sA = {k: a2[f"{k}_corr"] for k in ("xip", "xim")}
for p, ax in enumerate(axs.ravel()):
    ax.plot(th9, ap9[p] / fit["xiP"][p], "o-", c="C0", label="xi+")
    ax.plot(th9, am9[p] / fit["xiM"][p], "s-", c="C3", label="xi-")
    for k, c in (("xip", "C0"), ("xim", "C3")):
        ax.axhline(1 / sA[k]["A_pair"][p], c=c, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.2, 1.5); ax.axhline(1, c="k", lw=.5)
    ax.set_title(lab[p], fontsize=9)
axs[0, 0].legend(fontsize=7)
fig.supxlabel("theta [arcmin]"); fig.supylabel("ascii (weighted_rebin) / official FITS   [dotted: 1/A_pair from 1000-bin fit]")
fig.tight_layout(); fig.savefig(OUT / "fig1_ascii_over_FITS_by_pair.png", dpi=90); plt.close(fig)

fig, axs = plt.subplots(3, 7, figsize=(24, 9), sharex=True, sharey=True)
Ap_, Am_ = sA["xip"]["A_pair"], sA["xim"]["A_pair"]
for p, ax in enumerate(axs.ravel()):
    for key, A_, c in (("xip", Ap_, "C0"), ("xim", Am_, "C3")):
        z = ((T[key][p] - A_[p] * asc[key][p]) / asc["sigma_" + key][p]).reshape(50, 20).mean(1)
        ax.plot(th.reshape(50, 20).mean(1), z, c=c, lw=1)
    ax.set_xscale("log"); ax.set_ylim(-0.5, 0.5); ax.axhline(0, c="k", lw=.5)
    ax.axhspan(-1 / np.sqrt(20), 1 / np.sqrt(20), color="grey", alpha=.2)
    ax.set_title(lab[p], fontsize=9)
fig.supxlabel("theta [arcmin]")
fig.supylabel("(their - A_p ascii)/sigma_ascii, 20-bin averages  (grey: white-noise 1 sigma)")
fig.tight_layout(); fig.savefig(OUT / "fig2_their_over_ascii_residual_by_pair.png", dpi=90); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
x = np.arange(6)
for ax, tag, ttl in ((axs[0], "xip", "xi+"), (axs[1], "xim", "xi-")):
    d = a2[f"{tag}_corr"]
    ax.errorbar(x - .15, d["s_i_ascii_over_their"], yerr=d["s_i_ascii_over_their"] * d["B_i_err"] / d["B_i"], fmt="o", c="C0", label="1000-bin fit (s=ascii/their)")
    g = a2["gls9"][f"{tag}_FITS_over_ascii"]
    ax.errorbar(x, g["s_i"], yerr=g["s_i"] * g["B_i_err_cov_dominated"] / g["B_i"], fmt="s", c="C1", label="9-bin GLS, ascii/FITS")
    ax.plot(x + .15, PRIOR_S, "^", c="C2", label="prior project")
    ax.plot(x + .25, PROMPT_C * np.sqrt(PROMPT_CONST), "v", c="C4", label="prompt fit (converted)")
    kk_ = a2[f"{tag}_raw"]["global_const_model"]["ascii_over_raw_k"]
    ax.plot(x + .05, np.sqrt(kk_) * (1 + M_GOLD), "x", c="k", label="sqrt(k)(1+m_i): ascii = k * raw")
    ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x]); ax.set_xlabel("tomographic bin"); ax.set_title(ttl)
axs[0].set_ylabel("s_i = ascii / their  per-bin factor"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig3_B_bin_xip_vs_xim.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
xb = np.arange(9)
for key, c in (("xip", "C0"), ("xim", "C3")):
    axs[0].plot(th9, part[key]["per_theta_bin"], "o-", c=c, label=f"{key} C^-1 partition (sum={part[key]['total']:.1f})")
    axs[0].plot(th9, part[key]["diag_per_theta_bin"], "x--", c=c, alpha=.6, label=f"{key} diag-only")
axs[0].set_xscale("log"); axs[0].set_xlabel("theta [arcmin]"); axs[0].set_ylabel("chi2 contribution (189 pts per stat)")
axs[0].set_title("ascii(9 bins) - FITS"); axs[0].legend(fontsize=7)
axs[1].bar(xb, [s["chi2_En6_of_this_range_alone"] for s in swap]); axs[1].set_xticks(xb)
axs[1].set_xticklabels([f"{s['theta_lo']:.0f}-{s['theta_hi']:.0f}" for s in swap], rotation=45, fontsize=7)
axs[1].set_title("En(6 modes) chi2 of (ascii-their) restricted to one theta range"); axs[1].set_ylabel("chi2 /126 dof")
for key, c in (("xip", "C0"), ("xim", "C3")):
    zz = np.array([((T[key][p] - sA[key]["A_pair"][p] * asc[key][p]) / asc["sigma_" + key][p]) for p in range(21)])
    axs[2].plot(th.reshape(50, 20).mean(1), np.sqrt((zz.reshape(21, 50, 20).mean(2) ** 2).mean(0)), c=c, label=key)
axs[2].axhline(np.mean([fine[k]["rms_z_fine"].mean() for k in fine]) / np.sqrt(20), c="k", ls=":", label="white expectation")
axs[2].set_xscale("log"); axs[2].set_xlabel("theta [arcmin]"); axs[2].set_ylabel("rms of 20-bin-averaged z over 21 pairs"); axs[2].legend()
axs[2].set_title("their - A_p*ascii residual vs theta")
fig.tight_layout(); fig.savefig(OUT / "fig4_chi2_vs_theta.png", dpi=100); plt.close(fig)
print("plots written")
