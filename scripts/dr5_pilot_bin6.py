# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Pilot Bin6_Bin6: OUR exact (bin_slop=0, ordered) measurement vs the official release and vs the external
(their) TreeCorr measurement.  Run scripts/dr5_pilot_bin6_slop.py first for the bin_slop=1.5 test (item 4).
Outputs: runs/dr5_measure/pilot_bin6/{pilot.json,PILOT_BIN6.md}, plots/dr5_pilot_bin6_*.png
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe.measure import official as of  # noqa: E402
from kidspipe.measure.a1check import (A1B_THRESH, A3_MAX_ABS_OVER_SIGMA, a3_pass, boxcar_ratio, curves_z,  # noqa: E402
                                      ratio_of_sums)
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402
from kidspipe.theory.cosebis import CosebisFilters  # noqa: E402

OUT = ROOT / "runs/dr5_measure/pilot_bin6"; OUT.mkdir(parents=True, exist_ok=True)
PL = Path(os.environ.get("KIDSPIPE_PLOTS") or ROOT / "plots"); PL.mkdir(parents=True, exist_ok=True)
THEIRS = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin6_bin6.npz")
K = of.PAIRS.index((5, 5))
M6 = 0.045
R = {}

ours = {k: np.asarray(v, float) for k, v in np.load(ROOT / "runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz").items()}
th = {k: np.asarray(v, float) for k, v in np.load(THEIRS, allow_pickle=True).items() if np.asarray(v).dtype.kind == "f"}
asc_all = of.load_official_xi()
asc = {k: asc_all[k][K:K + 1] for k in ("xip", "xim", "sigma_xip", "sigma_xim", "npairs_weighted", "npairs", "weight")}
theta = ours["r_nom"]
R["pair_index"] = K
R["m_factor_theirs"] = float(th["m_factor"]); R["(1+m6)^2"] = (1 + M6) ** 2
R["theirs_corr_equals_raw_over_mfactor_maxrel"] = float(max(np.max(np.abs(th["xip"] * th["m_factor"] / th["xip_raw"] - 1)),
                                                              np.max(np.abs(th["xim"] * th["m_factor"] / th["xim_raw"] - 1))))
R["r_nom_max_abs_diff_ours_theirs_asc"] = float(max(np.abs(theta - th["r_nom"]).max(), np.abs(theta - asc_all["r_nom"][K]).max()))
w_asc = asc["npairs_weighted"]

# ------------------------------------------------------------------ 1. A1-style vs official fine ascii
def a1_block(tp, tm, label):
    z, A = curves_z(tp[None], tm[None], asc)
    rat = boxcar_ratio(z)
    return {"A_p_xip": float(A["xip"][0]), "A_p_xim": float(A["xim"][0]),
            "boxcar_over_white_xip": float(rat[0]), "boxcar_over_white_xim": float(rat[1]),
            "z_rms_xip": float(z[0].std()), "z_rms_xim": float(z[1].std()),
            "lag1_xip": float(np.corrcoef(z[0][:-1], z[0][1:])[0, 1]), "lag1_xim": float(np.corrcoef(z[1][:-1], z[1][1:])[0, 1]),
            "pass_threshold_1.03": bool(rat.max() <= A1B_THRESH["boxcar_over_white_max"])}, z

a1_o, z_o = a1_block(ours["xip"], ours["xim"], "ours")
a1_t, z_t = a1_block(th["xip_raw"], th["xim_raw"], "theirs")
R["A1_ours"], R["A1_theirs"] = a1_o, a1_t
R["A1_threshold"] = A1B_THRESH["boxcar_over_white_max"]

sal = json.loads((ROOT / "runs/dr5_measure/a1_salvage/a1_salvage.json").read_text())["step2_amplitude"]
errp, errm = sal["xip_raw"]["A_pair_err_frac"][K], sal["xim_raw"]["A_pair_err_frac"][K]
Cp, Cm = sal["xip_raw"]["global_const_model"]["C"], sal["xim_raw"]["global_const_model"]["C"]
Cpe, Cme = sal["xip_raw"]["global_const_model"]["C_err"], sal["xim_raw"]["global_const_model"]["C_err"]
Ap_t_sal = (sal["xip_raw"]["A_pair"][K], sal["xim_raw"]["A_pair"][K])
R["salvage"] = {"C_xip": Cp, "C_xim": Cm, "C_err_xip": Cpe, "C_err_xim": Cme, "pair_err_frac_xip": errp, "pair_err_frac_xim": errm,
                "A_p_theirs_from_salvage_json": Ap_t_sal,
                "const_only_value_xip": sal["xip_raw"]["const_only_value"], "const_only_value_xim": sal["xim_raw"]["const_only_value"]}
for key, A, e, C in (("xip", a1_o["A_p_xip"], errp, Cp), ("xim", a1_o["A_p_xim"], errm, Cm)):
    R["salvage"][f"pull_lnA_ours_vs_C_{key}"] = float(np.log(A / C) / e)
    R["salvage"][f"pull_lnA_theirs_vs_C_{key}"] = float(np.log(a1_t[f"A_p_{key}"] / C) / e)
    R["salvage"][f"ours_over_theirs_A_{key}"] = float(A / a1_t[f"A_p_{key}"])
# joint moving-block bootstrap (same recipe as dr5_a1_salvage.py: block 25, 40 starts) of lnA+ - lnA-
rng = np.random.default_rng(12345)
NREP, BLK = 4000, 25
def boot(tp, tm):
    out = np.zeros((NREP, 2))
    for r in range(NREP):
        st = rng.integers(0, 1000, 40); ix = ((st[:, None] + np.arange(BLK)[None]) % 1000).ravel()[:1000]
        out[r] = [np.log(ratio_of_sums(tp[ix], asc["xip"][0][ix], w_asc[0][ix])), np.log(ratio_of_sums(tm[ix], asc["xim"][0][ix], w_asc[0][ix]))]
    return out
b = boot(ours["xip"], ours["xim"])
d = b[:, 0] - b[:, 1]
dobs = np.log(a1_o["A_p_xip"] / a1_o["A_p_xim"])
R["xip_vs_xim"] = {"lnA_ratio_obs": float(dobs), "boot_sd_lnAp": float(b[:, 0].std()), "boot_sd_lnAm": float(b[:, 1].std()),
                   "boot_corr_p_m": float(np.corrcoef(b.T)[0, 1]), "boot_sd_diff_joint": float(d.std()),
                   "pull_joint": float(dobs / d.std()),
                   "sd_diff_quadrature_salvage": float(np.hypot(errp, errm)), "pull_quadrature_salvage": float(dobs / np.hypot(errp, errm)),
                   "obs_theirs_lnA_ratio": float(np.log(a1_t["A_p_xip"] / a1_t["A_p_xim"])),
                   "note": "xi+ and xi- share galaxies; the joint bootstrap carries their correlation, quadrature ignores it"}
# the pair's xi+/xi- difference of the 21-pair constants
R["xip_vs_xim"]["lnC_ratio_21pair"] = float(np.log(Cp / Cm))

# ------------------------------------------------------------------ 2. A3 vs 9-bin FITS
fx = of.load_official_xipm_fits()
cov = fx["cov"]; n = 189
sig9 = {"xip": np.sqrt(np.diag(cov)[K * 9:(K + 1) * 9]), "xim": np.sqrt(np.diag(cov)[n + K * 9:n + (K + 1) * 9])}
val9 = {"xip": fx["xiP"][K], "xim": fx["xiM"][K]}
edges9 = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
def rebin9(r, w, y):   # copy of dr5_adopt_01.py::rebin9
    o = np.zeros(9)
    for c in range(9):
        m = (r >= edges9[c]) & (r < (edges9[c + 1] if c < 8 else 300.01)); o[c] = np.sum(w[m] * y[m]) / np.sum(w[m])
    return o
meas = {"ours": {"xip": ours["xip"] / (1 + M6) ** 2, "xim": ours["xim"] / (1 + M6) ** 2, "w_own": ours["npairs"], "w_wt": ours["weight"]},
        "theirs": {"xip": th["xip"], "xim": th["xim"], "w_own": th["npairs"], "w_wt": th["weight"]}}
R["A3"] = {"threshold": A3_MAX_ABS_OVER_SIGMA, "m_convention": "m-corrected = raw/(1+m6)^2 = raw/1.092025 (auto pair, same as theirs xip=xip_raw/m_factor)"}
rb = {}
for who, d_ in meas.items():
    for key in ("xip", "xim"):
        # (a) adopt-style: own rebin9 weighted by the measurement's own npairs
        a = rebin9(theta, d_["w_own"], d_[key]) - val9[key]
        # (b) official.weighted_rebin weighted by the official ascii npairs_weighted (CosmoSIS bin_xi convention)
        y, tb, _ = of.weighted_rebin(theta, d_[key], w_asc[0], 9)
        bb_ = y - val9[key]
        # (c) weighted by the measurement's own sum w_i w_j
        c, _, _ = of.weighted_rebin(theta, d_[key], d_["w_wt"], 9)
        c = c - val9[key]
        rb[(who, key)] = (a / sig9[key], bb_ / sig9[key], c / sig9[key])
        R["A3"][f"{who}_{key}"] = {"max_abs_over_sigma_adoptstyle_ownnpairs": float(np.abs(a / sig9[key]).max()),
                                   "max_abs_over_sigma_weighted_rebin_officialnpw": float(np.abs(bb_ / sig9[key]).max()),
                                   "max_abs_over_sigma_weighted_rebin_ownweight": float(np.abs(c / sig9[key]).max()),
                                   "pass_adoptstyle": a3_pass(float(np.abs(a / sig9[key]).max())),
                                   "pass_weighted_rebin_officialnpw": a3_pass(float(np.abs(bb_ / sig9[key]).max())),
                                   "d_over_sigma_adoptstyle": (a / sig9[key]).tolist(),
                                   "d_over_sigma_weighted_rebin_officialnpw": (bb_ / sig9[key]).tolist()}
# ours - theirs rebinned directly
R["A3"]["ours_minus_theirs_max_over_sigma"] = {key: float(np.abs((rebin9(theta, ours["npairs"], meas["ours"][key]) - rebin9(theta, th["npairs"], meas["theirs"][key])) / sig9[key]).max()) for key in ("xip", "xim")}
R["A3"]["sigma9"] = {k: v.tolist() for k, v in sig9.items()}

# ------------------------------------------------------------------ 3. COSEBIs
cb = of.load_official_cosebis()
filt = CosebisFilters(2.0, 300.0, source="mpmath", dps=80)
def cosebis(xp, xm): return e_from_xi_binned(theta, xp[None], xm[None], filt, nmodes=20, scheme="edges", unit="rad")
Eo, Bo = cosebis(meas["ours"]["xip"], meas["ours"]["xim"])
Et, Bt = cosebis(meas["theirs"]["xip"], meas["theirs"]["xim"])
Eor, Bor = cosebis(ours["xip"], ours["xim"])    # raw (no m) for the record
cc = cb["cov"]
idxE = K * 20 + np.arange(20); idxB = 420 + idxE
CE, CB = cc[np.ix_(idxE, idxE)], cc[np.ix_(idxB, idxB)]
sE, sB = np.sqrt(np.diag(CE)), np.sqrt(np.diag(CB))
Eoff, Boff = cb["En"][K], cb["Bn"][K]
R["B1"] = {}
for who, E_, B_ in (("ours", Eo[0], Bo[0]), ("theirs", Et[0], Bt[0]), ("ours_raw_no_m", Eor[0], Bor[0])):
    dE = E_ - Eoff
    r_ = {"En_minus_official_over_sigma": (dE / sE).tolist(), "max_abs_En_20": float(np.abs(dE / sE).max()),
          "max_abs_En_6": float(np.abs(dE[:6] / sE[:6]).max()),
          "chi2_En_6": float(dE[:6] @ np.linalg.solve(CE[:6, :6], dE[:6])), "chi2_En_20": float(dE @ np.linalg.solve(CE, dE)),
          "rel_diff_En_6": (dE[:6] / Eoff[:6]).tolist()}
    for nm in (6, 20):
        c2 = float(B_[:nm] @ np.linalg.solve(CB[:nm, :nm], B_[:nm]))
        r_[f"Bn_null_chi2_{nm}"] = c2; r_[f"Bn_null_pte_{nm}"] = float(chi2dist.sf(c2, nm))
    R["B1"][who] = r_
    R["B1"][who]["Bn_over_sigma"] = (B_ / sB).tolist()
for nm in (6, 20):
    c2 = float(Boff[:nm] @ np.linalg.solve(CB[:nm, :nm], Boff[:nm]))
    R["B1"][f"official_Bn_null_chi2_{nm}"] = c2; R["B1"][f"official_Bn_null_pte_{nm}"] = float(chi2dist.sf(c2, nm))
R["B1"]["En_official_over_sigma"] = (Eoff / sE).tolist()
R["B1"]["note"] = "Delta/sigma uses diag of the official En cov block of this pair; ours/theirs are m-corrected as in dr5_adopt_01 B1; chi2 uses the full per-pair block"

# ------------------------------------------------------------------ 4. bin_slop test
S = ROOT / "runs/dr5_measure/pilot_bin6/bin6_bin6_slop15_assembled.npz"
def amp_cmp(x, y, g):
    """x/y amplitude (ratio of g-weighted sums), corr, max|d|/sigma_y, rms of fractional diff in 25 log blocks"""
    return {"amp": float(ratio_of_sums(x, y, g)), "corr": float(np.corrcoef(x, y)[0, 1])}
slop = None
if S.exists():
    slop = {k: np.asarray(v, float) for k, v in np.load(S).items()}
    R["slop15"] = {"npairs_total": float(slop["npairs"].sum()), "weight_total": float(slop["weight"].sum())}
    for lab, a_, b_ in (("ours_vs_theirs", ours, None), ("slop15_vs_theirs", slop, None), ("ours_vs_slop15", ours, slop)):
        ref = th if b_ is None else b_
        tgt = {"xip": ref["xip_raw"] if b_ is None else ref["xip"], "xim": ref["xim_raw"] if b_ is None else ref["xim"]}
        gw = th["weight"] if b_ is None else slop["weight"]
        sg = {"xip": th["sigma_xip"] if b_ is None else np.sqrt(slop["varxi"]), "xim": th["sigma_xim"] if b_ is None else np.sqrt(slop["varxi"])}
        o = {}
        for key in ("xip", "xim"):
            x = a_[key]; y = tgt[key]
            dd = (x - y) / sg[key]
            o[key] = {"amp": float(ratio_of_sums(x, y, gw)), "corr": float(np.corrcoef(x, y)[0, 1]),
                      "max_abs_d_over_sigma": float(np.abs(dd).max()), "rms_d_over_sigma": float(dd.std())}
            # amplitude by decade: 2-10, 10-60, 60-300 arcmin
            for lo, hi in ((2, 10), (10, 60), (60, 300)):
                m = (theta >= lo) & (theta < hi)
                o[key][f"amp_{lo}_{hi}"] = float(ratio_of_sums(x[m], y[m], gw[m]))
        R["slop15"][lab] = o
    R["slop15"]["npairs_slop15_over_theirs"] = float(slop["npairs"].sum() / th["npairs"].sum())
    R["slop15"]["weight_slop15_over_ours"] = float(slop["weight"].sum() / ours["weight"].sum())
    R["slop15"]["npairs_slop15_over_ours"] = float(slop["npairs"].sum() / ours["npairs"].sum())
    R["slop15"]["per_bin_npairs_ratio_slop15_over_ours_range"] = [float(np.min(slop["npairs"] / ours["npairs"])), float(np.max(slop["npairs"] / ours["npairs"]))]
    R["slop15"]["per_bin_npairs_ratio_slop15_over_theirs_x1_range"] = [float(np.min(slop["npairs"] / th["npairs"])), float(np.max(slop["npairs"] / th["npairs"]))]
    # A_p vs ascii, E_n for slop15
    zs, As = curves_z(slop["xip"][None], slop["xim"][None], asc)
    R["slop15"]["A_p_xip"], R["slop15"]["A_p_xim"] = float(As["xip"][0]), float(As["xim"][0])
    R["slop15"]["boxcar_over_white"] = boxcar_ratio(zs).tolist()
    Es, Bs = cosebis(slop["xip"] / (1 + M6) ** 2, slop["xim"] / (1 + M6) ** 2)
    R["slop15"]["max_abs_En_6_over_sigma_vs_official"] = float(np.abs((Es[0] - Eoff)[:6] / sE[:6]).max())
    R["slop15"]["En_ours_minus_slop15_over_sigma_official"] = ((Eo[0] - Es[0]) / sE).tolist()
    R["slop15"]["En_theirs_minus_slop15_over_sigma_official"] = ((Et[0] - Es[0]) / sE).tolist()
    R["slop15"]["En_slop15_minus_official_over_sigma"] = ((Es[0] - Eoff) / sE).tolist()
    R["slop15"]["max_abs_En_20_slop15_vs_official"] = float(np.abs((Es[0] - Eoff) / sE).max())
    # edge bins: npairs ratio slop15/ours in the first and last 5 fine bins
    q = slop["npairs"] / ours["npairs"]
    R["slop15"]["npairs_ratio_slop15_over_ours_first5"] = q[:5].tolist(); R["slop15"]["npairs_ratio_slop15_over_ours_last5"] = q[-5:].tolist()
    R["slop15"]["npairs_ratio_slop15_over_ours_mid"] = float(np.mean(q[100:900]))
    # Differential test of slop1.5 - slop0 on the SAME pairs: contribution of every chunk-pair k to the total amplitude shift
    #   Delta_k = sum_b g_b fold_k w_kb (xi15_kb - xi0_kb) / W_b / sum_b g_b xi0_b ,  sum_k Delta_k = amp_ratio - 1 (W_b of slop0)
    # then a delete-one-RA-strip jackknife (strip a removes all chunk-pairs (a,*) and (*,a)) for the error.
    ck = {}
    for f0 in sorted((ROOT / "runs/dr5_measure/xi_chunked/chunks/Bin6_Bin6").glob("c*.npz")):
        a_, b_ = int(f0.stem[1:3]), int(f0.stem[4:6])
        c0, c1 = np.load(f0), np.load(OUT / "slop15_chunks" / f0.name)
        ck[(a_, b_)] = (float(c0["fold"]), c0["weight"], c0["xip"], c0["xim"], c1["xip"], c1["xim"])
    Wb = ours["weight"]; g = w_asc[0]
    dm = {}
    for key, i0, i1 in (("xip", 2, 4), ("xim", 3, 5)):
        den = np.sum(g * ours[key]); D = {}
        for kk, v in ck.items():
            D[kk] = float(np.sum(g * v[0] * v[1] * (v[i1] - v[i0]) / Wb) / den)
        tot = sum(D.values())
        jk = []
        for a_ in range(16):
            jk.append(sum(val for (x, y), val in D.items() if x != a_ and y != a_))
        jk = np.array(jk); n_ = 16
        # jackknife of the total after dropping strip a: rescale not needed for the sd of the leave-one-out spread
        sd_jk = float(np.sqrt((n_ - 1) / n_ * np.sum((jk - jk.mean()) ** 2)))
        diag = sum(val for (x, y), val in D.items() if x == y)
        dm[key] = {"total_shift_sum_k": float(tot), "from_diagonal_chunkpairs": float(diag), "from_offdiag": float(tot - diag),
                   "jackknife_sd_over_strips": sd_jk, "t": float(tot / sd_jk),
                   "check_amp_minus_1": float(ratio_of_sums(slop[key], ours[key], g) - 1),
                   "n_nonzero_pairs": int(sum(1 for v in D.values() if v != 0.0)), "sign_counts_pos_neg": [int(sum(1 for v in D.values() if v > 0)), int(sum(1 for v in D.values() if v < 0))]}
    R["slop15"]["differential_shift_slop15_minus_slop0"] = dm

# ------------------------------------------------------------------ 5. sanity
sn = {}
sn["npairs_ours_over_theirs"] = float(ours["npairs"].sum() / th["npairs"].sum())
sn["npairs_ours_over_theirs_per_bin_range"] = [float((ours["npairs"] / th["npairs"]).min()), float((ours["npairs"] / th["npairs"]).max())]
rw = ours["weight"] / asc["npairs_weighted"][0]
sn["weight_ours_over_official_npairs_weighted"] = {"total": float(ours["weight"].sum() / asc["npairs_weighted"][0].sum()),
                                                   "bin_min": float(rw.min()), "bin_max": float(rw.max()), "bin_median": float(np.median(rw)), "bin_std_over_mean": float(rw.std() / rw.mean())}
rn = ours["npairs"] / asc["npairs"][0]
sn["npairs_ours_over_official_npairs"] = {"total": float(ours["npairs"].sum() / asc["npairs"][0].sum()), "bin_min": float(rn.min()), "bin_max": float(rn.max())}
rw2 = th["weight"] / asc["npairs_weighted"][0]
sn["weight_theirs_over_official_npairs_weighted_total"] = float(th["weight"].sum() / asc["npairs_weighted"][0].sum())
sn["weight_theirs_over_official_npairs_weighted_bin_range"] = [float(rw2.min()), float(rw2.max())]
sn["official_npairs_weighted_over_npairs_total"] = float(asc["npairs_weighted"][0].sum() / asc["npairs"][0].sum())
sn["ours_weight_over_npairs_total"] = float(ours["weight"].sum() / ours["npairs"].sum())
sn["official_npairs_frac_part_mean_abs_minus_half"] = float(np.mean(np.abs((asc["npairs"][0] % 1) - 0.5)))
# slope of official weight vs ours with theta?
sn["weight_ratio_vs_lnTheta_slope"] = float(np.polyfit(np.log(theta), rw / rw.mean(), 1)[0])
# xim_im
zim = ours["xim_im"] / np.sqrt(ours["varxi"])
zim_t = th["xim_im_raw"] / th["sigma_xim"]
def bc(z, B=20):
    nn = z.size // B * B; zb = z[:nn].reshape(-1, B).mean(1); return float(zb.std() / (z.std() / np.sqrt(B)))
sn["xim_im_over_sqrtvarxi"] = {"mean": float(zim.mean()), "std": float(zim.std()), "mean_over_sem_white": float(zim.mean() * np.sqrt(zim.size) / zim.std()),
                               "boxcar_over_white": bc(zim), "max_abs": float(np.abs(zim).max()), "lag1": float(np.corrcoef(zim[:-1], zim[1:])[0, 1]),
                               "weighted_chi2_rel_1000": float(np.sum(zim ** 2)), "theirs_std": float(zim_t.std()), "theirs_mean": float(zim_t.mean())}
# joint significance of the 9-bin xim_im
x9 = rebin9(theta, ours["weight"], ours["xim_im"]); 
sn["xim_im_ours_vs_theirs_corr"] = float(np.corrcoef(ours["xim_im"], th["xim_im_raw"])[0, 1])
sn["xim_im_9bin_over_sigma_xim"] = (x9 / sig9["xim"] ).tolist()
sig_o = np.sqrt(ours["varxi"])
sn["varxi_sqrt_ours_over_theirs_sigma_xip"] = {"median": float(np.median(sig_o / th["sigma_xip"])), "range": [float((sig_o / th["sigma_xip"]).min()), float((sig_o / th["sigma_xip"]).max())]}
so_ = asc["sigma_xip"][0]
sn["sigma_ours_over_official_sigma_xip"] = {"median": float(np.median(sig_o / so_)), "range": [float((sig_o / so_).min()), float((sig_o / so_).max())]}
sn["sigma_theirs_over_official_sigma_xip_median"] = float(np.median(th["sigma_xip"] / so_))
sn["sigma_ours_over_official_sigma_xim_median"] = float(np.median(sig_o / asc["sigma_xim"][0]))
sn["official_sigma_xip_over_sigma_xim_median"] = float(np.median(asc["sigma_xip"][0] / asc["sigma_xim"][0]))
sn["sigma_ratio_expected_if_ordered_vs_unordered"] = 1 / np.sqrt(2)
# is the scatter of (ours - A asc)/sigma consistent with sigma? (rms z already in A1)
sn["fine_scatter_note"] = "rms z (ours - A_p ascii)/sigma_ascii is in A1_ours: z_rms_*"
R["sanity"] = sn

(OUT / "pilot.json").write_text(json.dumps(R, indent=2, default=float))

# ------------------------------------------------------------------ plots
def logbin(x, w, y, nb=40):
    e = np.exp(np.linspace(np.log(2), np.log(300), nb + 1)); o = np.zeros(nb); t = np.zeros(nb)
    for c in range(nb):
        m = (x >= e[c]) & (x < e[c + 1]) if c < nb - 1 else (x >= e[c]); o[c] = np.sum(w[m] * y[m]) / np.sum(w[m]); t[c] = np.sum(w[m] * x[m]) / np.sum(w[m])
    return t, o
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for j, key in enumerate(("xip", "xim")):
    A = R["A1_ours"]["A_p_" + key]
    for lab, x, y, st in (("ours (bin_slop 0)", theta, ours[key], "o"), ("theirs raw (slop 1.5)", theta, th[key + "_raw"], "s"),
                          (f"official ascii x A_p={A:.3f}", theta, A * asc[key][0], "^")):
        t, o = logbin(x, ours["weight"], y); ax[0, j].plot(t, t * o * 1e4, st + "-", ms=4, lw=0.8, label=lab)
    ax[0, j].set_ylabel(r"$\theta\,\xi_{%s}$ [arcmin x1e-4] (raw, 40 log bins)" % ("+" if key == "xip" else "-"))
    ax[0, j].set_xscale("log"); ax[0, j].legend(fontsize=7); ax[0, j].set_title(r"Bin6$\times$Bin6 $\xi_{%s}$" % ("+" if key == "xip" else "-"))
    # residual in sigma units, 40 blocks of 25 bins
    zz = z_o[j]; nb = 40; zb = zz.reshape(nb, 25).mean(1) * np.sqrt(25); tb = np.exp(np.log(theta).reshape(nb, 25).mean(1))
    ax[1, j].plot(tb, zb, "o-", ms=3, lw=0.8, label="ours - A_p ascii (block mean x sqrt25, sigma_ascii)")
    zt = z_t[j].reshape(nb, 25).mean(1) * np.sqrt(25); ax[1, j].plot(tb, zt, "s-", ms=3, lw=0.8, label="theirs")
    ax[1, j].axhline(0, c="k", lw=0.5); ax[1, j].set_xlabel(r"$\theta$ [arcmin]"); ax[1, j].set_ylabel("z block"); ax[1, j].legend(fontsize=7)
fig.tight_layout(); fig.savefig(PL / "dr5_pilot_bin6_ascii.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tb9 = of.weighted_rebin(theta, ours["xip"], w_asc[0], 9)[1]
for j, key in enumerate(("xip", "xim")):
    for who, off, st in (("ours", -0.04, "o"), ("theirs", 0.04, "s")):
        ax[j].plot(tb9 * np.exp(off), rb[(who, key)][0], st, label=f"{who} (adopt-style, own npairs)")
        ax[j].plot(tb9 * np.exp(off), rb[(who, key)][1], st, mfc="none", label=f"{who} (weighted_rebin, official npairs_w)")
    ax[j].axhline(A3_MAX_ABS_OVER_SIGMA, c="r", ls="--"); ax[j].axhline(-A3_MAX_ABS_OVER_SIGMA, c="r", ls="--", label="gate +-0.1")
    ax[j].set_xscale("log"); ax[j].set_xlabel(r"$\theta$ [arcmin]"); ax[j].set_ylabel(r"$(\xi_{9bin}-\xi_{\rm FITS})/\sigma_{\rm FITS}$")
    ax[j].set_title(r"A3 for Bin6$\times$Bin6 $\xi_{%s}$" % ("+" if key == "xip" else "-")); ax[j].legend(fontsize=6)
fig.tight_layout(); fig.savefig(PL / "dr5_pilot_bin6_A3.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
nn = np.arange(1, 21)
ax[0, 0].errorbar(nn, Eoff * 1e9, sE * 1e9, fmt="k.", label="official $E_n$ (cov diag)")
ax[0, 0].plot(nn, Eo[0] * 1e9, "o", mfc="none", label="ours"); ax[0, 0].plot(nn, Et[0] * 1e9, "x", label="theirs")
ax[0, 0].set_ylabel(r"$E_n$ [1e-9]"); ax[0, 0].set_yscale("symlog", linthresh=1); ax[0, 0].legend(fontsize=7)
ax[0, 1].plot(nn, (Eo[0] - Eoff) / sE, "o-", label="ours"); ax[0, 1].plot(nn, (Et[0] - Eoff) / sE, "x-", label="theirs")
ax[0, 1].set_ylabel(r"$(E_n-E_n^{\rm off})/\sigma$"); ax[0, 1].legend(fontsize=7)
ax[1, 0].errorbar(nn, Boff * 1e9, sB * 1e9, fmt="k.", label="official $B_n$"); ax[1, 0].plot(nn + .1, Bo[0] * 1e9, "o", mfc="none", label="ours"); ax[1, 0].plot(nn + .2, Bt[0] * 1e9, "x", label="theirs")
ax[1, 0].axhline(0, c="k", lw=.5); ax[1, 0].set_ylabel(r"$B_n$ [1e-9]"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_yscale("symlog", linthresh=1)
ax[1, 1].plot(nn, Bo[0] / sB, "o-", label="ours"); ax[1, 1].plot(nn, Bt[0] / sB, "x-", label="theirs"); ax[1, 1].plot(nn, Boff / sB, "k.-", label="official")
ax[1, 1].axhline(0, c="k", lw=.5); ax[1, 1].set_ylabel(r"$B_n/\sigma_{B_n}$"); ax[1, 1].legend(fontsize=7)
for a_ in ax.ravel(): a_.set_xlabel("mode n")
fig.suptitle("COSEBIs Bin6xBin6"); fig.tight_layout(); fig.savefig(PL / "dr5_pilot_bin6_cosebis.png", dpi=110); plt.close(fig)

if slop is not None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, key in enumerate(("xip", "xim")):
        gg_ = w_asc[0]; dn = np.sum(gg_ * th[key + "_raw"])
        for lab, x in (("ours (slop 0)", ours[key]), ("slop 1.5 (our chunked ordered)", slop[key])):
            ax[0, j].plot(theta, np.cumsum(gg_ * (x - th[key + "_raw"])) / dn, lw=1.2, label=lab + " - theirs_raw")
        ax[0, j].axhline(0, c="k", lw=.5); ax[0, j].set_xscale("log")
        ax[0, j].set_ylabel(r"cumulative $\sum g(\xi-\xi_{theirs})/\sum g\,\xi_{theirs}$, $\xi_{%s}$" % ("+" if key == "xip" else "-")); ax[0, j].legend(fontsize=7)
        dz = (ours[key] - slop[key]) / np.sqrt(slop["varxi"]); dt = (slop[key] - th[key + "_raw"]) / th["sigma_" + key]
        ax[1, j].plot(theta, dz, lw=.4, label="(ours - slop1.5)/sigma"); ax[1, j].plot(theta, dt, lw=.4, label="(slop1.5 - theirs)/sigma")
        ax[1, j].set_xscale("log"); ax[1, j].set_xlabel(r"$\theta$ [arcmin]"); ax[1, j].legend(fontsize=7); ax[1, j].set_ylabel("per fine bin, sigma units")
    fig.tight_layout(); fig.savefig(PL / "dr5_pilot_bin6_slop.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].loglog(theta, ours["weight"] / asc["npairs_weighted"][0], lw=.6, label="ours weight / official npairs_weighted")
ax[0].loglog(theta, th["weight"] / asc["npairs_weighted"][0], lw=.6, label="theirs weight / official npairs_w"); ax[0].legend(fontsize=7); ax[0].set_xlabel("theta [arcmin]")
ax[1].plot(theta, zim, lw=.4); ax[1].set_xscale("log"); ax[1].set_title("xim_im / sqrt(varxi), ours"); ax[1].set_xlabel("theta [arcmin]")
ax[2].loglog(theta, sig_o, label="ours sqrt(varxi)"); ax[2].loglog(theta, th["sigma_xip"], label="theirs sigma_xip"); ax[2].loglog(theta, so_, label="official sigma_xip"); ax[2].legend(fontsize=7); ax[2].set_xlabel("theta [arcmin]")
fig.tight_layout(); fig.savefig(PL / "dr5_pilot_bin6_sanity.png", dpi=110); plt.close(fig)
print(json.dumps({k: R[k] for k in ("A1_ours", "A1_theirs", "xip_vs_xim")}, indent=1, default=float))
