# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Shear sign convention for the KiDS-Legacy DR5 catalogue (RESUME "Not yet done": handoff step 5).

Stages (python scripts/dr5_sign_convention.py --stage {parity,pilot,ggl,report,all}):
  parity : exact/synthetic + real-subsample numerical check of which ellipticity transformations leave
           Re xi+, Re xi- invariant and which flip Im xi+/-, xi_x, gamma_t, gamma_x; and TreeCorr's own frame.
  pilot  : what the Bin6_Bin6 pilot (ours, theirs raw, theirs e2-flipped, official ascii) says about the convention.
  ggl    : galaxy-galaxy lensing (lenses = tomo bin 1, sources = tomo bin 6) with 4 conventions, random-point
           subtraction, TreeCorr jackknife covariance.  The decisive physical test.
  report : (re)writes plots/dr5_sign_ggl.png from the saved ggl results.
Outputs: runs/dr5_measure/sign_convention/{sign.json, ggl_*.npz, *.log}; the .md is written by the lead/agent by hand.
Threads <= 8, RAM <= 4 GB.  Nothing under /mnt/d/HI-gamma-correlation or runs/dr5_measure/xi_chunked is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
OUT = ROOT / "runs/dr5_measure/sign_convention"
WORK = paths.dr5_work_dir()
HI = Path(os.environ.get("KIDSPIPE_THEIRS_XIPM", "/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm"))   # external, read only
PILOT = ROOT / "runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz"
NTH = 8
SEED = 20261030


def _json_load():
    f = OUT / "sign.json"
    return json.load(open(f)) if f.exists() else {}


def _json_save(d):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(OUT / "sign.json", "w"), indent=1, default=float)


# --------------------------------------------------------------------------------------------- parity
TRANS = {                      # name: (sign e1, sign e2, mirror RA about the box centre)
    "identity": (1, 1, False),
    "g->-g (e1,e2 both)": (-1, -1, False),
    "e2->-e2": (1, -1, False),
    "e1->-e1": (-1, 1, False),
    "mirror RA, e2->-e2": (1, -1, True),      # position-angle convention phi->-phi together with e2->-e2
    "mirror RA only": (1, 1, True),
}


def _gg(ra, dec, g1, g2, w, ra2=None, dec2=None, g1b=None, g2b=None, w2=None, nb=10, smax=100.0):
    import treecorr
    c1 = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2, w=w, ra_units="deg", dec_units="deg")
    gg = treecorr.GGCorrelation(nbins=nb, min_sep=2.0, max_sep=smax, sep_units="arcmin", bin_slop=0.0,
                                num_threads=NTH, verbose=0)
    if ra2 is None:
        gg.process(c1)
    else:
        c2 = treecorr.Catalog(ra=ra2, dec=dec2, g1=g1b, g2=g2b, w=w2, ra_units="deg", dec_units="deg")
        gg.process(c1, c2)
    return gg


def stage_parity(res):
    import treecorr
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d = np.load(WORK / "bin5.npz")
    ra, dec = d["ra"].astype("f8"), d["dec"].astype("f8")
    # densest 6x6 deg box on a coarse grid (avoid RA wrap: restrict to ra in [20, 340])
    H, xe, ye = np.histogram2d(ra, dec, bins=[np.arange(0, 360.1, 6), np.arange(-40, 10.1, 6)])
    ix, iy = np.unravel_index(np.argmax(H), H.shape)
    box = (xe[ix], xe[ix + 1], ye[iy], ye[iy + 1])
    m = (ra >= box[0]) & (ra < box[1]) & (dec >= box[2]) & (dec < box[3])
    ra, dec = ra[m], dec[m]
    e1, e2, w = d["e1"][m].astype("f8"), d["e2"][m].astype("f8"), d["w"][m].astype("f8")
    ra0 = 0.5 * (box[0] + box[1])
    out = {"box_ra_dec": [float(b) for b in box], "n_gal": int(m.sum())}
    half = rng.random(m.sum()) < 0.5
    rows = {}
    for name, (s1, s2, mir) in TRANS.items():
        r = (2 * ra0 - ra) if mir else ra
        auto = _gg(r, dec, s1 * e1, s2 * e2, w)
        crs = _gg(r[half], dec[half], (s1 * e1)[half], (s2 * e2)[half], w[half],
                  r[~half], dec[~half], (s1 * e1)[~half], (s2 * e2)[~half], w[~half])
        rows[name] = {"xip": auto.xip, "xim": auto.xim, "xim_im": auto.xim_im,
                      "x_xip": crs.xip, "x_xip_im": crs.xip_im, "x_xim": crs.xim, "x_xim_im": crs.xim_im}
    base = rows["identity"]
    sig = {k: float(np.std(base[k])) + 1e-30 for k in base}
    tab = {}
    for name, r in rows.items():
        tab[name] = {}
        for k in ("xip", "xim", "xim_im", "x_xip", "x_xip_im", "x_xim", "x_xim_im"):
            a, b = np.asarray(base[k]), np.asarray(r[k])
            # classify: same / sign-flipped / neither  (relative to the spread of the baseline curve)
            same = float(np.max(np.abs(b - a)) / sig[k])
            flip = float(np.max(np.abs(b + a)) / sig[k])
            tab[name][k] = {"max|b-a|/std(a)": same, "max|b+a|/std(a)": flip,
                            "verdict": "invariant" if same < 1e-3 else ("sign flips" if flip < 1e-3 else "changes")}
    out["real_subsample_table"] = tab
    out["xim_identity"] = np.asarray(base["xim"]).tolist()
    out["xim_e2flip"] = np.asarray(rows["e2->-e2"]["xim"]).tolist()
    out["xim_e2flip_over_identity_rms"] = float(np.sqrt(np.mean(np.asarray(rows["e2->-e2"]["xim"]) ** 2)) /
                                                np.sqrt(np.mean(np.asarray(base["xim"]) ** 2)))
    out["xip_identity"] = np.asarray(base["xip"]).tolist()
    out["xip_max_rel_diff_e2flip"] = float(np.max(np.abs(np.asarray(rows["e2->-e2"]["xip"]) - base["xip"])) /
                                           np.max(np.abs(base["xip"])))

    # ---- synthetic tangential-shear field: gamma_t, gamma_x under the transformations -------------
    ns, nl = 200000, 300
    side = 6.0
    ra_l = rng.uniform(0, side, nl) + 100.0
    dec_l = rng.uniform(0, side, nl) - 3.0
    ra_s = rng.uniform(0, side, ns) + 100.0
    dec_s = rng.uniform(0, side, ns) - 3.0
    gs = np.zeros(ns, complex)
    for a, b in zip(ra_l, dec_l):
        dx = -(ra_s - a) * np.cos(np.deg2rad(0.5 * (dec_s + b)))      # x = WEST (East on the left), y = North
        dy = dec_s - b
        r = np.hypot(dx, dy) * 60.0                                     # arcmin
        phi = np.arctan2(dy, dx)
        gs += np.where((r > 1) & (r < 100), -0.02 * (5.0 / np.maximum(r, 5.0)) * np.exp(2j * phi), 0)
    gs += rng.normal(0, 0.02, ns) + 1j * rng.normal(0, 0.02, ns)
    ww = np.ones(ns)
    lens_ctr = 100.0 + 0.5 * side
    syn = {}
    for name, (s1, s2, mir) in TRANS.items():
        rl = (2 * lens_ctr - ra_l) if mir else ra_l
        rs = (2 * lens_ctr - ra_s) if mir else ra_s
        L = treecorr.Catalog(ra=rl, dec=dec_l, ra_units="deg", dec_units="deg")
        S = treecorr.Catalog(ra=rs, dec=dec_s, g1=s1 * gs.real, g2=s2 * gs.imag, w=ww, ra_units="deg",
                             dec_units="deg")
        ng = treecorr.NGCorrelation(nbins=6, min_sep=2, max_sep=100, sep_units="arcmin", bin_slop=0.0,
                                    num_threads=NTH, verbose=0)
        ng.process(L, S)
        syn[name] = {"gt": ng.xi.tolist(), "gx": ng.xi_im.tolist()}
    base_gt = np.array(syn["identity"]["gt"])
    out["synthetic_input"] = ("shear defined in the frame x=West, y=North, g = -0.02 (5'/r) exp(2 i phi): tangential "
                              "shear around the lenses, so a correct convention returns gamma_t > 0 (injected) in TreeCorr")
    out["synthetic_table"] = {}
    for name in TRANS:
        gt, gx = np.array(syn[name]["gt"]), np.array(syn[name]["gx"])
        out["synthetic_table"][name] = {"gt": gt.tolist(), "gx": gx.tolist(),
                                        "gt_over_gt_identity_mean": float(np.mean(gt) / np.mean(base_gt)),
                                        "gx_rms_over_gt_identity_rms": float(np.sqrt(np.mean(gx ** 2)) /
                                                                             np.sqrt(np.mean(base_gt ** 2)))}

    # ---- TreeCorr's frame (single lens / single source, no noise) -------------------------------
    fr = {}
    for nm, (dra, ddec) in {"source NE of lens": (3 / 60, 3 / 60), "source NW of lens": (-3 / 60, 3 / 60)}.items():
        for g1, g2 in [(0, -0.1), (0, 0.1)]:
            L = treecorr.Catalog(ra=[10.0], dec=[0.0], ra_units="deg", dec_units="deg")
            S = treecorr.Catalog(ra=[10 + dra], dec=[ddec], ra_units="deg", dec_units="deg", g1=[g1], g2=[g2])
            ng = treecorr.NGCorrelation(nbins=1, min_sep=1, max_sep=10, sep_units="arcmin", bin_slop=0)
            ng.process(L, S)
            fr[f"{nm}, g=({g1},{g2})"] = float(ng.xi[0])
    out["treecorr_frame_single_pair_gt"] = fr
    out["treecorr_frame_conclusion"] = ("gt>0 for a NE source needs g2>0 and for a NW source needs g2<0: TreeCorr's frame "
                                        "is x=WEST, y=NORTH (East to the left, the usual sky-image convention), tangential "
                                        "shear = -Re(g e^{-2i phi}) with phi measured from +x(West) towards North.")
    # ---- what flip_g2=True means: PairCounter ((RA,Dec) frame, x=East) vs TreeCorr (x=West) on a 2x2 deg real-galaxy patch --------
    from kidspipe.measure.paircount import PairCounter
    sel = np.flatnonzero((ra > ra0 - 1) & (ra < ra0 + 1) & (dec > box[2] + 1) & (dec < box[2] + 3))   # 2x2 deg
    pc = PairCounter(ra[sel], dec[sel], e1[sel], e2[sel], w[sel], min_sep=2.0, max_sep=30.0, nbins=8)
    pxip, pxim, pxix, _, _ = pc.run()
    cmp = {}
    for lab, fl in {"TreeCorr no flip": False, "TreeCorr flip_g2=True": True}.items():
        cat = treecorr.Catalog(ra=ra[sel], dec=dec[sel], g1=e1[sel], g2=e2[sel], w=w[sel], ra_units="deg",
                               dec_units="deg", flip_g2=fl)
        gg = treecorr.GGCorrelation(nbins=8, min_sep=2.0, max_sep=30.0, sep_units="arcmin", bin_slop=0.0,
                                    num_threads=NTH, verbose=0)
        gg.process(cat)
        sc = np.std(gg.xim)
        cmp[lab] = {"max|PC.xip-TC.xip|/std(xip)": float(np.max(np.abs(pxip - gg.xip)) / np.std(gg.xip)),
                    "max|PC.xim-TC.xim|/std(xim)": float(np.max(np.abs(pxim - gg.xim)) / sc),
                    "max|PC.xix-TC.xim_im|/std": float(np.max(np.abs(pxix - gg.xim_im)) / sc),
                    "max|PC.xix+TC.xim_im|/std": float(np.max(np.abs(pxix + gg.xim_im)) / sc)}
    out["paircounter_vs_treecorr_real_2x2deg"] = dict(n_gal=int(len(sel)), **cmp)
    out["runtime_s"] = time.time() - t0
    res["parity"] = out
    return res


# --------------------------------------------------------------------------------------------- pilot
def _rebin(x, w, k=50):
    n = len(x) // k * k
    xs = (x[:n] * w[:n]).reshape(-1, k).sum(1)
    ws = w[:n].reshape(-1, k).sum(1)
    return xs / ws


def _ratio(a, b, s):
    """weighted-sum amplitude ratio a/b with weights 1/s^2: sum(a b/s^2)/sum(b^2/s^2)  (B = b is the template)."""
    return float(np.sum(a * b / s ** 2) / np.sum(b * b / s ** 2))


def stage_pilot(res):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from kidspipe.measure import official
    a = np.load(PILOT)
    th = np.load(HI / "fiducial/xipm_bin6_bin6.npz", allow_pickle=True)
    tf = np.load(HI / "e2flip/xipm_bin6_bin6.npz", allow_pickle=True)
    oa = official.load_official_xi("NScomb")
    k = official.PAIRS.index((5, 5))
    o = {key: oa[key][k] for key in ("xip", "xim", "xip_im", "xim_im", "sigma_xip", "sigma_xim")}
    ow = oa["npairs_weighted"][k]
    wt = th["weight"]
    out = {"pair": "Bin6_Bin6", "pair_index": int(k), "ours_file": str(PILOT.relative_to(ROOT)),
           "theirs_files": ["fiducial/xipm_bin6_bin6.npz", "e2flip/xipm_bin6_bin6.npz"],
           "theirs_settings_fiducial": str(th["settings"]), "theirs_settings_e2flip": str(tf["settings"])}
    # windows: 2-100 arcmin where the signal is, rebinned 50 fine bins -> 20 coarse bins
    sig = (_rebin(o["sigma_xip"], np.ones(1000)), _rebin(o["sigma_xim"], np.ones(1000)))
    def cb(x, w=wt):
        return _rebin(np.asarray(x, float), np.asarray(w, float))
    tbl = {}
    sets = {"ours (raw)": (a["xip"], a["xim"], a["xim_im"], a["weight"]),
            "theirs fiducial (raw)": (th["xip_raw"], th["xim_raw"], th["xim_im_raw"], th["weight"]),
            "theirs e2-flipped (raw)": (tf["xip_raw"], tf["xim_raw"], tf["xim_im_raw"], tf["weight"]),
            "official ascii": (o["xip"], o["xim"], o["xim_im"], ow)}
    cbn = {n: (cb(v[0], v[3]), cb(v[1], v[3]), cb(v[2], v[3])) for n, v in sets.items()}
    ref = cbn["theirs fiducial (raw)"]
    refs = (_rebin(np.asarray(th["sigma_xip"]), np.ones(1000)), _rebin(np.asarray(th["sigma_xim"]), np.ones(1000)))
    for n, v in cbn.items():
        tbl[n] = {
            "xip_amp_vs_theirs_raw": _ratio(v[0], ref[0], refs[0]), "xim_amp_vs_theirs_raw": _ratio(v[1], ref[1], refs[1]),
            "xip_corr_vs_theirs_raw": float(np.corrcoef(v[0], ref[0])[0, 1]),
            "xim_corr_vs_theirs_raw": float(np.corrcoef(v[1], ref[1])[0, 1]),
            "xim_im_corr_vs_theirs_raw": float(np.corrcoef(v[2], ref[2])[0, 1]),
            "xim_rms": float(np.sqrt(np.mean(v[1] ** 2))), "xip_rms": float(np.sqrt(np.mean(v[0] ** 2))),
        }
    out["coarse_bins_20_of_50_fine"] = tbl
    out["note"] = ("amplitude = sum(a b/s^2)/sum(b^2/s^2) over 20 coarse bins, b = theirs fiducial raw, s = their sigma; "
                   "corr = Pearson over the 20 coarse bins. xi_x (xim_im) corr is for the sign of the cross component.")
    # same on fine bins (all 1000), ours vs theirs, per the lead's quick look
    wts = np.asarray(a["npairs"]) / 1.0
    fine = {}
    for key, ok, tk in [("xip", "xip", "xip_raw"), ("xim", "xim", "xim_raw"), ("xim_im", "xim_im", "xim_im_raw")]:
        fine[key] = {"corr_ours_theirs_fine": float(np.corrcoef(a[ok], th[tk])[0, 1]),
                     "corr_ours_ascii_fine": float(np.corrcoef(a[ok], o[key])[0, 1]),
                     "corr_theirs_ascii_fine": float(np.corrcoef(th[tk], o[key])[0, 1])}
    fine["theirs_e2flip_vs_fiducial_xim_corr_fine"] = float(np.corrcoef(tf["xim_raw"], th["xim_raw"])[0, 1])
    fine["theirs_e2flip_xip_max_abs_diff_over_sigma"] = float(np.max(np.abs(np.asarray(tf["xip_raw"]) - th["xip_raw"]) /
                                                                     th["sigma_xip"]))
    fine["theirs_e2flip_xim_max_abs_diff_over_sigma"] = float(np.max(np.abs(np.asarray(tf["xim_raw"]) - th["xim_raw"]) /
                                                                     th["sigma_xim"]))
    out["fine_bins_1000"] = fine
    res["pilot"] = out
    return res


# --------------------------------------------------------------------------------------------- GGL
VARIANTS = {"a: catalogue e1,e2 as is": dict(flip_g1=False, flip_g2=False),
            "b: flip_g2": dict(flip_g1=False, flip_g2=True),
            "c: flip_g1": dict(flip_g1=True, flip_g2=False),
            "d: flip_g1+flip_g2": dict(flip_g1=True, flip_g2=True)}


def _ggl_arrays(n_lens, n_rand, seed=SEED, n_src=None):
    rng = np.random.default_rng(seed)
    l = np.load(WORK / "bin0.npz")
    s = np.load(WORK / "bin5.npz")
    ns = len(s["ra"])
    il = rng.choice(len(l["ra"]), n_lens, replace=False)
    ir = rng.choice(ns, n_rand, replace=False)
    isrc = np.arange(ns) if n_src is None else rng.choice(ns, n_src, replace=False)
    f8 = lambda x: np.ascontiguousarray(x, dtype="f8")
    return dict(lra=f8(l["ra"][il]), ldec=f8(l["dec"][il]), rra=f8(s["ra"][ir]), rdec=f8(s["dec"][ir]),
                sra=f8(s["ra"][isrc]), sdec=f8(s["dec"][isrc]), se1=f8(s["e1"][isrc]), se2=f8(s["e2"][isrc]),
                sw=f8(s["w"][isrc]))


def run_ggl(arr, flip_g1, flip_g2, npatch=50, nb=15, smin=1.0, smax=100.0, bin_slop=0.1, log=print):
    import treecorr
    t0 = time.time()
    L = treecorr.Catalog(ra=arr["lra"], dec=arr["ldec"], ra_units="deg", dec_units="deg", npatch=npatch,
                       rng=np.random.default_rng(SEED))
    cen = L.patch_centers
    R = treecorr.Catalog(ra=arr["rra"], dec=arr["rdec"], ra_units="deg", dec_units="deg", patch_centers=cen)
    S = treecorr.Catalog(ra=arr["sra"], dec=arr["sdec"], g1=arr["se1"], g2=arr["se2"], w=arr["sw"],
                         ra_units="deg", dec_units="deg", patch_centers=cen, flip_g1=flip_g1, flip_g2=flip_g2)
    kw = dict(nbins=nb, min_sep=smin, max_sep=smax, sep_units="arcmin", bin_slop=bin_slop, num_threads=NTH,
              var_method="jackknife", verbose=0)
    ng = treecorr.NGCorrelation(**kw)
    ng.process(L, S)
    t1 = time.time()
    rg = treecorr.NGCorrelation(**kw)
    rg.process(R, S)
    t2 = time.time()
    log(f"    process lens {t1 - t0:.0f}s (incl. cat build), randoms {t2 - t1:.0f}s")

    def f(cs):
        xi, xim, _ = cs[0].calculateXi(rg=cs[1])
        return np.concatenate([xi, xim])

    cov = treecorr.estimate_multi_cov([ng, rg], "jackknife", func=f)
    xi, xim, _ = ng.calculateXi(rg=rg)
    xi0, xim0, _ = ng.calculateXi()                       # no random subtraction
    return dict(r=ng.meanr.copy(), gt=np.asarray(xi).copy(), gx=np.asarray(xim).copy(), cov=cov,
                gt_nosub=np.asarray(xi0).copy(), gx_nosub=np.asarray(xim0).copy(),
                gt_rand=np.asarray(rg.xi).copy(), gx_rand=np.asarray(rg.xi_im).copy(),
                npairs=ng.npairs.copy(), npatch=npatch, runtime=time.time() - t0)


def chi2_vs0(v, C, npatch):
    nb = len(v)
    inv = np.linalg.inv(C) * (npatch - nb - 2) / (npatch - 1)       # Hartlap
    c2 = float(v @ inv @ v)
    from scipy.stats import chi2
    return c2, float(chi2.sf(c2, nb))


def summarise(r):
    nb = len(r["gt"])
    cgt, cgx = r["cov"][:nb, :nb], r["cov"][nb:, nb:]
    s = {}
    s["gt_chi2"], s["gt_pte"] = chi2_vs0(r["gt"], cgt, r["npatch"])
    s["gx_chi2"], s["gx_pte"] = chi2_vs0(r["gx"], cgx, r["npatch"])
    s["dof"] = nb
    sg = np.sqrt(np.diag(cgt))
    s["gt_over_sigma"] = (r["gt"] / sg).tolist()
    s["gt_total_snr_diag"] = float(np.sqrt(np.sum((r["gt"] / sg) ** 2)))
    # amplitude of gt relative to a fixed template shape r^-1 (sign of the signal), jackknife-weighted
    t = 1.0 / r["r"]
    inv = np.linalg.inv(cgt) * (r["npatch"] - nb - 2) / (r["npatch"] - 1)
    A = float(t @ inv @ r["gt"] / (t @ inv @ t)); sA = float(1 / np.sqrt(t @ inv @ t))
    s["amp_template_1_over_r"] = A
    s["amp_template_sigma"] = sA
    s["amp_template_snr"] = A / sA
    sx = np.sqrt(np.diag(cgx))
    Ax = float(t @ np.linalg.inv(cgx) * (r["npatch"] - nb - 2) / (r["npatch"] - 1) @ r["gx"] /
               (t @ np.linalg.inv(cgx) * (r["npatch"] - nb - 2) / (r["npatch"] - 1) @ t))
    s["gx_amp_template_1_over_r"] = Ax
    s["gx_over_sigma"] = (r["gx"] / sx).tolist()
    return s


def stage_ggl(res, n_lens=300000, n_rand=600000, small=False):
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if small:
        log("small timing run (30k lenses, 120k randoms, 600k sources)")
        arr = _ggl_arrays(30000, 120000, n_src=600000)
    else:
        arr = _ggl_arrays(n_lens, n_rand)
    log(f"loaded: lenses {len(arr['lra'])}, randoms {len(arr['rra'])}, sources {len(arr['sra'])} in {time.time() - t0:.0f}s")
    out = {"n_lenses": len(arr["lra"]), "n_randoms": len(arr["rra"]), "n_sources": len(arr["sra"]), "seed": SEED,
           "lens_file": "~/kids_dr5_work/bin0.npz (tomo bin 1), sources ~/kids_dr5_work/bin5.npz (tomo bin 6, weights w)",
           "randoms": "random subsample of the bin-6 source positions (follows the source footprint)",
           "treecorr": "NGCorrelation 15 log bins 1-100 arcmin, bin_slop 0.1, 50 kmeans patches from the lenses, "
                       "jackknife via estimate_multi_cov on [lens-source, random-source] (random subtraction included)",
           "variants": {}}
    for name, fl in VARIANTS.items():
        log(f"  variant {name}")
        r = run_ggl(arr, log=log, **fl)
        s = summarise(r)
        out["variants"][name] = {k: v for k, v in s.items()}
        out["variants"][name]["flags"] = fl
        out["variants"][name]["runtime_s"] = r["runtime"]
        np.savez(OUT / (("ggl_small_" if small else "ggl_") + name.split(":")[0] + ".npz"),
                 **{k: np.asarray(v) for k, v in r.items()})
        log(f"    gt chi2 {s['gt_chi2']:.1f}/{s['dof']} (pte {s['gt_pte']:.2g})  gx chi2 {s['gx_chi2']:.1f}/{s['dof']} "
            f"(pte {s['gx_pte']:.2g})  template amp {s['amp_template_1_over_r']:.3g}+-{s['amp_template_sigma']:.2g}")
    res["ggl_small" if small else "ggl"] = out
    return res


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True)
    cols = {"a": "C0", "b": "C3", "c": "C2", "d": "C1"}
    for j, key in enumerate("abcd"):
        f = OUT / f"ggl_{key}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        nb = len(d["gt"])
        sg = np.sqrt(np.diag(d["cov"])[:nb]); sx = np.sqrt(np.diag(d["cov"])[nb:])
        lab = [k for k in VARIANTS if k.startswith(key)][0]
        ax[0, j].errorbar(d["r"], d["r"] * d["gt"], d["r"] * sg, fmt="o", color=cols[key], label=r"$\gamma_t$")
        ax[0, j].errorbar(d["r"] * 1.05, d["r"] * d["gx"], d["r"] * sx, fmt="s", mfc="none", color="k", label=r"$\gamma_\times$")
        ax[0, j].axhline(0, color="gray", lw=0.7)
        ax[0, j].set_title(lab, fontsize=10)
        ax[0, j].set_xscale("log"); ax[0, j].set_xlim(0.8, 130)
        ax[0, j].set_ylabel(r"$\theta\,\gamma$ [arcmin]")
        ax[0, j].legend(fontsize=8)
        ax[1, j].plot(d["r"], d["gt"] / sg, "o", color=cols[key], label="gt/sigma")
        ax[1, j].plot(d["r"], d["gx"] / sx, "s", mfc="none", color="k", label="gx/sigma")
        ax[1, j].axhline(0, color="gray", lw=0.7)
        ax[1, j].set_xlabel("theta [arcmin]")
        ax[1, j].set_ylabel("value / jackknife sigma")
    fig.suptitle("KiDS-Legacy DR5: bin-1 lenses x bin-6 sources, random-subtracted; correct convention: gt>0, gx~0")
    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    fig.savefig(ROOT / "plots/dr5_sign_ggl.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = _json_load()
    if a.stage in ("parity", "all"):
        res = stage_parity(res); _json_save(res)
    if a.stage in ("pilot", "all"):
        res = stage_pilot(res); _json_save(res)
    if a.stage in ("ggl", "all"):
        res = stage_ggl(res, small=a.small); _json_save(res)
    if a.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
