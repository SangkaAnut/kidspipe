# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Pilot Bin6_Bin6 re-measured at (bin_slop=0, angle_slop=0) -- the setting that is exact by construction (brute-force
check in runs/dr5_measure/angle_slop/brute_check.json) -- with OUR chunked ordered approach (copy of
scripts/dr5_pilot_bin6_slop.py).  Chunk files go to runs/dr5_measure/pilot_bin6/<tag>_chunks/ (NOT xi_chunked/),
resumable.  Then A1/A3/E_n/B_n are evaluated against the official release exactly as scripts/dr5_pilot_bin6.py does.

  python scripts/dr5_pilot_bin6_exact.py run  [--bin-slop 0 --angle-slop 0 --threads 16]
  python scripts/dr5_pilot_bin6_exact.py eval [--bin-slop 0 --angle-slop 0]
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("drv", ROOT / "scripts/dr5_measure_xi_chunked.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
from kidspipe.measure.xi import run_pair, corr_to_arrays  # noqa: E402
from kidspipe.measure import official as of  # noqa: E402
from kidspipe.measure.a1check import (A1B_THRESH, A3_MAX_ABS_OVER_SIGMA, a3_pass, boxcar_ratio, curves_z)  # noqa: E402
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402
from kidspipe.theory.cosebis import CosebisFilters  # noqa: E402

PB = ROOT / "runs/dr5_measure/pilot_bin6"
THEIRS = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin6_bin6.npz")
K = of.PAIRS.index((5, 5)); M6 = 0.045


def tagof(b, a):
    return f"b{b:g}_a{a:g}"


def assemble(files):
    acc = None
    for f in files:
        a = np.load(f); fold = float(a["fold"]); w = np.asarray(a["weight"], float)
        if acc is None:
            acc = {k: np.zeros_like(w) for k in ("npairs", "weight", "wxip", "wxim", "wxim_im", "vn")}
            rnom = np.asarray(a["r_nom"], float); meanr = np.asarray(a["meanr"], float)
        acc["npairs"] += fold * a["npairs"]; acc["weight"] += fold * w
        acc["wxip"] += fold * w * a["xip"]; acc["wxim"] += fold * w * a["xim"]
        acc["wxim_im"] += fold * w * a["xim_im"]; acc["vn"] += fold * a["varxi"] * w ** 2
    ws = np.where(acc["weight"] > 0, acc["weight"], np.nan)
    return {"r_nom": rnom, "meanr": meanr, "xip": acc["wxip"] / ws, "xim": acc["wxim"] / ws, "xim_im": acc["wxim_im"] / ws,
            "varxi": acc["vn"] / ws ** 2, "npairs": acc["npairs"], "weight": acc["weight"]}


def do_run(args):
    out = PB / f"{tagof(args.bin_slop, args.angle_slop)}_chunks"; out.mkdir(parents=True, exist_ok=True)
    cats1, cats2 = drv.load_pair_cats(6, 6, 16, {})
    t0 = time.time()
    for (a, b) in drv.chunk_pairs(16, True):
        f = out / f"c{a:02d}_{b:02d}.npz"
        if f.exists():
            continue
        t1 = time.time()
        arr = corr_to_arrays(run_pair(cats1[a], cats2[b], bin_slop=args.bin_slop, num_threads=args.threads, angle_slop=args.angle_slop))
        arr["fold"] = np.float64(drv.fold_factor(a, b, True))
        arr["seconds"] = np.float64(time.time() - t1)
        arr.update(drv.slop_record(args.bin_slop, args.angle_slop))
        np.savez(f, **arr)
        print(f"[{time.time()-t0:7.0f}s] c{a:02d}_{b:02d} {time.time()-t1:6.1f}s npairs={arr['npairs'].sum():.3e}", flush=True)
    res = assemble(sorted(out.glob("c*.npz")))
    np.savez(PB / f"bin6_bin6_{tagof(args.bin_slop, args.angle_slop)}_assembled.npz", **res)
    print("done", time.time() - t0)


def do_eval(args):
    tag = tagof(args.bin_slop, args.angle_slop)
    new = {k: np.asarray(v, float) for k, v in np.load(PB / f"bin6_bin6_{tag}_assembled.npz").items()}
    ours = {k: np.asarray(v, float) for k, v in np.load(ROOT / "runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz").items()}
    th = {k: np.asarray(v, float) for k, v in np.load(THEIRS, allow_pickle=True).items() if np.asarray(v).dtype.kind == "f"}
    slop = {k: np.asarray(v, float) for k, v in np.load(PB / "bin6_bin6_slop15_assembled.npz").items()}
    asc_all = of.load_official_xi(); asc = {k: asc_all[k][K:K + 1] for k in ("xip", "xim", "sigma_xip", "sigma_xim", "npairs_weighted", "npairs", "weight")}
    theta = new["r_nom"]; w_asc = asc["npairs_weighted"]
    fx = of.load_official_xipm_fits(); cov = fx["cov"]; n = 189
    sig9 = {"xip": np.sqrt(np.diag(cov)[K * 9:(K + 1) * 9]), "xim": np.sqrt(np.diag(cov)[n + K * 9:n + (K + 1) * 9])}
    val9 = {"xip": fx["xiP"][K], "xim": fx["xiM"][K]}
    e9 = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
    def rebin9(r, w, y):
        o = np.zeros(9)
        for c in range(9):
            m = (r >= e9[c]) & (r < (e9[c + 1] if c < 8 else 300.01)); o[c] = np.sum(w[m] * y[m]) / np.sum(w[m])
        return o
    cb = of.load_official_cosebis(); filt = CosebisFilters(2.0, 300.0, source="mpmath", dps=80)
    idxE = K * 20 + np.arange(20); idxB = 420 + idxE
    CE, CB = cb["cov"][np.ix_(idxE, idxE)], cb["cov"][np.ix_(idxB, idxB)]
    sE = np.sqrt(np.diag(CE)); Eoff = cb["En"][K]
    R = {"setting": {"bin_slop": args.bin_slop, "angle_slop": args.angle_slop}, "npairs_total": float(new["npairs"].sum())}
    sets = {tag: new, "pilot_b0_adef": ours, "slop15": slop}
    for who, d in list(sets.items()) + [("theirs", {**th, "xip": th["xip_raw"], "xim": th["xim_raw"]})]:
        r = {}
        zz, A = curves_z(d["xip"][None], d["xim"][None], asc); rat = boxcar_ratio(zz)
        r["A1"] = {"A_p_xip": float(A["xip"][0]), "A_p_xim": float(A["xim"][0]), "boxcar_over_white": rat.tolist(),
                   "pass_1.03": bool(rat.max() <= A1B_THRESH["boxcar_over_white_max"])}
        mc = {k: (d[k] / (1 + M6) ** 2 if who != "theirs" else th[k]) for k in ("xip", "xim")}      # theirs: already m-corrected
        wown = d["npairs"] if who != "theirs" else th["npairs"]
        r["A3"] = {}
        for key in ("xip", "xim"):
            dd = (rebin9(theta, wown, mc[key]) - val9[key]) / sig9[key]
            y, _, _ = of.weighted_rebin(theta, mc[key], w_asc[0], 9)
            d2 = (y - val9[key]) / sig9[key]
            r["A3"][key] = {"d_over_sigma_adoptstyle": dd.tolist(), "max_adoptstyle": float(np.abs(dd).max()),
                            "max_weighted_rebin_officialnpw": float(np.abs(d2).max()), "pass": a3_pass(float(np.abs(dd).max()))}
        E, B = e_from_xi_binned(theta, mc["xip"][None], mc["xim"][None], filt, nmodes=20, scheme="edges", unit="rad")
        dE = (E[0] - Eoff) / sE
        r["B1"] = {"En_minus_official_over_sigma": dE.tolist(), "max_En_20": float(np.abs(dE).max()), "max_En_6": float(np.abs(dE[:6]).max()),
                   "chi2_En_6": float((E[0] - Eoff)[:6] @ np.linalg.solve(CE[:6, :6], (E[0] - Eoff)[:6])),
                   "chi2_En_20": float((E[0] - Eoff) @ np.linalg.solve(CE, E[0] - Eoff))}
        for nm in (6, 20):
            c2 = float(B[0][:nm] @ np.linalg.solve(CB[:nm, :nm], B[0][:nm])); r["B1"][f"Bn_null_chi2_{nm}"] = c2; r["B1"][f"Bn_null_pte_{nm}"] = float(chi2dist.sf(c2, nm))
        R[who] = r
    # direct comparison new vs pilot (0,default angle_slop)
    R["new_vs_pilot"] = {k: {"max_abs_fine_rel": float(np.max(np.abs(new[k] - ours[k]) / np.maximum(np.abs(ours[k]), 1e-30))),
                             "amp_ratio_weighted": float(np.sum(new["weight"] * new[k]) / np.sum(new["weight"] * ours[k]))} for k in ("xip", "xim")}
    R["new_vs_pilot"]["npairs_ratio"] = float(new["npairs"].sum() / ours["npairs"].sum())
    R["new_vs_pilot"]["max_9bin_d_over_sigma_official"] = {
        k: float(np.abs((rebin9(theta, new["npairs"], new[k]) - rebin9(theta, ours["npairs"], ours[k])) / (1 + M6) ** 2 / sig9[k]).max()) for k in ("xip", "xim")}
    (ROOT / "runs/dr5_measure/angle_slop" / f"exact_bin6_eval_{tag}.json").write_text(json.dumps(R, indent=1, default=float))
    for who in sets and list(R)[2:]:
        if isinstance(R[who], dict) and "A3" in R[who]:
            print(who, "A1", [round(x, 3) for x in R[who]["A1"]["boxcar_over_white"]], "A3 xi+/xi-", round(R[who]["A3"]["xip"]["max_adoptstyle"], 4),
                  round(R[who]["A3"]["xim"]["max_adoptstyle"], 4), "En20", round(R[who]["B1"]["max_En_20"], 3), "En6", round(R[who]["B1"]["max_En_6"], 3))
    print(R["new_vs_pilot"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("--bin-slop", type=float, default=0.0); ap.add_argument("--angle-slop", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(); (do_run if a.cmd == "run" else do_eval)(a)
