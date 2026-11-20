# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""southpatch verifier: run settings on a southern Bin6 patch (run) / analyze (analyze)."""
import sys, time, json, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
OUTD = ROOT/"runs/dr5_measure/angle_slop/verify/southpatch"
PATCH = (10.0, 24.0, -36.0, -27.0)
THREADS = 8
def cat():
    from kidspipe.measure.xi import load_bin, make_treecorr_cat
    r = load_bin(6); lo, hi, d0, d1 = PATCH
    m = (r["ra"]>=lo)&(r["ra"]<hi)&(r["dec"]>=d0)&(r["dec"]<d1)
    return make_treecorr_cat({k: r[k][m] for k in ("ra","dec","e1","e2","w")}), int(m.sum())
def run(settings):
    from kidspipe.measure.xi import run_pair, corr_to_arrays
    c, n = cat(); print("N", n, flush=True)
    for s in settings:
        bs, a = (float(x) for x in s.split(":")); f = OUTD/f"S_Bin6_Bin6_b{bs:g}_a{a:g}.npz"
        if f.exists(): continue
        t=time.time(); gg = run_pair(c, c, bin_slop=bs, num_threads=THREADS, angle_slop=a); dt=time.time()-t
        arr = corr_to_arrays(gg); arr.update(seconds=np.float64(dt), bin_slop=bs, angle_slop=a, ngal=n, threads=THREADS)
        np.savez(f, **arr); print(f.name, dt, flush=True)
if __name__=="__main__":
    if sys.argv[1]=="run": run(sys.argv[2].split(","))

def analyze():
    import importlib.util
    spec = importlib.util.spec_from_file_location("st", ROOT/"scripts/dr5_angle_slop_study.py")
    st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
    ctx = st.official_ctx(6, 6)
    L = {}
    for f in sorted(OUTD.glob("S_Bin6_Bin6_*.npz")):
        z = np.load(f); L[(float(z["bin_slop"]), float(z["angle_slop"]))] = {k: np.asarray(z[k], float) for k in z.files}
    ex = L[(0.0, 0.0)]; res = {}
    scale, vfac = st.fine_scale(ex, ctx)
    cen = np.sqrt(st.EDGES9[:-1]*st.EDGES9[1:])
    res["scale_sigma_ratio_by_9bin"] = {k: [float(np.mean(scale[k][(ex["r_nom"]>=st.EDGES9[c])&(ex["r_nom"]<st.EDGES9[c+1]+(0.01 if c==8 else 0))])) for c in range(9)] for k in scale}
    for (b, a), S in sorted(L.items()):
        m = st.metrics(ex, S, ctx)
        r = ex["r_nom"]; mf = ctx["mfac"]
        # own recomputation of E_n deviation (patch-scaled), coherence over n
        dx = {k: S[k]-ex[k] for k in ("xip","xim")}
        Ed, Bd = st.cosebis_E(ctx, r, dx["xip"]*scale["xip"]/mf, dx["xim"]*scale["xim"]/mf)
        e = Ed/ctx["sE"]
        sg = np.sign(e[:20]); nsc = int(np.sum(sg[1:] != sg[:-1]))
        ac = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if np.std(e) > 0 else float("nan")
        res[f"{b:g}:{a:g}"] = dict(seconds=float(S["seconds"]), threads=float(S["threads"]),
            xip9=m["xip"]["noise_like_full_survey_over_sigma_official"], xim9=m["xim"]["noise_like_full_survey_over_sigma_official"],
            max_xip=m["xip"]["max_abs_noise_like_full_survey_over_sigma_official"], max_xim=m["xim"]["max_abs_noise_like_full_survey_over_sigma_official"],
            E6=m["max_abs_En_6_noise_like"], E20=m["max_abs_En_20_noise_like"], En=e.tolist(), En_own_max20=float(np.abs(e).max()),
            En_patch_unscaled_max20=m["max_abs_En_20_patch"], sign_changes_20=nsc, lag1_autocorr=ac,
            d9_over_sigma_patch_xip=m["xip"]["d9_over_sigma_patch"], d9_over_sigma_patch_xim=m["xim"]["d9_over_sigma_patch"],
            npairs_ratio=m["npairs_ratio_to_exact"], Bn_max=float(np.abs(np.array(m["Bn_noise_like_full_survey_over_sigma_En"])).max()))
    if (0.0, 0.1) in L:
        d = {k: np.abs(L[(0.0,0.1)][k]-ex[k]).max() for k in ("xip","xim")}; res["0:0.1_vs_0:0_max_abs_fine"] = {k: float(v) for k, v in d.items()}
    (OUTD/"analysis.json").write_text(json.dumps(res, indent=1)); 
    for k, v in res.items():
        if isinstance(v, dict) and "E6" in v:
            print(k, f"t={v['seconds']:.0f}s xip {v['max_xip']:.3f} xim {v['max_xim']:.3f} E6 {v['E6']:.3f} E20 {v['E20']:.3f} signch {v['sign_changes_20']} ac {v['lag1_autocorr']:.2f}")
if __name__=="__main__" and sys.argv[1]=="analyze": analyze()

def extra():
    A = json.load(open(OUTD/"analysis.json")); F = json.load(open(ROOT/"runs/dr5_measure/angle_slop/fullpair_vs_exact.json"))["1.5:0.1"]
    e = np.array(A["1.5:0.1"]["En"]); f = np.array(F["En"]); n = min(len(e), len(f))
    out = {"En_ratio_south_over_full": (e[:n]/f[:n]).tolist(), "corr_En": float(np.corrcoef(e[:n], f[:n])[0,1])}
    for k in ("xip", "xim"):
        s = np.array(A["1.5:0.1"][k+"9"]); g = np.array(F[k+"9"]); out["corr_"+k+"9"] = float(np.corrcoef(s, g)[0,1])
        out["ratio_rms_"+k+"9_south_over_full"] = float(np.sqrt(np.mean(s**2))/np.sqrt(np.mean(g**2)))
    # patch-A same setting for En correlation with full pair: use A grid
    gz = ROOT/"runs/dr5_measure/angle_slop/grid"
    import importlib.util
    spec = importlib.util.spec_from_file_location("st", ROOT/"scripts/dr5_angle_slop_study.py"); st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
    ctx = st.official_ctx(6, 6)
    for P in ("A", "B", "S"):
        fe, fs = gz/f"{P}_Bin6_Bin6_b0_a0.npz", gz/f"{P}_Bin6_Bin6_b1.5_adef.npz"
        if not (fe.exists() and fs.exists()): continue
        ex = st.load_setting(fe); S = st.load_setting(fs); sc, _ = st.fine_scale(ex, ctx); mf = ctx["mfac"]
        Ed, _ = st.cosebis_E(ctx, ex["r_nom"], (S["xip"]-ex["xip"])*sc["xip"]/mf, (S["xim"]-ex["xim"])*sc["xim"]/mf)
        ee = Ed/ctx["sE"]; out[f"patch{P}_En_corr_with_full"] = float(np.corrcoef(ee[:n], f[:n])[0,1])
        out[f"patch{P}_En_ratio_first6"] = (ee[:6]/f[:6]).tolist(); out[f"patch{P}_En20"] = ee.tolist()
    # fractional deviation of south patch 1.5:0.1 vs exact per 9 bin (coherent bias test)
    ex = np.load(OUTD/"S_Bin6_Bin6_b0_a0.npz"); S = np.load(OUTD/"S_Bin6_Bin6_b1.5_a0.1.npz")
    for k in ("xip", "xim"):
        x9 = st.rebin9(ex["r_nom"], ex["weight"], ex[k]); d9 = st.rebin9(ex["r_nom"], ex["weight"], S[k]) - x9
        out["frac_dev_"+k] = (d9/x9).tolist()
    (OUTD/"extra.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:3000])
if __name__=="__main__" and sys.argv[1]=="extra": extra()
