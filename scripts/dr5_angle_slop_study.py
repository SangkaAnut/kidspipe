"""TreeCorr (bin_slop, angle_slop) accuracy/cost study on real KiDS-Legacy DR5 shear catalogues.

Subcommands
  run     measure one or more settings on a real-data patch (resumable; one npz per setting in
          runs/dr5_measure/angle_slop/grid/).  Production binning (1000 log bins, 2-300', sep_units arcmin) and the
          chunked driver's ordered call process(cat, cat) (same object twice => every pair counted twice).
  brute   prove that (bin_slop 0, angle_slop 0) equals TreeCorr brute force on a random 20k-galaxy subset
  chunks  time settings on real full-pair Bin6_Bin6 chunk-pairs (16 RA strips as the production driver)
  analyze reduce all stored runs to study.json, ANGLE_SLOP.md-ready tables and plots

Patches (bin-6 / bin-1 galaxies selected by RA/Dec box, North; all full density):
  A  ra 165-185, dec -5..4      B  ra 195-215, dec -5..4   (disjoint)      S  small test piece
Everything is written below runs/dr5_measure/angle_slop/ ; xi_chunked/ is never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "dr5_measure" / "angle_slop"
GRID = OUT / "grid"
PATCHES = {"A": (165.0, 185.0, -5.0, 4.0),      # 20 x 9 deg = ~180 deg2, 690k bin-6 galaxies
           "B": (195.0, 215.0, -5.0, 4.0),      # disjoint from A, ~190 deg2
           "S": (170.0, 176.0, -3.0, 1.0)}      # ~24 deg2 piece for cheap exactness / timing tests


def synth_signal(ra, dec, seed=11, n_lens=300, ra0=None, dec0=None):
    """Noiseless spin-2 test field on the REAL galaxy positions: tangential shear around n_lens random 'lenses' with
    log-uniform scale 2'-90' and random sign, g = -A thetas/(theta+thetas) exp(2 i phi) (flat sky, sky convention).
    It has xi+ and xi- signal at all separations 2-300' with no shape noise, so (setting - exact)/exact is a pure
    algorithmic bias fraction (it does not shrink with area)."""
    rng = np.random.default_rng(seed)
    lo, hi = ra.min(), ra.max(); dlo, dhi = dec.min(), dec.max()
    g = np.zeros(ra.size, complex)
    for _ in range(n_lens):
        rl, dl = rng.uniform(lo, hi), rng.uniform(dlo, dhi)
        ts = np.exp(rng.uniform(np.log(2.0), np.log(90.0))) / 60.0          # deg
        amp = 0.05 * rng.choice([-1.0, 1.0])
        dx = (ra - rl) * np.cos(np.deg2rad(dec)); dy = dec - dl
        th = np.hypot(dx, dy)
        g += -amp * ts / (th + ts) * np.exp(2j * np.arctan2(dy, dx))
    return g.real.copy(), g.imag.copy()


def patch_cat(b: int, patch: str, signal: bool = False):
    from kidspipe.measure.xi import load_bin, make_treecorr_cat
    r = load_bin(b)
    lo, hi, d0, d1 = PATCHES[patch]
    m = (r["ra"] >= lo) & (r["ra"] < hi) & (r["dec"] >= d0) & (r["dec"] < d1)
    rec = {k: r[k][m] for k in ("ra", "dec", "e1", "e2", "w")}
    if signal:
        rec["e1"], rec["e2"] = synth_signal(rec["ra"], rec["dec"])
    return make_treecorr_cat(rec), int(m.sum())


def tag(pair: str, patch: str, bs: float, a) -> str:
    return f"{patch}_{pair}_b{bs:g}_a{'def' if a is None else format(a, 'g')}"


def parse_settings(s: str):
    out = []
    for tok in s.split(","):
        b, a = tok.split(":")
        out.append((float(b), None if a in ("def", "None") else float(a)))
    return out


def cmd_run(args):
    from kidspipe.measure.xi import run_pair, corr_to_arrays
    GRID.mkdir(parents=True, exist_ok=True)
    pair = args.pair                       # "6-6" or "1-6"
    i, j = (int(x) for x in pair.split("-"))
    sig = getattr(args, "signal", False)
    assert not (sig and i != j), "signal-only test is for the auto pair"
    c1, n1 = patch_cat(i, args.patch, sig)
    c2, n2 = (c1, n1) if i == j else patch_cat(j, args.patch)
    pfx = ("SIG" + args.patch) if sig else args.patch
    print(f"patch {args.patch} pair {pair}: {n1:,} x {n2:,} galaxies", flush=True)
    for bs, a in parse_settings(args.settings):
        f = GRID / (tag(f"Bin{i}_Bin{j}", pfx, bs, a) + ".npz")
        if f.exists() and not args.force:
            print("skip", f.name); continue
        t = time.time()
        gg = run_pair(c1, c2, bin_slop=bs, num_threads=args.threads, angle_slop=a)
        dt = time.time() - t
        arr = corr_to_arrays(gg)
        arr.update(seconds=np.float64(dt), bin_slop=np.float64(bs), angle_slop=np.float64(np.nan if a is None else a),
                   ngal1=np.float64(n1), ngal2=np.float64(n2), threads=np.float64(args.threads))
        np.savez(f, **arr)
        print(f"{f.name}: {dt:8.1f} s  npairs={arr['npairs'].sum():.4e}", flush=True)


def cmd_brute(args):
    import treecorr
    from kidspipe.measure.xi import treecorr_kwargs, make_treecorr_cat, load_bin
    r = load_bin(6)
    m0 = (r["ra"] > 196) & (r["ra"] < 206) & (r["dec"] > -5) & (r["dec"] < 3)
    idx = np.where(m0)[0]
    res = {}
    for seed in (1, 2, 3):
        pick = np.sort(np.random.default_rng(seed).choice(idx, 20000, replace=False))
        cat = make_treecorr_cat({k: r[k][pick] for k in ("ra", "dec", "e1", "e2", "w")})
        g = {}
        for lab, (bs, a, br) in {"brute": (0.0, None, True), "0:0": (0.0, 0.0, False), "0:def": (0.0, None, False),
                                 "1.5:0": (1.5, 0.0, False), "1.5:def": (1.5, None, False)}.items():
            kw = treecorr_kwargs(1000, bs, args.threads, a)
            if br:
                kw["brute"] = True
            gg = treecorr.GGCorrelation(kw)
            gg.process(cat, cat)
            g[lab] = gg
        ref = g["brute"]
        for lab in ("0:0", "0:def", "1.5:0", "1.5:def"):
            d = {}
            for k in ("xip", "xim"):
                s = np.sqrt(ref.varxip)
                diff = np.abs(getattr(g[lab], k) - getattr(ref, k))
                d[k + "_max_abs_over_sigma"] = float(np.nanmax(diff / s))
                d[k + "_max_abs"] = float(diff.max())
            d["npairs_max_rel"] = float(np.max(np.abs(g[lab].npairs - ref.npairs) / np.maximum(ref.npairs, 1)))
            res[f"seed{seed}_{lab}"] = d
            print(seed, lab, d, flush=True)
    (OUT / "brute_check.json").write_text(json.dumps(res, indent=1))


CT = OUT / "chunk_timing"
STORED = ROOT / "runs" / "dr5_measure" / "xi_chunked" / "chunks" / "Bin6_Bin6"      # READ ONLY


def cmd_chunks(args):
    """Time settings on real full-pair Bin6_Bin6 chunk-pairs (16 equal-count RA strips, as the production driver)."""
    import importlib.util
    from kidspipe.measure.xi import run_pair, corr_to_arrays
    spec = importlib.util.spec_from_file_location("drv", ROOT / "scripts" / "dr5_measure_xi_chunked.py")
    drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
    CT.mkdir(parents=True, exist_ok=True)
    cats, _ = drv.load_pair_cats(6, 6, 16, {})
    for cp in args.cp.split(","):
        a, b = (int(x) for x in cp.split("-"))
        for bs, al in parse_settings(args.settings):
            f = CT / f"Bin6_Bin6_c{a:02d}_{b:02d}_b{bs:g}_a{'def' if al is None else format(al, 'g')}.npz"
            if f.exists() and not args.force:
                print("skip", f.name); continue
            t = time.time()
            arr = corr_to_arrays(run_pair(cats[a], cats[b], bin_slop=bs, num_threads=args.threads, angle_slop=al))
            dt = time.time() - t
            arr.update(seconds=np.float64(dt), bin_slop=np.float64(bs), angle_slop=np.float64(np.nan if al is None else al),
                       threads=np.float64(args.threads))
            np.savez(f, **arr)
            print(f"{f.name}: {dt:8.1f} s npairs={arr['npairs'].sum():.4e}", flush=True)


# ================================================================================================= analysis
THEIRS_DIR = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")   # READ ONLY
M_GOLD = {1: -0.023, 2: -0.016, 3: -0.011, 4: 0.020, 5: 0.030, 6: 0.045}         # Wright+25 Table 1 (as dr5_adopt_01)
EDGES9 = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
RANGES = {"2-10": (2.0, 10.0), "10-60": (10.0, 60.0), "60-300": (60.0, 300.01)}
AEFF = 0.1                                   # TreeCorr 5.1.4 default angle_slop (corr2base._default_angle_slop)
FIN = 49.0                                   # production hours at (0, default) on one 16-thread process (CHUNKING.md)


def rebin9(r, w, y):
    o = np.zeros(9)
    for c in range(9):
        m = (r >= EDGES9[c]) & (r < (EDGES9[c + 1] if c < 8 else 300.01))
        o[c] = np.sum(w[m] * y[m]) / np.sum(w[m])
    return o


def rebin9_var(r, w, v):
    o = np.zeros(9)
    for c in range(9):
        m = (r >= EDGES9[c]) & (r < (EDGES9[c + 1] if c < 8 else 300.01))
        o[c] = np.sum(w[m] ** 2 * v[m]) / np.sum(w[m]) ** 2
    return o


def official_ctx(i, j):
    from kidspipe.measure import official as of
    from kidspipe.theory.cosebis import CosebisFilters
    K = of.PAIRS.index((i - 1, j - 1))
    fx = of.load_official_xipm_fits(); cov = fx["cov"]; n = 189
    cb = of.load_official_cosebis(); cc = cb["cov"]
    idx = K * 20 + np.arange(20)
    th = {k: np.asarray(v, float) for k, v in np.load(THEIRS_DIR / f"xipm_bin{i}_bin{j}.npz", allow_pickle=True).items()
          if np.asarray(v).dtype.kind == "f"}
    return dict(K=K, i=i, j=j, mfac=(1 + M_GOLD[i]) * (1 + M_GOLD[j]),
                sig9={"xip": np.sqrt(np.diag(cov)[K * 9:(K + 1) * 9]), "xim": np.sqrt(np.diag(cov)[n + K * 9:n + (K + 1) * 9])},
                val9={"xip": fx["xiP"][K], "xim": fx["xiM"][K]}, Eoff=cb["En"][K], sE=np.sqrt(np.diag(cc[np.ix_(idx, idx)])), theirs=th,
                filt=CosebisFilters(2.0, 300.0, source="mpmath", dps=80))


def cosebis_E(ctx, theta, xp, xm, nmodes=20):
    from kidspipe.measure.xi2cosebis import e_from_xi_binned
    E, B = e_from_xi_binned(theta, np.asarray(xp)[None], np.asarray(xm)[None], ctx["filt"], nmodes=nmodes, scheme="edges", unit="rad")
    return E[0], B[0]


def load_setting(f):
    z = np.load(f)
    return {k: np.asarray(z[k], float) for k in z.files}


def sname(bs, a):
    return f"{bs:g}:{'def' if a is None else format(a, 'g')}"


def scan_grid(patch, pair):
    """{(bin_slop, angle_slop_eff): arrays}; angle_slop 'def' is stored as 0.1 (TreeCorr default, verified in source)."""
    out = {}
    i, j = (int(x) for x in pair.split("-"))
    for f in sorted(GRID.glob(f"{patch}_Bin{i}_Bin{j}_b*_a*.npz")):
        d = load_setting(f)
        a = AEFF if np.isnan(d["angle_slop"]) else float(d["angle_slop"])
        key = (float(d["bin_slop"]), a)
        if key in out and not np.isnan(d["angle_slop"]):
            continue                                   # an explicit 0.1 run never overrides the default run
        d["is_default"] = bool(np.isnan(d["angle_slop"]))
        out[key] = d
    return out


def fine_scale(ex, ctx):
    """Per-fine-bin factor sigma_survey / sigma_patch (shape-noise sigma, same estimator): converts a patch deviation
    (setting - exact) into its full-survey-equivalent size.  sigma_survey = the external full-pair TreeCorr sigma (proper
    unordered auto / cross); the patch run is process(cat,cat) (ordered auto: TreeCorr varxi is half the true variance)."""
    th = ctx["theirs"]
    vfac = 2.0 if ctx["i"] == ctx["j"] else 1.0
    return {k: th["sigma_" + k] / np.sqrt(vfac * ex["varxi"]) for k in ("xip", "xim")}, vfac


def interp_f(f9):
    cen = np.sqrt(EDGES9[:-1] * EDGES9[1:])
    return lambda r: np.interp(np.log(r), np.log(cen), f9)


def metrics(ex, S, ctx, sigf=None):
    r, g = ex["r_nom"], ex["weight"]
    mf = ctx["mfac"]; th = ctx["theirs"]
    scale, vfac = fine_scale(ex, ctx)
    o = {"seconds": float(S["seconds"]), "npairs_ratio_to_exact": float(S["npairs"].sum() / ex["npairs"].sum())}
    dxi = {}
    for key in ("xip", "xim"):
        d = S[key] - ex[key]; dxi[key] = d
        x9 = rebin9(r, g, ex[key]); d9 = rebin9(r, g, S[key]) - x9
        sp9 = np.sqrt(rebin9_var(r, g, vfac * ex["varxi"]))
        dfull9 = rebin9(r, th["npairs"], d * scale[key])             # same deviation at full-survey noise level
        pooled = {}
        for nm, (lo, hi) in RANGES.items():
            m = (r >= lo) & (r < hi)
            pooled[nm] = float(np.sum(g[m] * d[m]) / np.sum(g[m] * ex[key][m]))
        o[key] = {"d9_over_sigma_patch": (d9 / sp9).tolist(),
                  "xi_exact9_over_sigma_patch": (x9 / sp9).tolist(),
                  "d9_patch_over_sigma_official": (d9 / mf / ctx["sig9"][key]).tolist(),
                  "noise_like_full_survey_over_sigma_official": (dfull9 / mf / ctx["sig9"][key]).tolist(),
                  "max_abs_noise_like_full_survey_over_sigma_official": float(np.abs(dfull9 / mf / ctx["sig9"][key]).max()),
                  "rms_d9_over_sigma_patch": float(np.sqrt(np.mean((d9 / sp9) ** 2))),
                  "frac_pooled_noisy": pooled,
                  "rms_fine_d_over_sigma_patch": float(np.sqrt(np.mean((d / np.sqrt(vfac * ex["varxi"])) ** 2)))}
    # E_n / B_n: deviation (linear in xi) scaled to the full-survey noise level, and the patch deviation as is
    sE = ctx["sE"]
    Ed, Bd = cosebis_E(ctx, r, dxi["xip"] * scale["xip"] / mf, dxi["xim"] * scale["xim"] / mf)
    Ep, _ = cosebis_E(ctx, r, dxi["xip"] / mf, dxi["xim"] / mf)
    o["En_noise_like_full_survey_over_sigma_official"] = (Ed / sE).tolist()
    o["En_patch_over_sigma_official"] = (Ep / sE).tolist()
    o["Bn_noise_like_full_survey_over_sigma_En"] = (Bd / sE).tolist()
    o["max_abs_En_20_noise_like"] = float(np.abs(Ed / sE).max()); o["max_abs_En_6_noise_like"] = float(np.abs(Ed[:6] / sE[:6]).max())
    o["max_abs_En_20_patch"] = float(np.abs(Ep / sE).max())
    if sigf is not None:                                              # systematic (signal) part from the noiseless field
        E0, _ = cosebis_E(ctx, th["r_nom"], th["xip"], th["xim"])
        f = {k: interp_f(np.asarray(sigf[k]))(th["r_nom"]) for k in ("xip", "xim")}
        Ef, _ = cosebis_E(ctx, th["r_nom"], th["xip"] * (1 + f["xip"]), th["xim"] * (1 + f["xim"]))
        o["signal_f9"] = {k: list(map(float, sigf[k])) for k in ("xip", "xim")}
        for key in ("xip", "xim"):
            c = np.asarray(sigf[key]) * ctx["val9"][key] / ctx["sig9"][key]
            o[key]["signal_bias_over_sigma_official"] = c.tolist()
            o[key]["max_abs_signal_bias_over_sigma_official"] = float(np.abs(c).max())
        o["En_signal_over_sigma_official"] = ((Ef - E0) / sE).tolist()
        o["max_abs_En_20_signal"] = float(np.abs((Ef - E0) / sE).max())
    return o


def signal_fractions(patches=("SIGA", "SIGS")):
    """{setting_name: {xip: f9[9], xim: f9[9], patch}} = (setting - exact)/exact per 9-bin on the noiseless field."""
    out = {}
    for pn in patches:
        G = scan_grid(pn, "6-6")
        if (0.0, 0.0) not in G:
            continue
        ex = G[(0.0, 0.0)]; r, g = ex["r_nom"], ex["weight"]
        for (b, a), S in G.items():
            nm = sname(b, a)
            if nm in out:
                continue
            fr = {}
            for key in ("xip", "xim"):
                x9 = rebin9(r, g, ex[key]); fr[key] = ((rebin9(r, g, S[key]) - x9) / x9)
            out[nm] = {"xip": fr["xip"].tolist(), "xim": fr["xim"].tolist(), "patch": pn, "seconds": float(S["seconds"]),
                       "xi_exact9_min_abs": {k: float(np.abs(rebin9(r, g, ex[k])).min()) for k in ("xip", "xim")}}
    return out


def reduce_patch(patch, pair, ctx, sigfr=None):
    G = scan_grid(patch, pair)
    if (0.0, 0.0) not in G:
        return None
    ex = G[(0.0, 0.0)]
    ref = G.get((0.0, AEFF))
    res = {"ngal": [float(ex["ngal1"]), float(ex["ngal2"])], "exact_seconds": float(ex["seconds"]), "settings": {}}
    for (b, a), S in sorted(G.items()):
        nm = sname(b, a)
        sf = None if (sigfr is None or nm not in sigfr) else sigfr[nm]
        m = metrics(ex, S, ctx, sf)
        m["seconds_ratio_vs_0_def"] = float(S["seconds"] / ref["seconds"]) if ref is not None else None
        m["hours_if_49h_at_0_def"] = None if ref is None else float(FIN * S["seconds"] / ref["seconds"])
        m["is_default_run"] = bool(S["is_default"])
        res["settings"][nm] = m
    return res


def cmd_analyze(args):
    out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "treecorr_default_angle_slop": AEFF, "production_hours_at_0_def": FIN,
           "note_def": "setting names are bin_slop:angle_slop with the TreeCorr default angle_slop written 0.1"}
    bf = OUT / "brute_check.json"
    if bf.exists():
        out["brute_check"] = json.loads(bf.read_text())
    sigfr = signal_fractions()
    out["signal_fractions"] = sigfr
    for pair in args.pairs.split(","):
        i, j = (int(x) for x in pair.split("-"))
        ctx = official_ctx(i, j)
        out[f"Bin{i}_Bin{j}"] = {"mfac": ctx["mfac"], "official_sigma9": {k: v.tolist() for k, v in ctx["sig9"].items()},
                                  "official_val9": {k: v.tolist() for k, v in ctx["val9"].items()}, "sigma_En": ctx["sE"].tolist(),
                                  "patches": {}}
        for patch in args.patches.split(","):
            r = reduce_patch(patch, pair, ctx, sigfr if i == j else None)
            if r is not None:
                out[f"Bin{i}_Bin{j}"]["patches"][patch] = r
                print(f"{pair} patch {patch}: {len(r['settings'])} settings", flush=True)
    (OUT / "study_grid.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT / "study_grid.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run"); p.add_argument("--patch", default="A"); p.add_argument("--pair", default="6-6")
    p.add_argument("--settings", required=True, help="comma list bin_slop:angle_slop (def = TreeCorr default)")
    p.add_argument("--threads", type=int, default=16); p.add_argument("--force", action="store_true")
    p.add_argument("--signal", action="store_true", help="replace e1,e2 by a noiseless synthetic spin-2 field (tag SIG<patch>)")
    p.set_defaults(fn=cmd_run)
    p = sp.add_parser("chunks"); p.add_argument("--cp", required=True, help="comma list a-b of chunk-pairs")
    p.add_argument("--settings", required=True); p.add_argument("--threads", type=int, default=16)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_chunks)
    p = sp.add_parser("analyze"); p.add_argument("--pairs", default="6-6,1-6"); p.add_argument("--patches", default="A,B")
    p.set_defaults(fn=cmd_analyze)
    p = sp.add_parser("brute"); p.add_argument("--threads", type=int, default=16); p.set_defaults(fn=cmd_brute)
    a = ap.parse_args(); a.fn(a)
