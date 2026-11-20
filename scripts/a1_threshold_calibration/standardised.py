# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... 
# One-off A1b threshold-calibration study, not part of the Colab chain; see scripts/README.md.
"""Per-curve standardised criterion (i): calibration + power study.  Usage: standardised.py <stage>
stages: nulls | real | power | robust | plots   (each caches to runs/.../standardised/)."""
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[2]      # repository root
import sys, json, time
sys.path.insert(0, str(_R) + "/scripts/a1_threshold_calibration")
import numpy as np
from stdcommon import *

NA, NB, NC = 12000, 6000, 4000
CH = 250


def run_null(S, n, seed, **kw):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(0, n, CH):
        tp, tm = S.sim_their(min(CH, n - i), rng, **kw)
        parts.append(S.stats(tp, tm))
    return merge(parts)


def merge(parts):
    out = {}
    for k, v in parts[0].items():
        if isinstance(v, dict):
            out[k] = {kk: np.concatenate([p[k][kk] for p in parts]) for kk in v}
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


def save_st(path, st):
    flat = {}
    for k, v in st.items():
        if isinstance(v, dict):
            for kk, vv in v.items(): flat[f"{k}__{kk}"] = vv
        else: flat[k] = v
    np.savez_compressed(path, **flat)


def load_st(path):
    d = np.load(path); out = {}
    for k in d.files:
        if "__" in k:
            a, b = k.split("__"); out.setdefault(a, {})[b] = d[k]
        else: out[k] = d[k]
    return out


def stage_nulls():
    S = Study(); t = time.time()
    # cross-curve correlation of the boxcar means as well (the quantity that matters for the ratio)
    zb = (S.z / S.prof).reshape(42, 50, 20).mean(-1); cb = np.corrcoef(zb); iu = np.triu_indices(42, 1)
    print("rms corr of z/prof (lag 0): %.3f ; of its 20-bin means: %.3f (indep expectation 0.032 / 0.14)" % (np.sqrt((S.corr[iu] ** 2).mean()), np.sqrt((cb[iu] ** 2).mean())))
    for name, seed, n in (("A", 101, NA), ("B", 102, NB), ("C", 103, NC)):
        st = run_null(S, n, seed); save_st(OUT / f"null_{name}.npz", st); print(name, n, "%.0fs" % (time.time() - t), flush=True)


def get_tables():
    return Tables(load_st(OUT / "null_A.npz"))


def thresholds(tab):
    B = all_statistics(load_st(OUT / "null_B.npz"), tab)
    return {k: float(np.percentile(v, 99)) for k, v in B.items()}, B


def stage_real():
    S = Study(); tab = get_tables(); thr, B = thresholds(tab)
    C = all_statistics(load_st(OUT / "null_C.npz"), tab)
    stc = load_st(OUT / "null_C.npz"); stB = load_st(OUT / "null_B.npz")
    res = {"thresholds_1pct": thr, "false_fail_on_C": {k: float((C[k] > thr[k]).mean()) for k in thr}}
    res["const_fixed_false_fail"] = {c: float((np.concatenate([stB["ratio"].max(1), stc["ratio"].max(1)]) > c).mean()) for c in (0.7, 0.81, 0.9, 1.0, 1.33)}
    st = S.stats(S.rp[None], S.rm[None])
    real = all_statistics(st, tab)
    res["real"] = {k: float(v[0]) for k, v in real.items()}
    res["real_pte"] = {k: float(((np.concatenate([B[k], C[k]])) >= real[k][0]).mean()) for k in real}
    # per-curve table
    r = st["ratio"][0].astype(float)
    percurve = {}
    for m in METHODS:
        rp = st[m]["r_pred"][0].astype(float); D = r ** 2 - rp ** 2
        p = tab.pvals(m, D[None])[0]
        percurve[m] = [{"curve": LAB[i], "ratio": float(r[i]), "r_pred": float(rp[i]), "rho1": float(st[m]["rho"][0, i, 0]), "rho2": float(st[m]["rho"][0, i, 1]),
                        "S": float(r[i] / rp[i]), "D": float(D[i]), "p": float(p[i])} for i in range(42)]
    res["per_curve"] = percurve
    # null distribution diagnostics: homogeneity of D, S
    nA = load_st(OUT / "null_A.npz")
    res["null_diag"] = {}
    for m in METHODS:
        rr = nA["ratio"].astype(float); rp = nA[m]["r_pred"].astype(float); D = rr ** 2 - rp ** 2
        res["null_diag"][m] = {"D_mean_range": [float(D.mean(0).min()), float(D.mean(0).max())], "D_sd_range": [float(D.std(0).min()), float(D.std(0).max())],
                               "S_mean": float((rr / rp).mean()), "S_sd_over_mean": float(((rr / rp).std(0) / (rr / rp).mean(0)).mean()),
                               "ratio_sd_over_mean": float((rr.std(0) / rr.mean(0)).mean()),
                               "rho_est_sd_lag1": float(nA[m]["rho"][..., 0].std(0).mean()), "rho_est_sd_lag2": float(nA[m]["rho"][..., 1].std(0).mean())}
    # heteroscedasticity: hom vs het r_pred at the real data, and null-sim mean ratio vs r_pred
    z = S.z
    res["het"] = {}
    for m in ("plain", "hp15"):
        a = A.standardised_stats(z, m, True)["r_pred"]; b = A.standardised_stats(z, m, False)["r_pred"]
        res["het"][m] = {"max_abs_rel_diff_het_vs_hom_r_pred": float(np.abs(a / b - 1).max()), "median_rel": float(np.median(a / b - 1))}
    res["het"]["null_mean_ratio_over_r_pred_plain"] = float((nA["ratio"] / nA["plain"]["r_pred"]).mean())
    json.dump(res, open(OUT / "real_and_calibration.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("thresholds_1pct", "false_fail_on_C", "const_fixed_false_fail", "real", "real_pte", "het")}, indent=1))
    for m in METHODS:
        top = sorted(percurve[m], key=lambda d: d["p"])[:3]; print(m, "lowest p curves:", [(d["curve"], round(d["p"], 4)) for d in top])


POWER_EXTRA_AMPS = {"tilt xi- 1-1 only": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8], "tilt pair 4-4 only (xi+,xi-)": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
                    "tilt xi- only (all pairs)": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3], "tilt": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
                    "bump at 20'": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4], "step <5'": [0.05, 0.1, 0.15, 0.2, 0.3]}


def stage_power(NP=600):
    S = Study(); tab = get_tables(); thr, B = thresholds(tab)
    amps = dict(S.fam_amps); amps.update(POWER_EXTRA_AMPS)
    rows, contam = [], []
    for fam, am in amps.items():
        for a in am:
            rng = np.random.default_rng(hash((fam, a)) % 2 ** 31)
            # --- real data single draw
            tp, tm = inject(S.fam, fam, a, S.rp, S.rm, S.asc, np.random.default_rng(7))
            stR = S.stats(tp[None], tm[None]); realv = all_statistics(stR, tab)
            # --- null + injection probability
            parts, base_parts = [], []
            for i in range(0, NP, CH):
                n = min(CH, NP - i)
                tp0, tm0 = S.sim_their(n, rng)
                tpi, tmi = inject(S.fam, fam, a, tp0, tm0, S.asc, rng)
                parts.append(S.stats(tpi, tmi)); base_parts.append(S.stats(tp0, tm0))
            st = merge(parts); st0 = merge(base_parts)
            v = all_statistics(st, tab)
            row = {"family": fam, "amp": a,
                   "real": {k: bool(realv[k][0] > thr[k]) for k in thr},
                   "real_const_fixed": {"1.0": bool(stR["ratio"].max() > 1.0), "0.81": bool(stR["ratio"].max() > 0.81)},
                   "prob": {k: float((v[k] > thr[k]).mean()) for k in thr},
                   "prob_const_fixed": {"1.0": float((st["ratio"].max(1) > 1.0).mean()), "0.81": float((st["ratio"].max(1) > 0.81).mean())},
                   "crit2_prob": float((S.crit2_pte(st) < 0.01).mean())}
            rows.append(row)
            for m in METHODS:
                contam.append({"family": fam, "amp": a, "method": m,
                               "d_rho1": float((st[m]["rho"][..., 0] - st0[m]["rho"][..., 0]).mean()), "d_rho2": float((st[m]["rho"][..., 1] - st0[m]["rho"][..., 1]).mean()),
                               "d_rho1_max_curve": float((st[m]["rho"][..., 0] - st0[m]["rho"][..., 0]).mean(0).max()),
                               "mean_r_pred_ratio": float((st[m]["r_pred"] / st0[m]["r_pred"]).mean()), "mean_ratio_inj_over_null": float((st["ratio"] / st0["ratio"]).mean())})
            print(fam, a, {k: round(row["prob"][k], 2) for k in ("const", "D|plain", "D|hp15", "minp|hp15", "fisher|hp15", "stouffer|hp15")}, row["prob_const_fixed"], flush=True)
    json.dump({"rows": rows, "contam": contam, "thr": thr}, open(OUT / "power.json", "w"), indent=1)


def stage_robust():
    """Sensitivity of the 1% thresholds to the null model: independent vs correlated curves, profile window,
    lag-1/2 misspecification (+-0.03), and a hom (no profile) null."""
    S0 = Study(); tab = get_tables(); thr, _ = thresholds(tab); res = {}
    keys = ["const", "D|plain", "D|hp15", "D|hp31", "minp|hp15", "fisher|hp15", "stouffer|hp15", "Dz|hp15"]
    def run(S, label, seed, theta=None, prof=None, n=3000):
        st = run_null(S, n, seed, theta=theta, prof=prof) if (theta is not None or prof is not None) else run_null(S, n, seed)
        v = all_statistics(st, tab)
        res[label] = {"q99": {k: float(np.percentile(v[k], 99)) for k in keys}, "false_fail_at_baseline_thr": {k: float((v[k] > thr[k]).mean()) for k in keys}}
        print(label, {k: round(res[label]["false_fail_at_baseline_thr"][k], 3) for k in keys}, flush=True)
    run(S0, "baseline (indep, win51)", 201)
    Sc = Study(correlated=S0.corr); run(Sc, "correlated curves (sample corr)", 202)
    for w_ in (101, 201):
        prof = A.running_rms(S0.z - S0.z.mean(-1, keepdims=True), w_); run(S0, f"profile window {w_}", 203, prof=prof)
    run(S0, "homoscedastic profile (flat)", 204, prof=np.tile(S0.prof.mean(1, keepdims=True), (1, 1000)))
    for dl in (-0.04, 0.04):
        r1 = np.clip(S0.rho[:, 0] + dl, -0.6, 0.0); th_ = A.ma2_from_rho(r1, S0.rho[:, 1]); run(S0, f"lag-1 shifted {dl:+.2f}", 205, theta=th_)
    json.dump(res, open(OUT / "robustness.json", "w"), indent=1)


def stage_plots(est="hp15"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = json.load(open(OUT / "power.json")); rows = P["rows"]
    fams = list(dict.fromkeys(r["family"] for r in rows))
    meth = [("const 1.0 (adopted)", lambda r: r["prob_const_fixed"]["1.0"], "k", "-"),
            ("const 0.81 (FF 79%!)", lambda r: r["prob_const_fixed"]["0.81"], "0.6", "--"),
            ("const 1.03 (1% size)", lambda r: r["prob"]["const"], "tab:blue", "-"),
            (f"S=ratio/r_pred ({est})", lambda r: r["prob"][f"S|{est}"], "tab:orange", ":"),
            (f"D=ratio2-rpred2 ({est})", lambda r: r["prob"][f"D|{est}"], "tab:green", "-"),
            (f"min-p ({est})", lambda r: r["prob"][f"minp|{est}"], "tab:red", "-"),
            (f"Fisher ({est})", lambda r: r["prob"][f"fisher|{est}"], "tab:purple", "-"),
            (f"Stouffer ({est})", lambda r: r["prob"][f"stouffer|{est}"], "tab:brown", "--")]
    fig, axs = plt.subplots(2, 4, figsize=(18, 8)); axs = axs.ravel()
    for ax, f in zip(axs, fams):
        rr = sorted([r for r in rows if r["family"] == f], key=lambda r: r["amp"])
        x = [r["amp"] for r in rr]
        for n, g, c, ls in meth: ax.plot(x, [g(r) for r in rr], color=c, ls=ls, marker="o", ms=3, label=n)
        if f == "pair scale scatter": ax.plot(x, [r["crit2_prob"] for r in rr], color="m", ls="-.", label="criterion (ii)")
        ax.set_title(f, fontsize=10); ax.set_ylim(-0.03, 1.03); ax.axhline(0.01, color="gray", lw=0.5); ax.set_xlabel("injection amplitude"); ax.set_ylabel("detection prob.")
    for ax in axs[len(fams):]: ax.axis("off")
    axs[0].legend(fontsize=7, loc="lower right"); fig.suptitle(f"Detection probability vs injection amplitude (600 null sims + injection; estimator {est}; 1% size for calibrated stats)")
    fig.tight_layout(); fig.savefig(OUT / "power_curves.png", dpi=110)
    # null/real diagnostics plot
    st = load_st(OUT / "null_A.npz"); R = json.load(open(OUT / "real_and_calibration.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    rr = st["ratio"].astype(float); rp = st[est]["r_pred"].astype(float)
    ax[0].errorbar(np.arange(42), rr.mean(0), rr.std(0), fmt="o", ms=3, label="null ratio mean+-sd"); ax[0].plot(np.arange(42), [d["ratio"] for d in R["per_curve"][est]], "r*", label="real")
    ax[0].plot(np.arange(42), [d["r_pred"] for d in R["per_curve"][est]], "k_", label="real r_pred"); ax[0].axhline(1.0, c="gray"); ax[0].axhline(0.81, c="gray", ls=":"); ax[0].legend(fontsize=7); ax[0].set_xlabel("curve (0-20 xi+, 21-41 xi-)"); ax[0].set_title("per-curve boxcar ratio")
    D = rr ** 2 - rp ** 2; ax[1].errorbar(np.arange(42), D.mean(0), D.std(0), fmt="o", ms=3); ax[1].plot(np.arange(42), [d["D"] for d in R["per_curve"][est]], "r*"); ax[1].set_title(f"D = ratio^2 - r_pred^2 ({est}): null mean+-sd, real"); ax[1].set_xlabel("curve")
    p = np.array([d["p"] for d in R["per_curve"][est]]); ax[2].hist(p, bins=np.linspace(0, 1, 11), color="C0"); ax[2].set_title("real-data per-curve null p-values (uniform if null)"); ax[2].set_xlabel("p")
    fig.tight_layout(); fig.savefig(OUT / "null_and_real.png", dpi=110)


def sim_acf_z(acf, prof, n, rng):
    """Gaussian stationary z with autocorrelation acf (C, L+1) (acf[:,0]=1), heteroscedastic by prof (spectral method)."""
    C, N = prof.shape; M = 4096
    w = np.arange(L := acf.shape[1])
    om = 2 * np.pi * np.arange(M // 2 + 1) / M
    f = acf[:, :1] + 2 * (acf[:, None, 1:] * np.cos(om[None, :, None] * np.arange(1, L)[None, None, :])).sum(-1)
    f = np.clip(f, 1e-6, None)
    e = rng.standard_normal((n, C, M))
    x = np.fft.irfft(np.fft.rfft(e, axis=-1) * np.sqrt(f)[None], n=M, axis=-1)[..., :N]
    return x * prof[None]


def stage_tail(NT=4000, NR=3000):
    """Null-model misspecification: add the POOLED real lag 3..12 autocorrelation (mean over curves) to each curve's
    MA(2) null, rebuild the per-curve tables from that alt-null and recompute the real-data pte of each statistic."""
    S = Study(); res = {}
    Rz = np.load(ROOT / "runs/dr5_measure/a1_salvage/threshold_calibration/z.npz")["R"]       # (42, 40) raw ACF lags 1..40
    tail = Rz.mean(0)[2:12]
    k = np.arange(1, 13)
    th = S.theta
    base = (np.stack([np.ones(42)] + [np.zeros(42)] * 12, 1)); acf = np.zeros((42, 13)); acf[:, 0] = 1
    den = 1 + th[:, 0] ** 2 + th[:, 1] ** 2
    acf[:, 1] = (th[:, 0] + th[:, 0] * th[:, 1]) / den; acf[:, 2] = th[:, 1] / den
    for lbl, tl in (("MA2 only (spectral sim check)", np.zeros(10)), ("MA2 + pooled lags 3-12", tail)):
        a = acf.copy(); a[:, 3:13] = tl[None]
        rng = np.random.default_rng(31)
        def run(n):
            parts = []
            for i in range(0, n, 100):
                zz = sim_acf_z(a, S.prof, min(100, n - i), rng)
                tp = S.Ap[:, None] * S.asc["xip"] + S.asc["sigma_xip"] * zz[:, :21]; tm = S.Am[:, None] * S.asc["xim"] + S.asc["sigma_xim"] * zz[:, 21:]
                parts.append(S.stats(tp, tm))
            return merge(parts)
        stT = run(NT); stR = run(NR)
        tab = Tables(stT); real = all_statistics(S.stats(S.rp[None], S.rm[None]), tab); nul = all_statistics(stR, tab)
        keys = ["const", "minp|hp15", "fisher|hp15", "stouffer|hp15", "minp|plain", "fisher|plain", "stouffer|plain", "fisher|hp31"]
        res[lbl] = {"tail": [float(x) for x in tl], "q99": {q: float(np.percentile(nul[q], 99)) for q in keys}, "real": {q: float(real[q][0]) for q in keys},
                    "real_pte": {q: float((nul[q] >= real[q][0]).mean()) for q in keys}}
        print(lbl, res[lbl]["real_pte"], flush=True)
    json.dump(res, open(OUT / "tail_misspec.json", "w"), indent=1)


def stage_equiv(NS=200):
    """Which injection amplitude reproduces the real-data statistics?  median/16/84 pct of each statistic vs amplitude."""
    S = Study(); tab = get_tables(); thr, _ = thresholds(tab)
    real = all_statistics(S.stats(S.rp[None], S.rm[None]), tab); keys = ["minp|hp15", "fisher|hp15", "stouffer|hp15", "fisher|plain", "const"]
    res = {"real": {k: float(real[k][0]) for k in keys}, "thr": {k: thr[k] for k in keys}, "curves": {}}
    grid = {"tilt": [0, 0.02, 0.04, 0.06, 0.08, 0.1], "bump at 20'": [0, 0.02, 0.05, 0.08, 0.1, 0.12], "step <5'": [0, 0.04, 0.08, 0.1, 0.12]}
    for fam, am in grid.items():
        res["curves"][fam] = {}
        for a in am:
            rng = np.random.default_rng(77)
            parts = []
            for i in range(0, NS, 100):
                tp0, tm0 = S.sim_their(100, rng); tpi, tmi = inject(S.fam, fam, a, tp0, tm0, S.asc, rng); parts.append(S.stats(tpi, tmi))
            v = all_statistics(merge(parts), tab)
            res["curves"][fam][str(a)] = {k: [float(x) for x in np.percentile(v[k], [16, 50, 84])] for k in keys}
            print(fam, a, {k: round(float(np.median(v[k])), 2) for k in keys}, flush=True)
    json.dump(res, open(OUT / "equivalent_amplitude.json", "w"), indent=1)


def stage_save():
    """Write the shipped null table for the library gate (hp15) and smoke-test a1b_noise_standardised on the real data."""
    S = Study(); tab = get_tables(); thr, _ = thresholds(tab)
    null = {"method": "hp15", "size": 0.01, "sortedD": tab.sortedD["hp15"], "thr": {k: thr[f"{k}|hp15"] for k in ("minp", "fisher", "stouffer")}}
    A.save_null(OUT / "null_table_hp15.npz", null)
    r = A.a1b_noise_standardised(S.rp, S.rm, S.asc, S.sd, null=A.load_null(OUT / "null_table_hp15.npz"))
    r0 = A.a1b_noise(S.rp, S.rm, S.asc, S.sd)
    out = {"pass_standardised_minp": r["pass"], "pass_adopted_gate": r0["pass"], "standardised": {k: r["standardised"][k] for k in ("value", "threshold", "all", "thr_all", "argmin_p")},
           "argmin_curve": LAB[r["standardised"]["argmin_p"]], "const_pte": {k: r[k]["const_pte"] for k in ("xip", "xim")}}
    json.dump(out, open(OUT / "library_smoke_real.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"nulls": stage_nulls, "real": stage_real, "power": stage_power, "robust": stage_robust, "tail": stage_tail, "equiv": stage_equiv, "save": stage_save, "plots": lambda: stage_plots(*sys.argv[2:])}[sys.argv[1]]()
