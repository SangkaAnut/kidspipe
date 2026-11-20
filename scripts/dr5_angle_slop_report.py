# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Reduce runs/dr5_measure/angle_slop/study_grid.json (from `dr5_angle_slop_study.py analyze`) to study.json,
a generated markdown table (ANGLE_SLOP_generated.md) and plots/dr5_angle_slop_*.png.

All biases are in units of the OFFICIAL full-survey sigma (9-bin xi, E_n):
  noise-like : (setting - exact) measured on a real patch, rescaled bin by bin by sigma_survey/sigma_patch.  A typical
               size of the algorithmic deviation if it behaved like shape noise (contains that patch's random draw).
  signal     : (setting - exact)/exact on a noiseless synthetic spin-2 field on the real positions (patch SIGA / SIGS)
               times xi_official/sigma_official: the coherent bias the setting puts on the real signal.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dr5_measure" / "angle_slop"
PLOTS = ROOT / "plots"
FIN = 49.0
_E = np.exp(np.linspace(np.log(2.0), np.log(300.0), 10))
CEN = np.sqrt(_E[:-1] * _E[1:])
KEY = ["0:0", "0:0.1", "0.5:0.1", "1:0.1", "1:0.003", "1.5:0.1", "1.5:0.01", "1.5:0.005", "1.5:0.003", "1.5:0.002", "1.5:0.001"]


def mx(x):
    x = [v for v in x if v is not None]
    return float(np.max(np.abs(x))) if x else None


def collect():
    g = json.loads((OUT / "study_grid.json").read_text())
    rows = {}
    A = g["Bin6_Bin6"]["patches"]["A"]["settings"]
    refA = A["0:0.1"]["seconds"]
    for nm in A:
        rows[nm] = {"seconds_A_uncontended": A[nm]["seconds"], "ratio_vs_0_def": A[nm]["seconds"] / refA,
                    "hours": FIN * A[nm]["seconds"] / refA, "hours_basis": "patch A, 16 threads, uncontended"}
    S = g["Bin6_Bin6"]["patches"]["S"]["settings"]
    refS = S["0:0.1"]["seconds"]
    for nm in S:
        if nm not in rows:
            rows[nm] = {"seconds_A_uncontended": None, "ratio_vs_0_def": S[nm]["seconds"] / refS,
                        "hours": FIN * S[nm]["seconds"] / refS, "hours_basis": "small patch S ratio only (less reliable)"}
    for nm, r in rows.items():
        r["noise_like"], r["signal"] = {}, {}
        for pair in ("Bin6_Bin6", "Bin1_Bin6"):
            if pair not in g:
                continue
            for pt, P in g[pair]["patches"].items():
                m = P["settings"].get(nm)
                if m is None:
                    continue
                key = f"{pair}/{pt}"
                r["noise_like"][key] = {"xip9": m["xip"]["max_abs_noise_like_full_survey_over_sigma_official"],
                                        "xim9": m["xim"]["max_abs_noise_like_full_survey_over_sigma_official"],
                                        "En6": m["max_abs_En_6_noise_like"], "En20": m["max_abs_En_20_noise_like"],
                                        "xip9_by_bin": m["xip"]["noise_like_full_survey_over_sigma_official"],
                                        "xim9_by_bin": m["xim"]["noise_like_full_survey_over_sigma_official"],
                                        "seconds_this_run": m["seconds"]}
                if "max_abs_signal_bias_over_sigma_official" in m["xim"]:
                    r["signal"][key] = {"xip9": m["xip"]["max_abs_signal_bias_over_sigma_official"],
                                        "xim9": m["xim"]["max_abs_signal_bias_over_sigma_official"],
                                        "En20": m["max_abs_En_20_signal"],
                                        "xip9_by_bin": m["xip"]["signal_bias_over_sigma_official"],
                                        "xim9_by_bin": m["xim"]["signal_bias_over_sigma_official"]}
        nl, sg = r["noise_like"], r["signal"]
        w = {}
        for k in ("xip9", "xim9", "En6", "En20"):
            vals = [v[k] for v in nl.values()] + [v[k] for v in sg.values() if k in v]
            w[k] = float(max(vals)) if vals else None
        w["signal_xim9"] = mx([v["xim9"] for v in sg.values()])
        w["signal_xip9"] = mx([v["xip9"] for v in sg.values()])
        w["signal_En20"] = mx([v["En20"] for v in sg.values()])
        r["worst"] = w
        vals = [w[k] for k in ("xip9", "xim9", "En6", "En20") if w[k] is not None]
        r["worst_all"] = max(vals) if vals else None
        ab = [nl[f"Bin6_Bin6/{p}"]["xim9"] for p in ("A", "B") if f"Bin6_Bin6/{p}" in nl]
        r["stable_AB_xim9"] = None if len(ab) < 2 else bool(max(ab) < 2 * min(ab) + 0.02)
    return g, rows


def verdict(nm, r, f=None):
    w = r["worst_all"]
    if f:
        w = max(w or 0, f["max_xip9"], f["max_xim9"], f["max_En6"], f["max_En20"])
    if nm == "0:0":
        return "exact reference (= brute force to 1e-14)"
    if w is None:
        return "not evaluated"
    if w <= 0.03:
        return "negligible (<=0.03)"
    if w <= 0.1:
        return "acceptable (<=0.1)"
    return "too large (>0.1)"


def make_plots(rows):
    PLOTS.mkdir(exist_ok=True)
    names = [n for n in rows if rows[n]["noise_like"] and n != "0:0"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for a, (k, ttl) in zip(ax, (("xim9", "xi- (9 bins)"), ("En20", "E_n (20 modes)"))):
        for nm in names:
            h = rows[nm]["hours"]; y = max(rows[nm]["worst"][k], 1e-4)
            c = "tab:red" if nm.startswith("1.5") else "tab:blue" if nm.startswith("0.5") else "tab:green" if nm.startswith("1:") else "k"
            a.plot(h, y, "o", color=c, ms=6)
            if nm in KEY:
                a.annotate(nm, (h, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        a.axhline(0.1, color="grey", ls="--", lw=1)
        a.set_yscale("log"); a.set_xlabel("production hours = 49 h x runtime ratio vs 0:0.1 (uncontended)")
        a.set_ylabel("worst |bias| / official sigma"); a.set_title(ttl + ", worst over patches/pairs; dashed = 0.1 sigma"); a.grid(alpha=0.3)
    fig.suptitle("DR5 Bin6_Bin6 cost vs bias; setting = bin_slop:angle_slop (0.1 = TreeCorr default). red 1.5:*, green 1:*, blue 0.5:*, black 0:*")
    fig.savefig(PLOTS / "dr5_angle_slop_bias_vs_hours.png", dpi=130); plt.close(fig)

    sel = [n for n in ("0:0.1", "0.5:0.1", "1:0.1", "1.5:0.1", "1.5:0.003") if n in rows]
    cols = {"0:0.1": "k", "0.5:0.1": "tab:blue", "1:0.1": "tab:green", "1.5:0.1": "tab:red", "1.5:0.003": "tab:orange"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharex=True)
    for j, key in enumerate(("xip9_by_bin", "xim9_by_bin")):
        for i in range(2):
            a = ax[i, j]
            for nm in sel:
                if i == 0:
                    for pt, ls in (("Bin6_Bin6/A", "-"), ("Bin6_Bin6/B", ":")):
                        d = rows[nm]["noise_like"].get(pt)
                        if d is not None:
                            a.plot(CEN, d[key], ls, marker="o", ms=3, color=cols[nm], label=f"{nm} [{pt[-1]}]")
                else:
                    d = rows[nm]["signal"].get("Bin6_Bin6/A") or rows[nm]["signal"].get("Bin6_Bin6/S")
                    if d is not None:
                        a.plot(CEN, d[key], "-", marker="o", ms=3, color=cols[nm], label=nm)
            a.axhline(0, color="grey", lw=0.8); a.set_xscale("log"); a.grid(alpha=0.3)
            a.set_title(("xi+" if j == 0 else "xi-") + " 9-bin, " + ("noise-like on real patch (solid A, dotted B)" if i == 0 else "coherent signal bias, noiseless field"), fontsize=10)
            a.set_ylabel("bias / official sigma")
            if i == 1:
                a.set_xlabel("theta [arcmin]")
            a.legend(fontsize=7, ncol=2)
    fig.savefig(PLOTS / "dr5_angle_slop_bias_per_bin.png", dpi=130); plt.close(fig)


def fmt(x, p=3):
    return "n/a" if x is None or x != x else f"{x:.{p}f}"


def main():
    g, rows = collect()
    keyrows = [n for n in KEY if n in rows and rows[n]["noise_like"]]
    study = {"definitions": __doc__, "production_hours_at_0_def": FIN, "settings": rows,
             "brute_check": g.get("brute_check")}
    (OUT / "study.json").write_text(json.dumps(study, indent=1, default=float))
    make_plots(rows)
    fp = json.loads((OUT / "fullpair_vs_exact.json").read_text()) if (OUT / "fullpair_vs_exact.json").exists() else {}
    study["fullpair_vs_exact"] = fp
    (OUT / "study.json").write_text(json.dumps(study, indent=1, default=float))
    L = ["| setting | hours | xi+ 9-bin | xi- 9-bin | E_n (6) | E_n (20) | signal-only xi- | stable A/B | MEASURED full pair (xi+ / xi- / E6 / E20) | verdict |",
         "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for n in keyrows:
        r = rows[n]; w = r["worst"]
        st = {True: "yes", False: "NO", None: "n/a"}[r["stable_AB_xim9"]]
        star = "*" if r["hours_basis"].startswith("small") else ""
        f = fp.get(n)
        fs = "not measured" if not f else f"{f['max_xip9']:.3f} / {f['max_xim9']:.3f} / {f['max_En6']:.3f} / {f['max_En20']:.3f}"
        L.append(f"| {n} | {r['hours']:.1f}{star} | {fmt(w['xip9'])} | {fmt(w['xim9'])} | {fmt(w['En6'])} | {fmt(w['En20'])} | {fmt(w['signal_xim9'])} | {st} | {fs} | {verdict(n, r, f)} |")
    (OUT / "ANGLE_SLOP_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
