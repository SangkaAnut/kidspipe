"""Compact figures for docs/PIPELINE_ARCHITECTURE_COMPARISON.tex: xi+/-, COSEBIs and a 21-pair summary,
ours (exact, bin_slop 1.5) against the official release.  Writes docs/figs/*.pdf (or $KIDSPIPE_PLOTS/figs if set).
Inputs: the exact and bin_slop 1.5 products (<runs>/dr5_measure/..., else the shipped copy).   python scripts/dr5_plot_report_figs.py"""
import importlib.util, os, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
spec = importlib.util.spec_from_file_location("dr5plots", ROOT / "scripts/dr5_make_plots.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
of = P.of
OUT = Path(os.environ["KIDSPIPE_PLOTS"]).expanduser() / "figs" if os.environ.get("KIDSPIPE_PLOTS") else ROOT / "docs/figs"
OUT.mkdir(parents=True, exist_ok=True)
EXACT = paths.find_result("dr5_measure/product_exact/dr5_xipm_cosebis_exact.npz")
C_EX, C_B, C_OFF = "#0072B2", "#6b6b6b", "#D55E00"
SHOW = [(3, 3), (4, 5), (6, 6)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P.style()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "savefig.dpi": 200})
    ex, b = dict(np.load(EXACT)), dict(np.load(P.PRODUCT))
    fx, cb = of.load_official_xipm_fits(), of.load_official_cosebis()
    n = P.NP * 9; sig = np.sqrt(np.diag(fx["cov"])); th = fx["theta"]
    sE = np.sqrt(np.diag(cb["cov"]))[:420].reshape(P.NP, P.NM); sB = np.sqrt(np.diag(cb["cov"]))[420:].reshape(P.NP, P.NM)
    m = np.arange(1, P.NM + 1)
    kk = {p: P.PAIRS1.index(p) for p in SHOW}

    # ---- Fig 1: xi+/-
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.1, "wspace": 0.38})
    for c, p in enumerate(SHOW):
        k = kk[p]
        for lab, e9, b9, off, s0, col, mk in (("+", ex["xip9"][k], b["xip9"][k], fx["xiP"][k], k * 9, C_EX, "o"),
                                              ("-", ex["xim9"][k], b["xim9"][k], fx["xiM"][k], n + k * 9, "#CC79A7", "s")):
            sg = sig[s0:s0 + 9]
            axs[0, c].errorbar(th, th * off * 1e4, th * sg * 1e4, fmt=mk, mfc="white", mec=C_OFF if lab == "+" else "#8c3b00",
                               ecolor=C_OFF if lab == "+" else "#8c3b00", ms=6, elinewidth=1.0, capsize=1.5, zorder=2)
            axs[0, c].plot(th, th * e9 * 1e4, mk, color=col, ms=3.6, ls="none", zorder=4)
            axs[1, c].plot(th, (b9 - off) / sg, "--", color=C_B, lw=0.9)
            axs[1, c].plot(th, (e9 - off) / sg, "-" + mk, color=col, ms=2.5, lw=0.9)
        axs[0, c].set_title(f"Bin{p[0]}-{p[1]}", loc="left"); axs[0, c].set_xscale("log")
        axs[1, c].axhline(0, color="k", lw=0.4); axs[1, c].set_ylim(-0.2, 0.2); axs[1, c].set_xlabel(r"$\theta$ [arcmin]")
    axs[0, 0].set_ylabel(r"$\theta\,\xi_\pm\ [10^{-4}$ arcmin]"); axs[1, 0].set_ylabel(r"(ours $-$ official)$/\sigma$")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=C_OFF, color=C_OFF, ls="none", label="official $\\xi_+$ (open, bars)"),
                              Line2D([], [], marker="s", mfc="white", mec="#8c3b00", color="#8c3b00", ls="none", label="official $\\xi_-$"),
                              Line2D([], [], marker="o", color=C_EX, ls="none", label="exact $\\xi_+$"),
                              Line2D([], [], marker="s", color="#CC79A7", ls="none", label="exact $\\xi_-$"),
                              Line2D([], [], color=C_B, ls="--", label="bin_slop 1.5 residual")], fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.savefig(OUT / "fig_xipm.pdf", bbox_inches="tight"); plt.close(fig)

    # ---- Fig 2: COSEBIs
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.1, "wspace": 0.38})
    for c, p in enumerate(SHOW):
        k = kk[p]
        axs[0, c].axvspan(0.5, 6.5, color="#cccccc", alpha=0.35, lw=0)
        axs[0, c].errorbar(m - 0.2, cb["En"][k] * 1e10, sE[k] * 1e10, fmt="s", mfc="white", mec=C_OFF, ecolor=C_OFF, ms=4, elinewidth=0.9, capsize=1.2)
        axs[0, c].plot(m, b["En"][k] * 1e10, "o", color=C_B, ms=2.2, ls="none")
        axs[0, c].plot(m + 0.2, ex["En"][k] * 1e10, "o", color=C_EX, ms=2.8, ls="none")
        axs[0, c].axhline(0, color="k", lw=0.4); axs[0, c].set_title(f"Bin{p[0]}-{p[1]}", loc="left")
        axs[1, c].axvspan(0.5, 6.5, color="#cccccc", alpha=0.35, lw=0)
        axs[1, c].plot(m, (b["En"][k] - cb["En"][k]) / sE[k], "--", color=C_B, lw=0.9)
        axs[1, c].plot(m, (ex["En"][k] - cb["En"][k]) / sE[k], "-o", color=C_EX, ms=2.2, lw=0.9)
        axs[1, c].axhline(0, color="k", lw=0.4); axs[1, c].set_ylim(-0.7, 0.7); axs[1, c].set_xlabel("COSEBI mode $n$")
    axs[0, 0].set_ylabel(r"$E_n\ [10^{-10}]$"); axs[1, 0].set_ylabel(r"(ours $-$ official)$/\sigma$")
    fig.legend(handles=[Line2D([], [], marker="s", mfc="white", mec=C_OFF, color=C_OFF, ls="none", label="official (bars: $\\sigma$)"),
                              Line2D([], [], marker="o", color=C_B, ls="none", ms=3, label="bin_slop 1.5"),
                              Line2D([], [], marker="o", color=C_EX, ls="none", label="exact"), Line2D([], [], color=C_B, ls="--", label="bin_slop 1.5 residual")], fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.savefig(OUT / "fig_cosebis.pdf", bbox_inches="tight"); plt.close(fig)

    # ---- Fig 3: 21-pair summary, max |ours - official| / sigma
    xi_b, xi_e, E_b, E_e = [], [], [], []
    for k in range(P.NP):
        s = []
        for pr in (b, ex):
            s.append(max(np.abs((pr["xip9"][k] - fx["xiP"][k]) / sig[k * 9:(k + 1) * 9]).max(),
                         np.abs((pr["xim9"][k] - fx["xiM"][k]) / sig[n + k * 9:n + (k + 1) * 9]).max()))
        xi_b.append(s[0]); xi_e.append(s[1])
        E_b.append(np.abs((b["En"][k] - cb["En"][k]) / sE[k]).max()); E_e.append(np.abs((ex["En"][k] - cb["En"][k]) / sE[k]).max())
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, gridspec_kw={"wspace": 0.2})
    x = np.arange(P.NP); lab = [f"{i}{j}" for i, j in P.PAIRS1]
    for ax, yb, ye, t in ((axs[0], xi_b, xi_e, r"$\xi_\pm$, 9 bins: max over $\xi_+,\xi_-$ and bins"),
                          (axs[1], E_b, E_e, r"$E_n$, modes 1--20: max over modes")):
        ax.bar(x - 0.2, yb, 0.4, color=C_B, label="bin_slop 1.5"); ax.bar(x + 0.2, ye, 0.4, color=C_EX, label="exact")
        ax.set_title(t, loc="left"); ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90, fontsize=8); ax.set_xlabel("tomographic pair")
    axs[0].set_ylabel(r"max |ours $-$ official| $/\ \sigma_{\rm official}$"); axs[0].legend(fontsize=8.5, loc="upper right")
    fig.savefig(OUT / "fig_summary.pdf", bbox_inches="tight"); plt.close(fig)
    print("xi b15 range %.3f-%.3f, exact %.3f-%.3f; En b15 %.3f-%.3f, exact %.3f-%.3f" % (min(xi_b), max(xi_b), min(xi_e), max(xi_e), min(E_b), max(E_b), min(E_e), max(E_e)))


if __name__ == "__main__":
    main()
