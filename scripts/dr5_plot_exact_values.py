"""xi+/- and COSEBIs VALUES of the exact run, beside the bin_slop 1.5 product and the official release (same style as
scripts/dr5_make_plots.py plots 1 and 3).

  python scripts/dr5_plot_exact_values.py

Inputs (read only; <runs>/... first, else the copy shipped in <root>/runs/...): dr5_measure/product_exact/dr5_xipm_cosebis_exact.npz,
dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz, the official FITS (kidspipe.measure.official).  Writes dr5_08_xipm_exact_vs_official.png,
dr5_09_En_exact.png, dr5_09b_Bn_exact.png into $KIDSPIPE_PLOTS (default <root>/plots).  The exact run is not gated (PI 2026-11-02): no gate band is drawn.
"""
import importlib.util, sys, textwrap
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
_spec = importlib.util.spec_from_file_location("dr5plots", ROOT / "scripts/dr5_make_plots.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)
of = P.of
EXACT = paths.find_result("dr5_measure/product_exact/dr5_xipm_cosebis_exact.npz")
C_B15 = "#6b6b6b"


def plot_xi(ex, b15, fx):
    import matplotlib.pyplot as plt
    sig = np.sqrt(np.diag(fx["cov"])); n = P.NP * 9; th = fx["theta"]
    fig = plt.figure(figsize=(21, 17))
    gs = fig.add_gridspec(6, 6, hspace=0.35, wspace=0.28, left=0.045, right=0.99, top=0.93, bottom=0.05)
    mx = {"xi+": 0.0, "xi-": 0.0}
    for k, (i, j) in enumerate(P.PAIRS1):
        sub = gs[j - 1, i - 1].subgridspec(2, 1, height_ratios=[3, 1.3], hspace=0.06)
        a = fig.add_subplot(sub[0]); b = fig.add_subplot(sub[1], sharex=a)
        for lab, e9, b9, off, s0, col, mk in (("xi+", ex["xip9"][k], b15["xip9"][k], fx["xiP"][k], k * 9, P.C_OURS, "o"),
                                              ("xi-", ex["xim9"][k], b15["xim9"][k], fx["xiM"][k], n + k * 9, P.C_OFF, "s")):
            sg = sig[s0:s0 + 9]
            a.errorbar(th, th * off * 1e4, th * sg * 1e4, fmt="none", ecolor=col, elinewidth=1, capsize=2, alpha=0.8, zorder=2)
            a.plot(th, th * off * 1e4, mk, mfc="white", mec=col, ms=6, ls="none", zorder=3)
            a.plot(th, th * e9 * 1e4, mk, color=col, ms=3.2, ls="none", zorder=4)
            b.plot(th, (b9 - off) / sg, "--", color=col, lw=0.9, alpha=0.8)
            b.plot(th, (e9 - off) / sg, "-" + mk, color=col, ms=3, lw=1.1)
            mx[lab] = max(mx[lab], float(np.abs((e9 - off) / sg).max()))
        a.set_xscale("log"); a.set_title(P.pname((i, j)), fontsize=9, loc="left", pad=2)
        plt.setp(a.get_xticklabels(), visible=False)
        b.axhline(0, color="k", lw=0.5); b.set_ylim(-0.2, 0.2); b.set_yticks([-0.15, 0, 0.15]); b.set_xscale("log")
        if j == 6:
            b.set_xlabel(r"$\theta$ [arcmin]")
        else:
            plt.setp(b.get_xticklabels(), visible=False)
        if i == 1:
            a.set_ylabel(r"$\theta\,\xi_\pm\ [10^{-4}$ arcmin]"); b.set_ylabel(r"$\Delta/\sigma$")
        b.tick_params(labelsize=7)
    ax = fig.add_axes([0.52, 0.66, 0.46, 0.22]); ax.axis("off")
    ax.text(0, 1, "DR5 xi+/- (9 bins): EXACT TreeCorr run (bin_slop 0, angle_slop 0, float64, m-corrected) vs official KiDS-Legacy FITS\n\n"
            "Top: theta*xi (filled = exact; open with bars = official, bars = sqrt(diag COVMAT)); blue circles xi+, vermilion squares xi-.\n"
            "The bin_slop 1.5 product is not drawn on top: at this scale it lies on the official points (plot 1).\n"
            "Bottom: difference from official in official sigma; solid with markers = exact, dashed = bin_slop 1.5 product.\n\n"
            f"Max |exact - official|/sigma: xi+ {mx['xi+']:.3f}, xi- {mx['xi-']:.3f}. The release used bin_slop 1.5, so the exact run\n"
            "differs from it by the size of that approximation. Not gated (PI decision 2026-11-02); see EXACT_VS_B1P5.md.",
            va="top", fontsize=11, transform=ax.transAxes)
    fig.suptitle("Plot 8: xi+/- (9 bins), exact run vs official, 21 tomographic pairs", fontsize=13, x=0.045, ha="left")
    return fig


def plot_modes(ex, b15, off, cov, offset, title, ylab):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(6, 6, figsize=(21, 15), sharex=True)
    m = np.arange(1, P.NM + 1)
    for ax in axs.ravel():
        ax.axis("off")
    for k, (i, j) in enumerate(P.PAIRS1):
        ax = axs[j - 1, i - 1]; ax.axis("on")
        sg = np.sqrt(np.diag(cov)[k * P.NM + np.arange(P.NM) + offset])
        ax.axvspan(0.5, 6.5, color="#cccccc", alpha=0.35, lw=0)
        ax.errorbar(m - 0.2, off[k] * 1e10, sg * 1e10, fmt="s", mfc="white", mec=P.C_OFF, ecolor=P.C_OFF, ms=4, elinewidth=1, capsize=1.5)
        ax.plot(m, b15[k] * 1e10, "o", color=C_B15, ms=2.6, ls="none")
        ax.plot(m + 0.2, ex[k] * 1e10, "o", color=P.C_OURS, ms=3.4, ls="none")
        ax.axhline(0, color="k", lw=0.5); ax.set_title(P.pname((i, j)), loc="left", fontsize=9, pad=2)
        ax.set_xticks([1, 6, 10, 15, 20])
        if j == 6:
            ax.set_xlabel("COSEBI mode n")
        if i == 1:
            ax.set_ylabel(ylab)
        ax.tick_params(labelbottom=True)
    ax = axs[0, 2]; ax.axis("off")
    ax.text(0, 1, textwrap.fill(title, 62), va="top", fontsize=11, transform=ax.transAxes)
    ax.legend(handles=[Line2D([], [], marker="o", color=P.C_OURS, ls="none", label="exact (bin_slop 0)"),
                       Line2D([], [], marker="o", color=C_B15, ms=4, ls="none", label="bin_slop 1.5 product"),
                       Line2D([], [], marker="s", mfc="white", mec=P.C_OFF, color=P.C_OFF, ls="none", label="official FITS (bars: official sigma)")],
              loc="lower left", fontsize=11, bbox_to_anchor=(0, 0.0))
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")
    P.style()
    ex, b15 = dict(np.load(EXACT)), dict(np.load(P.PRODUCT))
    fx, cb = of.load_official_xipm_fits(), of.load_official_cosebis()
    assert np.array_equal(ex["pair_ids"], np.array(P.PAIRS1)) and np.array_equal(b15["pair_ids"], np.array(P.PAIRS1))

    def save(fig, name):
        fig.savefig(P.PLOTS / name, bbox_inches="tight"); print("wrote", P.PLOTS / name, flush=True)
    save(plot_xi(ex, b15, fx), "dr5_08_xipm_exact_vs_official.png")
    save(plot_modes(ex["En"], b15["En"], cb["En"], cb["cov"], 0,
                    "Plot 9: COSEBIs E_n, modes 1-20: exact run (blue), bin_slop 1.5 product (grey, mostly hidden behind the official squares), official (open squares, bars = official "
                    "sqrt(diag COVMAT) E block). Grey band = analysis range n<=6. Not gated.", r"$E_n\ [10^{-10}]$"), "dr5_09_En_exact.png")
    save(plot_modes(ex["Bn"], b15["Bn"], cb["Bn"], cb["cov"], 420,
                    "Plot 9b: COSEBIs B_n, modes 1-20: exact (blue), bin_slop 1.5 (grey, mostly hidden behind the official squares), official (open squares, bars = official "
                    "sigma, B block). B_n should be consistent with zero. Not gated.", r"$B_n\ [10^{-10}]$"), "dr5_09b_Bn_exact.png")


if __name__ == "__main__":
    main()
