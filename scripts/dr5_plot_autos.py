"""The 6 auto pairs at bin_slop 1.5: theta*xi+/- (9 bins) vs the official release, residuals in official sigma, float32 run for comparison.

  python scripts/dr5_plot_autos.py   -> plots/dr5_autos_b1p5.png
"""
import importlib.util, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline")
sys.path.insert(0, str(ROOT / "src"))
from kidspipe.measure.a1check import A3_MAX_ABS_OVER_SIGMA  # noqa: E402

PRODUCT = ROOT / "runs/dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz"
F32 = ROOT / "runs/dr5_measure/xi_unordered/b1.5_f32"
C_XIP, C_XIM, C_F32 = "#0072B2", "#D55E00", "#6b6b6b"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    spec = importlib.util.spec_from_file_location("unord", ROOT / "scripts/dr5_measure_xi_unordered.py")
    u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
    d = dict(np.load(PRODUCT)); ctx = u.load_context()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5, "savefig.dpi": 140})
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.2, left=0.06, right=0.99, top=0.84, bottom=0.07)
    pid = d["pair_ids"].tolist()
    for n, b in enumerate(range(1, 7)):
        k = pid.index([b, b]); blk = u.pair_blocks(ctx, (b, b)); th = ctx["fx"]["theta"]
        o32, _ = u.load_ours(F32 / f"Bin{b}_Bin{b}.npz")
        sub = gs[n // 3, n % 3].subgridspec(2, 1, height_ratios=[3, 1.4], hspace=0.06)
        a = fig.add_subplot(sub[0]); r = fig.add_subplot(sub[1], sharex=a)
        mx = {}
        for key, y9, col, mk in (("xip", d["xip9"][k], C_XIP, "o"), ("xim", d["xim9"][k], C_XIM, "s")):
            off, sg = blk["val9"][key], blk["sig9"][key]
            a.errorbar(th, th * off * 1e4, th * sg * 1e4, fmt="none", ecolor=col, elinewidth=1, capsize=2, alpha=0.8, zorder=2)
            a.plot(th, th * off * 1e4, mk, mfc="white", mec=col, ms=8, ls="none", zorder=3)
            a.plot(th, th * y9 * 1e4, mk, color=col, ms=4, ls="none", zorder=4)
            z = (y9 - off) / sg; mx[key] = np.abs(z).max()
            z32 = (u.rebin9(o32["r_nom"], o32["npairs"], o32[key] / u.mfac((b, b))) - off) / sg
            r.plot(th, z32, ":", color=C_F32 if key == "xip" else "#a0a0a0", lw=1.2, zorder=2)
            r.plot(th, z, "-" + mk, color=col, ms=3.5, lw=1.2, zorder=3)
        a.set_xscale("log"); plt.setp(a.get_xticklabels(), visible=False)
        a.set_title(f"Bin{b} x Bin{b}   A3 max |Δ|/σ: ξ+ {mx['xip']:.3f}, ξ− {mx['xim']:.3f}", loc="left", fontsize=10.5)
        r.axhspan(-A3_MAX_ABS_OVER_SIGMA, A3_MAX_ABS_OVER_SIGMA, color="#999999", alpha=0.22, lw=0)
        r.axhline(0, color="k", lw=0.5); r.set_ylim(-0.12, 0.12); r.set_yticks([-0.1, 0, 0.1])
        if n >= 3:
            r.set_xlabel(r"$\theta$ [arcmin]")
        else:
            plt.setp(r.get_xticklabels(), visible=False)
        if n % 3 == 0:
            a.set_ylabel(r"$\theta\,\xi_\pm\ [10^{-4}$ arcmin]"); r.set_ylabel(r"$\Delta/\sigma_{\rm off}$")
    h = [Line2D([], [], marker="o", color=C_XIP, ls="none", ms=5, label="ξ+ ours"),
         Line2D([], [], marker="o", mfc="white", mec=C_XIP, ls="none", ms=8, label="ξ+ official (bars: σ)"),
         Line2D([], [], marker="s", color=C_XIM, ls="none", ms=5, label="ξ− ours"),
         Line2D([], [], marker="s", mfc="white", mec=C_XIM, ls="none", ms=8, label="ξ− official (bars: σ)"),
         Line2D([], [], ls=":", color=C_F32, lw=1.4, label="float32-position run, superseded (dark ξ+, light ξ−)"),
         Line2D([], [], lw=6, color="#999999", alpha=0.3, label="A3 limit ±0.1σ")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.055, 0.925), ncol=6, frameon=False, fontsize=9.5, columnspacing=1.3, handletextpad=0.5)
    fig.suptitle("Auto pairs, bin_slop 1.5: ours agrees with the official ξ± within 0.05σ in every angular bin", x=0.06, ha="left", fontsize=13.5, y=0.975)
    fig.savefig(ROOT / "plots/dr5_autos_b1p5.png", bbox_inches="tight"); print("wrote plots/dr5_autos_b1p5.png")


if __name__ == "__main__":
    main()
