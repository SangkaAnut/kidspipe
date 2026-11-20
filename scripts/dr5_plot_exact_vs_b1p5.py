"""Figure: exact run minus bin_slop 1.5 product, per pair, in units of the release's sigma.

Reads runs/dr5_measure/product_exact/exact_vs_b1p5.json (written by scripts/dr5_make_exact_product.py); writes
plots/dr5_exact_vs_b1p5.png.  Diverging blue-grey-red, symmetric limits; one shared scale for the four COSEBIs panels so the
with/without-edge-bins panels compare directly.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.dont_write_bytecode = True
ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline")
REP = ROOT / "runs/dr5_measure/product_exact/exact_vs_b1p5.json"
OUTF = ROOT / "plots/dr5_exact_vs_b1p5.png"
CMAP = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#f0efec", "#e34948"])
INK, MUTED = "#1f1f1e", "#6b6a66"


def main():
    r = json.loads(REP.read_text())
    pairs = list(r["per_pair"]); a = {k: np.array(v) for k, v in r["arrays"].items()}
    D, e = r.get("D"), r["edge_bins"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 9.5, "savefig.dpi": 130})
    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.32, wspace=0.18)
    vx = 0.15; vc = 0.62
    panels = [((0, slice(0, 2)), "dxip9_over_sigma", r"$\xi_+$, 9 official bins", vx, "bin"),
              ((0, slice(2, 4)), "dxim9_over_sigma", r"$\xi_-$, 9 official bins", vx, "bin"),
              ((1, 0), "dEn_over_sigma", r"$E_n$", vc, "mode n"),
              ((1, 1), "dEn_edgezeroed_over_sigma", rf"$E_n$, edge difference zeroed", vc, "mode n"),
              ((1, 2), "dBn_over_sigma", r"$B_n$", vc, "mode n"),
              ((1, 3), "dBn_edgezeroed_over_sigma", rf"$B_n$, edge difference zeroed", vc, "mode n")]
    note = (f"'edge difference zeroed': the exact-minus-bin_slop-1.5 xi difference is set to zero in the outermost {e} fine bins "
            "(292.6-300 arcmin) before the COSEBIs are computed; it shows how much of the difference comes from those bins.")
    ims = {}
    for (row, col), key, title, vm, xl in panels:
        ax = fig.add_subplot(gs[row, col]); z = a[key]
        im = ax.imshow(z, aspect="auto", cmap=CMAP, vmin=-vm, vmax=vm, interpolation="nearest")
        ims.setdefault(row, (im, []))[1].append(ax)
        ax.set_title(f"{title}\nmax |difference| {np.abs(z).max():.2f}", color=INK, loc="left")
        n = z.shape[1]; ticks = np.arange(n) if n <= 9 else np.arange(0, n, 4)
        ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1); ax.set_xlabel(xl)
        ax.set_yticks(np.arange(len(pairs)))
        ax.set_yticklabels(pairs if (col == 0 or col == slice(0, 2)) else [], fontsize=7.5)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    for row, lab in ((0, "(exact - bin_slop 1.5) / sigma_release"), (1, "(exact - bin_slop 1.5) / sigma_release, shared scale")):
        im, axs = ims[row]
        cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01); cb.set_label(lab, color=INK); cb.outline.set_linewidth(0.5)
    t = ("DR5 exact TreeCorr run (bin_slop 0, angle_slop 0) minus the bin_slop 1.5 product, all 21 pairs, m-corrected; "
         "not gated (PI 2026-11-02)")
    sub = (f"Joint xi chi2 of the difference {r['fullvector']['xi_ex_minus_b15']['chi2_joint_378']:.2f} / 378;  "
           f"E_n 1-6 {r['fullvector']['cosebis_ex_minus_b15']['E_6m_126']:.2f} / 126.")
    if D:
        sub += (f"  Model chi2 of E_n 1-6 at the release best fit: official {D['chi2_official_data']:.2f}, "
                f"bin_slop 1.5 {D['chi2_b15']:.2f}, exact {D['chi2_exact']:.2f}.")
    fig.suptitle(t, x=0.06, ha="left", y=0.985, fontsize=11, color=INK)
    fig.text(0.06, 0.945, sub, ha="left", fontsize=9, color=MUTED)
    fig.text(0.06, 0.035, note, ha="left", fontsize=8.5, color=MUTED)
    OUTF.parent.mkdir(exist_ok=True)
    fig.savefig(OUTF, bbox_inches="tight"); print("wrote", OUTF)


if __name__ == "__main__":
    main()
