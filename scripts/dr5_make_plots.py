"""Six DR5 measurement plots (PI option (a): TreeCorr 5.1.4, bin_slop 1.5, unordered, float64 positions) in plots/.

  python scripts/dr5_make_plots.py a4      # fresh small-subsample estimator cross-check -> <runs>/dr5_measure/product/a4_recheck.json
  python scripts/dr5_make_plots.py plots   # all six figures (needs a4_recheck.json for plot 6)

Paths (kidspipe.paths): <runs> = $KIDSPIPE_RUNS (default <root>/runs); figures go to $KIDSPIPE_PLOTS (default <root>/plots).
Inputs (read only; each looked up as <runs>/..., else the copy shipped in <root>/runs/...): dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz,
fullvector_checks.json, a4_recheck.json (plot 6 is skipped without it), the official FITS/ascii (via kidspipe.measure.official),
dr5_measure/b1_integration.json (B1 scalars; arrays are recomputed from the official ascii: it stores none),
and the earlier pipeline's xi (theirs, $KIDSPIPE_THEIRS, optional: the two ours-vs-theirs panels of plot 2 are omitted when it is absent);
the float64 extract $KIDSPIPE_DR5_WORK/bin{0,5}.npz (a4 only).  The a4 TreeCorr runs use min(16, os.cpu_count()) threads.
Plots: dr5_01_xipm_vs_official, dr5_02_npairs_weights_ratio, dr5_03_En_vs_official, dr5_03b_Bn_vs_official, dr5_04_integration_vs_measurement,
dr5_05_Bn_null, dr5_06_estimator_crosscheck.
"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2dist

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
from kidspipe.measure import official as of  # noqa: E402
from kidspipe.measure.a1check import A3_MAX_ABS_OVER_SIGMA  # noqa: E402
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402

PRODUCT = paths.find_result("dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz")     # input
FULLV = paths.find_result("dr5_measure/product/fullvector_checks.json")                 # input
A4JSON = paths.find_result("dr5_measure/product/a4_recheck.json")                       # input of `plots`
A4OUT = paths.runs_dir() / "dr5_measure" / "product" / "a4_recheck.json"                # output of `a4`
PLOTS = Path(os.environ["KIDSPIPE_PLOTS"]).expanduser() if os.environ.get("KIDSPIPE_PLOTS") else ROOT / "plots"
PAIRS1 = [(i + 1, j + 1) for i, j in of.PAIRS]
NM, NP = of.N_MODES, of.N_PAIR

# colours: Okabe-Ito (colour-blind safe); ours = blue, official = vermilion, third = grey/green
C_OURS, C_OFF, C_THEIRS, C_GREY, C_GREEN = "#0072B2", "#D55E00", "#009E73", "#6b6b6b", "#009E73"
BIN_COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#7f7f7f"]   # lines coloured by first bin index


def _u():
    spec = importlib.util.spec_from_file_location("unord", ROOT / "scripts/dr5_measure_xi_unordered.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 130})


def pname(p):
    return f"Bin{p[0]}-{p[1]}"


# ------------------------------------------------------------------------------------------------ plot 1
def plot1(d, fx):
    import matplotlib.pyplot as plt
    cov = fx["cov"]; sig = np.sqrt(np.diag(cov)); n = NP * 9
    th = fx["theta"]
    fig = plt.figure(figsize=(21, 17))
    gs = fig.add_gridspec(6, 6, hspace=0.35, wspace=0.28, left=0.045, right=0.99, top=0.93, bottom=0.05)
    for k, (i, j) in enumerate(PAIRS1):
        sub = gs[j - 1, i - 1].subgridspec(2, 1, height_ratios=[3, 1.3], hspace=0.06)
        a = fig.add_subplot(sub[0]); b = fig.add_subplot(sub[1], sharex=a)
        for lab, ours, off, s0, col, mk in (("xi+", d["xip9"][k], fx["xiP"][k], k * 9, C_OURS, "o"),
                                           ("xi-", d["xim9"][k], fx["xiM"][k], n + k * 9, C_OFF, "s")):
            sg = sig[s0:s0 + 9]
            a.errorbar(th, th * off * 1e4, th * sg * 1e4, fmt="none", ecolor=col, elinewidth=1, capsize=2, alpha=0.8, zorder=2)
            a.plot(th, th * off * 1e4, mk, mfc="white", mec=col, ms=6, ls="none", zorder=3)
            a.plot(th, th * ours * 1e4, mk, color=col, ms=3.2, ls="none", zorder=4)
            b.plot(th, (ours - off) / sg, "-" + mk, color=col, ms=3, lw=1)
        a.set_xscale("log"); a.set_title(pname((i, j)), fontsize=9, loc="left", pad=2)
        plt.setp(a.get_xticklabels(), visible=False)
        b.axhspan(-A3_MAX_ABS_OVER_SIGMA, A3_MAX_ABS_OVER_SIGMA, color="#999999", alpha=0.25, lw=0)
        b.axhline(0, color="k", lw=0.5); b.set_ylim(-0.13, 0.13); b.set_yticks([-0.1, 0, 0.1]); b.set_xscale("log")
        if j == 6:
            b.set_xlabel(r"$\theta$ [arcmin]")
        else:
            plt.setp(b.get_xticklabels(), visible=False)
        if i == 1:
            a.set_ylabel(r"$\theta\,\xi_\pm\ [10^{-4}$ arcmin]"); b.set_ylabel(r"$\Delta/\sigma$")
        b.tick_params(labelsize=7)
    ax = fig.add_axes([0.52, 0.66, 0.46, 0.22]); ax.axis("off")
    j378 = json.loads(FULLV.read_text())["A3"]
    ax.text(0, 1, "DR5 xi+/- (9 bins): our TreeCorr 5.1.4 run (bin_slop 1.5, unordered, float64, m-corrected) vs official KiDS-Legacy FITS\n\n"
            "Top: theta*xi (filled = ours; open with bars = official, bars = sqrt(diag COVMAT)); blue circles xi+, vermilion squares xi-.\n"
            "Bottom: (ours - official)/sigma_official; grey band = A3 gate |Delta|/sigma < 0.1.\n\n"
            f"A3 max |Delta|/sigma: xi+ {j378['xip']['max_abs_over_sigma']:.3f}, xi- {j378['xim']['max_abs_over_sigma']:.3f} (both below the gate; thresholds were set after seeing results).\n"
            "Release used TreeCorr 4.2.3; this is agreement with the release, not proof of independent validation.\n"
            "Only the A3 comparison is amplitude-sensitive (~1 %).", va="top", fontsize=11, wrap=True, transform=ax.transAxes)
    fig.suptitle("Plot 1: xi+/- (9 bins) ours vs official, 21 tomographic pairs", fontsize=13, x=0.045, ha="left")
    return fig


# ------------------------------------------------------------------------------------------------ plot 2
def plot2(d, asc, u):
    import matplotlib.pyplot as plt
    th = d["r_nom"][0]
    ours_th = {k: u.load_theirs(p) for k, p in enumerate(PAIRS1)}
    have_theirs = all(t is not None for t in ours_th.values())
    if not have_theirs:
        print(f"NOTE plot 2: the earlier pipeline's xi ({u.THEIRS}) is not available; the two ours-vs-theirs panels are omitted "
              f"(set KIDSPIPE_THEIRS to include them)", flush=True)
    nrow = 2 if have_theirs else 1
    fig, axs = plt.subplots(nrow, 2, figsize=(14, 9.5 if have_theirs else 5.5), sharex=True, squeeze=False)
    panels = [("weight (ours) / weight (official ascii)", lambda k: d["weight"][k] / asc["weight"][k]),
              ("npairs (ours) / npairs (official ascii)", lambda k: d["npairs"][k] / asc["npairs"][k])]
    if have_theirs:
        panels += [("npairs ours / theirs (earlier pipeline)", lambda k: d["npairs"][k] / ours_th[k]["npairs"]),
                   ("weight ours / theirs (earlier pipeline)", lambda k: d["weight"][k] / ours_th[k]["weight"])]
    stats = []
    for ax, (ttl, fn) in zip(axs.ravel(), panels):
        allr = []
        for k, (i, j) in enumerate(PAIRS1):
            r = fn(k); allr.append(r)
            ax.plot(th, r, color=BIN_COL[i - 1], lw=1.8 if i == j else 0.9, ls="-" if i == j else "--", alpha=0.9)
        allr = np.array(allr); stats.append((ttl, np.nanmin(allr), np.nanmax(allr)))
        ax.set_xscale("log"); ax.set_title(ttl, loc="left"); ax.set_ylabel("ratio")
    for ax in axs[-1]:
        ax.set_xlabel(r"$\theta$ [arcmin] (1000 fine bins)")
    axs[0, 0].text(0.02, 0.5, "autos ~0.5: official weight counts ordered pairs, ours unordered;\ncrosses ~1.0", transform=axs[0, 0].transAxes, fontsize=8.5, va="center")
    axs[0, 1].text(0.02, 0.62, "official `npairs` column is a normalised column, not a raw count\n(ratio ~1.41 for crosses, ~0.71 for autos = 1.41/2; report section 3.5).\nUse npairs_weighted/weight for count comparisons.",
                   transform=axs[0, 1].transAxes, fontsize=8.5, va="center")
    if have_theirs:
        for ax in axs[1]:
            ax.set_ylim(0.99998, 1.00002); ax.ticklabel_format(axis="y", useOffset=False, style="plain"); ax.set_yticks([0.99998, 0.99999, 1.0, 1.00001, 1.00002])
        axs[1, 0].text(0.02, 0.9, f"min {stats[2][1]:.8f}  max {stats[2][2]:.8f}", transform=axs[1, 0].transAxes, fontsize=9)
        axs[1, 1].text(0.02, 0.9, f"min {stats[3][1]:.8f}  max {stats[3][2]:.8f}", transform=axs[1, 1].transAxes, fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BIN_COL[b], lw=2, label=f"first bin {b + 1}") for b in range(6)]
    h += [Line2D([], [], color="k", lw=1.8, label="auto (solid)"), Line2D([], [], color="k", lw=0.9, ls="--", label="cross (dashed)")]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=9)
    fig.suptitle("Plot 2: pair counts and weights, per pair, fine bins. Ours vs official ascii (top), ours vs theirs (bottom; exactly 1 expected)" if have_theirs
                 else "Plot 2: pair counts and weights, per pair, fine bins. Ours vs official ascii (the ours-vs-theirs panels need the earlier pipeline's xi: omitted)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig, stats


# ------------------------------------------------------------------------------------------------ plot 3 / 3b
def plot_modes(vals, off, cov, off_cov_offset, name, title, scale, ylab):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(6, 6, figsize=(21, 15), sharex=True)
    m = np.arange(1, NM + 1)
    for ax in axs.ravel():
        ax.axis("off")
    for k, (i, j) in enumerate(PAIRS1):
        ax = axs[j - 1, i - 1]; ax.axis("on")
        idx = k * NM + np.arange(NM) + off_cov_offset
        sg = np.sqrt(np.diag(cov)[idx])
        ax.axvspan(0.5, 6.5, color="#cccccc", alpha=0.35, lw=0)
        ax.errorbar(m - 0.12, off[k] * scale, sg * scale, fmt="s", mfc="white", mec=C_OFF, ecolor=C_OFF, ms=4, elinewidth=1, capsize=1.5, label="official")
        ax.errorbar(m + 0.12, vals[k] * scale, sg * scale, fmt="o", color=C_OURS, ms=3, elinewidth=0.7, capsize=1.5, alpha=0.9, label="ours")
        ax.axhline(0, color="k", lw=0.5); ax.set_title(pname((i, j)), loc="left", fontsize=9, pad=2)
        ax.set_xticks([1, 6, 10, 15, 20])
        if j == 6:
            ax.set_xlabel("COSEBI mode n")
        if i == 1:
            ax.set_ylabel(ylab)
        ax.tick_params(labelbottom=True)
    import textwrap
    from matplotlib.lines import Line2D
    ax = axs[0, 2]; ax.axis("off")
    ax.text(0, 1, textwrap.fill(title, 62), va="top", fontsize=11, transform=ax.transAxes)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_OURS, ls="none", label="ours (product, m-corrected)"),
                       Line2D([], [], marker="s", mfc="white", mec=C_OFF, color=C_OFF, ls="none", label="official FITS")],
              loc="lower left", fontsize=11, bbox_to_anchor=(0, 0.0))
    return fig


def plot3(d, cb, which):
    if which == "En":
        return plot_modes(d["En"], cb["En"], cb["cov"], 0, "En",
                          "Plot 3: COSEBIs E_n, modes 1-20, ours (blue dots) vs official (open squares); bars = official sqrt(diag COVMAT) E block. "
                          "Grey = analysis range n<=6. No numeric gate for E_n.", 1e10, r"$E_n\ [10^{-10}]$")
    return plot_modes(d["Bn"], cb["Bn"], cb["cov"], 420, "Bn",
                      "Plot 3b: COSEBIs B_n, modes 1-20, ours vs official; bars = official sqrt(diag COVMAT) B block (offset 420). "
                      "Grey = n<=6. B_n should be consistent with zero; ungated.", 1e10, r"$B_n\ [10^{-10}]$")


# ------------------------------------------------------------------------------------------------ plot 4
def b1_arrays(asc, cb, filt):
    """Recompute B1 (official ascii xi -> our E_n), trial meanr|edges|rad as in scripts/dr5_b1_integration.py; ascii is m-uncorrected."""
    E, B = e_from_xi_binned(asc["meanr"][0], asc["xip"], asc["xim"], filt, nmodes=NM, scheme="edges", unit="rad")
    return E, B


def plot4(d, cb, asc, filt):
    import matplotlib.pyplot as plt
    E1, B1 = b1_arrays(asc, cb, filt)
    sig = np.sqrt(np.diag(cb["cov"])[:NP * NM]).reshape(NP, NM)
    z1 = (E1 - cb["En"]) / sig; z2 = (d["En"] - cb["En"]) / sig
    b1json = json.loads(paths.find_result("dr5_measure/b1_integration.json").read_text())["trials"]["meanr|edges|rad"]
    chk = abs(np.abs(z1).max() - b1json["max_abs_dE_over_sigma"])
    assert chk < 1e-6, f"recomputed B1 max/sigma {np.abs(z1).max()} != stored {b1json['max_abs_dE_over_sigma']}"
    m = np.arange(1, NM + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1.3, 1.3, 1.0]})
    for ax, z, col, t in ((axs[0], z1, C_OFF, "B1: official ascii xi (m-uncorrected) -> our E_n"), (axs[1], z2, C_OURS, "B2: our measured xi (m-corrected) -> our E_n")):
        for k in range(NP):
            ax.plot(m + (k - 10) * 0.02, z[:, :][k], "o", color=col, ms=3, alpha=0.5)
        ax.plot(m, np.sqrt((z ** 2).mean(0)), "k-", lw=1.5, label="rms over 21 pairs")
        ax.axhline(0, color="k", lw=0.5); ax.axvspan(0.5, 6.5, color="#cccccc", alpha=0.3, lw=0)
        ax.set_ylim(-2, 2); ax.set_xlabel("COSEBI mode n"); ax.set_ylabel(r"$(E_n^{\rm ours}-E_n^{\rm official})/\sigma$"); ax.set_title(t, loc="left")
        ax.set_xticks([1, 5, 10, 15, 20]); ax.legend(loc="upper right")
        ax.text(0.02, 0.04, f"max |z| = {np.abs(z).max():.3f}   (21 x 20 points)", transform=ax.transAxes, fontsize=9)
    ax = axs[2]
    ax.plot(m, np.sqrt((z1 ** 2).mean(0)), "o-", color=C_OFF, label="B1 rms")
    ax.plot(m, np.sqrt((z2 ** 2).mean(0)), "o-", color=C_OURS, label="B2 rms")
    ax.set_yscale("log"); ax.set_xlabel("COSEBI mode n"); ax.set_ylabel("rms over pairs of residual / sigma"); ax.legend(); ax.set_title("rms per mode", loc="left")
    ax.set_xticks([1, 5, 10, 15, 20])
    fig.suptitle("Plot 4: integration test B1 vs measurement test B2, E_n residuals in official sigma (grey: n<=6). "
                 "B1 and B2 use different xi inputs (official ascii vs our product); neither is gated.", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, {"B1_max": float(np.abs(z1).max()), "B2_max": float(np.abs(z2).max()), "B1_rms_all": float(np.sqrt((z1 ** 2).mean())),
                 "B2_rms_all": float(np.sqrt((z2 ** 2).mean())), "B1_rms_n1to6": float(np.sqrt((z1[:, :6] ** 2).mean())),
                 "B2_rms_n1to6": float(np.sqrt((z2[:, :6] ** 2).mean())), "recompute_vs_stored_B1_max_diff": float(chk)}


# ------------------------------------------------------------------------------------------------ plot 5
def plot5(d, cb):
    import matplotlib.pyplot as plt
    fv = json.loads(FULLV.read_text())["B3"]
    sig = np.sqrt(np.diag(cb["cov"])[420:]).reshape(NP, NM)
    zo, zu = cb["Bn"] / sig, d["Bn"] / sig
    vm = 3.5
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1.0], hspace=0.45, wspace=0.28, left=0.05, right=0.98, top=0.88, bottom=0.09)
    for c, (z, t) in enumerate(((zu, "ours"), (zo, "official"))):
        ax = fig.add_subplot(gs[:, c])
        im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower", interpolation="nearest")
        ax.set_xticks(range(NP)); ax.set_xticklabels([pname(p) for p in PAIRS1], rotation=90, fontsize=7)
        ax.set_yticks(range(0, NM, 2)); ax.set_yticklabels(range(1, NM + 1, 2)); ax.set_ylabel("COSEBI mode n")
        ax.axhline(5.5, color="k", lw=0.8, ls=":"); ax.grid(False); ax.set_title(f"B_n / sigma, {t}", loc="left")
        if c == 1:
            cbr = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cbr.set_label(r"$B_n/\sigma$ (diag of official B block)")
    for r, nm in enumerate(("6m", "20m")):
        ax = fig.add_subplot(gs[r, 2]); b = fv[nm]; dof = b["dof"]
        x = np.linspace(chi2dist.ppf(0.0005, dof), chi2dist.ppf(0.9995, dof), 400)
        ax.plot(x, chi2dist.pdf(x, dof), color=C_GREY, lw=1.2, label=rf"$\chi^2_{{{dof}}}$ null")
        ax.axvline(b["chi2_ours"], color=C_OURS, lw=2, label=f"ours {b['chi2_ours']:.1f}, PTE {b['pte_ours']:.3f}")
        ax.axvline(b["chi2_official"], color=C_OFF, lw=2, ls="--", label=f"official {b['chi2_official']:.1f}, PTE {b['pte_official']:.3f}")
        ax.set_title(f"B_n null, modes 1-{nm[:-1]} ({dof} points); published PTE {b['published_pte']}", loc="left")
        ax.set_xlabel(r"$\chi^2$"); ax.set_yticks([]); ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Plot 5: B_n null test, B_n/sigma per pair and mode (left, dotted line = upper edge of n<=6) and chi2/PTE ours vs official (right). Ungated.",
                 x=0.01, ha="left", fontsize=11.5)
    return fig


# ------------------------------------------------------------------------------------------------ a4 recheck
A4_PAIRS = [(6, 6), (1, 6)]
A4_NBINS = 100
A4_RMAX = 60.0       # plotted/summarised range: the ~2 deg patch has few, edge-dominated pairs beyond ~half its diameter
A4_TARGET = {6: 15000, 1: 18000}      # galaxies in the patch (~0.25 % / 0.24 % of bins 6 / 1)


def run_a4():
    import treecorr
    from kidspipe.measure.paircount import PairCounter
    from kidspipe.measure.xi import corr_to_arrays, make_treecorr_cat, treecorr_kwargs
    t0 = time.time()
    work = paths.dr5_work_dir()
    for b in (6, 1):
        if not (work / f"bin{b - 1}.npz").is_file():
            sys.exit(f"a4 needs the per-bin extract {work / f'bin{b - 1}.npz'}: run scripts/dr5_extract.py first "
                     f"(or set KIDSPIPE_DR5_WORK)")
    nthreads = min(16, os.cpu_count() or 1)
    cat = {}
    ra0 = dec0 = None
    for b in (6, 1):
        z = np.load(work / f"bin{b - 1}.npz")
        ra, dec = z["ra"], z["dec"]
        if ra0 is None:                       # patch centre: densest-looking 1.0 deg disc of bin 6 around a fixed seed galaxy in the north footprint
            north = z["north"]
            i0 = np.random.default_rng(7).choice(np.flatnonzero(north)); ra0, dec0 = float(ra[i0]), float(dec[i0])
        dx = ((ra - ra0 + 180) % 360 - 180) * np.cos(np.deg2rad(dec0)); dy = dec - dec0
        rr = np.hypot(dx, dy); nt = A4_TARGET[b]
        rad = np.sort(rr)[nt] if rr.size > nt else rr.max()
        sel = rr <= rad
        cat[b] = {k: z[k][sel] for k in ("ra", "dec", "e1", "e2", "w")}
        cat[b]["n_total_bin"] = int(ra.size); cat[b]["radius_deg"] = float(rad)
        print(f"bin{b}: {sel.sum()} of {ra.size} galaxies ({100 * sel.sum() / ra.size:.3f} %) within {rad:.3f} deg of ({ra0:.3f},{dec0:.3f})", flush=True)
        del z
    out = {"what": "A4 recheck: independent pair counter (kidspipe.measure.paircount) vs TreeCorr on a SMALL SUBSAMPLE of the float64 DR5 extract",
           "subsample": f"disc of radius ~1 deg around RA,Dec = ({ra0:.4f},{dec0:.4f}); galaxies (bin, n, frac of bin): "
                        + "; ".join(f"bin{b}: {cat[b]['ra'].size} ({100 * cat[b]['ra'].size / cat[b]['n_total_bin']:.3f} %)" for b in cat),
           "nbins": A4_NBINS, "treecorr": treecorr.__version__, "pairs": {}}
    edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), A4_NBINS + 1))
    for p in A4_PAIRS:
        c1, c2 = cat[p[0]], cat[p[1]]
        r = {"n1": int(c1["ra"].size), "n2": int(c2["ra"].size), "auto": p[0] == p[1]}
        t1 = time.time()
        pc1 = PairCounter(c1["ra"], c1["dec"], c1["e1"], c1["e2"], c1["w"], 2.0, 300.0, A4_NBINS)
        pc2 = None if p[0] == p[1] else PairCounter(c2["ra"], c2["dec"], c2["e1"], c2["e2"], c2["w"], 2.0, 300.0, A4_NBINS)
        xip, xim, xix, npairs, wsum = pc1.run(pc2, chunk=1000)
        r["paircount"] = {"xip": xip, "xim": xim, "xix": xix, "npairs": npairs, "weight": wsum, "seconds": time.time() - t1}
        print(f"{pname(p)} paircount {time.time() - t1:.0f}s npairs={npairs.sum():.4e}", flush=True)
        k1 = make_treecorr_cat(c1); k2 = k1 if p[0] == p[1] else make_treecorr_cat(c2)
        for bs in (0.0, 1.5):
            for flip in (False, True):
                if bs == 1.5 and flip:
                    continue
                f1 = make_treecorr_cat(c1, flip_g2=flip); f2 = f1 if p[0] == p[1] else make_treecorr_cat(c2, flip_g2=flip)
                gg = treecorr.GGCorrelation(treecorr_kwargs(A4_NBINS, bs, nthreads)); t2 = time.time()
                gg.process(f1) if p[0] == p[1] else gg.process(f1, f2)
                a = corr_to_arrays(gg); a["seconds"] = time.time() - t2
                r[f"treecorr_bs{bs:g}_flipg2{int(flip)}"] = a
        out["pairs"][pname(p)] = r
        print(f"{pname(p)} done {time.time() - t0:.0f}s", flush=True)
    out["r_nom"] = np.sqrt(edges[:-1] * edges[1:]); out["runtime_s"] = time.time() - t0
    A4OUT.parent.mkdir(parents=True, exist_ok=True)
    A4OUT.write_text(json.dumps(out, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    print(f"wrote {A4OUT} in {time.time() - t0:.0f}s")


def plot6():
    import matplotlib.pyplot as plt
    A = json.loads(A4JSON.read_text()); r = np.array(A["r_nom"])
    fig, axs = plt.subplots(len(A["pairs"]), 4, figsize=(21, 9.5), squeeze=False)
    summ = {}
    for row, (pn, P) in enumerate(A["pairs"].items()):
        pc = {k: np.array(v) if isinstance(v, list) else v for k, v in P["paircount"].items()}
        T = {k: {kk: np.array(vv) for kk, vv in P[k].items()} for k in P if k.startswith("treecorr")}
        ok_all = pc["npairs"] > 0; ok = ok_all & (r <= A4_RMAX)
        tr0, trf, tr15 = T["treecorr_bs0_flipg2" + "0"], T["treecorr_bs0_flipg21"], T["treecorr_bs1.5_flipg20"]
        sg = np.sqrt(tr0["varxi"])
        for c, (key, lab) in enumerate((("xip", r"$\theta\,\xi_+$"), ("xim", r"$\theta\,\xi_-$"))):
            ax = axs[row, c]
            ax.plot(r[ok], r[ok] * 1e4 * tr0[key][ok], "-", color=C_OFF, lw=2.2, label="TreeCorr, g2 as stored")
            ax.plot(r[ok], r[ok] * 1e4 * trf[key][ok], "-", color=C_GREEN, lw=1.3, label="TreeCorr, flip_g2")
            ax.plot(r[ok], r[ok] * 1e4 * pc[key][ok], ".", color=C_OURS, ms=5, label="independent counter")
            ax.set_xscale("log"); ax.set_ylabel(lab + r" $[10^{-4}$ arcmin]"); ax.set_xlabel(r"$\theta$ [arcmin]")
            ax.set_title(f"{pn}: {key}", loc="left")
            if row == 0 and c == 0:
                ax.legend(fontsize=8.5)
        ax = axs[row, 2]
        for key, col, mk in (("xip", C_OURS, "o"), ("xim", C_OFF, "s")):
            ax.plot(r[ok], (pc[key][ok] - tr0[key][ok]) / sg[ok], mk, color=col, ms=3, alpha=0.8, label=f"{key} g2 as stored")
            ax.plot(r[ok], (pc[key][ok] - trf[key][ok]) / sg[ok], mk, mfc="none", color=col, ms=5, alpha=0.8, label=f"{key} flip_g2")
        ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-6, 6); ax.set_ylabel(r"(counter - TreeCorr)/$\sigma_{\rm TreeCorr}$ (bin_slop 0)")
        ax.set_title("difference in TreeCorr shape-noise sigma", loc="left"); ax.set_xlabel(r"$\theta$ [arcmin]")
        if row == 0:
            ax.legend(fontsize=7.5, ncol=2)
        ax = axs[row, 3]
        ax.plot(r[ok], pc["npairs"][ok] / tr0["npairs"][ok], "-", color=C_OURS, label="npairs counter / TreeCorr bs0")
        ax.plot(r[ok], pc["weight"][ok] / tr0["weight"][ok], "--", color=C_GREEN, label="weight counter / TreeCorr bs0")
        ax.set_ylim(0.99, 1.01); ax.plot(r[ok], tr15["npairs"][ok] / tr0["npairs"][ok], ":", color=C_OFF, label="npairs TreeCorr bs1.5 / bs0")
        ax.set_xscale("log"); ax.set_xlabel(r"$\theta$ [arcmin]"); ax.set_ylabel("ratio"); ax.set_title("pair counts", loc="left")
        ax.legend(fontsize=8, loc="best")
        s = {}
        for pre, mk in (("", ok), ("allbins_", ok_all)):
            for key in ("xip", "xim"):
                for lab, t in (("stored", tr0), ("flip_g2", trf)):
                    s[f"{pre}{key}_{lab}_max_abs_z"] = float(np.abs((pc[key][mk] - t[key][mk]) / sg[mk]).max())
                    s[f"{pre}{key}_{lab}_max_rel_scale"] = float(np.abs(pc[key][mk] - t[key][mk]).max() / np.abs(t[key][mk]).max())
            s[f"{pre}npairs_ratio_min"] = float((pc["npairs"][mk] / tr0["npairs"][mk]).min()); s[f"{pre}npairs_ratio_max"] = float((pc["npairs"][mk] / tr0["npairs"][mk]).max())
            s[f"{pre}weight_ratio_min"] = float((pc["weight"][mk] / tr0["weight"][mk]).min()); s[f"{pre}weight_ratio_max"] = float((pc["weight"][mk] / tr0["weight"][mk]).max())
            s[f"{pre}n_bins"] = int(mk.sum())
        s["xip_mean_abs_z_stored"] = float(np.abs((pc["xip"][ok] - tr0["xip"][ok]) / sg[ok]).mean())
        s["xim_mean_abs_z_flip"] = float(np.abs((pc["xim"][ok] - trf["xim"][ok]) / sg[ok]).mean())
        s["xim_mean_abs_z_stored"] = float(np.abs((pc["xim"][ok] - tr0["xim"][ok]) / sg[ok]).mean())
        summ[pn] = s
    fig.suptitle("Plot 6: estimator cross-check on a SUBSAMPLE: " + A["subsample"] + f"; {A['nbins']} log bins.\nIndependent counter (kidspipe.measure.paircount) vs TreeCorr {A['treecorr']}, bin_slop 0; shown theta <= {A4_RMAX:g} arcmin (patch ~2 deg across)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, summ


# ------------------------------------------------------------------------------------------------ main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plots"
    if cmd == "a4":
        return run_a4()
    import matplotlib
    matplotlib.use("Agg")
    style(); PLOTS.mkdir(parents=True, exist_ok=True)
    u = _u()
    d = dict(np.load(PRODUCT))
    ctx = u.load_context(); fx, cb, asc, filt = ctx["fx"], ctx["cb"], ctx["asc"], ctx["filt"]
    info = {}

    def save(fig, name):
        fig.savefig(PLOTS / name, bbox_inches="tight"); print("wrote", PLOTS / name, flush=True)
    save(plot1(d, fx), "dr5_01_xipm_vs_official.png")
    f, st = plot2(d, asc, u); save(f, "dr5_02_npairs_weights_ratio.png"); info["plot2"] = st
    save(plot3(d, cb, "En"), "dr5_03_En_vs_official.png")
    save(plot3(d, cb, "Bn"), "dr5_03b_Bn_vs_official.png")
    f, s4 = plot4(d, cb, asc, filt); save(f, "dr5_04_integration_vs_measurement.png"); info["plot4"] = s4
    save(plot5(d, cb), "dr5_05_Bn_null.png")
    if A4JSON.exists():
        f, s6 = plot6(); save(f, "dr5_06_estimator_crosscheck.png"); info["plot6"] = s6
    else:
        print("a4_recheck.json missing: plot 6 skipped (run `a4` first)")
    print(json.dumps(info, indent=1, default=float))


if __name__ == "__main__":
    main()
