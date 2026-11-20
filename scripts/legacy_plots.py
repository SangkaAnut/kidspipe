"""Figures and summary numbers for the KiDS-Legacy COSEBIs validation (called by legacy_validate.py plots).

Inputs <runs>/legacy_val/{chi2,selection}.npz (else the copies shipped in <root>/runs); figures to $KIDSPIPE_PLOTS (default <root>/plots),
summary.json to <runs>/legacy_val/."""
import json, os, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
OUT = paths.runs_dir() / "legacy_val"                    # summary.json is written here
PLOTS = Path(os.environ["KIDSPIPE_PLOTS"]).expanduser() if os.environ.get("KIDSPIPE_PLOTS") else ROOT / "plots"   # figures
CACHE = str(paths.runs_dir() / "legacy" / "cache")
C = dict(off="#1f4e79", ours="#d95f02", top="#1b9e77", rnd="#7570b3", gray="#666666")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100, "savefig.dpi": 140})


def wquant(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, x[o])


def wstats(x, w):
    w = w / w.sum(); m = np.sum(w * x); s = np.sqrt(np.sum(w * (x - m) ** 2))
    q = wquant(x, w, [0.1587, 0.5, 0.8413])
    return dict(mean=m, std=s, median=q[1], lo=q[1] - q[0], hi=q[2] - q[1])


def ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2))


def credible_levels(H, fr=(0.6827, 0.9545)):
    f = H.ravel(); o = np.argsort(f)[::-1]; cs = np.cumsum(f[o]) / f.sum()
    return sorted([f[o][np.searchsorted(cs, q)] for q in fr])


def stage_plots(a):
    from kidspipe.legacy import (LegacyCosebisModel, LegacyCosebisLikelihood, LegacyOptions, LegacyData, load_chain,
                                 chain_samples)
    from kidspipe.legacy.data import PAIRS
    from kidspipe.theory.cosebis import wn_of_l
    chain = load_chain()
    data = LegacyData.load()
    like = LegacyCosebisLikelihood(data=data)
    z = np.load(paths.find_result("legacy_val/chi2.npz")); sel = np.load(paths.find_result("legacy_val/selection.npz"))   # yours, else shipped
    summary = {}

    # ===================================================================== 1. chi2 comparison
    ok = np.isfinite(z["chi2"])
    row = z["row"][ok]; c_our = z["chi2"][ok]; c_off = -2 * (chain[row, 43] - chain[row, 42])
    d = c_our - c_off
    topset = set(sel["top"].tolist()); is_top = np.array([r in topset for r in row])
    draw_set = np.unique(sel["draws"]); is_draw = np.isin(row, draw_set)
    summary["chi2_comparison"] = dict(n_requested=int(len(z["row"])), n_ok=int(ok.sum()), n_failed=int((~ok).sum()))

    def dstat(m):
        x = d[m]
        if x.size < 2: return None
        return dict(n=int(m.sum()), mean=float(x.mean()), std=float(x.std(ddof=1)), median=float(np.median(x)),
                    max_abs=float(np.abs(x).max()), p95_abs=float(np.percentile(np.abs(x), 95)),
                    frac_abs_lt_0p5=float(np.mean(np.abs(x) < 0.5)), frac_abs_lt_1=float(np.mean(np.abs(x) < 1)),
                    corr=float(np.corrcoef(c_our[m], c_off[m])[0, 1]),
                    chi2_off_range=[float(c_off[m].min()), float(c_off[m].max())])
    summary["chi2_comparison"].update(all=dstat(np.ones_like(is_top)), top150=dstat(is_top), drawn=dstat(is_draw),
                                      # posterior-weighted statistics of the difference (counts of the draws)
                                      )
    cnt = {r: c for r, c in zip(*np.unique(sel["draws"], return_counts=True))}
    wd = np.array([cnt.get(r, 0) for r in row], float)
    mw = wd > 0
    mean_w = np.sum(wd[mw] * d[mw]) / wd[mw].sum(); std_w = np.sqrt(np.sum(wd[mw] * (d[mw] - mean_w) ** 2) / wd[mw].sum())
    summary["chi2_comparison"]["posterior_weighted_diff"] = dict(mean=float(mean_w), std=float(std_w))
    worst = np.argsort(-np.abs(d))[:5]
    summary["chi2_comparison"]["worst5"] = [dict(row=int(row[i]), diff=float(d[i]), chi2_off=float(c_off[i]),
                                                 omch2=float(chain[row[i], 0]), log_t_agn=float(chain[row[i], 5]),
                                                 sigma8_off=float(chain[row[i], 21]), sigma8_ours=float(z["sigma8"][ok][i]))
                                            for i in worst]
    sig_rel = z["sigma8"][ok] / chain[row, 21] - 1
    summary["chi2_comparison"]["sigma8_ours_over_official_minus1"] = dict(mean=float(sig_rel.mean()), std=float(sig_rel.std()))
    summary["chi2_comparison"]["corr_diff_vs_sigma8_mismatch"] = float(np.corrcoef(d, sig_rel)[0, 1])
    summary["chi2_comparison"]["corr_diff_vs_omch2"] = float(np.corrcoef(d, chain[row, 0])[0, 1])
    from scipy.stats import spearmanr
    names = ["omch2", "ombh2", "h0", "n_s", "S8_input", "log_t_agn", "a_uncorr", "beta_uncorr"] + [f"lm{i}" for i in range(1, 7)] + [f"dz{i}" for i in range(1, 7)]
    summary["chi2_comparison"]["spearman_diff_vs_param"] = {n: float(spearmanr(chain[row, k], d)[0]) for k, n in enumerate(names)}
    summary["chi2_comparison"]["spearman_diff_vs_chi2_off"] = float(spearmanr(c_off, d)[0])
    best_row = int(np.argmax(np.where(np.isfinite(chain[:, 43]), chain[:, 43], -np.inf)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    A = ax[0]
    A.scatter(c_off[~is_top], c_our[~is_top], s=6, c=C["rnd"], alpha=0.5, label=f"weighted draws (n={int((~is_top).sum())})", rasterized=True)
    A.scatter(c_off[is_top], c_our[is_top], s=12, c=C["top"], alpha=0.9, label=f"top posterior (n={int(is_top.sum())})")
    lo, hi = min(c_off.min(), c_our.min()) - 1, max(c_off.max(), c_our.max()) + 1
    A.plot([lo, hi], [lo, hi], "k-", lw=1, label="1:1")
    A.set(xlabel=r"official $\chi^2$ = $-2(\mathrm{post}-\mathrm{prior})$", ylabel=r"our $\chi^2$", xlim=(lo, hi), ylim=(lo, hi))
    A.set_aspect("equal"); A.legend(loc="upper left", fontsize=8)
    A.set_title(f"N={len(d)}: corr {summary['chi2_comparison']['all']['corr']:.4f}")
    A = ax[1]
    bins = np.linspace(-4, 4, 65)
    A.hist(d[~is_top], bins=bins, color=C["rnd"], alpha=0.6, label="weighted draws")
    A.hist(d[is_top], bins=bins, color=C["top"], alpha=0.7, label="top posterior")
    A.axvline(0, c="k", lw=1)
    s = summary["chi2_comparison"]["all"]
    A.set(xlabel=r"$\chi^2_{\rm ours}-\chi^2_{\rm off}$", ylabel="samples")
    A.set_title(f"mean {s['mean']:+.3f}, std {s['std']:.3f}, max|d| {s['max_abs']:.2f}")
    A.legend(fontsize=8)
    A = ax[2]
    sc = A.scatter(c_off, d, c=chain[row, 5], s=6, cmap="viridis", alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=A, label=r"$\log_{10}T_{\rm AGN}$")
    A.axhline(0, c="k", lw=1)
    A.set(xlabel=r"official $\chi^2$", ylabel=r"$\Delta\chi^2$ (ours - official)", ylim=(-8, 4))
    A.set_title(f"residual vs official chi2 (corr {np.corrcoef(c_off, d)[0, 1]:+.2f})")
    fig.savefig(PLOTS / "legacy_01_chi2_vs_official.png"); plt.close(fig)

    # ===================================================================== 2. best-sample data vs model
    nth = int(a.workers) if False else 2
    model = LegacyCosebisModel(nthreads=nth, cache_dir=CACHE, cache_size=2)
    s = chain_samples(chain, [best_row])[0]
    mu = model.mu(s)
    sel6 = data.select(6)
    dvec = data.value[sel6]; err = np.sqrt(np.diag(data.cov)[sel6]); cov6 = data.cov[np.ix_(sel6, sel6)]
    chi2_best = like.chi2(mu)
    Lc = np.linalg.cholesky(cov6)
    white = solve_triangular(Lc, dvec - mu, lower=True)
    summary["best_sample"] = dict(row=best_row, chi2_ours=float(chi2_best), chi2_off=float(s["chi2_off"]),
                                  chi2_over_126=float(chi2_best / 126), pte_126dof=float(chi2dist.sf(chi2_best, 126)),
                                  white_resid_mean=float(white.mean()), white_resid_std=float(white.std()),
                                  chi2_per_pair=[float(np.sum(white[6 * k:6 * k + 6] ** 2)) for k in range(21)])
    # B-mode check
    selB = 420 + sel6
    dB = data.value_B[sel6]; covB = data.cov[np.ix_(selB, selB)]
    chiB = float(dB @ np.linalg.solve(covB, dB))
    sel20 = np.arange(420); dB20 = data.value_B; cov20 = data.cov[420:, 420:]
    chiB20 = float(dB20 @ np.linalg.solve(cov20, dB20))
    # joint En-Bn covariance (modes 1-6): Bn vs 0 using the conditional on nothing (marginal) is the Bn block; E-B cross-corr size
    cross = data.cov[np.ix_(sel6, selB)]
    summary["bmode"] = dict(chi2_Bn_modes1to6=chiB, dof=126, pte=float(chi2dist.sf(chiB, 126)),
                            chi2_Bn_all20modes=chiB20, dof20=420, pte20=float(chi2dist.sf(chiB20, 420)),
                            max_abs_En_Bn_cross_corr=float(np.abs(cross / np.outer(err, np.sqrt(np.diag(covB)))).max()),
                            chi2_En_null_model_zero=float(dvec @ np.linalg.solve(cov6, dvec)))
    # per-mode: first Bn modes
    whiteB = solve_triangular(np.linalg.cholesky(covB), dB, lower=True)

    fig, axs = plt.subplots(6, 6, figsize=(15, 11), sharex=True, constrained_layout=True)
    modes = np.arange(1, 7)
    for k, (i, j) in enumerate(PAIRS):
        A = axs[j, i]
        A.errorbar(modes, dvec[6 * k:6 * k + 6] * 1e10, err[6 * k:6 * k + 6] * 1e10, fmt="o", c=C["off"], ms=4, capsize=2, label="data")
        A.plot(modes, mu[6 * k:6 * k + 6] * 1e10, "s-", c=C["ours"], ms=4, lw=1, label="our model")
        A.axhline(0, c="k", lw=0.5)
        A.set_title(f"bins {i+1}x{j+1}  $\\chi^2$={summary['best_sample']['chi2_per_pair'][k]:.1f}", fontsize=9)
        A.set_xticks(modes)
        if j == 5: A.set_xlabel("mode n")
        if i == 0: A.set_ylabel(r"$E_n\,[10^{-10}]$")
    for j in range(6):
        for i in range(j + 1, 6):
            axs[j, i].axis("off")
    axs[0, 3].legend(*axs[0, 0].get_legend_handles_labels(), loc="center", fontsize=11, frameon=False)
    axs[0, 3].text(0.5, -0.45, f"official best sample (chain row {best_row})\n$\\chi^2_{{ours}}$={chi2_best:.2f}, "
                   f"$\\chi^2_{{off}}$={s['chi2_off']:.2f}, 126 pts", transform=axs[0, 3].transAxes, ha="center", fontsize=10)
    axs[0, 3].axis("off") if False else None
    fig.suptitle("KiDS-Legacy COSEBIs: E_n data (error bars = sqrt(diag cov)) vs our model at the official max-posterior sample")
    fig.savefig(PLOTS / "legacy_02_En_data_vs_model.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    A = ax[0]
    A.plot(np.arange(126), white, "o", ms=3, c=C["ours"])
    A.axhline(0, c="k", lw=0.7)
    for k in range(1, 21): A.axvline(6 * k - 0.5, c="gray", lw=0.3)
    A.set(xlabel="data index (21 pairs x 6 modes)", ylabel=r"whitened residual $L^{-1}(d-\mu)$",
          title=f"E_n: chi2={chi2_best:.1f}; mean {white.mean():+.2f}, std {white.std():.2f}")
    A = ax[1]
    A.hist(white, bins=np.linspace(-4, 4, 25), density=True, color=C["ours"], alpha=0.7)
    xx = np.linspace(-4, 4, 200); A.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k-", label="N(0,1)")
    A.set(xlabel="whitened residual", ylabel="density", title="E_n whitened residuals"); A.legend()
    A = ax[2]
    A.errorbar(np.arange(126), dB * 1e10, np.sqrt(np.diag(covB)) * 1e10, fmt="o", ms=3, c=C["off"], capsize=1.5)
    A.axhline(0, c="k", lw=0.7)
    for k in range(1, 21): A.axvline(6 * k - 0.5, c="gray", lw=0.3)
    A.set(xlabel="data index (21 pairs x 6 modes)", ylabel=r"$B_n\,[10^{-10}]$",
          title=f"B_n vs 0: chi2={chiB:.1f}/126 (PTE {summary['bmode']['pte']:.2f}); all 20 modes {chiB20:.0f}/420 (PTE {summary['bmode']['pte20']:.2f})")
    A.title.set_fontsize(9)
    fig.savefig(PLOTS / "legacy_02b_residuals_Bmodes.png"); plt.close(fig)

    # ===================================================================== 3. importance reweighting
    idx_all = np.where(np.isfinite(chain[:, 43]))[0]
    lw = chain[idx_all, 41]; w_off_full = np.exp(lw - lw.max()); w_off_full /= w_off_full.sum()
    rows_u, counts = np.unique(sel["draws"], return_counts=True)
    pos = {int(r): k for k, r in enumerate(row)}
    keep = np.array([int(r) in pos and np.isfinite(c_our[pos[int(r)]]) for r in rows_u])
    rows_u, counts = rows_u[keep], counts[keep].astype(float)
    dlnl = -(c_our[[pos[int(r)] for r in rows_u]] - c_off[[pos[int(r)] for r in rows_u]]) / 2   # lnL_ours - lnL_off
    ratio = np.exp(dlnl - dlnl.max())
    w_sub = counts; w_rw = counts * ratio
    pars = {"S8": 4, "Omega_m": 23, "sigma8": 21}
    res = {}
    for nme, c in pars.items():
        x_full = chain[idx_all, c]; x_sub = chain[rows_u, c]
        res[nme] = dict(official_full_chain=wstats(x_full, w_off_full), official_draw_subset=wstats(x_sub, w_sub),
                        reweighted=wstats(x_sub, w_rw))
    # bootstrap uncertainty of the shift reweighted - subset (resample the unique draws with replacement)
    rng = np.random.default_rng(3)
    boot = {nme: [] for nme in pars}
    for _ in range(300):
        b = rng.integers(0, len(rows_u), len(rows_u))
        for nme, c in pars.items():
            x = chain[rows_u[b], c]
            boot[nme].append(np.sum(w_rw[b] * x) / w_rw[b].sum() - np.sum(w_sub[b] * x) / w_sub[b].sum())
    for nme in pars:
        res[nme]["shift_reweighted_minus_subset"] = dict(mean=float(np.mean(boot[nme])), bootstrap_std=float(np.std(boot[nme])))
        res[nme]["shift_reweighted_minus_fullchain"] = res[nme]["reweighted"]["mean"] - res[nme]["official_full_chain"]["mean"]
    res["ess"] = dict(n_unique_rows=int(len(rows_u)), n_draws=int(counts.sum()), ess_subset_counts=ess(w_sub), ess_reweighted=ess(w_rw),
                      ess_ratio_only=ess(ratio), ess_fraction=ess(w_rw) / ess(w_sub), official_chain_neff=ess(w_off_full),
                      dlnl_mean_weighted=float(np.sum(w_sub * dlnl) / w_sub.sum()), dlnl_std_weighted=float(np.sqrt(np.sum(w_sub * (dlnl - np.sum(w_sub * dlnl) / w_sub.sum()) ** 2) / w_sub.sum())),
                      ratio_max_over_min=float(np.exp(dlnl.max() - dlnl.min())))
    res["published_S8_Wright2025_COSEBIs"] = dict(mean=0.815, plus=0.016, minus=0.021, note="external value; see doc for verification status")
    summary["reweighting"] = res
    S8o, S8s = chain[idx_all, 4], chain[rows_u, 4]
    fig = plt.figure(figsize=(15, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    for k, (nme, c, lab) in enumerate([("S8", 4, r"$S_8$"), ("Omega_m", 23, r"$\Omega_{\rm m}$"), ("sigma8", 21, r"$\sigma_8$")]):
        A = fig.add_subplot(gs[0, k])
        xa = chain[idx_all, c]; xb = chain[rows_u, c]
        lo, hi = wquant(xa, w_off_full, [0.0005, 0.9995])
        bins = np.linspace(lo, hi, 50)
        A.hist(xa, bins=bins, weights=w_off_full, density=True, histtype="step", color=C["off"], lw=2,
               label=f"official ({res[nme]['official_full_chain']['mean']:.4f}$\\pm${res[nme]['official_full_chain']['std']:.4f})")
        A.hist(xb, bins=bins, weights=w_rw / w_rw.sum(), density=True, histtype="step", color=C["ours"], lw=2, ls="--",
               label=f"reweighted ({res[nme]['reweighted']['mean']:.4f}$\\pm${res[nme]['reweighted']['std']:.4f})")
        A.hist(xb, bins=bins, weights=w_sub / w_sub.sum(), density=True, histtype="step", color=C["gray"], lw=1, ls=":",
               label="official, same draws (noise ref.)")
        if nme == "S8":
            A.axvspan(0.815 - 0.021, 0.815 + 0.016, color="gold", alpha=0.3, label="published 0.815 (+0.016/-0.021)")
        A.set(xlabel=lab, ylabel="posterior density"); A.legend(fontsize=7, loc="upper left" if nme != "S8" else "upper right")
    A = fig.add_subplot(gs[1, 0:2])
    xr, yr = (chain[idx_all, 23], chain[idx_all, 4])
    xe = np.linspace(0.05, 0.6, 70); ye = np.linspace(0.65, 1.0, 70)
    def H2(x, y, w):
        h, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=w); return gaussian_filter(h, 1.2)
    for H, col, ls, lab in [(H2(xr, yr, w_off_full), C["off"], "-", "official (full chain)"),
                            (H2(chain[rows_u, 23], S8s, w_rw), C["ours"], "--", "reweighted"),
                            (H2(chain[rows_u, 23], S8s, w_sub), C["gray"], ":", "official, same draws")]:
        lv = credible_levels(H)
        A.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=lv, colors=col, linestyles=ls, linewidths=1.8)
        A.plot([], [], c=col, ls=ls, label=lab)
    A.axhspan(0.815 - 0.021, 0.815 + 0.016, color="gold", alpha=0.25)
    A.set(xlabel=r"$\Omega_{\rm m}$", ylabel=r"$S_8$", xlim=(0.1, 0.5), ylim=(0.72, 0.9)); A.legend(fontsize=8)
    A.set_title("68% and 95% contours (gold band: published S8)")
    A = fig.add_subplot(gs[1, 2])
    A.hist(dlnl, bins=40, weights=counts, color=C["ours"], alpha=0.7)
    A.set(xlabel=r"$\ln L_{\rm ours}-\ln L_{\rm off}$", ylabel="draws",
          title=f"ESS {res['ess']['ess_reweighted']:.0f} / {res['ess']['ess_subset_counts']:.0f} ({100*res['ess']['ess_fraction']:.1f}%)")
    fig.savefig(PLOTS / "legacy_03_S8_official_vs_ours.png"); plt.close(fig)

    # ===================================================================== 4. filters
    filt = model.filters
    th = np.logspace(np.log10(2.0), np.log10(300.0), 800)
    Tp = filt.Tplus(th, 6); Tm = filt.Tminus(th, 6)
    ell = np.logspace(0, 5, 500)
    Wn = wn_of_l(ell, filt, 6, nthreads=nth)
    G = filt.gram(6) / (filt.tmax - filt.tmin)
    summary["filters"] = dict(gram_max_offdiag=float(np.abs(G - np.diag(np.diag(G))).max()), gram_diag=np.diag(G).tolist(),
                              tmin=filt.tmin, tmax=filt.tmax, roots_source=filt.source)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    cm = plt.cm.viridis(np.linspace(0, 0.9, 6))
    for n in range(6):
        ax[0].plot(th, Tp[n], c=cm[n], label=f"n={n+1}"); ax[1].plot(th, Tm[n], c=cm[n])
        ax[2].plot(ell, ell * Wn[n], c=cm[n])
    ax[0].set(xscale="log", xlabel=r"$\theta$ [arcmin]", ylabel=r"$T_{+n}(\theta)$", title="T_+n, 2-300 arcmin"); ax[0].legend(ncol=2, fontsize=8)
    ax[1].set(xscale="log", xlabel=r"$\theta$ [arcmin]", ylabel=r"$T_{-n}(\theta)$", title="T_-n")
    ax[2].set(xscale="log", xlabel=r"$\ell$", ylabel=r"$\ell\,W_n(\ell)$ [rad$^2$]", title=f"W_n(l); filter Gram off-diag max {summary['filters']['gram_max_offdiag']:.1e}")
    ax[2].axvline(1e5, c="gray", ls=":")
    for A in ax: A.axhline(0, c="k", lw=0.5)
    fig.savefig(PLOTS / "legacy_04_filters.png"); plt.close(fig)

    # ===================================================================== 5. GG / IA contributions
    m_gg = LegacyCosebisModel(replace(model.opt, n_amp_scale=0.0), nthreads=nth, cache_dir=CACHE, cache_size=2, filters=model.filters)
    mu_gg = m_gg.mu(s); mu_ia = mu - mu_gg
    frac = mu_ia / mu
    summary["contributions"] = dict(IA_over_total_mode1_per_pair=[float(frac[6 * k]) for k in range(21)],
                                    IA_over_total_mean_over_all=float(np.mean(frac)),
                                    IA_over_total_range=[float(frac.min()), float(frac.max())],
                                    chi2_without_IA=float(like.chi2(mu_gg)), chi2_with_IA=float(chi2_best),
                                    A_i=[float(x) for x in model.amplitudes(s)], a=float(s["a"]), beta=float(s["beta"]),
                                    IA_negative_fraction=float(np.mean(mu_ia < 0)))
    fig, axs = plt.subplots(6, 6, figsize=(15, 11), sharex=True, constrained_layout=True)
    for k, (i, j) in enumerate(PAIRS):
        A = axs[j, i]; sl = slice(6 * k, 6 * k + 6)
        A.fill_between(modes, 1e-3 * 0 + 1e10 * 1e-30 + 1e-1, 1e-1, alpha=0)  # placeholder to keep log axis
        A.semilogy(modes, mu_gg[sl] * 1e10, "o-", c=C["off"], ms=4, label="GG")
        pos_ = mu_ia[sl] > 0
        A.semilogy(modes, np.abs(mu_ia[sl]) * 1e10, "s--", c=C["ours"], ms=4, label="|IA| (GI+II)")
        A.semilogy(modes, err[sl] * 1e10, ":", c=C["gray"], label="data error")
        A.set_title(f"bins {i+1}x{j+1}  IA/tot(n=1)={frac[6*k]:+.2f}", fontsize=9); A.set_xticks(modes)
        if j == 5: A.set_xlabel("mode n")
        if i == 0: A.set_ylabel(r"$E_n\,[10^{-10}]$")
        if np.any(pos_): A.plot(modes[pos_], np.abs(mu_ia[sl])[pos_] * 1e10, "k^", ms=6)
    for j in range(6):
        for i in range(j + 1, 6): axs[j, i].axis("off")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 4].legend(h, l, loc="center", fontsize=11, frameon=False)
    axs[0, 4].text(0.5, 0.05, "IA part is negative (GI-dominated) at all points\nexcept black triangles (positive)", transform=axs[0, 4].transAxes, ha="center", fontsize=9)
    fig.suptitle(f"E_n contributions at the official best sample: shear-shear (no IA) and IA part (total - GG), a={s['a']:.2f}, beta={s['beta']:.2f}")
    fig.savefig(PLOTS / "legacy_05_contributions.png"); plt.close(fig)

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    json.dump(clean(summary), open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(clean(summary), indent=1)[:6000])
