"""COSEBIs validation numbers + plots (writes plots/cosebis_*.png and runs/cosebis/validation_numbers.json).

Run:  PYTHONPATH=src OMP_NUM_THREADS=4 python3 scripts/cosebis_validate.py [stage]
  stage = main   filters, W_n, routes, data-vs-model, covariance, chi2 (plots 01-05, 07, 08)  [~2 min]
  stage = post   COSEBIs-only posterior (plot 06) from runs/cosebis/chain_n5_c*.npz (made by runs/cosebis/mcmc.py)
                 + importance reweighting of the xi+/- chain (runs/cosebis/reweight_n5.npz, computed if missing)
  stage = all    both (default)
"""
import glob
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular

from kidspipe.api import build_model
from kidspipe.theory import cosebis as C
from kidspipe.likelihood import cosebis as LC
from kidspipe.likelihood.data import PAIRS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
RUN = ROOT / "runs" / "cosebis"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
LAB = [f"{i + 1}-{j + 1}" for (j, i) in PAIRS]          # tomographic bin labels (lower bin first)
best = json.load(open(ROOT / "runs/kv450_final/map_best.json"))
th_best = np.array(best["theta"])
NAMES = best["names"]


def xi_chain(step=40):
    fs = sorted(glob.glob(str(ROOT / "runs/kv450_final/prod*/prod_c*.npz")))
    S, D, LP = [], [], []
    for f in fs:
        z = np.load(f)
        S.append(z["samples"][1000::step]); D.append(z["derived"][1000::step]); LP.append(z["log_post"][1000::step])
    return np.concatenate(S), np.concatenate(D), np.concatenate(LP)


def pval(c, n):
    return float(stats.chi2.sf(c, n))


# ======================================================================================== main stage
def stage_main():
    res = {}
    t00 = time.time()
    F = C.CosebisFilters(0.5, 300.0, 20)
    F100 = C.CosebisFilters(0.5, 100.0, 20)

    # ---------------------------------------------------------------- 01 filters
    th = np.geomspace(0.5, 300, 3000)
    th100 = np.geomspace(0.5, 100, 2000)
    acc = {}
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 1.0])
    Tp, Tm = F.Tplus(th, 5), F.Tminus(th, 5)
    for k, (T, nm) in enumerate(((Tp, "$T_{+n}$"), (Tm, "$T_{-n}$"))):
        a = fig.add_subplot(gs[0, k])
        for n in range(5):
            a.plot(th, T[n], label=f"n={n + 1}")
        a.set_xscale("log"); a.set_xlabel(r"$\theta$ [arcmin]"); a.set_title(nm + r"($\theta$), computed, 0.5'-300' (roots from table)")
        a.set_ylim(-3.2, 3.2) if k == 0 else a.set_ylim(-6, 6)
        a.legend(ncol=5, fontsize=7, loc="upper right")
    # overlay of tabulated 0.5-100 tables on ours
    Tp100, Tm100 = F100.Tplus(th100, 20), F100.Tminus(th100, 20)
    ovl = (1, 3, 5)
    for k, (Tc, name) in enumerate(((Tp100, "Tplus"), (Tm100, "Tminus"))):
        a = fig.add_subplot(gs[1, k])
        for c, n in enumerate(ovl):
            tab = np.loadtxt(ROOT / f"data/cosebis/Tplus_minus/{name}{n}_0.50-100.00.table")
            a.plot(th100, Tc[n - 1], "-", color=f"C{c}", lw=2.2, alpha=0.6, label=f"computed n={n}")
            a.plot(tab[::25, 0], tab[::25, 1], "k.", ms=3, label="tabulated" if c == 0 else None)
        a.set_xscale("log"); a.set_xlabel(r"$\theta$ [arcmin]")
        a.set_title(f"{name} overlay, 0.5'-100' filters (line = ours, dots = table)"); a.legend(ncol=4, fontsize=7)
    # residuals
    for k, (Tc, name) in enumerate(((Tp100, "Tplus"), (Tm100, "Tminus"))):
        a = fig.add_subplot(gs[2, k])
        for n in (1, 2, 5, 10, 20):
            tab = np.loadtxt(ROOT / f"data/cosebis/Tplus_minus/{name}{n}_0.50-100.00.table")
            Tn = (F100.Tplus(tab[:, 0], 20) if k == 0 else F100.Tminus(tab[:, 0], 20))[n - 1]
            r = (Tn - tab[:, 1]) / np.abs(tab[:, 1]).max()
            acc[f"{name}_{n}_maxdiff_over_peak"] = float(np.abs(r).max())
            a.plot(tab[:, 0], np.maximum(np.abs(r), 1e-18), lw=0.8, label=f"n={n}")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\theta$ [arcmin]")
        a.set_ylabel("|ours - table| / peak"); a.set_title(f"{name}: residual vs table"); a.legend(ncol=5, fontsize=7)
        a.set_ylim(1e-18, 1e-3)
    fig.tight_layout(); fig.savefig(OUT / "cosebis_01_filters.png"); plt.close(fig)

    # filter identities
    G = F.gram(20)
    acc["gram_max_abs_err_over_299.5"] = float(np.abs(G - 299.5 * np.eye(20)).max() / 299.5)
    a_, b_, s_ = F.constraints(20)
    acc["int_e2z_T_over_int|T|e4z_max"] = float(np.abs(a_ / s_).max())
    acc["int_e4z_T_over_int|T|e4z_max"] = float(np.abs(b_ / s_).max())
    acc["gram_first5_max_rel_err"] = float(np.abs(F.gram(5) - 299.5 * np.eye(5)).max() / 299.5)
    roots, norms, _ = C.regenerate_roots_norms(10, 0.5, 300.0, dps=80)
    acc["roots_maxabs_diff_mpmath_vs_table_by_n"] = {n: float(np.abs(np.array([float(x) for x in roots[n]]) - F.roots[n]).max()) for n in range(1, 11)}
    acc["norm_rel_diff_mpmath_vs_table_max"] = float(max(abs(float(norms[n]) / F.norms[n] - 1) for n in range(1, 11)))
    # independent check of T_- : mpmath adaptive quadrature at 3 points for n=2,5
    import mpmath as mp
    mp.mp.dps = 30
    dev = 0.0
    for n in (2, 5):
        r = [mp.mpf(float(x)) for x in F.roots[n]]
        Tpf = lambda y: mp.mpf(F.norms[n]) * mp.fprod([y - ri for ri in r])
        for tt in (3.0, 20.0, 150.0):
            z = mp.log(mp.mpf(tt) / mp.mpf("0.5"))
            ref = Tpf(z) + mp.quad(lambda y: 4 * Tpf(y) * (mp.e ** (2 * (y - z)) - 3 * mp.e ** (4 * (y - z))), [0, z / 2, z])
            dev = max(dev, float(abs(ref - F.Tminus(np.array([tt]), n)[n - 1][0])))
    acc["Tminus_vs_mpmath_quad_maxabs_n2_n5"] = dev
    res["filters"] = acc

    # ---------------------------------------------------------------- model + C_ell
    m = build_model(linear_backend="emulator")
    ell, Cl = m.cl(th_best)
    M20 = C.cosebis_operator(ell, F, 20)
    E_cl = C.E_from_cl(Cl, M20)
    thr, xp, xm = m.binner.xi_theta(Cl)
    E_xi, B_xi = C.E_from_xi(thr, xp, xm, F, 20)
    data = LC.CosebisData.load()
    sig = np.sqrt(np.diag(data.cov)).reshape(15, 20)
    res["route_agreement"] = {
        "max_abs_diff_over_sigma_by_mode": (np.abs(E_cl - E_xi) / sig).max(0).tolist(),
        "max_abs_diff_over_sigma_all": float((np.abs(E_cl - E_xi) / sig).max()),
        "max_rel_diff_modes1_5": float((np.abs(E_cl - E_xi)[:, :5] / np.abs(E_xi)[:, :5]).max()),
        "max_B_over_max_E_by_mode": (np.abs(B_xi).max(0) / np.abs(E_xi).max(0)).tolist()}

    # ---------------------------------------------------------------- 02 W_n(l) + integrand + l-convergence
    conv = []
    for lmax in (1e3, 3e3, 1e4, 3e4, 6e4, C.L_MAX_KCAP, 3e5, 5e5):
        M = C.cosebis_operator(ell, F, 5, lmin=1.0, lmax=lmax, use_cache=False, dl_max=24 if lmax < 2e5 else 48)
        E = C.E_from_cl(Cl, M)
        conv.append((lmax, float((np.abs(E - E_xi[:, :5]) / sig[:, :5]).max())))
    res["lmax_convergence_max_diff_over_sigma_n<=5"] = [{"lmax": a, "diff_sigma": b} for a, b in conv]
    lw = np.geomspace(1, 2e5, 6000)
    W = C.wn_of_l(lw, F, 5)
    pp = PAIRS.index((2, 2))                                       # bins 3-3
    Cint = np.exp(np.interp(np.log(lw), np.log(ell), np.log(Cl[2, 2])))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    lz = np.linspace(1, 1500, 6000)
    Wz = C.wn_of_l(lz, F, 5)
    for n in range(5):
        ax[0, 0].plot(lz, Wz[n] * 1e3, lw=0.7, label=f"n={n + 1}")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("l"); ax[0, 0].set_ylabel(r"$W_n(l)\ [10^{-3}]$"); ax[0, 0].legend(ncol=5, fontsize=7)
    ax[0, 0].set_title(r"$W_n(l)=\int d\theta\,\theta\,T_{+n}(\theta)J_0(l\theta)$, $\theta$ in rad (l<1500)", fontsize=9)
    for n in range(5):
        ax[0, 1].plot(lw, lw ** 2 * W[n] * Cint / (2 * np.pi), lw=0.6, label=f"n={n + 1}")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("l"); ax[0, 1].legend(ncol=5, fontsize=7)
    ax[0, 1].set_ylabel(r"$l^2 W_n C_{33}/2\pi$  (integrand per $d\ln l$)")
    ax[0, 1].set_title("E_n integrand, bins 3-3, xi+/- best fit")
    # cumulative E_n(<lmax)/E_n(total)  (Simpson on the dense ln l grid)
    from scipy.integrate import cumulative_trapezoid
    ax[1, 0].set_title(r"cumulative $E_n(<l)$, bins 3-3 (normalised to total)")
    for n in range(5):
        f = lw ** 2 * W[n] * Cint / (2 * np.pi)
        cum = cumulative_trapezoid(f, np.log(lw), initial=0)
        tot = np.interp(C.L_MAX_KCAP, lw, cum)
        ax[1, 0].plot(lw, cum / tot, label=f"n={n + 1}")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_ylim(-0.6, 1.8); ax[1, 0].axhline(1, color="gray", lw=0.5)
    ax[1, 0].axvline(C.L_MAX_KCAP, color="gray", ls="--"); ax[1, 0].set_xlabel("l_max"); ax[1, 0].legend(ncol=5, fontsize=7)
    ax[1, 1].loglog([c[0] for c in conv], [max(c[1], 1e-6) for c in conv], "o-")
    ax[1, 1].axvline(C.L_MAX_KCAP, color="gray", ls="--"); ax[1, 1].set_xlabel("l_max")
    ax[1, 1].set_ylabel(r"max$_{pairs,n\leq5}$ |E$_n$(C$_l$;l$_{max}$) - E$_n$(xi)| / $\sigma$")
    ax[1, 1].set_title("l_max convergence (dashed: kcap l_max)")
    fig.tight_layout(); fig.savefig(OUT / "cosebis_02_Wn.png"); plt.close(fig)

    # ---------------------------------------------------------------- 03 routes
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    rel = (E_cl - E_xi) / np.abs(E_xi)
    for p in range(15):
        ax[0].plot(range(1, 6), np.abs(rel[p, :5]), "o-", ms=3, lw=0.7, label=LAB[p])
    ax[0].set_yscale("log"); ax[0].set_xlabel("mode n"); ax[0].set_ylabel("|E(Cl) - E(xi)| / |E(xi)|"); ax[0].set_xticks(range(1, 6))
    ax[0].set_title("relative difference, n=1..5, all 15 pairs"); ax[0].legend(ncol=3, fontsize=6)
    ds = np.abs(E_cl - E_xi) / sig
    for p in range(15):
        ax[1].semilogy(range(1, 21), np.maximum(ds[p], 1e-8), lw=0.6, color="C0", alpha=0.7)
    ax[1].semilogy(range(1, 21), ds.max(0), "k-", lw=2, label="max over pairs")
    ax[1].set_xlabel("mode n"); ax[1].set_ylabel(r"|E(Cl) - E(xi)| / $\sigma_{data}$"); ax[1].legend(fontsize=7)
    ax[1].set_title("absolute difference in units of data error")
    for p in (0, 7, 14):
        ax[2].plot(range(1, 21), E_cl[p] * 1e10, "o", ms=4, label=f"C_l route, bins {LAB[p]}")
        ax[2].plot(range(1, 21), E_xi[p] * 1e10, "kx", ms=6)
    ax[2].set_xlabel("mode n"); ax[2].set_ylabel(r"$E_n\ [10^{-10}]$"); ax[2].legend(fontsize=7)
    ax[2].set_title("E_n values: C_l route (o), xi route (x)")
    fig.tight_layout(); fig.savefig(OUT / "cosebis_03_routes.png"); plt.close(fig)

    # ---------------------------------------------------------------- 04 data vs model
    cm = LC.CosebisModel(m, 20)
    S, Dv, LP = xi_chain()
    th_mean = S.mean(0)
    mu_best = cm.mu(th_best).reshape(15, 20)
    mu_mean = cm.mu(th_mean).reshape(15, 20)
    res["posterior_mean_theta"] = dict(zip(NAMES, th_mean.tolist()))
    L20 = {am: LC.CosebisLikelihood(20, am) for am in (True, False)}
    dm = L20[True].d.reshape(15, 20)
    rng = np.random.default_rng(0)
    Ssub = S[rng.choice(len(S), size=min(200, len(S)), replace=False)]
    Es = np.array([cm.mu(s) for s in Ssub]).reshape(len(Ssub), 15, 20)
    lo, hi = np.percentile(Es, [16, 84], axis=0)

    def grid(nmax, fname, title, ms):
        fig, axs = plt.subplots(5, 5, figsize=(13, 9.5), sharex=True)
        n = np.arange(1, nmax + 1)
        for p, (j, i) in enumerate(PAIRS):
            a = axs[j, i]
            a.fill_between(n, lo[p, :nmax] * 1e10, hi[p, :nmax] * 1e10, color="C0", alpha=0.2, lw=0)
            a.plot(n, mu_best[p, :nmax] * 1e10, "C0-", lw=1.2, label="xi+/- best fit")
            a.plot(n, mu_mean[p, :nmax] * 1e10, "C3--", lw=1.0, label="xi+/- posterior-mean theta")
            a.errorbar(n, dm[p, :nmax] * 1e10, sig[p, :nmax] * 1e10, fmt="k.", ms=ms, lw=0.8, capsize=1.5, label="KV450 data (m-corr.)")
            a.set_title(f"bins {i + 1}-{j + 1}", fontsize=7, pad=2)
            a.set_xticks(n if nmax <= 5 else [1, 5, 10, 15, 20])
        for j in range(5):
            for i in range(5):
                if i > j:
                    axs[j, i].axis("off")
        for a in axs[-1]:
            a.set_xlabel("mode n")
        axs[0, 4].legend(*axs[0, 0].get_legend_handles_labels(), fontsize=8, loc="center") if False else None
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.2)], l + ["xi+/- posterior 68%"], loc="upper right", fontsize=9, bbox_to_anchor=(0.95, 0.9))
        fig.suptitle(title + r"   ($E_n$ in units of $10^{-10}$; error bars = sqrt(diag COVMAT))", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / fname); plt.close(fig)

    grid(5, "cosebis_04_data_vs_model.png", "KV450 COSEBIs, modes 1-5: data vs xi+/- model", 5)
    grid(20, "cosebis_04b_data_vs_model_20modes.png", "KV450 COSEBIs, modes 1-20: data vs xi+/- model", 3)
    # normalised residuals
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    Lw = L20[True]
    for k, (mu, nm) in enumerate(((mu_best, "xi+/- best-fit theta"), (mu_mean, "xi+/- posterior-mean theta"))):
        r = (dm - mu) / sig
        im = ax[k].imshow(r, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, origin="upper", extent=(0.5, 20.5, 14.5, -0.5))
        ax[k].set_yticks(range(15)); ax[k].set_yticklabels(LAB, fontsize=6); ax[k].set_xlabel("mode n")
        ax[k].set_title(f"(data - model)/sqrt(diag), {nm}", fontsize=8); ax[k].grid(False)
        plt.colorbar(im, ax=ax[k], fraction=0.04)
    w = solve_triangular(Lw._L, (Lw.d - mu_best.ravel()) / Lw.scale, lower=True)
    ax[2].hist(w, 24, density=True, alpha=0.8, label="whitened (L$^{-1}$r), 20 modes")
    xx = np.linspace(-4, 4, 100); ax[2].plot(xx, stats.norm.pdf(xx), "k", label="N(0,1)")
    ax[2].set_title(f"whitened residuals: mean {w.mean():.2f}, std {w.std():.2f}", fontsize=8); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "cosebis_04c_residuals.png"); plt.close(fig)
    res["whitened_residual_20modes_bestfit"] = {"mean": float(w.mean()), "std": float(w.std()), "max_abs": float(np.abs(w).max())}
    r1 = (dm - mu_best) / sig
    res["normalised_diag_residual_best"] = {"rms_n<=5": float(np.sqrt((r1[:, :5] ** 2).mean())), "rms_all20": float(np.sqrt((r1 ** 2).mean())),
                                           "max_abs": float(np.abs(r1).max())}

    # ---------------------------------------------------------------- 05 covariance
    cov = data.cov
    dsd = np.sqrt(np.diag(cov)); corr = cov / np.outer(dsd, dsd)
    sel5 = data.select(5)
    fig, ax4 = plt.subplots(1, 4, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1, 1, 0.04, 0.9]}); ax = [ax4[0], ax4[1], ax4[3]]
    im = ax[0].imshow(corr[np.ix_(sel5, sel5)], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    for b in range(1, 15):
        ax[0].axhline(b * 5 - 0.5, color="k", lw=0.3); ax[0].axvline(b * 5 - 0.5, color="k", lw=0.3)
    ax[0].set_xticks(np.arange(15) * 5 + 2); ax[0].set_xticklabels(LAB, rotation=90, fontsize=6)
    ax[0].set_yticks(np.arange(15) * 5 + 2); ax[0].set_yticklabels(LAB, fontsize=6)
    ax[0].set_title("correlation, 75 points (15 pairs x modes 1-5)"); ax[0].grid(False)
    ax[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    for b in range(1, 15):
        ax[1].axhline(b * 20 - 0.5, color="k", lw=0.3); ax[1].axvline(b * 20 - 0.5, color="k", lw=0.3)
    ax[1].set_xticks(np.arange(15) * 20 + 10); ax[1].set_xticklabels(LAB, rotation=90, fontsize=6)
    ax[1].set_yticks(np.arange(15) * 20 + 10); ax[1].set_yticklabels(LAB, fontsize=6)
    ax[1].set_title("correlation, 300 points (15 pairs x 20 modes)"); ax[1].grid(False)
    fig.colorbar(im, cax=ax4[2], label="correlation")
    # same-pair mode-mode correlation (within-pair blocks, averaged) and eigenvalues
    blk = np.mean([corr[p * 20:(p + 1) * 20, p * 20:(p + 1) * 20] for p in range(15)], axis=0)
    ev = np.linalg.eigvalsh(corr)[::-1]
    ax[2].semilogy(np.arange(1, 301), ev, ".", ms=3)
    ax[2].set_xlabel("index"); ax[2].set_ylabel("eigenvalue of correlation matrix"); ax[2].set_title("correlation-matrix spectrum (all > 0)")
    fig.tight_layout(); fig.savefig(OUT / "cosebis_05_covariance.png"); plt.close(fig)
    off = corr[sel5][:, sel5] - np.eye(75)
    res["covariance"] = {"min_eig_cov": float(np.linalg.eigvalsh(cov).min()), "min_eig_corr": float(ev.min()),
                         "max_offdiag_corr_n5": float(np.abs(off).max()),
                         "mean_within_pair_mode_corr_offdiag_n<=5": float(np.mean(np.abs(blk[:5, :5] - np.eye(5)).sum() / 20)),
                         "cond_n5": LC.CosebisLikelihood(5).condition_number(), "cond_n20": LC.CosebisLikelihood(20).condition_number()}

    # ---------------------------------------------------------------- chi2 at xi+/- best fit
    chi = {}
    for N in (5, 20):
        for am in (True, False):
            L = LC.CosebisLikelihood(N, am)
            mu = mu_best[:, :N].ravel()
            c = L.chi2(mu)
            chi[f"n{N}_apply_m_{am}"] = {"chi2": c, "ndata": L.n_data, "chi2_per_ndata": c / L.n_data,
                                         "p_value_dof=ndata": pval(c, L.n_data), "null_chi2": L.null_chi2(),
                                         "chi2_diag_only": float(np.sum((L.d - mu) ** 2 / np.diag(L.cov)))}
    res["chi2_at_xi_MAP"] = chi
    res["xi_pm_MAP_chi2_on_195"] = best["chi2"]
    # chi2 at posterior-mean theta
    res["chi2_at_xi_posterior_mean_theta"] = {f"n{N}": LC.CosebisLikelihood(N, True).chi2(mu_mean[:, :N].ravel()) for N in (5, 20)}
    # chi2 distribution over xi posterior draws
    L5, L20m = LC.CosebisLikelihood(5), L20[True]
    c5 = np.array([L5.chi2(e[:, :5].ravel()) for e in Es]); c20 = np.array([L20m.chi2(e.ravel()) for e in Es])
    res["chi2_posterior_samples"] = {"n_samples": int(len(Es)),
                                     "n5": {"min": float(c5.min()), "median": float(np.median(c5)), "max": float(c5.max())},
                                     "n20": {"min": float(c20.min()), "median": float(np.median(c20)), "max": float(c20.max())}}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(c5, 25, color="C0", alpha=0.8); ax[0].axvline(75, color="k", ls="--", label="N=75"); ax[0].set_title("chi2 (5 modes) over xi+/- posterior draws"); ax[0].legend()
    ax[1].hist(c20, 25, color="C1", alpha=0.8); ax[1].axvline(300, color="k", ls="--", label="N=300"); ax[1].set_title("chi2 (20 modes)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "cosebis_07_chi2_samples.png"); plt.close(fig)

    # NZ_SOURCE table of the FITS file vs the n(z) used by the pipeline (histogram midpoints vs interpolated pipeline n(z))
    a = data.nz_source
    wbin = a[:, 2] - a[:, 0]
    nzp = m.nz
    nzc = {}
    for b in range(5):
        f = a[:, 3 + b] / np.sum(a[:, 3 + b] * wbin)
        pipe = np.interp(a[:, 1], nzp.z, nzp.n[b])
        nzc[f"bin{b + 1}"] = {"mean_z_file": float(np.sum(f * a[:, 1] * wbin)), "mean_z_pipeline": float(nzp.mean_z()[b]),
                              "max_abs_diff_over_peak": float(np.abs(f - pipe).max() / pipe.max())}
    res["nz_source_file_vs_pipeline"] = nzc

    # B-mode statement
    res["B_mode_data"] = "not available (En table only: no Bn data); B_n theory = 0 placeholder; only the pure-E B_n leakage test of the filters is possible"
    res["timing_seconds"] = time.time() - t00
    json.dump(res, open(RUN / "validation_numbers_main.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return res


# ======================================================================================== posterior stage
def rhat_split(x):
    """x (nchain, nsamp) -> split R-hat (Gelman-Rubin on half chains)."""
    n = x.shape[1] // 2
    y = np.concatenate([x[:, :n], x[:, n:2 * n]])
    W = y.var(axis=1, ddof=1).mean(); B = n * y.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def stage_post(nm=5, tag="n5", burn_extra=0):
    from kidspipe.inference.mcmc import split_rhat, effective_sample_size
    from kidspipe import params
    res = {}
    fs = sorted(glob.glob(str(RUN / f"chain_{tag}_c*.npz")))
    if not fs:
        print("no COSEBIs chains found"); return
    ch = [np.load(f) for f in fs]
    n_burn = 2000
    # samples as returned by blocked_metropolis are post burn-in (burn returned separately)
    X = [c["samples"] for c in ch]
    ncyc = min(len(x) for x in X)
    X = np.array([x[:ncyc] for x in X])                              # (nch, ncyc, 14)
    D = np.array([c["derived5"][:ncyc // 5] for c in ch])                        # every 5th sample: sigma8, Om, S8
    s8 = D[:, :, 2]
    names = ["sigma8", "Omega_m", "S8"]
    out = {"n_chains": len(fs), "cycles_per_chain": int(ncyc), "accept": [np.atleast_1d(c["accept"]).tolist() for c in ch],
           "chain_seconds": [float(c["time"]) for c in ch]}
    Dflat = D.reshape(-1, 3)
    out["rhat_derived_thin5"] = {n: rhat_split(D[:, :, k]) for k, n in enumerate(names)}
    out["rhat_theta"] = dict(zip(NAMES, split_rhat(X[:, ::5, :].transpose(0, 1, 2)).tolist())) if True else None
    out["note_ess"] = "ESS not quoted: 500 post-burn cycles per chain, S8 is thinned by 5"
    qs = np.percentile(s8, [16, 50, 84])
    out["S8"] = {"mean": float(s8.mean()), "std": float(s8.std()), "median": float(qs[1]), "p16": float(qs[0]), "p84": float(qs[2]),
                 "chain_means": s8.mean(1).tolist()}
    out["sigma8"] = {"mean": float(D[:, :, 0].mean()), "std": float(D[:, :, 0].std())}
    out["Omega_m"] = {"mean": float(D[:, :, 1].mean()), "std": float(D[:, :, 1].std())}
    # chi2 stats of the COSEBIs chain
    lp = np.array([c["log_post"][:ncyc] for c in ch])
    out["max_log_post"] = float(lp.max())
    # xi chain
    S, Dxi, LPxi = xi_chain(step=10)
    s8x = Dxi[:, 2]
    out["xi_S8"] = {"mean": float(s8x.mean()), "std": float(s8x.std()), "median": float(np.median(s8x)),
                    "p16": float(np.percentile(s8x, 16)), "p84": float(np.percentile(s8x, 84))}
    out["delta_S8_cosebis_minus_xi_mean"] = out["S8"]["mean"] - out["xi_S8"]["mean"]
    out["delta_S8_in_units_of_xi_std"] = out["delta_S8_cosebis_minus_xi_mean"] / out["xi_S8"]["std"]
    # emulator box: fraction of samples outside (omch2, ombh2, h)
    from kidspipe.theory import emulator as em
    try:
        e = em.TransferEmulator()
        bx = e.box
        Xf = X.reshape(-1, 14)
        inside = np.all((Xf[:, :3] >= bx[:, 0]) & (Xf[:, :3] <= bx[:, 1]), axis=1)
        out["frac_samples_outside_emulator_box"] = float(1 - inside.mean())
        Sf = S
        out["frac_xi_samples_outside_emulator_box"] = float(1 - np.all((Sf[:, :3] >= bx[:, 0]) & (Sf[:, :3] <= bx[:, 1]), axis=1).mean())
    except Exception as ex:                                           # noqa: BLE001
        out["frac_samples_outside_emulator_box"] = f"not computed: {ex}"

    # chi2 along the chain (thinned) and best chi2 sample
    m = build_model(linear_backend="emulator")
    cm = LC.CosebisModel(m, nm)
    like = LC.CosebisLikelihood(nm, True)
    Xt = X[:, ::100, :].reshape(-1, 14)
    chi2c = np.array([like.chi2(cm.mu(t)) for t in Xt])
    ib = int(np.argmin(chi2c))
    out["chi2_chain_thin100"] = {"n": int(len(chi2c)), "min": float(chi2c.min()), "median": float(np.median(chi2c)), "ndata": like.n_data,
                                 "min_p_value_dof=ndata": pval(chi2c.min(), like.n_data)}
    # chi2 minimum over all samples (cheaper proxy: highest log-post sample, flat priors => ~min chi2 up to prior)
    flat = X.reshape(-1, 14)
    ib2 = int(np.argmax(lp.reshape(-1)))
    out["chi2_at_max_logpost_sample"] = float(like.chi2(cm.mu(flat[ib2])))
    out["theta_at_max_logpost_sample"] = dict(zip(NAMES, flat[ib2].tolist()))
    # xi+/- chi2 on this theta is not computed (needs the xi likelihood); report only COSEBIs.
    # importance reweighting of xi chain
    rw = reweight(nm, tag, m, cm, like)
    out["reweight"] = rw["summary"]

    # ---------------- plot 06
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(0.55, 0.95, 60)
    ax[0].hist(s8x, bins, density=True, histtype="step", lw=2, color="k", label=f"xi+/-: {out['xi_S8']['mean']:.4f} $\\pm$ {out['xi_S8']['std']:.4f}")
    ax[0].hist(s8.ravel(), bins, density=True, histtype="step", lw=2, color="C3",
               label=f"COSEBIs (MCMC, {nm} modes): {out['S8']['mean']:.4f} $\\pm$ {out['S8']['std']:.4f}")
    wts = rw["w"]; ax[0].hist(rw["s8"], bins, weights=wts / wts.sum() / (bins[1] - bins[0]), histtype="step", lw=1.5, ls="--", color="C0",
                              label=f"xi+/- chain reweighted (ESS {rw['summary']['ess']:.0f}/{rw['summary']['n']}): {rw['summary']['S8_mean']:.4f} $\\pm$ {rw['summary']['S8_std']:.4f}")
    ax[0].set_xlabel(r"$S_8=\sigma_8(\Omega_m/0.3)^{0.5}$"); ax[0].set_ylabel("posterior density"); ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title("S8 marginal"); ax[0].set_xlim(0.55, 0.95)
    ax[1].scatter(Dxi[::4, 1], Dxi[::4, 0], s=1, c="0.6", alpha=0.4, label="xi+/-")
    ax[1].scatter(Dflat[::3, 1], Dflat[::3, 0], s=1, c="C3", alpha=0.3, label="COSEBIs")
    ax[1].set_xlabel(r"$\Omega_m$"); ax[1].set_ylabel(r"$\sigma_8$"); ax[1].set_xlim(0.05, 0.7); ax[1].set_ylim(0.4, 1.4)
    ax[1].set_title("sigma8 - Omega_m (prior-bounded, banana)"); ax[1].legend(markerscale=6, fontsize=8)
    for c in range(len(fs)):
        ax[2].plot(np.arange(s8.shape[1]) * 5, s8[c], lw=0.4, alpha=0.7)
    ax[2].set_xlabel("cycle (post burn-in)"); ax[2].set_ylabel("S8"); ax[2].set_title(f"S8 traces (thin 5), {len(fs)} chains, split R-hat = {out['rhat_derived_thin5']['S8']:.3f}")
    fig.tight_layout(); fig.savefig(OUT / "cosebis_06_posterior.png"); plt.close(fig)

    res["cosebis_posterior"] = out
    json.dump(res, open(RUN / f"validation_numbers_post_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


def reweight(nm, tag, m, cm, like, n=4000):
    """Importance weights w = L_cosebis(theta)/L_xi(theta) for a thinned xi+/- chain (prior identical)."""
    f = RUN / f"reweight_{tag}.npz"
    S, Dxi, LPxi = xi_chain(step=max(1, 272840 // n))
    if f.exists():
        z = np.load(f)
        if len(z["lnw"]) == len(S):
            lnw, lcos, lxi = z["lnw"], z["lcos"], z["lxi"]
        else:
            f.unlink()
    if not f.exists():
        from kidspipe.api import build_likelihood
        gl = build_likelihood()
        lcos = np.array([like.log_like(cm.mu(t)) for t in S])
        lxi = np.array([gl.log_like(m.mu(t), A_c=t[8], delta_c=t[7]) for t in S])
        lnw = lcos - lxi
        np.savez(f, lnw=lnw, lcos=lcos, lxi=lxi)
    w = np.exp(lnw - lnw.max())
    ess = w.sum() ** 2 / (w ** 2).sum()
    s8 = Dxi[:, 2]
    mean = np.sum(w * s8) / w.sum(); sd = np.sqrt(np.sum(w * (s8 - mean) ** 2) / w.sum())
    return {"w": w, "s8": s8, "summary": {"n": int(len(S)), "ess": float(ess), "S8_mean": float(mean), "S8_std": float(sd),
                                         "lnw_std": float(lnw.std()), "corr_lcos_lxi": float(np.corrcoef(lcos, lxi)[0, 1]),
                                         "mean_lcos": float(lcos.mean()), "mean_lxi": float(lxi.mean())}}


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    if STAGE in ("main", "all"):
        stage_main()
    if STAGE in ("post", "all"):
        stage_post()
