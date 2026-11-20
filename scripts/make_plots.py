"""Diagnostic plots for kidspipe -> plots/*.png  (run from anywhere: python3 scripts/make_plots.py).

Inputs are saved artefacts only (chains, MAP, profile, benchmark and validation JSON) plus a few model evaluations.
  01 data vs best-fit model (15 tomographic pairs)       02 HMcode P_nl vs stored reference + CAMB 'mead'
  03 posterior: S8, Omega_m, sigma8 and nuisance          04 convergence: S8/omch2 traces and per-chain means
  05 chi2 profiles (ln A_s and S8)                        06 emulator-vs-CAMB accuracy
  07 timing per move type                                 08 chi2 split by tomographic pair
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)
RUN = ROOT / "runs" / "kv450_final"
PUB_S8 = (0.737, 0.036, 0.040)   # external published KV450 value (not in the repo)


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def load_chains(tags=("prod1", "prod2"), burn=0.05):
    S, D, chain_id = [], [], []
    n = 0
    for t in tags:
        for f in sorted((RUN / t).glob("prod_c*.npz")):
            z = np.load(f, allow_pickle=True)
            k = int(burn * len(z["samples"]))
            S.append(z["samples"][k:]); D.append(z["derived"][k:]); chain_id.append(np.full(len(z["samples"]) - k, n)); n += 1
    return np.vstack(S), np.vstack(D), np.concatenate(chain_id)


# ----------------------------------------------------------------------------------------------- 01 data vs model
def plot_data_model():
    from kidspipe.theory.model import CosmicShearModel
    from kidspipe.likelihood.gauss import GaussianLikelihood
    from kidspipe.likelihood.data import PAIRS, split_vector
    from kidspipe.params import PARAM_NAMES
    best = json.load(open(RUN / "map_best.json"))
    theta = np.array(best["theta"]); names = best["names"]
    assert list(names) == list(PARAM_NAMES)
    lik = GaussianLikelihood()
    model = CosmicShearModel(linear_backend="emulator")
    mu = lik.add_systematics(model.mu(theta), A_c=theta[names.index("A_c")], delta_c=theta[names.index("delta_c")])
    d, err = lik.d, np.sqrt(np.diag(lik.cov))
    dp, dm = split_vector(d); ep, em = split_vector(err); mp, mm = split_vector(mu)
    tp, tm = lik.data.theta_plus, lik.data.theta_minus
    fig, axs = plt.subplots(3, 5, figsize=(17, 9), sharex=True)
    for p, (j, i) in enumerate(PAIRS):
        ax = axs.flat[p]
        ax.errorbar(tp, tp * dp[p] * 1e4, tp * ep[p] * 1e4, fmt="o", ms=3.5, color="C0", label=r"$\xi_+$ data")
        ax.plot(tp, tp * mp[p] * 1e4, "-", color="C0")
        ax.errorbar(tm, tm * dm[p] * 1e4, tm * em[p] * 1e4, fmt="s", ms=3.5, color="C3", label=r"$\xi_-$ data")
        ax.plot(tm, tm * mm[p] * 1e4, "-", color="C3")
        ax.set_xscale("log"); ax.set_title(f"bins {j + 1}-{i + 1}", fontsize=10)
        if p % 5 == 0: ax.set_ylabel(r"$\theta\,\xi_\pm\ [10^{-4}\,$arcmin]")
        if p >= 10: ax.set_xlabel(r"$\theta$ [arcmin]")
    axs.flat[0].legend(fontsize=8)
    chi2 = lik.chi2(model.mu(theta), A_c=theta[names.index("A_c")], delta_c=theta[names.index("delta_c")])
    fig.suptitle(f"KV450 cosmic shear: data (m-corrected, 195 pts) vs best fit of kidspipe, chi2 = {chi2:.1f}")
    save(fig, "01_data_vs_model.png")
    r = (d - mu) / err
    return r


# ----------------------------------------------------------------------------------------------- 02 P_nl check
def plot_pnl():
    from kidspipe.theory.linear import LinearBackend
    from kidspipe.theory.nonlinear import get_pnl
    kref = np.loadtxt(ROOT / "data/validation/k_h.txt"); zref = np.loadtxt(ROOT / "data/validation/z.txt")
    pref = np.loadtxt(ROOT / "data/validation/pk_mead.txt")           # (nz, nk), z ascending
    idx = [0, 12, 25, 50, 75, 99]
    zsub = zref[idx]
    lin = LinearBackend(z_grid=zsub).compute(0.1181838, 0.02228089, 0.679565, 0.9679701, np.log(2.170229e-9 * 1e10))
    hm = get_pnl(lin, 2.0, backend="hmcode"); cb = get_pnl(lin, 2.0, backend="camb")
    m = (kref > lin.k_h[0]) & (kref < lin.k_h[-1])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, i in enumerate(idx):
        ref = pref[i][m]
        for ax, P, lab in ((axs[0], hm, "HMcode (kidspipe)"), (axs[1], cb, "CAMB mead 1.5.8")):
            ax.semilogx(kref[m], np.exp(np.interp(np.log(kref[m]), np.log(lin.k_h), np.log(P[n]))) / ref - 1,
                        color=f"C{n}", label=f"z={zsub[n]:.2f}")
            ax.set_title(f"{lab} / stored reference array - 1"); ax.set_xlabel("k [h/Mpc]")
    for ax in axs: ax.axhline(0, color="k", lw=.5); ax.axhspan(-.01, .01, color="gray", alpha=.15); ax.set_ylim(-.02, .02)
    axs[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Non-linear P(k,z), A_baryon=2, eta0=0.74 (grey band = repo tolerance 1%)")
    save(fig, "02_pnl_vs_reference.png")


# ----------------------------------------------------------------------------------------------- 03 posterior
def plot_posterior():
    S, D, cid = load_chains()
    sig8, om, s8 = D[:, 0], D[:, 1], D[:, 2]
    from kidspipe.params import PARAM_NAMES
    col = {n: S[:, i] for i, n in enumerate(PARAM_NAMES)}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), gridspec_kw={"hspace": 0.4, "wspace": 0.25})
    ax = axs[0, 0]
    ax.hist(s8, bins=80, density=True, color="C0", alpha=.7, label=f"kidspipe  {s8.mean():.3f}$\\pm${s8.std():.3f}")
    ax.axvspan(PUB_S8[0] - PUB_S8[1], PUB_S8[0] + PUB_S8[2], color="C1", alpha=.3, label="published KV450 (external)")
    ax.axvline(PUB_S8[0], color="C1"); ax.set_xlabel(r"$S_8$"); ax.legend(fontsize=8); ax.set_title(r"$S_8$ marginal")
    ax = axs[0, 1]; h = ax.hist2d(om, s8, bins=90, cmap="viridis", cmin=1, range=[[0, 0.7], [0.55, 0.95]])
    ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$S_8$"); ax.set_title(r"$\Omega_m$ - $S_8$")
    ax = axs[0, 2]; ax.hist2d(sig8, om, bins=90, cmap="viridis", cmin=1, range=[[0.4, 1.5], [0.05, 0.7]])
    ax.set_xlabel(r"$\sigma_8$"); ax.set_ylabel(r"$\Omega_m$"); ax.set_title(r"$\sigma_8$ - $\Omega_m$ (prior-box shaped)")
    ax = axs[1, 0]; ax.hist(col["A_IA"], bins=80, density=True, alpha=.7); ax.set_xlabel(r"$A_{IA}$")
    ax.set_title(f"A_IA = {col['A_IA'].mean():.2f}$\\pm${col['A_IA'].std():.2f}   (published 0.98$\\pm$0.69, external)")
    ax = axs[1, 1]; ax.hist(col["A_baryon"], bins=60, density=True, alpha=.7); ax.set_xlabel(r"$A_{baryon}$")
    ax.set_title(f"A_baryon = {col['A_baryon'].mean():.2f}$\\pm${col['A_baryon'].std():.2f}   (bounds 2.0-3.13)")
    ax = axs[1, 2]
    sig = [0.039, 0.023, 0.026, 0.012, 0.011]
    for k in range(5):
        x = col[f"dz{k + 1}"] / sig[k]
        ax.hist(x, bins=60, density=True, histtype="step", color=f"C{k}", label=f"dz{k + 1}")
    g = np.linspace(-4, 4, 200); ax.plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), "k--", label="prior N(0,1)")
    ax.set_xlabel(r"$\delta z_i/\sigma_{prior}$"); ax.legend(fontsize=7); ax.set_title("photo-z shifts: prior-dominated")
    fig.suptitle(f"KV450 posterior, {len(S):,} samples from 14 chains (MCMC, emulator + HMcode)")
    save(fig, "03_posterior.png")


# ----------------------------------------------------------------------------------------------- 04 convergence
def plot_convergence():
    S, D, cid = load_chains(burn=0.0)
    from kidspipe.params import PARAM_NAMES
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (y, lab) in zip(axs[0], ((D[:, 2], r"$S_8$"), (S[:, PARAM_NAMES.index("omch2")], "omch2"))):
        for c in np.unique(cid)[:14]:
            v = y[cid == c]; t = np.arange(len(v))
            ax.plot(t[::40], v[::40], lw=.5, alpha=.7)
        ax.set_xlabel("cycle"); ax.set_ylabel(lab)
    for ax, (y, lab) in zip(axs[1], ((D[:, 2], r"$S_8$"), (S[:, PARAM_NAMES.index("ln_1e10_A_s")], r"$\ln 10^{10}A_s$"))):
        for c in np.unique(cid):
            v = y[cid == c]; ax.plot(np.cumsum(v)[::100] / np.arange(1, len(v) + 1)[::100], lw=.8)
        ax.set_xlabel("sample index / 100"); ax.set_ylabel(f"running mean {lab}")
        ax.set_xscale("log")
    fig.suptitle("Chain traces and running means (14 chains; target split R-hat < 1.02: max reached 1.017)")
    save(fig, "04_convergence.png")


# ----------------------------------------------------------------------------------------------- 05 profiles
def plot_profiles():
    P = json.load(open(RUN / "profile.json"))
    a = sorted((p for p in P if p["kind"] == "lnAs"), key=lambda p: p["target"])
    s = sorted((p for p in P if p["kind"].lower().startswith("s8")), key=lambda p: p["target"])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    axs[0].plot([p["target"] for p in a], [p["chi2_spec"] for p in a], "o-"); axs[0].set_xlabel(r"$\ln 10^{10}A_s$")
    axs[0].set_ylabel(r"$\chi^2_{min}$"); axs[0].set_title(f"chi2 changes by only {max(p['chi2_spec'] for p in a) - min(p['chi2_spec'] for p in a):.2f} over a factor 20 in A_s")
    axs[1].plot([p["omega_m"] for p in a], [p["S8"] for p in a], "o-"); axs[1].set_xlabel(r"$\Omega_m$"); axs[1].set_ylabel(r"$S_8$")
    axs[1].set_title("best-fit valley: low Omega_m, high sigma8")
    if s:
        axs[2].plot([p["target"] for p in s], [p["chi2_spec"] for p in s], "o-"); axs[2].set_xlabel(r"$S_8$")
        axs[2].set_ylabel(r"$\chi^2_{min}$"); axs[2].set_title(r"profile in $S_8$"); axs[2].set_ylim(top=min(max(p["chi2_spec"] for p in s), 215))
    save(fig, "05_profiles.png")


# ----------------------------------------------------------------------------------------------- 06 emulator accuracy
def plot_emulator():
    v = json.load(open(ROOT / "runs/validate_emulator_pipeline_spec.json"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].hist(v["dchi2"], bins=25); axs[0].set_xlabel(r"$\Delta\chi^2$ (emulator - exact CAMB)")
    axs[0].set_title(f"{v['n']} held-out points, max |d chi2| = {v['max_abs_dchi2']:.4f}")
    axs[1].scatter(v["chi2_camb"], np.abs(v["dchi2"]), s=14); axs[1].set_yscale("log"); axs[1].set_xscale("log")
    axs[1].axhline(0.05, color="r", ls="--", label="acceptance 0.05"); axs[1].set_xlabel(r"$\chi^2$ (exact CAMB)"); axs[1].set_ylabel(r"$|\Delta\chi^2|$")
    axs[1].legend()
    save(fig, "06_emulator_accuracy.png")


# ----------------------------------------------------------------------------------------------- 07 timing
def plot_timing():
    b = json.load(open(ROOT / "runs/bench_t4.json"))
    cfg = [k for k in ("i", "ii", "iii", "iv") if k in b["configs"]]
    moves = [("slow block", "cosmology move\n(omch2/ombh2/h + ...)"), ("lnAs/A_baryon", "medium move\n(lnAs/A_baryon)"), ("A_IA/dz", "nuisance move\n(A_IA, dz)")]
    fig, ax = plt.subplots(figsize=(10, 4.8)); w = 0.2
    for n, k in enumerate(cfg):
        vals = [b["configs"][k]["moves"][m]["ms_median"] for m, _ in moves]
        bars = ax.bar(np.arange(len(moves)) + n * w, vals, w, label=f"({k}) {b['configs'][k]['label']}")
    ax.set_yscale("log"); ax.set_ylabel("ms per evaluation (median, 4 threads)")
    ax.set_xticks(np.arange(len(moves)) + 1.5 * w); ax.set_xticklabels([l for _, l in moves]); ax.legend(fontsize=8)
    ax.set_title("Cost per likelihood evaluation by type of parameter move (bench_t4.json)")
    save(fig, "07_timing.png")


# ----------------------------------------------------------------------------------------------- 08 chi2 split
def plot_residuals(r):
    from kidspipe.likelihood.data import PAIRS
    sp = json.load(open(RUN / "chi2_split.json"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    axs[0].hist(r, bins=30, density=True, alpha=.7); g = np.linspace(-4, 4, 200)
    axs[0].plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), "k--"); axs[0].set_xlabel("(data - model)/sigma, diagonal errors only")
    axs[0].set_title("normalised residuals (correlated, so only indicative)")
    axs[1].imshow(r.reshape(15, 13), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    axs[1].set_yticks(range(15)); axs[1].set_yticklabels([f"{j + 1}-{i + 1}" for j, i in PAIRS], fontsize=7)
    axs[1].set_xticks(range(13)); axs[1].set_xticklabels([f"+{k}" for k in range(7)] + [f"-{k + 3}" for k in range(6)], fontsize=7)
    axs[1].set_title("residual map: pair x (xi+ bins | xi- bins)")
    save(fig, "08_residuals.png")


if __name__ == "__main__":
    r = plot_data_model()
    plot_residuals(r)
    plot_pnl(); plot_posterior(); plot_convergence(); plot_profiles(); plot_emulator(); plot_timing()
