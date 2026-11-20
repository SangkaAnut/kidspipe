"""Adopted DR5 measurement gates A1 (A1b-noise) and A3 (max |Delta|/sigma).

A1 = A1b-noise (adopted 2026-10-29, PI sign-off; evidence in runs/dr5_measure/a1_salvage/A1_SALVAGE.md).
  Compares the measured RAW (m-uncorrected) xi+/- with the official fine ascii (1000 log bins, 2-300')
  after one scale per pair A_p (ratio of npairs_weighted-weighted sums; NOT a least-squares slope,
  which noise in the ascii dilutes).  The official ascii is a blinded realisation of the same sample
  (m-uncorrected, amplitude x~0.85), so only shape and the pair-to-pair consistency of A_p are tested:
    (i)  z = (their - A_p ascii)/sigma_ascii; the rms of 20-bin boxcar averages of z, divided by the
         white-noise expectation rms(z)/sqrt(20), must be <= 1.03 on all 42 curves (1.0 until 2026-10-29).
         CAVEAT (found 2026-10-29 after adoption): for a truly WHITE residual this ratio scatters around 1
         (max over 42 curves: median 1.20, 99th pct 1.33), so 1.0 rejects white residuals.  It passes
         the real comparison only because two estimates from the SAME galaxies differ by anti-correlated
         jitter (lag-1 ~ -0.43 -> ratio ~0.4-0.8).  The gate is therefore valid for same-sample
         comparisons only.  UPDATE 2026-10-29 (standardised/STANDARDISED.md): a null that also REFITS A_p (as the gate does)
         gives 99th pct of max ratio 1.03 (not the 0.81 of a no-refit null); const 1.0 then has ~9 % false-fail, 0.81 ~79 %.
         So the 1 %-size same-sample constant is 1.03 (1.33 is the WHITE-residual 99th pct, too loose here).
         Real max 0.787 has null pte 0.82.  PI decision 2026-10-29: threshold moved 1.0 -> 1.03 (1 % size).
    (ii) ln A_p must be consistent with one global constant: chi2 (21 pairs, dof 20) with the bootstrap
         fractional errors of A_p, pte > 0.01, separately for xi+ and xi-.
A3 = their xi rebinned to the 9 official bins vs KiDS_Legacy_xipm.fits: PASS iff max |Delta|/sigma < 0.1
  (sigma = sqrt(diag) of the official covariance).  chi2/dof < 1 is not used: against the official
  covariance it cannot detect a 10 % amplitude error.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2dist

A1B_THRESH = {"boxcar_over_white_max": 1.03, "const_pte_min": 0.01}   # 1.03 = 99th pct of same-sample refit null
A3_MAX_ABS_OVER_SIGMA = 0.1
BOXCAR = 20


def ratio_of_sums(t, a, g):
    """Per-pair scale sum(g t)/sum(g a) over the last axis."""
    return (g * t).sum(-1) / (g * a).sum(-1)


def a1b_noise(tp_raw, tm_raw, asc, sd_frac, thresh=A1B_THRESH):
    """A1b-noise gate.  tp_raw, tm_raw: (21, 1000) raw xi+/-; asc: official ascii dict with keys
    xip, xim, sigma_xip, sigma_xim, npairs_weighted (each (21, 1000)); sd_frac: {'xip','xim'} ->
    (21,) bootstrap fractional errors of A_p.  Returns per-statistic diagnostics and 'pass'."""
    w = asc["npairs_weighted"]
    out, ratios, pt = {}, [], {}
    for key, tr in (("xip", tp_raw), ("xim", tm_raw)):
        A = ratio_of_sums(tr, asc[key], w)
        z = (tr - A[:, None] * asc[key]) / asc["sigma_" + key]
        n = z.shape[1] // BOXCAR * BOXCAR
        zb = z[:, :n].reshape(z.shape[0], -1, BOXCAR).mean(-1)
        rat = zb.std(1) / (z.std(1) / np.sqrt(BOXCAR))
        ratios.append(rat)
        sd = np.asarray(sd_frac[key], float)
        wg = 1 / sd ** 2
        lA = np.log(np.abs(A))
        lc = np.sum(lA * wg) / wg.sum()
        ch = float(np.sum(((lA - lc) / sd) ** 2))
        pt[key] = float(chi2dist.sf(ch, lA.size - 1))
        out[key] = {"max_boxcar_over_white": float(rat.max()), "median_boxcar_over_white": float(np.median(rat)),
                    "const_chi2": ch, "const_pte": pt[key], "C": float(np.exp(lc))}
    out["thresholds"] = dict(thresh)
    out["pass"] = bool(max(r.max() for r in ratios) <= thresh["boxcar_over_white_max"]
                       and min(pt.values()) > thresh["const_pte_min"])
    return out


def a3_pass(max_abs_over_sigma: float) -> bool:
    """A3 gate: max |Delta|/sigma < 0.1."""
    return bool(max_abs_over_sigma < A3_MAX_ABS_OVER_SIGMA)


# ---------------------------------------------------------------------------------------------------
# Per-curve STANDARDISED version of criterion (i)   (OPTION added 2026-10-29; the adopted gate above is unchanged)
#
# Each of the 42 curves is compared with ITS OWN null.  The null is a heteroscedastic MA(2) jitter model whose
# lag-1/lag-2 autocorrelation and rms profile are measured on the curve itself, pushed through the SAME pipeline as
# the data (their = A asc + sigma z_jitter, per-pair scale A_p REFIT, 20-bin boxcar ratio).  Because the refit adds a
# smooth component to z (its size matches the independent bootstrap error of A_p), the per-curve null ratio is NOT
# the white/MA(2) identity value and differs a lot from curve to curve (xi- 1-1: median 0.94; typical 0.35).
#   D_c = ratio_c^2 - r_pred_c^2   (excess boxcar variance over the lag-1/2 prediction; r_pred^2 = 1 + 2 sum_k rho_k W_k)
#   p_c = P_null,c(D >= D_c)       (per-curve ECDF from the null simulation)
#   combined: min-p (single bad curve), Fisher (weak coherent excess), Stouffer.   Gate: statistic <= calibrated threshold.
# Lags are estimated from a HIGH-PASS filtered z (method 'hpM': z minus its M-bin running mean, filter inverted
# exactly) so that smooth shape errors do not contaminate their own null; 'plain' uses z itself (contaminated).
# Evidence: runs/dr5_measure/a1_salvage/threshold_calibration/standardised/STANDARDISED.md
# ---------------------------------------------------------------------------------------------------
PROF_WIN = 51
STD_DEFAULT = {"method": "hp15", "stat": "minp", "size": 0.01, "const_pte_min": 0.01}


def curves_z(tp_raw, tm_raw, asc):
    """Residuals z (..., 42, N) (rows 0-20 xi+, 21-41 xi-; leading axes allowed) and per-pair scales {'xip','xim'}."""
    w = asc["npairs_weighted"]
    zs, As = [], {}
    for key, tr in (("xip", tp_raw), ("xim", tm_raw)):
        Ak = ratio_of_sums(tr, asc[key], w)
        As[key] = Ak
        zs.append((tr - Ak[..., None] * asc[key]) / asc["sigma_" + key])
    return np.concatenate(zs, -2), As


def boxcar_ratio(z, B=BOXCAR):
    n = z.shape[-1] // B * B
    zb = z[..., :n].reshape(z.shape[:-1] + (-1, B)).mean(-1)
    return zb.std(-1) / (z.std(-1) / np.sqrt(B))


def running_rms(x, win=PROF_WIN):
    """Running rms over `win` bins, edge-corrected (divides by the actual number of bins in the window)."""
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x ** 2, -1)], -1)
    n = x.shape[-1]
    i = np.arange(n)
    lo, hi = np.clip(i - win // 2, 0, n), np.clip(i + win // 2 + 1, 0, n)
    return np.sqrt((c[..., hi] - c[..., lo]) / (hi - lo))


def _acf12(u):
    u = u - u.mean(-1, keepdims=True)
    v = (u * u).sum(-1)
    return np.stack([(u[..., :-k] * u[..., k:]).sum(-1) / v for k in (1, 2)], -1)


def _jitter_rho_hp(z, m, win):
    n = z.shape[-1]
    h = -np.ones(m) / m
    h[m // 2] += 1.0
    a = np.correlate(h, h, "full")                    # filter autocorrelation a_l, l = -(m-1)..(m-1)
    A = lambda l: a[l + m - 1] if abs(l) <= m - 1 else 0.0
    M = np.zeros((3, 3))                              # G_x(k) = sum_j M[k, j] gamma_j  (gamma_j = 0 for j > 2)
    for k in range(3):
        for j in range(3):
            M[k, j] = A(k) if j == 0 else A(k - j) + A(k + j)
    from numpy.lib.stride_tricks import sliding_window_view
    box = sliding_window_view(z, m, axis=-1).mean(-1)                 # centred at t = m//2 .. n-m//2-1
    x = z[..., m // 2: n - m // 2] - box
    px = running_rms(x, win)
    u = x - x.mean(-1, keepdims=True)
    G = np.stack([(u * u).sum(-1) / u.shape[-1]] + [(u[..., :-k] * u[..., k:]).sum(-1) / u.shape[-1] for k in (1, 2)], -1)
    g = np.linalg.solve(M, G[..., None])[..., 0]                      # gamma_0..2 of z
    prof = np.concatenate([np.repeat(px[..., :1], m // 2, -1), px, np.repeat(px[..., -1:], m // 2, -1)], -1)
    return g[..., 1:] / g[..., :1], prof


def jitter_rho(z, method="hp15", win=PROF_WIN):
    """Lag-1, lag-2 autocorrelation of the residual (variance-weighted like the boxcar ratio itself) and the rms
    profile.  'plain': from z.  'hpM' (e.g. 'hp15', M odd): from z minus its M-bin running mean, with the filter's
    effect on the lag-0..2 autocovariances inverted exactly (stationary MA(2)); smooth shape errors on scales >> M
    bins barely enter.  'diff': from the first difference (noisy; for reference)."""
    if method == "plain":
        return _acf12(z), running_rms(z - z.mean(-1, keepdims=True), win)
    if method.startswith("hp") and method[2:].isdigit():
        return _jitter_rho_hp(z, int(method[2:]), win)
    if method == "diff":
        d = np.diff(z, axis=-1)
        pd = running_rms(d - d.mean(-1, keepdims=True), win)
        g = _acf12(d)
        g1, g2 = g[..., 0], g[..., 1]
        a11, a12, b1 = 2 + 2 * g1, -1.0, 1 + 2 * g1
        a21, a22, b2 = 2 * g2 - 1, 2.0, 2 * g2
        det = a11 * a22 - a12 * a21
        return np.stack([(b1 * a22 - a12 * b2) / det, (a11 * b2 - a21 * b1) / det], -1), np.concatenate([pd[..., :1], pd], -1)
    raise ValueError(method)


def predict_ratio(rho, prof, B=BOXCAR, het=True):
    """Predicted boxcar ratio r_pred from lag-1/2 autocorrelations (het=True: weights W_k from the rms profile; the
    effect is < 1 % for the real profile)."""
    n = prof.shape[-1] // B * B
    s = prof[..., :n].reshape(prof.shape[:-1] + (-1, B))
    out = np.ones(prof.shape[:-1])
    s0 = (s ** 2).sum((-1, -2))
    for k in (1, 2):
        W = (s[..., :-k] * s[..., k:]).sum((-1, -2)) / s0 if het else (1 - k / B) * np.ones_like(s0)
        out = out + 2 * rho[..., k - 1] * W
    return np.sqrt(np.clip(out, 0.04, None))


def standardised_stats(z, method="hp15", het=True, win=PROF_WIN):
    """ratio_c, rho (lag1, lag2), r_pred_c, S_c = ratio/r_pred and D_c = ratio^2 - r_pred^2 for each row of z."""
    rho, prof = jitter_rho(z, method, win)
    ratio = boxcar_ratio(z)
    rp = predict_ratio(rho, prof, het=het)
    return {"ratio": ratio, "rho": rho, "prof": prof, "r_pred": rp, "S": ratio / rp, "D": ratio ** 2 - rp ** 2}


def ma2_from_rho(r1, r2):
    """MA(2) coefficients (th1, th2) with the given lag-1/lag-2 autocorrelation (method of moments, per curve)."""
    from scipy.optimize import least_squares
    out = []
    for a, b in zip(np.atleast_1d(r1), np.atleast_1d(r2)):
        f = lambda t: [(t[0] + t[0] * t[1]) / (1 + t[0] ** 2 + t[1] ** 2) - a, t[1] / (1 + t[0] ** 2 + t[1] ** 2) - b]
        out.append(least_squares(f, [-0.6, 0.0], bounds=([-0.99, -0.99], [0.99, 0.99])).x)
    return np.array(out)


def simulate_null_z(theta, prof, n, rng, chol=None):
    """n pre-fit null realisations (n, C, N) of z: heteroscedastic MA(2) jitter (theta (C,2), prof (C,N));
    chol (C,C) optional cross-curve correlation applied to the innovations."""
    C, N = prof.shape
    e = rng.standard_normal((n, C, N + 2))
    if chol is not None:
        e = np.einsum("ij,njt->nit", chol, e)
    x = e[..., 2:] + theta[None, :, 0:1] * e[..., 1:-1] + theta[None, :, 1:2] * e[..., :-2]
    return x * prof[None]


def combine_pvalues(p):
    """Per-realisation combined statistics from per-curve p-values (last axis); larger = worse."""
    from scipy.stats import norm
    p = np.clip(p, 1e-12, 1.0)
    return {"minp": -np.log10(p.min(-1)), "fisher": -2 * np.log(p).sum(-1),
            "stouffer": norm.isf(np.clip(p, 1e-12, 1 - 1e-12)).sum(-1) / np.sqrt(p.shape[-1])}


def null_pvalues(D, sortedD):
    """Per-curve upper-tail p = (1 + #{null D >= D}) / (n+1) from the sorted null table sortedD (n, C)."""
    n = sortedD.shape[0]
    out = np.empty(np.shape(D))
    for c in range(sortedD.shape[1]):
        out[..., c] = (1 + n - np.searchsorted(sortedD[:, c], D[..., c], "left")) / (n + 1)
    return out


def calibrate_standardised(tp_raw, tm_raw, asc, method="hp15", n_tab=12000, n_thr=6000, size=0.01, seed=0,
                           prof_win=PROF_WIN, chunk=250, corr=None):
    """Build the per-curve null for the standardised gate from the data being tested.
    Fits MA(2) jitter (lag-1/2 from the plain z, once) + rms profile, simulates PRE-FIT residuals, runs them through
    the same pipeline as the data (A_p refit) and returns {'sortedD' (n_tab, 42), 'thr': {stat: threshold at `size`
    false-fail}, 'method', ...}.  The ECDF p-value resolves 1/(n_tab+1); min-p needs that below ~size/(2*42)
    (n_tab >= 12000 for size 0.01), otherwise a warning is issued and min-p loses power.  corr: optional (42,42) cross-curve correlation (the real data are consistent with 0)."""
    import warnings
    if 1.0 / (n_tab + 1) > size / 84:
        warnings.warn(f"n_tab={n_tab} too small to resolve min-p at size {size}; use n_tab >= {int(84 / size)}")
    rng = np.random.default_rng(seed)
    z, As = curves_z(tp_raw, tm_raw, asc)
    rho, _ = jitter_rho(z, "plain")
    prof = running_rms(z - z.mean(-1, keepdims=True), prof_win)
    theta = ma2_from_rho(rho[:, 0], rho[:, 1])
    chol = None
    if corr is not None:
        ev, V = np.linalg.eigh(corr)
        C2 = (V * np.clip(ev, 1e-6, None)) @ V.T
        d = np.sqrt(np.diag(C2)); chol = np.linalg.cholesky(C2 / np.outer(d, d))

    def sim(n):
        D = []
        for i in range(0, n, chunk):
            zz = simulate_null_z(theta, prof, min(chunk, n - i), rng, chol)
            tp = As["xip"][:, None] * asc["xip"] + asc["sigma_xip"] * zz[:, :21]
            tm = As["xim"][:, None] * asc["xim"] + asc["sigma_xim"] * zz[:, 21:]
            D.append(standardised_stats(curves_z(tp, tm, asc)[0], method)["D"])
        return np.concatenate(D)

    sortedD = np.sort(sim(n_tab), 0)
    comb = combine_pvalues(null_pvalues(sim(n_thr), sortedD))
    thr = {k: float(np.percentile(v, 100 * (1 - size))) for k, v in comb.items()}
    return {"method": method, "size": size, "n_tab": n_tab, "n_thr": n_thr, "sortedD": sortedD, "thr": thr,
            "theta": theta, "prof_win": prof_win}


def save_null(path, null):
    np.savez_compressed(path, sortedD=null["sortedD"].astype(np.float32), method=null["method"], size=null["size"],
                        thr_keys=np.array(list(null["thr"])), thr_vals=np.array(list(null["thr"].values())))


def load_null(path):
    d = np.load(path)
    return {"sortedD": d["sortedD"].astype(float), "method": str(d["method"]), "size": float(d["size"]),
            "thr": dict(zip(d["thr_keys"].tolist(), d["thr_vals"].tolist()))}


def a1b_noise_standardised(tp_raw, tm_raw, asc, sd_frac, null=None, stat=None, thresh=None, **calib_kw):
    """EXPERIMENTAL, NOT A GATE: end-to-end size fails (standardised/verify/VERIFY.md).  Calibrating the null on the
    data under test over-disperses the per-curve p-values (per-curve rho1/rho2 fit noise ~0.03/0.05 vs f(0) ~ 0.1):
    false-fail at the nominal 1 % thresholds is ~50 % (Fisher), ~24 % (Stouffer), worse for min-p.
    Standardised A1b-noise (OPTION; adopted a1b_noise unchanged).  Criterion (i): the chosen combined per-curve
    statistic ('minp' | 'fisher' | 'stouffer') <= its calibrated threshold (size 1 %); criterion (ii) as in a1b_noise.
    null: dict from calibrate_standardised / load_null (calibrated on the data itself if None; slow, ~1 min with the defaults).
    Returns a1b_noise-style dict + 'standardised' (per-curve ratio, r_pred, rho, D, p, all three statistics)."""
    th = dict(STD_DEFAULT if thresh is None else thresh)
    stat = stat or th["stat"]
    if null is None:
        null = calibrate_standardised(tp_raw, tm_raw, asc, th["method"], size=th["size"], **calib_kw)
    base = a1b_noise(tp_raw, tm_raw, asc, sd_frac)
    z, _ = curves_z(tp_raw, tm_raw, asc)
    st = standardised_stats(z, null["method"])
    p = null_pvalues(st["D"], null["sortedD"])
    comb = combine_pvalues(p)
    out = {k: base[k] for k in ("xip", "xim")}
    out["standardised"] = {"method": null["method"], "stat": stat, "value": float(comb[stat]), "threshold": null["thr"][stat],
                           "all": {k: float(v) for k, v in comb.items()}, "thr_all": dict(null["thr"]),
                           "p": p, "D": st["D"], "ratio": st["ratio"], "r_pred": st["r_pred"],
                           "rho1": st["rho"][:, 0], "rho2": st["rho"][:, 1], "argmin_p": int(p.argmin())}
    out["thresholds"] = {"stat": stat, "size": null["size"], "const_pte_min": th["const_pte_min"]}
    pt = min(base["xip"]["const_pte"], base["xim"]["const_pte"])
    out["pass"] = bool(comb[stat] <= null["thr"][stat] and pt > th["const_pte_min"])
    return out
