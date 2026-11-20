"""Weighted-sample utilities: marginal statistics, S8, PJ-HPD, burn-in, chain I/O, text summary.

All functions work on *weighted samples*: ``samples (n, d)`` and non-negative ``weights (n,)``
(``None`` = equal weights).  MCMC chains carry integer repeat counts or unit weights; nested
sampling carries posterior weights.

Projected joint highest posterior density (PJ-HPD)
--------------------------------------------------
The 1-D "PJ-HPD" credible interval used by the KiDS analyses (Joachimi et al. 2021, Asgari et
al. 2021) is built from the *full-dimensional* posterior: take the joint HPD region
``R_q = { theta : P(theta) >= P_q }`` that contains a fraction ``q`` of the posterior mass, and
project it on each axis, ``[min_{R_q} theta_i, max_{R_q} theta_i]``.  With samples this needs
the posterior density at each sample (``log_post = ln pi + ln L``, available for MCMC and, for
nested samples, ``ln L + ln pi``).  Sorting by density gives ``R_q`` for **every** ``q`` at once,
and the extents follow from running min/max along the sorted order: ``O(N log N + N d)``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["Chain", "weighted_mean", "weighted_cov", "weighted_quantile", "hpd_interval_1d",
           "marginal_mode", "marginal_stats", "s8_column", "append_s8", "pjhpd_extents",
           "pjhpd_mask", "remove_burn_in", "estimate_burn_in", "save_chain", "load_chain",
           "summary_table", "correlation_table"]


def _w(weights, n):
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, float)
    if w.shape != (n,) or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite, non-negative with shape (n,)")
    return w / w.sum()


def weighted_mean(samples, weights=None) -> np.ndarray:
    """Weighted mean per column: ``sum_i w_i x_i`` with ``sum w = 1``."""
    x = np.asarray(samples, float)
    return _w(weights, len(x)) @ x


def weighted_cov(samples, weights=None) -> np.ndarray:
    """Weighted covariance (reliability weights, bias-corrected by ``1/(1 - sum w^2)``)."""
    x = np.asarray(samples, float)
    w = _w(weights, len(x))
    mu = w @ x
    xc = x - mu
    return (xc * w[:, None]).T @ xc / (1.0 - np.sum(w ** 2))


def weighted_quantile(x, weights, q) -> np.ndarray:
    """Weighted quantiles of a 1-D sample using the mid-point CDF ``(cumsum(w) - w/2)/sum(w)``."""
    x = np.asarray(x, float)
    w = _w(weights, len(x))
    o = np.argsort(x, kind="stable")
    cdf = np.cumsum(w[o]) - 0.5 * w[o]
    return np.interp(np.asarray(q, float), cdf, x[o])


def hpd_interval_1d(x, weights=None, level: float = 0.68):
    """Shortest 1-D interval containing posterior mass ``level`` (marginal HPD), O(N log N).

    For each sorted sample ``i`` the smallest ``j`` with ``F_j - F_{i-1} >= level`` is found
    by ``searchsorted``; the narrowest ``[x_i, x_j]`` is returned as ``(lo, hi)``.
    """
    x = np.asarray(x, float)
    w = _w(weights, len(x))
    o = np.argsort(x, kind="stable")
    xs = x[o]
    F = np.cumsum(w[o])
    Fprev = np.concatenate([[0.0], F[:-1]])
    j = np.searchsorted(F, Fprev + level - 1e-12, side="left")
    ok = j < len(xs)
    if not ok.any():
        return float(xs[0]), float(xs[-1])
    i = np.nonzero(ok)[0]
    width = xs[j[i]] - xs[i]
    k = i[np.argmin(width)]
    return float(xs[k]), float(xs[j[k]])


def marginal_mode(x, weights=None, bins: int = 100, smooth: float = 2.0) -> float:
    """Mode of the weighted 1-D marginal: Gaussian-smoothed histogram (``smooth`` in bins)."""
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(x, float)
    w = _w(weights, len(x))
    h, e = np.histogram(x, bins=bins, weights=w)
    h = gaussian_filter1d(h, smooth, mode="nearest")
    k = int(np.argmax(h))
    return float(0.5 * (e[k] + e[k + 1]))


def s8_column(sigma8, omega_m):
    """Derived ``S8 = sigma8 * sqrt(Omega_m / 0.3)`` (spec s1.2); works elementwise on arrays."""
    return np.asarray(sigma8, float) * np.sqrt(np.asarray(omega_m, float) / 0.3)


def append_s8(samples, names, sigma8="sigma8", omega_m="omega_m", name="S8"):
    """Append an S8 column to ``samples`` using the named sigma8 and omega_m columns.

    Returns ``(samples_with_S8, names + (name,))``.
    """
    names = list(names)
    x = np.asarray(samples, float)
    s8 = s8_column(x[:, names.index(sigma8)], x[:, names.index(omega_m)])
    return np.column_stack([x, s8]), tuple(names + [name])


def pjhpd_extents(samples, log_post, weights=None, levels=(0.68, 0.95)) -> np.ndarray:
    """Projected joint-HPD intervals for several mass fractions at once.

    Parameters
    ----------
    samples : (n, d) (include derived columns such as S8 if wanted).
    log_post : (n,) ln posterior density at each sample (``ln pi + ln L``; any monotone
        function of the density gives the same regions).
    weights : (n,) posterior weights (``None`` = equal).
    levels : mass fractions ``q`` of the joint HPD region.

    Returns
    -------
    ext : (n_levels, d, 2) array of ``[min, max]`` of every column over the joint HPD region
        holding mass ``q`` (the smallest density-ordered set with cumulative weight >= q).
        O(N log N) (one sort) + O(N d) running min/max.
    """
    x = np.asarray(samples, float)
    lp = np.asarray(log_post, float)
    w = _w(weights, len(x))
    o = np.argsort(-lp, kind="stable")
    cw = np.cumsum(w[o])
    xs = x[o]
    cmin = np.minimum.accumulate(xs, axis=0)
    cmax = np.maximum.accumulate(xs, axis=0)
    out = np.empty((len(levels), x.shape[1], 2))
    for a, q in enumerate(levels):
        k = min(int(np.searchsorted(cw, q - 1e-12, side="left")), len(x) - 1)
        out[a, :, 0] = cmin[k]
        out[a, :, 1] = cmax[k]
    return out


def pjhpd_mask(log_post, weights=None, level: float = 0.68) -> np.ndarray:
    """Boolean mask of the samples inside the joint HPD region of mass ``level``."""
    lp = np.asarray(log_post, float)
    w = _w(weights, len(lp))
    o = np.argsort(-lp, kind="stable")
    k = min(int(np.searchsorted(np.cumsum(w[o]), level - 1e-12, side="left")), len(lp) - 1)
    m = np.zeros(len(lp), bool)
    m[o[:k + 1]] = True
    return m


def marginal_stats(samples, weights=None, names: Optional[Sequence[str]] = None, level: float = 0.68,
                   log_post=None) -> dict:
    """Per-parameter marginal summary.

    Returns ``{name: dict(mean, std, median, lo, hi, hpd_lo, hpd_hi, mode[, pj_lo, pj_hi])}``
    where ``lo/hi`` are the equal-tailed ``level`` credible limits, ``hpd_*`` the 1-D
    marginal HPD, and ``pj_*`` (if ``log_post`` is given) the projected joint HPD.
    """
    x = np.asarray(samples, float)
    n, d = x.shape
    names = tuple(names) if names is not None else tuple(f"p{i}" for i in range(d))
    w = _w(weights, n)
    mu = w @ x
    sd = np.sqrt(np.maximum(w @ (x - mu) ** 2 / (1 - np.sum(w ** 2)), 0.0))
    ext = pjhpd_extents(x, log_post, w, (level,))[0] if log_post is not None else None
    out = {}
    for i, nm in enumerate(names):
        lo, med, hi = weighted_quantile(x[:, i], w, [0.5 - level / 2, 0.5, 0.5 + level / 2])
        hl, hh = hpd_interval_1d(x[:, i], w, level)
        r = dict(mean=float(mu[i]), std=float(sd[i]), median=float(med), lo=float(lo), hi=float(hi),
                 hpd_lo=hl, hpd_hi=hh, mode=marginal_mode(x[:, i], w))
        if ext is not None:
            r["pj_lo"], r["pj_hi"] = float(ext[i, 0]), float(ext[i, 1])
        out[nm] = r
    return out


# --------------------------------------------------------------------------- burn-in
def estimate_burn_in(log_post, frac_max: float = 0.5) -> int:
    """Heuristic burn-in length for one chain: first step at which ``ln P`` reaches the
    median of the second half of the chain (capped at ``frac_max * n``)."""
    lp = np.asarray(log_post, float)
    n = len(lp)
    ref = np.median(lp[n // 2:])
    above = np.nonzero(lp >= ref)[0]
    k = int(above[0]) if len(above) else n // 2
    return min(k, int(frac_max * n))


def remove_burn_in(chain, n_burn: Optional[int] = None, frac: Optional[float] = None, log_post=None):
    """Drop the first steps along axis 0 of ``chain`` (any trailing shape, also (steps, walkers, d)).

    Give ``n_burn`` (steps), or ``frac`` (fraction of steps), or ``log_post`` (1-D or
    (steps, walkers); the burn-in is estimated with :func:`estimate_burn_in`, using the
    walker-mean ln P for ensembles).
    """
    c = np.asarray(chain)
    if n_burn is None:
        if frac is not None:
            n_burn = int(frac * len(c))
        elif log_post is not None:
            lp = np.asarray(log_post, float)
            n_burn = estimate_burn_in(lp if lp.ndim == 1 else lp.mean(axis=1))
        else:
            raise ValueError("give n_burn, frac or log_post")
    return c[int(n_burn):]


# --------------------------------------------------------------------------- Chain container + I/O
@dataclass
class Chain:
    """A weighted sample with metadata.

    ``samples`` (n, d), ``names`` (d,), ``weights`` (n,) (default ones), optional ``log_post`` (ln pi + ln L)
    and ``log_like`` (n,), and a JSON-serialisable ``meta`` dict (sampler, log_z, ...).
    """
    samples: np.ndarray
    names: tuple
    weights: Optional[np.ndarray] = None
    log_post: Optional[np.ndarray] = None
    log_like: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.samples = np.asarray(self.samples, float)
        self.names = tuple(self.names)
        if self.weights is None:
            self.weights = np.ones(len(self.samples))
        self.weights = np.asarray(self.weights, float)

    @property
    def n_eff(self) -> float:
        w = self.weights / self.weights.sum()
        return float(1.0 / np.sum(w ** 2))

    def column(self, name):
        return self.samples[:, self.names.index(name)]

    def mean(self):
        return weighted_mean(self.samples, self.weights)

    def cov(self):
        return weighted_cov(self.samples, self.weights)

    def with_s8(self, sigma8="sigma8", omega_m="omega_m") -> "Chain":
        """Copy with an extra ``S8`` column."""
        x, nm = append_s8(self.samples, self.names, sigma8, omega_m)
        return Chain(x, nm, self.weights, self.log_post, self.log_like, dict(self.meta))

    def stats(self, level: float = 0.68):
        return marginal_stats(self.samples, self.weights, self.names, level, self.log_post)

    def map_point(self):
        if self.log_post is None:
            raise ValueError("no log_post stored")
        return self.samples[int(np.argmax(self.log_post))]

    def burn(self, n_burn: Optional[int] = None, frac: Optional[float] = None) -> "Chain":
        """Copy without the first ``n_burn`` samples (single-chain ordering assumed)."""
        k = int(n_burn if n_burn is not None else (frac or 0.0) * len(self.samples))
        s = slice(k, None)
        g = lambda a: None if a is None else a[s]
        return Chain(self.samples[s], self.names, self.weights[s], g(self.log_post), g(self.log_like),
                     dict(self.meta))


def _prefix(path):
    p = str(path)
    for ext in (".npz", ".txt"):
        if p.endswith(ext):
            return p[:-len(ext)]
    return p


def save_chain(path, chain: Chain, fmt: str = "%.10e") -> tuple:
    """Write ``<prefix>.npz`` (exact, with metadata) and ``<prefix>.txt`` (human readable).

    The text file starts with ``#`` header lines (format tag, sizes, JSON metadata, column
    names) followed by rows ``weight [log_post] [log_like] p1 ... pd``.  Returns the two paths.
    """
    pre = _prefix(path)
    if any((not n) or any(c.isspace() for c in n) for n in chain.names):
        raise ValueError("save_chain: parameter names must be non-empty and contain no whitespace")
    os.makedirs(os.path.dirname(os.path.abspath(pre)), exist_ok=True)
    npz = pre + ".npz"
    txt = pre + ".txt"
    meta = json.dumps(chain.meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    arrs = dict(samples=chain.samples, weights=chain.weights, names=np.array(chain.names), meta=np.array(meta))
    cols = ["weight"]
    mats = [chain.weights[:, None]]
    if chain.log_post is not None:
        arrs["log_post"] = chain.log_post
        cols.append("log_post")
        mats.append(np.asarray(chain.log_post)[:, None])
    if chain.log_like is not None:
        arrs["log_like"] = chain.log_like
        cols.append("log_like")
        mats.append(np.asarray(chain.log_like)[:, None])
    np.savez_compressed(npz, **arrs)
    cols += list(chain.names)
    mats.append(chain.samples)
    header = (f"kidspipe chain v1\nn_samples = {len(chain.samples)}  n_params = {len(chain.names)}\n"
              f"meta = {meta}\ncolumns: " + " ".join(cols))
    np.savetxt(txt, np.hstack(mats), header=header, fmt=fmt)
    return npz, txt


def load_chain(path) -> Chain:
    """Load a chain written by :func:`save_chain` (prefers ``.npz``; falls back to the text file)."""
    pre = _prefix(path)
    if os.path.exists(pre + ".npz") and not str(path).endswith(".txt"):
        z = np.load(pre + ".npz", allow_pickle=False)
        return Chain(z["samples"], tuple(str(s) for s in z["names"]), z["weights"],
                     z["log_post"] if "log_post" in z else None,
                     z["log_like"] if "log_like" in z else None, json.loads(str(z["meta"])))
    with open(pre + ".txt") as fh:
        hdr = [ln[1:].strip() for ln in fh if ln.startswith("#")]
    cols = [h for h in hdr if h.startswith("columns:")][0].split(":", 1)[1].split()
    meta = json.loads([h for h in hdr if h.startswith("meta =")][0].split("=", 1)[1])
    data = np.loadtxt(pre + ".txt", ndmin=2)
    j = 1
    lpost = llike = None
    if cols[j] == "log_post":
        lpost, j = data[:, j], j + 1
    if cols[j] == "log_like":
        llike, j = data[:, j], j + 1
    return Chain(data[:, j:], tuple(cols[j:]), data[:, 0], lpost, llike, meta)


# --------------------------------------------------------------------------- text summaries
def summary_table(chain_or_samples, weights=None, names=None, log_post=None, level: float = 0.68,
                  map_point=None, fmt: str = "{:10.4f}") -> str:
    """Plain-text marginal table: mean, std, median, equal-tailed limits, marginal HPD, PJ-HPD, MAP.

    Accepts a :class:`Chain` or ``samples`` with ``weights``/``names``/``log_post``.
    """
    if isinstance(chain_or_samples, Chain):
        c = chain_or_samples
        samples, weights, names, log_post = c.samples, c.weights, c.names, c.log_post
        if map_point is None and log_post is not None:
            map_point = c.map_point()
    else:
        samples = np.asarray(chain_or_samples, float)
    st = marginal_stats(samples, weights, names, level, log_post)
    pj = log_post is not None
    head = ["parameter", "mean", "std", "median", f"lo{int(level*100)}", f"hi{int(level*100)}",
            "hpd_lo", "hpd_hi"] + (["pj_lo", "pj_hi"] if pj else []) + (["MAP"] if map_point is not None else [])
    lines = ["  ".join([f"{head[0]:<14s}"] + [f"{h:>10s}" for h in head[1:]])]
    for i, (nm, r) in enumerate(st.items()):
        vals = [r["mean"], r["std"], r["median"], r["lo"], r["hi"], r["hpd_lo"], r["hpd_hi"]]
        if pj:
            vals += [r["pj_lo"], r["pj_hi"]]
        if map_point is not None:
            vals.append(float(np.asarray(map_point)[i]))
        lines.append("  ".join([f"{nm:<14s}"] + [fmt.format(v) for v in vals]))
    ne = 1.0 / np.sum(_w(weights, len(samples)) ** 2)
    lines.append(f"# {len(samples)} samples, Kish n_eff = {ne:.0f}; limits = {level:.0%}; pj = projected joint HPD")
    return "\n".join(lines)


def correlation_table(samples, weights=None, names=None, width: int = 6) -> str:
    """Lower-triangular text table of the weighted correlation matrix (two decimals)."""
    x = np.asarray(samples, float)
    d = x.shape[1]
    names = tuple(names) if names is not None else tuple(f"p{i}" for i in range(d))
    C = weighted_cov(x, weights)
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    lines = [" " * 14 + " ".join(f"{n[:width]:>{width}s}" for n in names)]
    for i in range(d):
        row = " ".join(f"{R[i, j]:{width}.2f}" if j <= i else " " * width for j in range(d))
        lines.append(f"{names[i][:13]:<14s}{row}")
    return "\n".join(lines)
