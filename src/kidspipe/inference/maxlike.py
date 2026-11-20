"""Maximum-a-posteriori (MAP) search and numerical-Hessian covariance.

Everything operates on an abstract ``logpost(theta) -> float`` (a
:class:`~kidspipe.inference.posterior.Posterior` or any callable) and a bounds
array ``(n, 2)``.  The optimiser works in the *rescaled unit cube*

.. math::  u_i = (\\theta_i - lo_i)/(hi_i - lo_i) \\in [0,1],

so that one step size means the same relative move for every parameter
(omch2 spans 1, delta_c spans 1e-3).  Strategy (multi-start):

1. L-BFGS-B with bounds and forward-difference gradients (step ``eps`` in u),
2. a derivative-free polish (Nelder-Mead and/or Powell, bounded) from the
   L-BFGS-B result, accepted only if it lowers ``-ln P``,
3. best of all starts is returned together with how many starts agree.

The covariance is the inverse Hessian of ``-ln P`` at the MAP computed by
central finite differences with per-parameter *step control* (the step is grown
or shrunk until the quadratic change of ``-ln P`` lies in a window well above
numerical noise and well below non-linearity) and a positive-definite repair
(eigenvalue floor in unit-cube coordinates, so flat directions get at most
``max_sigma_frac`` of the prior range).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

__all__ = ["MapResult", "HessianResult", "find_map", "numerical_hessian", "hessian_to_cov",
           "repair_positive_definite", "map_and_covariance"]

_BIG = 1.0e12  # finite penalty used instead of +inf (L-BFGS-B line search cannot digest inf)


@dataclass
class MapResult:
    """Outcome of :func:`find_map`.

    Attributes
    ----------
    theta : (n,) MAP point.
    neg_log_post : float, ``-ln P(theta)``.
    log_post : float, ``ln P(theta)``.
    n_eval : number of objective evaluations used.
    n_agree : number of starts whose optimum is within ``agree_tol`` of the best ``-ln P``.
    starts : list of per-start dicts (x0, theta, fun, n_eval, stage).
    success : True if at least one start finished with a finite optimum.
    time : wall time in seconds.
    """
    theta: np.ndarray
    neg_log_post: float
    n_eval: int
    n_agree: int
    starts: list = field(default_factory=list)
    success: bool = True
    time: float = 0.0

    @property
    def log_post(self) -> float:
        return -self.neg_log_post


@dataclass
class HessianResult:
    """Finite-difference Hessian of ``-ln P`` at ``x`` and derived covariance.

    ``hess`` is the repaired (symmetric positive definite) Hessian, ``hess_raw`` the
    raw finite-difference one, ``cov = inv(hess)``, ``steps`` the final steps (physical
    units), ``n_repaired`` the number of eigenvalues raised by the repair and
    ``x_eval`` the (possibly boundary-shifted) centre actually used.
    """
    hess: np.ndarray
    hess_raw: np.ndarray
    cov: np.ndarray
    steps: np.ndarray
    n_repaired: int
    x_eval: np.ndarray
    n_eval: int
    at_boundary: np.ndarray

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))


def _as_func(logpost):
    return logpost.log_post if hasattr(logpost, "log_post") else logpost


def _neg(logpost_f, counter):
    def f(theta):
        counter[0] += 1
        v = logpost_f(theta)
        if v is None or not np.isfinite(v):
            return _BIG
        return -float(v)
    return f


def find_map(logpost, bounds, x0: Optional[Sequence] = None, n_start: int = 6, seed: int = 0,
             sampler: Optional[Callable] = None, eps: float = 1e-6, maxiter: int = 300,
             polish: str = "both", ftol: float = 1e-12, agree_tol: float = 0.05,
             start_spread: float = 0.15, n_rounds: int = 2, verbose: bool = False) -> MapResult:
    """Multi-start bounded maximisation of ``ln P``.

    Parameters
    ----------
    logpost : callable ``theta -> ln P`` (or object with ``.log_post``).
    bounds : (n, 2) array of box bounds ``[lo, hi]``.
    x0 : initial point(s), shape (n,) or (m, n); used as the first starts.
    n_start : total number of starts (``x0`` rows count; the rest are random).
    sampler : optional ``rng -> theta`` returning a random start (e.g. prior draw); default is
        uniform in a box of relative half-width ``start_spread`` around the first start
        (or the whole cube if no ``x0``).
    eps : relative finite-difference step in unit-cube coordinates for the L-BFGS-B gradient.
    polish : 'both' (Nelder-Mead then Powell), 'nelder-mead', 'powell' or 'none'.
    n_rounds : number of (L-BFGS-B -> polish) rounds per start (restarting L-BFGS-B at the
        polished point cures stalls caused by noisy numerical gradients).
    """
    t_start = time.perf_counter()
    f_lp = _as_func(logpost)
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng_ = hi - lo
    n = len(lo)
    cnt = [0]
    fneg = _neg(f_lp, cnt)
    fu = lambda u: fneg(lo + np.clip(u, 0.0, 1.0) * rng_)
    rng = np.random.default_rng(seed)

    starts = []
    if x0 is not None:
        starts.extend(list(np.atleast_2d(np.asarray(x0, float))))
    centre = starts[0] if starts else None
    while len(starts) < max(n_start, 1):
        for _ in range(100):
            if sampler is not None:
                x = np.asarray(sampler(rng), float)
            elif centre is not None:
                uc = (centre - lo) / rng_
                x = lo + np.clip(uc + start_spread * rng.uniform(-1, 1, n), 0.0, 1.0) * rng_
            else:
                x = lo + rng.uniform(0.05, 0.95, n) * rng_
            if fneg(x) < _BIG:
                break
        starts.append(x)

    bu = [(0.0, 1.0)] * n
    results = []
    for x_start in starts:
        n0 = cnt[0]
        u = np.clip((np.asarray(x_start, float) - lo) / rng_, 0.0, 1.0)
        best_u, best_f = u.copy(), fu(u)
        stage = "init"
        for rnd in range(n_rounds):
            f_before = best_f
            r = minimize(fu, best_u, method="L-BFGS-B", bounds=bu,
                         options=dict(maxiter=maxiter, ftol=ftol, gtol=1e-9, eps=eps, maxfun=50 * (n + 1) * maxiter))
            if r.fun < best_f:
                best_u, best_f, stage = np.clip(r.x, 0, 1), r.fun, "lbfgsb"
            methods = {"both": ("Nelder-Mead", "Powell"), "nelder-mead": ("Nelder-Mead",),
                       "powell": ("Powell",), "none": ()}[polish.lower()]
            for m in methods:
                if m == "Nelder-Mead":
                    simplex = np.vstack([best_u] + [np.clip(best_u + 2e-3 * np.eye(n)[i], 0, 1) for i in range(n)])
                    opts = dict(maxiter=400 * n, xatol=1e-8, fatol=1e-10, adaptive=True, initial_simplex=simplex)
                    rr = minimize(fu, best_u, method=m, bounds=bu, options=opts)
                else:
                    rr = minimize(fu, best_u, method=m, bounds=bu,
                                  options=dict(maxiter=40 * n, xtol=1e-8, ftol=1e-12))
                if rr.fun < best_f - 1e-12:
                    best_u, best_f, stage = np.clip(rr.x, 0, 1), rr.fun, stage + "+" + m
            if verbose:
                print(f"  start round {rnd}: -lnP = {best_f:.6f}")
            if f_before - best_f < 1e-7:
                break
        results.append(dict(x0=np.asarray(x_start), theta=lo + best_u * rng_, fun=float(best_f),
                            n_eval=cnt[0] - n0, stage=stage))
    funs = np.array([r["fun"] for r in results])
    ib = int(np.argmin(funs))
    ok = bool(np.isfinite(funs[ib]) and funs[ib] < _BIG)
    n_agree = int(np.sum(funs < funs[ib] + agree_tol))
    return MapResult(theta=results[ib]["theta"], neg_log_post=float(funs[ib]), n_eval=cnt[0],
                     n_agree=n_agree, starts=results, success=ok, time=time.perf_counter() - t_start)


# --------------------------------------------------------------------------- Hessian
def repair_positive_definite(H: np.ndarray, scale: Optional[np.ndarray] = None,
                             max_sigma_frac: float = 0.5, floor_rel: float = 1e-10):
    """Symmetrise ``H`` and make it positive definite by an eigenvalue repair.

    Work in unit-cube coordinates ``H_u = S H S`` with ``S = diag(scale)`` (parameter
    ranges).  With eigen-decomposition ``H_u = V diag(w) V^T`` and
    ``w_floor = floor_rel * max|w|``, every eigenvalue ``w <= w_floor`` (non-positive or
    negligible curvature, i.e. a flat direction or numerical noise) is replaced by
    ``max(|w|, 1/max_sigma_frac^2)``: a direction with no usable curvature gets a standard
    deviation of at most ``max_sigma_frac`` of the prior range, and a clearly negative
    curvature keeps its magnitude.  Healthy matrices are returned unchanged.

    Returns ``(H_repaired, n_changed)`` in the original coordinates.
    """
    H = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    n = H.shape[0]
    S = np.ones(n) if scale is None else np.asarray(scale, float)
    Hu = H * np.outer(S, S)
    w, V = np.linalg.eigh(Hu)
    wfloor = floor_rel * max(np.max(np.abs(w)), 1e-300)
    bad = w <= wfloor
    if not bad.any():
        return H, 0
    w_new = np.where(bad, np.maximum(np.abs(w), 1.0 / max_sigma_frac ** 2), w)
    Hn = ((V * w_new) @ V.T) / np.outer(S, S)
    return 0.5 * (Hn + Hn.T), int(bad.sum())


def hessian_to_cov(H: np.ndarray) -> np.ndarray:
    """Covariance = inverse of a positive definite Hessian (via Cholesky); symmetric output."""
    L = np.linalg.cholesky(H)
    Li = np.linalg.solve(L, np.eye(H.shape[0]))
    C = Li.T @ Li
    return 0.5 * (C + C.T)


def numerical_hessian(logpost, x, bounds, h0: float = 1e-3, target=(0.02, 0.5),
                      max_step_iter: int = 14, repair: bool = True,
                      max_sigma_frac: float = 0.5) -> HessianResult:
    """Central-difference Hessian of ``-ln P`` at ``x`` with step control.

    Diagonal and mixed second derivatives (physical coordinates, units of the parameters):

    .. math::
        H_{ii} \\approx \\frac{f(x+h_ie_i)+f(x-h_ie_i)-2f(x)}{h_i^2},\\quad
        H_{ij} \\approx \\frac{f_{++}-f_{+-}-f_{-+}+f_{--}}{4h_ih_j},\\quad f=-\\ln P .

    Step control: ``h_i`` starts at ``h0*(hi_i-lo_i)`` and is doubled / halved (at most
    ``max_step_iter`` times) until ``f(x+h)+f(x-h)-2f(x)`` lies in ``target`` (units of
    ln P), i.e. well above rounding noise but where the quadratic approximation holds.
    The stencil is kept inside ``bounds`` by shifting the centre inward if ``x`` is closer
    than ``h_i`` to a bound (``at_boundary`` flags such parameters).
    """
    f_lp = _as_func(logpost)
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng_ = hi - lo
    n = len(lo)
    x = np.clip(np.asarray(x, float).copy(), lo, hi)
    cnt = [0]
    f = lambda th: (cnt.__setitem__(0, cnt[0] + 1), -float(f_lp(th)))[1]
    f0 = f(x)
    if not np.isfinite(f0):
        raise ValueError("numerical_hessian: -ln P is not finite at x")
    steps = np.full(n, h0) * rng_
    hmin, hmax = 1e-8 * rng_, 0.25 * rng_
    e = np.eye(n)
    at_b = np.zeros(n, bool)
    for i in range(n):
        h = np.clip(steps[i], hmin[i], hmax[i])
        for _ in range(max_step_iter):
            xc = x.copy()
            xc[i] = np.clip(x[i], lo[i] + 1.001 * h, hi[i] - 1.001 * h) if 2.002 * h < rng_[i] else x[i]
            fp, fm = f(xc + h * e[i]), f(xc - h * e[i])
            d2 = fp + fm - 2.0 * f(xc)
            if not np.isfinite(d2):
                h *= 0.5
            elif d2 < target[0]:
                if h >= hmax[i]:
                    break
                h = min(h * (2.0 if d2 > 0.1 * target[0] else 4.0), hmax[i])
            elif d2 > target[1]:
                if h <= hmin[i]:
                    break
                h *= 0.5
            else:
                break
            h = float(np.clip(h, hmin[i], hmax[i]))
        steps[i] = h
    xe = np.clip(x, lo + 1.001 * steps, hi - 1.001 * steps)
    at_b = np.abs(xe - x) > 0
    fc = f(xe)
    fp = np.array([f(xe + steps[i] * e[i]) for i in range(n)])
    fm = np.array([f(xe - steps[i] * e[i]) for i in range(n)])
    H = np.zeros((n, n))
    H[np.diag_indices(n)] = (fp + fm - 2 * fc) / steps ** 2
    for i in range(n):
        for j in range(i + 1, n):
            fpp = f(xe + steps[i] * e[i] + steps[j] * e[j])
            fmm = f(xe - steps[i] * e[i] - steps[j] * e[j])
            fpm = f(xe + steps[i] * e[i] - steps[j] * e[j])
            fmp = f(xe - steps[i] * e[i] + steps[j] * e[j])
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * steps[i] * steps[j])
    if not np.all(np.isfinite(H)):
        bad = sorted(set(np.nonzero(~np.isfinite(H))[0].tolist()))
        raise ValueError(f"numerical_hessian: non-finite -ln P within the finite-difference stencil "
                         f"(parameters {bad}); x is too close to the edge of the posterior support")
    H_raw = H.copy()
    nrep = 0
    if repair:
        H, nrep = repair_positive_definite(H, scale=rng_, max_sigma_frac=max_sigma_frac)
    try:
        cov = hessian_to_cov(H)
    except np.linalg.LinAlgError:
        H, nrep2 = repair_positive_definite(H + 1e-12 * np.eye(n) * np.trace(H) / n, scale=rng_,
                                            max_sigma_frac=max_sigma_frac)
        nrep += nrep2
        cov = hessian_to_cov(H)
    return HessianResult(hess=H, hess_raw=H_raw, cov=cov, steps=steps, n_repaired=nrep,
                         x_eval=xe, n_eval=cnt[0], at_boundary=at_b)


def map_and_covariance(logpost, bounds, **kw):
    """Convenience: :func:`find_map` followed by :func:`numerical_hessian` at the MAP.

    ``kw`` is passed to :func:`find_map`.  Returns ``(MapResult, HessianResult)``.
    """
    res = find_map(logpost, bounds, **kw)
    hr = numerical_hessian(logpost, res.theta, bounds)
    return res, hr
