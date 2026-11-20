"""MCMC samplers and convergence diagnostics.

Three samplers, all driven by an abstract ``logpost(theta) -> float`` (``-inf`` for
rejected points) and numpy arrays:

``adaptive_metropolis``
    Haario-Bachelier-Saksman-Lahdesmaki adaptive random-walk Metropolis.  The proposal
    covariance ``s^2 * C`` with ``s^2 = (2.38^2/d) * exp(2 log_s)`` is learned **only during
    burn-in** (recursive sample covariance blended with a user prior covariance, e.g. the
    inverse MAP Hessian; the scalar ``log_s`` is tuned by Robbins-Monro to a target
    acceptance).  After burn-in the proposal is frozen, so the production chain is an
    exact Markov chain.

``blocked_metropolis``
    Fast/slow oversampling.  One *cycle* executes a schedule ``[(block_1, r_1), (block_2, r_2),
    ...]`` of single-block random-walk Metropolis updates, e.g. one move of the slow
    (Boltzmann) block, then ``r`` moves of the medium block, then ``r'`` moves of the
    fast block, which are cheap because the model caches its stages.  The proposal for
    block ``b`` is Gaussian with the *conditional* covariance
    ``(P_bb)^{-1}`` of the learned full covariance (``P = C^{-1}``), the optimal choice for a
    Gibbs-like block move.  Every block has its own Robbins-Monro scale and acceptance
    rate.  Because rejection of a slow move leaves the proposed point in the model cache,
    the model should hold >= 2 slow-stage entries (``LinearBackend(cache_size>=2)``).

``ensemble_sampler``
    Affine-invariant ensemble sampler of Goodman & Weare (2010), stretch move

    .. math:: Y = X_c + Z (X_k - X_c),\\quad g(z) \\propto z^{-1/2},\\ z\\in[1/a, a],\\quad
              \\ln q = (d-1)\\ln Z + \\ln P(Y) - \\ln P(X_k),

    implemented in the parallel (two half-ensembles) form of Foreman-Mackey et al. (2013),
    vectorised over the walkers of a half-ensemble.

Diagnostics: split Gelman-Rubin ``R-hat`` (optionally rank-normalised), FFT-based
integrated autocorrelation time ``tau`` (Sokal windowing) and effective sample size
``ESS = N_samples / tau``.  Array convention: axis 0 = steps, optional axis 1 = chains
(walkers), last axis = parameters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["ChainResult", "EnsembleResult", "adaptive_metropolis", "blocked_metropolis",
           "ensemble_sampler", "split_rhat", "integrated_autocorr_time",
           "effective_sample_size", "autocorr_function", "default_proposal_cov"]


def _as_func(logpost):
    return logpost.log_post if hasattr(logpost, "log_post") else logpost


def default_proposal_cov(bounds, frac: float = 0.02) -> np.ndarray:
    """Diagonal fallback proposal covariance: ``(frac * range_i)^2``."""
    b = np.asarray(bounds, float)
    return np.diag((frac * (b[:, 1] - b[:, 0])) ** 2)


def _opt_accept(d: int) -> float:
    """Roberts-Rosenthal optimal acceptance for a d-dimensional Gaussian random walk."""
    return 0.234 + (0.44 - 0.234) / max(d, 1) ** 0.9 if d < 8 else 0.234


def _safe_chol(C, jitter_rel=1e-10):
    C = 0.5 * (C + C.T)
    j = jitter_rel * max(np.mean(np.diag(C)), 1e-300)
    for k in range(8):
        try:
            return np.linalg.cholesky(C + (j * 10 ** k) * np.eye(len(C)))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("proposal covariance not positive definite")


class _RunningCov:
    """Recursive (Welford) mean and covariance."""

    def __init__(self, d):
        self.n = 0
        self.mean = np.zeros(d)
        self.M2 = np.zeros((d, d))

    def add(self, x):
        self.n += 1
        dx = x - self.mean
        self.mean += dx / self.n
        self.M2 += np.outer(dx, x - self.mean)

    def cov(self):
        return self.M2 / max(self.n - 1, 1)


@dataclass
class ChainResult:
    """Output of the Metropolis samplers.

    ``samples`` (n, d) and ``log_post`` (n,) are the *post-burn-in* states (one per step /
    cycle after thinning); ``burn_samples``/``burn_log_post`` the burn-in part.
    ``accept`` is the production acceptance rate (per block in ``accept_blocks``),
    ``cov`` the final proposal covariance (the learned full covariance), ``scales`` the
    final proposal scale factors, ``n_eval`` the number of ``logpost`` calls and ``time`` the
    wall time in seconds.
    """
    samples: np.ndarray
    log_post: np.ndarray
    burn_samples: np.ndarray
    burn_log_post: np.ndarray
    accept: float
    cov: np.ndarray
    n_eval: int
    time: float
    accept_blocks: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)

    @property
    def all_samples(self):
        return np.vstack([self.burn_samples, self.samples])


def adaptive_metropolis(logpost, x0, n_steps: int, n_burn: int = 2000, cov0=None, bounds=None,
                        seed: int = 0, adapt_every: int = 25, cov_prior_weight: float = 100.0,
                        target_accept: Optional[float] = None, adapt_scale: bool = True,
                        thin: int = 1, init_scale: float = 1.0) -> ChainResult:
    """Adaptive Metropolis (Haario et al. 2001) with burn-in-only adaptation.

    Parameters
    ----------
    logpost : callable ``theta -> ln P`` (``-inf`` allowed).
    x0 : start point (d,); must have finite ``ln P``.
    n_steps : production steps after burn-in.
    n_burn : burn-in steps during which the proposal is adapted (these are returned separately).
    cov0 : prior proposal covariance (d,d), e.g. the inverse MAP Hessian; default
        ``default_proposal_cov(bounds)`` (or 1e-2 identity without bounds).
    cov_prior_weight : pseudo-sample count of ``cov0`` in the blend
        ``C = (w C0 + n C_emp)/(w + n)`` (n = burn-in states seen so far).
    target_accept : target acceptance of the Robbins-Monro scale tuning (default 0.234 for
        d>=8).  The scale starts at ``init_scale * 2.38/sqrt(d)``.
    thin : keep every ``thin``-th production state.
    Returns :class:`ChainResult`.
    """
    t0 = time.perf_counter()
    f = _as_func(logpost)
    x = np.array(x0, float).ravel()
    d = x.size
    lp = f(x)
    n_eval = 1
    if not np.isfinite(lp):
        raise ValueError("adaptive_metropolis: ln P(x0) is not finite")
    if cov0 is None:
        cov0 = default_proposal_cov(bounds) if bounds is not None else 1e-2 * np.eye(d)
    cov0 = np.asarray(cov0, float)
    target = _opt_accept(d) if target_accept is None else target_accept
    rng = np.random.default_rng(seed)
    rc = _RunningCov(d)
    rc.add(x)
    C = cov0.copy()
    L = _safe_chol(C)
    log_s = np.log(init_scale * 2.38 / np.sqrt(d))
    n_tot = n_burn + n_steps
    out_x = np.empty((n_tot, d))
    out_lp = np.empty(n_tot)
    acc_burn = acc_prod = 0
    t_adapt = 0
    i = 0
    while i < n_tot:
        m = min(adapt_every, n_tot - i)
        LZ = rng.standard_normal((m, d)) @ L.T
        lu = np.log(rng.uniform(size=m))
        for k in range(m):
            s = np.exp(log_s)
            y = x + s * LZ[k]
            lpy = f(y)
            n_eval += 1
            a = lpy - lp
            ok = (lpy > -np.inf) and (lu[k] < a)
            if ok:
                x, lp = y, lpy
            step = i + k
            if step < n_burn:
                acc_burn += ok
                rc.add(x)
                if adapt_scale:
                    t_adapt += 1
                    p_acc = 1.0 if (lpy > -np.inf and a >= 0) else (np.exp(a) if lpy > -np.inf else 0.0)
                    log_s += (p_acc - target) / (t_adapt + 10.0) ** 0.6 * 2.0
            else:
                acc_prod += ok
            out_x[step] = x
            out_lp[step] = lp
        i += m
        if i <= n_burn:   # covariance update at the end of each adaptation block, burn-in only
            w = cov_prior_weight
            C = (w * cov0 + rc.n * rc.cov()) / (w + rc.n)
            L = _safe_chol(C)
    prod_n = max(n_steps, 1)
    return ChainResult(samples=out_x[n_burn::thin].copy(), log_post=out_lp[n_burn::thin].copy(),
                       burn_samples=out_x[:n_burn].copy(), burn_log_post=out_lp[:n_burn].copy(),
                       accept=acc_prod / prod_n, cov=C, n_eval=n_eval, time=time.perf_counter() - t0,
                       accept_blocks={"all": acc_prod / prod_n, "all_burn": acc_burn / max(n_burn, 1)},
                       scales={"all": float(np.exp(log_s))})


def _conditional_cov(C, idx):
    """Conditional covariance of block ``idx`` given all other coordinates = inv(inv(C)[idx, idx])."""
    P = np.linalg.inv(C)
    Pb = P[np.ix_(idx, idx)]
    return np.linalg.inv(0.5 * (Pb + Pb.T))


def blocked_metropolis(logpost, x0, schedule: Sequence, n_cycles: int, n_burn_cycles: int = 500,
                       cov0=None, bounds=None, seed: int = 0, adapt_every: int = 25,
                       cov_prior_weight: float = 100.0, thin: int = 1,
                       target_accept: Optional[float] = None, allow_partial: bool = False) -> ChainResult:
    """Fast/slow blocked random-walk Metropolis with oversampling of cheap blocks.

    Parameters
    ----------
    schedule : sequence of ``(indices, repeats)``.  One cycle performs, in this order and for
        each entry, ``repeats`` Metropolis updates proposing only the coordinates ``indices``.
        Typical: ``[(slow_and_medium_idx, 1), (fast_idx, 8)]`` or
        ``[(slow_idx, 1), (medium_idx, 3), (fast_idx, 8)]`` (cost-ordered: expensive first).
    n_cycles, n_burn_cycles : production / burn-in cycles; state is recorded once per cycle.
    Other parameters as in :func:`adaptive_metropolis`.  The full covariance is learned from
    the per-cycle states during burn-in (blend with ``cov0``); the block proposals are the
    conditional covariances scaled by ``exp(2 log_s_b)*2.38^2/d_b`` with ``log_s_b`` tuned per
    block to the optimal acceptance for ``d_b`` dimensions during burn-in only.

    The union of the blocks must cover every coordinate (``ValueError`` otherwise, because an
    uncovered coordinate would silently never move); ``allow_partial=True`` permits deliberately
    frozen coordinates.

    Result ``accept_blocks`` is keyed by schedule position; ``n_eval`` counts ``logpost`` calls.
    Statistics per change kind (slow/medium/fast) are available from a
    :class:`~kidspipe.inference.posterior.Posterior` passed as ``logpost``.
    """
    t0 = time.perf_counter()
    f = _as_func(logpost)
    x = np.array(x0, float).ravel()
    d = x.size
    lp = f(x)
    n_eval = 1
    if not np.isfinite(lp):
        raise ValueError("blocked_metropolis: ln P(x0) is not finite")
    sched = [(np.asarray(idx, int), int(r)) for idx, r in schedule if len(idx) > 0 and r > 0]
    nb = len(sched)
    if nb == 0:
        raise ValueError("blocked_metropolis: empty schedule")
    covered = np.zeros(d, bool)
    for idx, _ in sched:
        if idx.min() < 0 or idx.max() >= d:
            raise ValueError(f"blocked_metropolis: block index out of range for d={d}")
        covered[idx] = True
    if not allow_partial and not covered.all():
        raise ValueError("blocked_metropolis: schedule never updates coordinates "
                         f"{np.nonzero(~covered)[0].tolist()} (pass allow_partial=True to freeze them)")
    if cov0 is None:
        cov0 = default_proposal_cov(bounds) if bounds is not None else 1e-2 * np.eye(d)
    cov0 = np.asarray(cov0, float)
    rng = np.random.default_rng(seed)
    rc = _RunningCov(d)
    rc.add(x)
    C = cov0.copy()
    targets = [(_opt_accept(len(idx)) if target_accept is None else target_accept) for idx, _ in sched]
    log_s = np.array([np.log(2.38 / np.sqrt(len(idx))) for idx, _ in sched])
    t_ad = np.zeros(nb)
    Ls = [_safe_chol(_conditional_cov(C, idx)) for idx, _ in sched]
    n_tot = n_burn_cycles + n_cycles
    out_x = np.empty((n_tot, d))
    out_lp = np.empty(n_tot)
    acc = np.zeros(nb)
    prop = np.zeros(nb)
    acc_b = np.zeros(nb)
    prop_b = np.zeros(nb)
    cyc = 0
    while cyc < n_tot:
        m = min(adapt_every, n_tot - cyc)
        Z = [rng.standard_normal((m * r, len(idx))) @ Ls[b].T for b, (idx, r) in enumerate(sched)]
        LU = [np.log(rng.uniform(size=m * r)) for b, (idx, r) in enumerate(sched)]
        for k in range(m):
            burning = (cyc + k) < n_burn_cycles
            for b, (idx, r) in enumerate(sched):
                s = np.exp(log_s[b])
                for j in range(r):
                    y = x.copy()
                    y[idx] += s * Z[b][k * r + j]
                    lpy = f(y)
                    n_eval += 1
                    a = lpy - lp
                    ok = (lpy > -np.inf) and (LU[b][k * r + j] < a)
                    if ok:
                        x, lp = y, lpy
                    if burning:
                        prop_b[b] += 1
                        acc_b[b] += ok
                        t_ad[b] += 1
                        p_acc = 0.0 if lpy == -np.inf else min(1.0, np.exp(min(a, 0.0)))
                        log_s[b] += 2.0 * (p_acc - targets[b]) / (t_ad[b] + 10.0) ** 0.6
                    else:
                        prop[b] += 1
                        acc[b] += ok
            if burning:
                rc.add(x)
            out_x[cyc + k] = x
            out_lp[cyc + k] = lp
        cyc += m
        if cyc <= n_burn_cycles:
            w = cov_prior_weight
            C = (w * cov0 + rc.n * rc.cov()) / (w + rc.n)
        Ls = [_safe_chol(_conditional_cov(C, idx)) for idx, _ in sched]   # frozen after burn-in
    ab = {f"block{b}": acc[b] / max(prop[b], 1) for b in range(nb)}
    tot = acc.sum() / max(prop.sum(), 1)
    ab["all"] = tot
    for b in range(nb):
        ab[f"block{b}_burn"] = acc_b[b] / max(prop_b[b], 1)
    return ChainResult(samples=out_x[n_burn_cycles::thin].copy(), log_post=out_lp[n_burn_cycles::thin].copy(),
                       burn_samples=out_x[:n_burn_cycles].copy(), burn_log_post=out_lp[:n_burn_cycles].copy(),
                       accept=tot, cov=C, n_eval=n_eval, time=time.perf_counter() - t0,
                       accept_blocks=ab, scales={f"block{b}": float(np.exp(log_s[b])) for b in range(nb)})


# --------------------------------------------------------------------------- ensemble
@dataclass
class EnsembleResult:
    """Output of :func:`ensemble_sampler`.

    ``chain`` (n_steps, n_walkers, d), ``log_post`` (n_steps, n_walkers),
    ``acceptance_fraction`` (n_walkers,), ``n_eval`` number of ln P evaluations, ``time`` s.
    """
    chain: np.ndarray
    log_post: np.ndarray
    acceptance_fraction: np.ndarray
    n_eval: int
    time: float

    def flat(self, burn: int = 0, thin: int = 1):
        """Flattened (n*nw, d) samples and ln P after discarding ``burn`` steps."""
        c = self.chain[burn::thin]
        lp = self.log_post[burn::thin]
        return c.reshape(-1, c.shape[-1]), lp.reshape(-1)


def ensemble_sampler(logpost, p0, n_steps: int, a: float = 2.0, vectorized: bool = False,
                     seed: int = 0) -> EnsembleResult:
    """Goodman-Weare affine-invariant ensemble sampler (parallel stretch move).

    Parameters
    ----------
    logpost : callable.  ``theta (d,) -> float``, or if ``vectorized`` ``thetas (m,d) -> (m,)``.
    p0 : initial walker positions (n_walkers, d); n_walkers even and ideally >= 2 d, all with
        finite ``ln P``.
    a : stretch scale (2 is the standard choice).
    """
    t0 = time.perf_counter()
    f = _as_func(logpost)
    X = np.array(p0, float)
    nw, d = X.shape
    if nw % 2:
        raise ValueError("ensemble_sampler: n_walkers must be even")
    rng = np.random.default_rng(seed)

    def ev(Y):
        if vectorized:
            v = np.asarray(f(Y), float)
        else:
            v = np.array([f(y) for y in Y], float)
        return np.where(np.isfinite(v), v, -np.inf)

    lp = ev(X)
    n_eval = nw
    if not np.all(np.isfinite(lp)):
        raise ValueError("ensemble_sampler: all initial walkers must have finite ln P")
    half = nw // 2
    sets = (np.arange(half), np.arange(half, nw))
    chain = np.empty((n_steps, nw, d))
    lps = np.empty((n_steps, nw))
    acc = np.zeros(nw)
    for t in range(n_steps):
        for s in (0, 1):
            act, comp = sets[s], sets[1 - s]
            z = ((a - 1.0) * rng.uniform(size=half) + 1.0) ** 2 / a
            Xc = X[comp[rng.integers(len(comp), size=half)]]
            Xa = X[act]
            Y = Xc + z[:, None] * (Xa - Xc)
            lpy = ev(Y)
            n_eval += half
            lnq = (d - 1.0) * np.log(z) + lpy - lp[act]
            ok = np.log(rng.uniform(size=half)) < lnq
            ok &= np.isfinite(lpy)
            X[act[ok]] = Y[ok]
            lp[act[ok]] = lpy[ok]
            acc[act] += ok
        chain[t] = X
        lps[t] = lp
    return EnsembleResult(chain=chain, log_post=lps, acceptance_fraction=acc / n_steps,
                          n_eval=n_eval, time=time.perf_counter() - t0)


# --------------------------------------------------------------------------- diagnostics
def _as3(x):
    """Return array (n, m, d) from (n,), (n,d) [single chain] or (n,m,d)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        return x[:, None, None]
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError("expected 1-3 dimensional array")


def split_rhat(x, rank_normalize: bool = False) -> np.ndarray:
    """Split Gelman-Rubin statistic per parameter.

    Each chain is cut into two halves (so that a single chain also yields a diagnostic and
    within-chain trends are detected).  With ``m`` split chains of length ``n``,
    ``W`` the mean within-chain variance, ``B/n`` the variance of the chain means,

    .. math:: \\hat R = \\sqrt{\\frac{(n-1)/n\\,W + B/n}{W}} .

    ``rank_normalize=True`` applies the rank-normalisation of Vehtari et al. (2021) first
    (robust to heavy tails).  Input axes: (steps,) one chain / one parameter, (steps, params)
    ONE chain (a 2-D array is never read as (steps, chains)), or (steps, chains, params).
    Returns shape (d,).
    """
    x = _as3(x)
    n, m, d = x.shape
    h = n // 2
    if h < 2:
        raise ValueError("chains too short for split R-hat")
    y = np.concatenate([x[:h], x[h:2 * h]], axis=1)         # (h, 2m, d)
    if rank_normalize:
        from scipy.stats import norm, rankdata
        flat = y.reshape(-1, d)
        r = np.apply_along_axis(rankdata, 0, flat)
        z = norm.ppf((r - 0.375) / (flat.shape[0] + 0.25))
        y = z.reshape(y.shape)
    nn = y.shape[0]
    W = y.var(axis=0, ddof=1).mean(axis=0)
    Bn = y.mean(axis=0).var(axis=0, ddof=1)
    var_plus = (nn - 1.0) / nn * W + Bn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_plus / W)


def autocorr_function(x) -> np.ndarray:
    """Normalised autocorrelation function along axis 0 (FFT), averaged over chains.

    ``x`` has axes (steps[, chains][, params]); returns (steps, d) with ``rho(0) = 1``.
    """
    x = _as3(x)
    n, m, d = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(xc, n=nfft, axis=0)
    acf = np.fft.irfft(F * np.conj(F), axis=0)[:n]
    acf = acf.mean(axis=1)                      # average the autocovariance over chains
    with np.errstate(divide="ignore", invalid="ignore"):
        return acf / acf[0]


def integrated_autocorr_time(x, c: float = 5.0, tol: float = 50.0, return_flag: bool = False):
    """Integrated autocorrelation time ``tau = 1 + 2 sum_{t>=1} rho(t)`` per parameter.

    Sokal's automatic window: the smallest ``M`` with ``M >= c * tau(M)`` is used.  If no such
    window exists (chain shorter than ``~tol*tau``) the full-length estimate is returned and,
    with ``return_flag=True``, the boolean array ``reliable`` (False where ``n < tol * tau``).
    """
    rho = autocorr_function(x)
    n, d = rho.shape
    taus = 2.0 * np.cumsum(rho, axis=0) - 1.0
    tau = np.empty(d)
    rel = np.ones(d, bool)
    for i in range(d):
        ok = np.arange(n) >= c * taus[:, i]
        w = int(np.argmax(ok)) if ok.any() else n - 1
        tau[i] = taus[w, i]
        rel[i] = bool(ok.any()) and (n >= tol * tau[i])
    tau = np.maximum(tau, 1.0)
    return (tau, rel) if return_flag else tau


def effective_sample_size(x, **kw) -> np.ndarray:
    """ESS per parameter = (steps * chains) / tau; ``x`` axes (steps[, chains][, params])."""
    x3 = _as3(x)
    tau = integrated_autocorr_time(x3, **kw)
    return x3.shape[0] * x3.shape[1] / tau
