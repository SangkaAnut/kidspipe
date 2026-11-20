"""Nested-sampling wrappers (nautilus 1.0.5 and dynesty 2.1.4).

Both wrappers take the same two callables

* ``prior_transform(u) -> theta``: unit hypercube -> parameters through the exact inverse prior
  CDFs (``kidspipe.params.prior_transform``; a normalised prior, so the evidence is the
  properly normalised ``Z = int L(theta) pi(theta) dtheta``),
* ``loglike(theta) -> ln L`` (scalar; ``-inf`` or NaN are both treated as zero likelihood),

and return a :class:`NestedResult` with weighted posterior samples and ``ln Z``.  The
posterior weights are normalised to sum to one; the log-weights are those of the
sampler (``ln w_i = ln L_i + ln(prior-volume shell)_i - ln Z`` up to the sampler's own
bookkeeping).  Heavy imports are done lazily so that the module can be imported without
the samplers installed.

nautilus does not report an evidence error; ``log_z_err`` is ``nan`` there.  dynesty's
``logzerr`` is the usual information-based estimate (known to be optimistic in high
dimensions).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = ["NestedResult", "run_nautilus", "run_dynesty", "safe_loglike"]

_NEG = -1.0e300


@dataclass
class NestedResult:
    """Weighted samples from a nested-sampling run.

    Attributes
    ----------
    samples : (n, d) parameter values; weights : (n,) normalised posterior weights (sum 1);
    log_like : (n,) ln L at the samples; log_z, log_z_err : evidence and its error estimate;
    n_eval : number of likelihood calls; time : wall time (s); sampler : 'nautilus'|'dynesty';
    ess : Kish effective sample size ``1/sum w^2``; info : sampler-specific extras.
    """
    samples: np.ndarray
    weights: np.ndarray
    log_like: np.ndarray
    log_z: float
    log_z_err: float
    n_eval: int
    time: float
    sampler: str
    info: dict = field(default_factory=dict)

    @property
    def ess(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def log_post(self, log_prior: Callable) -> np.ndarray:
        """ln pi + ln L at the samples, as needed by :func:`~kidspipe.inference.chains.pjhpd_extents`.

        ``log_like`` alone is the posterior density only for a flat prior; with the truncated
        Gaussian priors of ``kidspipe.params`` pass ``log_prior=params.log_prior`` (or
        ``Posterior.log_prior``).
        """
        return self.log_like + np.array([log_prior(t) for t in self.samples], float)

    def equal_weight(self, n: Optional[int] = None, seed: int = 0) -> np.ndarray:
        """Resample to an equal-weight sample set (systematic resampling), shape (n, d)."""
        n = len(self.weights) if n is None else n
        rng = np.random.default_rng(seed)
        pos = (rng.uniform() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(self.weights), pos)
        return self.samples[np.minimum(idx, len(self.weights) - 1)]


class _SafeLoglike:
    """Picklable wrapper (a closure cannot be sent to multiprocessing workers, e.g. a dynesty ``pool``)."""

    def __init__(self, loglike: Callable):
        self.loglike = loglike

    def __call__(self, theta):
        try:
            v = float(self.loglike(theta))
        except Exception:  # noqa: BLE001
            return _NEG
        return v if np.isfinite(v) else _NEG


def safe_loglike(loglike: Callable) -> Callable:
    """Wrap ``loglike`` so exceptions / NaN / +-inf return a very negative finite number.

    The returned object is picklable whenever ``loglike`` is (needed for dynesty with a process pool)."""
    return _SafeLoglike(loglike)


def _norm_weights(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


def run_nautilus(loglike: Callable, prior_transform: Callable, ndim: int, n_live: int = 1000,
                 n_eff: float = 5000, f_live: float = 0.01, n_networks: int = 4,
                 vectorized: bool = False, prior_vectorized: bool = False, seed: Optional[int] = None,
                 pool=None, filepath: Optional[str] = None, resume: bool = False,
                 timeout: float = np.inf, n_like_max: float = np.inf, verbose: bool = False,
                 discard_exploration: bool = False, **sampler_kwargs) -> NestedResult:
    """Run nautilus (neural-network-boosted importance nested sampling).

    Parameters
    ----------
    loglike : ``theta (d,) -> ln L`` or, with ``vectorized=True``, ``thetas (m,d) -> (m,)``.
    prior_transform : ``u (d,) -> theta``; if ``prior_vectorized`` it must accept ``(m,d)`` arrays
        (nautilus passes batches when the sampler is vectorised; an unvectorised transform
        is applied row by row here).
    n_live : live points per bound (default 1000; the nautilus default is 2000, ~d x 100 is
        plenty for smooth unimodal posteriors).
    n_eff : target effective sample size of the posterior (nautilus stops once reached).
    f_live : evidence fraction in the live set at which the exploration phase stops.
    Remaining keyword arguments go to ``nautilus.Sampler``.  ``filepath``/``resume`` give HDF5
    checkpointing.  Returns :class:`NestedResult` (``log_z_err = nan``).
    """
    import nautilus  # lazy

    ll = safe_loglike(loglike) if not vectorized else (
        lambda X: np.nan_to_num(np.asarray(loglike(X), float), nan=_NEG, posinf=_NEG, neginf=_NEG))

    def prior(u):
        u = np.asarray(u, float)
        if u.ndim == 1 or prior_vectorized:
            return np.asarray(prior_transform(u), float)
        return np.array([prior_transform(r) for r in u])

    t0 = time.perf_counter()
    s = nautilus.Sampler(prior, ll, n_dim=ndim, n_live=n_live, n_networks=n_networks,
                         vectorized=vectorized, seed=seed, pool=pool,
                         filepath=filepath, resume=resume, **sampler_kwargs)
    ok = s.run(f_live=f_live, n_eff=n_eff, n_like_max=n_like_max, timeout=timeout,
               discard_exploration=discard_exploration, verbose=verbose)
    pts, logw, logl = s.posterior()
    return NestedResult(samples=np.asarray(pts, float), weights=_norm_weights(np.asarray(logw)),
                        log_like=np.asarray(logl, float), log_z=float(s.log_z), log_z_err=float("nan"),
                        n_eval=int(s.n_like), time=time.perf_counter() - t0, sampler="nautilus",
                        info={"success": bool(ok), "n_eff": float(s.n_eff)})


def run_dynesty(loglike: Callable, prior_transform: Callable, ndim: int, nlive: int = 500,
                dynamic: bool = False, bound: str = "multi", sample: str = "auto",
                dlogz: float = 0.1, seed: Optional[int] = None, pool=None, queue_size=None,
                maxiter: Optional[int] = None, maxcall: Optional[int] = None,
                nlive_batch: int = 250, maxbatch: Optional[int] = None,
                wt_kwargs: Optional[dict] = None, checkpoint_file: Optional[str] = None,
                resume: bool = False, verbose: bool = False, **sampler_kwargs) -> NestedResult:
    """Run dynesty (static or dynamic nested sampling).

    Parameters
    ----------
    nlive : live points (static), or initial live points (dynamic).
    dynamic : if True use ``DynamicNestedSampler`` with posterior-weighted batches
        (``wt_kwargs={'pfrac': 1.0}`` by default, i.e. optimise for the posterior, not the evidence).
    bound, sample : dynesty bounding / sampling methods; ``sample='auto'`` picks 'unif' for d<10,
        'rwalk' for 10<=d<=20 and 'rslice' above.
    dlogz : evidence-remaining stopping tolerance.
    Remaining keyword arguments go to the sampler constructor.  Returns :class:`NestedResult`
    with ``log_z_err`` = dynesty's ``logzerr[-1]``.
    """
    import dynesty  # lazy

    rstate = np.random.default_rng(seed)
    ll = safe_loglike(loglike)
    t0 = time.perf_counter()
    common = dict(bound=bound, sample=sample, rstate=rstate, pool=pool, queue_size=queue_size, **sampler_kwargs)
    if dynamic:
        s = dynesty.DynamicNestedSampler(ll, prior_transform, ndim, nlive=nlive, **common)
        wt = {"pfrac": 1.0} if wt_kwargs is None else wt_kwargs
        s.run_nested(dlogz_init=dlogz, nlive_init=nlive, nlive_batch=nlive_batch, maxiter=maxiter,
                     maxcall=maxcall, maxbatch=maxbatch, wt_kwargs=wt, print_progress=verbose,
                     checkpoint_file=checkpoint_file, resume=resume)
    else:
        s = dynesty.NestedSampler(ll, prior_transform, ndim, nlive=nlive, **common)
        s.run_nested(dlogz=dlogz, maxiter=maxiter, maxcall=maxcall, print_progress=verbose,
                     checkpoint_file=checkpoint_file, resume=resume)
    r = s.results
    w = np.asarray(r.importance_weights(), float)
    w = w / w.sum()
    keep = w > 0
    return NestedResult(samples=np.asarray(r.samples, float)[keep], weights=w[keep],
                        log_like=np.asarray(r.logl, float)[keep], log_z=float(r.logz[-1]),
                        log_z_err=float(r.logzerr[-1]), n_eval=int(np.sum(r.ncall)),
                        time=time.perf_counter() - t0, sampler="dynesty-dynamic" if dynamic else "dynesty",
                        info={"niter": int(r.niter)})


def make_nested_functions(post):
    """Return ``(loglike, prior_transform, ndim)`` from a :class:`Posterior` for the wrappers."""
    return post.log_like, post.prior_transform, post.ndim
