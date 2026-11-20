"""Abstract posterior for the kidspipe inference layer.

The samplers in this package never see the cosmological model.  They only see a
:class:`Posterior`, which glues together (spec s1.3, s12)

* a *model* ``model_mu(theta) -> mu`` (e.g. ``CosmicShearModel.mu``),
* a *likelihood* with ``log_like(mu) -> float`` (e.g. ``GaussianLikelihood``) and
* a *parameter module* (``kidspipe.params``-like) providing ``PARAM_NAMES``,
  ``BOUNDS``, ``START``, ``log_prior(theta)`` and ``prior_transform(u)``,

into the functions

.. math::  \\ln P(\\theta) = \\ln\\pi(\\theta) + \\ln L(\\mu(\\theta)),\\qquad
           \\ln L = -\\chi^2/2 ,

with the conventions of spec s1.3: ``-inf`` outside the prior support, and any
exception or non-finite value coming out of the model/likelihood is mapped to
``ln L = -inf`` ("a failing module gives ln L = -inf").

Parameter blocks
----------------
For blocked Metropolis sampling the 14 parameters are split by the cost of
re-evaluating the model after changing them (DESIGN.md stage caches):

* ``FAST_PARAMS``   : A_IA, delta_c, A_c, dz1..dz5 (never touch the Boltzmann
  code or the non-linear P(k); only the cheap projection / likelihood stage),
* ``MEDIUM_PARAMS`` : n_s, ln_1e10_A_s, A_baryon (exact rescale of the cached P_lin, new HMcode P_nl),
* ``SLOW_PARAMS``   : omch2, ombh2, h (new CAMB transfer functions).

These are identical to ``kidspipe.api.BLOCKS`` (n_s is an exact rescale of the cached transfer function).

Everything is float64 numpy; theta is the 14-vector in ``PARAM_NAMES`` order
(units: those of each parameter, dimensionless except where stated).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

FAST_PARAMS = ("A_IA", "delta_c", "A_c", "dz1", "dz2", "dz3", "dz4", "dz5")
MEDIUM_PARAMS = ("n_s", "ln_1e10_A_s", "A_baryon")
SLOW_PARAMS = ("omch2", "ombh2", "h")
DEFAULT_PARAM_NAMES = SLOW_PARAMS + MEDIUM_PARAMS + FAST_PARAMS
AUTO_LIKE_KWARGS = ("A_c", "delta_c")   # parameters forwarded to the likelihood when it accepts them

__all__ = [
    "FAST_PARAMS", "MEDIUM_PARAMS", "SLOW_PARAMS", "DEFAULT_PARAM_NAMES", "AUTO_LIKE_KWARGS",
    "ParamBlocks", "SimpleParams", "Posterior",
]


# --------------------------------------------------------------------------- blocks
@dataclass(frozen=True)
class ParamBlocks:
    """Index lists of the slow / medium / fast parameter blocks.

    Attributes are integer numpy arrays indexing into the theta vector.  The
    three blocks need not cover theta (a toy problem may leave some unused) but
    must be disjoint.
    """

    slow: np.ndarray
    medium: np.ndarray
    fast: np.ndarray

    @classmethod
    def from_names(cls, names: Sequence[str],
                   slow: Sequence[str] = SLOW_PARAMS,
                   medium: Sequence[str] = MEDIUM_PARAMS,
                   fast: Sequence[str] = FAST_PARAMS) -> "ParamBlocks":
        """Resolve block name lists against ``names`` (names absent from ``names`` are skipped)."""
        idx = {n: i for i, n in enumerate(names)}
        mk = lambda lst: np.array([idx[n] for n in lst if n in idx], dtype=int)
        return cls(mk(slow), mk(medium), mk(fast))

    @classmethod
    def from_indices(cls, slow=(), medium=(), fast=()) -> "ParamBlocks":
        """Explicit index lists (for toy problems)."""
        return cls(np.asarray(slow, int), np.asarray(medium, int), np.asarray(fast, int))

    def as_dict(self) -> dict:
        return {"slow": self.slow, "medium": self.medium, "fast": self.fast}

    def classify_change(self, a: np.ndarray, b: np.ndarray) -> str:
        """Most expensive block in which ``a`` and ``b`` differ: 'slow' | 'medium' | 'fast' | 'same' | 'other'."""
        if a is None or b is None:
            return "slow"
        diff = a != b
        if not diff.any():
            return "same"
        if diff[self.slow].any():
            return "slow"
        if diff[self.medium].any():
            return "medium"
        if diff[self.fast].any():
            return "fast"
        return "other"


# --------------------------------------------------------------------------- toy params
class SimpleParams:
    """Minimal parameter module (box prior, optional Gaussian factors) for toys and tests.

    Mimics the interface of ``kidspipe.params``: ``PARAM_NAMES``, ``BOUNDS`` (n,2),
    ``START`` (n,), ``log_prior(theta)`` and ``prior_transform(u)``.  The prior is
    flat on ``BOUNDS`` unless ``gauss_mean``/``gauss_sigma`` (arrays with NaN for
    "flat") are given, in which case it is a truncated normal on those entries
    (spec s1.3, the normalisation constants are included so evidences are right).
    """

    def __init__(self, names, bounds, start=None, gauss_mean=None, gauss_sigma=None):
        from scipy.stats import norm
        self.PARAM_NAMES = tuple(names)
        self.BOUNDS = np.asarray(bounds, float).reshape(-1, 2)
        n = len(self.PARAM_NAMES)
        self.START = (self.BOUNDS.mean(axis=1) if start is None else np.asarray(start, float))
        self._gm = np.full(n, np.nan) if gauss_mean is None else np.asarray(gauss_mean, float)
        self._gs = np.full(n, np.nan) if gauss_sigma is None else np.asarray(gauss_sigma, float)
        self._isg = np.isfinite(self._gs)
        lo, hi = self.BOUNDS[:, 0], self.BOUNDS[:, 1]
        self._norm = norm
        a = np.where(self._isg, (lo - self._gm) / np.where(self._isg, self._gs, 1.0), 0.0)
        b = np.where(self._isg, (hi - self._gm) / np.where(self._isg, self._gs, 1.0), 1.0)
        self._cdf_a, self._cdf_b = norm.cdf(a), norm.cdf(b)
        self._lognorm = np.where(
            self._isg,
            np.log(np.where(self._isg, self._cdf_b - self._cdf_a, 1.0))
            + 0.5 * np.log(2 * np.pi * np.where(self._isg, self._gs, 1.0) ** 2),
            np.log(hi - lo))

    def log_prior(self, theta) -> float:
        """ln pi(theta): flat -ln(hi-lo) or truncated normal (spec s1.3); -inf outside bounds."""
        t = np.asarray(theta, float)
        lo, hi = self.BOUNDS[:, 0], self.BOUNDS[:, 1]
        if np.any(t < lo) or np.any(t > hi) or not np.all(np.isfinite(t)):
            return -np.inf
        g = np.where(self._isg, -0.5 * ((t - self._gm) / np.where(self._isg, self._gs, 1.0)) ** 2, 0.0)
        return float(np.sum(g - self._lognorm))

    def prior_transform(self, u) -> np.ndarray:
        """Unit cube -> parameters through the exact inverse CDFs (vectorised over leading axes)."""
        u = np.asarray(u, float)
        lo, hi = self.BOUNDS[:, 0], self.BOUNDS[:, 1]
        flat = lo + u * (hi - lo)
        if not self._isg.any():
            return flat
        p = self._cdf_a + u * (self._cdf_b - self._cdf_a)
        gs = self._gm + self._gs * self._norm.ppf(np.clip(p, 1e-300, 1 - 1e-16))
        return np.where(self._isg, np.clip(gs, lo, hi), flat)


# --------------------------------------------------------------------------- posterior
@dataclass
class _Stats:
    n_prior: int = 0
    n_prior_reject: int = 0
    n_like: int = 0
    n_fail: int = 0          # exceptions or non-finite values from model/likelihood
    n_memo_hits: int = 0
    t_prior: float = 0.0
    t_model: float = 0.0
    t_like: float = 0.0
    by_kind_n: dict = field(default_factory=lambda: {k: 0 for k in ("slow", "medium", "fast", "same", "other")})
    by_kind_t: dict = field(default_factory=lambda: {k: 0.0 for k in ("slow", "medium", "fast", "same", "other")})


class Posterior:
    """ln posterior = ln prior + ln likelihood(model(theta)).

    Parameters
    ----------
    model_mu : callable or object with ``.mu``
        ``theta (14,) -> mu (195,)`` (data-ordered model vector).
    likelihood : callable or object with ``.log_like``
        ``mu -> ln L`` (float).
    params_module : module/object, optional
        Provides ``PARAM_NAMES``, ``BOUNDS``, ``log_prior``, ``prior_transform``
        (and optionally ``START``).  Default: ``kidspipe.params`` (imported lazily).
    names, bounds, log_prior, prior_transform : optional
        Overrides / replacements when no parameter module is given.  Without a
        ``log_prior`` callable a flat prior on ``bounds`` is used.
    blocks : ParamBlocks, optional
        Slow/medium/fast index lists.  Default: from parameter names via
        :data:`FAST_PARAMS`, :data:`MEDIUM_PARAMS`, :data:`SLOW_PARAMS`.
    memo : bool
        Remember the last evaluated point (exact equality) and return the
        stored result instead of recomputing (counted in ``stats.n_memo_hits``).
    like_kwargs : dict, optional
        ``{likelihood keyword: parameter name}`` forwarded from theta to the likelihood,
        ``likelihood(mu, **{kw: theta[names.index(pname)]})``.  Default (``None``):
        detected automatically -- every keyword among ``A_c``, ``delta_c`` that is an
        explicit argument of the likelihood callable *and* a parameter name is bound.
        This matters for ``kidspipe.likelihood.GaussianLikelihood.log_like(mu, A_c, delta_c)``,
        which takes the PURE theory vector and adds the c-term / delta_c^2 itself (spec s9):
        without the binding A_c and delta_c would silently be fixed at 1 and 0.
        Pass ``{}`` to disable.  The mapping actually used is ``self.like_kwargs``.

    Notes
    -----
    Counters and wall-time accounting live in :attr:`stats`; evaluations are
    classified by the most expensive block that changed with respect to the
    previous *model* evaluation (``stats.by_kind_n/t``), which is what the
    fast/slow oversampling needs to be judged on.  Instances are callable:
    ``post(theta) == post.log_post(theta)``.
    """

    def __init__(self, model_mu, likelihood, params_module=None, *, names=None, bounds=None,
                 log_prior: Optional[Callable] = None, prior_transform: Optional[Callable] = None,
                 start=None, blocks: Optional[ParamBlocks] = None, memo: bool = True,
                 like_kwargs: Optional[dict] = None):
        self._model = model_mu.mu if hasattr(model_mu, "mu") else model_mu
        self._like = likelihood.log_like if hasattr(likelihood, "log_like") else likelihood
        pm = params_module
        if pm is None and names is None and bounds is None:
            import importlib
            pm = importlib.import_module("kidspipe.params")
        self.params_module = pm
        if names is None:
            names = getattr(pm, "PARAM_NAMES", None)
        if bounds is None:
            bounds = getattr(pm, "BOUNDS", None)
        if bounds is None:
            raise ValueError("Posterior needs bounds (params_module.BOUNDS or bounds=)")
        self.bounds = np.asarray(bounds, float).reshape(-1, 2)
        self.ndim = self.bounds.shape[0]
        self.names = tuple(names) if names is not None else tuple(f"p{i}" for i in range(self.ndim))
        st = start if start is not None else getattr(pm, "START", None)
        self.start = np.asarray(st, float) if st is not None else self.bounds.mean(axis=1)
        self._lp = log_prior if log_prior is not None else getattr(pm, "log_prior", None)
        self._pt = prior_transform if prior_transform is not None else getattr(pm, "prior_transform", None)
        self.blocks = blocks if blocks is not None else ParamBlocks.from_names(self.names)
        self.memo = bool(memo)
        self.like_kwargs = self._resolve_like_kwargs(like_kwargs)
        self._like_idx = {k: self.names.index(v) for k, v in self.like_kwargs.items()}
        self.stats = _Stats()
        self._last_theta = None      # last theta sent to the model
        self._memo_theta = None
        self._memo_val = None
        self.last_error: Optional[BaseException] = None

    def _resolve_like_kwargs(self, like_kwargs) -> dict:
        """Mapping likelihood keyword -> parameter name (see class docstring)."""
        if like_kwargs is not None:
            lk = dict(like_kwargs)
            for k, v in lk.items():
                if v not in self.names:
                    raise ValueError(f"like_kwargs: parameter {v!r} (for keyword {k!r}) not in names")
            return lk
        import inspect
        try:
            sig = inspect.signature(self._like)
        except (TypeError, ValueError):
            return {}
        return {k: k for k in AUTO_LIKE_KWARGS
                if k in sig.parameters and k in self.names
                and sig.parameters[k].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                               inspect.Parameter.KEYWORD_ONLY)}

    # ----------------------------------------------------------------- prior
    def in_bounds(self, theta) -> bool:
        """True if lo <= theta <= hi for all components (ValueError for a wrong length)."""
        t = np.asarray(theta, float)
        if t.shape != (self.ndim,):
            raise ValueError(f"theta must have shape ({self.ndim},), got {t.shape}")
        return bool(np.all(t >= self.bounds[:, 0]) and np.all(t <= self.bounds[:, 1]) and np.all(np.isfinite(t)))

    def log_prior(self, theta) -> float:
        """ln pi(theta) (spec s1.3); -inf outside the bounds or if the prior returns NaN."""
        t0 = time.perf_counter()
        t = np.asarray(theta, float)
        self.stats.n_prior += 1
        if not self.in_bounds(t):
            v = -np.inf
        elif self._lp is None:
            v = float(-np.sum(np.log(self.bounds[:, 1] - self.bounds[:, 0])))
        else:
            try:
                v = float(self._lp(t))
            except Exception as e:  # noqa: BLE001
                self.last_error = e
                v = -np.inf
            if not np.isfinite(v):
                v = -np.inf
        if v == -np.inf:
            self.stats.n_prior_reject += 1
        self.stats.t_prior += time.perf_counter() - t0
        return v

    def prior_transform(self, u) -> np.ndarray:
        """Unit cube -> theta (exact inverse prior CDF); flat-box fallback."""
        if self._pt is not None:
            return np.asarray(self._pt(np.asarray(u, float)), float)
        u = np.asarray(u, float)
        return self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])

    # ----------------------------------------------------------------- likelihood
    def _eval_like(self, t: np.ndarray) -> float:
        kind = self.blocks.classify_change(self._last_theta, t)
        t0 = time.perf_counter()
        ll = -np.inf
        try:
            mu = self._model(t.copy())
            t1 = time.perf_counter()
            self.stats.t_model += t1 - t0
            kw = {k: float(t[i]) for k, i in self._like_idx.items()}
            ll = float(self._like(mu, **kw))
            self.stats.t_like += time.perf_counter() - t1
            if not np.isfinite(ll):
                self.stats.n_fail += 1
                ll = -np.inf
        except Exception as e:  # noqa: BLE001  (spec s1.3: failing module -> -inf)
            self.last_error = e
            self.stats.n_fail += 1
            ll = -np.inf
        self._last_theta = t.copy()
        self.stats.n_like += 1
        self.stats.by_kind_n[kind] += 1
        self.stats.by_kind_t[kind] += time.perf_counter() - t0
        return ll

    def eval(self, theta):
        """Return ``(ln post, ln like)`` at theta (``(-inf, -inf)`` if out of the prior support)."""
        t = np.array(theta, float).ravel()
        if self.memo and self._memo_theta is not None and np.array_equal(t, self._memo_theta):
            self.stats.n_memo_hits += 1
            return self._memo_val
        lp = self.log_prior(t)
        if lp == -np.inf:
            res = (-np.inf, -np.inf)
        else:
            ll = self._eval_like(t)
            res = (lp + ll, ll) if ll > -np.inf else (-np.inf, -np.inf)
        if self.memo:
            self._memo_theta, self._memo_val = t, res
        return res

    def log_post(self, theta) -> float:
        """ln P(theta) = ln pi + ln L; ``-inf`` outside bounds or on any model failure."""
        return self.eval(theta)[0]

    def log_like(self, theta) -> float:
        """ln L(theta) (``-inf`` outside the bounds or on model failure; prior not included)."""
        t = np.array(theta, float).ravel()
        if not self.in_bounds(t):
            return -np.inf
        if self.memo and self._memo_theta is not None and np.array_equal(t, self._memo_theta):
            self.stats.n_memo_hits += 1
            return self._memo_val[1]
        ll = self._eval_like(t)
        return ll

    __call__ = log_post

    def neg_log_post(self, theta) -> float:
        """-ln P (``+inf`` if invalid)."""
        return -self.log_post(theta)

    def log_post_batch(self, thetas) -> np.ndarray:
        """ln P for each row of ``thetas`` (n, ndim) (a plain loop; the model is not vectorised)."""
        th = np.atleast_2d(np.asarray(thetas, float))
        return np.array([self.log_post(t) for t in th])

    # ----------------------------------------------------------------- bookkeeping
    def reset_stats(self):
        """Zero all counters and timers and forget the memo."""
        self.stats = _Stats()
        self._memo_theta = self._memo_val = self._last_theta = None

    def timing_report(self) -> dict:
        """Counters and wall times (s) as a plain dict, incl. mean ms per model call by change kind."""
        s = self.stats
        per = {k: (1e3 * s.by_kind_t[k] / s.by_kind_n[k] if s.by_kind_n[k] else float("nan"))
               for k in s.by_kind_n}
        return {"n_prior": s.n_prior, "n_prior_reject": s.n_prior_reject, "n_like": s.n_like,
                "n_fail": s.n_fail, "n_memo_hits": s.n_memo_hits,
                "t_prior": s.t_prior, "t_model": s.t_model, "t_like": s.t_like,
                "by_kind_n": dict(s.by_kind_n), "by_kind_t": dict(s.by_kind_t),
                "ms_per_call_by_kind": per}

    @property
    def n_evals(self) -> int:
        """Number of model+likelihood evaluations performed."""
        return self.stats.n_like

    def random_start(self, rng, n=1, max_tries=200) -> np.ndarray:
        """Draw ``n`` finite-posterior points through the prior transform (shape (n, ndim))."""
        out = []
        tries = 0
        while len(out) < n and tries < max_tries * n:
            tries += 1
            th = self.prior_transform(rng.uniform(0.02, 0.98, self.ndim))
            if np.isfinite(self.log_post(th)):
                out.append(th)
        if len(out) < n:
            raise RuntimeError("could not find finite-posterior starting points")
        return np.array(out)
