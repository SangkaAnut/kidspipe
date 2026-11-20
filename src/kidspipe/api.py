"""High-level API: build the model / likelihood / posterior and fetch derived quantities.

Typical use::

    from kidspipe import api
    post = api.build_posterior()                     # hmcode backend, fast fidelity
    lnp = post.log_post(api.preset("test"))          # float
    api.derived_for(post, theta)                     # {'sigma8':..., 'omega_m':..., 'S8':...}

Everything returned by :func:`build_posterior` is a plain :class:`kidspipe.inference.Posterior`
with three extra attributes: ``post.model`` (the :class:`CosmicShearModel`), ``post.likelihood``
(:class:`GaussianLikelihood`) and ``post.recorder`` (:class:`DerivedRecorder`).

The model returns the PURE theory vector; ``GaussianLikelihood.log_like(mu, A_c, delta_c)`` adds the
``delta_c^2`` (xi+ only) and ``A_c^2 c`` terms itself, and ``Posterior`` forwards ``theta[A_c]``,
``theta[delta_c]`` to it (docs/interface_notes/inference.md, item 8).  ``build_posterior`` asserts this
wiring (``post.like_kwargs == {'A_c': 'A_c', 'delta_c': 'delta_c'}``) so a future change that silently
fixes A_c = 1 / delta_c = 0 cannot go unnoticed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from . import params
from .params import PARAM_NAMES

__all__ = [
    "PRESETS", "preset", "theta_from", "load_theta_file", "build_model", "build_likelihood",
    "build_posterior", "DerivedRecorder", "derived_for", "log_post_with_derived",
    "derived_columns", "DERIVED_NAMES", "BLOCKS", "block_indices", "default_runs_dir",
    "random_prior_draws",
]

#: names of the derived quantities, in the order used for chain columns
DERIVED_NAMES = ("sigma8", "omega_m", "S8")

#: spec s15 anchors: test cosmology (A_baryon = 2), amplitude scan (A_baryon = 3.13)
TEST_COSMOLOGY = dict(omch2=0.1, ombh2=0.022, h=0.7, n_s=0.96, ln_1e10_A_s=3.0, A_baryon=2.0,
                      A_IA=0.0, delta_c=0.0, A_c=1.0, dz1=0.0, dz2=0.0, dz3=0.0, dz4=0.0, dz5=0.0)
AMP_SCAN_BASE = dict(TEST_COSMOLOGY, omch2=0.12, ombh2=0.0222, h=0.7, n_s=0.97, A_baryon=3.13)
#: spec anchors: chi2 (A_c=1, delta_c=0, m-correction on)
ANCHOR_TEST_CHI2 = 196.06
ANCHOR_SCAN = {2.6: 192.39, 2.8: 188.77, 3.0: 199.43, 3.2: 239.69}

PRESETS = {
    "test": TEST_COSMOLOGY,
    "scan": dict(AMP_SCAN_BASE, ln_1e10_A_s=2.8),
    "start": {n: float(v) for n, v in zip(PARAM_NAMES, params.START)},
}

#: parameter blocks by re-evaluation cost
BLOCKS = {
    "slow": ("omch2", "ombh2", "h"),            # new CAMB transfer function (~0.5 s)
    "medium": ("n_s", "ln_1e10_A_s", "A_baryon"),   # exact rescale of the cached transfer fn + new P_nl (~15 ms)
    "fast": ("A_IA", "delta_c", "A_c", "dz1", "dz2", "dz3", "dz4", "dz5"),
}


def block_indices(kind: str) -> np.ndarray:
    """Indices (into the 14-vector) of the ``slow``/``medium``/``fast`` parameter block."""
    return np.array([PARAM_NAMES.index(n) for n in BLOCKS[kind]], dtype=int)


def default_runs_dir() -> Path:
    """``$KIDSPIPE_RUNS`` if set, else ``<repo>/runs`` (see :mod:`kidspipe.paths`)."""
    from . import paths
    return paths.runs_dir()


# ----------------------------------------------------------------------------------------------
# parameter vectors
# ----------------------------------------------------------------------------------------------
def preset(name: str = "start") -> np.ndarray:
    """Parameter vector (14,) of a named preset: ``test`` (spec s4.1 cosmology, A_baryon = 2),
    ``scan`` (spec s4.2 base at ln_1e10_A_s = 2.8, A_baryon = 3.13), ``start`` (``params.START``)."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return params.from_dict(PRESETS[name], default=params.START)


def theta_from(values: Optional[Mapping[str, float]] = None, base="start") -> np.ndarray:
    """Parameter vector from a ``{name: value}`` mapping over ``base`` (preset name or vector)."""
    b = preset(base) if isinstance(base, str) else np.asarray(base, dtype=np.float64)
    out = b.copy()
    for k, v in (values or {}).items():
        if k not in PARAM_NAMES:
            raise KeyError(f"unknown parameter {k!r}; valid: {', '.join(PARAM_NAMES)}")
        out[PARAM_NAMES.index(k)] = float(v)
    return out


def load_theta_file(path, base="start") -> np.ndarray:
    """Read parameter values from a JSON object (``{"omch2": 0.12, ...}``; a list of 14 numbers is also
    accepted; the MAP file written by ``kidspipe map`` has a ``"theta"`` object) or an ini-like text file
    (``name = value`` lines, ``#``/``;`` comments).  Missing names are taken from ``base``."""
    txt = Path(path).read_text()
    s = txt.lstrip()
    if s.startswith("{") or s.startswith("["):
        obj = json.loads(txt)
        if isinstance(obj, list):
            arr = np.asarray(obj, dtype=np.float64)
            if arr.shape != (len(PARAM_NAMES),):
                raise ValueError(f"{path}: expected a list of {len(PARAM_NAMES)} numbers")
            return arr
        if isinstance(obj.get("theta"), dict):      # `kidspipe map` output
            obj = obj["theta"]
        return theta_from(obj, base)
    vals = {}
    for ln in txt.splitlines():
        ln = ln.split("#", 1)[0].split(";", 1)[0].strip()
        if not ln or ln.startswith("["):
            continue
        if "=" not in ln:
            raise ValueError(f"{path}: cannot parse line {ln!r} (expected 'name = value')")
        k, v = ln.split("=", 1)
        vals[k.strip()] = float(v.strip().split()[0])
    return theta_from(vals, base)


def random_prior_draws(n: int, seed: int = 0, box: float = 1.0) -> np.ndarray:
    """``n`` draws from the prior through ``params.prior_transform``; ``box < 1`` shrinks the unit-cube
    sampling to the central fraction ``box`` of every marginal (moderate box = fewer extreme cosmologies)."""
    rng = np.random.default_rng(seed)
    u = 0.5 + box * (rng.uniform(size=(n, params.N_PARAM)) - 0.5)
    return params.prior_transform(u)


# ----------------------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------------------
def build_model(**kw):
    """Construct a :class:`kidspipe.theory.model.CosmicShearModel`.

    Keywords are passed through: ``fidelity`` ('fast'|'kcap'), ``pnl_backend`` ('hmcode'|'camb'),
    ``z_grid`` ('spec'|'reduced'), ``n_ell``, ``n_nodes``, ``nthreads``, ``cache_size``, ``data_dir``.
    """
    from .theory.model import CosmicShearModel
    return CosmicShearModel(**kw)


def build_likelihood(apply_m: bool = True, use_cterm: bool = True, m=None, data=None):
    """Construct the :class:`GaussianLikelihood` (KV450 data, cut covariance, m-bias, c-term)."""
    from .likelihood.gauss import GaussianLikelihood
    return GaussianLikelihood(data=data, m=m, apply_m=apply_m, use_cterm=use_cterm)


class DerivedRecorder:
    """Callable model wrapper: ``theta -> mu`` that records sigma8, Omega_m, S8 for every cosmology evaluated.

    These derived quantities depend only on the first five parameters (omch2, ombh2, h, n_s, ln_1e10_A_s),
    so they are stored under that 5-tuple.  This makes chain post-processing free: every accepted point of a
    chain has been evaluated once, hence recorded.  ``lookup`` falls back to the linear backend (CAMB
    transfer function, cached/rescaled when possible) for points that were never evaluated.
    """

    N_KEY = 5

    def __init__(self, model, max_entries: int = 2_000_000):
        self.model = model
        self.max_entries = int(max_entries)
        self.table: dict = {}
        self.last_error: Optional[BaseException] = None

    def mu(self, theta):
        mu = self.model.mu(theta)
        if len(self.table) < self.max_entries:
            key = self._key(theta)
            if key not in self.table:
                d = self.model.last_derived
                self.table[key] = (d["sigma8"], d["omega_m"], d["S8"])
        return mu

    __call__ = mu

    @staticmethod
    def _key(theta):
        t = np.asarray(theta, dtype=np.float64)
        return tuple(float(x) for x in t[: DerivedRecorder.N_KEY])

    def lookup(self, theta) -> dict:
        """Derived quantities at theta as ``{'sigma8','omega_m','S8'}``; NaN if the point is invalid."""
        key = self._key(theta)
        v = self.table.get(key)
        if v is None:
            try:
                lin = self.model.linear.compute(*key)
                v = (lin.sigma8_0, lin.omega_m, lin.S8)
                if len(self.table) < self.max_entries:
                    self.table[key] = v
            except Exception as e:      # out-of-range theta etc.
                self.last_error = e
                v = (math.nan, math.nan, math.nan)
        return dict(zip(DERIVED_NAMES, (float(x) for x in v)))

    def columns(self, thetas) -> np.ndarray:
        """(n, 3) array of (sigma8, omega_m, S8) for the rows of ``thetas``."""
        th = np.atleast_2d(np.asarray(thetas, dtype=np.float64))
        out = np.empty((len(th), len(DERIVED_NAMES)))
        cache = {}
        for i, t in enumerate(th):
            key = self._key(t)
            if key not in cache:
                cache[key] = [self.lookup(t)[n] for n in DERIVED_NAMES]
            out[i] = cache[key]
        return out


def build_posterior(model=None, likelihood=None, *, memo: bool = True, record_derived: bool = True, **model_kw):
    """Wire model + likelihood + params into a :class:`kidspipe.inference.Posterior`.

    Parameters
    ----------
    model : CosmicShearModel, optional (default: ``build_model(**model_kw)``)
    likelihood : GaussianLikelihood, optional (default: ``build_likelihood()``)
    record_derived : wrap the model in a :class:`DerivedRecorder` (cheap: one dict insert per new cosmology).

    Returns
    -------
    Posterior with attributes ``model``, ``likelihood``, ``recorder`` (or None) and ``blocks`` =
    FAST (A_IA, delta_c, A_c, dz1..5) / MEDIUM (n_s, ln_1e10_A_s, A_baryon) / SLOW (omch2, ombh2, h);
    (identical to ``inference.SLOW_PARAMS`` / ``MEDIUM_PARAMS`` / ``FAST_PARAMS``; n_s is an exact rescale, see ``BLOCKS``).
    """
    from .inference.posterior import Posterior, ParamBlocks
    if model is None:
        model = build_model(**model_kw)
    elif model_kw:
        raise TypeError("pass either a model instance or model keywords, not both")
    if likelihood is None:
        likelihood = build_likelihood()
    rec = DerivedRecorder(model) if record_derived else None
    blocks = ParamBlocks.from_names(PARAM_NAMES, slow=BLOCKS["slow"], medium=BLOCKS["medium"], fast=BLOCKS["fast"])
    post = Posterior(rec if rec is not None else model, likelihood, params, blocks=blocks, memo=memo)
    expected = {"A_c": "A_c", "delta_c": "delta_c"}
    if post.like_kwargs != expected:        # pragma: no cover - defensive wiring check
        raise RuntimeError(f"likelihood does not receive A_c/delta_c from theta (got {post.like_kwargs})")
    post.model, post.likelihood, post.recorder = model, likelihood, rec
    return post


# ----------------------------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------------------------
def derived_for(post, theta) -> dict:
    """``{'sigma8','omega_m','S8'}`` at theta (uses the recorder table; computes via CAMB if unseen)."""
    rec = getattr(post, "recorder", None)
    if rec is None:
        rec = DerivedRecorder(post.model)
        post.recorder = rec
    return rec.lookup(theta)


def log_post_with_derived(post, theta):
    """``(ln post, ln like, {'sigma8','omega_m','S8'})`` at theta; derived entries are NaN if ln post = -inf."""
    lp, ll = post.eval(theta)
    d = derived_for(post, theta) if np.isfinite(lp) else {n: math.nan for n in DERIVED_NAMES}
    return lp, ll, d


def derived_columns(post, samples) -> np.ndarray:
    """(n, 3) derived columns (sigma8, omega_m, S8) for chain samples; see :meth:`DerivedRecorder.columns`."""
    rec = getattr(post, "recorder", None)
    if rec is None:
        rec = DerivedRecorder(post.model)
        post.recorder = rec
    return rec.columns(samples)
