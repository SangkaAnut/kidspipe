"""Gaussian chi2 likelihood for the KV450 xi+/- data (spec s9, s11).

Model vector (spec s9, per pair and kept theta bin k)::

    mu+[k] = B_k[xi+] + delta_c^2 + A_c^2 c+[k]        (delta_c^2 on xi+ ONLY, "montepython" branch)
    mu-[k] = B_k[xi-]             + A_c^2 c-[k]
    d_ab   = x_ab / ((1 + m_a)(1 + m_b))               (m divides the DATA; covariance untouched)

Likelihood (spec s11)::

    r = d - mu ,   chi2 = r^T C^-1 r ,   ln L = -chi2 / 2          (no ln|C|, no Hartlap)

``chi2`` is evaluated through the Cholesky factor C = L L^T: chi2 = |L^-1 r|^2 by a triangular solve
(no explicit inverse).  ``mu`` passed to the methods is the *pure theory* 195-vector B_k[xi] in data
ordering; this class adds the delta_c / A_c terms (see docs/interface_notes/params_data_likelihood.md).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .data import KV450Data, PLUS_MASK, N_DATA
from ..params import M_BIAS


class GaussianLikelihood:
    """Cholesky-based Gaussian chi2 with m-bias data correction and c-term/delta_c additions.

    Parameters
    ----------
    data : KV450Data, optional (default: ``KV450Data.load()``)
    m : (5,) multiplicative shear bias per tomographic bin (default ``params.M_BIAS``)
    apply_m : divide the data by (1+m_a)(1+m_b) (default True, spec s9)
    use_cterm : include the A_c^2 c-term (default True)
    """

    def __init__(self, data: KV450Data | None = None, m=None, apply_m: bool = True,
                 use_cterm: bool = True):
        self.data = KV450Data.load() if data is None else data
        self.m = np.asarray(M_BIAS if m is None else m, dtype=np.float64)
        self.apply_m = bool(apply_m)
        self.use_cterm = bool(use_cterm)
        raw = self.data.d
        self.d = raw / self.data.m_factor(self.m) if self.apply_m else raw.copy()
        self.cov = self.data.cov
        self._L = cholesky(self.cov, lower=True)               # C = L L^T
        self._L.flags.writeable = False
        self.d.flags.writeable = False
        self.logdet = 2.0 * np.sum(np.log(np.diag(self._L)))   # informational (not in ln L)
        self._cterm = self.data.cterm if self.use_cterm else np.zeros(N_DATA)
        self._plus = PLUS_MASK.astype(np.float64)
        self.n_calls = 0

    # ---- model assembly ----------------------------------------------------------------
    def add_systematics(self, mu, A_c=1.0, delta_c=0.0) -> np.ndarray:
        """Add delta_c^2 (xi+ entries only) and A_c^2 c-term to a pure-theory vector (spec s9).

        mu : (..., 195);  A_c, delta_c : scalars or arrays broadcastable to mu.shape[:-1].
        """
        mu = np.asarray(mu, dtype=np.float64)
        A_c = np.asarray(A_c, dtype=np.float64)[..., None]
        delta_c = np.asarray(delta_c, dtype=np.float64)[..., None]
        return mu + (delta_c ** 2) * self._plus + (A_c ** 2) * self._cterm

    def residual(self, mu, A_c=1.0, delta_c=0.0) -> np.ndarray:
        """r = d - mu_total, shape (..., 195)."""
        return self.d - self.add_systematics(mu, A_c, delta_c)

    # ---- chi2 / likelihood -------------------------------------------------------------
    def chi2_from_residual(self, r) -> np.ndarray:
        """chi2 = |L^-1 r|^2 for r of shape (195,) or (N, 195)."""
        r = np.asarray(r, dtype=np.float64)
        if r.ndim == 1:
            y = solve_triangular(self._L, r, lower=True, check_finite=False)
            return float(y @ y)
        y = solve_triangular(self._L, r.T, lower=True, check_finite=False)   # (195, N)
        return np.einsum("ij,ij->j", y, y)

    def chi2(self, mu, A_c=1.0, delta_c=0.0) -> float:
        """chi2 of one theory vector mu (195,) (pure theory; systematics added here)."""
        self.n_calls += 1
        mu = np.asarray(mu, dtype=np.float64)
        if mu.shape != (N_DATA,):
            raise ValueError(f"mu must have shape ({N_DATA},), got {mu.shape}")
        if not np.all(np.isfinite(mu)):
            return np.inf
        with np.errstate(invalid="ignore", over="ignore"):
            c = self.chi2_from_residual(self.residual(mu, A_c, delta_c))
        return c if np.isfinite(c) else np.inf      # non-finite A_c / delta_c -> inf, never NaN

    def chi2_batch(self, mus, A_c=1.0, delta_c=0.0) -> np.ndarray:
        """chi2 for many theory vectors, mus (N, 195) -> (N,). ``A_c``/``delta_c`` scalar or (N,).

        One triangular solve with N right-hand sides (BLAS-3).  Non-finite rows give ``inf``.
        """
        mus = np.atleast_2d(np.asarray(mus, dtype=np.float64))
        if mus.shape[-1] != N_DATA:
            raise ValueError(f"mus must have last axis {N_DATA}, got {mus.shape}")
        n = mus.shape[0]
        A_c = np.broadcast_to(np.asarray(A_c, dtype=np.float64), (n,))
        delta_c = np.broadcast_to(np.asarray(delta_c, dtype=np.float64), (n,))
        self.n_calls += n
        good = np.all(np.isfinite(mus), axis=1)
        out = np.full(n, np.inf)
        if good.any():
            with np.errstate(invalid="ignore", over="ignore"):
                r = self.residual(mus[good], A_c[good], delta_c[good])
                c = self.chi2_from_residual(r)
            out[good] = np.where(np.isfinite(c), c, np.inf)   # NaN/inf A_c, delta_c -> inf
        return out

    def log_like(self, mu, A_c=1.0, delta_c=0.0) -> float:
        """ln L = -chi2/2 (spec s11); ``-inf`` if mu is non-finite."""
        c = self.chi2(mu, A_c, delta_c)
        return -0.5 * c if np.isfinite(c) else -np.inf

    def log_like_batch(self, mus, A_c=1.0, delta_c=0.0) -> np.ndarray:
        """Vectorised ln L, shape (N,)."""
        return -0.5 * self.chi2_batch(mus, A_c, delta_c)

    # ---- diagnostics -------------------------------------------------------------------
    def condition_number(self) -> float:
        """2-norm condition number of the cut covariance (spec: ~2.4e5)."""
        w = np.linalg.eigvalsh(self.cov)
        return float(w[-1] / w[0])

    def chi2_explicit_inverse(self, mu, A_c=1.0, delta_c=0.0) -> float:
        """Reference chi2 via an explicit ``np.linalg.inv`` (for testing only)."""
        r = self.residual(mu, A_c, delta_c)
        return float(r @ np.linalg.inv(self.cov) @ r)

    def null_chi2(self) -> float:
        """chi2 of the (m-corrected) data against a zero model, no systematics terms (spec s15.2: 269.3)."""
        return self.chi2_from_residual(self.d)
