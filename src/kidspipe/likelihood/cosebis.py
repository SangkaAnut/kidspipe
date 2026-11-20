"""KV450 COSEBIs E_n data, covariance and Gaussian likelihood (spec s14.1).

Data file ``data/cosebis/COSEBIs_En_nModes20.fits``: ``En`` table (300 rows = 15 tomographic pairs x 20 modes;
columns BIN1, BIN2, ANGBIN (= mode n, 1-based), VALUE, ANG (= n again, unit label 'arcmin' is a mislabel)) and
``COVMAT`` (300 x 300, entries ~1e-20, min eigenvalue 2.3e-22; positive definite).  Row order: pair-major, mode
fastest; pairs (BIN1<=BIN2, 1-based) (1,1),(1,2),..,(1,5),(2,2),..,(5,5).  This is EXACTLY ``data.PAIRS`` order
(pair (j,i), j>=i, i outer loop) with j=BIN2-1, i=BIN1-1, which is verified at load time.

Vector used by the likelihood with ``n_modes = N`` (1..20): rows with ANGBIN <= N, i.e. 15*N entries in
(pair, mode) order; the covariance is the corresponding sub-block (modes are NOT re-orthogonalised).

Numerics: chi2 = |L^-1 r|^2 with C = L L^T; to avoid working with 1e-20 numbers, d and C are divided by s and s^2
(s = 1e-10 default) before the Cholesky factorisation.

m-bias treatment (DECISION, spec is silent on whether the shipped E_n are m-corrected)
    E_n is linear in xi, so a multiplicative shear bias gives E_n^obs = (1+m_a)(1+m_b) E_n^true.  Default
    ``apply_m=True``: data are divided by (1+m_a)(1+m_b) with the KV450 m-vector (``params.M_BIAS``), covariance
    untouched -- identical to the xi+/- convention of ``GaussianLikelihood``.  ``apply_m=False`` uses the data as
    shipped.  Both chi2 are reported by ``scripts``/docs; the effect is tiny (|m| <~ 1.3 %) compared with the
    COSEBIs noise.
delta_c / A_c (additive c-term offsets of xi+/-) do NOT apply to COSEBIs here: the E_n of the shipped file already
    have the filters' response to constants removed (T_+n integrate to zero against theta^-2... only up to the
    e^{2z}, e^{4z} constraints, i.e. constant-in-xi_+ and ~theta^-2 offsets are rejected by construction).
    ``CosebisModel`` therefore ignores those two entries of theta.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .data import N_BIN, N_PAIR, PAIRS, pair_index, default_data_dir
from ..params import M_BIAS

N_MODES_FILE = 20
SCALE_E = 1.0e-10


def default_fits_path() -> Path:
    return default_data_dir() / "cosebis" / "COSEBIs_En_nModes20.fits"


@dataclass
class CosebisData:
    """Full 20-mode E_n data: ``value`` (300,) and ``cov`` (300, 300) in (pair, mode) order; ``pair``/``mode`` per row."""
    value: np.ndarray
    cov: np.ndarray
    pair: np.ndarray        # pair index p in PAIRS order, per row
    mode: np.ndarray        # 1-based mode per row
    bin1: np.ndarray
    bin2: np.ndarray
    nz_source: np.ndarray | None = None

    @classmethod
    def load(cls, path=None) -> "CosebisData":
        from astropy.io import fits
        path = default_fits_path() if path is None else Path(path)
        with fits.open(path) as h:
            t = h["En"].data
            cov = np.array(h["COVMAT"].data, dtype=np.float64)
            nz = np.array(h["NZ_SOURCE"].data.tolist(), dtype=float) if "NZ_SOURCE" in h else None
            b1, b2 = np.array(t["BIN1"], int), np.array(t["BIN2"], int)
            mode = np.array(t["ANGBIN"], int)
            val = np.array(t["VALUE"], dtype=np.float64)
        n = N_PAIR * N_MODES_FILE
        if val.size != n or cov.shape != (n, n):
            raise ValueError(f"unexpected COSEBIs file layout: {val.size} values, covariance {cov.shape}")
        pair = np.array([pair_index(int(b2[r]) - 1, int(b1[r]) - 1) for r in range(n)])
        exp_pair = np.repeat(np.arange(N_PAIR), N_MODES_FILE)
        exp_mode = np.tile(np.arange(1, N_MODES_FILE + 1), N_PAIR)
        if not (np.array_equal(pair, exp_pair) and np.array_equal(mode, exp_mode)):
            raise ValueError("COSEBIs En rows are not in (PAIRS order, mode) order")
        if not np.allclose(cov, cov.T, rtol=1e-12, atol=0):
            raise ValueError("COVMAT is not symmetric")
        return cls(val, 0.5 * (cov + cov.T), pair, mode, b1, b2, nz)

    def select(self, n_modes: int) -> np.ndarray:
        """Row indices (15*n_modes,) with mode <= n_modes, in (pair, mode) order."""
        if not 1 <= n_modes <= N_MODES_FILE:
            raise ValueError(f"n_modes must be in 1..{N_MODES_FILE}")
        return np.where(self.mode <= n_modes)[0]


def m_factor(m, n_modes: int) -> np.ndarray:
    """(1+m_a)(1+m_b) per row of the (pair, mode) vector, shape (15*n_modes,)."""
    m = np.asarray(m, float)
    j = np.array([pr[0] for pr in PAIRS])
    i = np.array([pr[1] for pr in PAIRS])
    return np.repeat((1.0 + m[j]) * (1.0 + m[i]), n_modes)


class CosebisLikelihood:
    """Gaussian chi2 of a model E_n vector (15*n_modes, pair-major) against the KV450 COSEBIs data."""

    def __init__(self, n_modes: int = 5, apply_m: bool = True, m=None, data: CosebisData | None = None,
                 scale: float = SCALE_E):
        self.data = CosebisData.load() if data is None else data
        self.n_modes, self.apply_m, self.scale = int(n_modes), bool(apply_m), float(scale)
        self.m = np.asarray(M_BIAS if m is None else m, float)
        idx = self.data.select(self.n_modes)
        d = self.data.value[idx]
        self.d_raw = d.copy()
        self.d = d / m_factor(self.m, self.n_modes) if self.apply_m else d.copy()
        self.cov = self.data.cov[np.ix_(idx, idx)]
        self._L = cholesky(self.cov / self.scale ** 2, lower=True)          # scaled covariance
        self.n_data = self.d.size

    def chi2(self, E) -> float:
        E = np.asarray(E, float).ravel()
        if E.size != self.n_data:
            raise ValueError(f"E must have {self.n_data} entries, got {E.size}")
        if not np.all(np.isfinite(E)):
            return np.inf
        y = solve_triangular(self._L, (self.d - E) / self.scale, lower=True, check_finite=False)
        return float(y @ y)

    def log_like(self, E) -> float:
        c = self.chi2(E)
        return -0.5 * c if np.isfinite(c) else -np.inf

    def null_chi2(self) -> float:
        return self.chi2(np.zeros(self.n_data))

    def condition_number(self) -> float:
        w = np.linalg.eigvalsh(self.cov)
        return float(w[-1] / w[0])


class CosebisModel:
    """Parameter vector (14,) -> E_n theory vector (15*n_modes,) in the data's (pair, mode) order.

    Uses ``CosmicShearModel.cl(theta)`` (same stage caches as the xi+/- model) and the cached linear operator of
    ``theory.cosebis.cosebis_operator``.  delta_c and A_c are ignored (see module docstring).  ``B_n`` theory = 0.
    """

    def __init__(self, model=None, n_modes: int = 5, tmin: float = 0.5, tmax: float = 300.0,
                 lmin: float = 1.0, lmax: float | None = None, filters=None, nthreads: int = 4, cache_dir=None):
        from ..theory.cosebis import CosebisFilters, cosebis_operator
        if model is None:
            from ..api import build_model
            model = build_model(linear_backend="emulator")
        self.model, self.n_modes = model, int(n_modes)
        self.filters = filters if filters is not None else CosebisFilters(tmin, tmax, max(self.n_modes, 1))
        self.M = cosebis_operator(model.ell, self.filters, self.n_modes, lmin=lmin, lmax=lmax,
                                  nthreads=nthreads, cache_dir=cache_dir)

    def mu_from_cl(self, C) -> np.ndarray:
        from ..theory.cosebis import E_from_cl
        return E_from_cl(C, self.M).reshape(-1)          # (15, n_modes) -> pair-major

    def mu(self, theta) -> np.ndarray:
        _, C = self.model.cl(theta)
        return self.mu_from_cl(C)

    __call__ = mu

    @staticmethod
    def B_theory(n_modes: int) -> np.ndarray:
        """Placeholder: B-mode COSEBIs are zero in the lensing-only model."""
        return np.zeros(N_PAIR * n_modes)


class CosebisPosterior:
    """ln posterior = log prior (params, normalized) + COSEBIs ln L (flat/truncated priors as for xi+/-)."""

    def __init__(self, cmodel: CosebisModel, like: CosebisLikelihood):
        if cmodel.n_modes != like.n_modes:
            raise ValueError("model and likelihood n_modes differ")
        self.model, self.like = cmodel, like

    def log_like(self, theta) -> float:
        return self.like.log_like(self.model.mu(theta))

    def log_post(self, theta) -> float:
        from ..params import log_prior
        lp = float(log_prior(theta))
        return lp + self.log_like(theta) if np.isfinite(lp) else -np.inf


def build_cosebis(n_modes: int = 5, apply_m: bool = True, model=None, **kw):
    """Convenience: (CosebisModel, CosebisLikelihood) pair."""
    return CosebisModel(model, n_modes, **kw), CosebisLikelihood(n_modes, apply_m)
