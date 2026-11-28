"""Parameter vector, bounds, priors and prior transform (spec s1).

The 14 varied parameters, in vector order (``PARAM_NAMES``)::

    omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon, A_IA, delta_c, A_c, dz1 .. dz5

Priors (spec s1.1, s1.3).  All are defined on the box ``BOUNDS`` and are zero outside it:

* flat on [lo, hi]:  ``ln pi = -ln(hi - lo)``
* truncated normal N(mu, sig) on [lo, hi]::

      ln pi(x) = -(x-mu)^2/(2 sig^2) - ln(Phi(b) - Phi(a)) - 0.5 ln(2 pi sig^2),
      a = (lo-mu)/sig,  b = (hi-mu)/sig

  used for ``delta_c`` N(0, 2e-4), ``A_c`` N(1.01, 0.13) and ``dz_i`` N(0, sigma_i) with
  sigma = (0.039, 0.023, 0.026, 0.012, 0.011).

Units: all parameters are dimensionless except ``h`` (little h) -- nothing here carries units.
Fixed model constants (Omega_k = 0, w = -1, m_nu = 0.06 eV, 1 massive nu) and the fixed
multiplicative shear-bias vector ``M_BIAS`` are also defined here.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

# --------------------------------------------------------------------------------------
# names, bounds, start (spec s1.1)
# --------------------------------------------------------------------------------------
PARAM_NAMES: tuple[str, ...] = (
    "omch2", "ombh2", "h", "n_s", "ln_1e10_A_s", "A_baryon", "A_IA",
    "delta_c", "A_c", "dz1", "dz2", "dz3", "dz4", "dz5",
)
N_PARAM = len(PARAM_NAMES)

#: (lo, hi) per parameter, shape (14, 2)
BOUNDS: np.ndarray = np.array([
    [0.01, 0.99],      # omch2
    [0.019, 0.026],    # ombh2
    [0.64, 0.82],      # h
    [0.7, 1.3],        # n_s
    [1.7, 5.0],        # ln_1e10_A_s
    [2.0, 3.13],       # A_baryon
    [-6.0, 6.0],       # A_IA
    [-6e-4, 6e-4],     # delta_c
    [0.62, 1.4],       # A_c
    [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1],   # dz1..5
], dtype=np.float64)

#: sampler start point (KV450_values.ini "start" column), shape (14,)
START: np.ndarray = np.array([
    0.10, 0.022, 0.70, 1.09, 3.09, 3.0, 0.0, 0.0, 1.13, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)

# --------------------------------------------------------------------------------------
# Gaussian priors: index -> (mu, sigma)  (spec s1.1)
# --------------------------------------------------------------------------------------
#: width of the photo-z shift priors, bins 1..5
SIGMA_DZ: np.ndarray = np.array([0.039, 0.023, 0.026, 0.012, 0.011])

_IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
GAUSSIAN_PRIORS: dict[str, tuple[float, float]] = {
    "delta_c": (0.0, 2e-4),
    "A_c": (1.01, 0.13),
    **{f"dz{i + 1}": (0.0, float(SIGMA_DZ[i])) for i in range(5)},
}

_G_IDX = np.array([_IDX[n] for n in GAUSSIAN_PRIORS], dtype=int)
_G_MU = np.array([v[0] for v in GAUSSIAN_PRIORS.values()])
_G_SIG = np.array([v[1] for v in GAUSSIAN_PRIORS.values()])
_G_LO = BOUNDS[_G_IDX, 0]
_G_HI = BOUNDS[_G_IDX, 1]
_G_A = (_G_LO - _G_MU) / _G_SIG
_G_B = (_G_HI - _G_MU) / _G_SIG
# log of truncation mass Phi(b)-Phi(a); all brackets here straddle zero so there is no cancellation
_G_LOGZ = np.log(ndtr(_G_B) - ndtr(_G_A))
_G_NORM = -_G_LOGZ - 0.5 * np.log(2.0 * np.pi * _G_SIG ** 2)    # additive constant of ln pi

_IS_GAUSS = np.zeros(N_PARAM, dtype=bool)
_IS_GAUSS[_G_IDX] = True
_FLAT_IDX = np.nonzero(~_IS_GAUSS)[0]
_FLAT_LO = BOUNDS[_FLAT_IDX, 0]
_FLAT_HI = BOUNDS[_FLAT_IDX, 1]
_FLAT_SPAN = _FLAT_HI - _FLAT_LO
_FLAT_LOGNORM = -np.log(_FLAT_SPAN)

_G_PA = ndtr(_G_A)
_G_PB = ndtr(_G_B)
_G_DP = _G_PB - _G_PA
_G_QA = ndtr(-_G_B)
_G_QB = ndtr(-_G_A)
_G_DQ = _G_QB - _G_QA
_G_USE_MIRROR = _G_A >= 0.0
_G_ANY_MIRROR = bool(np.any(_G_USE_MIRROR))

# --------------------------------------------------------------------------------------
# fixed constants
# --------------------------------------------------------------------------------------
#: multiplicative shear bias per tomographic bin, fixed (spec s1.1 "Fixed likelihood constants")
M_BIAS: np.ndarray = np.array([-0.0128, -0.0104, -0.0114, 0.0072, 0.0061])
#: fixed cosmology settings (spec s1.1, s2.1)
FIXED = {"omega_k": 0.0, "w": -1.0, "wa": 0.0, "mnu_ev": 0.06, "num_massive_neutrinos": 1}
#: HMcode baryon-feedback relation eta0 = a0 + a1 A_baryon (spec s1.2)
ETA0_A0, ETA0_A1 = 0.98, -0.12


def eta0(A_baryon):
    """HMcode-2016 eta0 = 0.98 - 0.12 A_baryon (spec s1.2); A=3.13 -> 0.6044, A=2 -> 0.74."""
    return ETA0_A0 + ETA0_A1 * np.asarray(A_baryon, dtype=np.float64)


def A_s(ln_1e10_A_s):
    """Primordial amplitude A_s = exp(ln(1e10 A_s)) * 1e-10 (spec s1.2)."""
    return np.exp(np.asarray(ln_1e10_A_s, dtype=np.float64)) * 1e-10


# --------------------------------------------------------------------------------------
# prior
# --------------------------------------------------------------------------------------
def in_bounds(theta) -> np.ndarray:
    """Boolean array (broadcast over leading axes): all 14 parameters inside ``BOUNDS`` (closed)."""
    th = np.asarray(theta, dtype=np.float64)
    return np.all((th >= BOUNDS[:, 0]) & (th <= BOUNDS[:, 1]), axis=-1)


def log_prior(theta, normalized: bool = True):
    """Log prior density ln pi(theta), spec s1.3.

    ``-inf`` outside ``BOUNDS`` (or if any entry is NaN).  Inside:
    ``sum_flat -ln(hi-lo) + sum_gauss [ -(x-mu)^2/(2 sig^2) - ln(Phi(b)-Phi(a)) - 0.5 ln(2 pi sig^2) ]``.
    With ``normalized=False`` the (parameter-independent) normalisation constants are dropped, which
    is irrelevant for sampling and only matters for evidences.

    Parameters
    ----------
    theta : array (..., 14)
    Returns float for a 1-D input, array (...) for batched input.
    """
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim == 0 or th.shape[-1] != N_PARAM:
        raise ValueError(f"log_prior expects last axis {N_PARAM}, got shape {th.shape}")
    ok = in_bounds(th)                       # NaN compares False -> not ok
    z = (th[..., _G_IDX] - _G_MU) / _G_SIG
    lp = -0.5 * np.sum(z * z, axis=-1)
    if normalized:
        lp = lp + np.sum(_G_NORM) + np.sum(_FLAT_LOGNORM)
    out = np.where(ok, lp, -np.inf)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------------------
# prior transform (exact inverse CDFs)
# --------------------------------------------------------------------------------------
def _trunc_norm_ppf(u, a, b, mu, sig):
    """Inverse CDF of N(mu, sig) truncated to standardised interval [a, b], elementwise.

    x = mu + sig * Phi^-1( Phi(a) + u (Phi(b) - Phi(a)) ), evaluated on whichever side of zero
    keeps full relative precision in the tail (mirror trick for a >= 0), then clipped to [a, b].
    """
    u = np.asarray(u, dtype=np.float64)
    pa, pb = ndtr(a), ndtr(b)
    lower = ndtri(pa + u * (pb - pa))
    # mirrored evaluation (x -> -x) for intervals lying on the positive side
    qa, qb = ndtr(-b), ndtr(-a)           # = sf(b), sf(a)
    mirrored = -ndtri(qa + (1.0 - u) * (qb - qa))
    x = np.where(a >= 0.0, mirrored, lower)
    x = np.clip(x, a, b)
    return mu + sig * x


def prior_transform(u):
    """Map the unit cube to parameter space with the exact prior inverse CDFs (spec s1.1).

    Flat parameters: ``lo + u (hi - lo)``.  Truncated Gaussians: ``mu + sig Phi^-1(Phi(a) + u (Phi(b)-Phi(a)))``.
    Vectorised; ``u`` may have shape (14,), (M, 14), or (..., 14).  Output has the same shape and always lies
    inside ``BOUNDS`` (so ``log_prior`` is finite on it).  ``u`` outside [0, 1] is clipped to [0, 1]
    (identically for flat and Gaussian parameters); a NaN in ``u`` raises ``ValueError`` rather than
    silently producing NaN parameters.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 0 or u.shape[-1] != N_PARAM:
        raise ValueError(f"prior_transform expects last axis {N_PARAM}, got {u.shape}")
    if np.isnan(u).any():
        raise ValueError("prior_transform: NaN in unit-cube coordinates")
    u = np.clip(u, 0.0, 1.0)
    out = np.empty_like(u)
    out[..., _FLAT_IDX] = np.clip(_FLAT_LO + u[..., _FLAT_IDX] * _FLAT_SPAN, _FLAT_LO, _FLAT_HI)
    ug = u[..., _G_IDX]
    x = ndtri(_G_PA + ug * _G_DP)
    if _G_ANY_MIRROR:
        mirrored = -ndtri(_G_QA + (1.0 - ug) * _G_DQ)
        x = np.where(_G_USE_MIRROR, mirrored, x)
    x = np.clip(x, _G_A, _G_B)
    out[..., _G_IDX] = np.clip(_G_MU + _G_SIG * x, _G_LO, _G_HI)
    return out


def prior_cdf(theta):
    """Inverse of :func:`prior_transform`: marginal prior CDF of each parameter, in [0, 1]. Shape as input."""
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim == 0 or th.shape[-1] != N_PARAM:
        raise ValueError(f"prior_cdf expects last axis {N_PARAM}, got shape {th.shape}")
    out = np.empty_like(th)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    out[..., _FLAT_IDX] = (th[..., _FLAT_IDX] - lo[_FLAT_IDX]) / (hi - lo)[_FLAT_IDX]
    z = (th[..., _G_IDX] - _G_MU) / _G_SIG
    pa, pb = ndtr(_G_A), ndtr(_G_B)
    out[..., _G_IDX] = (ndtr(z) - pa) / (pb - pa)
    return np.clip(out, 0.0, 1.0)


def to_dict(theta) -> dict:
    """Parameter vector (14,) -> ``{name: float}`` in ``PARAM_NAMES`` order."""
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.size != N_PARAM:
        raise ValueError(f"expected {N_PARAM} parameters, got {th.size}")
    return {n: float(v) for n, v in zip(PARAM_NAMES, th)}


def from_dict(d: dict, default: np.ndarray | None = None) -> np.ndarray:
    """``{name: value}`` -> parameter vector (14,); missing names are taken from ``default`` (START)."""
    base = (START if default is None else np.asarray(default, dtype=np.float64)).copy()
    for k, v in d.items():
        base[_IDX[k]] = v
    return base


def dz_from_theta(theta) -> np.ndarray:
    """Photo-z shifts dz1..dz5 from a parameter vector, shape (..., 5)."""
    return np.asarray(theta, dtype=np.float64)[..., 9:14]
