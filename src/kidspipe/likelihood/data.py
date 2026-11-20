"""KV450 cosmic-shear data vector, covariance, c-term, ordering and scale cuts (spec s10).

Ordering ("montepython", spec s10.2).  Tomographic bins are 0-based.  The 15 pairs are

    PAIRS = [(j, i) for i in range(5) for j in range(i, 5)]      # theory pair (j, i), j >= i

i.e. (0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),... ; theory xi for pair (j,i) is
``shear_xi_plus/bin_{j+1}_{i+1}``.  Data-file column of pair index ``p`` is ``1 + p``.

Theta bins (spec s8): 9 log bins between 0.5' and 300'.  Cuts (spec s10.3), applied identically to
every pair: keep xi+ bins 0..6 (theta <= 72.4') and xi- bins 3..8 (theta >= 4.2').  The 195-vector is

    [ pair 0: xi+(7 bins), xi-(6 bins) | pair 1: xi+(7), xi-(6) | ... | pair 14 ]   (15 x 13 = 195)

The uncut 270-covariance has layout, per pair, [xi+(9), xi-(9)]; the cut covariance is obtained by
row/column deletion with the same index set (``KV450Data.full_index``).

Theory arrays use ``xip[..., 15, 9]`` / ``xim[..., 15, 9]`` (pair index in PAIRS order, 9 theta bins).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# layout constants
# --------------------------------------------------------------------------------------
N_BIN = 5
PAIRS: list[tuple[int, int]] = [(j, i) for i in range(N_BIN) for j in range(i, N_BIN)]
N_PAIR = len(PAIRS)                 # 15
N_THETA = 9
KEEP_PLUS = np.arange(0, 7)         # xi+ theta-bin indices kept (cut_xi_plus = 7 8 dropped)
KEEP_MINUS = np.arange(3, 9)        # xi- theta-bin indices kept (cut_xi_minus = 0 1 2 dropped)
N_PLUS, N_MINUS = len(KEEP_PLUS), len(KEEP_MINUS)     # 7, 6
N_PER_PAIR = N_PLUS + N_MINUS       # 13
N_DATA = N_PAIR * N_PER_PAIR        # 195
N_FULL = N_PAIR * 2 * N_THETA       # 270

#: theta bin edges in arcmin (spec s8): exp(linspace(ln 0.5, ln 300, 10))
THETA_EDGES_ARCMIN = np.exp(np.linspace(np.log(0.5), np.log(300.0), N_THETA + 1))
#: pair-index lookup {(j, i): p}
PAIR_INDEX = {pr: p for p, pr in enumerate(PAIRS)}
#: boolean mask over the 195-vector, True on xi+ entries (receives the delta_c^2 offset)
PLUS_MASK = np.tile(np.r_[np.ones(N_PLUS, bool), np.zeros(N_MINUS, bool)], N_PAIR)


def pair_index(j: int, i: int) -> int:
    """Index p in PAIRS of the unordered bin pair {i, j} (0-based); (j,i) and (i,j) are equivalent."""
    if j < i:
        j, i = i, j
    return PAIR_INDEX[(j, i)]


def data_column(p: int) -> int:
    """Column of the data file for pair index p: 1 + p, equal to ``1 + i(2n-i+1)/2 + (j-i)`` (spec s10.2)."""
    j, i = PAIRS[p]
    return 1 + i * (2 * N_BIN - i + 1) // 2 + (j - i)


# --------------------------------------------------------------------------------------
# ordering helpers
# --------------------------------------------------------------------------------------
def full_index() -> np.ndarray:
    """Indices (195,) of the kept entries inside the uncut 270-layout [pair][xi+(9), xi-(9)]."""
    base = np.arange(N_PAIR)[:, None] * 2 * N_THETA
    plus = base + KEEP_PLUS[None, :]
    minus = base + N_THETA + KEEP_MINUS[None, :]
    return np.concatenate([plus, minus], axis=1).ravel()


def assemble_vector(xip, xim) -> np.ndarray:
    """Theory xi arrays -> 195-vector in data ordering (cuts applied).

    Parameters
    ----------
    xip, xim : arrays (..., 15, 9)   pair index in PAIRS order, 9 theta bins (uncut)
    Returns array (..., 195); leading axes are preserved (batches allowed).
    """
    xip = np.asarray(xip)
    xim = np.asarray(xim)
    if xip.shape[-2:] != (N_PAIR, N_THETA) or xim.shape[-2:] != (N_PAIR, N_THETA):
        raise ValueError(f"expected (...,{N_PAIR},{N_THETA}) arrays, got {xip.shape}, {xim.shape}")
    both = np.concatenate([xip[..., KEEP_PLUS], xim[..., KEEP_MINUS]], axis=-1)   # (..., 15, 13)
    return both.reshape(both.shape[:-2] + (N_DATA,))


def split_vector(vec) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the cut part of :func:`assemble_vector`: (..., 195) -> xip (...,15,7), xim (...,15,6)."""
    v = np.asarray(vec)
    if v.shape[-1] != N_DATA:
        raise ValueError(f"expected last axis {N_DATA}, got {v.shape}")
    v = v.reshape(v.shape[:-1] + (N_PAIR, N_PER_PAIR))
    return v[..., :N_PLUS], v[..., N_PLUS:]


def cut_cterm(c_plus, c_minus) -> np.ndarray:
    """c-term 9-vectors (xi+_c, xi-_c) -> 195-vector: the same 9-vector for all 15 pairs, then cut (spec s9)."""
    cp = np.asarray(c_plus, dtype=np.float64).reshape(N_THETA)
    cm = np.asarray(c_minus, dtype=np.float64).reshape(N_THETA)
    return assemble_vector(np.broadcast_to(cp, (N_PAIR, N_THETA)),
                           np.broadcast_to(cm, (N_PAIR, N_THETA)))


def from_pm_blocks(vec) -> np.ndarray:
    """Convert the block layout [xi+ (15 x 7) | xi- (15 x 6)] (``xi_pm-bin-theta``, e.g. the Flinc
    theory file; pair-major within each block) to the interleaved montepython layout of this module."""
    v = np.asarray(vec)
    xp = v[..., :N_PAIR * N_PLUS].reshape(v.shape[:-1] + (N_PAIR, N_PLUS))
    xm = v[..., N_PAIR * N_PLUS:].reshape(v.shape[:-1] + (N_PAIR, N_MINUS))
    return np.concatenate([xp, xm], axis=-1).reshape(v.shape[:-1] + (N_DATA,))


def to_pm_blocks(vec) -> np.ndarray:
    """Inverse of :func:`from_pm_blocks`."""
    xp, xm = split_vector(vec)
    return np.concatenate([xp.reshape(xp.shape[:-2] + (-1,)), xm.reshape(xm.shape[:-2] + (-1,))], axis=-1)


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def default_data_dir() -> Path:
    """``$KIDSPIPE_DATA`` if set, else ``<repo>/data`` (see :mod:`kidspipe.paths`)."""
    from .. import paths
    return paths.data_dir()


def load_data_vector(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``xipm_data_vector.dat`` (18 x 16).

    Returns (theta_arcmin (9,), xip (15, 9), xim (15, 9)) with pair index in PAIRS order.
    Rows 0-8 are xi+, rows 9-17 xi-; column 0 is theta [arcmin], columns 1..15 the pairs (spec s10.1).
    """
    raw = np.loadtxt(path)
    if raw.shape != (2 * N_THETA, 1 + N_PAIR):
        raise ValueError(f"unexpected data vector shape {raw.shape}")
    cols = [data_column(p) for p in range(N_PAIR)]
    xip = raw[:N_THETA, cols].T.copy()
    xim = raw[N_THETA:, cols].T.copy()
    return raw[:N_THETA, 0].copy(), xip, xim


def load_covariance(path) -> np.ndarray:
    """Read the uncut analytic covariance (270 x 270), symmetric; layout per pair [xi+(9), xi-(9)]."""
    c = np.loadtxt(path)
    if c.shape != (N_FULL, N_FULL):
        raise ValueError(f"unexpected covariance shape {c.shape}")
    return c


def load_cterm(path) -> tuple[np.ndarray, np.ndarray]:
    """Read the TreeCorr c-term file (9 x 10); columns 3, 4 = (xi+_c, xi-_c) (spec s9)."""
    raw = np.loadtxt(path)
    if raw.shape[0] != N_THETA:
        raise ValueError(f"unexpected c-term shape {raw.shape}")
    return raw[:, 3].copy(), raw[:, 4].copy()


def load_flinc_theory(path=None) -> np.ndarray:
    """Read ``data/validation/flinc_theory_vector.ascii`` (195 values; block layout, see :func:`from_pm_blocks`)."""
    if path is None:
        path = default_data_dir() / "validation" / "flinc_theory_vector.ascii"
    v = np.loadtxt(path).ravel()
    if v.size != N_DATA:
        raise ValueError(f"Flinc theory vector has {v.size} values, expected {N_DATA}")
    return v


@dataclass
class KV450Data:
    """KV450 xi+/- data after scale cuts.

    Attributes
    ----------
    d : (195,)   measured xi+/- (as read, i.e. WITHOUT the m-bias division; see GaussianLikelihood)
    cov : (195, 195)  cut analytic covariance
    cterm_plus, cterm_minus : (9,)  uncut c-term amplitudes
    cterm : (195,)  cut c-term vector (same 9-vector for every pair)
    theta_plus (7,), theta_minus (6,) : pair-weighted mean theta [arcmin] of the kept bins
    xip_full, xim_full : (15, 9) uncut data arrays;  cov_full : (270, 270) uncut covariance
    """
    d: np.ndarray
    cov: np.ndarray
    cterm_plus: np.ndarray
    cterm_minus: np.ndarray
    cterm: np.ndarray
    theta: np.ndarray
    xip_full: np.ndarray = field(repr=False)
    xim_full: np.ndarray = field(repr=False)
    cov_full: np.ndarray = field(repr=False)

    # class-level views of the module constants for convenience
    pairs = PAIRS
    n_data = N_DATA
    plus_mask = PLUS_MASK

    @classmethod
    def load(cls, data_dir=None) -> "KV450Data":
        """Load from ``<data_dir>/kv450/{xipm_data_vector.dat, cov_analytic.txt, c_term_treecorr.out}``."""
        root = Path(data_dir) if data_dir is not None else default_data_dir()
        k = root / "kv450"
        theta, xip, xim = load_data_vector(k / "xipm_data_vector.dat")
        cov_full = load_covariance(k / "cov_analytic.txt")
        cp, cm = load_cterm(k / "c_term_treecorr.out")
        idx = full_index()
        cov = cov_full[np.ix_(idx, idx)]
        cov = 0.5 * (cov + cov.T)            # remove %.4e-level asymmetry, if any
        return cls(d=assemble_vector(xip, xim), cov=cov, cterm_plus=cp, cterm_minus=cm,
                   cterm=cut_cterm(cp, cm), theta=theta, xip_full=xip, xim_full=xim,
                   cov_full=cov_full)

    # ---- helpers -------------------------------------------------------------------
    @property
    def theta_plus(self) -> np.ndarray:
        return self.theta[KEEP_PLUS]

    @property
    def theta_minus(self) -> np.ndarray:
        return self.theta[KEEP_MINUS]

    @staticmethod
    def assemble(xip, xim) -> np.ndarray:
        """See :func:`assemble_vector`."""
        return assemble_vector(xip, xim)

    @staticmethod
    def split(vec):
        """See :func:`split_vector`."""
        return split_vector(vec)

    @staticmethod
    def m_factor(m) -> np.ndarray:
        """Per-entry factor (1+m_a)(1+m_b), shape (195,), for the pair of each entry (spec s9).

        Pair (j,i) uses m[j], m[i]; the factor is repeated over the 13 kept entries of that pair.
        """
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (N_BIN,):
            raise ValueError(f"m must have shape ({N_BIN},), got {m.shape}")
        j = np.array([pr[0] for pr in PAIRS])
        i = np.array([pr[1] for pr in PAIRS])
        per_pair = (1.0 + m[j]) * (1.0 + m[i])
        return np.repeat(per_pair, N_PER_PAIR)
