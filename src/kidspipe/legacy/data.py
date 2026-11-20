"""KiDS-Legacy COSEBIs data, covariance, n(z), dz covariance, official chain, and the Gaussian likelihood."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular

N_BIN = 6
N_PAIR = 21
N_MODES_FILE = 20
SCALE_E = 1.0e-10
PAIRS = [(i, j) for i in range(N_BIN) for j in range(i, N_BIN)]      # (BIN1-1, BIN2-1), BIN1 <= BIN2, FITS order


def legacy_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        p = parent / "data" / "kids_legacy"
        if p.is_dir():
            return p
    raise FileNotFoundError("data/kids_legacy not found")


@dataclass
class LegacyData:
    value: np.ndarray        # (420,)  E_n  (pair-major, mode fastest)
    value_B: np.ndarray      # (420,)
    cov: np.ndarray          # (840, 840)  [En | Bn]
    pair: np.ndarray         # pair index per row (420,)
    mode: np.ndarray         # 1-based mode per row
    z_mid: np.ndarray        # (119,)
    nz: np.ndarray           # (6, 119) histogram heights (NOT normalised as a density)
    dz_cov: np.ndarray       # (6, 6)

    @classmethod
    def load(cls, path=None, dz_cov_path=None) -> "LegacyData":
        from astropy.io import fits
        d = legacy_dir()
        path = d / "KiDS_Legacy_cosebis.fits" if path is None else Path(path)
        with fits.open(path) as h:
            t, tb = h["En"].data, h["Bn"].data
            cov = np.array(h["COVMAT"].data, float)
            b1, b2, mode = np.array(t["BIN1"], int), np.array(t["BIN2"], int), np.array(t["ANGBIN"], int)
            val, valB = np.array(t["VALUE"], float), np.array(tb["VALUE"], float)
            nzt = h["NZ_SOURCE"].data
            z_mid = np.array(nzt["Z_MID"], float)
            nz = np.array([nzt[f"BIN{i}"] for i in range(1, N_BIN + 1)], float)
        pidx = {p: k for k, p in enumerate(PAIRS)}
        pair = np.array([pidx[(b1[r] - 1, b2[r] - 1)] for r in range(b1.size)])
        if not (np.array_equal(pair, np.repeat(np.arange(N_PAIR), N_MODES_FILE))
                and np.array_equal(mode, np.tile(np.arange(1, N_MODES_FILE + 1), N_PAIR))):
            raise ValueError("En rows not in (pair, mode) FITS order")
        if cov.shape != (840, 840) or not np.allclose(cov, cov.T, rtol=1e-10, atol=0):
            raise ValueError("unexpected COVMAT")
        dzc = np.loadtxt(d / "Nz_covariance.txt" if dz_cov_path is None else dz_cov_path)
        return cls(val, valB, 0.5 * (cov + cov.T), pair, mode, z_mid, nz, dzc)

    def select(self, n_modes: int = 6) -> np.ndarray:
        return np.where(self.mode <= n_modes)[0]


class LegacyCosebisLikelihood:
    """Gaussian chi2 for the 126 (21 pairs x 6 modes) En data points.  chi2 = |L^-1 (d - E)|^2, covariance = En block."""

    def __init__(self, n_modes: int = 6, data: LegacyData | None = None, scale: float = SCALE_E):
        self.data = LegacyData.load() if data is None else data
        self.n_modes, self.scale = int(n_modes), float(scale)
        idx = self.data.select(self.n_modes)
        self.d = self.data.value[idx]
        self.cov = self.data.cov[np.ix_(idx, idx)]
        self._L = cholesky(self.cov / scale ** 2, lower=True)
        self.n_data = self.d.size

    def chi2(self, E) -> float:
        E = np.asarray(E, float).ravel()
        if E.size != self.n_data:
            raise ValueError(f"E must have {self.n_data} entries")
        if not np.all(np.isfinite(E)):
            return np.inf
        y = solve_triangular(self._L, (self.d - E) / self.scale, lower=True, check_finite=False)
        return float(y @ y)

    def log_like(self, E) -> float:
        return -0.5 * self.chi2(E)


# ------------------------------------------------------------------ official chain
CHAIN_COLUMNS = {
    "omch2": 0, "ombh2": 1, "h0": 2, "n_s": 3, "s_8_input": 4, "log_t_agn": 5,
    "sigma_8": 21, "ln1e10As": 22, "omega_m": 23, "omega_nu": 24, "a": 27, "beta": 28,
    "log_weight": 41, "prior": 42, "post": 43,
}


def load_chain(path=None) -> np.ndarray:
    p = legacy_dir() / "official_cosebis" / "output_nautilus_cosebis_Fiducial.txt" if path is None else Path(path)
    return np.loadtxt(p, comments="#")


def chain_samples(chain: np.ndarray, rows) -> list[dict]:
    """Physical parameter dicts (and official quantities) for chain rows.  NOTE: the 'A_S' column is ln(1e10 A_s)."""
    out = []
    for r in np.atleast_1d(rows):
        c = chain[r]
        d = dict(omch2=c[0], ombh2=c[1], h0=c[2], n_s=c[3], A_s=np.exp(c[22]) * 1e-10, log_t_agn=c[5],
                 a=c[27], beta=c[28])
        for i in range(6):
            d[f"log10_M_mean_{i+1}"] = c[29 + i]
            d[f"bias_{i+1}"] = c[35 + i]
        d.update(sigma_8_off=c[21], omega_m_off=c[23], loglike_off=c[43] - c[42], chi2_off=-2 * (c[43] - c[42]),
                 row=int(r), log_weight=c[41])
        out.append(d)
    return out
