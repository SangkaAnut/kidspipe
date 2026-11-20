"""Shared loaders for official KiDS-Legacy DR5 products + the xi->En integration used in the DR5 checks.

Only official files are touched here; no catalogue access.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

OFFICIAL = Path("/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/"
                "KiDS_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/data")

N_BIN = 6
N_PAIR = 21
N_MODES = 20
N_THETA_FINE = 1000
TMIN, TMAX = 2.0, 300.0
PAIRS = [(i, j) for i in range(N_BIN) for j in range(i, N_BIN)]  # (bin1-1, bin2-1), i<=j


def fine_bin_centres(tmin=TMIN, tmax=TMAX, n=N_THETA_FINE) -> np.ndarray:
    """TreeCorr Log binning: n bins, edges geometric from tmin to tmax; r_nom = geometric centre."""
    e = np.exp(np.linspace(np.log(tmin), np.log(tmax), n + 1))
    return np.sqrt(e[:-1] * e[1:])


def load_official_xi(tag: str = "NScomb") -> dict:
    """Official TreeCorr xi+/xi- ascii: returns arrays (21, 1000) for xip, xim, xip_im, xim_im,
    sigma_xip, sigma_xim, npairs, weight, npairs_weighted, and the theta columns."""
    files = sorted(glob.glob(str(OFFICIAL / "cosmosis_xipm" / f"XI_KiDS_Legacy_{tag}_nBins_6_Bin*_Bin*.ascii")))
    if len(files) != N_PAIR:
        raise FileNotFoundError(f"expected {N_PAIR} ascii files, found {len(files)}")
    keys = ["xip", "xim", "xip_im", "xim_im", "sigma_xip", "sigma_xim", "npairs",
            "weight", "npairs_weighted", "r_nom", "meanr", "meanlogr"]
    out = {k: np.zeros((N_PAIR, N_THETA_FINE)) for k in keys}
    pairs = []
    for f in files:
        m = re.search(r"Bin(\d+)_Bin(\d+)\.ascii$", f)
        i, j = int(m.group(1)), int(m.group(2))
        k = PAIRS.index((i - 1, j - 1))
        a = np.genfromtxt(f, names=True)
        for key in keys:
            out[key][k] = a[key]
        pairs.append((i, j))
    if pairs != [(i + 1, j + 1) for i, j in PAIRS]:
        raise ValueError(f"file order mismatch: {pairs}")
    out["pairs"] = PAIRS
    return out


def load_official_xipm_fits() -> dict:
    """Official 9-bin xi+/xi- FITS (xiP, xiM, COVMAT 378x378)."""
    from astropy.io import fits
    with fits.open(OFFICIAL / "KiDS_Legacy_xipm.fits") as h:
        tp, tm = h["xiP"].data, h["xiM"].data
        cov = np.array(h["COVMAT"].data, float)
        hdr = {k: h["NZ_SOURCE"].header[k] for k in h["NZ_SOURCE"].header
               if k.startswith("NGAL") or k.startswith("SIG_E")}
        xiP = np.array(tp["VALUE"], float)
        xiM = np.array(tm["VALUE"], float)
        ang = np.array(tp["ANG"], float)
        b1, b2, ab = np.array(tp["BIN1"], int), np.array(tp["BIN2"], int), np.array(tp["ANGBIN"], int)
    nang = ang.size // N_PAIR
    pidx = {p: k for k, p in enumerate(PAIRS)}
    pair = np.array([pidx[(b1[r] - 1, b2[r] - 1)] for r in range(b1.size)])
    order = np.lexsort((ab, pair))
    if not np.array_equal(pair[order], np.repeat(np.arange(N_PAIR), nang)):
        raise ValueError("xiP rows not pair-major")
    return {
        "xiP": xiP[order].reshape(N_PAIR, nang),
        "xiM": xiM[order].reshape(N_PAIR, nang),
        "cov": 0.5 * (cov + cov.T),
        "theta": ang[:nang],
        "nang": nang,
        "header": hdr,
        "pair": pair[order].reshape(N_PAIR, nang),
        "ang": ab[order].reshape(N_PAIR, nang),
    }


def load_official_cosebis() -> dict:
    """Official COSEBIs FITS: En, Bn (21,20), COVMAT (840,840) = [En | Bn], plus NZ header."""
    from astropy.io import fits
    with fits.open(OFFICIAL / "KiDS_Legacy_cosebis.fits") as h:
        te, tb = h["En"].data, h["Bn"].data
        cov = np.array(h["COVMAT"].data, float)
        hdr = {k: h["NZ_SOURCE"].header[k] for k in h["NZ_SOURCE"].header
               if k.startswith("NGAL") or k.startswith("SIG_E")}
        b1, b2, ab = np.array(te["BIN1"], int), np.array(te["BIN2"], int), np.array(te["ANGBIN"], int)
        En = np.array(te["VALUE"], float)
        Bn = np.array(tb["VALUE"], float)
    pidx = {p: k for k, p in enumerate(PAIRS)}
    pair = np.array([pidx[(b1[r] - 1, b2[r] - 1)] for r in range(b1.size)])
    order = np.lexsort((ab, pair))
    return {
        "En": En[order].reshape(N_PAIR, N_MODES),
        "Bn": Bn[order].reshape(N_PAIR, N_MODES),
        "cov": 0.5 * (cov + cov.T),
        "header": hdr,
    }


def cov_e_block(cov: np.ndarray, n_modes: int, n_pair: int = N_PAIR) -> np.ndarray:
    """Take the En-only block of the 840x840 [En|Bn] covariance for the first n_modes."""
    idx = np.array([p * N_MODES + m for p in range(n_pair) for m in range(n_modes)])
    return cov[np.ix_(idx, idx)]


def cov_xip_block(cov: np.ndarray, nang: int) -> np.ndarray:
    """Take the xiP-only block of the 378x378 [xiP|xiM] covariance."""
    idx = np.array([p * nang + a for p in range(N_PAIR) for a in range(nang)])
    return cov[np.ix_(idx, idx)]


def cov_xim_block(cov: np.ndarray, nang: int) -> np.ndarray:
    idx = np.array([N_PAIR * nang + p * nang + a for p in range(N_PAIR) for a in range(nang)])
    return cov[np.ix_(idx, idx)]


def weighted_rebin(theta_fine: np.ndarray, xi: np.ndarray, npair_w: np.ndarray, nang: int = 9):
    """Re-bin fine log bins onto the official nang log bins, weighting by npairs_weighted, matching
    the CosmoSIS bin_xi module (weighted_binning=1):  xi_bin = sum(w xi)/sum(w) over fine bins in bin,
    and theta_bin = sum(w theta)/sum(w)."""
    e = np.exp(np.linspace(np.log(TMIN), np.log(TMAX), nang + 1))
    out = np.zeros(xi.shape[:-1] + (nang,))
    th = np.zeros(nang)
    for k in range(nang):
        m = (theta_fine >= e[k]) & (theta_fine < e[k + 1])
        if k == nang - 1:
            m = (theta_fine >= e[k]) & (theta_fine <= e[k + 1])
        w = npair_w[..., m]
        out[..., k] = np.sum(w * xi[..., m], axis=-1) / np.sum(w, axis=-1)
        w0 = w.reshape(-1, w.shape[-1])[0]      # theta from the first row's weights (one theta per bin)
        th[k] = np.sum(w0 * theta_fine[m]) / np.sum(w0)
    return out, th, e
