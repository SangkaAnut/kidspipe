"""E_n / B_n from *binned* xi+/- (the DR5 measurement case).

The official DR5 xi+/- live on 1000 log bins covering 2-300 arcmin; the bin *centres*
span only 2.005-299.25 arcmin, so a spline of xi over the whole [tmin, tmax] cannot be
built at the endpoints.  Here the integral

    E_n = 1/2 int_{tmin}^{tmax} dtheta theta [ T_+n(theta) xi_+(theta) + T_-n(theta) xi_-(theta) ]
    B_n = 1/2 int                 dtheta theta [ T_+n xi_+ - T_-n xi_- ]

is a weighted sum over the fine bins.  The three schemes differ only in the weight of a
bin; T and xi are always evaluated at the bin's theta value:

  "edges"  : w_k = (b_{k+1}^2 - b_k^2)/2   exact integral of theta over the bin (default)
  "centre" : w_k = theta_k (b_{k+1} - b_k)  midpoint rule
  "trapz"  : composite trapezoid in ln(theta) of f_k = theta_k^2 T_k xi_k  (theta dtheta = theta^2 dln theta);
             omits the outer half-bins, an O(dln theta) error: vs "edges" 0.2 % (n=1) to 2 % (n=4) at 1000 bins

``unit`` sets the length unit of dtheta*theta: 'arcmin' (T dimensionless, theta in arcmin)
or 'rad' (theta in radians, i.e. the arcmin result x ARCMIN**2).  Which one reproduces the
official E_n is decided empirically in scripts/dr5_b1_integration.py.
"""
from __future__ import annotations

import numpy as np

from ..theory.cosebis import CosebisFilters

ARCMIN = np.deg2rad(1.0 / 60.0)   # radians per arcmin


def bin_edges(theta_nom: np.ndarray) -> np.ndarray:
    """Recover the n+1 log-bin edges from n geometric bin centres (TreeCorr Log binning)."""
    theta_nom = np.asarray(theta_nom, float)
    n = theta_nom.size
    ratio = (theta_nom[-1] / theta_nom[0]) ** (1.0 / (n - 1))
    lo = theta_nom[0] / np.sqrt(ratio)
    return lo * ratio ** np.arange(n + 1)


def _weights(theta: np.ndarray, edges: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "edges":
        return 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
    if scheme == "centre":
        return theta * np.diff(edges)
    raise ValueError(f"weights not defined for scheme {scheme}")


def e_from_xi_binned(theta_arcmin: np.ndarray, xip: np.ndarray, xim: np.ndarray,
                     filt: CosebisFilters, nmodes: int = 20, scheme: str = "edges",
                     unit: str = "arcmin") -> tuple:
    """(E, B), each shape (..., nmodes), from xi sampled at ``theta_arcmin`` (arcmin)."""
    th = np.asarray(theta_arcmin, float)
    xip = np.asarray(xip, float)
    xim = np.asarray(xim, float)
    edges = bin_edges(th)
    z = filt.z_of_theta(th)
    Tp = filt.Tplus_z(z, nmodes)          # (nmodes, nth)
    Tm = filt.Tminus_z(z, nmodes)

    if scheme == "trapz":
        # theta T xi dtheta = theta^2 T xi dln(theta), sampled at the bin centres; trapezoid in ln(theta)
        t = np.log(th)
        ap = np.trapz(Tp * th ** 2 * xip[..., None, :], t, axis=-1)      # -> (..., nmodes)
        am = np.trapz(Tm * th ** 2 * xim[..., None, :], t, axis=-1)
    else:
        w = _weights(th, edges, scheme)
        ap = np.einsum("nk,...k->...n", Tp, w * xip)
        am = np.einsum("nk,...k->...n", Tm, w * xim)

    scale = ARCMIN ** 2 if unit == "rad" else 1.0
    return 0.5 * scale * (ap + am), 0.5 * scale * (ap - am)
