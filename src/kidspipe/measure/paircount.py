"""Independent pair counter: KD-tree pair counting on 3-D unit vectors with chord distances,
plus explicit tangential/cross shear projection.  Used as the cross-check (check A4) against
TreeCorr; it shares no code with TreeCorr and no correlation function with it.

Estimator (the standard shear two-point estimator, as in TreeCorr's "simple" mode):

    xi_+(theta) = sum_ij w_i w_j (g1_i g1_j + g2_i g2_j) / sum_ij w_i w_j
    xi_-(theta) = sum_ij w_i w_j Re[(g1_i + i g2_i) (g1_j + i g2_j) e^{-4 i phi_ij}] / sum
    xi_x(theta) = sum_ij w_i w_j Im[...]                                              / sum

with phi_ij the position angle of the separation vector from i to j measured in the same
frame as the shears, so that e^{-4i phi} rotates both shears onto the separation axis.  The
shears are given in the (RA, Dec) sky frame, so phi is measured from +RA towards +Dec, i.e.
phi = atan2(dDec, dRA * cos(Dec_mean)) with the same handedness as the e1/e2 definition.

The rotation is applied separately to each galaxy's shear (the correct tangential
projection), not just to the product's phase, so this is a genuinely independent
implementation path from TreeCorr's.

Usage:
    pc = PairCounter(ra, dec, g1, g2, w, min_sep=2.0, max_sep=300.0, nbins=1000)
    xip, xim, xix, npairs, weight = pc.run(other=None)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, float))
    dec = np.deg2rad(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _log_edges(min_sep: float, max_sep: float, nbins: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(min_sep), np.log(max_sep), nbins + 1))


class PairCounter:
    """Pair counter for one catalogue (auto) or two (cross)."""

    def __init__(self, ra, dec, g1, g2, w, min_sep=2.0, max_sep=300.0, nbins=1000):
        self.ra = np.asarray(ra, float)
        self.dec = np.asarray(dec, float)
        self.g1 = np.asarray(g1, float)
        self.g2 = np.asarray(g2, float)
        self.w = np.asarray(w, float)
        self.min_sep, self.max_sep, self.nbins = float(min_sep), float(max_sep), int(nbins)
        self.edges = _log_edges(self.min_sep, self.max_sep, self.nbins)
        self.xyz = unit_vectors(self.ra, self.dec)

    def run(self, other: "PairCounter | None" = None, chunk: int = 200_000):
        """Return (xip, xim, xix, npairs, weight), each shape (nbins,).

        Chord distance d = 2 sin(theta/2); we bin on theta (arcmin) to match TreeCorr's Log
        binning in separation angle.
        """
        o = self if other is None else other
        auto = other is None
        # max chord length for max_sep
        dmax = 2.0 * np.sin(np.deg2rad(self.max_sep / 60.0) / 2.0)
        tree = cKDTree(o.xyz)
        nb = self.nbins
        num_p = np.zeros(nb)
        num_m_re = np.zeros(nb)
        num_m_im = np.zeros(nb)
        npairs = np.zeros(nb)
        wsum = np.zeros(nb)

        n1 = self.xyz.shape[0]
        for start in range(0, n1, chunk):
            stop = min(start + chunk, n1)
            idx = np.arange(start, stop)
            nb_list = tree.query_ball_point(self.xyz[idx], r=dmax)
            counts = np.fromiter((len(v) for v in nb_list), dtype=np.int64, count=idx.size)
            if counts.sum() == 0:
                continue
            i_all = np.repeat(idx, counts)
            j_all = np.concatenate([np.asarray(v, dtype=np.int64) for v in nb_list])
            if auto:
                m = j_all > i_all
                i_all, j_all = i_all[m], j_all[m]
            if i_all.size == 0:
                continue

            chord = np.linalg.norm(self.xyz[i_all] - o.xyz[j_all], axis=1)
            theta_arcmin = np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1))) * 60.0
            m = (theta_arcmin >= self.min_sep) & (theta_arcmin < self.max_sep)
            i_all, j_all, theta_arcmin = i_all[m], j_all[m], theta_arcmin[m]
            if i_all.size == 0:
                continue

            # position angle of j as seen from i, in the (RA, Dec) frame of the shears
            dra = (o.ra[j_all] - self.ra[i_all] + 180.0) % 360.0 - 180.0
            ddec = o.dec[j_all] - self.dec[i_all]
            dec_mid = np.deg2rad(0.5 * (self.dec[i_all] + o.dec[j_all]))
            phi = np.arctan2(ddec, dra * np.cos(dec_mid))

            wi = self.w[i_all]
            wj = o.w[j_all]
            wij = wi * wj

            g1i, g2i = self.g1[i_all], self.g2[i_all]
            g1j, g2j = o.g1[j_all], o.g2[j_all]

            # xi_+ : no rotation
            p = wij * (g1i * g1j + g2i * g2j)

            # xi_-, xi_x : rotate BOTH shears onto the separation axis.
            # With complex shear g = g1 + i g2, rotating the frame by phi multiplies the shear
            # by e^{-2i*phi}.  The pure-rotation 2-point function is
            #     xi_- + i xi_x = < g1 g2 e^{-4 i phi} >          (NO conjugate)
            # so both shears must be multiplied, giving a net e^{-4i phi}.  Using
            # (g1 e^{-2i phi}) * conj(g2 e^{-2i phi}) would cancel the phases and reproduce
            # xi_+ -- that was an earlier bug, caught by comparing with TreeCorr.
            gAm = (g1i + 1j * g2i) * np.exp(-2j * phi)
            gBm = (g1j + 1j * g2j) * np.exp(-2j * phi)
            mm = wij * gAm * gBm

            bins = np.digitize(theta_arcmin, self.edges) - 1
            ok = (bins >= 0) & (bins < nb)
            np.add.at(num_p, bins[ok], p[ok])
            np.add.at(num_m_re, bins[ok], mm.real[ok])
            np.add.at(num_m_im, bins[ok], mm.imag[ok])
            np.add.at(wsum, bins[ok], wij[ok])
            np.add.at(npairs, bins[ok], 1.0)

        good = wsum > 0
        xip = np.zeros(nb)
        xim = np.zeros(nb)
        xix = np.zeros(nb)
        xip[good] = num_p[good] / wsum[good]
        xim[good] = num_m_re[good] / wsum[good]
        xix[good] = num_m_im[good] / wsum[good]
        return xip, xim, xix, npairs, wsum
