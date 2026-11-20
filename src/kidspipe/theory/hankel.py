"""C_ell -> xi_+/- (Hankel transform) and theta binning (MASTER_MATH_SPEC s7, s8).

Definitions (flat sky, continuous multipole, spec s7.1)::

    xi_+(theta) = 1/(2 pi) * int_0^inf dl  l  C(l)  J_0(l theta)
    xi_-(theta) = 1/(2 pi) * int_0^inf dl  l  C(l)  J_4(l theta)

with C(l) the dimensionless convergence-like spectrum of spec s6 and theta in radians.

Two routes to the 9 binned values per pair and sign are provided:

``fidelity="kcap"``  (spec s7.2 (F) + s8 (F))
    FFT-log Hankel transform on the fixed 1024-point grid l in [1e-4, 5e6]
    (log-log interpolation of C with slopes +1 / -3 outside the input range, kernel
    ``K_mu(k) = exp(i k ln2) Gamma((1+mu)/2 + i k/2) / Gamma((1+mu)/2 - i k/2)``, evaluated
    once and cached), followed by the node-trapezoid, theta-weighted bin average.

``fidelity="fast"``  (spec s7.2 (E) + s8 (E))
    A precomputed dense linear operator ``M_pm`` of shape ``(nbin, nell)`` so that
    ``xi_pm_binned = M_pm @ C`` for a FIXED ell grid.  Row b of ``M_mu`` is

        M_mu[b, :] = int dl  (1/2pi) l  Jbar_mu,b(l)  L_k(l)

    where ``L_k`` is the (linear) interpolation basis of the C(l) nodes and

        Jbar_mu,b(l) = int_{th_a}^{th_b} th J_mu(l th) dth / int_{th_a}^{th_b} th dth

    is the *exact* theta-weighted bin average of the Bessel function over the true bin edges
    (closed form below).  This is the continuous definition of s7.1 followed by the exact
    bin average of s8 (E).

Closed forms (x = l*theta; checked numerically in tests)::

    int x J_0(x) dx = x J_1(x)
    int x J_4(x) dx = x J_1(x) - 4 J_0(x) - 12 J_2(x)  (+ const, chosen so F_4(0) = 0)
                    = sum_k (-1)^k x^(2k+6) / ((2k+6) 2^(2k+4) k! (k+4)!)    (small x series)
    int_{th_a}^{th_b} th J_mu(l th) dth = [F_mu(l th_b) - F_mu(l th_a)] / l^2

Treatment of C(l) off the ell grid in the fast operator: C is interpolated between the nodes with
the local 4-point Lagrange cubic in ln l (linear in the node values, error O(h^4) ~ 1e-7 for the
KiDS ell grid), and outside [ell_0, ell_{n-1}] it follows the same power laws as the FFT route,
C = C_0 (l/l_0)^{+1} below and C = C_{n-1} (l/l_{n-1})^{-3} above (integrated to l = 0 and to
l_max = 5e6, the end of the FFT grid).  Both are linear in C_0, C_{n-1} so they only modify the first
and last columns of M.

Units: ell dimensionless, theta radians internally (arcmin at the API boundary for the edges),
xi dimensionless.

Public API::

    PAIRS, THETA_EDGES_ARCMIN, theta_edges_arcmin()
    fft_grid(), hankel_kernel(mu), ClInterpolator, cl_to_xi_fft(ell, C)
    kcap_bin_operator(theta, edges_rad), exact_bin_operator(theta, edges_rad)
    bessel_bin_average(ell, theta_a, theta_b, mu), bin_operator(ell, edges_rad)
    HankelBinner(ell, theta_edges_arcmin=None, fidelity="fast")
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.special import j0, j1, loggamma

__all__ = [
    "PAIRS", "NBIN", "THETA_EDGES_ARCMIN", "theta_edges_arcmin", "ARCMIN",
    "N_FFT", "L_MIN_FFT", "L_MAX_FFT", "LN2_KCAP",
    "fft_grid", "hankel_kernel", "ClInterpolator", "cl_to_xi_fft",
    "kcap_bin_operator", "exact_bin_operator",
    "bessel_antiderivative", "bessel_bin_average", "bin_operator", "HankelBinner",
]

#: arcmin in radians.
ARCMIN = np.pi / (180.0 * 60.0)
#: tomographic pair order (j >= i), "montepython" order, 15 pairs (spec s10.2).
PAIRS = [(j, i) for i in range(5) for j in range(i, 5)]
_PJ = np.array([p[0] for p in PAIRS])
_PI = np.array([p[1] for p in PAIRS])
NBIN = 9

N_FFT = 1024
L_MIN_FFT = 1.0e-4
L_MAX_FFT = 5.0e6
#: kcap's 8-digit ln 2 (spec s7.2 step 3); the exact value differs by <= 7e-8 rad of phase.
LN2_KCAP = 0.69314718


def theta_edges_arcmin(n_edges: int = NBIN + 1, lo: float = 0.5, hi: float = 300.0) -> np.ndarray:
    """Bin edges in arcmin, ``exp(linspace(ln lo, ln hi, n_edges))`` (spec s8)."""
    return np.exp(np.linspace(np.log(lo), np.log(hi), n_edges))


#: default 10 edges (9 bins): 0.5, 1.0178, 2.0717, 4.2172, 8.5843, 17.474, 35.569, 72.403, 147.38, 300 arcmin.
THETA_EDGES_ARCMIN = theta_edges_arcmin()
THETA_EDGES_ARCMIN.setflags(write=False)      # module-level constant: no hidden global state


def _check_ell(ell, min_nodes: int) -> np.ndarray:
    """Validated float copy of an ell grid: 1-D, finite, strictly increasing, positive."""
    ell = np.array(ell, dtype=float)             # copy: later in-place edits by the caller must not leak in
    if ell.ndim != 1 or ell.size < min_nodes or not np.all(np.isfinite(ell)) \
            or np.any(ell <= 0) or np.any(np.diff(ell) <= 0):
        raise ValueError(f"ell must be a finite, strictly increasing, positive 1-D array (>= {min_nodes} nodes)")
    ell.setflags(write=False)
    return ell


def _check_edges(edges) -> np.ndarray:
    """Validated float copy of bin edges: 1-D, >= 2 entries, finite, positive, strictly increasing."""
    edges = np.array(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) \
            or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("theta bin edges must be a finite, strictly increasing, positive 1-D array (>= 2 entries)")
    return edges


# ---------------------------------------------------------------------------------------------
# FFT-log Hankel transform (spec s7.2)
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _fft_grid_cached():
    dlnl = np.log(L_MAX_FFT / L_MIN_FFT) / (N_FFT - 1)
    lnrc = 0.5 * (np.log(L_MAX_FFT) + np.log(L_MIN_FFT))
    nc = N_FFT // 2 + 1
    l = np.exp(lnrc + (np.arange(N_FFT) - nc) * dlnl)
    l.setflags(write=False)
    return l, dlnl


def fft_grid():
    """Fixed FFT grid of spec s7.2.

    Returns ``(l, dlnl)``: ``l_i = exp(lnrc + (i - nc) Dl)``, i = 0..1023, nc = 513,
    Dl = ln(5e6/1e-4)/1023 = 0.0240814163 (l_0 = 9.645e-5, l_1023 = 4.8226e6).  The output theta
    grid is ``theta_j = 1/l_{N-1-j}`` in radians, ascending.
    """
    return _fft_grid_cached()


@lru_cache(maxsize=None)
def hankel_kernel(mu: int, ln2: float = LN2_KCAP) -> np.ndarray:
    """Cached FFT kernel ``K_mu(k_m)``, m = 0..N/2, unit modulus (complex128, shape (513,)).

    ``K_mu(k) = exp(i k ln2) Gamma((1+mu)/2 + i k/2) / Gamma((1+mu)/2 - i k/2)`` with
    ``k_m = 2 pi m / (Dl N)`` (spec s7.2 step 3, bias q = 0).  The Gamma ratio of conjugate
    arguments is ``exp(2 i Im loggamma((1+mu)/2 + i k/2))`` (computed with
    ``scipy.special.loggamma`` on its continuous branch, so no cancellation / branch problems).
    The imaginary parts at m = 0 and m = N/2 of the transformed product are discarded by the
    real-signal inverse FFT (spec: "Force Im[conv_0] = Im[conv_{N/2}] = 0").
    """
    _, dlnl = fft_grid()
    m = np.arange(N_FFT // 2 + 1)
    k = 2.0 * np.pi * m / (dlnl * N_FFT)
    phase = k * ln2 + 2.0 * loggamma(0.5 * (1.0 + mu) + 0.5j * k).imag
    K = np.exp(1j * phase)
    K.setflags(write=False)
    return K


class ClInterpolator:
    """Interpolation of C(l) (rows) onto the 1024-point FFT grid (spec s7.2 step 1).

    Per row (a row is one tomographic pair):

    * all C > 0:   ln C linear in ln l between the input nodes; below ell[0] slope +1
      (C ~ l), above ell[-1] slope -3 (C ~ l^-3); entries below 1e-30 are floored to 1e-30;
    * all C < 0:   same on ln(-C), sign restored;
    * otherwise (mixed signs or any exact zero): C itself linear in ln l, constant extrapolation.

    The input nodes may be any ascending positive grid (kcap assumes log-uniform; for a log-uniform
    grid the result is identical).  The index/weight tables are built once per ell grid.
    """

    def __init__(self, ell):
        ell = _check_ell(ell, 2)
        self.ell = ell
        self.lnl = np.log(ell)
        l, _ = fft_grid()
        lnx = np.log(l)
        n = ell.size
        idx = np.clip(np.searchsorted(self.lnl, lnx) - 1, 0, n - 2)
        t = (lnx - self.lnl[idx]) / (self.lnl[idx + 1] - self.lnl[idx])
        self.idx = idx
        self.t = np.clip(t, 0.0, 1.0)
        self.below = lnx < self.lnl[0]
        self.above = lnx > self.lnl[-1]
        self.d_below = np.where(self.below, lnx - self.lnl[0], 0.0)
        self.d_above = np.where(self.above, lnx - self.lnl[-1], 0.0)

    def __call__(self, C):
        """C: (P, nell) -> (P, N_FFT) values on the FFT grid."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        if C.shape[-1] != self.ell.size:
            raise ValueError("C last axis must match ell")
        i0, t = self.idx, self.t
        allpos = np.all(C > 0, axis=-1)
        allneg = np.all(C < 0, axis=-1)
        same = allpos | allneg
        sgn = np.where(allneg, -1.0, 1.0)[:, None]
        # branch A: log-log
        lc = np.log(np.maximum(np.abs(C), 1e-30))
        lin = lc[:, i0] * (1.0 - t) + lc[:, i0 + 1] * t
        lin = np.where(self.below, lc[:, :1] + 1.0 * self.d_below, lin)
        lin = np.where(self.above, lc[:, -1:] - 3.0 * self.d_above, lin)
        out_ll = sgn * np.exp(lin)
        if same.all():
            return out_ll
        # branch B: linear in ln l, constant extrapolation
        out_lin = C[:, i0] * (1.0 - t) + C[:, i0 + 1] * t
        return np.where(same[:, None], out_ll, out_lin)


def cl_to_xi_fft(ell, C, ln2: float = LN2_KCAP, interp: ClInterpolator | None = None):
    """Faithful FFT Hankel transform of spec s7.2, batched over rows.

    Parameters
    ----------
    ell : (nell,) ascending multipoles (dimensionless).
    C : (..., nell) spectra (e.g. (15, nell)).
    ln2 : value of ln 2 in the kernel phase (kcap: 0.69314718).

    Returns
    -------
    theta : (1024,) ascending, radians, ``theta_j = 1/l_{N-1-j}``.
    xip, xim : (..., 1024), ``(1/2pi) int l C J_{0,4}(l theta) dl`` on the theta grid.

    One ``rfft`` of the (P, N) array ``f = l C(l)`` and one ``irfft`` of the stacked (2, P, N/2+1)
    products with the cached kernels K_0, K_4.
    """
    C = np.asarray(C, dtype=float)
    lead = C.shape[:-1]
    C2 = C.reshape(-1, C.shape[-1])
    if interp is None:
        interp = ClInterpolator(ell)
    l, _ = fft_grid()
    f = interp(C2) * l
    F = np.fft.rfft(f, axis=-1)                               # (P, N/2+1)
    K = np.stack([hankel_kernel(0, ln2), hankel_kernel(4, ln2)])  # (2, N/2+1)
    conv = np.fft.irfft(F[None] * K[:, None, :], n=N_FFT, axis=-1)  # (2, P, N)
    xi = (conv * (l / (2.0 * np.pi)))[..., ::-1]
    theta = 1.0 / l[::-1]
    xi = xi.reshape((2,) + lead + (N_FFT,))
    return theta, xi[0], xi[1]


# ---------------------------------------------------------------------------------------------
# theta binning of nodes (spec s8)
# ---------------------------------------------------------------------------------------------
def kcap_bin_operator(theta, edges_rad) -> np.ndarray:
    """Spec s8 (F): ``(nbin, ntheta)`` weights for the kcap node-trapezoid bin average.

    ``xi_bin[b] = trapz(theta_S xi_S, theta_S) / trapz(theta_S, theta_S)`` over the nodes
    S = {j: edges[b] <= theta_j < edges[b+1]} (partial end intervals dropped).
    """
    theta = np.asarray(theta, float)
    edges = _check_edges(edges_rad)
    W = np.zeros((edges.size - 1, theta.size))
    for b in range(edges.size - 1):
        s = np.nonzero((theta >= edges[b]) & (theta < edges[b + 1]))[0]
        if s.size < 2:
            raise ValueError(f"bin {b} contains fewer than 2 theta nodes")
        th = theta[s]
        dth = np.diff(th)
        w = np.zeros(s.size)
        w[:-1] += 0.5 * dth
        w[1:] += 0.5 * dth
        W[b, s] = w * th / (0.5 * (th[-1] ** 2 - th[0] ** 2))
    return W


def exact_bin_operator(theta, edges_rad, order: int = 16, pad: int = 6) -> np.ndarray:
    """Spec s8 (E): ``(nbin, ntheta)`` weights for the exact bin average
    ``int th xi dth / int th dth`` over the true edges, with xi a cubic (not-a-knot) spline in
    ln theta through the nodes (only nodes within ``pad`` nodes of the edge range are used).
    The weights are linear in xi, built from the spline of the identity matrix.
    """
    theta = np.asarray(theta, float)
    edges = _check_edges(edges_rad)
    lo = max(np.searchsorted(theta, edges[0]) - pad, 0)
    hi = min(np.searchsorted(theta, edges[-1]) + pad, theta.size)
    sub = np.arange(lo, hi)
    x = np.log(theta[sub])
    spl = CubicSpline(x, np.eye(sub.size), axis=0)       # evaluates to (npts, nsub)
    gx, gw = np.polynomial.legendre.leggauss(order)
    W = np.zeros((edges.size - 1, theta.size))
    for b in range(edges.size - 1):
        xa, xb = np.log(edges[b]), np.log(edges[b + 1])
        xs = 0.5 * (xb - xa) * gx + 0.5 * (xb + xa)
        ws = 0.5 * (xb - xa) * gw * np.exp(2.0 * xs)     # th dth = th^2 dln th
        W[b, sub] = ws @ spl(xs) / (0.5 * (edges[b + 1] ** 2 - edges[b] ** 2))
    return W


# ---------------------------------------------------------------------------------------------
# Exact bin-averaged Bessel functions and the fast linear operator
# ---------------------------------------------------------------------------------------------
def _f4_series(x):
    """Series of F_4(x) = int_0^x t J_4(t) dt, accurate for x < ~3."""
    x = np.asarray(x, float)
    x2 = x * x
    term = x ** 6 / (6.0 * 16.0 * 24.0)
    s = term.copy()
    for k in range(0, 24):
        term = -term * x2 * (2 * k + 6) / ((2 * k + 8) * 4.0 * (k + 1) * (k + 5))
        s += term
    return s


def bessel_antiderivative(x, mu: int, j0x=None, j1x=None):
    """``F_mu(x) = int_0^x t J_mu(t) dt`` for mu in {0, 4} (closed form, series for small x).

    mu=0: ``x J_1(x)``.  mu=4: ``x J_1 - 4 J_0 - 12 J_2 + 4`` with ``J_2 = 2 J_1/x - J_0``
    for x >= 2 (stable upward recurrence) and the power series for x < 2 (the closed form
    cancels catastrophically there).  Optional precomputed ``j0x``, ``j1x`` avoid re-evaluation.
    """
    x = np.asarray(x, float)
    if j1x is None:
        j1x = j1(x)
    if mu == 0:
        return x * j1x
    if mu != 4:
        raise ValueError("mu must be 0 or 4")
    if j0x is None:
        j0x = j0(x)
    small = x < 2.0
    xs = np.where(small, 1.0, x)
    j2 = 2.0 * j1x / xs - j0x
    big = x * j1x - 4.0 * j0x - 12.0 * j2 + 4.0
    if not small.any():
        return big
    out = big
    out = np.where(small, _f4_series(np.where(small, x, 0.0)), out)
    return out


def bessel_bin_average(ell, theta_a, theta_b, mu: int):
    """Exact theta-weighted bin average of J_mu(l theta) (shape of ``ell``)::

        Jbar = int_{th_a}^{th_b} th J_mu(l th) dth / int_{th_a}^{th_b} th dth
             = 2 [F_mu(l th_b) - F_mu(l th_a)] / (l^2 (th_b^2 - th_a^2)).

    theta in radians, ell dimensionless.  Jbar -> 1 (mu=0) as l -> 0.
    """
    ell = np.asarray(ell, float)
    return 2.0 * (bessel_antiderivative(ell * theta_b, mu) - bessel_antiderivative(ell * theta_a, mu)) \
        / (ell ** 2 * (theta_b ** 2 - theta_a ** 2))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def _panels(breaks, wmax):
    """Gauss-Legendre-8 panels covering consecutive intervals ``breaks``, each interval cut into
    ceil(len/wmax) panels.  Returns (nodes, weights, interval index per node)."""
    breaks = np.asarray(breaks, float)
    length = np.diff(breaks)
    n = np.maximum(1, np.ceil(length / wmax)).astype(np.int64)
    iv = np.repeat(np.arange(n.size), n)
    first = np.cumsum(n) - n
    j = np.arange(n.sum()) - np.repeat(first, n)
    h = length[iv] / n[iv]
    a = breaks[iv] + j * h
    nodes = a[:, None] + 0.5 * h[:, None] * (_GL_X[None, :] + 1.0)
    w = 0.5 * h[:, None] * _GL_W[None, :]
    return nodes.ravel(), w.ravel(), np.repeat(iv, _GL_X.size)


def _lagrange4(lnl, lnx, iv):
    """Local 4-point Lagrange weights in ln l: returns (start index s (M,), weights (M,4))."""
    n = lnl.size
    s = np.clip(iv - 1, 0, max(n - 4, 0))
    cols = s[:, None] + np.arange(4)[None, :]
    xn = lnl[cols]                                       # (M, 4)
    w = np.ones((lnx.size, 4))
    for a in range(4):
        for b in range(4):
            if a != b:
                w[:, a] *= (lnx - xn[:, b]) / (xn[:, a] - xn[:, b])
    return s, w


def bin_operator(ell, edges_rad, l_cap: float = L_MAX_FFT, ppp: float = 1.0,
                 tail_stretch: float = 1.0):
    """Linear operators ``M_plus, M_minus`` of shape ``(nbin, nell)``, ``xi_bin = M @ C(ell)``.

    ``M_mu[b, k] = (1/2pi) int dl l Jbar_{mu,b}(l) B_k(l)`` where B_k is the interpolation basis of
    node k (local cubic Lagrange in ln l inside [ell_0, ell_-1]; ``(l/ell_0)`` below and
    ``(l/ell_-1)^-3`` above, so the first/last columns absorb the tails), integrated with
    8-point Gauss-Legendre panels no wider than ``ppp * 2 pi / theta_b`` (one oscillation of the
    bin-edge Bessel function) from 0 to ``l_cap`` (tail panels ``tail_stretch`` times wider beyond
    ell_-1, where the integrand is negligible).

    Needs >= 4 ell nodes.  Costs O(0.5-1 s) per construction for the default 400 nodes; reuse it.
    """
    ell = _check_ell(ell, 4)
    edges = _check_edges(edges_rad)
    nb = edges.size - 1
    n = ell.size
    lnl = np.log(ell)
    Mp = np.zeros((nb, n))
    Mm = np.zeros((nb, n))
    top = max(l_cap, ell[-1])
    for b in range(nb):
        ta, tb = edges[b], edges[b + 1]
        wmax = ppp * 2.0 * np.pi / tb
        norm = 2.0 / (tb ** 2 - ta ** 2) / (2.0 * np.pi)
        # --- pieces: low tail [0, ell0], main [ell0, ell_-1] (per-node intervals), high tail
        pieces = []
        # low tail
        x, w, _ = _panels(np.array([0.0, ell[0]]), wmax)
        pieces.append(("lo", x, w, None))
        x, w, iv = _panels(ell, wmax)
        pieces.append(("mid", x, w, iv))
        if top > ell[-1]:
            x, w, _ = _panels(np.array([ell[-1], top]), wmax * tail_stretch)
            pieces.append(("hi", x, w, None))
        for kind, x, w, iv in pieces:
            xa, xb = x * ta, x * tb
            j0a, j1a, j0b, j1b = j0(xa), j1(xa), j0(xb), j1(xb)
            rp = (bessel_antiderivative(xb, 0, j0b, j1b) - bessel_antiderivative(xa, 0, j0a, j1a))
            rm = (bessel_antiderivative(xb, 4, j0b, j1b) - bessel_antiderivative(xa, 4, j0a, j1a))
            q = w * norm / x                                  # (1/2pi) l * 2 dF/(l^2 dth2) * w
            qp, qm = q * rp, q * rm
            if kind == "lo":
                g = x / ell[0]
                Mp[b, 0] += np.sum(qp * g)
                Mm[b, 0] += np.sum(qm * g)
            elif kind == "hi":
                g = (x / ell[-1]) ** -3
                Mp[b, -1] += np.sum(qp * g)
                Mm[b, -1] += np.sum(qm * g)
            else:
                s, lw = _lagrange4(lnl, np.log(x), iv)
                rows = np.repeat(np.arange(x.size), 4)
                cols = (s[:, None] + np.arange(4)[None, :]).ravel()
                P = sparse.csr_matrix((lw.ravel(), (rows, cols)), shape=(x.size, n))
                Mp[b] += P.T @ qp
                Mm[b] += P.T @ qm
    return Mp, Mm


def _require_finite(*arrays):
    """Raise instead of letting NaN/inf C_ell (or an overflow) leak silently into chi2."""
    for a in arrays:
        if not np.isfinite(a).all():
            raise ValueError("non-finite xi: the input C_ell contains NaN/inf")


# ---------------------------------------------------------------------------------------------
# user-facing class
# ---------------------------------------------------------------------------------------------
class HankelBinner:
    """C_ell (5,5,nell) -> binned xi_+/- (15, nbin) for the 15 pairs ``PAIRS``.

    Parameters
    ----------
    ell : (nell,) ascending positive multipoles at which C is supplied (fixed for the life of the
        object; for ``fidelity="fast"`` the operator is built for exactly this grid).
    theta_edges_arcmin : bin edges in arcmin; default ``exp(linspace(ln 0.5, ln 300, 10))``.
    fidelity : ``"fast"`` -> exact bin-averaged Bessel operator (spec s7.2(E)+s8(E));
        ``"kcap"`` -> FFT Hankel + node-trapezoid binning (spec s7.2(F)+s8(F)).
        The other combinations are available through :meth:`xi_binned_fft`.

    Attributes: ``edges_rad``, ``theta_fft`` (radians), ``M_plus``/``M_minus`` (fast only, (nbin,nell)),
    ``W_kcap``/``W_exact`` (theta binning weights on the FFT theta grid).
    """

    def __init__(self, ell, theta_edges_arcmin=None, fidelity: str = "fast", ln2: float = LN2_KCAP):
        if fidelity not in ("fast", "kcap"):
            raise ValueError("fidelity must be 'fast' or 'kcap'")
        self.fidelity = fidelity
        self.ell = _check_ell(ell, 4 if fidelity == "fast" else 2)   # private read-only copy
        edges = THETA_EDGES_ARCMIN if theta_edges_arcmin is None else theta_edges_arcmin
        self.theta_edges_arcmin = _check_edges(edges)
        self.edges_rad = self.theta_edges_arcmin * ARCMIN
        self.nbin = self.edges_rad.size - 1
        self.ln2 = ln2
        self.theta_fft = 1.0 / fft_grid()[0][::-1]
        self._interp = None
        self._W_kcap = None
        self._W_exact = None
        self._M = None
        if fidelity == "fast":
            self._ensure_operator()
        else:
            self._interp = ClInterpolator(self.ell)
            self._W_kcap = kcap_bin_operator(self.theta_fft, self.edges_rad)

    # -- lazy builders -------------------------------------------------------------------
    def _ensure_operator(self):
        if self._M is None:
            Mp, Mm = bin_operator(self.ell, self.edges_rad)
            self._M = (Mp, Mm)
            self._MT = np.ascontiguousarray(np.concatenate([Mp, Mm], axis=0).T)  # (nell, 2 nbin)
        return self._M

    @property
    def M_plus(self):
        return self._ensure_operator()[0]

    @property
    def M_minus(self):
        return self._ensure_operator()[1]

    @property
    def W_kcap(self):
        if self._W_kcap is None:
            self._W_kcap = kcap_bin_operator(self.theta_fft, self.edges_rad)
        return self._W_kcap

    @property
    def W_exact(self):
        if self._W_exact is None:
            self._W_exact = exact_bin_operator(self.theta_fft, self.edges_rad)
        return self._W_exact

    # -- helpers -------------------------------------------------------------------------
    @staticmethod
    def _pairs(C):
        C = np.asarray(C, float)
        if C.ndim == 3:
            if C.shape[0] != 5 or C.shape[1] != 5:
                raise ValueError("C must have shape (5, 5, nell)")
            return C[_PJ, _PI]                   # (15, nell), pair (j, i) j >= i
        if C.ndim == 2 and C.shape[0] == len(PAIRS):
            return C
        raise ValueError("C must have shape (5,5,nell) or (15,nell)")

    # -- main entry points ---------------------------------------------------------------
    def xi_binned(self, C):
        """Binned xi_+, xi_- for the 15 pairs.  ``C``: (5,5,nell) (pairs (j,i), j>=i used) or (15,nell).

        Returns ``(xip, xim)``, each (15, nbin), dimensionless, pair order ``PAIRS``.
        """
        C15 = self._pairs(C)
        if C15.shape[-1] != self.ell.size:
            raise ValueError("C last axis must match the ell grid given at construction")
        if self.fidelity == "fast":
            self._ensure_operator()
            out = C15 @ self._MT                       # (15, 2 nbin)
            _require_finite(out)
            return out[:, :self.nbin].copy(), out[:, self.nbin:].copy()
        return self.xi_binned_fft(C15, binning="kcap")

    def xi_binned_fft(self, C, binning: str = "kcap"):
        """FFT route with ``binning`` in {"kcap" (node trapezoid), "exact" (spline over true edges)}."""
        if binning not in ("kcap", "exact"):
            raise ValueError("binning must be 'kcap' or 'exact'")
        C15 = self._pairs(C)
        if self._interp is None:
            self._interp = ClInterpolator(self.ell)
        _, xip, xim = cl_to_xi_fft(self.ell, C15, ln2=self.ln2, interp=self._interp)
        W = self.W_kcap if binning == "kcap" else self.W_exact
        xip, xim = xip @ W.T, xim @ W.T
        _require_finite(xip, xim)
        return xip, xim

    def xi_theta(self, C):
        """Unbinned FFT result on the fixed theta grid: ``(theta_rad (1024,), xip (15,1024), xim)``."""
        if self._interp is None:
            self._interp = ClInterpolator(self.ell)
        theta, xip, xim = cl_to_xi_fft(self.ell, self._pairs(C), ln2=self.ln2, interp=self._interp)
        _require_finite(xip, xim)
        return theta, xip, xim
