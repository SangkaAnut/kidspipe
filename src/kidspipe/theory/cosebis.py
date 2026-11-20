"""COSEBIs (Complete Orthogonal Sets of E/B-Integrals) filters and E_n / B_n computation (spec s14.1).

Definitions (z = ln(theta/theta_min), theta in arcmin inside the filters, radians in the integrals)::

    T_+n(z) = N_n prod_{i=1}^{n+1} (z - r_{n,i})                          (roots/norms from the staged tables)
    T_-n(z) = T_+n(z) + int_0^z dy 4 T_+n(y) [ e^{2(y-z)} - 3 e^{4(y-z)} ]
    E_n = 1/2 int dtheta theta [ T_+n xi_+ + T_-n xi_- ]  =  1/(2 pi) int dl l W_n(l) C_E(l)
    B_n = 1/2 int dtheta theta [ T_+n xi_+ - T_-n xi_- ]
    W_n(l) = int_{theta_min}^{theta_max} dtheta theta T_+n(theta) J_0(l theta)

The defining conditions of T_+n (used by :func:`regenerate_roots_norms`, an independent mpmath
construction): degree n+1 polynomial in z; int_0^Z e^{2z} T dz = int_0^Z e^{4z} T dz = 0 (Z = ln(tmax/tmin));
orthonormality int dtheta T_n T_m = (theta_max - theta_min) delta_nm with theta in arcmin; positive leading
coefficient.  T is dimensionless; the *integrals* E_n, B_n use theta in RADIANS (E_n ~ 1e-10 for KV450).

Two routes to E_n are provided:

* ``E_from_cl``  : linear operator ``M[n, k]`` (cached) acting on the Limber C_ell nodes.  C is interpolated by a
  natural cubic spline in ln l of *linear* C (as kcap does) inside the model ell grid, with the power laws
  l^{+1} / l^{-3} (same as theory/hankel.py) outside it.  The l integral runs over ``[lmin, lmax]``
  (default [1, 1.146e5] = [1, 1e4/theta_max_rad], the kcap range for 300') with W_n computed by Gauss-Legendre
  panels in theta that follow the Bessel oscillation (no oscillatory-quadrature shortcut, error ~1e-10).
* ``E_from_xi``  : the theta-space formula with unbinned xi from ``HankelBinner.xi_theta`` (FFT route).

Public API: ``CosebisFilters``, ``parse_roots_table``, ``parse_norm_table``, ``regenerate_roots_norms``,
``wn_of_l``, ``cosebis_operator``, ``E_from_cl``, ``E_from_xi``.
"""
from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import j0

ARCMIN = np.pi / (180.0 * 60.0)

__all__ = ["ARCMIN", "default_cosebis_dir", "parse_roots_table", "parse_norm_table", "regenerate_roots_norms",
           "CosebisFilters", "wn_of_l", "l_quadrature", "cosebis_operator", "E_from_cl", "E_from_xi",
           "L_MAX_KCAP"]

#: kcap upper l limit for theta_max = 300': 1e4 / theta_max_rad
L_MAX_KCAP = 1.0e4 / (300.0 * ARCMIN)


def default_cosebis_dir() -> Path:
    """``$KIDSPIPE_DATA/cosebis`` if set else ``<repo>/data/cosebis``."""
    from .. import paths
    return paths.data_dir() / "cosebis"


# --------------------------------------------------------------------------------------
# table parsing and regeneration
# --------------------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


def _numbers(line: str):
    return [t.replace("D", "e").replace("d", "e") for t in _NUM.findall(line)]


def parse_roots_table(path) -> dict[int, np.ndarray]:
    """``Root_*.table``: line ``n r_1 ... r_{n+1}`` -> {n: roots (n+1,)}.  Blank/comment lines skipped; the
    leading index and the root count are checked."""
    out: dict[int, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = _numbers(s)
        n = int(float(tok[0]))
        r = np.array([float(t) for t in tok[1:]])
        if r.size != n + 1:
            raise ValueError(f"{path}: mode {n} has {r.size} roots, expected {n + 1}")
        out[n] = r
    if not out:
        raise ValueError(f"{path}: no roots found")
    return out


def parse_norm_table(path):
    """``Normalization_*.table``: line ``n norm`` -> ({n: float}, {n: decimal string at full precision})."""
    vals, strs = {}, {}
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = _numbers(s)
        n = int(float(tok[0]))
        vals[n] = float(tok[1])
        strs[n] = tok[1]
    if not vals:
        raise ValueError(f"{path}: no norms found")
    return vals, strs


def regenerate_roots_norms(nmax: int, tmin: float = 0.5, tmax: float = 300.0, dps: int = 80):
    """Independent mpmath construction of the COSEBIs roots and norms from the defining conditions.

    Unknown coefficients a_0..a_{n+1} of T_n(z) = sum a_k z^k.  Linear conditions (exact moments):
    int_0^Z e^{2z}T = 0, int_0^Z e^{4z}T = 0, int dtheta T_n T_m = 0 for m<n (dtheta = tmin e^z dz), then the
    one-dimensional null space is scaled so that int dtheta T_n^2 = tmax - tmin and the leading coefficient is
    positive.  Returns (roots {n: list of mpf (sorted)}, norms {n: mpf}, coeffs {n: list of mpf}).
    """
    import mpmath as mp
    mp.mp.dps = dps
    tmin_, tmax_ = mp.mpf(str(tmin)), mp.mpf(str(tmax))
    Z = mp.log(tmax_ / tmin_)
    kmax = 2 * (nmax + 1) + 1
    # M[k][c] = int_0^Z z^k e^{c z} dz  by  M_k = [z^k e^{cz}/c]_0^Z - (k/c) M_{k-1}
    mom = {}
    for c in (1, 2, 3, 4):   # c=1 : theta-weight; 2,4 : constraints;  c=1 needed for products (e^z z^(j+k))
        cm = mp.mpf(c)
        prev = (mp.exp(cm * Z) - 1) / cm
        mom[(0, c)] = prev
        for k in range(1, kmax + 1):
            prev = (Z ** k * mp.exp(cm * Z)) / cm - (k / cm) * prev
            mom[(k, c)] = prev
    coeffs, roots, norms = {}, {}, {}
    for n in range(1, nmax + 1):
        rows = []
        rows.append([mom[(k, 2)] for k in range(n + 2)])
        rows.append([mom[(k, 4)] for k in range(n + 2)])
        for m in range(1, n):
            am = coeffs[m]
            rows.append([tmin_ * sum(am[j] * mom[(j + k, 1)] for j in range(len(am))) for k in range(n + 2)])
        A = mp.matrix(rows)                       # (n+1) x (n+2)
        # null space: fix a_{n+1}=1, solve the (n+1)x(n+1) system for the others
        Asub = A[:, :n + 1]
        rhs = -A[:, n + 1]
        a = list(mp.lu_solve(Asub, rhs)) + [mp.mpf(1)]
        nrm2 = tmin_ * sum(a[j] * a[k] * mom[(j + k, 1)] for j in range(n + 2) for k in range(n + 2))
        scale = mp.sqrt((tmax_ - tmin_) / nrm2)
        a = [x * scale for x in a]
        coeffs[n] = a
        norms[n] = a[-1]
        rts = mp.polyroots(a[::-1], maxsteps=500, extraprec=4 * dps)
        roots[n] = sorted([mp.re(r) for r in rts])
    return roots, norms, coeffs


# --------------------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------------------
_GL = {}


def _gl(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


class CosebisFilters:
    """T_+n, T_-n for theta in [tmin, tmax] arcmin, modes n = 1..nmax (1-based, as in the tables).

    Parameters
    ----------
    tmin, tmax : arcmin.
    nmax : number of modes kept (<= number in the table).
    source : ``"table"`` (staged Root/Normalization tables) or ``"mpmath"`` (regenerated; nmax <~ 12).
    """

    def __init__(self, tmin: float = 0.5, tmax: float = 300.0, nmax: int = 20, source: str = "table",
                 data_dir=None, dps: int = 80):
        self.tmin, self.tmax = float(tmin), float(tmax)
        self.Z = math.log(self.tmax / self.tmin)
        self.source = source
        if source == "table":
            d = Path(data_dir) / "roots_norms" if data_dir is not None else default_cosebis_dir() / "roots_norms"
            tag = f"{self.tmin:.2f}-{self.tmax:.2f}"
            roots = parse_roots_table(d / f"Root_{tag}.table")
            norms, self.norm_strings = parse_norm_table(d / f"Normalization_{tag}.table")
            nmax = min(int(nmax), max(roots))
            self.roots = {n: roots[n] for n in range(1, nmax + 1)}
            self.norms = {n: norms[n] for n in range(1, nmax + 1)}
        elif source == "mpmath":
            r, nm, _ = regenerate_roots_norms(int(nmax), self.tmin, self.tmax, dps=dps)
            self.roots = {n: np.array([float(x) for x in r[n]]) for n in r}
            self.norms = {n: float(nm[n]) for n in nm}
        else:
            raise ValueError("source must be 'table' or 'mpmath'")
        self.nmax = int(nmax)

    # -- evaluation ---------------------------------------------------------------------
    def z_of_theta(self, theta_arcmin):
        return np.log(np.asarray(theta_arcmin, float) / self.tmin)

    def Tplus_z(self, z, nmodes=None) -> np.ndarray:
        """T_+n(z) for n = 1..nmodes: shape (nmodes, *z.shape)."""
        z = np.asarray(z, float)
        nmodes = self.nmax if nmodes is None else int(nmodes)
        out = np.empty((nmodes,) + z.shape)
        for n in range(1, nmodes + 1):
            r = self.roots[n]
            p = np.full(z.shape, self.norms[n])
            for ri in r:
                p = p * (z - ri)
            out[n - 1] = p
        return out

    def Tplus(self, theta_arcmin, nmodes=None) -> np.ndarray:
        return self.Tplus_z(self.z_of_theta(theta_arcmin), nmodes)

    def Tminus_z(self, z, nmodes=None, ngl: int = 96) -> np.ndarray:
        """T_-n(z) = T_+n(z) + int_0^z 4 T_+n(y)[e^{2(y-z)} - 3 e^{4(y-z)}] dy, Gauss-Legendre with ``ngl`` nodes on
        [0, z] (the integrand is polynomial x exponential, an entire function: converged to ~1e-13)."""
        z = np.asarray(z, float)
        nmodes = self.nmax if nmodes is None else int(nmodes)
        x, w = _gl(ngl)
        zf = z.ravel()
        y = 0.5 * zf[:, None] * (x[None, :] + 1.0)                 # (nz, ngl)
        kern = 4.0 * (np.exp(2.0 * (y - zf[:, None])) - 3.0 * np.exp(4.0 * (y - zf[:, None])))
        wk = 0.5 * zf[:, None] * w[None, :] * kern
        Tp_y = self.Tplus_z(y, nmodes)                              # (nmodes, nz, ngl)
        integ = np.einsum("nij,ij->ni", Tp_y, wk)
        res = self.Tplus_z(zf, nmodes) + integ
        return res.reshape((nmodes,) + z.shape)

    def Tminus(self, theta_arcmin, nmodes=None) -> np.ndarray:
        return self.Tminus_z(self.z_of_theta(theta_arcmin), nmodes)

    # -- checks -------------------------------------------------------------------------
    def gram(self, nmodes=None, ngl: int = 48, panels: int = 8) -> np.ndarray:
        """int dtheta T_n T_m [arcmin] (should be (tmax - tmin) delta_nm)."""
        nmodes = self.nmax if nmodes is None else int(nmodes)
        z, w = self._z_quad(ngl, panels)
        T = self.Tplus_z(z, nmodes)
        return (T * (w * self.tmin * np.exp(z))) @ T.T

    def constraints(self, nmodes=None, ngl: int = 48, panels: int = 8):
        """(int e^{2z}T_n dz, int e^{4z}T_n dz, scale) for n=1..nmodes; scale = int |T| e^{4z} dz for relative size."""
        nmodes = self.nmax if nmodes is None else int(nmodes)
        z, w = self._z_quad(ngl, panels)
        T = self.Tplus_z(z, nmodes)
        return T @ (w * np.exp(2 * z)), T @ (w * np.exp(4 * z)), np.abs(T) @ (w * np.exp(4 * z))

    def _z_quad(self, ngl, panels):
        x, w = _gl(ngl)
        e = np.linspace(0.0, self.Z, panels + 1)
        z = (0.5 * (e[1:, None] - e[:-1, None]) * (x[None, :] + 1.0) + e[:-1, None]).ravel()
        ww = (0.5 * (e[1:, None] - e[:-1, None]) * w[None, :]).ravel()
        return z, ww


# --------------------------------------------------------------------------------------
# W_n(l)
# --------------------------------------------------------------------------------------
def _theta_nodes(l_ref: float, filt: CosebisFilters, n_log_panels: int, ngl: int, dphi: float):
    """Gauss-Legendre nodes/weights in theta [rad] resolving Bessel phase up to ``l_ref``: ``n_log_panels``
    log-spaced base panels, each sub-divided so that no sub-panel carries more than ``dphi`` radians of l_ref*theta."""
    x, w = _gl(ngl)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    th_min = filt.tmin * ARCMIN
    edges = th_min * np.exp(np.linspace(0.0, filt.Z, n_log_panels + 1))
    nodes, wts = [], []
    for j in range(n_log_panels):
        a, b = edges[j], edges[j + 1]
        m = max(1, int(math.ceil(l_ref * (b - a) / dphi)))
        e = a + (b - a) * np.arange(m + 1) / m
        h = np.diff(e)
        nodes.append((e[:-1, None] + h[:, None] * x[None, :]).ravel())
        wts.append((h[:, None] * w[None, :]).ravel())
    return np.concatenate(nodes), np.concatenate(wts), th_min


def wn_of_l(l, filt: CosebisFilters, nmodes: int = 5, n_log_panels: int = 24, ngl: int = 10, dphi: float = 6.0,
            nthreads: int = 4, chunk: int = 128) -> np.ndarray:
    """W_n(l) = int dtheta theta T_+n(theta) J_0(l theta) [theta in rad], shape (nmodes, len(l)).

    l values are grouped in octave tiers; each tier shares one fixed set of theta nodes sized for the tier's upper l
    (phase per sub-panel <= ``dphi`` rad, GL-``ngl``: quadrature error ~ (dphi/2)^(2 ngl)/(2 ngl)! ~ 1e-9), so T_+n is
    evaluated once per tier and J_0 on (l x theta) blocks.
    """
    l = np.atleast_1d(np.asarray(l, float))
    out = np.empty((nmodes, l.size))
    tier = np.maximum(0, np.ceil(np.log2(np.maximum(l, 1e-300) / 8.0))).astype(int)
    jobs = []
    for k in np.unique(tier):
        idx = np.where(tier == k)[0]
        th, wt, th_min = _theta_nodes(8.0 * 2.0 ** k, filt, n_log_panels, ngl, dphi)
        A = filt.Tplus_z(np.log(th / th_min), nmodes) * (wt * th)[None, :]       # (nmodes, nodes)
        for c in range(0, idx.size, chunk):
            jobs.append((idx[c:c + chunk], th, A))

    def run(job):
        ii, th, A = job
        out[:, ii] = A @ j0(np.outer(th, l[ii]))
    if nthreads <= 1 or len(jobs) == 1:
        for jb in jobs:
            run(jb)
    else:
        with ThreadPoolExecutor(nthreads) as ex:
            list(ex.map(run, jobs))
    return out


def l_quadrature(lmin: float, lmax: float, dl_max: float = 24.0, rel: float = 0.1, ngl: int = 5):
    """Composite Gauss-Legendre cells on [lmin, lmax]: width min(rel*l, dl_max).  Returns (l_nodes, weights).
    dl_max = 24 resolves the 2pi/theta_max_rad = 72 oscillation period of W_n(l) with 3 cells per period."""
    edges = [float(lmin)]
    while edges[-1] < lmax:
        edges.append(min(lmax, edges[-1] + min(rel * edges[-1], dl_max)))
    e = np.array(edges)
    x, w = _gl(ngl)
    h = np.diff(e)
    l = (e[:-1, None] + 0.5 * h[:, None] * (x[None, :] + 1.0)).ravel()
    ww = (0.5 * h[:, None] * w[None, :]).ravel()
    return l, ww


_OP_CACHE: dict = {}


def _interp_matrix(ell, l_nodes):
    """S[i, k] with C(l_i) = sum_k S[i,k] C_k : natural cubic spline in ln l of linear C inside [ell_0, ell_-1];
    outside: C_0 (l/l_0)^{+1} below, C_-1 (l/l_-1)^{-3} above (as theory/hankel.py)."""
    ell = np.asarray(ell, float)
    lnl = np.log(ell)
    n = ell.size
    cs = CubicSpline(lnl, np.eye(n), bc_type="natural", axis=0)
    ln_x = np.log(l_nodes)
    S = np.empty((l_nodes.size, n))
    inside = (l_nodes >= ell[0]) & (l_nodes <= ell[-1])
    S[inside] = cs(ln_x[inside])
    lo, hi = l_nodes < ell[0], l_nodes > ell[-1]
    S[lo] = 0.0
    S[lo, 0] = l_nodes[lo] / ell[0]
    S[hi] = 0.0
    S[hi, -1] = (l_nodes[hi] / ell[-1]) ** (-3.0)
    return S


def cosebis_operator(ell, filt: CosebisFilters, nmodes: int = 5, lmin: float = 1.0, lmax: float | None = None,
                     dl_max: float = 24.0, cache_dir=None, nthreads: int = 4, use_cache: bool = True) -> np.ndarray:
    """Linear operator M (nmodes, nell): E_n = sum_k M[n,k] C_k  (one C-spectrum) -- cached.

    E_n = 1/(2pi) int_{lmin}^{lmax} dl l W_n(l) C(l); lmax default = 1e4/theta_max_rad (kcap).
    Cache key: (ell grid, tmin, tmax, nmodes, lmin, lmax, dl_max, roots source); disk cache in ``cache_dir``
    (default ``<cosebis data>/cache``) as .npz, because W_n at ~25000 l nodes costs ~10-20 s.
    """
    ell = np.asarray(ell, float)
    lmax = 1.0e4 / (filt.tmax * ARCMIN) if lmax is None else float(lmax)
    key = (ell.size, float(ell[0]), float(ell[-1]), filt.tmin, filt.tmax, int(nmodes), float(lmin), lmax,
           float(dl_max), filt.source)
    if use_cache and key in _OP_CACHE:
        return _OP_CACHE[key]
    fn = None
    if use_cache:
        cd = Path(cache_dir) if cache_dir is not None else default_cosebis_dir() / "cache"
        name = "W_" + "_".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in key) + ".npz"
        fn = cd / name
        if fn.exists():
            try:
                with np.load(fn) as f:
                    W, l = f["W"], f["l"]
                    ww = f["w"]
                S = _interp_matrix(ell, l)
                M = ((W * (ww * l / (2 * np.pi))[None, :]) @ S)
                _OP_CACHE[key] = M
                return M
            except Exception:
                pass
    l, ww = l_quadrature(lmin, lmax, dl_max=dl_max)
    W = wn_of_l(l, filt, nmodes, nthreads=nthreads)
    if fn is not None:
        try:
            fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fn, W=W, l=l, w=ww)
        except OSError:
            pass
    S = _interp_matrix(ell, l)
    M = (W * (ww * l / (2 * np.pi))[None, :]) @ S
    if use_cache:
        _OP_CACHE[key] = M
    return M


def E_from_cl(C, M: np.ndarray) -> np.ndarray:
    """E_n from C_ell.  C : (nell,), (15, nell) or (5,5,nell) -> (nmodes,), (15, nmodes), (5,5,nmodes) (using
    the same pair selection as the rest of the code for (5,5,nell) -> (15, nmodes) in ``PAIRS`` order)."""
    C = np.asarray(C, float)
    if C.ndim == 3:
        from .hankel import _PI, _PJ
        C = C[_PJ, _PI]
    return C @ M.T


# --------------------------------------------------------------------------------------
# xi route
# --------------------------------------------------------------------------------------
def E_from_xi(theta_rad, xip, xim, filt: CosebisFilters, nmodes: int = 5, panels: int = 24, ngl: int = 16):
    """(E_n, B_n) = 1/2 int dtheta theta [T_+n xi_+ +- T_-n xi_-] from unbinned xi on a log-theta grid.

    theta_rad : (nt,) ascending radians; xip, xim : (..., nt).  xi is interpolated by a cubic spline in ln theta
    onto Gauss-Legendre nodes in z = ln(theta/theta_min) (dtheta theta = theta^2 dz).  Returns (E, B) each (..., nmodes).
    """
    theta_rad = np.asarray(theta_rad, float)
    lo, hi = filt.tmin * ARCMIN, filt.tmax * ARCMIN
    if theta_rad[0] > lo or theta_rad[-1] < hi:
        raise ValueError("theta grid does not cover [tmin, tmax]")
    z, w = filt._z_quad(ngl, panels)
    th = lo * np.exp(z)
    lt = np.log(theta_rad)
    xp = CubicSpline(lt, np.asarray(xip, float), axis=-1)(np.log(th))
    xm = CubicSpline(lt, np.asarray(xim, float), axis=-1)(np.log(th))
    Tp = filt.Tplus_z(z, nmodes)
    Tm = filt.Tminus_z(z, nmodes)
    wt = 0.5 * w * th ** 2
    ap = xp @ (Tp * wt).T
    am = xm @ (Tm * wt).T
    return ap + am, ap - am
