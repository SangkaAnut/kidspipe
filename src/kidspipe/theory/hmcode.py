r"""Stand-alone numba HMcode-2016 ("mead") non-linear matter power spectrum.

Implements the halo-model fitting scheme of Mead et al. (2015, MNRAS 454, 1958; 2016,
MNRAS 459, 1468) with the one-parameter baryon model (``A_baryon`` in the halo concentration,
``eta0`` in the window-function mass dependence), see ``docs/MASTER_MATH_SPEC.md`` s2.5.
It reproduces CAMB's ``halofit_version='mead'`` (HMcode-2016) at the few 1e-4 level for
LCDM + one massive neutrino, using only the linear power spectra as input (no CAMB import).

Units: wavenumbers k in h/Mpc, radii R in Mpc/h, power P in (Mpc/h)^3; the dimensionless
spectrum is ``Delta^2(k) = k^3 P(k) / (2 pi^2)``.

Model (spec s2.5; ``cb`` = CDM+baryons, ``tot`` = all matter)::

    Delta2_HM^alpha = Delta2_2h^alpha + Delta2_1h^alpha
    alpha  = 3.24 * 1.85^neff                          (clipped to [0.5, 2])
    Delta2_2h = Delta2_lin,tot [1 - f tanh^2(k sigma_v / sqrt f)],  f = 0.0095 sigma_v100^1.37
    sigma_v^2 = 1/3 int dlnk Delta2_lin,tot / k^2 ,  sigma_v100^2 same with W_th(100 k)^2
    Delta2_1h = [1 - exp(-(k/k*)^2)] (2 Delta_v k^3 / 3 pi) int dnu g(nu) R_v^3 |W(nu^eta k R_v/c ; c)|^2
    k* = 0.584 / sigma_v ;  eta = eta0 - 0.3 sigma_8(z)
    Delta_v = 418 Om_m(z)^-0.352 (1 + 0.916 f_nu)
    delta_c = [1.59 + 0.0314 ln sigma_8(z)] (1 + 0.262 f_nu) [1 + 0.0123 log10 Om_m(z)]
    nu = delta_c / sigma_cb(R, z),   g(nu) = 0.21616 [1 + (0.707 nu^2)^-0.3] exp(-0.707 nu^2 / 2)
    R_v = R / Delta_v^(1/3),  c = A_baryon (1+z_f)/(1+z),  g(z_f) = delta_c g(z) / sigma_cb(0.01^(1/3) R, z)
    neff = -3 - dln sigma_cb^2 / dlnR |_{nu(R)=1}     (clipped to [n_s-4, n_s])

Implementation choices where the spec is silent (all validated against CAMB 1.5.8, see
``benchmarks/hmcode_vs_camb.py``):

* sigma(R) = sqrt(int dlnk Delta2 W_th(kR)^2) is evaluated on a *dense uniform ln k grid*
  (spacing ``dl``) obtained by local cubic (4-point Lagrange) interpolation of ln Delta2 between
  the native k nodes; above the last native node ln Delta2 is continued as a power law whose slope
  is the slope between the last two nodes (empirically the best match to CAMB 1.5.8), out to ``K_EXT`` = 5e6 h/Mpc (needed for the
  sigma(R) of the lightest halos).  Below the first native node Delta2 is dropped (<1e-9 effect).
  The (R x k) quadrature matrix is z-independent, so sigma(R_j, z) for all z is one matrix product.
* R is gridded uniformly in ln R with ``ms`` nodes per factor 0.01^(1/3) in radius (i.e. per factor 100
  in mass), so sigma(0.01^(1/3) R) is an exact grid shift.  Halo masses ``M = 4 pi/3 rho_bar R^3`` are
  restricted to ``[mmin, mmax]`` = [1, 1e18] Msun/h (rho_bar = 2.775e11 Om_m h^2 Msun/Mpc^3 -> Msun/h
  per (Mpc/h)^3), the range of the spec.
* The nu-integral is done as a sum over ln R nodes with the *analytic-derivative* Jacobian
  dnu/dlnR = -nu dln sigma/dlnR (4th-order central differences of ln sigma on the grid), which converges
  much faster than trapezoid-in-nu.
* The linear growth factor g(z) (for the Bullock formation redshift and Om_m(z)) is the exact flat
  matter+Lambda solution D(a) = a 2F1(1/3, 1; 11/6; -(1-Om_m)/Om_m a^3), normalised to g(0)=1; radiation
  is kept in E(z) but not in the growth ODE; the Dolag et al. dark-energy correction is
  (g_w/g_LCDM)^1.5 = 1 for w = -1 (argument ``dolag`` to override).
* sigma_8(z), sigma_v, sigma_v100 use the TOTAL matter spectrum; sigma(R) of the halo model and neff use the
  COLD (CDM+baryon) spectrum (this matters at the 1% level for m_nu = 0.06 eV).
* If nu(R) = 1 has no root in the allowed mass range the nonlinear radius is clamped to the end of the range
  and neff is the log slope there (reproduces CAMB in the low-amplitude corners).
* ``kmin_nl`` (default 0): CAMB 1.5.8 applies the 2-halo damping smoothly down to the smallest k, so no
  hard switch at Min_kh_nonlinear = 0.005 is applied unless requested.
* The NFW window uses the exact Ci/Si expression through the non-oscillatory auxiliary functions
  f, g:  W m(c) = g(x) + [f(y) - 1/y] sin(c x) - g(y) cos(c x),  x = k r_s, y = (1+c) x,
  m(c) = ln(1+c) - c/(1+c); f, g are tabulated (linear in ln z) with an asymptotic series for z > 60.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit, prange

#: fastmath without 'reassoc': lets LLVM use FMA/approximate functions but not reorder sums, so results do not depend on
#: thread count, memory alignment or the order of the redshifts (bitwise; numba >= 0.60 otherwise differs by ~1 ulp)
_FASTMATH = {"nnan", "ninf", "nsz", "arcp", "contract", "afn"}

__all__ = ["hmcode_pnl", "pk_cb_approx", "hmcode_pnl_diag", "nfw_window", "growth_table", "omega_m_of_z"]

# ----------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------
RHO_CRIT_H = 2.775e11          # critical density today [h^2 Msun / Mpc^3]
K_EXT = 5.0e6                  # h/Mpc, upper end of dense k grid (power-law continuation)
R0 = 1.0e-5                    # Mpc/h, smallest radius of the sigma(R) grid
R_MAX = 400.0                  # Mpc/h, largest radius of the sigma(R) grid
F_RADIUS = 0.01 ** (1.0 / 3.0)  # radius ratio of the formation mass M_f = 0.01 M
LN_F = math.log(F_RADIUS)
Z_ASYM = 60.0                  # switch to asymptotic series for the auxiliary functions f, g
FG_LNZ0 = math.log(1.0e-6)
FG_DU = 0.001


# ----------------------------------------------------------------------------------------------
# auxiliary tables for the NFW window  (Ci, Si  ->  non-oscillatory f, g)
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _fg_tables():
    """Tables F(z) = z f(z) and G(z) = z^2 g(z) on a uniform ln z grid.

    f(z) = Ci(z) sin z + (pi/2 - Si(z)) cos z,  g(z) = -Ci(z) cos z + (pi/2 - Si(z)) sin z
    (so Ci = f sin - g cos, Si = pi/2 - f cos - g sin).  Asymptotically F, G -> 1.
    """
    from scipy.special import sici

    n = int(math.ceil((math.log(Z_ASYM) - FG_LNZ0) / FG_DU)) + 2
    lnz = FG_LNZ0 + FG_DU * np.arange(n)
    z = np.exp(lnz)
    si, ci = sici(z)
    f = ci * np.sin(z) + (0.5 * np.pi - si) * np.cos(z)
    g = -ci * np.cos(z) + (0.5 * np.pi - si) * np.sin(z)
    return np.ascontiguousarray(z * f), np.ascontiguousarray(z * z * g)


@njit(cache=True, inline="always")
def _fg_eval(lnz, z, ftab, gtab):
    """Return (F, G) = (z f(z), z^2 g(z)) by table lookup or asymptotic series (z > Z_ASYM)."""
    if z > Z_ASYM:
        iz2 = 1.0 / (z * z)
        # F = sum (-1)^n (2n)! z^-2n ; G = sum (-1)^n (2n+1)! z^-2n
        fa = 1.0 - iz2 * (2.0 - iz2 * (12.0 - iz2 * (120.0 - iz2 * (1680.0 - iz2 * 30240.0))))
        ga = 1.0 - iz2 * (6.0 - iz2 * (60.0 - iz2 * (840.0 - iz2 * (15120.0 - iz2 * 332640.0))))
        return fa, ga
    t = (lnz - FG_LNZ0) / FG_DU
    if t < 0.0:
        t = 0.0
    i = int(t)
    w = t - i
    return ftab[i] + w * (ftab[i + 1] - ftab[i]), gtab[i] + w * (gtab[i + 1] - gtab[i])


@njit(cache=True)
def _nfw_w(x, c, ftab, gtab):
    """NFW window W(x; c), x = k r_s, clipped to [0, 1] (spec s2.5)."""
    y = (1.0 + c) * x
    if y < 5.0e-3:
        return 1.0
    mc = math.log(1.0 + c) - c / (1.0 + c)
    lnx = math.log(x)
    _, gx = _fg_eval(lnx, x, ftab, gtab)
    fy, gy = _fg_eval(lnx + math.log1p(c), y, ftab, gtab)
    s = math.sin(c * x)
    co = math.cos(c * x)
    w = (gx / (x * x) + ((fy - 1.0) / y) * s - (gy / (y * y)) * co) / mc
    return min(max(w, 0.0), 1.0)


def nfw_window(x, c):
    """Normalised NFW Fourier transform W(k r_s; c), spec s2.5 (vectorised numpy wrapper).

    ``W = [cos x (Ci((1+c)x) - Ci x) + sin x (Si((1+c)x) - Si x) - sin(c x)/((1+c) x)] / [ln(1+c) - c/(1+c)]``
    with x = k R_v / c, clipped to [0, 1].  Evaluated through non-oscillatory auxiliary functions
    (see module docstring); accuracy ~1e-6 absolute.
    """
    ftab, gtab = _fg_tables()
    xb, cb = np.broadcast_arrays(np.asarray(x, float), np.asarray(c, float))
    out = np.empty(xb.shape)
    _nfw_vec(np.array(xb, dtype=float).ravel(), np.array(cb, dtype=float).ravel(), out.reshape(-1), ftab, gtab)
    return out


@njit(cache=True)
def _nfw_vec(x, c, out, ftab, gtab):
    for i in range(x.size):
        out[i] = _nfw_w(x[i], c[i], ftab, gtab)


# ----------------------------------------------------------------------------------------------
# background / growth helpers (numpy)
# ----------------------------------------------------------------------------------------------
def omega_m_of_z(z, omega_m, omega_rad=0.0):
    """Om_m(z) = Om_m (1+z)^3 / E^2(z) for flat LCDM (+radiation), E^2 = Om_m(1+z)^3 + Om_r (1+z)^4 + 1-Om_m-Om_r.

    ``omega_m`` is the total late-time matter density parameter (CDM + baryons + massive nu), spec s2.3.
    """
    z = np.asarray(z, float)
    e2 = omega_m * (1 + z) ** 3 + omega_rad * (1 + z) ** 4 + (1.0 - omega_m - omega_rad)
    return omega_m * (1 + z) ** 3 / e2


def _growth(a, omega_m):
    from scipy.special import hyp2f1

    a = np.asarray(a, float)
    d = a * hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -(1.0 - omega_m) / omega_m * a ** 3)
    d1 = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -(1.0 - omega_m) / omega_m)
    return d / d1


def growth_table(omega_m, n=1200, a_min=1.0e-5):
    """Linear growth g(a) = D(a)/D(1) of flat matter+Lambda (hypergeometric closed form).

    D(a) = a 2F1(1/3, 1; 11/6; -(1-Om_m)/Om_m a^3).  Returns (ln a, ln g) on a uniform ln a grid
    in [ln a_min, 0] (g is monotonic, so the pair is directly invertible by interpolation).
    """
    lna = np.linspace(math.log(a_min), 0.0, n)
    return lna, np.log(_growth(np.exp(lna), omega_m))


def pk_cb_approx(z, k_h, pk_lin, omega_m, omega_nu_frac):
    """Approximate linear CDM+baryon spectrum from the total one (fallback when CAMB's is unavailable).

    ``P_cb = P_tot * [1 + ((1-f_nu)^-2 - 1) S(k,z)]`` with ``S = 1 - exp(-(k/k_t)^0.8265)``,
    ``k_t = 0.0278 (1+z)^-0.434 sqrt(Om_m/0.3)`` h/Mpc (free-streaming transition), fitted to CAMB for
    m_nu = 0.06 eV, 0.05 <= Om_m <= 0.5, z <= 5: max abs error of the ratio 1.4e-3 (rms 2e-4), i.e. ~10 %
    of the ~1 % effect.  Prefer the exact spectrum (``get_linear_matter_power_spectrum(var1=var2='delta_nonu')``).
    """
    z = np.atleast_1d(np.asarray(z, float))[:, None]
    k = np.asarray(k_h, float)[None, :]
    f = float(omega_nu_frac)
    if f <= 0.0:
        return np.array(pk_lin, float)
    kt = 0.0278 * (1.0 + z) ** (-0.4342) * math.sqrt(omega_m / 0.3)
    s = 1.0 - np.exp(-((k / kt) ** 0.8265))
    return np.asarray(pk_lin, float) * (1.0 + ((1.0 - f) ** -2 - 1.0) * s)


# ----------------------------------------------------------------------------------------------
# k-grid dependent precomputation (cached)
# ----------------------------------------------------------------------------------------------
def _wth(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    big = x > 5.0e-2
    xb = x[big]
    out[big] = 3.0 * (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
    xs = x[~big]
    out[~big] = 1.0 - xs ** 2 / 10.0 + xs ** 4 / 280.0 - xs ** 6 / 15120.0
    return out


class _Grid:
    """z-independent operators for one native k grid: dense ln k grid, 4-point Lagrange weights,
    window-quadrature matrices WR[j, d] = dl * W_th(q_d R_j)^2 and the vectors for sigma_8, sigma_v, sigma_v100."""

    def __init__(self, k_h, ms, dl):
        k = np.asarray(k_h, float)
        lk = np.log(k)
        self.k, self.lk = k, lk
        self.ms, self.dl = ms, dl
        n_dense = int(math.floor((math.log(K_EXT) - lk[0]) / dl)) + 1
        lq = lk[0] + dl * np.arange(n_dense)
        self.lq, self.q = lq, np.exp(lq)
        nin = int(np.searchsorted(lq, lk[-1] * (1 + 1e-14), side="right"))
        self.nin = nin
        # 4-point Lagrange stencil on the nonuniform native grid
        pos = np.searchsorted(lk, lq[:nin], side="right") - 1
        i0 = np.clip(pos - 1, 0, k.size - 4)
        w = np.ones((nin, 4))
        xs = lq[:nin, None]
        nodes = lk[i0[:, None] + np.arange(4)[None, :]]
        for m in range(4):
            for j in range(4):
                if j != m:
                    w[:, m] *= (xs[:, 0] - nodes[:, j]) / (nodes[:, m] - nodes[:, j])
        self.i0 = i0.astype(np.int64)
        self.w = np.ascontiguousarray(w)
        # radius grid (uniform in ln R); ms nodes per factor F_RADIUS
        self.dR = -LN_F / ms
        self.nR = int(math.ceil(math.log(R_MAX / R0) / self.dR)) + 1
        self.lnR = math.log(R0) + self.dR * np.arange(self.nR)
        R = np.exp(self.lnR)
        self.R = R
        # quadrature matrix (nR x N)
        self.WR = np.ascontiguousarray(dl * _wth(np.outer(R, self.q)) ** 2)
        self.W8 = dl * _wth(8.0 * self.q) ** 2
        self.W100 = dl * _wth(100.0 * self.q) ** 2 / (3.0 * self.q ** 2)
        self.Wv = dl / (3.0 * self.q ** 2)


@lru_cache(maxsize=6)
def _grid_cached(key, ms, dl):
    return _Grid(np.frombuffer(key, dtype=np.float64), ms, dl)


def _get_grid(k_h, ms, dl):
    k = np.ascontiguousarray(k_h, dtype=np.float64)
    return _grid_cached(k.tobytes(), int(ms), float(dl))


# ----------------------------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=_FASTMATH)
def _dense_dsq(lnd2, i0, w, nin, lq, lk_last, slope, out):
    """exp of interpolated ln Delta^2 on the dense grid (inside) / power-law continuation (outside)."""
    for d in range(nin):
        i = i0[d]
        v = w[d, 0] * lnd2[i] + w[d, 1] * lnd2[i + 1] + w[d, 2] * lnd2[i + 2] + w[d, 3] * lnd2[i + 3]
        out[d] = math.exp(v)
    last = lnd2[lnd2.size - 1]
    for d in range(nin, lq.size):
        out[d] = math.exp(last + slope * (lq[d] - lk_last))


@njit(cache=True, fastmath=_FASTMATH)
def _dot(a, b):
    s = 0.0
    for i in range(a.size):
        s += a[i] * b[i]
    return s


@njit(cache=True, fastmath=_FASTMATH)
def _hm_one_z(k, lk, pk_t, pk_c, i0, w, nin, lq, WR, W8, W100, Wv,
              lnR, dR, ia, ib, ms, om_z, g_z, z_z, f_nu, n_s, A_b, eta0, dolag,
              lna_t, lng_t, ftab, gtab, kmin_nl, tol_win, out, diag):
    nk = k.size
    nR = lnR.size
    N = lq.size
    pi = math.pi
    # --- dimensionless spectra (log) -------------------------------------------------------
    lnd_t = np.empty(nk)
    lnd_c = np.empty(nk)
    for j in range(nk):
        pref = k[j] ** 3 / (2.0 * pi * pi)
        lnd_t[j] = math.log(pk_t[j] * pref)
        lnd_c[j] = math.log(pk_c[j] * pref)
    lk_last = lk[nk - 1]
    # continuation slope above the last native node: from the LAST TWO nodes (empirically the best
    # match to CAMB 1.5.8 'mead').  A 3-node least-squares slope was tried and is worse
    # against CAMB (up to 4.6e-3 instead of 0.9e-3 at z>=2 in the Om=0.05 corner).
    s_t = (lnd_t[nk - 1] - lnd_t[nk - 2]) / (lk[nk - 1] - lk[nk - 2])
    s_c = (lnd_c[nk - 1] - lnd_c[nk - 2]) / (lk[nk - 1] - lk[nk - 2])
    dt = np.empty(N)
    dc = np.empty(N)
    _dense_dsq(lnd_t, i0, w, nin, lq, lk_last, s_t, dt)
    _dense_dsq(lnd_c, i0, w, nin, lq, lk_last, s_c, dc)
    # --- scalar integrals (total matter) ---------------------------------------------------
    sig8 = math.sqrt(_dot(W8, dt))
    sv = math.sqrt(_dot(Wv, dt))
    sv100 = math.sqrt(_dot(W100, dt))
    fdamp = min(max(0.0095 * sv100 ** 1.37, 1.0e-3), 0.99)
    # --- halo-model parameters --------------------------------------------------------------
    Dv = 418.0 * om_z ** (-0.352) * (1.0 + 0.916 * f_nu)
    dcr = (1.59 + 0.0314 * math.log(sig8)) * (1.0 + 0.262 * f_nu) * (1.0 + 0.0123 * math.log10(om_z))
    eta = eta0 - 0.3 * sig8
    kstar = 0.584 / sv
    # --- sigma_cb(R) on the grid ------------------------------------------------------------
    # nodes needed: [ia - ms, ib]
    lsg = np.zeros(nR)
    j_lo = max(ia - ms - 2, 0)
    j_hi = min(ib + 2, nR - 1)
    for j in range(j_lo, j_hi + 1):
        lsg[j] = 0.5 * math.log(_dot(WR[j], dc))
    # dln sigma / dlnR : 4th-order central differences, 2nd order one-sided at the ends
    dsg = np.zeros(nR)
    for j in range(j_lo + 2, j_hi - 1):
        dsg[j] = (-lsg[j + 2] + 8.0 * lsg[j + 1] - 8.0 * lsg[j - 1] + lsg[j - 2]) / (12.0 * dR)
    dsg[j_lo] = (-3.0 * lsg[j_lo] + 4.0 * lsg[j_lo + 1] - lsg[j_lo + 2]) / (2.0 * dR)
    dsg[j_lo + 1] = (lsg[j_lo + 2] - lsg[j_lo]) / (2.0 * dR)
    dsg[j_hi] = (3.0 * lsg[j_hi] - 4.0 * lsg[j_hi - 1] + lsg[j_hi - 2]) / (2.0 * dR)
    dsg[j_hi - 1] = (lsg[j_hi] - lsg[j_hi - 2]) / (2.0 * dR)
    # --- nonlinear radius (nu = 1) and effective index --------------------------------------
    lndc = math.log(dcr)
    jr = -1
    for j in range(ia, ib):
        if lndc - lsg[j] < 0.0 <= lndc - lsg[j + 1]:   # ln nu goes from <0 to >=0
            jr = j
            break
    if jr < 0:
        if lndc - lsg[ia] >= 0.0:
            dlns = dsg[ia]            # nu > 1 everywhere: R_nl below the mass range
        else:
            dlns = dsg[ib]            # nu < 1 everywhere
    else:
        # cubic Hermite for y(t) = ln nu on [jr, jr+1]
        y0 = lndc - lsg[jr]
        y1 = lndc - lsg[jr + 1]
        m0 = -dsg[jr] * dR
        m1 = -dsg[jr + 1] * dR
        t = -y0 / (y1 - y0)
        for _ in range(6):
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            yv = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
            dy = (6 * t2 - 6 * t) * y0 + (3 * t2 - 4 * t + 1) * m0 + (-6 * t2 + 6 * t) * y1 + (3 * t2 - 2 * t) * m1
            if dy == 0.0:
                break
            t -= yv / dy
            t = min(max(t, 0.0), 1.0)
        t2 = t * t
        dy = (6 * t2 - 6 * t) * y0 + (3 * t2 - 4 * t + 1) * m0 + (-6 * t2 + 6 * t) * y1 + (3 * t2 - 2 * t) * m1
        dlns = -dy / dR   # dln sigma/dlnR = - dln nu/dlnR
    neff = -3.0 - 2.0 * dlns
    neff = min(max(neff, n_s - 4.0), n_s)
    alpha = min(max(3.24 * 1.85 ** neff, 0.5), 2.0)
    # --- halo nodes: Jacobian weights, window scales, concentrations --------------------------
    nh = ib - ia + 1
    wt = np.zeros(nh)      # g(nu) Rv^3 (dnu/dlnR) dlnR
    lnsc = np.zeros(nh)    # ln( nu^eta R_v / c ) = ln r_s,eff
    cc = np.zeros(nh)
    l1c = np.zeros(nh)
    mcc = np.zeros(nh)
    lng_z = math.log(g_z)
    lna_z = -math.log1p(z_z)
    Dv13 = Dv ** (1.0 / 3.0)
    total = 0.0
    for ii in range(nh):
        j = ia + ii
        nu = dcr / math.exp(lsg[j])
        ltgt = lndc + lng_z - lsg[j - ms]          # ln[ delta_c g(z) / sigma(0.01^(1/3) R, z) ]
        if ltgt >= lng_z:
            cval = A_b * dolag                      # formation later than z -> z_f = z
        else:
            lnaf = np.interp(ltgt, lng_t, lna_t)    # g(a_f) = target
            cval = A_b * dolag * math.exp(lna_z - lnaf)
        cc[ii] = cval
        l1c[ii] = math.log1p(cval)
        mcc[ii] = l1c[ii] - cval / (1.0 + cval)
        Rv = math.exp(lnR[j]) / Dv13
        gnu = 0.21616 * (1.0 + (0.707 * nu * nu) ** (-0.3)) * math.exp(-0.5 * 0.707 * nu * nu)
        wt[ii] = gnu * Rv * Rv * Rv * (-nu * dsg[j]) * dR
        lnsc[ii] = eta * math.log(nu) + math.log(Rv) - math.log(cval)
        total += wt[ii]
    # prune negligible tails of the nu-integral (W^2 <= 1)
    i_lo = 0
    acc = 0.0
    while i_lo < nh - 1 and acc + wt[i_lo] < tol_win * total:
        acc += wt[i_lo]
        i_lo += 1
    i_hi = nh - 1
    acc = 0.0
    while i_hi > i_lo and acc + wt[i_hi] < tol_win * total:
        acc += wt[i_hi]
        i_hi -= 1
    # --- assemble P(k) ---------------------------------------------------------------------------
    inv_alpha = 1.0 / alpha
    sqf = math.sqrt(fdamp)
    for j in range(nk):
        kj = k[j]
        if kj < kmin_nl:
            out[j] = pk_t[j]
            continue
        lkj = lk[j]
        acc = 0.0
        for ii in range(i_lo, i_hi + 1):
            lnx = lkj + lnsc[ii]
            x = math.exp(lnx)
            c = cc[ii]
            y = (1.0 + c) * x
            if y < 5.0e-3:
                acc += wt[ii]
                continue
            _, gx = _fg_eval(lnx, x, ftab, gtab)
            fy, gy = _fg_eval(lnx + l1c[ii], y, ftab, gtab)
            cx = c * x
            wv = (gx / (x * x) + ((fy - 1.0) / y) * math.sin(cx) - (gy / (y * y)) * math.cos(cx)) / mcc[ii]
            wv = min(max(wv, 0.0), 1.0)
            acc += wt[ii] * wv * wv
        kk = kj / kstar
        d1 = (1.0 - math.exp(-kk * kk)) * (2.0 * Dv * kj ** 3 / (3.0 * pi)) * acc
        th = math.tanh(kj * sv / sqf)
        d2 = pk_t[j] * kj ** 3 / (2.0 * pi * pi) * max(1.0 - fdamp * th * th, 0.0)
        dh = (d1 ** alpha + d2 ** alpha) ** inv_alpha
        out[j] = dh * 2.0 * pi * pi / kj ** 3
    diag[0] = sig8
    diag[1] = sv
    diag[2] = sv100
    diag[3] = fdamp
    diag[4] = neff
    diag[5] = alpha
    diag[6] = dcr
    diag[7] = Dv
    diag[8] = eta
    diag[9] = kstar


@njit(cache=True, parallel=True, fastmath=_FASTMATH)
def _hm_all(k, lk, pk_t, pk_c, i0, w, nin, lq, WR, W8, W100, Wv, lnR, dR, ia, ib, ms,
            om_z, g_z, z_arr, f_nu, n_s, A_b, eta0, dolag, lna_t, lng_t, ftab, gtab, kmin_nl, tol_win,
            out, diag):
    for iz in prange(z_arr.size):
        _hm_one_z(k, lk, pk_t[iz], pk_c[iz], i0, w, nin, lq, WR, W8, W100, Wv, lnR, dR, ia, ib, ms,
                  om_z[iz], g_z[iz], z_arr[iz], f_nu, n_s, A_b, eta0, dolag, lna_t, lng_t, ftab, gtab,
                  kmin_nl, tol_win, out[iz], diag[iz])


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def hmcode_pnl_diag(z, k_h, pk_lin, omega_m, omega_nu_frac, h, n_s, A_baryon, eta0, pk_cb=None, *,
                    ms=24, dl=0.04, mmin=1.0, mmax=1.0e18, kmin_nl=0.0, omega_rad_h2=3.6e-5,
                    dolag=1.0, tol_win=1.0e-10):
    """As :func:`hmcode_pnl` but returns ``(pk_nl, diag)``; ``diag[nz, 10]`` holds, per redshift,
    ``sigma_8(z), sigma_v, sigma_v100, f_damp, neff, alpha, delta_c, Delta_v, eta, k_star``."""
    z = np.ascontiguousarray(np.atleast_1d(np.asarray(z, float)))
    k = np.ascontiguousarray(k_h, float)
    pk_t = np.ascontiguousarray(pk_lin, float)
    if z.ndim != 1 or k.ndim != 1:
        raise ValueError("z and k_h must be one-dimensional")
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("z must be finite and >= 0")
    if pk_t.shape != (z.size, k.size):
        raise ValueError(f"pk_lin shape {pk_t.shape} != (nz, nk) = {(z.size, k.size)}")
    if k.size < 4 or not np.all(np.isfinite(k)) or k[0] <= 0.0 or np.any(np.diff(k) <= 0):
        raise ValueError("k_h must be finite, positive, strictly increasing, with at least 4 nodes")
    # scalar parameters: fail loudly instead of returning silent NaN/garbage (the kernels use fastmath)
    for name, val, lo, hi in (("omega_m", omega_m, 0.0, np.inf), ("h", h, 0.0, np.inf),
                              ("A_baryon", A_baryon, 0.0, np.inf)):
        if not (np.isfinite(val) and lo < val < hi):
            raise ValueError(f"{name} must be finite and > 0, got {val!r}")
    if not (np.isfinite(omega_nu_frac) and 0.0 <= omega_nu_frac < 1.0):
        raise ValueError(f"omega_nu_frac must be in [0, 1), got {omega_nu_frac!r}")
    if not (np.isfinite(n_s) and np.isfinite(eta0)):
        raise ValueError("n_s and eta0 must be finite")
    if not (int(ms) >= 2 and 0.0 < dl <= 0.2):
        raise ValueError("need ms >= 2 and 0 < dl <= 0.2")
    if not (0.0 < mmin < mmax):
        raise ValueError("need 0 < mmin < mmax")
    if not (np.all(np.isfinite(pk_t)) and np.all(pk_t > 0.0)):
        raise ValueError("pk_lin must be finite and strictly positive")
    if pk_cb is None:
        pk_c = np.ascontiguousarray(pk_cb_approx(z, k, pk_t, omega_m, omega_nu_frac))
    else:
        pk_c = np.ascontiguousarray(pk_cb, float)
        if pk_c.shape != pk_t.shape:
            raise ValueError("pk_cb must have the same shape as pk_lin")
        if not (np.all(np.isfinite(pk_c)) and np.all(pk_c > 0.0)):
            raise ValueError("pk_cb must be finite and strictly positive")
    omega_m = float(omega_m)
    om_rad = omega_rad_h2 / (h * h) if omega_rad_h2 else 0.0
    grid = _get_grid(k, ms, dl)
    ftab, gtab = _fg_tables()
    lna_t, lng_t = growth_table(omega_m)
    om_z = omega_m_of_z(z, omega_m, om_rad)
    g_z = _growth(1.0 / (1.0 + z), omega_m)
    # active radius window from the mass range M = 4 pi/3 rho_bar R^3
    rho = RHO_CRIT_H * omega_m
    r_lo = (3.0 * mmin / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    r_hi = (3.0 * mmax / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    ia = max(int(math.ceil((math.log(r_lo) - grid.lnR[0]) / grid.dR)), grid.ms + 2)
    ib = min(int(math.floor((math.log(r_hi) - grid.lnR[0]) / grid.dR)), grid.nR - 4)
    if ib - ia < grid.ms + 4:
        raise ValueError("mass range [mmin, mmax] too narrow / outside the radius grid")
    out = np.empty((z.size, k.size))
    diag = np.empty((z.size, 10))
    _hm_all(k, grid.lk, pk_t, pk_c, grid.i0, grid.w, grid.nin, grid.lq, grid.WR, grid.W8, grid.W100,
            grid.Wv, grid.lnR, grid.dR, ia, ib, grid.ms, om_z, g_z, z, float(omega_nu_frac), float(n_s),
            float(A_baryon), float(eta0), float(dolag), lna_t, lng_t, ftab, gtab, float(kmin_nl),
            float(tol_win), out, diag)
    return out, diag


def hmcode_pnl(z, k_h, pk_lin, omega_m, omega_nu_frac, h, n_s, A_baryon, eta0, pk_cb=None, **kw):
    """HMcode-2016 non-linear matter power spectrum ``P_nl(z, k)`` (spec s2.5).

    Parameters
    ----------
    z : (nz,) redshifts (any order; each is independent).
    k_h : (nk,) strictly increasing wavenumbers [h/Mpc] (CAMB's native nodes, ~195, work best;
        the first and last node define the integration range of sigma(R); P is returned on the same nodes).
    pk_lin : (nz, nk) linear TOTAL-matter power spectrum [(Mpc/h)^3] (used for sigma_8, sigma_v, 2-halo term).
    omega_m : late-time matter density parameter Om_m = (omch2 + ombh2 + omnuh2)/h^2 (CAMB ``omegam``).
    omega_nu_frac : f_nu = omnuh2 / (omch2 + ombh2 + omnuh2).
    h : Hubble parameter H0/100 (only used for the tiny radiation term in E(z)).
    n_s : scalar spectral index (only used for the clip range of n_eff).
    A_baryon : halo concentration amplitude A (2.0 ... 3.13 in KV450; 3.13 = dark matter only).
    eta0 : window-function parameter, eta = eta0 - 0.3 sigma_8(z); KV450 uses eta0 = 0.98 - 0.12 A_baryon.
    pk_cb : (nz, nk) linear CDM+baryon power spectrum (CAMB ``delta_nonu`` auto spectrum).  It is used for
        sigma(R) and n_eff of the halo model.  If omitted, :func:`pk_cb_approx` is used (exact for f_nu = 0;
        ~1e-3 accurate for m_nu = 0.06 eV).  Passing ``pk_lin`` instead would bias P_nl by ~1 % at high k.

    Keyword options
    ---------------
    ms (24): ln R nodes per factor 100 in halo mass (accuracy/cost; ms=16 is ~1e-3 less accurate at k > 10 h/Mpc, 35 vs 54 ms for 150 z),
    dl (0.04) dense ln k spacing, mmin/mmax (1, 1e18 Msun/h), kmin_nl (0), omega_rad_h2 (3.6e-5),
    dolag (1.0), tol_win (1e-10).

    Returns
    -------
    pk_nl : (nz, nk) non-linear power spectrum [(Mpc/h)^3] on ``k_h``.
    """
    return hmcode_pnl_diag(z, k_h, pk_lin, omega_m, omega_nu_frac, h, n_s, A_baryon, eta0, pk_cb, **kw)[0]
