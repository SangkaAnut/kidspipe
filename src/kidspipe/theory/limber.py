"""Lensing kernels, NLA intrinsic alignments and the Limber projection (spec s4-s6).

Everything here is the convergence-like spectrum of spec s6.1,

    C_ij(l) = int_0^chi_max dchi  q_i(chi) q_j(chi) / chi^2  P_nl( (l+1/2)/chi , z(chi) ),
    q_i(chi) = s W_i(chi) + F(z(chi)) n_i(chi),                                  (spec s6.1)

with ``s = (3/2) Omega_m (100/c)^2`` [(h/Mpc)^2], ``c = 299792.458 km/s``, ``chi`` in Mpc/h,
``n_i(chi) = n_i(z) dz/dchi`` [h/Mpc] (spec s4), ``P_nl`` in (Mpc/h)^3, ``k`` in h/Mpc.  Expanding
the product gives GG + GI + IG + II exactly (the "folded" kernel, spec s6.1 / D19).  The C_l are
dimensionless, are NOT multiplied by l(l+1)/2pi and carry no (l-1)l(l+1)(l+2)/(l+1/2)^4 factor.

Numerics ("fast", spec s6.3 (E))
--------------------------------
* The integral is done over ``z`` on ``n_nodes`` uniform nodes in ``[0, z_max]`` with the
  trapezoid rule:  ``int dchi f = int dz (dchi/dz) f``.
* ``W_i = (chi/a)[A_i(z) - chi B_i(z)]`` with ``A_i(z) = int_z^inf n_i dz'`` and
  ``B_i(z) = int_z^inf n_i/chi(z') dz'`` by reverse cumulative trapezoid (spec s4 (E)).
* ``ln P_nl`` is interpolated linearly in ``z`` and (default) with 4-point Lagrange cubics in
  ``ln k``; the integrand is exactly 0 outside the (z, k) table (spec s6.2, D13).
* The expensive, parameter dependent look-up ``Q[m, l] = P_nl((l+1/2)/chi_m, z_m)`` is built once
  per cosmology (:meth:`Limber.set_power`); changes of photo-z shifts / ``A_IA`` only re-do the
  cheap kernel contraction ``C[pair, l] = sum_m KK[pair, m] Q[m, l]`` (:meth:`Limber.cl`).

``fidelity="kcap"`` adds the cheap, reproducible parts of the kcap numerics (spec s6.2, D10, D15):
the power spectrum itself (not ln P) is interpolated, and ``W_i`` is divided by the numerically
integrated ``Z_i = int n_i dz`` of the sampled distribution.  It does NOT emulate GSL Akima/bicubic
splines or the adaptive QAG quadrature, so it differs from kcap by ~1e-3, not bit-wise.
"""
from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numba
import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.extending import intrinsic
from scipy.interpolate import CubicSpline

__all__ = [
    "C_LIGHT",
    "C1_RHO_CRIT",
    "default_ell",
    "coarse_ell",
    "lensing_prefactor",
    "lensing_efficiency",
    "nla_growth_factor",
    "Limber",
    "cl_shear",
]

C_LIGHT = 299792.458  # km/s
C1_RHO_CRIT = 0.0138731  # dimensionless C1 rho_crit (spec s5)


def default_ell(n: int = 400) -> np.ndarray:
    """kcap multipole grid: ``logspace(log10 0.1, log10 5e5, 400)`` (spec s6.2)."""
    return np.logspace(np.log10(0.1), np.log10(5.0e5), n)


def coarse_ell(n: int = 120) -> np.ndarray:
    """Coarse grid ``logspace(0, log10 3e5, 120)`` (spec D20: Delta chi2 = 0.01)."""
    return np.logspace(0.0, np.log10(3.0e5), n)


def lensing_prefactor(omega_m: float) -> float:
    """``s = (3/2) Omega_m (100/c)^2`` in (h/Mpc)^2, c = 299792.458 km/s (spec s0.1)."""
    return 1.5 * omega_m * (100.0 / C_LIGHT) ** 2


def _rev_cumtrapz(f: np.ndarray, dz: float) -> np.ndarray:
    """``out[..., m] = int_{z_m}^{z_end} f dz`` on a uniform grid (reverse cumulative trapezoid)."""
    seg = 0.5 * dz * (f[..., 1:] + f[..., :-1])
    out = np.zeros_like(f)
    out[..., :-1] = np.cumsum(seg[..., ::-1], axis=-1)[..., ::-1]
    return out


def lensing_efficiency(z: np.ndarray, chi: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Lensing efficiency ``W_i(chi)`` [Mpc/h] on a *uniform* z grid (spec s4 (E)).

    ``W = (chi/a) [A(z) - chi B(z)]``, ``A = int_z^zmax n dz'``, ``B = int_z^zmax n/chi(z') dz'``,
    ``a = 1/(1+z)``.  Equivalent to
    ``W(chi) = (chi/a) int_chi^chi_max n(chi') (chi'-chi)/chi' dchi'`` for ``n`` normalised to 1.
    ``n`` has shape ``(..., nz)`` and is the density in z (not chi).
    """
    z = np.asarray(z, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    dz = (z[-1] - z[0]) / (z.size - 1)
    inv_chi = np.divide(1.0, chi, out=np.zeros_like(chi), where=chi > 0)
    A = _rev_cumtrapz(n, dz)
    B = _rev_cumtrapz(n * inv_chi, dz)
    return chi * (1.0 + z) * (A - chi * B)


def nla_growth_factor(z_pk: np.ndarray, pk_nl: np.ndarray) -> np.ndarray:
    """``D(z) = sqrt(P_nl(k_small, z) / P_nl(k_small, 0))`` with ``k_small = k_h[0]`` (spec s5).

    ``pk_nl`` has shape ``(nz, nk)`` with ``z_pk[0] == 0``.  Returns shape ``(nz,)``.
    """
    if abs(z_pk[0]) > 1e-8:
        raise ValueError("the P_nl redshift grid must start at z = 0 (spec s5)")
    return np.sqrt(pk_nl[:, 0] / pk_nl[0, 0])


#: ``ln k`` grids whose spacings all agree with the mean spacing ``dlnk`` to this relative accuracy (and whose
#: nodes lie within this fraction of ``dlnk`` of ``lnk[0] + j dlnk``) take the uniform-grid index path of
#: :func:`_fill_q` (``s = (x - ln k_0) / dlnk``, ``j = int(s)``, constant Lagrange weights in ``s - j0``).
UNIFORM_LNK_RTOL = 1e-11

#: node chunks per numba thread in :func:`_fill_q` (load balance; each chunk owns one scratch row of nk doubles)
_CHUNKS_PER_THREAD = 8


@njit(cache=True)
def _fill_q_prepare(z_pk, z_nodes, lnchi, lnk, rtol):
    """Everything of :func:`_fill_q` that does not depend on the multipole, computed once per call.

    Returns ``(iz, t, uniform, inv_dlnk, inv_dk, inv_den)``:

    * ``iz[m], t[m]`` -- linear-in-z stencil of node ``m``: ``z_pk[iz] <= z_m <= z_pk[iz + 1]``,
      ``t = (z_m - z_pk[iz]) / (z_pk[iz + 1] - z_pk[iz])`` (``searchsorted`` and the expression of the former
      per-node code); ``iz = -1`` marks a node whose row is identically 0 (``ln chi_m`` not finite, i.e. chi = 0,
      ``z_m`` NaN or outside ``[z_pk[0], z_pk[-1]]``, or a single-redshift table).
    * ``uniform`` -- every spacing equals the mean spacing ``dlnk`` within ``rtol |dlnk|`` (and every node
      lies within ``rtol |dlnk|`` of ``lnk[0] + j dlnk``); ``inv_dlnk = 1/dlnk``.
    * ``inv_dk[j] = 1/(lnk[j+1] - lnk[j])`` (linear weights) and ``inv_den[j0, i] = 1/prod_{n != i}(x_i - x_n)``
      over the stencil ``x_n = lnk[j0 + n]`` (cubic Lagrange weights; the same products as the former code).
    """
    nzp = z_pk.size
    nm = z_nodes.size
    iz = np.empty(nm, dtype=np.int64)
    t = np.zeros(nm)
    for m in range(nm):
        z = z_nodes[m]
        if nzp < 2 or (not np.isfinite(lnchi[m])) or not (z >= z_pk[0] and z <= z_pk[nzp - 1]):
            iz[m] = -1
            continue
        i = np.searchsorted(z_pk, z) - 1
        if i < 0:
            i = 0
        if i > nzp - 2:
            i = nzp - 2
        iz[m] = i
        t[m] = (z - z_pk[i]) / (z_pk[i + 1] - z_pk[i])
    nk = lnk.size
    inv_dk = np.zeros(max(nk - 1, 1))
    inv_den = np.zeros((max(nk - 3, 1), 4))
    uniform = False
    inv_dlnk = 0.0
    if nk >= 2:
        dlnk = (lnk[nk - 1] - lnk[0]) / (nk - 1)
        dev = 0.0
        drift = 0.0
        for j in range(nk - 1):
            d = lnk[j + 1] - lnk[j]
            inv_dk[j] = 1.0 / d
            dev = max(dev, abs(d - dlnk))
            drift = max(drift, abs(lnk[j + 1] - (lnk[0] + (j + 1) * dlnk)))
        uniform = dev < rtol * abs(dlnk) and drift < rtol * abs(dlnk)
        inv_dlnk = 1.0 / dlnk
    for j0 in range(nk - 3):
        x0 = lnk[j0]
        x1 = lnk[j0 + 1]
        x2 = lnk[j0 + 2]
        x3 = lnk[j0 + 3]
        inv_den[j0, 0] = 1.0 / ((x0 - x1) * (x0 - x2) * (x0 - x3))
        inv_den[j0, 1] = 1.0 / ((x1 - x0) * (x1 - x2) * (x1 - x3))
        inv_den[j0, 2] = 1.0 / ((x2 - x0) * (x2 - x1) * (x2 - x3))
        inv_den[j0, 3] = 1.0 / ((x3 - x0) * (x3 - x1) * (x3 - x2))
    return iz, t, uniform, inv_dlnk, inv_dk, inv_den


@intrinsic
def _f64_bits(typingctx, x):
    """Bitcast a float64 scalar to int64 without modifying bit representation."""
    if x != types.float64:
        return None

    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], ir.IntType(64))

    return types.int64(types.float64), codegen


@intrinsic
def _bits_f64(typingctx, x):
    """Bitcast an int64 scalar to float64 without modifying bit representation."""
    if x != types.int64:
        return None

    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], ir.DoubleType())

    return types.float64(types.int64), codegen


_SHIFT = 6755399441055744.0  # 1.5 * 2**52 for Cody-Waite round-to-nearest-int
_SHIFT_BITS = 0x4338000000000000
_LOG2E = 1.4426950408889634
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10


@njit(inline="always")
def _vexp1(x):
    """Branch-free 1-ulp ``exp(x)`` for ``-708.0 <= x <= 709.0`` (Cody-Waite + degree-13 Estrin)."""
    tt = x * _LOG2E + _SHIFT
    kf = tt - _SHIFT
    r = (x - kf * _LN2_HI) - kf * _LN2_LO
    r2 = r * r
    r4 = r2 * r2
    p0 = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0)))
    p1 = (1.0 / 24.0) + r * ((1.0 / 120.0) + r * ((1.0 / 720.0) + r * (1.0 / 5040.0)))
    p2 = (1.0 / 40320.0) + r * ((1.0 / 362880.0) + r * ((1.0 / 3628800.0) + r * (1.0 / 39916800.0)))
    p3 = (1.0 / 479001600.0) + r * (1.0 / 6227020800.0)
    p = (p0 + r4 * p1) + (r4 * r4) * (p2 + r4 * p3)
    n = _f64_bits(tt) - _SHIFT_BITS
    return p * _bits_f64((n + 1023) << 52)


@njit(inline="always")
def _vexp_1d(buf, n):
    """In-place ``exp`` on contiguous 1-D ``buf[0:n]``; uses AVX2-vectorizable ``_vexp1`` when in range.

    Operating on a 1-D scratch buffer (never a 2-D row slice) allows LLVM to auto-vectorize with AVX2.
    """
    ok = True
    for i in range(n):
        v = buf[i]
        ok = ok & (v >= -708.0) & (v <= 709.0)
    if ok:
        for i in range(n):
            buf[i] = _vexp1(buf[i])
    else:
        for i in range(n):
            buf[i] = np.exp(buf[i])


@njit(inline="always")
def _z_row(tab, i0, tt, row):
    """``row[c] = (1 - t) tab[i0, c] + t tab[i0 + 1, c]`` (the former per-node row, now in a scratch buffer)."""
    omt = 1.0 - tt
    for c in range(row.size):
        row[c] = omt * tab[i0, c] + tt * tab[i0 + 1, c]


@njit(inline="always")
def _zero_row(out_m, nl):
    """Zero every multipole entry of 1-D row ``out_m``."""
    for l in range(nl):
        out_m[l] = 0.0


@njit(inline="always")
def _zero_margins(out_m, la, lb, nl):
    """Zero out-of-range multipole margins ``[0, la)`` and ``[lb, nl)`` of 1-D row ``out_m``."""
    for l in range(la):
        out_m[l] = 0.0
    for l in range(lb, nl):
        out_m[l] = 0.0


@njit(inline="always")
def _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi):
    """Zero out-of-range margins of 1-D row ``out_m`` and return the contiguous active slice ``(la, lb)``."""
    nl = ln_nu.size
    la = 0
    while la < nl and (ln_nu[la] - lc) < lo:
        out_m[la] = 0.0
        la += 1
    lb = nl
    while lb > la and (ln_nu[lb - 1] - lc) > hi:
        lb -= 1
        out_m[lb] = 0.0
    return la, lb


@njit(cache=True)
def _prep_newton_geom(lnk):
    """Precompute stencil spacings and reciprocals ``(h1, h2, ih1, ih21, ih32, ih2, ih31, ih3)`` of length ``nk - 3``."""
    n = max(lnk.size - 3, 0)
    h1 = np.empty(n, dtype=np.float64)
    h2 = np.empty(n, dtype=np.float64)
    ih1 = np.empty(n, dtype=np.float64)
    ih21 = np.empty(n, dtype=np.float64)
    ih32 = np.empty(n, dtype=np.float64)
    ih2 = np.empty(n, dtype=np.float64)
    ih31 = np.empty(n, dtype=np.float64)
    ih3 = np.empty(n, dtype=np.float64)
    for j0 in range(n):
        x0 = lnk[j0]
        x1 = lnk[j0 + 1]
        x2 = lnk[j0 + 2]
        x3 = lnk[j0 + 3]
        d10 = x1 - x0
        d21 = x2 - x1
        d32 = x3 - x2
        d20 = x2 - x0
        d31 = x3 - x1
        d30 = x3 - x0
        h1[j0] = d10
        h2[j0] = d20
        ih1[j0] = 1.0 / d10
        ih21[j0] = 1.0 / d21
        ih32[j0] = 1.0 / d32
        ih2[j0] = 1.0 / d20
        ih31[j0] = 1.0 / d31
        ih3[j0] = 1.0 / d30
    return h1, h2, ih1, ih21, ih32, ih2, ih31, ih3


@njit(parallel=True, cache=True)
def _build_newton_from_tab(tab, ih1, ih21, ih32, ih2, ih31, ih3):
    """Build Newton divided-difference SoA ``(d0, d1, d2, d3)`` of shape ``(nz_pk, nk - 3)`` from ``tab``."""
    nzp = tab.shape[0]
    ns = ih1.size
    d0 = np.empty((nzp, ns), dtype=np.float64)
    d1 = np.empty((nzp, ns), dtype=np.float64)
    d2 = np.empty((nzp, ns), dtype=np.float64)
    d3 = np.empty((nzp, ns), dtype=np.float64)
    for i in prange(nzp):
        for j0 in range(ns):
            y0 = tab[i, j0]
            y1 = tab[i, j0 + 1]
            y2 = tab[i, j0 + 2]
            y3 = tab[i, j0 + 3]
            f01 = (y1 - y0) * ih1[j0]
            f12 = (y2 - y1) * ih21[j0]
            f23 = (y3 - y2) * ih32[j0]
            f012 = (f12 - f01) * ih2[j0]
            f123 = (f23 - f12) * ih31[j0]
            d0[i, j0] = y0
            d1[i, j0] = f01
            d2[i, j0] = f012
            d3[i, j0] = (f123 - f012) * ih3[j0]
    return d0, d1, d2, d3


@njit(parallel=True, cache=True)
def _validate_log_and_newton(pk_nl, ih1, ih21, ih32, ih2, ih31, ih3):
    """Fused validation (finite > 0), ``ln(max(pk, 1e-300))``, and Newton divided-difference SoA build."""
    nzp = pk_nl.shape[0]
    nk = pk_nl.shape[1]
    ns = ih1.size
    d0 = np.empty((nzp, ns), dtype=np.float64)
    d1 = np.empty((nzp, ns), dtype=np.float64)
    d2 = np.empty((nzp, ns), dtype=np.float64)
    d3 = np.empty((nzp, ns), dtype=np.float64)
    row_ok = np.empty(nzp, dtype=np.bool_)
    for i in prange(nzp):
        r_ok = True
        if ns > 0:
            v0 = pk_nl[i, 0]
            g0 = np.isfinite(v0) and v0 > 0.0
            r_ok = r_ok and g0
            y0 = np.log(v0 if (g0 and v0 > 1e-300) else 1e-300)
            v1 = pk_nl[i, 1]
            g1 = np.isfinite(v1) and v1 > 0.0
            r_ok = r_ok and g1
            y1 = np.log(v1 if (g1 and v1 > 1e-300) else 1e-300)
            v2 = pk_nl[i, 2]
            g2 = np.isfinite(v2) and v2 > 0.0
            r_ok = r_ok and g2
            y2 = np.log(v2 if (g2 and v2 > 1e-300) else 1e-300)
            for j0 in range(ns):
                v3 = pk_nl[i, j0 + 3]
                g3 = np.isfinite(v3) and v3 > 0.0
                r_ok = r_ok and g3
                y3 = np.log(v3 if (g3 and v3 > 1e-300) else 1e-300)
                f01 = (y1 - y0) * ih1[j0]
                f12 = (y2 - y1) * ih21[j0]
                f23 = (y3 - y2) * ih32[j0]
                f012 = (f12 - f01) * ih2[j0]
                f123 = (f23 - f12) * ih31[j0]
                d0[i, j0] = y0
                d1[i, j0] = f01
                d2[i, j0] = f012
                d3[i, j0] = (f123 - f012) * ih3[j0]
                y0 = y1
                y1 = y2
                y2 = y3
        else:
            for j in range(nk):
                v = pk_nl[i, j]
                if not (np.isfinite(v) and v > 0.0):
                    r_ok = False
        row_ok[i] = r_ok
    ok = True
    for i in range(nzp):
        if not row_ok[i]:
            ok = False
            break
    return ok, d0, d1, d2, d3


@njit(parallel=True, cache=True)
def _validate_and_log(pk_nl):
    """Validate ``pk_nl`` (finite and > 0) and compute ``ln(max(pk_nl, 1e-300))`` in one parallel pass."""
    nzp = pk_nl.shape[0]
    nk = pk_nl.shape[1]
    tab = np.empty((nzp, nk), dtype=np.float64)
    row_ok = np.empty(nzp, dtype=np.bool_)
    for i in prange(nzp):
        r_ok = True
        for j in range(nk):
            v = pk_nl[i, j]
            good = np.isfinite(v) and v > 0.0
            r_ok = r_ok and good
            tab[i, j] = np.log(v if (good and v > 1e-300) else 1e-300)
        row_ok[i] = r_ok
    ok = True
    for i in range(nzp):
        if not row_ok[i]:
            ok = False
            break
    return ok, tab


@njit(parallel=True, cache=True)
def _validate_pk_positive(pk_nl):
    """Parallel check that every entry of ``pk_nl`` is finite and strictly positive."""
    nzp = pk_nl.shape[0]
    nk = pk_nl.shape[1]
    row_ok = np.empty(nzp, dtype=np.bool_)
    for i in prange(nzp):
        r_ok = True
        for j in range(nk):
            v = pk_nl[i, j]
            if not (np.isfinite(v) and v > 0.0):
                r_ok = False
                break
        row_ok[i] = r_ok
    for i in range(nzp):
        if not row_ok[i]:
            return False
    return True


@njit(parallel=True, cache=True)
def _precompute_cubic_stencils(lnk, iz, lnchi, ln_nu):
    """Precompute per-node active bounds ``(las, lbs)`` and stencil offsets ``(j0s, us)`` using int32 indices."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    lo = lnk[0]
    hi = lnk[nk - 1]
    las = np.zeros(nm, dtype=np.int32)
    lbs = np.zeros(nm, dtype=np.int32)
    j0s = np.empty((nm, nl), dtype=np.int32)
    us = np.empty((nm, nl), dtype=np.float64)
    for m in prange(nm):
        if iz[m] < 0:
            continue
        lc = lnchi[m]
        la = 0
        while la < nl and (ln_nu[la] - lc) < lo:
            la += 1
        lb = nl
        while lb > la and (ln_nu[lb - 1] - lc) > hi:
            lb -= 1
        if lb <= la:
            continue
        jm = j0s[m]
        um = us[m]
        j = 0
        for l in range(la, lb):
            x = ln_nu[l] - lc
            while j < nk - 2 and lnk[j + 1] <= x:
                j += 1
            j0 = j - 1
            if j0 < 0:
                j0 = 0
            if j0 > nk - 4:
                j0 = nk - 4
            jm[l] = j0
            um[l] = x - lnk[j0]
        las[m] = la
        lbs[m] = lb
    return las, lbs, j0s, us


@njit(parallel=True, cache=True)
def _fill_q_newton_cursor_fast(d0, d1, d2, d3, h1, h2, lnk, iz, t, lnchi, ln_nu, work_buf, work_j0, work_u, out):
    """Uncached cubic ``fast`` kernel using Newton divided differences, monotone cursor, and 1D ``_vexp_1d``."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_buf.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        buf = work_buf[c]
        j0_arr = work_j0[c]
        u_arr = work_u[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la = 0
            while la < nl and (ln_nu[la] - lc) < lo:
                out_m[la] = 0.0
                la += 1
            lb = nl
            while lb > la and (ln_nu[lb - 1] - lc) > hi:
                lb -= 1
                out_m[lb] = 0.0
            n_act = lb - la
            if n_act <= 0:
                continue
            j = 0
            for k in range(n_act):
                x = ln_nu[la + k] - lc
                while j < nk - 2 and lnk[j + 1] <= x:
                    j += 1
                j0 = j - 1
                if j0 < 0:
                    j0 = 0
                if j0 > nk - 4:
                    j0 = nk - 4
                j0_arr[k] = j0
                u_arr[k] = x - lnk[j0]
            r00 = d0[i0]
            r01 = d1[i0]
            r02 = d2[i0]
            r03 = d3[i0]
            r10 = d0[i0 + 1]
            r11 = d1[i0 + 1]
            r12 = d2[i0 + 1]
            r13 = d3[i0 + 1]
            tt = t[m]
            omt = 1.0 - tt
            for k in range(n_act):
                j0 = j0_arr[k]
                u = u_arr[k]
                u1 = u - h1[j0]
                u2 = u - h2[j0]
                p0 = ((r03[j0] * u2 + r02[j0]) * u1 + r01[j0]) * u + r00[j0]
                p1 = ((r13[j0] * u2 + r12[j0]) * u1 + r11[j0]) * u + r10[j0]
                buf[k] = omt * p0 + tt * p1
            _vexp_1d(buf, n_act)
            for k in range(n_act):
                out_m[la + k] = buf[k]


@njit(parallel=True, cache=True)
def _fill_q_newton_cached_fast(d0, d1, d2, d3, h1, h2, iz, t, las, lbs, j0s, us, work_buf, out):
    """Cached cubic ``fast`` kernel using precomputed ``(las, lbs, j0s, us)``, Newton SoA, and 1D ``_vexp_1d``."""
    nm = iz.size
    nl = out.shape[1]
    nchunk = work_buf.shape[0]
    for c in prange(nchunk):
        buf = work_buf[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            la = las[m]
            lb = lbs[m]
            n_act = lb - la
            if n_act <= 0:
                _zero_row(out_m, nl)
                continue
            _zero_margins(out_m, la, lb, nl)
            r00 = d0[i0]
            r01 = d1[i0]
            r02 = d2[i0]
            r03 = d3[i0]
            r10 = d0[i0 + 1]
            r11 = d1[i0 + 1]
            r12 = d2[i0 + 1]
            r13 = d3[i0 + 1]
            jm = j0s[m]
            um = us[m]
            tt = t[m]
            omt = 1.0 - tt
            for k in range(n_act):
                l = la + k
                j0 = jm[l]
                u = um[l]
                u1 = u - h1[j0]
                u2 = u - h2[j0]
                p0 = ((r03[j0] * u2 + r02[j0]) * u1 + r01[j0]) * u + r00[j0]
                p1 = ((r13[j0] * u2 + r12[j0]) * u1 + r11[j0]) * u + r10[j0]
                buf[k] = omt * p0 + tt * p1
            _vexp_1d(buf, n_act)
            for k in range(n_act):
                out_m[la + k] = buf[k]


@njit(parallel=True, cache=True)
def _fill_q_uniform_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, work_buf, out):
    """Uniform-grid cubic ``fast`` kernel using equispaced Lagrange weights and 1D ``_vexp_1d``."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        buf = work_buf[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            n_act = lb - la
            if n_act <= 0:
                continue
            _z_row(tab, i0, t[m], row)
            for k in range(n_act):
                x = ln_nu[la + k] - lc
                s = (x - lo) * inv_dlnk
                j0 = int(s) - 1
                if j0 < 0:
                    j0 = 0
                if j0 > nk - 4:
                    j0 = nk - 4
                u = s - j0
                a1 = u - 1.0
                a2 = u - 2.0
                a3 = u - 3.0
                w0 = -(a1 * a2 * a3) * (1.0 / 6.0)
                w1 = (u * a2 * a3) * 0.5
                w2 = -(u * a1 * a3) * 0.5
                w3 = (u * a1 * a2) * (1.0 / 6.0)
                buf[k] = w0 * row[j0] + w1 * row[j0 + 1] + w2 * row[j0 + 2] + w3 * row[j0 + 3]
            _vexp_1d(buf, n_act)
            for k in range(n_act):
                out_m[la + k] = buf[k]


# Lagrange (inv_den) is kept for kcap because Newton form suffers catastrophic cancellation on coarse P grids.
@njit(parallel=True, cache=True)
def _fill_q_cursor_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_den, work_row, out):
    """Uncached cubic ``kcap`` kernel using Lagrange ``inv_den``, monotone cursor, and zero-clipping."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            if lb <= la:
                continue
            _z_row(tab, i0, t[m], row)
            j = 0
            for l in range(la, lb):
                x = ln_nu[l] - lc
                while j < nk - 2 and lnk[j + 1] <= x:
                    j += 1
                j0 = j - 1
                if j0 < 0:
                    j0 = 0
                if j0 > nk - 4:
                    j0 = nk - 4
                d0 = x - lnk[j0]
                d1 = x - lnk[j0 + 1]
                d2 = x - lnk[j0 + 2]
                d3 = x - lnk[j0 + 3]
                w0 = d1 * d2 * d3 * inv_den[j0, 0]
                w1 = d0 * d2 * d3 * inv_den[j0, 1]
                w2 = d0 * d1 * d3 * inv_den[j0, 2]
                w3 = d0 * d1 * d2 * inv_den[j0, 3]
                v = w0 * row[j0] + w1 * row[j0 + 1] + w2 * row[j0 + 2] + w3 * row[j0 + 3]
                out_m[l] = v if v > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _fill_q_cached_kcap(tab, lnk, inv_den, iz, t, lnchi, ln_nu, las, lbs, j0s, work_row, out):
    """Cached cubic ``kcap`` kernel using precomputed ``(las, lbs, j0s)`` and Lagrange ``inv_den``."""
    nm = iz.size
    nl = ln_nu.size
    nchunk = work_row.shape[0]
    for c in prange(nchunk):
        row = work_row[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            la = las[m]
            lb = lbs[m]
            if lb <= la:
                _zero_row(out_m, nl)
                continue
            _zero_margins(out_m, la, lb, nl)
            _z_row(tab, i0, t[m], row)
            lc = lnchi[m]
            jm = j0s[m]
            for l in range(la, lb):
                x = ln_nu[l] - lc
                j0 = jm[l]
                d0 = x - lnk[j0]
                d1 = x - lnk[j0 + 1]
                d2 = x - lnk[j0 + 2]
                d3 = x - lnk[j0 + 3]
                w0 = d1 * d2 * d3 * inv_den[j0, 0]
                w1 = d0 * d2 * d3 * inv_den[j0, 1]
                w2 = d0 * d1 * d3 * inv_den[j0, 2]
                w3 = d0 * d1 * d2 * inv_den[j0, 3]
                v = w0 * row[j0] + w1 * row[j0 + 1] + w2 * row[j0 + 2] + w3 * row[j0 + 3]
                out_m[l] = v if v > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _fill_q_uniform_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, out):
    """Uniform-grid cubic ``kcap`` kernel using equispaced Lagrange weights and zero-clipping."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            if lb <= la:
                continue
            _z_row(tab, i0, t[m], row)
            for l in range(la, lb):
                x = ln_nu[l] - lc
                s = (x - lo) * inv_dlnk
                j0 = int(s) - 1
                if j0 < 0:
                    j0 = 0
                if j0 > nk - 4:
                    j0 = nk - 4
                u = s - j0
                a1 = u - 1.0
                a2 = u - 2.0
                a3 = u - 3.0
                w0 = -(a1 * a2 * a3) * (1.0 / 6.0)
                w1 = (u * a2 * a3) * 0.5
                w2 = -(u * a1 * a3) * 0.5
                w3 = (u * a1 * a2) * (1.0 / 6.0)
                v = w0 * row[j0] + w1 * row[j0 + 1] + w2 * row[j0 + 2] + w3 * row[j0 + 3]
                out_m[l] = v if v > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _fill_q_linear_cursor_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dk, work_row, work_buf, out):
    """Linear-in-ln-k ``fast`` kernel with monotone cursor and 1D ``_vexp_1d``."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        buf = work_buf[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            n_act = lb - la
            if n_act <= 0:
                continue
            _z_row(tab, i0, t[m], row)
            if nk >= 2:
                j = 0
                for k in range(n_act):
                    x = ln_nu[la + k] - lc
                    while j < nk - 2 and lnk[j + 1] <= x:
                        j += 1
                    buf[k] = row[j] + (row[j + 1] - row[j]) * (x - lnk[j]) * inv_dk[j]
            else:
                for k in range(n_act):
                    buf[k] = row[0]
            _vexp_1d(buf, n_act)
            for k in range(n_act):
                out_m[la + k] = buf[k]


@njit(parallel=True, cache=True)
def _fill_q_linear_cursor_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dk, work_row, out):
    """Linear-in-ln-k ``kcap`` kernel with monotone cursor and zero-clipping."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            if lb <= la:
                continue
            _z_row(tab, i0, t[m], row)
            if nk >= 2:
                j = 0
                for l in range(la, lb):
                    x = ln_nu[l] - lc
                    while j < nk - 2 and lnk[j + 1] <= x:
                        j += 1
                    v = row[j] + (row[j + 1] - row[j]) * (x - lnk[j]) * inv_dk[j]
                    out_m[l] = v if v > 0.0 else 0.0
            else:
                for l in range(la, lb):
                    v = row[0]
                    out_m[l] = v if v > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _fill_q_linear_uniform_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, work_buf, out):
    """Uniform-grid linear-in-ln-k ``fast`` kernel with 1D ``_vexp_1d``."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        buf = work_buf[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            n_act = lb - la
            if n_act <= 0:
                continue
            _z_row(tab, i0, t[m], row)
            if nk >= 2:
                for k in range(n_act):
                    x = ln_nu[la + k] - lc
                    s = (x - lo) * inv_dlnk
                    j = int(s)
                    if j > nk - 2:
                        j = nk - 2
                    if j < 0:
                        j = 0
                    buf[k] = row[j] + (row[j + 1] - row[j]) * (s - j)
            else:
                for k in range(n_act):
                    buf[k] = row[0]
            _vexp_1d(buf, n_act)
            for k in range(n_act):
                out_m[la + k] = buf[k]


@njit(parallel=True, cache=True)
def _fill_q_linear_uniform_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, out):
    """Uniform-grid linear-in-ln-k ``kcap`` kernel with zero-clipping."""
    nm = iz.size
    nl = ln_nu.size
    nk = lnk.size
    nchunk = work_row.shape[0]
    lo = lnk[0]
    hi = lnk[nk - 1]
    for c in prange(nchunk):
        row = work_row[c]
        for m in range((c * nm) // nchunk, ((c + 1) * nm) // nchunk):
            out_m = out[m]
            i0 = iz[m]
            if i0 < 0:
                _zero_row(out_m, nl)
                continue
            lc = lnchi[m]
            la, lb = _active_bounds_and_zero(out_m, ln_nu, lc, lo, hi)
            if lb <= la:
                continue
            _z_row(tab, i0, t[m], row)
            if nk >= 2:
                for l in range(la, lb):
                    x = ln_nu[l] - lc
                    s = (x - lo) * inv_dlnk
                    j = int(s)
                    if j > nk - 2:
                        j = nk - 2
                    if j < 0:
                        j = 0
                    v = row[j] + (row[j + 1] - row[j]) * (s - j)
                    out_m[l] = v if v > 0.0 else 0.0
            else:
                for l in range(la, lb):
                    v = row[0]
                    out_m[l] = v if v > 0.0 else 0.0


def _fill_q_cursor(tab, lnk, iz, t, lnchi, ln_nu, use_exp, inv_den, work, out):
    """Backward-compatible dispatcher for general-grid cubic look-up."""
    if use_exp:
        h1, h2, ih1, ih21, ih32, ih2, ih31, ih3 = _prep_newton_geom(lnk)
        d0, d1, d2, d3 = _build_newton_from_tab(tab, ih1, ih21, ih32, ih2, ih31, ih3)
        nchunk = work.shape[0]
        nl_buf = max(ln_nu.size, 1)
        work_buf = np.empty((nchunk, nl_buf))
        work_j0 = np.empty((nchunk, nl_buf), dtype=np.int32)
        work_u = np.empty((nchunk, nl_buf))
        _fill_q_newton_cursor_fast(d0, d1, d2, d3, h1, h2, lnk, iz, t, lnchi, ln_nu, work_buf, work_j0, work_u, out)
    else:
        _fill_q_cursor_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_den, work, out)


def _fill_q_uniform(tab, lnk, iz, t, lnchi, ln_nu, use_exp, inv_dlnk, work, out):
    """Backward-compatible dispatcher for uniform-grid cubic look-up."""
    if use_exp:
        work_buf = np.empty((work.shape[0], max(ln_nu.size, 1)))
        _fill_q_uniform_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work, work_buf, out)
    else:
        _fill_q_uniform_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work, out)


def _fill_q_linear(tab, lnk, iz, t, lnchi, ln_nu, use_exp, uniform, inv_dlnk, inv_dk, work, out):
    """Backward-compatible dispatcher for linear-in-ln-k look-up."""
    if use_exp:
        work_buf = np.empty((work.shape[0], max(ln_nu.size, 1)))
        if uniform:
            _fill_q_linear_uniform_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work, work_buf, out)
        else:
            _fill_q_linear_cursor_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dk, work, work_buf, out)
    else:
        if uniform:
            _fill_q_linear_uniform_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work, out)
        else:
            _fill_q_linear_cursor_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dk, work, out)


def _fill_q(tab, z_pk, lnk, z_nodes, lnchi, ln_nu, order, use_exp, out):
    """out[m, l] = interp of ``tab`` at (z_m, ln k = ln nu_l - ln chi_m); 0 outside the table.

    ``tab`` is ln P (use_exp=True) or P (use_exp=False), shape (nz_pk, nk).  ``ln_nu`` ascending.
    Linear in z; linear (order=1) or 4-point cubic (order=3, needs nk >= 4) in ln k; the cubic stencil
    is ``j0 = max(0, min(j - 1, nk - 4))`` around the interval ``lnk[j] <= x < lnk[j + 1]``.

    Cost structure (spec s6.3 (E), prompt 07): the per-node z stencil and grid-only geometry are prepared once
    (:func:`_fill_q_prepare`); the specialized parallel kernels make no heap allocation per node or per chunk
    and dispatch in Python with no runtime ``use_exp`` / ``uniform`` branches in inner loops.  For ``fast``
    non-uniform cubics, Newton divided-difference SoA removes the nk-wide scratch row and feeds a contiguous
    1-D scratch buffer to the branch-free 1-ulp AVX2-vectorizable ``_vexp_1d``; ``kcap`` keeps Lagrange
    ``inv_den`` to avoid cancellation on coarse grids.  The result does not depend on the number of threads.
    """
    tab = np.ascontiguousarray(tab, dtype=np.float64)
    lnk = np.ascontiguousarray(lnk, dtype=np.float64)
    ln_nu = np.ascontiguousarray(ln_nu, dtype=np.float64)
    lnchi = np.ascontiguousarray(lnchi, dtype=np.float64)
    iz, t, uniform, inv_dlnk, inv_dk, inv_den = _fill_q_prepare(
        np.ascontiguousarray(z_pk, dtype=np.float64), np.ascontiguousarray(z_nodes, dtype=np.float64), lnchi, lnk,
        UNIFORM_LNK_RTOL)
    nchunk = max(1, min(iz.size, _CHUNKS_PER_THREAD * numba.get_num_threads()))
    use_exp = bool(use_exp)
    if order == 3 and lnk.size >= 4:
        if use_exp:
            if uniform:
                work_row = np.empty((nchunk, lnk.size))
                work_buf = np.empty((nchunk, max(ln_nu.size, 1)))
                _fill_q_uniform_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, work_buf, out)
            else:
                h1, h2, ih1, ih21, ih32, ih2, ih31, ih3 = _prep_newton_geom(lnk)
                d0, d1, d2, d3 = _build_newton_from_tab(tab, ih1, ih21, ih32, ih2, ih31, ih3)
                nl_buf = max(ln_nu.size, 1)
                work_buf = np.empty((nchunk, nl_buf))
                work_j0 = np.empty((nchunk, nl_buf), dtype=np.int32)
                work_u = np.empty((nchunk, nl_buf))
                _fill_q_newton_cursor_fast(d0, d1, d2, d3, h1, h2, lnk, iz, t, lnchi, ln_nu, work_buf, work_j0, work_u,
                                           out)
        else:
            work_row = np.empty((nchunk, lnk.size))
            if uniform:
                _fill_q_uniform_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, out)
            else:
                _fill_q_cursor_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_den, work_row, out)
    else:
        work_row = np.empty((nchunk, max(lnk.size, 1)))
        if use_exp:
            work_buf = np.empty((nchunk, max(ln_nu.size, 1)))
            if uniform:
                _fill_q_linear_uniform_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, work_buf, out)
            else:
                _fill_q_linear_cursor_fast(tab, lnk, iz, t, lnchi, ln_nu, inv_dk, work_row, work_buf, out)
        else:
            if uniform:
                _fill_q_linear_uniform_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dlnk, work_row, out)
            else:
                _fill_q_linear_cursor_kcap(tab, lnk, iz, t, lnchi, ln_nu, inv_dk, work_row, out)
    return out


# ------------------------------------------------------------------ thread-safe LRU geometry/stencil/growth caches
_CACHE_LOCK = threading.RLock()
_DIST_CACHE_MAX = 8
_STENCIL_CACHE_MAX = 4
_GROWTH_CACHE_MAX = 8
_QT_POOL_MAX = 4
_DIST_CACHE: OrderedDict[int, "_DistEntry"] = OrderedDict()
_STENCIL_CACHE: OrderedDict[int, "_StencilEntry"] = OrderedDict()
_GROWTH_CACHE: OrderedDict[int, "_GrowthEntry"] = OrderedDict()
_QT_POOL: list[np.ndarray] = []
_NEXT_DIST_ID = 0
_NEXT_STENCIL_ID = 0
_NEXT_GROWTH_ID = 0


def _clear_limber_caches() -> None:
    """Clear module-level Limber distance, stencil, growth caches, and ``_QT_POOL`` (for tests)."""
    with _CACHE_LOCK:
        _DIST_CACHE.clear()
        _STENCIL_CACHE.clear()
        _GROWTH_CACHE.clear()
        _QT_POOL.clear()


def _acquire_qt(nm: int, nl: int) -> np.ndarray:
    """Pop a matching pooled ``Qt`` buffer if available, else allocate a fresh ``(nm, nl)`` array."""
    shape = (nm, nl)
    with _CACHE_LOCK:
        for i in range(len(_QT_POOL) - 1, -1, -1):
            if _QT_POOL[i].shape == shape:
                return _QT_POOL.pop(i)
    return np.empty(shape, dtype=np.float64)


def _is_poolable_qt(qt) -> bool:
    """Return True iff ``qt`` is a standalone large C-contiguous writeable 2-D float64 array."""
    return (
        isinstance(qt, np.ndarray)
        and qt.base is None
        and qt.dtype == np.float64
        and qt.flags.c_contiguous
        and qt.flags.writeable
        and qt.ndim == 2
        and qt.size >= 100_000
    )


def _push_qt_pool(qt: np.ndarray) -> None:
    """Return an unreferenced ``Qt`` buffer to ``_QT_POOL`` up to ``_QT_POOL_MAX``."""
    with _CACHE_LOCK:
        if len(_QT_POOL) < _QT_POOL_MAX:
            _QT_POOL.append(qt)


def _freeze(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _ro_store(a: np.ndarray) -> np.ndarray:
    c = np.array(a, dtype=np.float64, copy=True, order="C")
    c.flags.writeable = False
    return c


def _arr_match(a: np.ndarray, ro: np.ndarray) -> bool:
    return np.array_equal(a, ro)


@dataclass
class _DistEntry:
    """Cached background-distance spline evaluation and integration weights on ``[0, z_max]``."""

    dist_id: int
    bucket: tuple
    z_tab_ro: np.ndarray
    chi_tab_ro: np.ndarray
    z_max: float
    z: np.ndarray
    dz: float
    chi: np.ndarray
    _zf: np.ndarray
    _chif: np.ndarray
    dchidz: np.ndarray
    w: np.ndarray
    inv_chi: np.ndarray
    lnchi: np.ndarray


@dataclass
class _StencilEntry:
    """Cached ``_fill_q_prepare`` outputs, Newton geometry, and cubic stencils."""

    stencil_id: int
    bucket: tuple
    z_pk_ro: np.ndarray
    k_h_ro: np.ndarray
    ell_ro: np.ndarray
    lnk: np.ndarray
    ln_nu: np.ndarray
    iz: np.ndarray
    t: np.ndarray
    uniform: bool
    inv_dlnk: float
    inv_dk: np.ndarray
    inv_den: np.ndarray
    h1: np.ndarray | None
    h2: np.ndarray | None
    ih1: np.ndarray | None
    ih21: np.ndarray | None
    ih32: np.ndarray | None
    ih2: np.ndarray | None
    ih31: np.ndarray | None
    ih3: np.ndarray | None
    las: np.ndarray | None
    lbs: np.ndarray | None
    j0s: np.ndarray | None
    us: np.ndarray | None
    hits: int = 0


@dataclass
class _GrowthEntry:
    """Cached NLA growth factor ``D(z)`` evaluated on integration nodes."""

    growth_id: int
    bucket: tuple
    z_pk_ro: np.ndarray
    D_pk_ro: np.ndarray
    D_ro: np.ndarray


def _get_dist_entry(chi_tab: np.ndarray, z_tab: np.ndarray, z_max_val: float, n_nodes: int,
                    kernel_oversample: int) -> _DistEntry:
    global _NEXT_DIST_ID
    bucket = (z_tab.size, z_max_val, n_nodes, kernel_oversample)
    with _CACHE_LOCK:
        for did in reversed(_DIST_CACHE):
            entry = _DIST_CACHE[did]
            if (entry.bucket == bucket
                    and _arr_match(z_tab, entry.z_tab_ro)
                    and _arr_match(chi_tab, entry.chi_tab_ro)):
                _DIST_CACHE.move_to_end(did)
                return entry
        if (not (np.all(np.isfinite(z_tab)) and np.all(np.isfinite(chi_tab)))
                or np.any(np.diff(z_tab) <= 0) or np.any(np.diff(chi_tab) <= 0)
                or abs(z_tab[0]) > 1e-8):
            raise ValueError("z_tab must start at 0 and chi_tab must increase strictly with z_tab")
        z_tab_ro = _ro_store(z_tab)
        chi_tab_ro = _ro_store(chi_tab)
        z = np.linspace(0.0, z_max_val, n_nodes)
        spl = CubicSpline(z_tab_ro, chi_tab_ro)
        chi = spl(z)
        chi[0] = max(chi[0], 0.0)
        zf = np.linspace(0.0, z_max_val, (n_nodes - 1) * kernel_oversample + 1)
        chif = spl(zf)
        chif[0] = max(chif[0], 0.0)
        dz = float(z[1] - z[0])
        dchidz = spl(z, 1)
        w = np.full(z.size, dz)
        w[[0, -1]] *= 0.5
        inv_chi = np.divide(1.0, chi, out=np.zeros_like(chi), where=chi > 0)
        with np.errstate(divide="ignore"):
            lnchi = np.where(chi > 0, np.log(np.where(chi > 0, chi, 1.0)), np.nan)
        _NEXT_DIST_ID += 1
        did = _NEXT_DIST_ID
        entry = _DistEntry(
            dist_id=did,
            bucket=bucket,
            z_tab_ro=z_tab_ro,
            chi_tab_ro=chi_tab_ro,
            z_max=z_max_val,
            z=_freeze(z),
            dz=dz,
            chi=_freeze(chi),
            _zf=_freeze(zf),
            _chif=_freeze(chif),
            dchidz=_freeze(dchidz),
            w=_freeze(w),
            inv_chi=_freeze(inv_chi),
            lnchi=_freeze(lnchi),
        )
        _DIST_CACHE[did] = entry
        while len(_DIST_CACHE) > _DIST_CACHE_MAX:
            _DIST_CACHE.popitem(last=False)
        return entry


def _get_stencil_entry(dist: _DistEntry, z_pk: np.ndarray, k_h: np.ndarray, ell: np.ndarray,
                       interp_order: int) -> _StencilEntry:
    global _NEXT_STENCIL_ID
    bucket = (dist.dist_id, z_pk.size, k_h.size, ell.size, interp_order)
    with _CACHE_LOCK:
        for sid in reversed(_STENCIL_CACHE):
            entry = _STENCIL_CACHE[sid]
            if (entry.bucket == bucket
                    and _arr_match(z_pk, entry.z_pk_ro)
                    and _arr_match(k_h, entry.k_h_ro)
                    and _arr_match(ell, entry.ell_ro)):
                _STENCIL_CACHE.move_to_end(sid)
                return entry
        if (not (np.all(np.isfinite(z_pk)) and np.all(np.isfinite(k_h)))
                or np.any(np.diff(z_pk) <= 0) or np.any(np.diff(k_h) <= 0) or k_h[0] <= 0):
            raise ValueError("z_pk and k_h must be strictly ascending (k_h > 0)")
        z_pk_ro = _ro_store(z_pk)
        k_h_ro = _ro_store(k_h)
        ell_ro = _ro_store(ell)
        lnk = _freeze(np.log(k_h_ro))
        ln_nu = _freeze(np.log(ell_ro + 0.5))
        iz, t, uniform, inv_dlnk, inv_dk, inv_den = _fill_q_prepare(
            z_pk_ro, dist.z, dist.lnchi, lnk, UNIFORM_LNK_RTOL)
        iz = _freeze(iz)
        t = _freeze(t)
        inv_dk = _freeze(inv_dk)
        inv_den = _freeze(inv_den)
        if interp_order == 3 and lnk.size >= 4:
            h1, h2, ih1, ih21, ih32, ih2, ih31, ih3 = (_freeze(a) for a in _prep_newton_geom(lnk))
        else:
            h1 = h2 = ih1 = ih21 = ih32 = ih2 = ih31 = ih3 = None
        las = lbs = j0s = us = None
        _NEXT_STENCIL_ID += 1
        sid = _NEXT_STENCIL_ID
        entry = _StencilEntry(
            stencil_id=sid,
            bucket=bucket,
            z_pk_ro=z_pk_ro,
            k_h_ro=k_h_ro,
            ell_ro=ell_ro,
            lnk=lnk,
            ln_nu=ln_nu,
            iz=iz,
            t=t,
            uniform=bool(uniform),
            inv_dlnk=float(inv_dlnk),
            inv_dk=inv_dk,
            inv_den=inv_den,
            h1=h1,
            h2=h2,
            ih1=ih1,
            ih21=ih21,
            ih32=ih32,
            ih2=ih2,
            ih31=ih31,
            ih3=ih3,
            las=las,
            lbs=lbs,
            j0s=j0s,
            us=us,
            hits=0,
        )
        _STENCIL_CACHE[sid] = entry
        while len(_STENCIL_CACHE) > _STENCIL_CACHE_MAX:
            _STENCIL_CACHE.popitem(last=False)
        return entry


def _ensure_cubic_stencils(st: _StencilEntry, dist: _DistEntry) -> bool:
    """Lazily precompute cubic stencil indices on the second hit so cold one-shot calls stay fast."""
    with _CACHE_LOCK:
        if st.j0s is not None:
            return True
        if st.hits == 0:
            st.hits = 1
            return False
        las, lbs, j0s, us = (_freeze(a) for a in _precompute_cubic_stencils(st.lnk, st.iz, dist.lnchi, st.ln_nu))
        st.las = las
        st.lbs = lbs
        st.us = us
        st.j0s = j0s
        return True


def _get_growth_entry(dist: _DistEntry, z_pk: np.ndarray, D_pk: np.ndarray) -> np.ndarray:
    global _NEXT_GROWTH_ID
    bucket = (dist.dist_id, z_pk.size)
    with _CACHE_LOCK:
        for gid in reversed(_GROWTH_CACHE):
            entry = _GROWTH_CACHE[gid]
            if (entry.bucket == bucket
                    and _arr_match(z_pk, entry.z_pk_ro)
                    and np.array_equal(D_pk, entry.D_pk_ro)):
                _GROWTH_CACHE.move_to_end(gid)
                return entry.D_ro
        D = CubicSpline(z_pk, D_pk)(dist.z) if z_pk.size > 3 else np.interp(dist.z, z_pk, D_pk)
        D_ro = _freeze(D)
        _NEXT_GROWTH_ID += 1
        gid = _NEXT_GROWTH_ID
        entry = _GrowthEntry(
            growth_id=gid,
            bucket=bucket,
            z_pk_ro=_ro_store(z_pk),
            D_pk_ro=_ro_store(D_pk),
            D_ro=D_ro,
        )
        _GROWTH_CACHE[gid] = entry
        while len(_GROWTH_CACHE) > _GROWTH_CACHE_MAX:
            _GROWTH_CACHE.popitem(last=False)
        return D_ro


class Limber:
    """Limber projection with a reusable power-spectrum look-up.

    Typical use (the cosmology changes rarely, the nuisance parameters often)::

        lim = Limber(ell=default_ell(), fidelity="fast", n_nodes=2000)
        lim.set_power(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl)   # slow: builds Q[m, l]
        C = lim.cl(z_nz, nz_shifted, A_IA)                         # fast: kernels + contraction

    Parameters
    ----------
    ell : (nell,) strictly ascending multipoles (floats allowed); default :func:`default_ell`.
    fidelity : ``"fast"`` (spec E) or ``"kcap"`` (P interpolated linearly in value, W / Z_i).
    n_nodes : number of uniform z nodes of the trapezoid rule (spec s15.4: 2000 -> 1.5e-4).
    interp_order : 3 (cubic Lagrange in ln k, default) or 1 (linear).
    kernel_oversample : the cumulative integrals A_i, B_i defining W_i are evaluated on a grid that is
        this many times finer than the Limber nodes (cheap: O(nbin * nodes)) and sub-sampled, which
        removes the O(dz^2) trapezoid error caused by the kinks of the piecewise-linear n(z).
    """

    def __init__(self, ell=None, fidelity: str = "fast", n_nodes: int = 2000, interp_order: int = 3,
                 kernel_oversample: int = 4):
        if fidelity not in ("fast", "kcap"):
            raise ValueError("fidelity must be 'fast' or 'kcap'")
        self.ell = default_ell() if ell is None else np.ascontiguousarray(ell, dtype=np.float64)
        if self.ell.ndim != 1 or self.ell.size == 0 or not np.all(np.isfinite(self.ell)) or self.ell[0] <= -0.5:
            raise ValueError("ell must be a non-empty 1-D array of finite values > -1/2 (nu = l + 1/2 > 0)")
        if np.any(np.diff(self.ell) <= 0):
            raise ValueError("ell must be strictly ascending")
        if int(n_nodes) < 3:
            raise ValueError("n_nodes must be >= 3")
        self.fidelity = fidelity
        self.n_nodes = int(n_nodes)
        self.interp_order = int(interp_order)
        self.kernel_oversample = max(1, int(kernel_oversample))
        self._ready = False
        self._nz_cache = None

    def __del__(self):
        try:
            qt = getattr(self, "Qt", None)
            if qt is not None:
                self.Qt = None
                if _is_poolable_qt(qt) and sys.getrefcount(qt) == 2:
                    _push_qt_pool(qt)
        except Exception:
            pass

    # ------------------------------------------------------------------ power-spectrum stage
    def set_power(self, chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl, z_max=None):
        """Build everything that depends on the cosmology only.

        Parameters
        ----------
        chi_tab, z_tab : (nzb,) comoving radial distance [Mpc/h] vs redshift (z_tab ascending).
        omega_m : Omega_m including massive neutrinos.
        z_pk : (nz,) redshifts of ``pk_nl`` (ascending, ``z_pk[0] = 0``).
        k_h : (nk,) wavenumbers [h/Mpc] ascending (already extended to 500 h/Mpc, spec s2.6).
        pk_nl : (nz, nk) non-linear matter power [(Mpc/h)^3].
        z_max : upper integration limit; default ``min(z_tab[-1], z_pk[-1])``.
        """
        z_tab_in = np.asarray(z_tab, dtype=np.float64)
        chi_tab_in = np.asarray(chi_tab, dtype=np.float64)
        z_pk_in = np.ascontiguousarray(z_pk, dtype=np.float64)
        k_h_in = np.ascontiguousarray(k_h, dtype=np.float64)
        pk_nl_in = np.ascontiguousarray(pk_nl, dtype=np.float64)
        if z_pk_in.ndim != 1 or k_h_in.ndim != 1 or z_pk_in.size < 1 or k_h_in.size < 1:
            raise ValueError("z_pk and k_h must be non-empty 1-D arrays")
        if pk_nl_in.shape != (z_pk_in.size, k_h_in.size):
            raise ValueError("pk_nl must have shape (len(z_pk), len(k_h))")
        if not np.isfinite(omega_m) or omega_m <= 0:
            raise ValueError("omega_m must be positive and finite")
        if z_tab_in.ndim != 1 or z_tab_in.shape != chi_tab_in.shape or z_tab_in.size < 4:
            raise ValueError("chi_tab and z_tab must be 1-D arrays of equal length >= 4")
        z_max_val = float(min(z_tab_in[-1], z_pk_in[-1]) if z_max is None else z_max)
        if not np.isfinite(z_max_val) or not (0.0 < z_max_val <= z_tab_in[-1]):
            raise ValueError("z_max must lie in (0, z_tab[-1]] (the distance table is not extrapolated)")

        dist = _get_dist_entry(chi_tab_in, z_tab_in, z_max_val, self.n_nodes, self.kernel_oversample)
        st = _get_stencil_entry(dist, z_pk_in, k_h_in, self.ell, self.interp_order)

        nk = k_h_in.size
        use_exp = self.fidelity == "fast"
        cubic = self.interp_order == 3 and nk >= 4
        if cubic and use_exp and not st.uniform:
            ok, d0, d1, d2, d3 = _validate_log_and_newton(
                pk_nl_in, st.ih1, st.ih21, st.ih32, st.ih2, st.ih31, st.ih3)
            if not ok:
                raise ValueError("pk_nl must be finite and strictly positive")
        elif use_exp:
            ok, tab = _validate_and_log(pk_nl_in)
            if not ok:
                raise ValueError("pk_nl must be finite and strictly positive")
        else:
            if not _validate_pk_positive(pk_nl_in):
                raise ValueError("pk_nl must be finite and strictly positive")

        # Validate pk_nl before nla_growth_factor (division/sqrt) and validate everything before mutating self.
        D_pk = nla_growth_factor(z_pk_in, pk_nl_in)
        D_ro = _get_growth_entry(dist, z_pk_in, D_pk)

        nm = dist.z.size
        nl = self.ell.size
        nchunk = max(1, min(nm, _CHUNKS_PER_THREAD * numba.get_num_threads()))
        Qt = _acquire_qt(nm, nl)
        if cubic:
            if use_exp:
                work_buf = np.empty((nchunk, max(nl, 1)))
                if st.uniform:
                    work_row = np.empty((nchunk, nk))
                    _fill_q_uniform_fast(tab, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_dlnk,
                                         work_row, work_buf, Qt)
                elif _ensure_cubic_stencils(st, dist):
                    _fill_q_newton_cached_fast(d0, d1, d2, d3, st.h1, st.h2, st.iz, st.t,
                                               st.las, st.lbs, st.j0s, st.us, work_buf, Qt)
                else:
                    nl_buf = max(nl, 1)
                    work_j0 = np.empty((nchunk, nl_buf), dtype=np.int32)
                    work_u = np.empty((nchunk, nl_buf))
                    _fill_q_newton_cursor_fast(d0, d1, d2, d3, st.h1, st.h2, st.lnk, st.iz, st.t,
                                               dist.lnchi, st.ln_nu, work_buf, work_j0, work_u, Qt)
            else:
                work_row = np.empty((nchunk, nk))
                if st.uniform:
                    _fill_q_uniform_kcap(pk_nl_in, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_dlnk,
                                         work_row, Qt)
                elif _ensure_cubic_stencils(st, dist):
                    _fill_q_cached_kcap(pk_nl_in, st.lnk, st.inv_den, st.iz, st.t, dist.lnchi, st.ln_nu,
                                        st.las, st.lbs, st.j0s, work_row, Qt)
                else:
                    _fill_q_cursor_kcap(pk_nl_in, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_den,
                                        work_row, Qt)
        else:
            work_row = np.empty((nchunk, max(nk, 1)))
            if use_exp:
                work_buf = np.empty((nchunk, max(nl, 1)))
                if st.uniform:
                    _fill_q_linear_uniform_fast(tab, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_dlnk,
                                                work_row, work_buf, Qt)
                else:
                    _fill_q_linear_cursor_fast(tab, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_dk,
                                               work_row, work_buf, Qt)
            else:
                if st.uniform:
                    _fill_q_linear_uniform_kcap(pk_nl_in, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_dlnk,
                                                work_row, Qt)
                else:
                    _fill_q_linear_cursor_kcap(pk_nl_in, st.lnk, st.iz, st.t, dist.lnchi, st.ln_nu, st.inv_dk,
                                               work_row, Qt)

        old_qt = getattr(self, "Qt", None)
        self.omega_m = float(omega_m)
        self.s = lensing_prefactor(omega_m)
        self.z_max = dist.z_max
        self._zf = dist._zf
        self._chif = dist._chif
        self.z = dist.z
        self.dz = dist.dz
        self.chi = dist.chi
        self.dchidz = dist.dchidz
        self.w = dist.w
        self.inv_chi = dist.inv_chi
        self.D = D_ro
        self.Qt = Qt  # (nm, nell)
        self._nz_cache = None
        self._ready = True
        if old_qt is not None and _is_poolable_qt(old_qt) and sys.getrefcount(old_qt) == 2:
            _push_qt_pool(old_qt)
        return self

    # ------------------------------------------------------------------ kernel stage
    def nla_factor(self, A_IA: float) -> np.ndarray:
        """``F(z_m) = -A_IA * 0.0138731 * Omega_m / D(z_m)`` on the integration nodes (spec s5)."""
        return -A_IA * C1_RHO_CRIT * self.omega_m / self.D

    def _kernels(self, z_nz, nz_shifted):
        """Return (W, n_chi) on the integration nodes; cached for repeated identical n(z)."""
        key = self._nz_cache
        if key is not None and np.array_equal(key[0], nz_shifted) and np.array_equal(key[1], z_nz):
            return key[2], key[3]
        n_in = np.asarray(nz_shifted, dtype=np.float64)
        z_nz = np.asarray(z_nz, dtype=np.float64)
        if z_nz[-1] > self.z_max + 1e-12 and np.any(n_in[:, z_nz > self.z_max + 1e-12] != 0):
            raise ValueError("n(z) has support above the integration limit z_max")
        r = self.kernel_oversample
        nf = np.stack([np.interp(self._zf, z_nz, row, left=0.0, right=0.0) for row in n_in])
        Wf = lensing_efficiency(self._zf, self._chif, nf)
        W = Wf[:, ::r]
        n = nf[:, ::r]
        if self.fidelity == "kcap":
            W = W / (_rev_cumtrapz(nf, self._zf[1] - self._zf[0])[:, :1])
        n_chi = n / self.dchidz  # [h/Mpc]
        self._nz_cache = (n_in.copy(), z_nz.copy(), W, n_chi)
        return W, n_chi

    def kernels(self, z_nz, nz_shifted, A_IA: float = 0.0):
        """Return ``q (nbin, nm)`` = s W_i + F n_i(chi) on the integration nodes ``self.z``."""
        W, n_chi = self._kernels(z_nz, nz_shifted)
        A = np.asarray(A_IA, dtype=np.float64)
        if A.ndim == 0:
            return self.s * W + self.nla_factor(float(A))[None, :] * n_chi
        # per-bin amplitude array (nbin,): F_i(z) = -A_i C1rho_crit Omega_m / D(z)  (mass-dependent NLA, KiDS-Legacy)
        F = -A[:, None] * C1_RHO_CRIT * self.omega_m / self.D[None, :]
        return self.s * W + F * n_chi

    def cl(self, nz_shifted, A_IA: float = 0.0, z_nz=None) -> np.ndarray:
        """Limber ``C_ij(l)`` for all bin pairs, shape ``(nbin, nbin, nell)`` (symmetric).

        ``C = sum_m w_m (dchi/dz)_m q_i q_j / chi_m^2 Q[m, l]`` -- one matmul over all pairs.
        """
        if not self._ready:
            raise RuntimeError("call set_power first")
        z_nz = self._z_nz if z_nz is None else z_nz
        if z_nz is None:
            raise ValueError("z_nz required")
        q = self.kernels(z_nz, nz_shifted, A_IA)
        nb = q.shape[0]
        ii, jj = np.tril_indices(nb)
        meas = self.w * self.dchidz * self.inv_chi**2
        KK = q[ii] * q[jj] * meas[None, :]  # (npair, nm)
        Cp = KK @ self.Qt  # (npair, nell)
        C = np.empty((nb, nb, self.ell.size))
        C[ii, jj] = Cp
        C[jj, ii] = Cp
        return C

    # allow Limber(z_nz=...) style default
    _z_nz = None

    def set_nz_grid(self, z_nz):
        """Store the default z grid of the n(z) tables so :meth:`cl` can omit it."""
        self._z_nz = np.asarray(z_nz, dtype=np.float64)
        return self


def cl_shear(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl, z_nz, nz_shifted, A_IA, ell,
             fidelity: str = "fast", n_nodes: int = 2000) -> np.ndarray:
    """One-shot Limber convergence spectra ``C_ij(l)`` (GG+GI+IG+II folded), shape (nbin, nbin, nell).

    ``C_ij(l) = int dchi q_i q_j / chi^2 P_nl((l+1/2)/chi, z)``, see the module docstring.
    Inputs: ``chi_tab`` [Mpc/h] at ``z_tab``; ``omega_m``; ``pk_nl[z_pk, k_h]`` in (Mpc/h)^3 with
    k_h in h/Mpc extended to 500 h/Mpc; ``nz_shifted[nbin, len(z_nz)]`` normalised; ``A_IA`` the NLA
    amplitude; ``ell`` ascending multipoles.  For repeated calls use :class:`Limber` to reuse the
    power-spectrum look-up.
    """
    lim = Limber(ell=ell, fidelity=fidelity, n_nodes=n_nodes)
    lim.set_power(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl)
    return lim.cl(nz_shifted, A_IA, z_nz=z_nz)
