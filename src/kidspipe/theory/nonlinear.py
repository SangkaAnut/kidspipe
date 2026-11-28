"""Non-linear matter power spectrum dispatcher + high-k extension (spec s2.5, s2.6).

Public API
----------
``get_pnl(lin_or_params, A_baryon, backend="hmcode")``  ->  ``pk_nl`` (nz, nk) on the grid of ``lin``
``extend_high_k(k_h, pk, khmax=500., nnew=200)``        ->  ``(k_ext, pk_ext)``      (spec s2.6)
``BACKENDS``  dict ``name -> callable(lin, A_baryon, **kw) -> pk_nl``; a new backend is added by
``BACKENDS["name"] = func`` inside its own delimited section below (no other edit needed).

Units: k in h/Mpc, P in (Mpc/h)^3.

The file is organised in delimited sections so several authors can work on it without conflicts:
  SECTION 1  shared helpers (baryon-parameter map)
  SECTION 2  backend "camb"   (CAMB Halofit 'mead' = HMcode-2016)
  SECTION 3  high-k extension (spec s2.6)
  SECTION 4  backend "hmcode" (reserved: stand-alone numba HMcode, added by another agent)
  SECTION 5  registry + get_pnl dispatcher  (keep last)
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

__all__ = ["get_pnl", "extend_high_k", "extend_high_k_batch", "BACKENDS", "eta_baryon"]

# =====================================================================================================
# SECTION 1: shared helpers
# =====================================================================================================
#: eta0 = a0 + a1 * A_baryon,  a0 = 0.98, a1 = -0.12  (spec s1.2)  ->  A=3.13: 0.6044 ; A=2: 0.74
ETA_A0, ETA_A1 = 0.98, -0.12


def eta_baryon(A_baryon: float) -> float:
    """HMcode-2016 window-mass exponent eta0 = 0.98 - 0.12 A_baryon (spec s1.2)."""
    return ETA_A0 + ETA_A1 * float(A_baryon)


# =====================================================================================================
# SECTION 2: backend "camb"  (CAMB Halofit, halofit_version='mead' == HMcode-2016, spec s2.1/s2.5)
# =====================================================================================================
def _pnl_camb(lin, A_baryon: float, eta0: float | None = None) -> np.ndarray:
    """P_nl(z,k_h) [(Mpc/h)^3] from CAMB's HMcode-2016 with ``HMCode_A_baryon=A``, ``HMCode_eta_baryon=0.98-0.12A``.

    Re-uses the CAMB transfer functions cached by :class:`~kidspipe.theory.linear.LinearBackend`; only the
    cheap transfer->power + halofit stage is repeated (A_s, n_s and A_baryon are applied in place).
    The result lives on exactly the same (z, k_h) grid as ``lin.pk_lin`` (CAMB native nodes, k_h <= 21/h).
    """
    state = getattr(lin, "_state", None)
    if state is None:
        raise ValueError("backend 'camb' needs a LinearResult produced by LinearBackend.compute "
                         "(it carries the CAMB state); got a result without it")
    eta = eta_baryon(A_baryon) if eta0 is None else float(eta0)
    return state.camb_nonlinear(float(A_baryon), eta)


# =====================================================================================================
# SECTION 3: high-k extension (spec s2.6)
# =====================================================================================================
def extend_high_k(k_h, pk, khmax: float = 500.0, nnew: int = 200):
    """Log-log linear extrapolation of P(k) to ``khmax`` (spec s2.6), applied row by row.

    For each redshift row, a straight line  ln P = a + b ln k  is fitted by least squares to the LAST 3
    nodes, and evaluated on  ln k_new = linspace(ln k_max, ln khmax, nnew)[1:]  (``nnew-1`` = 199 new nodes,
    uniform in ln k).  No low-k extension.  Works for P_lin and P_nl.

    Parameters
    ----------
    k_h : (nk,) ascending [h/Mpc];  pk : (nz, nk) or (nk,) [(Mpc/h)^3] (strictly positive).

    Returns
    -------
    k_ext : (nk + nnew - 1,), pk_ext : (nz, nk + nnew - 1) (or 1-d if ``pk`` was 1-d).
    """
    k_h = np.asarray(k_h, dtype=np.float64)
    pk = np.asarray(pk, dtype=np.float64)
    one_d = pk.ndim == 1
    P = pk[None, :] if one_d else pk
    if P.shape[1] != k_h.size:
        raise ValueError("pk last axis must match k_h")
    if k_h.size < 3:
        raise ValueError("need at least 3 nodes for the fit")
    if khmax <= k_h[-1]:
        raise ValueError("khmax must exceed the last k node")
    x = np.log(k_h[-3:])
    y = np.log(P[:, -3:])                                   # (nz, 3)
    xm = x.mean()
    b = ((x - xm) * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / ((x - xm) ** 2).sum()  # LS slope
    a = y.mean(axis=1) - b * xm
    xnew = np.linspace(x[-1], np.log(khmax), int(nnew))[1:]
    ynew = a[:, None] + b[:, None] * xnew[None, :]
    k_ext = np.concatenate([k_h, np.exp(xnew)])
    pk_ext = np.concatenate([P, np.exp(ynew)], axis=1)
    return (k_ext, pk_ext[0]) if one_d else (k_ext, pk_ext)


def extend_high_k_batch(k_h_batch, pk_batch, khmax: float = 500.0, nnew: int = 200):
    """Batched log-log linear extrapolation of ``pk_batch`` ``(U, nz, nk)`` to ``khmax``.

    ``k_h_batch`` may have shape ``(U, nk)`` or ``(nk,)`` (broadcast across ``U``).
    Returns ``(k_ext_batch (U, nk + nnew - 1), pk_ext_batch (U, nz, nk + nnew - 1))``.
    """
    P = np.asarray(pk_batch, dtype=np.float64)
    if P.ndim != 3:
        raise ValueError(f"pk_batch must have shape (U, nz, nk), got {P.shape}")
    u, nz, nk = P.shape
    k_b = np.asarray(k_h_batch, dtype=np.float64)
    if k_b.ndim == 1:
        k_b = np.broadcast_to(k_b, (u, nk))
    if k_b.shape != (u, nk):
        raise ValueError(f"k_h_batch shape {k_b.shape} incompatible with pk_batch {P.shape}")
    if nk < 3:
        raise ValueError("need at least 3 nodes for the fit")
    if u == 0:
        nn = max(int(nnew) - 1, 0)
        return np.empty((0, nk + nn), dtype=np.float64), np.empty((0, nz, nk + nn), dtype=np.float64)
    if np.any(khmax <= k_b[:, -1]):
        raise ValueError("khmax must exceed the last k node")
    nnew_i = int(nnew)
    x = np.log(k_b[:, -3:])                                        # (U, 3)
    y = np.log(P[:, :, -3:])                                       # (U, nz, 3)
    xm = x.mean(axis=1, keepdims=True)                             # (U, 1)
    dx = x - xm                                                    # (U, 3)
    ym = y.mean(axis=2, keepdims=True)                             # (U, nz, 1)
    b = np.sum(dx[:, None, :] * (y - ym), axis=2) / np.sum(dx * dx, axis=1)[:, None]
    a = ym[:, :, 0] - b * xm
    t = np.arange(1, nnew_i, dtype=np.float64) / (nnew_i - 1)
    xnew = x[:, -1:] + (np.log(khmax) - x[:, -1:]) * t[None, :]    # (U, nnew - 1)
    ynew = a[:, :, None] + b[:, :, None] * xnew[:, None, :]        # (U, nz, nnew - 1)
    k_ext = np.concatenate([k_b, np.exp(xnew)], axis=1)
    pk_ext = np.concatenate([P, np.exp(ynew)], axis=2)
    return k_ext, pk_ext


# =====================================================================================================
# SECTION 4: backend "hmcode" (reserved for the stand-alone numba HMcode; add `BACKENDS["hmcode"] = ...`)
# =====================================================================================================
def _pnl_hmcode(lin, A_baryon: float, eta0: float | None = None, **kw) -> np.ndarray:
    """P_nl(z,k_h) from the stand-alone numba HMcode-2016 (:mod:`kidspipe.theory.hmcode`).

    Needs only the arrays of the ``LinearResult`` (no CAMB state), so it works on cached/rescaled linear
    spectra and is ~50 ms single-threaded for 150 z x 194 k.  eta0 defaults to 0.98 - 0.12 A_baryon.
    """
    from .hmcode import hmcode_pnl
    eta = eta_baryon(A_baryon) if eta0 is None else float(eta0)
    return hmcode_pnl(lin.z, lin.k_h, lin.pk_lin, lin.omega_m, lin.omega_nu_frac, lin.h, lin.n_s,
                      float(A_baryon), eta, lin.pk_cb, **kw)


# =====================================================================================================
# SECTION 5: registry + dispatcher (keep last)
# =====================================================================================================
#: Registered non-linear backends: name -> ``callable(lin, A_baryon, **kwargs) -> pk_nl (nz, nk)``.
BACKENDS: Dict[str, Callable[..., np.ndarray]] = {"camb": _pnl_camb, "hmcode": _pnl_hmcode}

_default_linear = None


def _linear_from_params(params):
    """Build a LinearResult from a dict / sequence of cosmological parameters (shared default backend)."""
    global _default_linear
    from .linear import LinearBackend
    if _default_linear is None:
        _default_linear = LinearBackend()
    if isinstance(params, dict):
        d = params
    else:
        from ..params import PARAM_NAMES  # optional dependency on params.py
        d = dict(zip(PARAM_NAMES, params))
    return _default_linear.compute(d["omch2"], d["ombh2"], d["h"], d["n_s"], d["ln_1e10_A_s"],
                                   d.get("mnu", 0.06))


def get_pnl(lin_or_params, A_baryon: float, backend: str = "hmcode", **kwargs) -> np.ndarray:
    """Non-linear P(k,z) [(Mpc/h)^3] on the grid of ``lin`` (shape (nz, nk)).

    ``lin_or_params`` is a :class:`~kidspipe.theory.linear.LinearResult`, or a dict with keys
    ``omch2, ombh2, h, n_s, ln_1e10_A_s`` (``mnu`` optional) from which one is built.  The HMcode baryon
    parameters are ``A_baryon`` and ``eta0 = 0.98 - 0.12 A_baryon`` (spec s1.2).
    ``backend``: ``"camb"`` (CAMB Halofit 'mead') or any key of :data:`BACKENDS`.
    """
    if backend not in BACKENDS:
        raise KeyError(f"unknown non-linear backend {backend!r}; available: {sorted(BACKENDS)}")
    lin = lin_or_params
    if not hasattr(lin, "pk_lin"):
        lin = _linear_from_params(lin_or_params)
    return BACKENDS[backend](lin, A_baryon, **kwargs)
