"""Theory stack of kidspipe (linear P(k), non-linear P(k), n(z), Limber, Hankel, model).

Names are exported lazily so that ``import kidspipe.theory.<module>`` does not pull in CAMB/numba.
"""
from importlib import import_module

_LAZY = {
    "LinearBackend": "linear", "LinearResult": "linear", "Z_GRID_SPEC": "linear", "Z_GRID_REDUCED": "linear",
    "get_pnl": "nonlinear", "extend_high_k": "nonlinear", "BACKENDS": "nonlinear",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module("." + _LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LAZY)
