"""Small shims so the package runs on numpy 1.26 and numpy 2.x alike."""
from __future__ import annotations

import numpy as np

#: trapezoidal rule; ``np.trapz`` was renamed ``np.trapezoid`` in numpy 2.0 and later removed
trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = ["trapezoid"]
