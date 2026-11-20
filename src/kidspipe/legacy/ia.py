"""Mass-dependent NLA amplitude (Wright et al. 2025, Eq. 11): A_i = a f_r,i (M_i / M_piv)^beta, M_i = 10^log10M_i."""
from __future__ import annotations

import numpy as np

F_RED = np.array([0.158, 0.198, 0.206, 0.258, 0.207, 0.026])   # values.ini f_r_1..6
LOG10_M_PIV = 13.5


def ia_amplitudes(a, beta, log10_M_mean, f_r=F_RED, log10_m_piv: float = LOG10_M_PIV) -> np.ndarray:
    """Per-bin effective NLA amplitude (nbin,)."""
    lm = np.asarray(log10_M_mean, float)
    return float(a) * np.asarray(f_r, float) * 10.0 ** (float(beta) * (lm - log10_m_piv))
