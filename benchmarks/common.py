"""Shared helpers for the performance benchmarks / accuracy studies (points in the prior box, chi2 evaluation)."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from kidspipe.params import BOUNDS, PARAM_NAMES, GAUSSIAN_PRIORS, START

IDX = {n: i for i, n in enumerate(PARAM_NAMES)}

#: default cosmology box for test points (omch2 restricted to the emulator box, rest = full prior box)
TEST_BOX = {"omch2": (0.04, 0.30), "ombh2": (0.019, 0.026), "h": (0.64, 0.82), "n_s": (0.8, 1.2)}


def draw_points(n: int, seed: int, box: dict | None = None, s8_range=(0.65, 0.85), lin=None) -> np.ndarray:
    """``n`` full 14-vectors: Sobol in (omch2, ombh2, h, n_s) inside ``box``, ln A_s chosen so that
    S8 = sigma8 sqrt(Om/0.3) ~ U(s8_range) (data-compatible amplitudes, chi2 ~ 190-300), A_baryon ~ U(2, 3.13),
    A_IA ~ U(-1.5, 1.5), delta_c/A_c/dz drawn from their (truncated) Gaussian priors."""
    from kidspipe.theory.linear import LinearBackend
    box = dict(TEST_BOX if box is None else box)
    lin = LinearBackend() if lin is None else lin
    rng = np.random.default_rng(seed)
    names = ("omch2", "ombh2", "h", "n_s")
    sob = qmc.Sobol(d=5, scramble=True, seed=seed).random(n)
    th = np.tile(np.asarray(START, float), (n, 1))
    for j, nm in enumerate(names):
        lo, hi = box[nm]
        th[:, IDX[nm]] = lo + (hi - lo) * sob[:, j]
    s8t = s8_range[0] + (s8_range[1] - s8_range[0]) * sob[:, 4]
    for i in range(n):
        r = lin.compute(th[i, 0], th[i, 1], th[i, 2], th[i, 3], 3.0)
        s8_ref = r.sigma8_0
        # S8 = sigma8 sqrt(Om/0.3)  ->  sigma8 target
        s8 = s8t[i] / np.sqrt(r.omega_m / 0.3)
        th[i, IDX["ln_1e10_A_s"]] = 3.0 + 2.0 * np.log(s8 / s8_ref)
    th[:, IDX["A_baryon"]] = rng.uniform(2.0, 3.13, n)
    th[:, IDX["A_IA"]] = rng.uniform(-1.5, 1.5, n)
    for nm, (mu, sg) in GAUSSIAN_PRIORS.items():
        lo, hi = BOUNDS[IDX[nm]]
        x = rng.normal(mu, sg, n)
        th[:, IDX[nm]] = np.clip(x, lo, hi)
    return th


def chi2_of(model, like, theta) -> float:
    mu = model.mu(theta)
    return float(like.chi2(mu, A_c=theta[IDX["A_c"]], delta_c=theta[IDX["delta_c"]]))
