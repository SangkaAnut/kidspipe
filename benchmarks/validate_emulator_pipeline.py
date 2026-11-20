#!/usr/bin/env python
"""Full-pipeline check of the transfer emulator: |Delta chi2| (emulator vs exact CAMB) at held-out prior-box points.

    python benchmarks/validate_emulator_pipeline.py [--n 50] [--seed 777] [--z-grid spec|reduced]

Points: Sobol in the emulator box (omch2, ombh2, h) x n_s in [0.8, 1.2], ln A_s set so that S8 ~ U(0.65, 0.85)
(data-compatible amplitudes), A_baryon, A_IA, delta_c, A_c, dz from their priors (see ``benchmarks/common.py``).
Writes runs/validate_emulator_pipeline.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import numpy as np  # noqa: E402

from common import chi2_of, draw_points  # noqa: E402
from kidspipe.likelihood.gauss import GaussianLikelihood  # noqa: E402
from kidspipe.theory.emulator import TransferEmulator  # noqa: E402
from kidspipe.theory.model import CosmicShearModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--z-grid", default="spec")
    ap.add_argument("--fidelity", default="fast")
    ap.add_argument("--pnl", default="hmcode")
    ap.add_argument("--out", default=str(ROOT / "runs" / "validate_emulator_pipeline.json"))
    a = ap.parse_args()
    em = TransferEmulator()
    box = {"omch2": tuple(em.box[0]), "ombh2": tuple(em.box[1]), "h": tuple(em.box[2]), "n_s": (0.8, 1.2)}
    like = GaussianLikelihood()
    m_c = CosmicShearModel(fidelity=a.fidelity, pnl_backend=a.pnl, z_grid=a.z_grid, nthreads=4, linear_backend="camb")
    pts = draw_points(a.n, a.seed, box=box, lin=m_c.linear)
    m_e = CosmicShearModel(fidelity=a.fidelity, pnl_backend=a.pnl, z_grid=a.z_grid, nthreads=4, linear_backend="emulator")
    c_c, c_e, s_c, s_e = [], [], [], []
    t0 = time.time()
    for th in pts:
        c_c.append(chi2_of(m_c, like, th)); s_c.append(m_c.last_derived["sigma8"])
        c_e.append(chi2_of(m_e, like, th)); s_e.append(m_e.last_derived["sigma8"])
    c_c, c_e = np.array(c_c), np.array(c_e)
    d = c_e - c_c
    res = dict(n=int(a.n), seed=a.seed, z_grid=a.z_grid, fidelity=a.fidelity, pnl=a.pnl, max_abs_dchi2=float(np.abs(d).max()),
               rms_dchi2=float(np.sqrt((d ** 2).mean())), mean_dchi2=float(d.mean()), chi2_range=[float(c_c.min()), float(c_c.max())],
               n_above_0p05=int((np.abs(d) > 0.05).sum()), n_fallback=int(m_e.linear.stats["fallback"]),
               sigma8_rel_diff_max=float(np.abs(np.array(s_e) / np.array(s_c) - 1).max()),
               dchi2=d.tolist(), chi2_camb=c_c.tolist(), points=pts.tolist(), seconds=time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if k not in ("dchi2", "chi2_camb", "points")}, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
