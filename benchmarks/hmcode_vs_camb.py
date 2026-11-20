#!/usr/bin/env python3
"""Benchmark of the numba HMcode-2016 (kidspipe.theory.hmcode) against CAMB 'mead'.

Prints, for the stored validation array and a set of cosmologies / baryon parameters including
KV450 prior corners, the max and rms fractional deviation (z<2 and z>=2 separately) and the timing
of the full 150 x ~195 evaluation (1 thread and N threads, after JIT).

Usage:  PYTHONPATH=src python benchmarks/hmcode_vs_camb.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import camb  # noqa: E402
import numba  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from kidspipe.theory import hmcode  # noqa: E402

ZS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9, 2.0, 3.0, 4.0, 6.0)

# name, omch2, ombh2, h, n_s, ln_1e10_As, A_baryon, mnu
CASES = [
    ("KV450 start, A=2.0", 0.10, 0.022, 0.70, 1.09, 3.09, 2.0, 0.06),
    ("KV450 start, A=3.13", 0.10, 0.022, 0.70, 1.09, 3.09, 3.13, 0.06),
    ("Planck-like, A=2.5", 0.12, 0.0222, 0.70, 0.97, 3.04, 2.5, 0.06),
    ("corner high Om/h/ns/As", 0.25, 0.025, 0.80, 1.20, 4.00, 2.0, 0.06),
    ("corner Om>1, h=0.64", 0.40, 0.022, 0.64, 1.00, 3.00, 3.13, 0.06),
    ("corner ns=0.7, lnAs=5", 0.12, 0.022, 0.70, 0.70, 5.00, 2.0, 0.06),
    ("corner Om=0.05, ns=1.3", 0.01, 0.022, 0.82, 1.30, 5.00, 2.0, 0.06),
    ("corner low Om, ns=0.8, lnAs=2.5", 0.05, 0.020, 0.65, 0.80, 2.50, 3.13, 0.06),
    ("massless neutrinos", 0.12, 0.0222, 0.70, 0.96, 3.05, 2.0, 0.0),
]


def camb_run(omch2, ombh2, h, ns, lnA, mnu, A, zs, nonlinear=True):
    p = camb.CAMBparams()
    if mnu > 0:
        p.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=mnu, num_massive_neutrinos=1, omk=0)
    else:
        p.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.0, num_massive_neutrinos=0, omk=0)
    p.InitPower.set_params(As=np.exp(lnA) * 1e-10, ns=ns)
    p.set_dark_energy(w=-1.0)
    p.NonLinearModel = camb.nonlinear.Halofit()
    p.NonLinearModel.set_params(halofit_version="mead", HMCode_A_baryon=A, HMCode_eta_baryon=0.98 - 0.12 * A)
    p.set_matter_power(redshifts=list(np.asarray(zs)[::-1]), kmax=20.0, nonlinear=nonlinear)
    p.NonLinear = camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none
    p.WantCls = False
    r = camb.get_results(p)
    kh, z, pl = r.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    pcb = r.get_linear_matter_power_spectrum(var1="delta_nonu", var2="delta_nonu",
                                             hubble_units=True, k_hunit=True)[2]
    pnl = r.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)[2] if nonlinear else None
    return dict(kh=kh, z=z, pl=pl, pcb=pcb, pnl=pnl, om=p.omegam, h=h, fnu=p.omnuh2 / (p.omegam * h * h), ns=ns)


def hm(o, A, eta0=None, **kw):
    eta0 = 0.98 - 0.12 * A if eta0 is None else eta0
    return hmcode.hmcode_pnl(o["z"], o["kh"], o["pl"], o["om"], o["fnu"], o["h"], o["ns"], A, eta0, o["pcb"], **kw)


def timeit(f, n=5):
    f()
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the timing section")
    args = ap.parse_args()

    print("=== stored array data/validation/pk_mead.txt (A=2.0, eta0=0.74), 1e-4<k<30 h/Mpc, all z ===")
    D = ROOT / "data" / "validation"
    kf, zf, pf = np.loadtxt(D / "k_h.txt"), np.loadtxt(D / "z.txt"), np.loadtxt(D / "pk_mead.txt")
    o = camb_run(0.1181838, 0.02228089, 0.679565, 0.9679701, np.log(2.170229e-9 * 1e10), 0.06, 2.0, zf)
    pn = hm(o, 2.0, 0.74)
    sel = (kf > 1e-4) & (kf < 30) & (kf > o["kh"][0]) & (kf < o["kh"][-1])
    for name, P in (("kidspipe hmcode", pn), ("CAMB 1.5.8 mead ", o["pnl"])):
        pm = np.exp(CubicSpline(np.log(o["kh"]), np.log(P), axis=1)(np.log(kf[sel])))
        r = pm / pf[:, sel] - 1
        print(f"  {name}: max {np.abs(r).max():.4f}  rms {np.sqrt((r ** 2).mean()):.4f}  (target rtol 1e-2)")
    d = np.abs(pn / o["pnl"] - 1)
    print(f"  kidspipe vs CAMB on the same cosmology: max {d.max():.2e}  rms {np.sqrt((d ** 2).mean()):.2e}")

    print("\n=== vs CAMB 'mead' (A_baryon, eta0 = 0.98-0.12 A); targets: z<2 <= 3e-3, z>=2 <= 1.5e-2 ===")
    print(f"{'case':34s} {'Om':>5s} | {'z<2 max':>9s} {'z<2 rms':>9s} | {'z>=2 max':>9s} {'z>=2 rms':>9s} | worst k, z")
    worst_lo = worst_hi = 0.0
    for name, omch2, ombh2, h, ns, lnA, A, mnu in CASES:
        o = camb_run(omch2, ombh2, h, ns, lnA, mnu, A, ZS)
        r = hm(o, A) / o["pnl"] - 1
        lo = o["z"] < 2
        ilo = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        worst_lo = max(worst_lo, np.abs(r[lo]).max())
        worst_hi = max(worst_hi, np.abs(r[~lo]).max())
        print(f"{name:34s} {o['om']:5.3f} | {np.abs(r[lo]).max():9.2e} {np.sqrt((r[lo] ** 2).mean()):9.2e} | "
              f"{np.abs(r[~lo]).max():9.2e} {np.sqrt((r[~lo] ** 2).mean()):9.2e} | "
              f"{o['kh'][ilo[1]]:.2f}, {o['z'][ilo[0]]:.2f}")
    print(f"worst over cases: z<2 {worst_lo:.2e}, z>=2 {worst_hi:.2e}")

    if args.quick:
        return
    print("\n=== timing: full KV450 grid (150 z x nk k nodes), after JIT ===")
    zs = np.concatenate([np.linspace(0, 2, 100, endpoint=False), np.linspace(2, 6, 50)])
    o = camb_run(0.10, 0.022, 0.70, 0.96, 3.0, 0.06, 2.0, zs, nonlinear=False)
    args_ = (o["z"], o["kh"], o["pl"], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74, o["pcb"])
    nmax = numba.get_num_threads()
    for nt in sorted({1, 4, nmax}):
        numba.set_num_threads(nt)
        for ms in (24, 16):
            t = timeit(lambda: hmcode.hmcode_pnl(*args_, ms=ms))
            print(f"  threads={nt:2d} ms={ms:2d}: {1e3 * t:7.1f} ms   ({o['pl'].shape[0]} z x {o['kh'].size} k)")
    numba.set_num_threads(1)
    sub = np.arange(0, 150, 6)
    a2 = (o["z"][sub], o["kh"], o["pl"][sub], o["om"], o["fnu"], o["h"], o["ns"], 2.0, 0.74, o["pcb"][sub])
    print(f"  threads= 1 reduced grid ({sub.size} z): {1e3 * timeit(lambda: hmcode.hmcode_pnl(*a2)):7.1f} ms")


if __name__ == "__main__":
    main()
