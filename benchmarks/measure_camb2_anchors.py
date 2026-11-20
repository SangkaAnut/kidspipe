"""Measure, under the installed CAMB, every physical value asserted by the spec-anchor tests.

Writes data/validation/anchors_camb2.json (used by tests/_anchors.py when CAMB >= 2):

    python benchmarks/measure_camb2_anchors.py data/validation/anchors_camb2.json
"""
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline

import camb
from kidspipe import api
from kidspipe.theory.linear import LinearBackend
from kidspipe.theory.nonlinear import get_pnl

SANITY = dict(omch2=0.10, ombh2=0.022, h=0.70, n_s=1.09, ln_1e10_A_s=3.09)
TESTCOSMO = dict(omch2=0.10, ombh2=0.022, h=0.70, n_s=0.96, ln_1e10_A_s=3.0)
v = {}
lb = LinearBackend(nthreads=4)
r = lb.compute(**SANITY)
v["linear.sanity.sigma8"] = float(r.sigma8_0)
v["linear.sanity.omega_m"] = float(r.omega_m)
v["linear.sanity.omnuh2"] = float(r.omnuh2)
v["linear.sanity.S8"] = float(r.S8)
v["linear.sanity.nk"] = int(r.k_h.size)
v["linear.sanity.kmin_h"] = float(r.k_h[0])
v["linear.sanity.kmax_h"] = float(r.k_h[-1])
k = r.k_h
i = int(np.searchsorted(k, 1.00298))
w = (1.00298 - k[i - 1]) / (k[i] - k[i - 1])
pn = get_pnl(r, 3.0, "camb")
v["nonlinear.sanity.pnl_z0_k1.00298"] = float(pn[0, i - 1] * (1 - w) + pn[0, i] * w)
v["nonlinear.sanity.plin_z0_k1.00298"] = float(r.pk_lin[0, i - 1] * (1 - w) + r.pk_lin[0, i] * w)
t = lb.compute(**TESTCOSMO)
v["linear.testcosmo.sigma8"] = float(t.sigma8_0)
v["linear.testcosmo.S8"] = float(t.S8)
v["linear.testcosmo.omega_m"] = float(t.omega_m)
v["linear.testcosmo.nk"] = int(t.k_h.size)
v["linear.testcosmo.kmax_h"] = float(t.k_h[-1])
chi = CubicSpline(t.z_chi, t.chi)
v["linear.testcosmo.chi_z0.5"] = float(chi(0.5))
v["linear.testcosmo.chi_z1"] = float(chi(1.0))
v["linear.testcosmo.Plin_z0_k0.1005"] = float(np.interp(0.1005, t.k_h, t.pk_lin[0]))
post = api.build_posterior()
d = api.derived_for(post, api.preset("test"))
v["e2e.test.sigma8"] = float(d["sigma8"])
v["e2e.test.omega_m"] = float(d["omega_m"])
v["e2e.test.S8"] = float(d["S8"])
out = {"camb_version": camb.__version__, "numpy_version": np.__version__,
       "note": "values measured by tests' own code paths under this CAMB; used by the spec-anchor tests when CAMB >= 2 "
               "(the spec anchors were produced with CAMB 1.x); tolerances are the same as for the spec anchors",
       "values": v}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
