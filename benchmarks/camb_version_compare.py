"""Evaluate fixed KV450 and KiDS-Legacy points under the installed CAMB / numpy, for comparing versions.

Run once per environment, then compare the two npz files (docs/COMPATIBILITY.md):

    python benchmarks/camb_version_compare.py runs/compat/points_camb<ver>_numpy<ver>.npz .

KV450: the test, scan and start presets, the MAP (runs/kv450_final/map_best.json) and 12 samples of the production
chain (prod1/prod_c0.npz, rng seed 1), each with exact CAMB, the transfer emulator and the CAMB 'mead' non-linear
backend: ln L and sigma8. KiDS-Legacy: the 10 highest-posterior rows of the official COSEBIs chain (row 0 = the
best-posterior row used for step D): chi2 of E_n 1-6 and sigma8.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

out, repo = Path(sys.argv[1]), Path(sys.argv[2])
import camb
from kidspipe import api
from kidspipe.params import PARAM_NAMES

res = {"camb_version": camb.__version__, "numpy_version": np.__version__}
t0 = time.time()

# ---------------- KV450 points
mb = json.load(open(repo / "runs/kv450_final/map_best.json"))
th_map = np.array([mb["theta"][n] for n in PARAM_NAMES]) if isinstance(mb["theta"], dict) else np.asarray(mb["theta"], float)
z = np.load(repo / "runs/kv450_final/prod1/prod_c0.npz")
S = z["samples"]
rng = np.random.default_rng(1)
pts = [api.preset("test"), api.preset("scan"), api.preset("start"), th_map] + [S[i, :14] for i in rng.choice(len(S), 12, replace=False)]
P = np.array(pts)
res["kv_points"] = P
for name, kw in [("exact", dict(linear_backend="camb")), ("emulator", dict(linear_backend="emulator")),
                 ("mead", dict(pnl_backend="camb"))]:
    post = api.build_posterior(api.build_model(**kw), api.build_likelihood())
    ll, s8 = [], []
    for th in P:
        lp, lnl, der = api.log_post_with_derived(post, th)
        ll.append(lnl); s8.append(der["sigma8"])
    res[f"kv_lnL_{name}"] = np.array(ll)
    res[f"kv_sigma8_{name}"] = np.array(s8)
    print(name, "done", round(time.time() - t0), "s", flush=True)

# ---------------- KiDS-Legacy COSEBIs (official chain rows)
from kidspipe.legacy import LegacyCosebisLikelihood, LegacyCosebisModel, chain_samples, load_chain

chain = load_chain()
post = np.where(np.isfinite(chain[:, 43]), chain[:, 43], -np.inf)
rows = list(np.argsort(post)[::-1][:10])                      # rows[0] = best-posterior row (step D)
samples = chain_samples(chain, rows)
model = LegacyCosebisModel(nthreads=4, cache_dir=str(repo / "runs/legacy/cache"), cache_size=1)
like = LegacyCosebisLikelihood()
c2, s8 = [], []
for s_ in samples:
    mu = model.mu(s_)
    c2.append(like.chi2(mu)); s8.append(model.last_derived.get("sigma_8", np.nan))
res["legacy_rows"] = np.array(rows)
res["legacy_chi2"] = np.array(c2)
res["legacy_sigma8"] = np.array(s8)
res["legacy_chi2_chain"] = np.array([s_["chi2_off"] for s_ in samples])
res["legacy_sigma8_chain"] = np.array([s_["sigma_8_off"] for s_ in samples])
print("legacy done", round(time.time() - t0), "s", flush=True)
np.savez(out, **res)
print("saved", out)
