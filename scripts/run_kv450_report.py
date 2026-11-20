"""Collect the numbers of runs/kv450_final/* into markdown tables (printed; used to write docs/RESULTS.md)."""
import glob
import json
import numpy as np
import run_kv450_common as C
from run_kv450_common import PARAM_NAMES, params
from kidspipe.inference.chains import marginal_stats, hpd_interval_1d, weighted_quantile

NAMES = list(PARAM_NAMES) + ["sigma8", "omega_m", "S8"]
A = json.load(open(C.OUT / "combined_prod1_prod2/analysis.json"))
PUB = {"S8": (0.737, 0.040, 0.036), "omega_m": (0.256, None, None), "A_IA": (0.98, 0.69, 0.69), "A_baryon": (2.48, None, None)}

print("## MCMC marginals (prod1+prod2)\n")
print("| param | mean | std | median | 68% lo | 68% hi | 1-D HPD 68% | PJ-HPD 68% | R-hat | ESS |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n in NAMES:
    s = A["stats"][n]
    print(f"| {n} | {s['mean']:.4g} | {s['std']:.3g} | {s['median']:.4g} | {s['lo']:.4g} | {s['hi']:.4g} | [{s['hpd_lo']:.4g}, {s['hpd_hi']:.4g}] | [{s['pj_lo']:.4g}, {s['pj_hi']:.4g}] | {A['rhat'][n]:.3f} | {A['ess'][n]:.0f} |")

# chi2 / lnL statistics from the chains
files = sorted(glob.glob(str(C.OUT / "prod1/prod_c*.npz"))) + sorted(glob.glob(str(C.OUT / "prod2/prod_c*.npz")))
X, LP, D = [], [], []
for f in files:
    z = np.load(f)
    b = int(0.05 * len(z["samples"]))
    X.append(z["samples"][b:]); LP.append(z["log_post"][b:]); D.append(z["derived"][b:])
X, LP, D = map(np.concatenate, (X, LP, D))
lnp = params.log_prior(X)
chi2 = -2 * (LP - lnp)
print(f"\nchi2 over posterior samples: min {chi2.min():.2f}, mean {chi2.mean():.2f}, median {np.median(chi2):.2f}, sd {chi2.std():.2f}; "
      f"mean - min = {chi2.mean()-chi2.min():.2f}; (mean chi2 - MAP chi2 181.874) = {chi2.mean()-181.874:.2f}")
s8 = D[:, 2]
i = np.argmax(LP)
print("max log_post sample:", dict(zip(PARAM_NAMES, np.round(X[i], 4))), "chi2", chi2[i])
# fractions of posterior near prior edges
for n, lo, hi in (("h", 0.64, 0.82), ("ombh2", 0.019, 0.026), ("ln_1e10_A_s", 1.7, 5.0), ("n_s", 0.7, 1.3), ("A_baryon", 2.0, 3.13), ("omch2", 0.01, 0.99)):
    x = X[:, params.PARAM_NAMES.index(n)]
    w = hi - lo
    print(f"{n}: frac within 5% of range of lower edge {np.mean(x < lo + 0.05*w):.3f}, of upper edge {np.mean(x > hi - 0.05*w):.3f}; "
          f"quantiles 2.5/16/50/84/97.5 = {np.round(np.percentile(x,[2.5,16,50,84,97.5]),4)}")
print("S8 quantiles 2.5/16/50/84/97.5:", np.round(np.percentile(s8, [2.5, 16, 50, 84, 97.5]), 4))
print("S8 1-D HPD 68%:", np.round(hpd_interval_1d(s8, None, 0.68), 4), " 95%:", np.round(hpd_interval_1d(s8, None, 0.95), 4))
print("S8 mode (hist):", end=" ")
h, e = np.histogram(s8, bins=80); print(round(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]), 4))
print("corr(S8, others):", {n: round(float(np.corrcoef(s8, X[:, j])[0, 1]), 2) for j, n in enumerate(PARAM_NAMES)})
print("corr(sigma8,omega_m)=", round(float(np.corrcoef(D[:, 0], D[:, 1])[0, 1]), 3))

# nested
print("\n## nautilus")
for t in ("nautilus1", "nautilus2", "nautilus3"):
    s = json.load(open(C.OUT / t / "nested_summary.json"))
    z = np.load(C.OUT / t / "nested.npz")
    w = z["weights"]; Xn = np.hstack([z["samples"], z["derived"]])
    m = w @ Xn
    sd = np.sqrt(w @ (Xn - m) ** 2)
    q = [weighted_quantile(Xn[:, 16], w, p) for p in (0.1587, 0.5, 0.8413)]
    print(t, f"z_grid={s['z_grid']} logZ_box={s['log_z_box']:.3f} logZ_full_est={s['log_z_full_estimate']:.3f} n_eval={s['n_eval']} ESS={s['ess']:.0f} wall={s['wall']:.0f}s S8={m[16]:.4f}+-{sd[16]:.4f} q={np.round(q,4)}")
    if t == "nautilus3" or t == "nautilus2":
        mm = np.array([A["stats"][n]["mean"] for n in NAMES]); ss = np.array([A["stats"][n]["std"] for n in NAMES])
        print("   (nested mean - MCMC mean)/MCMC std:", {n: round(float((m[k] - mm[k]) / ss[k]), 3) for k, n in enumerate(NAMES)})
        print("   nested sd / MCMC sd:", {n: round(float(sd[k] / ss[k]), 3) for k, n in enumerate(NAMES)})
