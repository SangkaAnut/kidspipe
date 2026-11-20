"""chi2 decomposition at the best-fit point (and at the posterior-mean S8 point of the MAP valley):
xi+ / xi- (diagonal-block-sum of r_i (C^-1 r)_i, which sums exactly to chi2), per tomographic pair, and the Gaussian-prior 'chi2' of
the nuisance parameters.   usage: run_kv450_chi2split.py [map_best.json]   -> runs/kv450_final/chi2_split.json"""
import json
import sys
import numpy as np
from scipy.linalg import cho_solve
import run_kv450_common as C
from run_kv450_common import api, params, PARAM_NAMES, IDX
from kidspipe.likelihood.data import PLUS_MASK, PAIRS, N_PER_PAIR

f = sys.argv[1] if len(sys.argv) > 1 else "map_best.json"
th = np.array(json.load(open(C.find(f)))["theta"])
post = C.make_post(linear="camb", threads=2)      # exact CAMB here (single point)
lik = post.likelihood
mu = post.model.mu(th)
r = lik.residual(mu, th[IDX["A_c"]], th[IDX["delta_c"]])
w = np.linalg.solve(lik.cov, r)
contrib = r * w
out = dict(chi2=float(contrib.sum()), chi2_check=float(-2 * post.log_like(th)), n_data=195,
           chi2_xi_plus=float(contrib[PLUS_MASK].sum()), n_xi_plus=int(PLUS_MASK.sum()),
           chi2_xi_minus=float(contrib[~PLUS_MASK].sum()), n_xi_minus=int((~PLUS_MASK).sum()))
pairs = contrib.reshape(15, N_PER_PAIR).sum(1)
out["per_pair"] = {f"{i+1}x{j+1}": float(pairs[p]) for p, (j, i) in enumerate(PAIRS)}
out["per_pair_n"] = N_PER_PAIR
out["null_chi2"] = float(lik.null_chi2())
gp = params.GAUSSIAN_PRIORS
out["prior_chi2_nuisance"] = {n: float(((th[IDX[n]] - m) / s) ** 2) for n, (m, s) in gp.items()}
out["prior_chi2_total"] = float(sum(out["prior_chi2_nuisance"].values()))
out["chi2_plus_prior"] = out["chi2"] + out["prior_chi2_total"]
out["dof"] = 195 - 14
C.jdump(C.OUT / "chi2_split.json", out)
print(json.dumps(out, indent=1))
