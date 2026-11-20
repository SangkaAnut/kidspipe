"""Convergence diagnostics + marginal summaries of production chains.   usage: run_kv450_analyze.py TAG [--burn-frac 0.2] [--thin 1]
Reads runs/kv450_final/TAG/prod_c*.npz (samples, log_post, derived = sigma8, omega_m, S8), writes TAG/analysis.json and prints tables."""
import argparse
import glob
import json
import numpy as np
import run_kv450_common as C
from run_kv450_common import PARAM_NAMES
from kidspipe.inference.mcmc import split_rhat, effective_sample_size, integrated_autocorr_time
from kidspipe.inference.chains import marginal_stats, pjhpd_extents

NAMES = list(PARAM_NAMES) + ["sigma8", "omega_m", "S8"]


def load(tag, burn_frac=0.2, thin=1):
    files = [f for t in tag.split(",") for f in sorted(glob.glob(str(C.OUT / t / "prod_c*.npz")))]
    chains, lps = [], []
    for f in files:
        z = np.load(f)
        X = np.hstack([z["samples"], z["derived"]])
        lp = z["log_post"]
        chains.append((X, lp))
    n = min(len(c[0]) for c in chains)
    b = int(burn_frac * n)
    arr = np.stack([c[0][b:n:thin] for c in chains], axis=1)         # (steps, chains, 17): equal-length part, for R-hat / tau
    lp = np.stack([c[1][b:n:thin] for c in chains], axis=1)
    full = np.concatenate([c[0][int(burn_frac * len(c[0]))::thin] for c in chains])       # all post-burn samples (unequal lengths)
    full_lp = np.concatenate([c[1][int(burn_frac * len(c[1]))::thin] for c in chains])
    return arr, lp, n, full, full_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--burn-frac", type=float, default=0.2)
    ap.add_argument("--thin", type=int, default=1)
    a = ap.parse_args()
    arr, lp, n_full, full, full_lp = load(a.tag, a.burn_frac, a.thin)
    ns, nc, nd = arr.shape
    rh = split_rhat(arr)
    rhr = split_rhat(arr, rank_normalize=True)
    tau = integrated_autocorr_time(arr)
    flat, flp = full, full_lp
    ess = len(flat) / tau          # tau from the equal-length part, applied to all post-burn samples
    st = marginal_stats(flat, names=NAMES, log_post=flp)
    box = C.in_emu_box(flat[:, :14])
    # mass outside the emulator box, by parameter
    out_info = {n: float(np.mean((flat[:, C.IDX[n]] < lo) | (flat[:, C.IDX[n]] > hi))) for n, (lo, hi) in C.EMU_BOX.items()}
    res = dict(tag=a.tag, steps_per_chain=ns, chains=nc, burn_frac=a.burn_frac, total_samples=len(flat),
               rhat=dict(zip(NAMES, rh)), rhat_rank=dict(zip(NAMES, rhr)), tau=dict(zip(NAMES, tau)), ess=dict(zip(NAMES, ess)),
               stats=st, mass_outside_box=float(1 - box.mean()), mass_outside_by_param=out_info,
               min_in_box=dict(zip(C.EMU_BOX, [float(flat[:, C.IDX[n]].min()) for n in C.EMU_BOX])),
               max_in_box=dict(zip(C.EMU_BOX, [float(flat[:, C.IDX[n]].max()) for n in C.EMU_BOX])))
    od = C.OUT / (a.tag if "," not in a.tag else "combined_" + a.tag.replace(",", "_"))
    od.mkdir(exist_ok=True)
    C.jdump(od / "analysis.json", res)
    print(f"{a.tag}: {len(flat)} post-burn samples; diagnostics on {ns} steps x {nc} chains (burn frac {a.burn_frac}, full length {n_full})   mass outside emulator box: {1-box.mean():.2e}")
    print(f"{'param':12s} {'mean':>10s} {'std':>9s} {'median':>10s} {'68% lo':>10s} {'68% hi':>10s} {'PJ lo':>10s} {'PJ hi':>10s} {'Rhat':>6s} {'Rhat_r':>6s} {'tau':>7s} {'ESS':>7s}")
    for i, nm in enumerate(NAMES):
        s = st[nm]
        print(f"{nm:12s} {s['mean']:10.4g} {s['std']:9.3g} {s['median']:10.4g} {s['lo']:10.4g} {s['hi']:10.4g} {s['pj_lo']:10.4g} {s['pj_hi']:10.4g} "
              f"{rh[i]:6.3f} {rhr[i]:6.3f} {tau[i]:7.1f} {ess[i]:7.0f}")
    print("chain means of S8:", np.round(arr[:, :, -1].mean(0), 4))


if __name__ == "__main__":
    main()
