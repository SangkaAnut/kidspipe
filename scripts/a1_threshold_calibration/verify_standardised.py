# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... 
# One-off A1b threshold-calibration study, not part of the Colab chain; see scripts/README.md.
"""Independent verification of the standardised A1b-noise criterion (i).  Modes (argv[1]):
  gen      alt noise generators (AR(2), MA2+tail GP, block bootstrap): size at shipped thresholds + real pte  -> gen.json
  circ     paired injection contamination of fitted lag-1/2 and of the plain-fit MA(2) theta               -> circ.json
  power    re-run 3 power entries with the SHIPPED fixed table (own injections)                            -> power.json
  e2e      end-to-end: refit null from each simulated/injected realisation, build its own table+threshold -> e2e_<cond>_<seed>.json
"""
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[2]      # repository root
import sys, json, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(_R) + "/scripts/a1_threshold_calibration")
from kidspipe.measure import a1check as A
from kidspipe.measure.official import PAIRS, load_official_xi
ROOT = _R
OUT = ROOT / "runs/dr5_measure/a1_salvage/threshold_calibration/standardised/verify"
SH = ROOT / "runs/dr5_measure/a1_salvage/threshold_calibration/standardised"
TH = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
LAB = [f"xi+ {i+1}-{j+1}" for i, j in PAIRS] + [f"xi- {i+1}-{j+1}" for i, j in PAIRS]


def load():
    asc = load_official_xi()
    rp = np.array([np.load(TH / f"xipm_bin{i+1}_bin{j+1}.npz")["xip_raw"] for i, j in PAIRS])
    rm = np.array([np.load(TH / f"xipm_bin{i+1}_bin{j+1}.npz")["xim_raw"] for i, j in PAIRS])
    return asc, rp, rm


asc, RP, RM = load()
THETA = asc["r_nom"][0]
Z0, As0 = A.curves_z(RP, RM, asc)
RHO0, _ = A.jitter_rho(Z0, "plain")
PROF = A.running_rms(Z0 - Z0.mean(-1, keepdims=True), 51)
MA = A.ma2_from_rho(RHO0[:, 0], RHO0[:, 1])
NULL = A.load_null(SH / "null_table_hp15.npz")
SD = np.array([1.0])


def build(zz, Ap=None, Am=None):
    Ap = As0["xip"] if Ap is None else Ap
    Am = As0["xim"] if Am is None else Am
    return (Ap[:, None] * asc["xip"] + asc["sigma_xip"] * zz[:, :21],
            Am[:, None] * asc["xim"] + asc["sigma_xim"] * zz[:, 21:])


def gate_stats(tp, tm, null=NULL):
    z, _ = A.curves_z(tp, tm, asc)
    st = A.standardised_stats(z, null["method"])
    p = A.null_pvalues(st["D"], null["sortedD"])
    c = A.combine_pvalues(p)
    c["const"] = st["ratio"].max(-1)
    return c, st


# ---------- independent generators (all return pre-fit z, shape (n, 42, 1000))
def acf_from_ma(th):
    den = 1 + th[:, 0] ** 2 + th[:, 1] ** 2
    return np.stack([(th[:, 0] + th[:, 0] * th[:, 1]) / den, th[:, 1] / den], 1)


def gp_from_acf(acf, n, rng, prof=PROF):
    """Gaussian process with given acf (C, L) at lags 1.. via circulant embedding (independent of MA recursion)."""
    C, L = acf.shape; N = prof.shape[1]; M = 4096
    r = np.zeros((C, M)); r[:, 0] = 1
    for k in range(L):
        r[:, k + 1] = acf[:, k]; r[:, M - k - 1] = acf[:, k]
    f = np.fft.rfft(r, axis=-1).real
    f = np.clip(f, 1e-6, None)
    e = rng.standard_normal((n, C, M))
    x = np.fft.irfft(np.fft.rfft(e, axis=-1) * np.sqrt(f)[None], n=M, axis=-1)[..., :N]
    return x * prof[None]


def gen_ar2(n, rng):
    r = acf_from_ma(MA); r1, r2 = r[:, 0], r[:, 1]
    # Yule-Walker: phi1 = r1(1-r2)/(1-r1^2), phi2 = (r2-r1^2)/(1-r1^2)
    p1 = r1 * (1 - r2) / (1 - r1 ** 2); p2 = (r2 - r1 ** 2) / (1 - r1 ** 2)
    N = PROF.shape[1]; B = 200
    e = rng.standard_normal((n, 42, N + B))
    x = np.zeros_like(e)
    for t in range(2, N + B):
        x[..., t] = e[..., t] + p1[None] * x[..., t - 1] + p2[None] * x[..., t - 2]
    x = x[..., B:]
    x = x / x.std(-1, keepdims=True)  # unit variance per curve
    return x * PROF[None]


TAIL = np.load(ROOT / "runs/dr5_measure/a1_salvage/threshold_calibration/z.npz")["R"].mean(0)[2:12]


def gen_tail(n, rng):
    acf = np.zeros((42, 12)); acf[:, :2] = acf_from_ma(MA); acf[:, 2:12] = TAIL[None]
    return gp_from_acf(acf, n, rng)


def gen_boot(n, rng, blk=50, shift=100):
    """Local block bootstrap of the real standardised z (z/prof): block at position s is replaced by a block of the SAME
    curve starting within +-shift bins (so the position-dependent lag-1 is preserved), then multiplied by prof."""
    x = (Z0 - Z0.mean(-1, keepdims=True)) / PROF
    N = x.shape[1]; nb = N // blk
    out = np.empty((n, 42, N))
    starts = np.arange(nb) * blk
    for c in range(42):
        src = np.clip(starts[None] + rng.integers(-shift, shift + 1, size=(n, nb)), 0, N - blk)
        idx = src[..., None] + np.arange(blk)
        out[:, c, :nb * blk] = x[c][idx].reshape(n, -1)
        if nb * blk < N: out[:, c, nb * blk:] = x[c][-(N - nb * blk):]
    return out * PROF[None]


def gen_ma(n, rng):
    return A.simulate_null_z(MA, PROF, n, rng)


GENS = {"ma2": gen_ma, "ar2": gen_ar2, "ma2+tail": gen_tail, "blockboot": gen_boot}


def sims(gen, n, rng, chunk=200, null=NULL):
    out = {k: [] for k in ("minp", "fisher", "stouffer", "const")}; Ds = []
    for i in range(0, n, chunk):
        zz = GENS[gen](min(chunk, n - i), rng)
        c, st = gate_stats(*build(zz), null)
        for k in out: out[k].append(c[k])
        Ds.append(st["D"])
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(Ds)


def mode_gen(n=4000, only=None, tag=''):
    res = {}
    real, rst = gate_stats(RP[None], RM[None])
    res["real_shipped"] = {k: float(v[0]) for k, v in real.items()}
    thr = NULL["thr"]; res["shipped_thr"] = thr
    for g in (only or GENS):
        t = time.time(); rng = np.random.default_rng(1000 + list(GENS).index(g))
        v, D = sims(g, n, rng)
        r = {"false_fail_at_shipped_1pct": {k: float((v[k] > thr[k]).mean()) for k in thr},
             "q99": {k: float(np.percentile(v[k], 99)) for k in thr},
             "real_pte_shipped_table": {k: float((v[k] >= real[k][0]).mean()) for k in thr},
             "median_D_mean_over_curves_minus_shipped_table": float(np.mean(np.median(D, 0) - np.median(NULL["sortedD"], 0)))}
        r["ff_mc_se"] = float(np.sqrt(0.01 * 0.99 / n))
        # own table from this generator (independent rebuild), n=6000, thresholds from n sims at 1%
        v6, D6 = sims(g, 6000, np.random.default_rng(2000 + list(GENS).index(g)))
        own = {"sortedD": np.sort(D6, 0), "method": "hp15", "size": 0.01}
        pr = A.combine_pvalues(A.null_pvalues(rst["D"], own["sortedD"]))
        pn = A.combine_pvalues(A.null_pvalues(D, own["sortedD"]))
        r["own_table"] = {"real": {k: float(pr[k][0]) for k in thr},
                          "thr99": {k: float(np.percentile(pn[k], 99)) for k in thr},
                          "real_pte": {k: float((pn[k] >= pr[k][0]).mean()) for k in thr}}
        res[g] = r; print(g, json.dumps(r["false_fail_at_shipped_1pct"]), json.dumps(r["own_table"]["real_pte"]), round(time.time() - t), flush=True)
    json.dump(res, open(OUT / f"gen{tag}.json", "w"), indent=1)


# ---------- injections (own implementation)
ltf = np.log(THETA / np.sqrt(600)); ltf = ltf / np.abs(ltf).max()
bumpf = np.exp(-0.5 * ((np.log(THETA) - np.log(20)) / 0.3) ** 2)


def inj(kind, amp, tp, tm):
    fp = np.ones((21, 1000)); fm = np.ones((21, 1000))
    if kind == "tilt": fp = fp + amp * ltf; fm = fm + amp * ltf
    elif kind == "bump": fp = fp + amp * bumpf; fm = fm + amp * bumpf
    elif kind == "tilt44":
        p = PAIRS.index((3, 3)); fp[p] += amp * ltf; fm[p] += amp * ltf
    elif kind == "null": pass
    else: raise KeyError(kind)
    return tp * fp, tm * fm


def mode_power(n=600):
    res = {}; thr = NULL["thr"]
    for kind, amp in [("null", 0), ("tilt", 0.05), ("tilt", 0.10), ("tilt", 0.20), ("tilt44", 0.20), ("bump", 0.15)]:
        rng = np.random.default_rng(31)
        v = {k: [] for k in ("minp", "fisher", "stouffer", "const")}
        for i in range(0, n, 200):
            tp, tm = build(gen_ma(200, rng)); tp, tm = inj(kind, amp, tp, tm)
            c, _ = gate_stats(tp, tm)
            for k in v: v[k].append(c[k])
        v = {k: np.concatenate(x) for k, x in v.items()}
        res[f"{kind} {amp}"] = {**{k: float((v[k] > thr[k]).mean()) for k in thr}, "const1.0": float((v["const"] > 1.0).mean()),
                                "const1.03": float((v["const"] > 1.0315).mean())}
        print(kind, amp, res[f"{kind} {amp}"], flush=True)
    json.dump(res, open(OUT / "power.json", "w"), indent=1)


def mode_circ(n=300):
    res = {}
    for kind, amp in [("tilt", 0.1), ("tilt", 0.2), ("bump", 0.2)]:
        rng = np.random.default_rng(55)
        d = {"plain": [], "hp15": [], "th0": [], "th1": [], "ratio": [], "rp15": []}
        for i in range(0, n, 100):
            tp, tm = build(gen_ma(100, rng)); tpi, tmi = inj(kind, amp, tp, tm)
            z0, _ = A.curves_z(tp, tm, asc); z1, _ = A.curves_z(tpi, tmi, asc)
            for m in ("plain", "hp15"):
                d[m].append((A.jitter_rho(z1, m)[0] - A.jitter_rho(z0, m)[0]))
            r0 = A.jitter_rho(z0, "plain")[0]; r1 = A.jitter_rho(z1, "plain")[0]
            d["ratio"].append(A.boxcar_ratio(z1) / A.boxcar_ratio(z0))
            s0 = A.standardised_stats(z0, "hp15"); s1 = A.standardised_stats(z1, "hp15")
            d["rp15"].append(s1["r_pred"] / s0["r_pred"])
        res[f"{kind} {amp}"] = {"d_rho_plain_lag1": float(np.concatenate(d["plain"])[..., 0].mean()), "d_rho_plain_lag2": float(np.concatenate(d["plain"])[..., 1].mean()),
                                "d_rho_hp15_lag1": float(np.concatenate(d["hp15"])[..., 0].mean()), "d_rho_hp15_lag2": float(np.concatenate(d["hp15"])[..., 1].mean()),
                                "ratio_inflation": float(np.concatenate(d["ratio"]).mean()), "rpred_hp15_inflation": float(np.concatenate(d["rp15"]).mean())}
        print(kind, amp, res[f"{kind} {amp}"], flush=True)
    json.dump(res, open(OUT / "circ.json", "w"), indent=1)


def e2e_one(args):
    """One realisation: draw/inject (or use the REAL data if cond == 'real'), refit the null from THIS data exactly as
    a1b_noise_standardised does (calibrate_standardised: plain lag-1/2 -> MA(2), rms profile), build its own table, return stats."""
    cond, seed, gen, ntab, nthr, size = args
    import warnings; warnings.simplefilter("ignore")
    rng = np.random.default_rng(seed)
    if cond == "real":
        tp, tm = RP[None], RM[None]
    else:
        kind, amp = {"null": ("null", 0), "tilt10": ("tilt", .1), "tilt20": ("tilt", .2), "tilt44_20": ("tilt44", .2), "bump15": ("bump", .15)}[cond]
        tp, tm = build(GENS[gen](1, rng)); tp, tm = inj(kind, amp, tp, tm)
    null = A.calibrate_standardised(tp[0], tm[0], asc, "hp15", n_tab=ntab, n_thr=nthr, size=size, seed=seed + 7, chunk=250)
    c, st = gate_stats(tp, tm, null)
    return {k: float(c[k][0]) for k in ("minp", "fisher", "stouffer", "const")}, {k: float(v) for k, v in null["thr"].items()}


def mode_e2e(cond, gen, nrep, nproc, ntab=2000, nthr=20, size=0.05):
    t = time.time()
    with Pool(nproc) as p:
        r = p.map(e2e_one, [(cond, 10000 + i, gen, ntab, nthr, size) for i in range(nrep)], chunksize=1)
    out = {"cond": cond, "gen": gen, "nrep": nrep, "ntab": ntab, "stats": {k: [a[0][k] for a in r] for k in r[0][0]}, "seconds": time.time() - t}
    json.dump(out, open(OUT / f"e2ev2_{cond}_{gen}.json", "w"))
    print(cond, "done", out["seconds"], flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "gen": mode_gen()
    elif m == "genbb": mode_gen(only=["blockboot"], tag="_bb")
    elif m == "circ": mode_circ()
    elif m == "power": mode_power()
    elif m == "e2e": mode_e2e(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif m == "time":
        t = time.time(); print(e2e_one(("null", 1, "ma2", 2000, 1000, 0.05)), time.time() - t)
