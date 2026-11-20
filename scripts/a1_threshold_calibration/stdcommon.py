# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... 
# One-off A1b threshold-calibration study, not part of the Colab chain; see scripts/README.md.
"""Shared engine for the standardised A1b-noise study (data loading, injections, null sims, statistics)."""
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[2]      # repository root
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from kidspipe.measure.official import PAIRS, load_official_xi
from kidspipe.measure import a1check as A

ROOT = _R
OUT = ROOT / "runs/dr5_measure/a1_salvage/threshold_calibration/standardised"
OUT.mkdir(parents=True, exist_ok=True)
TH = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
LAB = [f"xi+ {i+1}-{j+1}" for i, j in PAIRS] + [f"xi- {i+1}-{j+1}" for i, j in PAIRS]


def load_real():
    asc = load_official_xi()
    rp = np.array([np.load(TH / f"xipm_bin{i+1}_bin{j+1}.npz")["xip_raw"] for i, j in PAIRS])
    rm = np.array([np.load(TH / f"xipm_bin{i+1}_bin{j+1}.npz")["xim_raw"] for i, j in PAIRS])
    amp = json.load(open(ROOT / "runs/dr5_measure/a1_salvage/a1_salvage.json"))["step2_amplitude"]
    sd = {k: np.array(amp[f"{k}_raw"]["A_pair_err_frac"]) for k in ("xip", "xim")}
    return asc, rp, rm, sd


def z_of(tp, tm, asc):
    """tp, tm: (..., 21, 1000) -> z (..., 42, 1000)."""
    w = asc["npairs_weighted"]
    out = []
    for key, tr in (("xip", tp), ("xim", tm)):
        Ap = A.ratio_of_sums(tr, asc[key], w)
        out.append((tr - Ap[..., None] * asc[key]) / asc["sigma_" + key])
    return np.concatenate(out, -2), None


def pair_index(i, j):
    return PAIRS.index((i, j))


# ---------------- injections: return multiplicative factor arrays (21,1000) for xi+ and xi- (or callables)
def families(th):
    ltf = np.log(th / np.sqrt(2 * 300)); ltf = ltf / np.abs(ltf).max()
    one = np.ones((21, th.size))

    def mk(shape, which="both", pairs=None):
        def f(amp):
            fp, fm = one.copy(), one.copy()
            sel = slice(None) if pairs is None else list(pairs)
            if which in ("both", "p"): fp[sel] = 1 + amp * shape
            if which in ("both", "m"): fm[sel] = 1 + amp * shape
            return fp, fm
        return f
    bump = lambda a: np.exp(-0.5 * ((np.log(th) - np.log(20)) / 0.3) ** 2)
    step = (th < 5).astype(float)
    p11 = PAIRS.index((0, 0)); p44 = PAIRS.index((3, 3))
    return {
        "tilt": (mk(ltf), [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]),
        "bump at 20'": (mk(bump(0)), [0.05, 0.1, 0.2, 0.3]),
        "step <5'": (mk(step), [0.05, 0.1, 0.2]),
        "tilt xi- 1-1 only": (mk(ltf, "m", [p11]), [0.05, 0.1, 0.2, 0.3]),
        "tilt pair 4-4 only (xi+,xi-)": (mk(ltf, "both", [p44]), [0.05, 0.1, 0.2, 0.3]),
        "tilt xi- only (all pairs)": (mk(ltf, "m"), [0.05, 0.1, 0.15, 0.2]),
    }


def apply_inject(fam, amp, tp, tm, asc, rng):
    """fam: name; returns (tp', tm')."""
    if fam == "white noise":
        return (tp + amp * asc["sigma_xip"] * rng.standard_normal(tp.shape),
                tm + amp * asc["sigma_xim"] * rng.standard_normal(tm.shape))
    if fam == "pair scale scatter":
        f = np.exp(amp * rng.standard_normal(tp.shape[:-1]))[..., None]
        return tp * f, tm * f
    raise KeyError(fam)


def inject(famdict, name, amp, tp, tm, asc, rng):
    if name in famdict:
        fp, fm = famdict[name][0](amp)
        return tp * fp, tm * fm
    return apply_inject(name, amp, tp, tm, asc, rng)


EXTRA = {"white noise": [0.05, 0.1, 0.2], "pair scale scatter": [0.01, 0.02, 0.05, 0.1]}


METHODS = ["plain", "hp15", "hp31"]


class Study:
    """Real data + the null model (fitted MA(2) jitter, rms profile, cross-curve correlation)."""

    def __init__(self, correlated=None, prof_win=51):
        self.asc, self.rp, self.rm, self.sd = load_real()
        self.th = self.asc["r_nom"][0]
        self.w = self.asc["npairs_weighted"]
        self.z, _ = z_of(self.rp, self.rm, self.asc)
        self.Ap = A.ratio_of_sums(self.rp, self.asc["xip"], self.w)
        self.Am = A.ratio_of_sums(self.rm, self.asc["xim"], self.w)
        self.rho, _ = A.jitter_rho(self.z, "plain")
        self.prof = A.running_rms(self.z - self.z.mean(-1, keepdims=True), prof_win)
        self.theta = A.ma2_from_rho(self.rho[:, 0], self.rho[:, 1])
        x = self.z / self.prof
        self.corr = np.corrcoef(x)
        self.chol = None
        if correlated is not None:
            self.set_corr(correlated)
        self.fam = families(self.th)
        self.fam_all = dict(self.fam)
        self.fam_amps = {k: v[1] for k, v in self.fam.items()}
        self.fam_amps.update(EXTRA)

    def set_corr(self, C):
        ev, V = np.linalg.eigh(C)
        ev = np.clip(ev, 1e-6, None)
        C2 = (V * ev) @ V.T
        d = np.sqrt(np.diag(C2)); C2 = C2 / np.outer(d, d)
        self.chol = np.linalg.cholesky(C2)

    def sim_their(self, n, rng, theta=None, prof=None):
        """Pre-fit null: their = A_p asc + sigma * z_jitter  (A_p refit happens in the gate)."""
        zz = A.simulate_null_z(self.theta if theta is None else theta, self.prof if prof is None else prof, n, rng, self.chol)
        tp = self.Ap[:, None] * self.asc["xip"] + self.asc["sigma_xip"] * zz[:, :21]
        tm = self.Am[:, None] * self.asc["xim"] + self.asc["sigma_xim"] * zz[:, 21:]
        return tp, tm

    def stats(self, tp, tm, methods=METHODS):
        z, _ = z_of(tp, tm, self.asc)
        out = {"ratio": A.boxcar_ratio(z).astype(np.float32)}
        for m in methods:
            st = A.standardised_stats(z, m)
            out[m] = {"r_pred": st["r_pred"].astype(np.float32), "rho": st["rho"].astype(np.float32)}
        lA = {}
        for key, tr in (("xip", tp), ("xim", tm)):
            Ak = A.ratio_of_sums(tr, self.asc[key], self.w)
            lA[key] = np.log(np.abs(Ak))
        out["lnA_p"], out["lnA_m"] = lA["xip"].astype(np.float32), lA["xim"].astype(np.float32)
        return out

    def crit2_pte(self, st):
        from scipy.stats import chi2
        ptes = []
        for k, key in (("lnA_p", "xip"), ("lnA_m", "xim")):
            # sims use the real per-pair A_p as truth; remove their real scatter (criterion (ii) assumes one constant)
            Areal = self.Ap if key == "xip" else self.Am
            lcr = np.sum(np.log(np.abs(Areal)) / self.sd[key] ** 2) / np.sum(1 / self.sd[key] ** 2)
            lA = st[k] - np.log(np.abs(Areal)) + lcr; sdv = self.sd[key]; wg = 1 / sdv ** 2
            lc = (lA * wg).sum(-1, keepdims=True) / wg.sum()
            ch = (((lA - lc) / sdv) ** 2).sum(-1)
            ptes.append(chi2.sf(ch, 20))
        return np.minimum(*ptes)


class Tables:
    """Per-curve null tables from a training set of null stats: ECDF/mean/sd of D = ratio^2 - r_pred^2."""

    def __init__(self, st):
        self.n = st["ratio"].shape[0]
        self.D, self.sortedD, self.mu, self.sdv = {}, {}, {}, {}
        for m in METHODS:
            D = st["ratio"].astype(float) ** 2 - st[m]["r_pred"].astype(float) ** 2
            self.sortedD[m] = np.sort(D, 0); self.mu[m] = D.mean(0); self.sdv[m] = D.std(0)

    def pvals(self, m, D):
        out = np.empty(D.shape)
        for c in range(D.shape[-1]):
            out[..., c] = (1 + self.n - np.searchsorted(self.sortedD[m][:, c], D[..., c], "left")) / (self.n + 1)
        return out


def all_statistics(st, tab):
    """Map null/injected stats -> dict of scalar test statistics per realisation (larger = worse)."""
    out = {"const": st["ratio"].max(-1).astype(float)}
    for m in METHODS:
        r = st["ratio"].astype(float); rp = st[m]["r_pred"].astype(float)
        D = r ** 2 - rp ** 2
        p = tab.pvals(m, D)
        out[f"S|{m}"] = (r / rp).max(-1)
        out[f"D|{m}"] = D.max(-1)
        out[f"Dz|{m}"] = ((D - tab.mu[m]) / tab.sdv[m]).max(-1)
        out[f"minp|{m}"] = -np.log10(p.min(-1))
        out[f"fisher|{m}"] = -2 * np.log(p).sum(-1)
        out[f"stouffer|{m}"] = norm.isf(np.clip(p, 1e-12, 1 - 1e-12)).sum(-1) / np.sqrt(p.shape[-1])
    return out
