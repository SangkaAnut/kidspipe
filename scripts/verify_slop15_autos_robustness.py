# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Independent re-computation of the slop1.5 auto-pair eval (robustness check). Read-only; prints numbers."""
import json, re, sys, numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths
R = Path(__file__).resolve().parents[1] / "runs/dr5_measure"
OFF = paths.official_dir()
ev = json.load(open(R/"xi_unordered/b1.5/eval.json"))
M = [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045]
def boxr(z):
    zb = z[:1000//20*20].reshape(-1, 20).mean(1); return zb.std()/(z.std()/np.sqrt(20))
# --- hand-read official FITS
h = fits.open(OFF/"KiDS_Legacy_xipm.fits")
print("FITS exts:", [x.name for x in h])
tp, tm = h["xiP"].data, h["xiM"].data
cov = h["COVMAT"].data if "COVMAT" in [x.name for x in h] else None
print("cov shape", None if cov is None else cov.shape, "xiP cols", tp.columns.names)
n = 9
res = {}
for k in (2, 5, 1, 3, 4, 6):
    a = np.genfromtxt(OFF/f"cosmosis_xipm/XI_KiDS_Legacy_NScomb_nBins_6_Bin{k}_Bin{k}.ascii", names=True)
    o = np.load(R/f"xi_unordered/b1.5/Bin{k}_Bin{k}.npz")
    w = a["npairs_weighted"]
    out = {}
    for s in ("xip", "xim"):
        A = (w*o[s]).sum()/(w*a[s]).sum()
        z = (o[s]-A*a[s])/a["sigma_"+s]
        out[s] = (A, boxr(z))
    sel = (tp["BIN1"] == k) & (tp["BIN2"] == k)
    ix = np.where(sel)[0]
    # A3 by hand
    m = (1+M[k-1])**2
    edges = np.exp(np.linspace(np.log(2), np.log(300), 10)); edges[-1] = 300.01
    r = o["r_nom"]; wn = o["npairs"]
    d3 = {}
    for s, t in (("xip", tp), ("xim", tm)):
        reb = np.array([np.sum(wn[(r >= edges[c]) & (r < edges[c+1])]*o[s][(r >= edges[c]) & (r < edges[c+1])])/np.sum(wn[(r >= edges[c]) & (r < edges[c+1])]) for c in range(n)])/m
        off = t.columns.names
        val = t["VALUE"][ix]
        off_i = ix + (0 if s == "xip" else 0)
        # sigma from cov diag: xip block then xim block, pair-major
        K = [(i, j) for i in range(1, 7) for j in range(i, 7)].index((k, k))
        diag = np.diag(cov)
        sig = np.sqrt(diag[K*9:(K+1)*9] if s == "xip" else diag[189+K*9:189+(K+1)*9])
        d3[s] = np.abs((reb-val)/sig).max()
    res[k] = (out, d3, o["r_nom"][0], ix[:3], tp["ANGBIN"][ix][:3] if "ANGBIN" in tp.columns.names else None)
    e = ev["pairs"][f"Bin{k}_Bin{k}"]
    print(f"Bin{k}: A_p mine xip {out['xip'][0]:.6f} xim {out['xim'][0]:.6f} | eval.json {e['A1']['A_p_xip']:.6f} {e['A1']['A_p_xim']:.6f}")
    print(f"   boxcar mine {out['xip'][1]:.4f}/{out['xim'][1]:.4f} | json {e['A1']['boxcar_over_white']}")
    print(f"   A3 mine {d3['xip']:.4f}/{d3['xim']:.4f} | json {e['A3']['xip']['max']:.4f}/{e['A3']['xim']['max']:.4f}  rows {ix[:2]}..  BIN1/2 ok")
# --- const test by hand
S = json.load(open(R/"a1_salvage/a1_salvage.json"))["step2_amplitude"]
PAIRS = [(i, j) for i in range(6) for j in range(i, 6)]
idx = [PAIRS.index((i, i)) for i in range(6)]
print("auto idx", idx)
for s in ("xip", "xim"):
    A = np.array([res[k][0][s][0] for k in range(1, 7)])
    sd = np.array(S[f"{s}_raw"]["A_pair_err_frac"])[idx]
    sd_all = np.array(S[f"{s}_raw"]["A_pair_err_frac"])
    wg = 1/sd**2; lA = np.log(A); lc = (lA*wg).sum()/wg.sum()
    contrib = ((lA-lc)/sd)**2
    ch = contrib.sum()
    th = np.array(S[f"{s}_raw"]["A_pair"])[idx]
    print(s, "A_p ours", np.round(A, 4), "theirs(salv)", np.round(th, 4), "sd", np.round(sd, 4))
    print("  C", np.exp(lc), "chi2", ch, "pte(dof5)", chi2.sf(ch, 5), "contrib", np.round(contrib, 2))
    print("  json C", ev["A1_const"][s]["C"], "chi2", ev["A1_const"][s]["const_chi2"], "pte", ev["A1_const"][s]["const_pte"])
    # alternative: all-21 global C from salvage
    print("  salvage 21-pair C(global)", S[f"{s}_raw"]["global_const_model"]["C"], " pte if dof 6 (unfitted C):", chi2.sf(ch, 6))
    # fitted-C with salvage C fixed
    chf = (((lA-np.log(S[f'{s}_raw']['global_const_model']['C']))/sd)**2).sum()
    print("  chi2 vs salvage-21 C (fixed):", chf, "pte dof6", chi2.sf(chf, 6))
