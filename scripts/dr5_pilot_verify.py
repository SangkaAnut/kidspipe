"""Independent verification of the Bin6_Bin6 pilot (does not import dr5_pilot_bin6*.py)."""
import sys, json, numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2d
from astropy.io import fits
ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline"); sys.path.insert(0, str(ROOT/"src"))
from kidspipe.measure import official as O
from kidspipe.measure.a1check import ratio_of_sums
from kidspipe.measure.xi2cosebis import e_from_xi_binned
from kidspipe.theory.cosebis import CosebisFilters
TH = Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin6_bin6.npz")
k = O.PAIRS.index((5, 5)); print("pair index", k, O.PAIRS[k])
off = O.load_official_xi(); ob = O.load_official_cosebis()
ours = np.load(ROOT/"runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz")
sl = np.load(ROOT/"runs/dr5_measure/pilot_bin6/bin6_bin6_slop15_assembled.npz")
th = np.load(TH); M = th["m_factor"]; print("m_factor", M, 1.045**2)
res = {}
print("r_nom diff ours/ascii", np.abs(ours["r_nom"]-off["r_nom"][k]).max(), "theirs", np.abs(th["r_nom"]-off["r_nom"][k]).max())
w = off["npairs_weighted"][k]
for name, d, kp, km in (("ours", ours, "xip", "xim"), ("slop15", sl, "xip", "xim"), ("theirs", th, "xip_raw", "xim_raw")):
    for key, kk in (("xip", kp), ("xim", km)):
        y = d[kk]; a = off[key][k]
        A = (w*y).sum()/(w*a).sum()
        z = (y-A*a)/off["sigma_"+key][k]
        zb = z.reshape(-1, 20).mean(1)
        rat = zb.std()/(z.std()/np.sqrt(20))
        zz = z-z.mean(); lag1 = (zz[:-1]*zz[1:]).sum()/(zz*zz).sum()
        print(f"{name:7s} {key}: A_p={A:.5f} boxcar/white={rat:.3f} lag1={lag1:.3f}")
        res[f"{name}_{key}"] = dict(A=A, rat=rat)
# amplitude ratios with theirs weight
wt = th["weight"]
for key, kk in (("xip", "xip_raw"), ("xim", "xim_raw")):
    print(key, "ours/theirs", (wt*ours[key]).sum()/(wt*th[kk]).sum(), "slop15/theirs", (wt*sl[key]).sum()/(wt*th[kk]).sum(),
          "ours/slop15", (wt*ours[key]).sum()/(wt*sl[key]).sum())
print("npairs ours/theirs", ours["npairs"].sum()/th["npairs"].sum(), "slop15/theirs", sl["npairs"].sum()/th["npairs"].sum(),
      "ours/official", ours["npairs"].sum()/off["npairs"][k].sum())
# A3
h = fits.open(O.OFFICIAL/"KiDS_Legacy_xipm.fits"); cov = 0.5*(h["COVMAT"].data+h["COVMAT"].data.T)
fx = O.load_official_xipm_fits()
print("fits pair row k theta", fx["theta"])
edges = np.exp(np.linspace(np.log(2), np.log(300), 10))
def reb(r, w_, y):
    o = np.zeros(9)
    for c in range(9):
        m = (r >= edges[c]) & (r < edges[c+1] if c < 8 else r <= 300.01)
        o[c] = (w_[m]*y[m]).sum()/w_[m].sum()
    return o
for name, d, kp, km, corr in (("ours", ours, "xip", "xim", M), ("slop15", sl, "xip", "xim", M), ("theirs", th, "xip", "xim", 1.0)):
    for st, key, blk in (("xip", kp, 0), ("xim", km, 1)):
        y = d[key]/corr
        idx = blk*189 + k*9 + np.arange(9)
        sig = np.sqrt(np.diag(cov)[idx])
        fits_val = fx["xiP" if st == "xip" else "xiM"][k]
        dd = reb(ours["r_nom"], d["npairs"], y) - fits_val
        print(f"A3 {name:6s} {st}: max|d|/sig={np.abs(dd/sig).max():.4f}  d/sig={np.round(dd/sig,3)}")
        res[f"A3_{name}_{st}"] = float(np.abs(dd/sig).max())
# check sigma from cov row vs fits ordering: compare sigma with official sigma9 in fx
print("cov diag idx check: xiP sigma", np.sqrt(np.diag(cov)[k*9:(k+1)*9])[:3])
# also unflipped sign test/ swap: xi+ <-> xi- deliberately
# COSEBIs
filt = CosebisFilters(2.0, 300.0, source="mpmath", dps=60)
def cb(d, corr):
    return e_from_xi_binned(d["r_nom"], d["xip"]/corr, d["xim"]/corr, filt, nmodes=20, scheme="edges", unit="rad")
Cfull = ob["cov"]
for name, d, corr in (("ours", ours, M), ("slop15", sl, M), ("theirs", th, 1.0), ("ours_raw", ours, 1.0)):
    if name == "theirs":
        E, B = cb(dict(r_nom=th["r_nom"], xip=th["xip"], xim=th["xim"]), 1.0)
    else:
        E, B = cb(d, corr)
    for nm in (6, 20):
        ie = k*20+np.arange(nm); C = Cfull[np.ix_(ie, ie)]
        dE = E-ob["En"][k]; dE = dE[:nm]
        c2 = dE@np.linalg.solve(C, dE)
        ib = 420+k*20+np.arange(nm); Cb = Cfull[np.ix_(ib, ib)]
        b = B[:nm]; c2b = b@np.linalg.solve(Cb, b)
        print(f"COSEBI {name:8s} {nm:2d}m: En max|d|/sig={np.abs(dE/np.sqrt(np.diag(C))).max():.3f} chi2={c2:.3f}  Bn chi2={c2b:.2f} pte={chi2d.sf(c2b,nm):.4f}")
    if name == "ours": print("  dE/sig by mode", np.round((E-ob["En"][k])/np.sqrt(np.diag(Cfull)[k*20:(k+1)*20]),3))
ob_b = ob["Bn"][k]
for nm in (6, 20):
    ib = 420+k*20+np.arange(nm); Cb = Cfull[np.ix_(ib, ib)]
    c2 = ob_b[:nm]@np.linalg.solve(Cb, ob_b[:nm]); print("official Bn chi2", nm, c2, chi2d.sf(c2, nm))
# xip<->xim swap check on A1
print("swap check ours.xip vs ascii xim:", (w*ours["xip"]).sum()/(w*off["xim"][k]).sum())
# varxi factor, weight ratio
print("sqrt varxi ours/theirs sigma", np.median(np.sqrt(ours["varxi"])/th["sigma_xip"]), "official sigma / ours", np.median(off["sigma_xip"][k]/np.sqrt(ours["varxi"])))
print("sum weight/off npw", ours["weight"].sum()/w.sum(), (ours["weight"]/w)[:5], (ours["weight"]/w)[-5:])
print("xim_im z", np.mean(ours["xim_im"]/(np.sqrt(2)*np.sqrt(ours["varxi"]))), np.std(ours["xim_im"]/(np.sqrt(2)*np.sqrt(ours["varxi"]))))
