"""Independent recompute of A3 / E_n / B_n / A1 for the 6 slop-1.5 unordered auto pairs. Own code; only CosebisFilters imported."""
import sys, json, glob, os
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, "/mnt/d/cloning_kids_Kappa_pipeline/src")
from kidspipe.theory.cosebis import CosebisFilters
from astropy.io import fits
OFF = "/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_cosmic_shear_data_release_unpacked/KiDS_Legacy_cosmic_shear_data_release/data/"
OUT = "/mnt/d/cloning_kids_Kappa_pipeline/runs/dr5_measure/xi_unordered/b1.5/"
MM = [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045]
ARC = np.deg2rad(1/60)
res = {}
# ---- official, selected by header columns (no assumption on ordering)
hx = fits.open(OFF + "KiDS_Legacy_xipm.fits"); hc = fits.open(OFF + "KiDS_Legacy_cosebis.fits")
cov9 = np.array(hx["COVMAT"].data, float); covc = np.array(hc["COVMAT"].data, float)
tP, tM = hx["xiP"].data, hx["xiM"].data
En_t, Bn_t = hc["En"].data, hc["Bn"].data
print("xiM rows same order as xiP:", all(np.array_equal(tP[c], tM[c]) for c in ("BIN1", "BIN2", "ANGBIN")))
print("Bn rows same order as En:", all(np.array_equal(En_t[c], Bn_t[c]) for c in ("BIN1", "BIN2", "ANGBIN")))
print("cov shapes", cov9.shape, covc.shape, "N xi rows", len(tP), "N En rows", len(En_t))
nx = len(tP); ne = len(En_t)
filt = CosebisFilters(2.0, 300.0, source="mpmath", dps=80)
e9 = np.exp(np.linspace(np.log(2), np.log(300), 10))
R = {}
asc_cache = {}
for p in range(1, 7):
    # -- ours
    d = np.load(OUT + f"Bin{p}_Bin{p}.npz")
    r, xp, xm, npr, wt = (d[k].astype(float) for k in ("r_nom", "xip", "xim", "npairs", "weight"))
    # edges from r_nom
    q = (r[-1]/r[0])**(1/999); lo = r[0]/np.sqrt(q); edges = lo*q**np.arange(1001)
    f = (1+MM[p-1])**2
    xpc, xmc = xp/f, xm/f
    # -- official block rows by header
    ip = np.where((tP["BIN1"] == p) & (tP["BIN2"] == p))[0]; ip = ip[np.argsort(tP["ANGBIN"][ip])]
    assert len(ip) == 9 and np.all(tM["BIN1"][ip] == p) and np.all(tM["BIN2"][ip] == p)
    vp, vm = tP["VALUE"][ip], tM["VALUE"][ip]
    sp, sm = np.sqrt(np.diag(cov9)[ip]), np.sqrt(np.diag(cov9)[nx+ip])
    # rebin: assign by fine-bin geometric centre r_nom; also alt: by edges lower bound / meanr
    def reb(rr, y, w):
        o = []
        for c in range(9):
            m = (rr >= e9[c]) & (rr < e9[c+1]) if c < 8 else (rr >= e9[c]) & (rr <= 300.01)
            o.append((w[m]*y[m]).sum()/w[m].sum())
        return np.array(o)
    a3 = {}
    for nm, rr in (("r_nom", r), ("meanr", d["meanr"].astype(float))):
        a3[nm] = (np.abs((reb(rr, xpc, npr)-vp)/sp).max(), np.abs((reb(rr, xmc, npr)-vm)/sm).max())
    a3w = (np.abs((reb(r, xpc, wt)-vp)/sp).max(), np.abs((reb(r, xmc, wt)-vm)/sm).max())
    # E/B
    th = r; z = filt.z_of_theta(th); Tp = filt.Tplus_z(z, 20); Tm = filt.Tminus_z(z, 20)
    w = 0.5*(edges[1:]**2 - edges[:-1]**2)
    ap = Tp @ (w*xpc); am = Tm @ (w*xmc)
    E = 0.5*ARC**2*(ap+am); B = 0.5*ARC**2*(ap-am)
    ie = np.where((En_t["BIN1"] == p) & (En_t["BIN2"] == p))[0]; ie = ie[np.argsort(En_t["ANGBIN"][ie])]
    assert len(ie) == 20 and np.all(Bn_t["BIN1"][ie] == p) and np.all(Bn_t["BIN2"][ie] == p)
    Eo, Bo = En_t["VALUE"][ie], Bn_t["VALUE"][ie]
    sE = np.sqrt(np.diag(covc)[ie])
    # check which covariance layout: En then Bn (840 = 2*420)
    CB = covc[np.ix_(ie+ne, ie+ne)]
    dE = (E-Eo)/sE
    cb6 = B[:6] @ np.linalg.solve(CB[:6, :6], B[:6]); cb6o = Bo[:6] @ np.linalg.solve(CB[:6, :6], Bo[:6])
    cb20 = B @ np.linalg.solve(CB, B); cb20o = Bo @ np.linalg.solve(CB, Bo)
    # A1: official ascii of this pair
    fn = OFF + f"cosmosis_xipm/XI_KiDS_Legacy_NScomb_nBins_6_Bin{p}_Bin{p}.ascii"
    a = np.genfromtxt(fn, names=True)
    assert np.allclose(a["r_nom"], r, rtol=1e-5), "ascii r_nom mismatch"
    A1 = []
    for xo, key in ((xp, "xip"), (xm, "xim")):
        g = a["npairs_weighted"]
        A = (g*a[key]).sum()/(g*xo).sum()      # official/ours ... gate: A = sum(g t)/sum(g a), t=ours
        A = (g*xo).sum()/(g*a[key]).sum()
        zz = (xo - A*a[key])/a["sigma_"+key]
        zb = zz[:1000//20*20].reshape(-1, 20).mean(1)
        A1.append(zb.std()/(zz.std()/np.sqrt(20)))
    # catalogue
    c = np.load(f"/home/anut/kids_dr5_work/bin{p-1}.npz")
    nobj = len(c["w"]); sw = float(c["w"].astype(np.float64).sum()); sw2 = float((c["w"].astype(np.float64)**2).sum())
    prov = {k: (d[k].item() if d[k].ndim == 0 else None) for k in d.files if d[k].ndim == 0}
    R[p] = dict(A3_rnom=a3["r_nom"], A3_meanr=a3["meanr"], A3_wt=a3w, maxE6=np.abs(dE[:6]).max(), maxE20=np.abs(dE).max(),
                B6=cb6, B6o=cb6o, B20=cb20, B20o=cb20o, pte6=chi2.sf(cb6, 6), A1=A1, nobj=nobj, nobj_npz=(prov["nobj1"], prov["nobj2"]),
                sumw=sw, sumw2=sw2, prov=prov, npairs_sum=npr.sum(), wsum=wt.sum(), asc_np=a["npairs"].sum(),
                nan=int(np.isnan(xp).sum()+np.isnan(xm).sum()+np.isnan(npr).sum()), minnp=float(npr.min()),
                ncat_nan=int(np.isnan(c["e1"]).sum()+np.isnan(c["e2"]).sum()+np.isnan(c["w"]).sum()),
                Emean=(float(c["e1"].astype(float)@c["w"]/sw), float(c["e2"].astype(float)@c["w"]/sw)))
    print(p, {k: (np.round(v, 4) if isinstance(v, (float, np.floating, tuple, list)) else v) for k, v in R[p].items() if k != "prov"}, flush=True)
    print("   prov", prov)
json.dump(R, open("/tmp/claude-1000/-home-anut/8ecc82c7-22dd-490c-b045-63f93bce2233/scratchpad/recompute.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
