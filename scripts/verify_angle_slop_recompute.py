# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
import os
os.environ["OMP_NUM_THREADS"]="2"
import sys, json, numpy as np
from pathlib import Path
from scipy.integrate import simpson
from scipy.stats import chi2 as chi2d
from astropy.io import fits
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R/"src"))
from kidspipe.theory.cosebis import CosebisFilters
from kidspipe import paths
O=paths.official_dir()
out={}
# ---- official 9-bin xipm, own parsing (no official.py)
with fits.open(O/"KiDS_Legacy_xipm.fits") as h:
    print([x.name for x in h]); 
    cv=h["COVMAT"]; print({k:cv.header[k] for k in cv.header if k.startswith(("STRT","NAME","UNIT","COV"))})
    P,M=h["xiP"].data,h["xiM"].data; C=np.array(cv.data,float)
    print(P.columns.names, P["ANG"][:9], P["BIN1"][:3],P["BIN2"][:3],P["ANGBIN"][:3], P["BIN1"][9*5:9*7:9], P["BIN2"][9*5:9*7:9])
    # select bin6-bin6 by own mask; order by ANGBIN
    sel=lambda T: np.where((T["BIN1"]==6)&(T["BIN2"]==6))[0]
    iP,iM=sel(P),sel(M); iP=iP[np.argsort(P["ANGBIN"][iP])]; iM=iM[np.argsort(M["ANGBIN"][iM])]
    print("rows",iP,iM)
    xiP,xiM=np.array(P["VALUE"][iP],float),np.array(M["VALUE"][iM],float); ang=np.array(P["ANG"][iP],float)
    # covariance ordering: assume xiP rows then xiM rows in file order
    nP=len(P); print("nP",nP,C.shape)
    cvP=C[np.ix_(iP,iP)]; cvM=C[np.ix_(nP+iM,nP+iM)]
    sP,sM=np.sqrt(np.diag(cvP)),np.sqrt(np.diag(cvM))
    # sanity: sigma vs |xi| scale, off-diag cross
    print("sigP",sP,"sigM",sM,"xiP",xiP,"xiM",xiM)
    print("ANG units", h["xiP"].header.get("TUNIT4"), ang)
with fits.open(O/"KiDS_Legacy_cosebis.fits") as h:
    print([x.name for x in h]); te,tb=h["En"].data,h["Bn"].data; Cc=np.array(h["COVMAT"].data,float)
    print(te.columns.names, Cc.shape, {k:h["COVMAT"].header[k] for k in h["COVMAT"].header if k.startswith(("STRT","NAME"))})
    print(h["En"].header.get("TUNIT4"), te["VALUE"][:3], te["BIN1"][:3], te["ANGBIN"][:3])
    ie=np.where((te["BIN1"]==6)&(te["BIN2"]==6))[0]; ie=ie[np.argsort(te["ANGBIN"][ie])]
    ib=np.where((tb["BIN1"]==6)&(tb["BIN2"]==6))[0]; ib=ib[np.argsort(tb["ANGBIN"][ib])]
    nE=len(te); Eoff=np.array(te["VALUE"][ie],float); Boff=np.array(tb["VALUE"][ib],float)
    CE=Cc[np.ix_(ie,ie)]; CB=Cc[np.ix_(nE+ib,nE+ib)]
    print("nE",nE,"ie",ie[:3],"ib",ib[:3])
sE=np.sqrt(np.diag(CE)); print("Eoff",Eoff,"sE",sE)
# ---- data
def L(f): return {k:np.asarray(v,float) for k,v in np.load(f).items()}
D={"exact":L(R/"runs/dr5_measure/pilot_bin6/bin6_bin6_b0_a0_assembled.npz"),
   "pilot":L(R/"runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz"),
   "slop15":L(R/"runs/dr5_measure/pilot_bin6/bin6_bin6_slop15_assembled.npz")}
# official ascii bin6-bin6
a=np.genfromtxt(sorted((O/"cosmosis_xipm").glob("XI_KiDS_Legacy_NScomb_nBins_6_Bin6_Bin6.ascii"))[0],names=True)
wasc=a["npairs_weighted"]
th=D["exact"]["r_nom"]
for k in D: assert np.allclose(D[k]["r_nom"],th)
e9=np.exp(np.linspace(np.log(2),np.log(300),10))
def rb(w,y,mode="centre"):
    o=np.zeros(9)
    for c in range(9):
        m=(th>=e9[c])&((th<e9[c+1]) if c<8 else (th<=300.0*(1+1e-9)))
        o[c]=(w[m]*y[m]).sum()/w[m].sum()
    return o
m6=0.045; fac=(1+m6)**2
print("m=0.045 factor",fac)
# COSEBI filter
filt=CosebisFilters(2.0,300.0,source="mpmath",dps=80)
edges=np.exp(np.linspace(np.log(2),np.log(300),1001)); w_e=0.5*(edges[1:]**2-edges[:-1]**2)
z=np.log(th/2.0); Tp=filt.Tplus_z(z,20); Tm=filt.Tminus_z(z,20)
AR2=np.deg2rad(1/60)**2
def En(xp,xm,unit=AR2):
    ap=Tp@(w_e*xp); am=Tm@(w_e*xm); return 0.5*unit*(ap+am),0.5*unit*(ap-am)
res={}
for k,d in D.items():
    xp,xm=d["xip"]/fac,d["xim"]/fac
    r={}
    for wn,w in (("npairs",d["npairs"]),("weight",d["weight"]),("offascii_npw",wasc)):
        r["A3_xip_"+wn]=((rb(w,xp)-xiP)/sP).tolist(); r["A3_xim_"+wn]=((rb(w,xm)-xiM)/sM).tolist()
    E,B=En(xp,xm); r["dE"]=((E-Eoff)/sE).tolist(); r["E"]=E.tolist(); r["B"]=B.tolist()
    for n in (6,20): r[f"Bchi2_{n}"]=float(B[:n]@np.linalg.solve(CB[:n,:n],B[:n]))
    r["Echi2_6"]=float((E-Eoff)[:6]@np.linalg.solve(CE[:6,:6],(E-Eoff)[:6]))
    # also raw (no m) for contrast
    E0,_=En(d["xip"],d["xim"]); r["dE_noM_max6"]=float(np.abs((E0-Eoff)/sE)[:6].max())
    # with unit arcmin^2 (to prove unit)
    res[k]=r
    print(k,"A3 xi+ (npairs)",np.abs(r["A3_xip_npairs"]).max(),"xi-",np.abs(r["A3_xim_npairs"]).max(),
          "| weight:",np.abs(r["A3_xip_weight"]).max(),np.abs(r["A3_xim_weight"]).max(),
          "| offascii w:",np.abs(r["A3_xip_offascii_npw"]).max(),np.abs(r["A3_xim_offascii_npw"]).max(),
          "| E6/E20",np.abs(r["dE"][:6]).max(),np.abs(r["dE"]).max(),"B2 6/20",r["Bchi2_6"],r["Bchi2_20"],"Echi2_6",r["Echi2_6"])
    print("  xi- by bin",np.round(r["A3_xim_npairs"],4)); print("  xi+ by bin",np.round(r["A3_xip_npairs"],4)); print("  dE",np.round(r["dE"],3))
# C3: differences
def diff(a,b):
    da,db=D[a],D[b]; w=da["npairs"]
    dp=(rb(w,da["xip"]/fac)-rb(db["npairs"],db["xip"]/fac))/sP; dm=(rb(w,da["xim"]/fac)-rb(db["npairs"],db["xim"]/fac))/sM
    Ea,_=En(da["xip"]/fac,da["xim"]/fac); Eb,_=En(db["xip"]/fac,db["xim"]/fac); dE=(Ea-Eb)/sE
    return dict(xip=dp.tolist(),xim=dm.tolist(),dE=dE.tolist(),max_xip=float(abs(dp).max()),max_xim=float(abs(dm).max()),E6=float(abs(dE[:6]).max()),E20=float(abs(dE).max()))
for a_,b_ in (("slop15","exact"),("pilot","exact")):
    r=diff(a_,b_); res[f"{a_}-{b_}"]=r; print(a_,"-",b_,r["max_xip"],r["max_xim"],r["E6"],r["E20"]); print(" dE",np.round(r["dE"],3)); print(" xim",np.round(r["xim"],4)); print(" xip",np.round(r["xip"],4))
# ---- independent filter checks
# (1) Simpson E_1 with own quadrature, fine grid (log-spaced 20001 pts) from interpolated exact xi
zz=np.linspace(0,np.log(150),40001); tt=2*np.exp(zz)
xpi=np.interp(np.log(tt),np.log(th),D["exact"]["xip"]/fac); xmi=np.interp(np.log(tt),np.log(th),D["exact"]["xim"]/fac)
# independent T+_n from definition: polynomial in z with roots/norm -> same object; instead verify T+ orthonormality & constraints, and T- by independent ODE-free relation
Tpz=filt.Tplus_z(zz,3)
# independent T-: Schneider T-(t)=4 int_t^tmax dv/v T+(v)(1-3t^2/v^2) ... (+T+(t) form tested numerically both signs)
from scipy.integrate import cumulative_trapezoid
def Tm_indep(n):
    f=Tpz[n]; v=tt
    # int_t^tmax dv/v T+(v)(1-3 t^2/v^2) = int dz f - 3 t^2 int dz f/v^2 ; tail integrals via Simpson cumulative from right
    g1=f; g2=f/v**2
    c1=cumulative_trapezoid(g1[::-1],zz[::-1],initial=0)[::-1]*-1; c2=cumulative_trapezoid(g2[::-1],zz[::-1],initial=0)[::-1]*-1
    return 4*(c1-3*tt**2*c2)
for n in range(3):
    ti=Tm_indep(n); tk=filt.Tminus_z(zz,3)[n]
    for nm,c in (("-T+",-Tpz[n]),("+T+",Tpz[n]),("0",0*Tpz[n])):
        dd=np.abs(ti+c-tk)[2000:-2000]; print("Tm check mode",n+1,nm,"maxdiff",dd.max(),"scale",np.abs(tk).max())
Ep=simpson(0.5*Tpz[0]*xpi*tt*tt,x=zz); 
Em=simpson(0.5*filt.Tminus_z(zz,1)[0]*xmi*tt*tt,x=zz)
print("E1 simpson (arcmin^2 units->rad^2):",(Ep+Em)*AR2,"vs mine",res["exact"]["E"][0])
res["E1_simpson"]=float((Ep+Em)*AR2)
# (2) kernel check against official: official E from THEIR xi
th_=L("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin6_bin6.npz") if False else None
T=np.load("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin6_bin6.npz",allow_pickle=True)
Et,Bt=En(np.asarray(T["xip"],float),np.asarray(T["xim"],float)); dEt=(Et-Eoff)/sE
print("theirs(m-corr) E vs official sigma: max6/20",abs(dEt[:6]).max(),abs(dEt).max(), "A3 theirs xi+ xi-",np.abs((rb(np.asarray(T['npairs'],float),np.asarray(T['xip'],float))-xiP)/sP).max(),np.abs((rb(np.asarray(T['npairs'],float),np.asarray(T['xim'],float))-xiM)/sM).max())
res["theirs"]={"dE":dEt.tolist()}
res["unit_check"]={"E_over_Eoff":(np.array(res['exact']['E'])/Eoff).tolist()}
# (3) A_p sanity: ratio exact/pilot amplitude, same factor
for k in D: print(k,"weighted sum xip",(D[k]["npairs"]*D[k]["xip"]).sum()/D[k]["npairs"].sum(), "npairs",D[k]["npairs"].sum())
json.dump(res,open(R/"runs/dr5_measure/angle_slop/verify/recompute_numbers.json","w"),indent=1)
