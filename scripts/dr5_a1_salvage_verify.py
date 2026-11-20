# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Independent verification of A1 salvage. Own rebin, own covariance slicing, own B_i fit."""
import glob, re, json, numpy as np
from astropy.io import fits
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402
O=paths.official_dir()
T=Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
OUT=Path(__file__).resolve().parents[1]/"runs/dr5_measure/a1_salvage/verify"; OUT.mkdir(parents=True, exist_ok=True)
pairs=[(i,j) for i in range(1,7) for j in range(i,7)]
res={}
# ascii
asc={k:np.zeros((21,1000)) for k in "xip xim npairs_weighted r_nom meanr".split()}
for p,(i,j) in enumerate(pairs):
    a=np.genfromtxt(O/"cosmosis_xipm"/f"XI_KiDS_Legacy_NScomb_nBins_6_Bin{i}_Bin{j}.ascii",names=True)
    for k in asc: asc[k][p]=a[k]
# their
th={k:np.zeros((21,1000)) for k in "xip xim xip_raw xim_raw npairs r_nom".split()}
for p,(i,j) in enumerate(pairs):
    d=np.load(T/f"xipm_bin{i}_bin{j}.npz")
    for k in th: th[k][p]=d[k]
# FITS: read rows explicitly by BIN1,BIN2,ANGBIN
h=fits.open(O/"KiDS_Legacy_xipm.fits")
print([x.name for x in h])
def tab(ext):
    t=h[ext].data; v=np.full((21,9),np.nan)
    for r in range(len(t)):
        p=pairs.index((int(t['BIN1'][r]),int(t['BIN2'][r]))); v[p,int(t['ANGBIN'][r])-1]=t['VALUE'][r]
    return v, t
xiP,tp=tab("xiP"); xiM,tm=tab("xiM")
# check row order of table vs [pair-major, angbin] and of covariance
order=[(int(tp['BIN1'][r]),int(tp['BIN2'][r]),int(tp['ANGBIN'][r])) for r in range(len(tp))]
canon=[(i,j,a) for (i,j) in pairs for a in range(1,10)]
print("xiP row order pair-major canonical:",order==canon, "min ANGBIN",min(o[2] for o in order))
orderM=[(int(tm['BIN1'][r]),int(tm['BIN2'][r]),int(tm['ANGBIN'][r])) for r in range(len(tm))]
print("xiM same order:",orderM==order)
cov=np.array(h["COVMAT"].data,float); print(cov.shape, "asym",np.abs(cov-cov.T).max()/np.abs(cov).max())
cov=.5*(cov+cov.T)
print("COVMAT hdr", {k:h["COVMAT"].header[k] for k in h["COVMAT"].header if k.startswith(('COV','NAME','STRT','_'))} )
Cp=cov[:189,:189]; Cm=cov[189:,189:]
# sanity on which half is xiM: xi- noise > xi+ at small scales? compare S/N: xiP value^2/diag
print("mean xiP/sigP",np.nanmean(np.abs(xiP.ravel())/np.sqrt(np.diag(Cp))),"mean xiM/sigM",np.nanmean(np.abs(xiM.ravel())/np.sqrt(np.diag(Cm))))
print("mean xiP/sigM",np.nanmean(np.abs(xiP.ravel())/np.sqrt(np.diag(Cm))))
# own rebin: edges, select by r_nom, weight by w
e=np.exp(np.linspace(np.log(2),np.log(300),10))
def rebin(xi,w,r):
    out=np.zeros((21,9))
    for b in range(9):
        lo,hi=e[b],e[b+1]
        m=(r>=lo)&(r<hi) if b<8 else (r>=lo)&(r<=hi*(1+1e-9))
        out[:,b]=(w*xi*m).sum(1)/(w*m).sum(1)
    return out
def chi(d,C):
    d=d.ravel(); c=d@np.linalg.solve(C,d); return c, c/len(d), np.abs(d/np.sqrt(np.diag(C))).max()
fine=np.exp(np.linspace(np.log(2),np.log(300),1001)); rn=np.sqrt(fine[:-1]*fine[1:])
print("ascii r_nom==geom centre?",np.abs(asc['r_nom'][0]/rn-1).max(),"meanr vs rnom max",np.abs(asc['meanr']/asc['r_nom']-1).max())
for tag,r in (("rnom",asc['r_nom']),("meanr",asc['meanr'])):
    for nm,xi,V,C in (("xip",asc['xip'],xiP,Cp),("xim",asc['xim'],xiM,Cm)):
        c=chi(rebin(xi,asc['npairs_weighted'],r)-V,C); res[f"1_ascii_{nm}_{tag}"]=c; print("1 ascii",nm,tag,c)
# wrong block
c=chi(rebin(asc['xip'],asc['npairs_weighted'],asc['r_nom'])-xiP,Cm); print("1 xip with XIM block (wrong)",c)
# diag
d=(rebin(asc['xip'],asc['npairs_weighted'],asc['r_nom'])-xiP).ravel(); print("diag-only xip chi2/dof",np.mean(d**2/np.diag(Cp)))
d=(rebin(asc['xim'],asc['npairs_weighted'],asc['r_nom'])-xiM).ravel(); print("diag-only xim chi2/dof",np.mean(d**2/np.diag(Cm)))
# median ratio
for nm,xi,V in (("xip",asc['xip'],xiP),("xim",asc['xim'],xiM)):
    R=rebin(xi,asc['npairs_weighted'],asc['r_nom'])/V; res[f"med_ratio_{nm}"]=float(np.median(R)); print("median ratio",nm,np.median(R))
# alt: median rebin (the prompt's)
def medrebin(xi,r):
    out=np.zeros((21,9))
    for b in range(9):
        m=(r[0]>=e[b])&(r[0]<e[b+1]) if b<8 else (r[0]>=e[b])
        out[:,b]=np.median(xi[:,m],axis=1)
    return out
for nm,xi,V,C in (("xip",asc['xip'],xiP,Cp),("xim",asc['xim'],xiM,Cm)):
    R=medrebin(xi,asc['r_nom'])/V; print("median-rebin ratio median",nm,np.median(R), chi(medrebin(xi,asc['r_nom'])-V,C))
# 2 their
for wn,w in (("ascii_npw",asc['npairs_weighted']),("own_npairs",th['npairs'])):
    for nm,xi,V,C in (("xip",th['xip'],xiP,Cp),("xim",th['xim'],xiM,Cm)):
        c=chi(rebin(xi,w,th['r_nom'])-V,C); res[f"2_their_{nm}_{wn}"]=c; print("2 their",nm,wn,c)
# 3: B_i fit on 9-bin vectors.  model asc9 = s_i s_j * ... we fit ascii9 = F_p * FITS ; F_p=s_i s_j; GLS with FITS cov (linearised: residual asc - F*fits, cov C scaled)
a9={"xip":rebin(asc['xip'],asc['npairs_weighted'],asc['r_nom']),"xim":rebin(asc['xim'],asc['npairs_weighted'],asc['r_nom'])}
from scipy.optimize import least_squares
def fit(nm,V,C):
    L=np.linalg.cholesky(np.linalg.inv(C))   # C^-1 = L L^T
    def rf(ls):
        s=np.exp(ls); F=np.array([s[i-1]*s[j-1] for i,j in pairs])[:,None]
        return L.T@((a9[nm]-F*V).ravel())
    r=least_squares(rf,np.log(np.full(6,0.95)))
    J=r.jac; cv=np.linalg.inv(J.T@J); s=np.exp(r.x); return s, s*np.sqrt(np.diag(cv)), (r.fun**2).sum()
S={}
for nm,V,C in (("xip",xiP,Cp),("xim",xiM,Cm)):
    s,err,c2=fit(nm,V,C); S[nm]=s.tolist(); res[f"3_s_{nm}"]=dict(s=s.tolist(),err=err.tolist(),chi2=float(c2))
    print("3 s_i GLS",nm,np.round(s,3),np.round(err,3),"chi2/dof(189-6)",c2/183)
# 3b: 1000-bin ratio-of-sums fit (non-covariance footing), their raw vs ascii, and corrected vs ascii
for lab,key in (("corrected","xip"),("raw","xip_raw")):
  for nm in ("xip","xim"):
    k=nm if lab=="corrected" else nm+"_raw"
    A=np.array([th[k][p].dot(asc['npairs_weighted'][p]*0+1)/1 for p in range(21)])  # placeholder
    A=np.array([(th[k][p]*np.abs(asc[nm][p])**0).sum()/asc[nm][p].sum() for p in range(21)])  # plain sum ratio
    # fit ln A_p = lnB_i+lnB_j
    X=np.zeros((21,6))
    for p,(i,j) in enumerate(pairs): X[p,i-1]+=1; X[p,j-1]+=1
    lb=np.linalg.lstsq(X,np.log(A),rcond=None)[0]; resid=np.log(A)-X@lb
    res[f"3b_{lab}_{nm}"]=dict(B=np.exp(lb).tolist(),invB=np.exp(-lb).tolist(),rms=float(resid.std()))
    print("3b",lab,nm,"1/B=",np.round(np.exp(-lb),3),"resid rms",resid.std().round(4), "mean A",A.mean().round(4))
print("expected m factors (1+m)",)
M=[-0.023,-0.016,-0.011,0.020,0.030,0.045]; print("1/(1+m_i)",np.round(1/(1+np.array(M)),3))
json.dump(res,open(OUT/"verify.json","w"),indent=1,default=float)
