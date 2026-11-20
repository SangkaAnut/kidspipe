# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... 
# One-off A1b threshold-calibration study, not part of the Colab chain; see scripts/README.md.
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[2]      # repository root
import json, numpy as np
from pathlib import Path
from kidspipe.measure.official import PAIRS, load_official_xi
from kidspipe.measure.a1check import ratio_of_sums
asc=load_official_xi(); w=asc["npairs_weighted"]
TH=Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
Z={}
for key in ("xip","xim"):
    tr=np.array([np.load(TH/f"xipm_bin{i+1}_bin{j+1}.npz")[key+"_raw"] for i,j in PAIRS])
    A=ratio_of_sums(tr,asc[key],w); Z[key]=(tr-A[:,None]*asc[key])/asc["sigma_"+key]
z=np.vstack([Z["xip"],Z["xim"]])            # 42 x 1000
def acf(x,L):
    x=x-x.mean(); v=(x*x).mean(); return np.array([(x[:-k]*x[k:]).mean()/v for k in range(1,L+1)])
R=np.array([acf(c,40) for c in z])
print("pooled ACF lags 1..12:", np.round(R.mean(0)[:12],3))
print("pooled ACF lags 13..40 mean, sd:", round(R.mean(0)[12:].mean(),4), round(R.mean(0)[12:].std(),4), " (1/sqrt(N)=%.3f per curve)"%(1/np.sqrt(1000)))
print("lag1 per curve min/med/max:", np.round([R[:,0].min(),np.median(R[:,0]),R[:,0].max()],3))
print("lag2 per curve min/med/max:", np.round([R[:,1].min(),np.median(R[:,1]),R[:,1].max()],3))
# ratio implied by ACF, vs measured
def ratio_from_acf(r,B=20): return np.sqrt(1+2*sum((1-k/B)*r[k-1] for k in range(1,B)))
zb=z.reshape(42,50,20).mean(-1); meas=zb.std(1)/(z.std(1)/np.sqrt(20))
print("measured ratio: median %.3f max %.3f"%(np.median(meas),meas.max()))
print("ratio from lag1 only (MA1 null): median %.3f"%np.median(np.sqrt(np.clip(1+2*(19/20)*R[:,0],0,None))))
print("ratio from lags 1-2 only: median %.3f"%np.median(np.sqrt(np.clip(1+2*((19/20)*R[:,0]+(18/20)*R[:,1]),0,None))))
# does sigma vary across theta? (heteroscedastic z)
print("z rms by theta decade:", [round(z[:,m].std(),3) for m in np.array_split(np.arange(1000),5)])
np.savez(str(_R) + "/runs/dr5_measure/a1_salvage/threshold_calibration/z.npz", z=z, R=R, meas=meas)
