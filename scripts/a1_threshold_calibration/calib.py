# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... 
# One-off A1b threshold-calibration study, not part of the Colab chain; see scripts/README.md.
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[2]      # repository root
import numpy as np
from scipy.optimize import brentq
d=np.load(str(_R) + "/runs/dr5_measure/a1_salvage/threshold_calibration/z.npz"); z,R,meas=d["z"],d["R"],d["meas"]
rng=np.random.default_rng(11)
# per-curve MA(2) fit from lags 1,2 (method of moments): rho1=(t1+t1 t2)/(1+t1^2+t2^2), rho2=t2/(1+t1^2+t2^2)
from scipy.optimize import least_squares
def fit_ma2(r1,r2):
    f=lambda t:[(t[0]+t[0]*t[1])/(1+t[0]**2+t[1]**2)-r1, t[1]/(1+t[0]**2+t[1]**2)-r2]
    return least_squares(f,[-0.6,0.0],bounds=([-0.99,-0.99],[0.99,0.99])).x
TH=np.array([fit_ma2(r[0],r[1]) for r in R])
prof=np.array([np.sqrt(np.convolve(c**2,np.ones(51)/51,mode="same")) for c in z])   # local rms profile per curve
def stat(x):
    xb=x.reshape(x.shape[0],50,20).mean(-1); return xb.std(1)/(x.std(1)/np.sqrt(20))
def sim_null(n):
    out=np.empty((n,42))
    for s in range(n):
        e=rng.standard_normal((42,1002))
        x=e[:,2:]+TH[:,0:1]*e[:,1:-1]+TH[:,1:2]*e[:,:-2]
        out[s]=stat(x*prof)
    return out
N=sim_null(4000); mx=N.max(1)
print("per-curve null ratio median (sim) %.3f vs measured %.3f"%(np.median(N),np.median(meas)))
for q in (50,90,95,99,99.9): print(f"fitted-null max over 42: {q}th pct = {np.percentile(mx,q):.3f}")
print("measured max = %.3f  -> its null pte = %.3f"%(meas.max(),(mx>=meas.max()).mean()))
for t in (0.787,1.0,1.33): print(f"false-fail rate of threshold {t}: {(mx>t).mean():.4f}")
# per-curve standardized alternative: largest per-curve pte deficit
pc=np.array([(N[:,i]>=meas[i]).mean() for i in range(42)])
print("per-curve null pte: min %.3f (curve %d), median %.3f"%(pc.min(),pc.argmin(),np.median(pc)))
np.save(str(_R) + "/runs/dr5_measure/a1_salvage/threshold_calibration/nullmax.npy",mx)
