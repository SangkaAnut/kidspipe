# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... 
# One-off A1b threshold-calibration study, not part of the Colab chain; see scripts/README.md.
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[2]      # repository root
import json, numpy as np
from pathlib import Path
from kidspipe.measure.official import PAIRS, load_official_xi
from kidspipe.measure.a1check import a1b_noise
S=str(_R) + "/runs/dr5_measure/a1_salvage/threshold_calibration/"
d=np.load(S+"z.npz"); z,R,meas=d["z"],d["R"],d["meas"]
lab=[f"xi+ {i+1}-{j+1}" for i,j in PAIRS]+[f"xi- {i+1}-{j+1}" for i,j in PAIRS]
o=np.argsort(meas)[::-1][:5]
print("top-5 curves by ratio:", [(lab[i], round(meas[i],3), "lag1 %.3f"%R[i,0]) for i in o])
c=z[20]; zb=c.reshape(50,20).mean(-1)
print(lab[20],"boxcar means (first 50, units of z):", np.round(zb,2))
print(lab[20],"lag1 %.3f lag2 %.3f, lags3-10 %s"%(R[20,0],R[20,1],np.round(R[20,2:10],3)))
# power vs threshold on real data
asc=load_official_xi(); th=asc["r_nom"][0]
TH=Path("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial")
rp=np.array([np.load(TH/f"xipm_bin{i+1}_bin{j+1}.npz")["xip_raw"] for i,j in PAIRS]); rm=np.array([np.load(TH/f"xipm_bin{i+1}_bin{j+1}.npz")["xim_raw"] for i,j in PAIRS])
cp=np.array([np.load(TH/f"xipm_bin{i+1}_bin{j+1}.npz")["xip"] for i,j in PAIRS]); cm=np.array([np.load(TH/f"xipm_bin{i+1}_bin{j+1}.npz")["xim"] for i,j in PAIRS])
amp=json.load(open(_R / "runs/dr5_measure/a1_salvage/a1_salvage.json"))["step2_amplitude"]; sd={k:np.array(amp[f"{k}_raw"]["A_pair_err_frac"]) for k in ("xip","xim")}
ltf=np.log(th/np.sqrt(2*300)); ltf/=np.abs(ltf).max()
rng=np.random.default_rng(7)
def mr(tp,tm):
    r=a1b_noise(tp,tm,asc,sd); return max(r["xip"]["max_boxcar_over_white"],r["xim"]["max_boxcar_over_white"]), min(r["xip"]["const_pte"],r["xim"]["const_pte"])
cases=[("real data",rp,rm)]
for e in (0.05,0.1,0.15,0.2,0.3): cases.append((f"tilt {e}",rp*(1+e*ltf),rm*(1+e*ltf)))
for a_ in (0.05,0.1,0.2,0.3):
    b=1+a_*np.exp(-0.5*((np.log(th)-np.log(20))/0.3)**2); cases.append((f"bump {a_} at 20'",rp*b,rm*b))
for a_ in (0.05,0.1,0.2):
    b=1+a_*(th<5); cases.append((f"step {a_} below 5'",rp*b,rm*b))
for f in (0.05,0.1,0.2): cases.append((f"extra white noise {f} sig",rp+f*asc["sigma_xip"]*rng.standard_normal(rp.shape),rm+f*asc["sigma_xim"]*rng.standard_normal(rm.shape)))
print(f"\n{'case':26s} max_ratio  const_pte | caught at 0.81 / 1.00 / 1.33 (ratio criterion only)")
for n,a,b in cases:
    m,p=mr(a,b); print(f"{n:26s} {m:8.3f}  {p:9.4f} | {m>0.81!s:>5} {m>1.0!s:>5} {m>1.33!s:>5}")
