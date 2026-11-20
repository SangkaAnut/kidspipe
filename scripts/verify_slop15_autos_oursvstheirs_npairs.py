# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
import sys, numpy as np
import pathlib; _R=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(_R/"scripts")); import dr5_measure_xi_unordered as U
for i in range(1,7):
    o=np.load(_R/f"runs/dr5_measure/xi_unordered/b1.5/Bin{i}_Bin{i}.npz"); t=np.load(U.THEIRS/f"xipm_bin{i}_bin{i}.npz")
    dn=(o['npairs']-t['npairs']); rel=dn/t['npairs']; dw=(o['weight']-t['weight'])/t['weight']
    # predicted xi change from pair migration only? corr of dxi with dweight
    d=o['xip']-t['xip_raw']
    print(i,"sum dn %.0f"%dn.sum(),"frac bins npairs differ %.3f"%(dn!=0).mean(),"rms rel dn %.2e"%np.sqrt((rel**2).mean()),"rms rel dn 2-6' %.2e"%np.sqrt((rel[:130]**2).mean()),"lag1 dn %.2f"%np.corrcoef(dn[:-1],dn[1:])[0,1],"corr(dxip,dw) %.2f"%np.corrcoef(d,dw)[0,1])
