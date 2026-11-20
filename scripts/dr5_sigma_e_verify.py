# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Independent verification of sigma_e (does not import dr5_sigma_e.py). Run with OMP<=4."""
import numpy as np, json
from astropy.io import fits
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402
HDR=str(paths.official_dir()/'KiDS_Legacy_xipm.fits')
h=fits.open(HDR)['NZ_SOURCE'].header
sig=np.array([h['SIG_E_%d'%i] for i in range(1,7)])
m=np.array([-0.023,-0.016,-0.011,0.020,0.030,0.045])   # Wright+25 Table 1 last column (checked vs paper text)
res={k:[] for k in ['C9_div','C9_nom','C9_mul','C9_div_patchc','lin_div','C9_unw_div','implied_m','wmean_e1','wmean_e2','C9_div_float32']}
for b in range(6):
    d=np.load(str(paths.dr5_work_dir())+'/bin%d.npz'%b)
    e1=d['e1'].astype(np.float64); e2=d['e2'].astype(np.float64); w=d['w'].astype(np.float64); N=d['north']
    ex=np.load(str(paths.dr5_work_dir())+'/extra/bin%d.npz'%b)
    assert ex['TOMOBIN'][0]==b+1 and np.all(ex['TOMOBIN']==b+1)
    assert np.allclose(ex['weight'],w,rtol=1e-6)          # same weight column, same row order
    w2=w*w; S2=w2.sum()
    c9=np.sqrt((w2*(e1**2+e2**2)).sum()/(2*S2))          # per-component, no m
    res['C9_nom'].append(c9/sig[b]); res['C9_div'].append(c9/(1+m[b])/sig[b]); res['C9_mul'].append(c9*(1+m[b])/sig[b])
    res['implied_m'].append(c9/sig[b]-1)
    res['wmean_e1'].append((w*e1).sum()/w.sum()); res['wmean_e2'].append((w*e2).sum()/w.sum())
    # alt: subtract weighted-mean c per patch and per component
    s=0
    for msk in (N,~N):
        a=e1[msk]-np.average(e1[msk],weights=w[msk]); c=e2[msk]-np.average(e2[msk],weights=w[msk])
        s+=(w2[msk]*(a**2+c**2)).sum()
    res['C9_div_patchc'].append(np.sqrt(s/(2*S2))/(1+m[b])/sig[b])
    res['lin_div'].append(np.sqrt((w*(e1**2+e2**2)).sum()/(2*w.sum()))/(1+m[b])/sig[b])
    res['C9_unw_div'].append(np.sqrt((e1**2+e2**2).mean()/2)/(1+m[b])/sig[b])
    f1=d['e1'];f2=d['e2'];fw=d['w']   # float32 products accumulated in float64
    res['C9_div_float32'].append(np.sqrt(((fw.astype(np.float64)**2)*(f1.astype(np.float64)**2+f2.astype(np.float64)**2)).sum()/(2*(fw.astype(np.float64)**2).sum()))/(1+m[b])/sig[b])
    print('bin',b+1,{k:round(float(v[-1]),5) for k,v in res.items() if v},flush=True)
# also: implied m vs table
print('implied m',[round(x,5) for x in res['implied_m']])
print('table m  ',m.tolist())
# n_eff check: (1+m) cancels in Eq C.12 for uniform m
json.dump({k:[float(x) for x in v] for k,v in res.items()}|{'SIG_E':sig.tolist(),'m':m.tolist()},open(paths.runs_dir()/'dr5_measure/sigma_e/verify.json','w'),indent=1)
