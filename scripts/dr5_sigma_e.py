# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""sigma_e gate: every defensible variant vs official SIG_E_i. Reads only cached npz + release FITS header."""
import json, itertools, numpy as np
from astropy.io import fits
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402
W=str(paths.dr5_work_dir())+'/'
REL=str(paths.official_dir()/'KiDS_Legacy_xipm.fits')
OUT=str(paths.runs_dir()/'dr5_measure/sigma_e/sigma_e.json')
M=[-0.023,-0.016,-0.011,0.020,0.030,0.045]   # m_gold, Wright+25 Table 1 (x1e-2: -2.3,-1.6,-1.1,2.0,3.0,4.5)
A_EFF=967.4; A_MOS=1014.03
h=fits.getheader(REL,'NZ_SOURCE')
SIG=[h['SIG_E_%d'%i] for i in range(1,7)]; NGAL=[h['NGAL_%d'%i] for i in range(1,7)]
PAPER_SIG=[0.28,0.27,0.29,0.26,0.28,0.30]
res={'SIG_E':SIG,'NGAL':NGAL,'m_gold':M,'paper_sigma_e_rounded':PAPER_SIG,'bins':{}}
for b in range(6):
    d=np.load(W+'bin%d.npz'%b); x=np.load(W+'extra/bin%d.npz'%b)
    e1=d['e1'].astype(np.float64); e2=d['e2'].astype(np.float64); north=d['north']
    wts={'weight':d['w'].astype(np.float64),'shear':x['shear_weight_only'],'gold':x['gold_weight_only'],
         'shear*gold':x['shear_weight_only']*x['gold_weight_only'],'unit':np.ones(len(e1))}
    out={}
    # c subtraction variants
    def csub(mode,w):
        a,c=e1.copy(),e2.copy()
        if mode=='none': return a,c
        if mode=='bin':
            a-=np.average(a,weights=w); c-=np.average(c,weights=w); return a,c
        if mode=='patch':
            for m in (north,~north):
                a[m]-=np.average(a[m],weights=w[m]); c[m]-=np.average(c[m],weights=w[m])
            return a,c
    for wn,w in wts.items():
        for cm in ['none','bin','patch']:
            a,c=csub(cm,w); e2s=a*a+c*c; w2=w*w
            S1,S2=w.sum(),w2.sum()
            est={
             'C9_sum w2 |e|2 / sum w2 (/2)':np.sqrt((w2*e2s).sum()/S2/2),
             'sum w |e|2 / sum w (/2)':np.sqrt((w*e2s).sum()/S1/2),
             'sum w2 |e|2 / sum w2 (total, no /2)':np.sqrt((w2*e2s).sum()/S2),
             'sum w2 |e|2 /(sum w)^2 *N (/2)':np.sqrt((w2*e2s).sum()/S1**2*len(w)/2),
             'comp-wise weighted std e1,e2 avg (w)':np.sqrt(0.5*(np.average(a**2,weights=w)+np.average(c**2,weights=w))),
             'unweighted |e|2/2':np.sqrt(e2s.mean()/2),
            }
            for k,v in est.items():
                base=v
                for mn,f in [('m0',lambda m:1.0),('divide_(1+m)',lambda m:1/(1+m)),('multiply_(1+m)',lambda m:(1+m)),
                             ('divide_(1+m)^2 (sig2)',lambda m:1/(1+m)**2)]:
                    key=f'{wn} | c:{cm} | {k} | {mn}'
                    out[key]=base*f(M[b])
    res['bins'][b+1]={'n':int(len(e1)),'variants':out}
    # n_eff check
    w=wts['weight']; res['bins'][b+1]['neff_arcmin2_967.4']=float(w.sum()**2/(w*w).sum()/(A_EFF*3600))
    res['bins'][b+1]['NGAL_x_1014.03/967.4']=NGAL[b]*A_MOS/A_EFF
    print('bin',b+1,'done',flush=True)
keys=list(res['bins'][1]['variants'])
tab=[]
for k in keys:
    r=[res['bins'][b]['variants'][k]/SIG[b-1] for b in range(1,7)]
    tab.append({'variant':k,'ratios':r,'max_abs_dev':max(abs(x-1) for x in r),'mean_ratio':float(np.mean(r))})
tab.sort(key=lambda t:t['max_abs_dev'])
res['ranked']=tab
json.dump(res,open(OUT,'w'),indent=1)
print(len(tab),'variants'); 
for t in tab[:12]: print('%.4f'%t['max_abs_dev'],t['variant'],np.round(t['ratios'],4))
base=[t for t in tab if t['variant']=='weight | c:none | C9_sum w2 |e|2 / sum w2 (/2) | m0'][0]; print('BASE',np.round(base['ratios'],4))

# implied m per bin from Eq. C.9:  (1+m)_implied = sigma_raw(C9, weight, c none)/SIG_E
imp=[]
for b in range(1,7):
    raw=res['bins'][b]['variants']['weight | c:none | C9_sum w2 |e|2 / sum w2 (/2) | m0']
    imp.append(raw/SIG[b-1]-1)
res['implied_m_from_C9']=imp
json.dump(res,open(OUT,'w'),indent=1)
print('implied m',np.round(imp,5),'table m',M)
def row(k): return np.round([res['bins'][b]['variants'][k]/SIG[b-1] for b in range(1,7)],4)
for k in ['weight | c:none | C9_sum w2 |e|2 / sum w2 (/2) | m0','weight | c:none | C9_sum w2 |e|2 / sum w2 (/2) | divide_(1+m)','weight | c:none | C9_sum w2 |e|2 / sum w2 (/2) | multiply_(1+m)','weight | c:none | sum w |e|2 / sum w (/2) | m0','weight | c:none | sum w |e|2 / sum w (/2) | divide_(1+m)','weight | c:none | sum w2 |e|2 /(sum w)^2 *N (/2) | divide_(1+m)','weight | c:none | unweighted |e|2/2 | m0','shear | c:none | C9_sum w2 |e|2 / sum w2 (/2) | m0','gold | c:none | C9_sum w2 |e|2 / sum w2 (/2) | m0']: print(row(k),k)
print([ (b,res['bins'][b]['neff_arcmin2_967.4'],res['bins'][b]['NGAL_x_1014.03/967.4']) for b in range(1,7)])
