# NEEDS EXTERNAL INPUT (not in this repository): the earlier pipeline's outputs under /mnt/d/HI-gamma-correlation/pipeline/... (override the location where the script allows it).
# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""(a) ours vs theirs in sigma units, per fine bin and 9-bin official sigma."""
import sys, json, numpy as np
import pathlib; _R=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(_R/"scripts")); sys.path.insert(0,str(_R/"src"))
import dr5_measure_xi_unordered as U
from scipy.stats import chi2
ctx=U.load_context(); out={}
for i in range(1,7):
    o=np.load(_R/f"runs/dr5_measure/xi_unordered/b1.5/Bin{i}_Bin{i}.npz")
    t=np.load(U.THEIRS/f"xipm_bin{i}_bin{i}.npz"); mf=float(t['m_factor'])
    blk=U.pair_blocks(ctx,(i,i)); r={}
    r['np_ratio']=float(o['npairs'].sum()/t['npairs'].sum()); r['w_ratio']=float(o['weight'].sum()/t['weight'].sum())
    r['sig_ratio']=float(np.median(np.sqrt(o['varxi'])/t['sigma_xip']))
    for k in ('xip','xim'):
        sig=t['sigma_'+k]; d=(o[k]-t[k+'_raw']); z=d/sig; zo=d/np.sqrt(o['varxi'])
        n=t['npairs']; zz=z
        # lag-1 autocorr, mean, std, weighted mean
        d9=U.rebin9(o['r_nom'],o['weight'],o[k])-U.rebin9(o['r_nom'],o['weight'],t[k+'_raw'])
        # 9-bin: rebin of difference using weights (use theirs weights)
        d9=U.rebin9(t['r_nom'],t['weight'],d)
        z9=d9/blk['sig9'][k]/ (1.0)
        # normalise by m: official sigma is for m-corrected; raw sigma = sig9*mf
        z9=d9/(blk['sig9'][k]*mf)
        # z-std by scale band
        bands={}
        for lo,hi in [(2,6),(6,20),(20,60),(60,300)]:
            m=(t['r_nom']>=lo)&(t['r_nom']<hi); bands[f'{lo}-{hi}']=[float(zz[m].mean()),float(zz[m].std()),int(m.sum())]
        # cumulative wandering: mean over 10 consecutive bins (coherence): std of block means vs expect std/sqrt(10)
        bm=zz[:1000].reshape(100,10).mean(1)
        r[k]=dict(mean_z=float(zz.mean()),std_z=float(zz.std()),rms_z=float(np.sqrt((zz**2).mean())),max_abs_z=float(abs(zz).max()),
                  lag1=float(np.corrcoef(zz[:-1],zz[1:])[0,1]),lag5=float(np.corrcoef(zz[:-5],zz[5:])[0,1]),
                  blockmean10_std_over_expect=float(bm.std()/(zz.std()/np.sqrt(10))),
                  bands=bands,frac_gt2=float((abs(zz)>2).mean()),
                  rms_z_ownvar=float(np.sqrt((zo**2).mean())),
                  z9=[float(x) for x in z9], rms_z9=float(np.sqrt((z9**2).mean())),
                  mean_z9=float(z9.mean()), signed_frac_pos=float((zz>0).mean()),
                  chi2_9_total=float((z9**2).sum()))
    out[i]=r
json.dump(out,open("/tmp/claude-1000/-home-anut/8ecc82c7-22dd-490c-b045-63f93bce2233/scratchpad/a.json","w"),indent=1)
for i,r in out.items():
    print(i,'npr %.6f wr %.6f sigratio %.4f'%(r['np_ratio'],r['w_ratio'],r['sig_ratio']))
    for k in ('xip','xim'):
        q=r[k]; print('  ',k,'mean %.3f std %.3f rms %.3f max %.2f lag1 %.3f lag5 %.3f blk %.2f >2: %.3f pos %.3f | rms(own var) %.3f | z9 rms %.3f mean %.3f'%(q['mean_z'],q['std_z'],q['rms_z'],q['max_abs_z'],q['lag1'],q['lag5'],q['blockmean10_std_over_expect'],q['frac_gt2'],q['signed_frac_pos'],q['rms_z_ownvar'],q['rms_z9'],q['mean_z9']))
        print('     z9',np.round(q['z9'],2),' bands',{b:[round(x,2) for x in v[:2]] for b,v in q['bands'].items()})
