"""(c) tree-structure sensitivity test on a bin-6 subsample: permutation / float32 positions / bin_slop 0."""
import sys, time, json, numpy as np, treecorr
N=int(sys.argv[1]); SLOPS=[float(x) for x in sys.argv[2].split(",")]; OUT=sys.argv[3]; NT=6
d=np.load("/mnt/d/HI-gamma-correlation/pipeline/00_extracts/outputs/bin6.npz")
sel=np.sort(np.random.default_rng(12345).choice(len(d["ra"]),N,replace=False))
ra,dec,e1,e2,w=[d[k][sel].astype("f8") for k in ("ra","dec","e1","e2","weight")]
ra32,dec32=ra.astype("f4").astype("f8"),dec.astype("f4").astype("f8")
p1=np.random.default_rng(1).permutation(N); p2=np.random.default_rng(2).permutation(N)
def run(slop,r,dc,perm=None):
    ix=slice(None) if perm is None else perm
    c=treecorr.Catalog(ra=r[ix],dec=dc[ix],ra_units="deg",dec_units="deg",g1=e1[ix],g2=e2[ix],w=w[ix])
    g=treecorr.GGCorrelation(nbins=1000,bin_type="Log",min_sep=2.,max_sep=300.,sep_units="arcmin",bin_slop=slop,num_threads=NT,verbose=0)
    t=time.time(); g.process(c); print("slop",slop,"%.1fs"%(time.time()-t),flush=True)
    return dict(xip=g.xip.copy(),xim=g.xim.copy(),sp=np.sqrt(g.varxip),sm=np.sqrt(g.varxim),npairs=g.npairs.copy(),weight=g.weight.copy(),r=g.rnom.copy())
res={}
for s in SLOPS:
    res[f"{s}_A_f64"]=run(s,ra,dec); res[f"{s}_B_f32"]=run(s,ra32,dec32)
    res[f"{s}_C_f64_perm1"]=run(s,ra,dec,p1); res[f"{s}_D_f32_perm1"]=run(s,ra32,dec32,p1); res[f"{s}_E_f64_perm2"]=run(s,ra,dec,p2)
np.savez(OUT,**{f"{k}__{q}":v for k,r in res.items() for q,v in r.items()})
