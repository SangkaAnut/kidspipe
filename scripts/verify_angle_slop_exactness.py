import os, sys, json, time
os.environ["OMP_NUM_THREADS"]="6"
import numpy as np, treecorr
sys.path.insert(0,"src")
from kidspipe.measure import xi as X
N=int(sys.argv[1]) if len(sys.argv)>1 else 30000
rec=X.load_bin(6)
m=(rec["ra"]>20)&(rec["ra"]<40)&(rec["dec"]>-36)&(rec["dec"]<-20)
idx=np.where(m)[0]; print("region n",idx.size,flush=True)
rng=np.random.default_rng(7); idx=np.sort(rng.choice(idx,N,replace=False))
s={k:rec[k][idx].astype(np.float64) for k in("ra","dec","e1","e2","w")}
cat=X.make_treecorr_cat(s)
NB=1000
res={}
def run(bs,as_,brute=False):
    kw=X.treecorr_kwargs(NB,bs,6,as_)
    if brute: kw["brute"]=True
    t=time.time(); g=treecorr.GGCorrelation(kw); g.process(cat,cat); 
    return dict(xip=g.xip.copy(),xim=g.xim.copy(),np=g.npairs.copy(),w=g.weight.copy(),t=time.time()-t)
R={"brute_tc":run(0.0,None,True),"0,0":run(0.0,0.0),"0,def":run(0.0,None),"1.5,def":run(1.5,None)}
# own numpy brute: ordered pairs i!=j, great-circle bearing at each galaxy
ra=np.deg2rad(s["ra"]);de=np.deg2rad(s["dec"]);g=s["e1"]+1j*s["e2"];w=s["w"]
edges=np.exp(np.linspace(np.log(2.0),np.log(300.0),NB+1))
def acc(sign):
    Sp=np.zeros(NB);Sm=np.zeros(NB);Sw=np.zeros(NB);Sn=np.zeros(NB)
    B=400
    for i0 in range(0,N,B):
        i1=min(N,i0+B)
        r1=ra[i0:i1,None];d1=de[i0:i1,None];r2=ra[None,:];d2=de[None,:]
        dl=r2-r1
        h=np.sin((d2-d1)/2)**2+np.cos(d1)*np.cos(d2)*np.sin(dl/2)**2
        th=np.rad2deg(2*np.sqrt(h))*60  # chord (TreeCorr default Euclidean metric)
        k=np.searchsorted(edges,th,side="right")-1
        ok=(k>=0)&(k<NB)
        ii=np.arange(i0,i1)[:,None]*np.ones((1,N),int); jj=np.ones((i1-i0,1),int)*np.arange(N)[None,:]
        ok&=(ii!=jj)
        # bearing from 1 to 2 and 2 to 1 (north through east)
        b12=np.arctan2(np.sin(dl)*np.cos(d2),np.cos(d1)*np.sin(d2)-np.sin(d1)*np.cos(d2)*np.cos(dl))
        b21=np.arctan2(-np.sin(dl)*np.cos(d1),np.cos(d2)*np.sin(d1)-np.sin(d2)*np.cos(d1)*np.cos(dl))
        # shear (e1 along +RA=east, e2 along NE) -> angle from east axis: phi = pi/2 - bearing
        p1=sign*(np.pi/2-b12); p2=sign*(np.pi/2-b21)
        g1=g[i0:i1,None]*np.exp(-2j*p1); g2=g[None,:]*np.exp(-2j*p2)
        ww=w[i0:i1,None]*w[None,:]
        kk=k[ok]
        Sn+=np.bincount(kk,minlength=NB);Sw+=np.bincount(kk,ww[ok],NB)
        Sp+=np.bincount(kk,(ww*(g[i0:i1,None]*np.conj(g[None,:]) if False else (g1*np.conj(g2))).real)[ok],NB)
        Sm+=np.bincount(kk,(ww*(g1*g2).real)[ok],NB)
    gd=Sw>0
    return dict(xip=np.where(gd,Sp/np.where(gd,Sw,1),0),xim=np.where(gd,Sm/np.where(gd,Sw,1),0),np=Sn,w=Sw)
t=time.time()
cands={sgn:acc(sgn) for sgn in(-1,)}
print("numpy brute time",time.time()-t,flush=True)
b=R["brute_tc"]
for sgn,c in cands.items():
    print("sign",sgn,"xip vs TCbrute",np.abs(c["xip"]-b["xip"]).max(),"xim",np.abs(c["xim"]-b["xim"]).max(),"np",np.abs(c["np"]-b["np"]).max())
best=min(cands,key=lambda s:np.abs(cands[s]["xim"]-b["xim"]).max()); mine=cands[best]
out={"N":N,"sign":best}
for name,r in R.items():
    d={}
    for ref,rn in((mine,"numpy"),(b,"tcbrute")):
        d[rn]=dict(xip=float(np.abs(r["xip"]-ref["xip"]).max()),xim=float(np.abs(r["xim"]-ref["xim"]).max()),
                   np=float(np.abs(r["np"]-ref["np"]).max()),np_nbins_diff=int((r["np"]!=ref["np"]).sum()))
    d["time"]=r["t"]; out[name]=d
out["xip_scale"]=float(np.abs(mine["xip"]).max()); out["np_total"]=float(mine["np"].sum())
print(json.dumps(out,indent=1))
json.dump(out,open("runs/dr5_measure/angle_slop/verify/exactness_a.json","w"),indent=1)
print("tc np total",b["np"].sum(),"mine",mine["np"].sum())
eq=(b["np"]==mine["np"])&(b["np"]>0)
print("bins equal count",eq.sum(),"of",(b["np"]>0).sum())
print("max diff on equal-count bins xip",np.abs(b["xip"]-mine["xip"])[eq].max(),"xim",np.abs(b["xim"]-mine["xim"])[eq].max())
dn=(b["np"]-mine["np"])
print("diff sum",dn.sum(),"first bins",dn[:10],"last",dn[-10:])
print("cum diff by 100bin groups",[dn[i:i+100].sum() for i in range(0,1000,100)])
print("bins with weight diff", np.abs(b["w"]-mine["w"]).max()/mine["w"].max())
