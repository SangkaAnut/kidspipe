import numpy as np
from scipy.special import jv
from scipy.interpolate import CubicSpline
arc=np.pi/180/60
tmin,tmax=0.5*arc,300*arc; dx=0.5
edges=100*15**(np.arange(9)/8)
def Tap(th):
    x=np.log(th); xl=np.log(tmin); xu=np.log(tmax)
    r=np.zeros_like(th)
    m=(xl-dx/2<=x)&(x<xl+dx/2); r[m]=np.cos(np.pi/2*(x[m]-(xl+dx/2))/dx)**2
    m=(xl+dx/2<=x)&(x<xu-dx/2); r[m]=1
    m=(xu-dx/2<=x)&(x<xu+dx/2); r[m]=np.cos(np.pi/2*(x[m]-(xu-dx/2))/dx)**2
    return r
def g(mu,th,lo,hi):
    xu=hi*th; xl=lo*th
    if mu==0: return (hi*jv(1,xu)-lo*jv(1,xl))/th
    Gm=lambda x:(x-8/x)*jv(1,x)-8*jv(2,x)
    return (Gm(xu)-Gm(xl))/th**2
za,zb=np.log(tmin)-dx/2,np.log(tmax)+dx/2
nch=400; xg,wg=np.polynomial.legendre.leggauss(30)
zs=[];ws=[]
for c in range(nch):
    a=za+(zb-za)*c/nch;b=za+(zb-za)*(c+1)/nch
    zs.append(0.5*(b-a)*xg+0.5*(a+b)); ws.append(0.5*(b-a)*wg)
zs=np.concatenate(zs);ws=np.concatenate(ws); th=np.exp(zs); wt=ws*th**2*Tap(th)
def WE(l,lo,hi):
    g0=g(0,th,lo,hi); g4=g(4,th,lo,hi)
    out=np.empty(len(l))
    for i,x in enumerate(l):
        out[i]=0.5*(np.sum(wt*jv(0,x*th)*g0)+np.sum(wt*jv(4,x*th)*g4))
    return out
nodes=np.exp(np.log(0.1)+np.log(1e5)/999*np.arange(1000))
for b in (0,3,7):
    lo,hi=edges[b],edges[b+1]
    sel=(nodes>lo/6)&(nodes<hi*6)
    # need whole table for natural spline, but local effect: compute nodes in a wide window and spline only there
    idx=np.where(sel)[0]; idx=np.arange(max(idx[0]-5,0),min(idx[-1]+6,1000))
    Wn=WE(nodes[idx],lo,hi)
    cs=CubicSpline(np.log(nodes[idx]),Wn,bc_type='natural')
    lf=np.linspace(lo/5,hi*5,3000); Wf=WE(lf,lo,hi); Ws=cs(np.log(lf))
    # toy BP: C=l^-1.0 *exp? use l^-1.2
    C=lf**-1.2
    Nint=np.log(hi/lo)
    bp_f=np.trapz(lf*C*Wf,lf)/Nint; bp_s=np.trapz(lf*C*Ws,lf)/Nint
    inb=(lf>lo)&(lf<hi)
    print(f"band {b+1} [{lo:.1f},{hi:.1f}] table step at hi {hi*0.01153:.1f}; max|dW| {np.max(abs(Wf-Ws)):.2e} (peak W {np.max(abs(Wf)):.3f}); in-band max {np.max(abs(Wf-Ws)[inb]):.2e}; BP direct {bp_f:.6e} spline {bp_s:.6e} rel {(bp_s/bp_f-1):.2e}")
