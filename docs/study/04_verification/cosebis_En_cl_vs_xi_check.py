import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.special import j0
import nicaea_logfft_hankel_check as H
arc=np.pi/180/60
tmin,tmax=0.5*arc,300*arc
D=np.log(tmax/tmin)
roots={};norm={}
for ln in open('/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap/cosebis/TLogsRootsAndNorms/Root_0.50-300.00.table'):
    s=ln.split(); roots[int(s[0])]=np.array(s[1:],float)
for ln in open('/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap/cosebis/TLogsRootsAndNorms/Normalization_0.50-300.00.table'):
    s=ln.split(); norm[int(s[0])]=float(s[1])
def Tp(n,z): return norm[n]*np.prod([z-r for r in roots[n]],axis=0) if np.ndim(z)==0 else norm[n]*np.prod([z-r for r in roots[n]],axis=0)
def Tp(n,z):
    z=np.asarray(z,float); out=np.full(z.shape,norm[n])
    for r in roots[n]: out=out*(z-r)
    return out
xg,wg=np.polynomial.legendre.leggauss(60)
def Tm(n,z):
    z=np.atleast_1d(z); res=np.empty_like(z)
    for k,zz in enumerate(z):
        y=0.5*zz*(xg+1); w=0.5*zz*wg
        res[k]=Tp(n,zz)+np.sum(w*4*Tp(n,y)*(np.exp(2*(y-zz))-3*np.exp(4*(y-zz))))
    return res
C=lambda l: 1e-9*l/(1+(l/300.)**4)
# xi+/- on nicaea grid
xip=H.hank(0,C); xim=H.hank(4,C)
theta=1/H.l[::-1]
for n in (1,2,3):
    # (b) from xi: 0.5 * int dtheta theta [T+ xi+ + T- xi-]
    z=np.linspace(0,D,20001); th=tmin*np.exp(z)
    lt=np.log(theta)
    xp=np.interp(np.log(th),lt,xip); xm=np.interp(np.log(th),lt,xim)
    integrand=th*th*(Tp(n,z)*xp+Tm(n,z)*xm)     # dtheta*theta = theta^2 dz
    Eb=0.5*np.trapz(integrand,z)
    # (a) from C: 1/2pi int dl l W_n C
    zz,ww=np.polynomial.legendre.leggauss(3000); 
    # split z into chunks for accuracy
    nch=20; zs=[];ws=[]
    for c in range(nch):
        a,b=D*c/nch,D*(c+1)/nch
        zs.append(0.5*(b-a)*zz+0.5*(a+b)); ws.append(0.5*(b-a)*ww)
    zs=np.concatenate(zs);ws=np.concatenate(ws)
    ths=tmin*np.exp(zs); wt=ws*ths**2*Tp(n,zs)
    lg=np.linspace(1,6e4,30001); dl=lg[1]-lg[0]
    W=np.array([np.sum(wt*j0(L*ths)) for L in lg[::1]]) if False else None
    Wl=np.empty(lg.size)
    for i0 in range(0,lg.size,500):
        L=lg[i0:i0+500]
        Wl[i0:i0+500]=(j0(np.outer(L,ths))*wt).sum(axis=1)
    Ea=np.trapz(lg*Wl*C(lg),lg)/(2*np.pi)
    print(n,Ea,Eb)
