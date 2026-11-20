import numpy as np
from scipy.special import jv
arc=np.pi/180/60
tmin,tmax=0.5*arc,300*arc; dx=0.5
lo,hi=100.0,100.0*15**(1/8)
def Tap(th):
    x=np.log(th); xl=np.log(tmin); xu=np.log(tmax)
    r=np.zeros_like(th)
    m=(xl-dx/2<=x)&(x<xl+dx/2); r[m]=np.cos(np.pi/2*(x[m]-(xl+dx/2))/dx)**2
    m=(xl+dx/2<=x)&(x<xu-dx/2); r[m]=1
    m=(xu-dx/2<=x)&(x<xu+dx/2); r[m]=np.cos(np.pi/2*(x[m]-(xu-dx/2))/dx)**2
    return r
def g(mu,th):
    xu=hi*th; xl=lo*th
    if mu==0: G=lambda x,l: l*jv(1,x)    # (1/theta)*l*J1(l theta)
    if mu==0: return (hi*jv(1,xu)-lo*jv(1,xl))/th
    if mu==2: return -(xu*jv(1,xu)-xl*jv(1,xl)+2*jv(0,xu)-2*jv(0,xl))/th**2
    if mu==4:
        Gm=lambda x:(x-8/x)*jv(1,x)-8*jv(2,x)
        return (Gm(xu)-Gm(xl))/th**2
# GL in z=ln theta over [ln(tmin)-dx/2, ln(tmax)+dx/2]
za,zb=np.log(tmin)-dx/2,np.log(tmax)+dx/2
nch=200; xg,wg=np.polynomial.legendre.leggauss(40)
zs=[];ws=[]
for c in range(nch):
    a=za+(zb-za)*c/nch;b=za+(zb-za)*(c+1)/nch
    zs.append(0.5*(b-a)*xg+0.5*(a+b)); ws.append(0.5*(b-a)*wg)
zs=np.concatenate(zs);ws=np.concatenate(ws); th=np.exp(zs)
wt=ws*th**2*Tap(th)
ref=np.loadtxt('/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap/cosebis/Benjamins_Bandpower/bandpower_filter_output_0_100.0_140.29.dat')
L=ref[::100,0]
for mu in (0,4):
    gg=g(mu,th)
    W=np.array([np.sum(wt*jv(mu,l*th)*gg) for l in L])
    globals()['W%d'%mu]=W
N=np.log(hi/lo)
print("ell      (W0+W4)/2   (W0-W4)/2  ref cols")
for i,l in enumerate(L):
    print("%8.2f %12.5e %12.5e | %s"%(l,(W0[i]+W4[i])/2,(W0[i]-W4[i])/2,' '.join('%12.5e'%v for v in ref[i*100,1:])))
