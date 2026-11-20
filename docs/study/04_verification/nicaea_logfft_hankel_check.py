import numpy as np
from scipy.special import loggamma, jv
from scipy.integrate import quad
N=1024; lmin=1e-4; lmax=5e6
dlnl=(np.log(lmax)-np.log(lmin))/(N-1)
lnrc=0.5*(np.log(lmax)+np.log(lmin)); nc=N//2+1
i=np.arange(N)
l=np.exp(lnrc+(i-nc)*dlnl)
def C(l): return 1e-9*l/(1+(l/800.)**3.5)   # toy: ~l at low, l^-2.5 at high (>0)
def hank(mu,Cfun,q=0.0,norm=2*np.pi):
    lP=l*Cfun(l)
    F=np.fft.rfft(lP)
    k=2*np.pi*np.arange(N//2+1)/(dlnl*N)
    a1=0.5*(1+mu+q)+0.5j*k
    a2=0.5*(1+mu-q)-0.5j*k
    kern=2**q*np.exp(1j*k*np.log(2))*np.exp(loggamma(a1)-loggamma(a2))
    conv=F*kern
    conv[0]=conv[0].real; conv[N//2]=conv[N//2].real
    r=np.fft.irfft(conv,n=N)*N   # FFTW c2r unnormalised
    t=np.exp((nc-i)*dlnl-lnrc)
    xi=np.empty(N); xi[N-i-1]=r/(t*N*norm)
    return xi
theta=1/l[::-1]    # theta_j = 1/l_{N-1-j}
for mu in (0,4):
    xi=hank(mu,C)
    for th_arcmin in (1,10,100):
        th=th_arcmin/60*np.pi/180
        j=np.argmin(abs(np.log(theta/th)))
        f=lambda x: x*C(x)*jv(mu,x*theta[j])/(2*np.pi)
        # integrate by oscillatory-safe splitting
        xs=np.concatenate(([0],np.logspace(-3,7,60000)))
        ref=np.trapz(f(xs),xs)
        print(mu,th_arcmin,theta[j],xi[j],ref)
