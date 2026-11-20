#!/usr/bin/env python3
"""Independent, minimal implementation of the KV450 xi+/- likelihood following
docs/MASTER_MATH_SPEC.md (NOT kcap code).  Purpose: confirm that the written
specification is complete enough to reproduce a sensible chi^2 against the
KV450 data shipped in the reference clone.  Requires numpy, scipy, camb.

Usage: OMP_NUM_THREADS=4 python3 kv450_spec_crosscheck.py [--scan]
"""
import sys, time
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d, CubicSpline
from scipy.special import loggamma
import camb

REF = '/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap/data/KV450/'
C_KMS = 299792.4580           # value hard-coded in project_2d.py
C1RHOCRIT = 0.0138731         # linear_alignments.py
trapz = getattr(np, 'trapezoid', None) or np.trapz

# ------------------------------------------------------------------ 1. CAMB
def boltzmann(omch2, ombh2, h, ns, ln1e10As, A_bary=3.0, mnu=0.06, halofit='mead'):
    p = camb.CAMBparams(WantCls=False)
    p.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0.0)
    p.InitPower = camb.initialpower.InitialPowerLaw()
    p.InitPower.set_params(ns=ns, As=np.exp(ln1e10As)*1e-10)
    p.Reion = camb.reionization.TanhReionization(); p.Reion.Reionization = False
    p.NonLinearModel = camb.nonlinear.Halofit(halofit_version=halofit,
                         HMCode_A_baryon=A_bary, HMCode_eta_baryon=0.98-0.12*A_bary)
    z = np.concatenate([np.linspace(0, 2, 100, endpoint=False), np.linspace(2, 6, 50)])
    p.set_matter_power(redshifts=list(z[::-1]), kmax=20.0, nonlinear=True)
    p.NonLinear = camb.model.NonLinear_both
    r = camb.get_results(p)
    k, zz, pnl = r.get_nonlinear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    k2, z2, plin = r.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    zb = np.linspace(0, 6, 6000)
    chi_mpc = r.comoving_radial_distance(zb)          # Mpc
    return dict(k=k, z=zz, pnl=pnl, plin=plin, zb=zb, chi=chi_mpc*h, omega_m=p.omegam, sigma8=r.get_sigma8()[-1])

def extrapolate(k, P, kmax=500.0, npoint=3, nmax=200):
    """log-log straight line through last npoint nodes, per z row; nmax-1 new nodes."""
    x = np.log(k); xn = np.linspace(x[-1], np.log(kmax), nmax)[1:]
    out = np.empty((P.shape[0], len(k)+len(xn)))
    for iz in range(P.shape[0]):
        b, a = np.polyfit(x[-npoint:], np.log(P[iz, -npoint:]), 1)
        out[iz] = np.concatenate([P[iz], np.exp(a + b*xn)])
    return np.concatenate([k, np.exp(xn)]), out

# --------------------------------------------------------------------- 2. n(z)
def load_nz():
    names = ['z0.1t0.3', 'z0.3t0.5', 'z0.5t0.7', 'z0.7t0.9', 'z0.9t1.2']
    ns = []
    for nm in names:
        d = np.loadtxt(REF+f'nofz/Nz_DIR_{nm}.asc')
        z = d[:, 0] + 0.5*(d[1, 0]-d[0, 0]); n = d[:, 1]
        z = np.concatenate([[0.0], z]); n = np.concatenate([[0.0], n])
        ns.append(n/trapz(n, z))
    return z, np.array(ns)

def shift_nz(z, n, delta):
    out = np.empty_like(n)
    for i in range(n.shape[0]):
        f = interp1d(z, n[i], kind='linear', bounds_error=False, fill_value=0.0)
        m = f(z-delta[i]); out[i] = m/trapz(m, z)
    return out

# ------------------------------------------------------ 3. kernels + Limber
def limber_cl(B, zn, nz, A_IA, ells, nzgrid=2000):
    h_chi = B['chi']; zb = B['zb']; Om = B['omega_m']
    s = 1.5*(100.0/C_KMS)**2*Om                         # (h/Mpc)^2
    zg = np.linspace(0, 6, nzgrid)
    chi = np.interp(zg, zb, h_chi)                      # Mpc/h
    a = 1.0/(1+zg)
    # NLA (bk_corrected): D(z)=sqrt(Pnl(kmin,z)/Pnl(kmin,0)) on the CAMB (z,k) grid
    D = np.sqrt(B['pnl'][:, 0]/B['pnl'][0, 0])
    Dg = np.interp(zg, B['z'], D)
    F = -A_IA*C1RHOCRIT*Om/Dg                           # P_GI=F Pnl, P_II=F^2 Pnl
    Q = []
    for i in range(nz.shape[0]):
        n = np.interp(zg, zn, nz[i], left=0.0, right=0.0)
        n = n/trapz(n, zg)
        # A(z)=int_z^inf n dz', B(z)=int_z^inf n/chi dz'  (reverse cumulative trapezoid)
        def rc(y):
            c = np.concatenate([[0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(zg))])
            return c[-1]-c
        Aint = rc(n)
        Bint = rc(np.where(chi > 0, n/np.maximum(chi, 1e-30), 0.0))
        G = Aint - chi*Bint
        W = (chi/a)*G                                   # Mpc/h per unit z? -> convert below
        # kcap W is built with n(chi) dchi = n(z) dz  => same G; chi-space measure is dchi
        # N(chi) = n(z) dz/dchi
        dchi_dz = np.gradient(chi, zg)
        N = n/dchi_dz
        Q.append(s*W + F*N)                             # combined (GG+GI+II) kernel
    Q = np.array(Q)
    # P(k,z) interpolation in (z, ln k) of ln P
    k, P = extrapolate(B['k'], B['pnl'])
    spl = RectBivariateSpline(B['z'], np.log(k), np.log(P), kx=3, ky=3)
    kmin, kmaxv = k[0], k[-1]
    dchi = np.gradient(chi, zg)
    Cl = np.zeros((nz.shape[0], nz.shape[0], len(ells)))
    chi_pos = chi[1:]; zg_pos = zg[1:]
    for il, l in enumerate(ells):
        kk = (l+0.5)/chi_pos
        ok = (kk >= kmin) & (kk <= kmaxv)
        lnP = np.full(chi_pos.shape, -np.inf)
        lnP[ok] = spl.ev(zg_pos[ok], np.log(kk[ok]))
        Pk = np.where(ok, np.exp(lnP), 0.0)
        wgt = Pk*dchi[1:]/chi_pos**2
        M = Q[:, 1:]*wgt                                # (nbin, nchi)
        Cl[:, :, il] = M @ Q[:, 1:].T * 1.0
    # trapezoid weight: sum ~ integral dz; we used dz-step implicitly -> multiply dz
    Cl *= (zg[1]-zg[0])
    return Cl

# ------------------------------------------------ 4. nicaea-style Hankel
def hankel_xi(ell_in, cl_in, N=1024, lmin=1e-4, lmax=5e6, ln2=0.69314718):
    dlnl = (np.log(lmax)-np.log(lmin))/(N-1); lnrc = 0.5*(np.log(lmax)+np.log(lmin)); nc = N//2+1
    i = np.arange(N); l = np.exp(lnrc+(i-nc)*dlnl)
    lo, hi = np.log(ell_in[0]), np.log(ell_in[-1])
    out = {}
    for mu in (0, 4):
        res = []
        for cl in cl_in:
            lc = np.log(l)
            if np.all(cl > 0):
                lnc = np.interp(lc, np.log(ell_in), np.log(cl))
                lnc = np.where(lc < lo, np.log(cl[0])+1.0*(lc-lo), lnc)
                lnc = np.where(lc > hi, np.log(cl[-1])-3.0*(lc-hi), lnc)
                C = np.exp(lnc)
            else:
                C = np.interp(lc, np.log(ell_in), cl)       # constant extrapolation (EPS quirk)
            F = np.fft.rfft(l*C); kf = 2*np.pi*np.arange(N//2+1)/(dlnl*N)
            kern = np.exp(1j*kf*ln2)*np.exp(loggamma(0.5*(1+mu)+0.5j*kf)-loggamma(0.5*(1+mu)-0.5j*kf))
            cv = F*kern; cv[0] = cv[0].real; cv[N//2] = cv[N//2].real
            r = np.fft.irfft(cv, n=N)*N
            t = np.exp((nc-i)*dlnl-lnrc)
            xi = np.empty(N); xi[N-i-1] = r/(t*N*2*np.pi)
            res.append(xi)
        out[mu] = np.array(res)
    theta = 1.0/l[::-1]
    return theta, out[0], out[4]

def bin_theory(theta, xi, edges):
    res = np.zeros(len(edges)-1)
    for b in range(len(edges)-1):
        m = (theta >= edges[b]) & (theta < edges[b+1])
        res[b] = trapz(theta[m]*xi[m], theta[m])/trapz(theta[m], theta[m])
    return res

# ------------------------------------------------------ 5. likelihood
class KV450Like:
    def __init__(self):
        self.edges = np.exp(np.linspace(np.log(0.5), np.log(300.0), 10))*np.pi/180/60
        dat = np.loadtxt(REF+'data_vector/KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat')
        cov = np.loadtxt(REF+'covariance/cov_analytic_montepython_mcorr.txt')
        self.cut_p, self.cut_m = [7, 8], [0, 1, 2]
        n, nt = 5, 9
        idx = []
        for p in range(15):
            idx += [p*18+t for t in self.cut_p] + [p*18+nt+t for t in self.cut_m]
        keep = np.setdiff1d(np.arange(270), idx)
        self.inv = np.linalg.inv(cov[np.ix_(keep, keep)])
        m = np.array([-0.0128, -0.0104, -0.0114, 0.0072, 0.0061])
        x = []
        for i in range(n):
            for j in range(i, n):
                col = 1 + (i*(2*n-i+1))//2 + (j-i)
                f = 1.0/((1+m[i])*(1+m[j]))
                xp, xm = dat[:9, col]*f, dat[9:, col]*f
                x.append(np.concatenate([np.delete(xp, self.cut_p), np.delete(xm, self.cut_m)]))
        self.x = np.concatenate(x)
        c = np.loadtxt(REF+'systematics/KV450_ALL_c12_treecorr.out', usecols=[3, 4])
        self.cp, self.cm = np.delete(c[:, 0], self.cut_p), np.delete(c[:, 1], self.cut_m)
        self.zn, self.nz = load_nz()
        self.ells = np.logspace(np.log10(0.1), np.log10(5e5), 400)

    def model(self, omch2, ombh2, h, ns, ln1e10As, A_bary=3.0, A_IA=0.0, dz=np.zeros(5), delta_c=0.0, A_c=1.0, B=None):
        if B is None:
            B = boltzmann(omch2, ombh2, h, ns, ln1e10As, A_bary)
        nz = shift_nz(self.zn, self.nz, dz)
        Cl = limber_cl(B, self.zn, nz, A_IA, self.ells)
        mu = []
        cache = {}
        for i in range(5):
            for j in range(i, 5):
                cl = Cl[j, i]
                theta, xp, xm = hankel_xi(self.ells, [cl])
                bp = bin_theory(theta, xp[0], self.edges); bm = bin_theory(theta, xm[0], self.edges)
                bp = np.delete(bp, self.cut_p); bm = np.delete(bm, self.cut_m)
                bp = bp + delta_c**2 + A_c**2*self.cp
                bm = bm + A_c**2*self.cm
                mu.append(np.concatenate([bp, bm]))
        return np.concatenate(mu), B

    def chi2(self, *a, **k):
        mu, B = self.model(*a, **k)
        d = self.x-mu
        return float(d @ self.inv @ d), mu, B

if __name__ == '__main__':
    L = KV450Like()
    t = time.time()
    # KV450-like point (ombh2, h, ns fixed; scan amplitude)
    base = dict(omch2=0.12, ombh2=0.0222, h=0.70, ns=0.97, A_bary=3.13)
    print('theory data points:', len(L.x))
    best = None
    scan = [2.6, 2.8, 3.0, 3.2, 3.4, 3.6] if '--scan' in sys.argv else [3.0]
    for lnA in scan:
        c2, mu, B = L.chi2(base['omch2'], base['ombh2'], base['h'], base['ns'], lnA, base['A_bary'])
        print(f'ln1e10As={lnA:.2f} sigma8={B["sigma8"]:.4f} S8={B["sigma8"]*np.sqrt(B["omega_m"]/0.3):.4f} chi2={c2:.2f}  ({time.time()-t:.1f}s)')
