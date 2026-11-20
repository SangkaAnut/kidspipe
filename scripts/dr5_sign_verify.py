"""Independent verification of the 'sign' track (no import of dr5_sign_convention.py).
stages: flat | frame | ggl | pilot"""
import sys, json, time
import numpy as np, treecorr
from pathlib import Path
ROOT = Path("/mnt/d/cloning_kids_Kappa_pipeline"); OUT = ROOT / "runs/dr5_measure/sign_convention/verify"
WORK = Path.home() / "kids_dr5_work"
def save(key, val):
    f = OUT / "verify.json"; d = json.loads(f.read_text()) if f.exists() else {}
    d[key] = val; f.write_text(json.dumps(d, indent=1))

def stage_flat():
    """Flat-plane Gaussian E-mode random field (Kaiser-Squires), isotropic; TreeCorr Euclidean. Table check."""
    rng = np.random.default_rng(777); N = 512; L = 40.0   # box in degrees-like units; use x,y in arcmin*? just units
    k = np.fft.fftfreq(N)[:, None] * 2 * np.pi, np.fft.fftfreq(N)[None, :] * 2 * np.pi
    kx, ky = k[0] * np.ones((1, N)), k[1] * np.ones((N, 1)); k2 = kx**2 + ky**2; k2[0, 0] = 1
    P = (k2 + 0.05**2)**-1.0; P[0, 0] = 0
    kap = np.fft.fft2(rng.normal(size=(N, N))) * np.sqrt(P)
    # gamma(k) = (kx^2-ky^2 + 2i kx ky)/k^2 * kappa(k), axes: array[ix, iy]; x = axis0, y = axis1
    gk = (kx**2 - ky**2 + 2j * kx * ky) / k2 * kap
    g = np.fft.ifft2(gk); g /= g.std()
    n = 150000; ix = rng.integers(0, N, n); iy = rng.integers(0, N, n)
    x = ix + rng.random(n) - .5; y = iy + rng.random(n) - .5; gg = g[ix, iy]
    half = rng.random(n) < .5
    def xi(x, y, g, sel=None):
        c = treecorr.Catalog(x=x, y=y, g1=g.real, g2=g.imag)
        d = treecorr.GGCorrelation(min_sep=2, max_sep=60, nbins=12, bin_slop=0, num_threads=8); d.process(c)
        return d.xip.copy(), d.xim.copy(), d.xim_im.copy()
    def cross(x, y, g):
        a = treecorr.Catalog(x=x[half], y=y[half], g1=g.real[half], g2=g.imag[half]); b = treecorr.Catalog(x=x[~half], y=y[~half], g1=g.real[~half], g2=g.imag[~half])
        d = treecorr.GGCorrelation(min_sep=2, max_sep=60, nbins=12, bin_slop=0, num_threads=8); d.process(a, b)
        return d.xip_im.copy()
    T = {"identity": (x, y, gg), "g->-g": (x, y, -gg), "e2->-e2": (x, y, gg.conj()), "e1->-e1": (x, y, -gg.conj()),
         "phi->-phi (x->-x, shear unchanged)": (-x, y, gg), "mirror x->-x & e2->-e2": (-x, y, gg.conj())}
    base = xi(x, y, gg); bi = cross(x, y, gg); res = {}
    sd = {"xip": base[0].std(), "xim": base[1].std(), "ximim": base[2].std(), "xipim": bi.std()}
    for nm, (X, Y, G) in T.items():
        r = xi(X, Y, G); ci = cross(X, Y, G)
        def cl(a, b, s):
            if np.max(abs(a - b)) < 1e-9 * s + 1e-12: return "invariant"
            if np.max(abs(a + b)) < 1e-9 * s + 1e-12: return "sign flips"
            return f"changes (rms ratio {np.sqrt(np.mean(a**2))/np.sqrt(np.mean(b**2)):.3f})"
        res[nm] = {"Re xi+": cl(r[0], base[0], sd["xip"]), "Re xi-": cl(r[1], base[1], sd["xim"]), "Im xi- (xi_x)": cl(r[2], base[2], sd["ximim"]) if sd["ximim"] > 0 else "n/a",
                   "Im xi+ (cross-cat)": cl(ci, bi, sd["xipim"]), "xim/xim_identity max|diff|/std": float(np.max(abs(r[1] - base[1])) / sd["xim"])}
    save("flat_table", res); print(json.dumps(res, indent=1))

def stage_frame():
    # TreeCorr single pair: source NE of lens, which g2 sign gives gamma_t>0?
    out = {}
    for lab, dra, ddec in [("NE", .01, .01), ("NW", -.01, .01), ("E", .01, 0.), ("N", 0., .01)]:
        for g1, g2 in [(0, .1), (0, -.1), (.1, 0), (-.1, 0)]:
            L = treecorr.Catalog(ra=[10.], dec=[0.], ra_units="deg", dec_units="deg")
            S = treecorr.Catalog(ra=[10 + dra], dec=[ddec], ra_units="deg", dec_units="deg", g1=[g1], g2=[g2])
            ng = treecorr.NGCorrelation(min_sep=.3, max_sep=3., nbins=1, sep_units="arcmin", bin_slop=0)
            ng.process(L, S); out[f"{lab} g=({g1},{g2})"] = float(ng.xi[0])
    print(json.dumps(out, indent=0)); save("treecorr_frame_gt", out)
    # sphere mirror exactness + PairCounter vs TreeCorr on 2x2 deg real patch
    from kidspipe.measure import paircount
    s = np.load(WORK / "bin5.npz"); ra, dec = s["ra"], s["dec"]
    m = (ra > 180) & (ra < 182) & (dec > 0) & (dec < 2)
    if m.sum() < 3000: m = (ra > ra[0]) & (ra < ra[0] + 2) & (dec > dec[0]) & (dec < dec[0] + 2)
    sel = np.where(m)[0][:20000]; print("patch gals", len(sel))
    r, d, e1, e2, w = [np.asarray(s[k][sel], float) for k in ("ra", "dec", "e1", "e2", "w")]
    kw = dict(min_sep=2, max_sep=30, nbins=10, sep_units="arcmin", bin_slop=0, num_threads=8)
    def tc(r, d, e1, e2, **f):
        c = treecorr.Catalog(ra=r, dec=d, ra_units="deg", dec_units="deg", g1=e1, g2=e2, w=w, **f)
        g = treecorr.GGCorrelation(**kw); g.process(c); return g.xip.copy(), g.xim.copy(), g.xim_im.copy()
    a = tc(r, d, e1, e2); b = tc(r, d, e1, e2, flip_g2=True)
    pc = paircount.PairCounter(r, d, e1, e2, w, min_sep=2, max_sep=30, nbins=10)
    xp, xm, xx, npr, ws = pc.run()
    pc2 = paircount.PairCounter(r, d, e1, -e2, w, min_sep=2, max_sep=30, nbins=10); xp2, xm2, xx2, *_ = pc2.run()
    sd = lambda v: v.std()
    res = {"PC(e1,e2) vs TC(no flip): xip, xim, xix vs xim_im": [float(np.max(abs(xp - a[0])) / sd(a[0])), float(np.max(abs(xm - a[1])) / sd(a[1])), float(np.max(abs(xx - a[2])) / sd(a[2])), float(np.max(abs(xx + a[2])) / sd(a[2]))],
           "PC(e1,e2) vs TC(flip_g2)": [float(np.max(abs(xp - b[0])) / sd(b[0])), float(np.max(abs(xm - b[1])) / sd(b[1])), float(np.max(abs(xx - b[2])) / sd(b[2])), float(np.max(abs(xx + b[2])) / sd(b[2]))],
           "PC(e1,-e2) vs TC(no flip)": [float(np.max(abs(xp2 - a[0])) / sd(a[0])), float(np.max(abs(xm2 - a[1])) / sd(a[1]))],
           "n": len(sel), "xim_noflip_over_flip_rms": float(np.sqrt(np.mean(a[1]**2) / np.mean(b[1]**2)))}
    # sphere mirror exactness: ra->-ra with e2->-e2 (exact isometry). Only unambiguous with bin_slop=0.
    c = tc(-r, d, e1, -e2)
    res["sphere mirror(ra->-ra)+e2->-e2 vs identity: xip, xim max|d|/std, xim_im+ ratio"] = [float(np.max(abs(c[0] - a[0])) / sd(a[0])), float(np.max(abs(c[1] - a[1])) / sd(a[1])), float(np.max(abs(c[2] + a[2])) / sd(a[2]))]
    c2 = tc(r, d, -e1, -e2); res["g->-g xim diff"] = float(np.max(abs(c2[1] - a[1])) / sd(a[1]))
    print(json.dumps(res, indent=1)); save("paircounter_vs_treecorr", res)

def stage_ggl(seed=4242, nl=300000, nr=600000, npatch=40):
    import healpy as hp
    rng = np.random.default_rng(seed)
    l = np.load(WORK / "bin0.npz"); s = np.load(WORK / "bin5.npz")
    il = rng.choice(len(l["ra"]), nl, replace=False)
    f8 = lambda x: np.ascontiguousarray(x, dtype="f8")
    sra, sdec = f8(s["ra"]), f8(s["dec"])
    # randoms: uniform within occupied healpix pixels (nside 256) of the SOURCE footprint
    ns = 256; pix = hp.ang2pix(ns, sra, sdec, lonlat=True); cnt = np.bincount(pix, minlength=hp.nside2npix(ns))
    occ = np.where(cnt >= 3)[0]; print("occupied pixels", len(occ), "area deg2", len(occ) * hp.nside2pixarea(ns, degrees=True))
    # simpler: uniform on sphere sampling with rejection against occupied mask
    pts_r = []; pts_d = []; got = 0; mask = np.zeros(hp.nside2npix(ns), bool); mask[occ] = True
    while got < nr:
        u = rng.uniform(-1, 1, 4 * nr); lon = rng.uniform(0, 360, 4 * nr); lat = np.degrees(np.arcsin(u))
        ok = mask[hp.ang2pix(ns, lon, lat, lonlat=True)]
        pts_r.append(lon[ok]); pts_d.append(lat[ok]); got += ok.sum()
    rra = np.concatenate(pts_r)[:nr]; rdec = np.concatenate(pts_d)[:nr]
    res = {"seed": seed, "nl": nl, "nr": nr, "npatch": npatch, "randoms": f"uniform in occupied (>=3 sources) nside-{ns} healpix pixels of the source footprint"}
    t0 = time.time()
    Lc = treecorr.Catalog(ra=f8(l["ra"][il]), dec=f8(l["dec"][il]), ra_units="deg", dec_units="deg", npatch=npatch, rng=np.random.default_rng(seed))
    cen = Lc.patch_centers
    Rc = treecorr.Catalog(ra=rra, dec=rdec, ra_units="deg", dec_units="deg", patch_centers=cen)
    kw = dict(min_sep=1, max_sep=100, nbins=15, sep_units="arcmin", bin_slop=0.1, num_threads=8, var_method="jackknife")
    vs = {"a (as is)": (False, False), "b (flip_g2)": (False, True), "c (flip_g1)": (True, False), "d (both)": (True, True)}
    tmpl = None; res["variants"] = {}
    for nm, (f1, f2) in vs.items():
        Sc = treecorr.Catalog(ra=sra, dec=sdec, g1=f8(s["e1"]), g2=f8(s["e2"]), w=f8(s["w"]), ra_units="deg", dec_units="deg", patch_centers=cen, flip_g1=f1, flip_g2=f2)
        ng = treecorr.NGCorrelation(**kw); ng.process(Lc, Sc)
        rg = treecorr.NGCorrelation(**kw); rg.process(Rc, Sc)
        gt = ng.xi - rg.xi; gx = ng.xi_im - rg.xi_im; r = ng.meanr
        Ct = treecorr.estimate_multi_cov([ng, rg], "jackknife", func=lambda c: c[0].xi - c[1].xi)
        Cx = treecorr.estimate_multi_cov([ng, rg], "jackknife", func=lambda c: c[0].xi_im - c[1].xi_im)
        h = (npatch - 15 - 2) / (npatch - 1)
        def chi(v, C): return float(v @ (np.linalg.inv(C) * h) @ v)
        t = 1 / r; Ci = np.linalg.inv(Ct) * h; A = float(t @ Ci @ gt / (t @ Ci @ t)); sA = float(1 / np.sqrt(t @ Ci @ t))
        diag = np.sqrt(np.diag(Ct))
        res["variants"][nm] = {"gt_chi2": chi(gt, Ct), "gx_chi2": chi(gx, Cx), "amp": A, "amp_sig": sA, "snr": A / sA, "gt_over_sig": (gt / diag).tolist(),
                               "gt": gt.tolist(), "gt_rand": rg.xi.tolist(), "gt_nosub": ng.xi.tolist(), "r": r.tolist(), "sig": diag.tolist(), "gx_over_sig": (gx / np.sqrt(np.diag(Cx))).tolist()}
        print(nm, f"gt chi2 {chi(gt,Ct):.1f}/15 gx chi2 {chi(gx,Cx):.1f}/15 amp {A:.3e}+-{sA:.3e} S/N {A/sA:.2f}  t={time.time()-t0:.0f}s", flush=True)
        save("ggl_independent", res)

def stage_pilot():
    d = np.load(ROOT / "runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz")
    th = np.load("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin6_bin6.npz", allow_pickle=True)
    tf = np.load("/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/e2flip/xipm_bin6_bin6.npz", allow_pickle=True)
    print(list(th.keys())); 
    wt = th["weight"] if "weight" in th else d["weight"]
    out = {}
    def cb(x, w, k=50): return (x * w).reshape(-1, k).sum(1) / w.reshape(-1, k).sum(1)
    w = np.asarray(th["weight"], float)
    for nm, a, b in [("xip ours vs theirs", d["xip"], th["xip_raw"]), ("xim ours vs theirs", d["xim"], th["xim_raw"]), ("xip e2flip vs fid", tf["xip_raw"], th["xip_raw"]), ("xim e2flip vs fid", tf["xim_raw"], th["xim_raw"])]:
        A = cb(np.asarray(a, float), w); B = cb(np.asarray(b, float), w)
        out[nm] = {"coarse_amp": float(A @ B / (B @ B)), "coarse_corr": float(np.corrcoef(A, B)[0, 1]), "fine_corr": float(np.corrcoef(a, b)[0, 1]), "fine_amp": float(np.sum(w * a * b) / np.sum(w * b * b))}
    print(json.dumps(out, indent=1)); save("pilot_recheck", out)

if __name__ == "__main__":
    {"flat": stage_flat, "frame": stage_frame, "ggl": stage_ggl, "pilot": stage_pilot}[sys.argv[1]]()
