#!/usr/bin/env python
"""Train the (omch2, ombh2, h) -> linear-transfer emulator (see ``kidspipe.theory.emulator``).

Stages (default: all, in order)::

    python scripts/train_emulator.py generate   # parallel exact CAMB runs on a Sobol design -> <runs>/emulator_train/*.npz
    python scripts/train_emulator.py fit        # PCA + regression -> data/emulator/transfer_emulator_v1.npz ($KIDSPIPE_EMULATOR)
    python scripts/train_emulator.py validate   # held-out error report (JSON in <runs>/emulator_train/validation.json)

<runs> = $KIDSPIPE_RUNS (default <root>/runs).  ``fit`` OVERWRITES the emulator artefact the pipeline uses (``kidspipe.paths.emulator_artefact()``:
the shipped data/emulator/transfer_emulator_v1.npz unless $KIDSPIPE_EMULATOR points elsewhere); set $KIDSPIPE_EMULATOR to write a new
artefact without touching the shipped one.  ``--workers`` defaults to os.cpu_count() (about 17 core-hours for ``generate``).

Everything is seeded; re-running ``python scripts/train_emulator.py`` reproduces the artefact (up to CAMB
floating-point reproducibility).  CAMB runs use 1 OpenMP thread per worker process, ``--workers`` processes.

Training target (see the emulator module docstring for the formulation): the CAMB linear total- and
cold-matter spectra at the reference primordial spectrum (ln 10^10 A_s = 3, n_s = 1), converted to
P_Mpc(k, z) [Mpc^3] and resampled (cubic spline in ln k) from CAMB's cosmology-dependent native k nodes
onto one fixed k grid [1/Mpc]; ln P_Mpc is compressed by PCA and the PCA coefficients are regressed on
(omch2, ombh2, h).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from kidspipe import paths  # noqa: E402

TRAIN_DIR = paths.runs_dir() / "emulator_train"
ARTEFACT = paths.emulator_artefact()

# ---- the design (documented in docs/PERFORMANCE.md) -----------------------------------------------------
BOX = {"omch2": (0.03, 0.34), "ombh2": (0.019, 0.026), "h": (0.64, 0.82)}
SEED_TRAIN, SEED_TEST, SEED_AUG = 20241101, 20241102, 20241103
N_TRAIN, N_TEST, N_AUG = 1024, 256, 384
AUG_OMCH2_MAX = 0.12      # extra training points: omch2 log-uniform in [box lo, AUG_OMCH2_MAX] (the spectrum varies fastest there)
LN_AS_REF, NS_REF, MNU = 3.0, 1.0, 0.06
CENTRE = (0.5 * sum(BOX["omch2"]), 0.5 * sum(BOX["ombh2"]), 0.5 * sum(BOX["h"]))


def master_z() -> np.ndarray:
    """Union of the 150-node spec grid and the 41-node reduced grid (so both are exact subsets)."""
    from kidspipe.theory.linear import Z_GRID_REDUCED, Z_GRID_SPEC
    z = np.unique(np.round(np.concatenate([Z_GRID_SPEC, Z_GRID_REDUCED]), 12))
    return z


def design(n: int, seed: int, corners: bool) -> np.ndarray:
    """Scrambled Sobol design in the box (n x 3: omch2, ombh2, h); optionally + the 27 points of a 3^3 lattice
    (corners, edge mid-points, face centres, centre) so the box boundary is anchored."""
    from scipy.stats import qmc
    u = qmc.Sobol(d=3, scramble=True, seed=seed).random(n)
    if corners:
        g = np.array([[a, b, c] for a in (0, .5, 1) for b in (0, .5, 1) for c in (0, .5, 1)], float)
        u = np.vstack([u, g])
    lo = np.array([BOX[k][0] for k in ("omch2", "ombh2", "h")])
    hi = np.array([BOX[k][1] for k in ("omch2", "ombh2", "h")])
    return lo + (hi - lo) * u


def design_aug(n: int, seed: int) -> np.ndarray:
    """Extra design block concentrated at low omch2: Sobol in (ln omch2, ombh2, h), omch2 in [box lo, AUG_OMCH2_MAX]."""
    from scipy.stats import qmc
    u = qmc.Sobol(d=3, scramble=True, seed=seed).random(n)
    lo, hi = np.log(BOX["omch2"][0]), np.log(AUG_OMCH2_MAX)
    om = np.exp(lo + (hi - lo) * u[:, 0])
    ob = BOX["ombh2"][0] + (BOX["ombh2"][1] - BOX["ombh2"][0]) * u[:, 1]
    h = BOX["h"][0] + (BOX["h"][1] - BOX["h"][0]) * u[:, 2]
    return np.stack([om, ob, h], axis=1)


# ---- fixed output grid + worker ------------------------------------------------------------------------
K_PER_LOGINT_TRAIN = 40          # dense uniform-in-ln(k) CAMB sampling for TRAINING (default native sampling aliases the BAO)
KMAX_TRAIN = 21.0                # set_matter_power(kmax) [1/Mpc]; CAMB then delivers nodes up to >= 21.2


def k_out_grid() -> np.ndarray:
    """Fixed emulator output grid [1/Mpc]: 5e-5 ... 21, spacing dlnk = 0.33 below 1e-3, 0.04 on [1e-3, 1.2]
    (where the BAO wiggles live) and ~0.24 on [1.2, 21]; ~200 nodes (CAMB's default native grid has 183-212)."""
    lo = np.exp(np.linspace(np.log(5e-5), np.log(1e-3), 10))
    mid = np.exp(np.arange(np.log(1e-3), np.log(1.2), 0.04))[1:]
    hi = np.exp(np.linspace(np.log(mid[-1]), np.log(21.0), int(np.ceil((np.log(21.0) - np.log(mid[-1])) / 0.24)) + 1))[1:]
    return np.concatenate([lo, mid, hi])


_W = {}


def _init_worker(zgrid, kgrid):
    import camb
    camb.config.ThreadNum = 1
    from kidspipe.theory.linear import LinearBackend

    class Dense(LinearBackend):
        def _make_params(self, *a):
            p = super()._make_params(*a)
            p.Transfer.k_per_logint = K_PER_LOGINT_TRAIN
            return p

    _W["dense"] = Dense(z_grid=zgrid, kmax_per_mpc=KMAX_TRAIN, nthreads=1, cache_size=1)
    _W["native"] = LinearBackend(z_grid=zgrid, nthreads=1, cache_size=1)
    _W["k"] = kgrid


def resample(k_mpc, p_h, h, kgrid):
    """(nz, nk_native) P in (Mpc/h)^3 on native k nodes -> ln P_Mpc on ``kgrid`` (cubic spline in ln k)."""
    from scipy.interpolate import CubicSpline
    return CubicSpline(np.log(k_mpc), np.log(p_h / h ** 3), axis=1)(np.log(kgrid))


def _run_one(arg):
    x, with_native = arg
    kgrid = _W["k"]
    omch2, ombh2, h = map(float, x)
    lb = _W["dense"]
    lb.clear_cache()
    e = lb._run_camb(omch2, ombh2, h, NS_REF, LN_AS_REF, MNU)
    kmpc = e.k_h * e.h
    lnp = resample(kmpc, e.pk_ref, e.h, kgrid)
    lnc = resample(kmpc, e.pk_cb_ref, e.h, kgrid)
    rdrag = float(e.results.get_derived_params()["rdrag"])
    out = dict(lnp=lnp.astype(np.float32), rat=(lnc - lnp).astype(np.float32), sigma8=e.sigma8_ref, rdrag=rdrag,
               nk=len(kmpc), kmax=float(kmpc[-1]))
    if with_native:                    # what the production (default-sampling) CAMB path delivers, on the same grid
        nb = _W["native"]
        nb.clear_cache()
        n = nb._run_camb(omch2, ombh2, h, NS_REF, LN_AS_REF, MNU)
        km = n.k_h * n.h
        lp = resample(km, n.pk_ref, n.h, kgrid)
        out["lnp_native"] = lp.astype(np.float32)
        out["rat_native"] = (resample(km, n.pk_cb_ref, n.h, kgrid) - lp).astype(np.float32)
        out["sigma8_native"] = n.sigma8_ref
    return out


def generate(args):
    import multiprocessing as mp
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    z = master_z()
    kgrid = k_out_grid()
    np.save(TRAIN_DIR / "kgrid.npy", kgrid)
    np.save(TRAIN_DIR / "zgrid.npy", z)
    for name, n, seed, corners in (("test", args.n_test, SEED_TEST, False), ("train", args.n_train, SEED_TRAIN, True),
                                   ("train_aug", args.n_aug, SEED_AUG, None)):
        out = TRAIN_DIR / f"{name}.npz"
        if out.exists() and not args.force:
            print(f"[generate] {out} exists, skipping (use --force)")
            continue
        X = design_aug(n, seed) if name == "train_aug" else design(n, seed, corners)
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(z, kgrid)) as pool:
            res = []
            for i, r in enumerate(pool.imap(_run_one, [(x, name == "test") for x in X], chunksize=1)):
                res.append(r)
                if (i + 1) % 50 == 0:
                    print(f"[generate:{name}] {i + 1}/{len(X)}  {time.time() - t0:.0f} s", flush=True)
        keys = res[0].keys()
        np.savez(out, X=X, **{k: np.stack([r[k] for r in res]) for k in keys})
        print(f"[generate:{name}] {len(X)} CAMB runs in {time.time() - t0:.0f} s -> {out}")


# ---- fit ------------------------------------------------------------------------------------------------
def _scale(X):
    """Parameters -> [-1, 1]^3 in logarithmic coordinates (same as TransferEmulator.scale)."""
    lo = np.log([BOX[k][0] for k in ("omch2", "ombh2", "h")])
    hi = np.log([BOX[k][1] for k in ("omch2", "ombh2", "h")])
    return 2.0 * (np.log(X) - lo) / (hi - lo) - 1.0


def _pca(Y, npc_max):
    """PCA by eigendecomposition of the (N x N) Gram matrix of the centred data; returns mean, basis (npc, D), sv."""
    mean = Y.mean(0)
    A = Y - mean
    w, V = np.linalg.eigh(A @ A.T)
    o = np.argsort(w)[::-1][:npc_max]
    w, V = w[o], V[:, o]
    basis = (A.T @ V / np.sqrt(w)).T                    # orthonormal rows
    return mean, basis, np.sqrt(w / len(Y))


#: k window [1/Mpc] around CAMB's own numerical glitch (a single-node spike of up to ~2e-3 in the converged CAMB
#: output near k = 0.0396 /Mpc, present in the default-sampling output too; it is noise in the TRUTH, not emulatable)
GLITCH_WINDOW = (0.0385, 0.0425)


def _err_metric(dlnp, X, z, k, zmax=3.0, khmax=10.0, skip_glitch=False):
    """max over (z <= zmax, k_h < khmax) of |exp(dlnP) - 1| for each sample; dlnp (n, nz, nk).
    ``skip_glitch`` drops the k window ``GLITCH_WINDOW`` (CAMB numerical noise in the truth)."""
    zm = z <= zmax
    out = np.empty(len(X))
    kok = ~((k > GLITCH_WINDOW[0]) & (k < GLITCH_WINDOW[1])) if skip_glitch else np.ones_like(k, bool)
    for i in range(len(X)):
        km = (k / X[i, 2] < khmax) & kok
        out[i] = np.abs(np.expm1(dlnp[i][np.ix_(zm, km)])).max()
    return out


def _loo_gp_one(args):
    """Hyper-parameters (ell_1..3, nugget) of one PC by minimising the closed-form leave-one-out error of
    'polynomial + SE-ARD GP' (the polynomial is kept fixed; GP on the residual)."""
    from scipy.optimize import minimize
    from kidspipe.theory.emulator import gp_kernel
    Xfull, rfull = args
    sd = rfull.std() + 1e-300
    sub = np.random.default_rng(0).choice(len(Xfull), min(700, len(Xfull)), replace=False)   # hyper-parameter search subset
    Xs, y = Xfull[sub], rfull[sub] / sd
    N = len(Xs)

    def obj(lp):
        ell, lam = np.exp(lp[:3]), np.exp(lp[3])
        K = gp_kernel(Xs, Xs, ell) + lam * np.eye(N)
        try:
            Ki = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            return 1e3
        loo = (Ki @ y) / np.diag(Ki)
        return float(np.log(np.mean(loo ** 2) + 1e-30))

    best = None
    for l0 in ((1.0, 1.0, 1.0), (0.4, 0.8, 1.5)):
        x0 = np.log(np.array(list(l0) + [1e-7]))
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=120, xatol=2e-2, fatol=1e-3))
        if best is None or res.fun < best.fun:
            best = res
    ell, lam = np.exp(best.x[:3]), np.exp(best.x[3])
    K = gp_kernel(Xfull, Xfull, ell) + lam * np.eye(len(Xfull))
    return ell, np.linalg.solve(K, rfull / sd) * sd, best.fun


def _fit_gp_residual(Xs, C, deg, n_jobs=None):
    """Polynomial (degree ``deg``) + per-PC ARD squared-exponential GP on the residual; hyper-parameters per PC by
    leave-one-out error (numpy/scipy only)."""
    from joblib import Parallel, delayed
    from kidspipe.theory.emulator import poly_design
    n_jobs = n_jobs or os.cpu_count() or 1
    P = poly_design(Xs, deg)
    coef = np.linalg.lstsq(P, C, rcond=1e-13)[0]
    R = C - P @ coef
    out = Parallel(n_jobs=n_jobs)(delayed(_loo_gp_one)((Xs, R[:, j])) for j in range(C.shape[1]))
    return dict(kind="polygp", deg=deg, coef=coef, gp_X=Xs, gp_alpha=np.stack([o[1] for o in out], 1),
                gp_ell=np.stack([o[0] for o in out]))


def _fit_poly(Xs, C, deg):
    from kidspipe.theory.emulator import poly_design
    return dict(kind="poly", deg=deg, coef=np.linalg.lstsq(poly_design(Xs, deg), C, rcond=1e-13)[0])


def _predict(reg, Xs):
    from kidspipe.theory.emulator import _Regressor
    return _Regressor(reg)(Xs)


def fit(args):
    import camb
    parts = [np.load(TRAIN_DIR / "train.npz")]
    if (TRAIN_DIR / "train_aug.npz").exists() and not args.no_aug:
        parts.append(np.load(TRAIN_DIR / "train_aug.npz"))
    d_tr = {kk: np.concatenate([p[kk] for p in parts]) for kk in ("X", "lnp", "rat")}
    d_te = np.load(TRAIN_DIR / "test.npz")
    z, k = np.load(TRAIN_DIR / "zgrid.npy"), np.load(TRAIN_DIR / "kgrid.npy")
    nz, nk = len(z), len(k)
    Xtr, Xte = d_tr["X"], d_te["X"]
    Str, Ste = _scale(Xtr), _scale(Xte)
    meta = dict(version=1, camb_version=camb.__version__, box=BOX, seed_train=SEED_TRAIN, seed_test=SEED_TEST,
                n_train=int(len(Xtr)), n_test=int(len(Xte)), n_train_aug=int(len(parts[1]["X"])) if len(parts) > 1 else 0,
                coords="log(omch2, ombh2, h) scaled to [-1,1]", seed_aug=SEED_AUG, ln_As_ref=LN_AS_REF, ns_ref=NS_REF, mnu=MNU,
                k_per_logint_train=K_PER_LOGINT_TRAIN, kmax_train=KMAX_TRAIN, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                nk=int(nk), nz=int(nz), target="PCA(ln P_Mpc(k,z)) and PCA(ln(P_cb/P_tot)) on fixed k grid [1/Mpc]")
    arte = {}
    for tag, key, npc_max in (("tot", "lnp", args.npc_tot), ("rat", "rat", args.npc_rat)):
        Ytr = d_tr[key].reshape(len(Xtr), -1).astype(np.float64)
        Yte = d_te[key].reshape(len(Xte), -1).astype(np.float64)
        mean, basis, sv = _pca(Ytr, npc_max)
        # smallest npc whose PCA reconstruction error on the held-out spectra is below the target
        errs = {}
        for npc in range(4, npc_max + 1, 2):
            rec = ((Yte - mean) @ basis[:npc].T) @ basis[:npc] + mean
            errs[npc] = float(_err_metric((rec - Yte).reshape(-1, nz, nk), Xte, z, k).max())
        print(f"[fit:{tag}] PCA recon max err by npc:", {n: f"{e:.1e}" for n, e in errs.items()})
        target = args.pca_target if tag == "tot" else args.pca_target_rat
        npc = next((n for n, e in errs.items() if e <= target), npc_max)
        basis = basis[:npc]
        Ctr = (Ytr - mean) @ basis.T
        # --- choose the regressor on a validation split of the TRAINING set (the test set stays untouched)
        rng = np.random.default_rng(1)
        perm = rng.permutation(len(Xtr))
        nv = len(Xtr) // 5
        iv, it = perm[:nv], perm[nv:]
        cands = {}
        for name, fn in (("poly8", lambda S, C: _fit_poly(S, C, 8)), ("poly10", lambda S, C: _fit_poly(S, C, 10)),
                         ("polygp3", lambda S, C: _fit_gp_residual(S, C, 3)),
                         ("polygp2", lambda S, C: _fit_gp_residual(S, C, 2))):
            if args.regressors and name not in args.regressors:
                continue
            t0 = time.time()
            reg = fn(Str[it], Ctr[it])
            rec = _predict(reg, Str[iv]) @ basis + mean
            e = _err_metric((rec - Ytr[iv]).reshape(-1, nz, nk), Xtr[iv], z, k)
            cands[name] = (float(e.max()), float(np.quantile(e, 0.99)), fn)
            print(f"[fit:{tag}] {name:8s} validation max err {e.max():.2e} p99 {np.quantile(e, .99):.2e}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        best = min(cands, key=lambda n: cands[n][0])
        reg = cands[best][2](Str, Ctr)
        print(f"[fit:{tag}] selected {best} with npc={npc}")
        meta[f"{tag}_npc"] = int(npc)
        meta[f"{tag}_regressor"] = best
        meta[f"{tag}_validation_max_err"] = {n: c[0] for n, c in cands.items()}
        arte[f"{tag}_mean"] = mean.astype(np.float64)
        arte[f"{tag}_basis"] = basis.astype(np.float32)
        for kk, vv in reg.items():
            arte[f"reg_{tag}_{kk}"] = np.asarray(vv)
    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARTEFACT, meta_json=json.dumps(meta), box=np.array([BOX[k] for k in ("omch2", "ombh2", "h")]),
                        k_mpc=k, z_master=z, ln_As_ref=LN_AS_REF, ns_ref=NS_REF, mnu=MNU, **arte)
    print(f"[fit] wrote {ARTEFACT} ({ARTEFACT.stat().st_size / 1e6:.1f} MB)")


# ---- validate ---------------------------------------------------------------------------------------------
def validate(args):
    from kidspipe.theory.emulator import TransferEmulator
    em = TransferEmulator()
    d = np.load(TRAIN_DIR / "test.npz")
    z, k = em.z, em.k_mpc
    X = d["X"]
    out = {"n_test": int(len(X)), "meta": em.meta}
    ev = [em.predict(*x) for x in X]
    lnp = np.stack([e[0] for e in ev])
    lnc = np.stack([e[1] for e in ev])
    for label, truth_p, truth_r in (("vs_dense_truth", d["lnp"], d["rat"]), ("vs_default_native_resampled", d["lnp_native"], d["rat_native"])):
        e_tot = _err_metric(lnp - truth_p, X, z, k)
        e_cb = _err_metric(lnc - (truth_p + truth_r), X, z, k)
        e_ng = _err_metric(lnp - truth_p, X, z, k, skip_glitch=True)
        out[label + "_excluding_camb_glitch_window"] = {"P_tot_max": float(e_ng.max()), "P_tot_p99": float(np.quantile(e_ng, .99)),
                                                         "P_tot_median": float(np.median(e_ng)), "frac_below_5e-4": float((e_ng <= 5e-4).mean())}
        out[label] = {"P_tot_max": float(e_tot.max()), "P_tot_p99": float(np.quantile(e_tot, .99)), "P_tot_median": float(np.median(e_tot)),
                      "P_cb_max": float(e_cb.max()), "P_cb_p99": float(np.quantile(e_cb, .99)), "P_cb_median": float(np.median(e_cb)),
                      "frac_below_5e-4": float((e_tot <= 5e-4).mean()), "frac_below_2e-3": float((e_tot <= 2e-3).mean())}
        print(f"[validate] {label}: ", json.dumps(out[label]))
    # sigma8: quadrature of the emulated z=0 spectrum vs the converged (dense) CAMB sigma8 and the default-sampling one
    from kidspipe.theory.emulator import _Sigma8
    s8 = np.array([_Sigma8(k, ev[i][0][0], X[i, 2])(1.0, 0.0) for i in range(len(X))])
    out["sigma8_rel_vs_dense_camb"] = {"max": float(np.abs(s8 / d["sigma8"] - 1).max()), "rms": float(np.sqrt(np.mean((s8 / d["sigma8"] - 1) ** 2)))}
    out["sigma8_rel_vs_default_camb"] = {"max": float(np.abs(s8 / d["sigma8_native"] - 1).max())}
    out["sigma8_dense_vs_default_camb"] = {"max": float(np.abs(d["sigma8"] / d["sigma8_native"] - 1).max())}
    print("[validate] sigma8:", json.dumps({kk: out[kk] for kk in out if kk.startswith("sigma8")}))
    (TRAIN_DIR / "validation.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "generate", "fit", "validate"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="CAMB worker processes for `generate` (default os.cpu_count())")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-aug", type=int, default=N_AUG)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-aug", action="store_true", help="fit without the low-omch2 augmentation block")
    ap.add_argument("--npc-tot", type=int, default=40)
    ap.add_argument("--npc-rat", type=int, default=16)
    ap.add_argument("--pca-target", type=float, default=5e-5, help="held-out PCA reconstruction target (total P)")
    ap.add_argument("--pca-target-rat", type=float, default=2e-5)
    ap.add_argument("--regressors", nargs="*", default=None, help="restrict candidates (poly8 poly10 polygp3 polygp2)")
    a = ap.parse_args()
    if a.stage in ("all", "generate"):
        generate(a)
    if a.stage in ("all", "fit"):
        fit(a)
    if a.stage in ("all", "validate"):
        validate(a)
