"""COSEBIs-only MCMC (blocked Metropolis, emulator model).  Run (from any directory):
    python scripts/cosebis_mcmc.py NMODES NCHAINS CYCLES BURN [tag]          e.g.  COSEBIS_WALL=1500 python scripts/cosebis_mcmc.py 5 10 10000 1500 n5
Seeds: chain starts = random draws of the xi+/- posterior (prod1/prod2), proposal cov0 = xi+/- posterior covariance.
One process per chain (NCHAINS, each single-threaded; about 0.5 core-h per chain for the documented 5-mode run); COSEBIS_WALL = wall-clock
limit per chain in seconds (default 2400).

Paths (kidspipe.paths): the xi+/- production chains <runs>/kv450_final/prod*/prod_c*.npz (else the copy shipped in <root>/runs), the
emulator artefact (data/emulator or $KIDSPIPE_EMULATOR); output <runs>/cosebis/chain_<tag>_c<i>.npz, <runs> = $KIDSPIPE_RUNS (default
<root>/runs).  Moved here from runs/cosebis/mcmc.py.
"""
import os, sys, glob, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
from pathlib import Path
if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
    print(__doc__); sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
import numpy as np
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths
NM, NCH, CYC, BURN = (int(a) for a in sys.argv[1:5]); TAG = sys.argv[5] if len(sys.argv) > 5 else f"n{NM}"
OUT = paths.runs_dir() / "cosebis"

def xi_chain():
    for base in (paths.runs_dir(), paths.shipped_runs_dir()):        # your production chains, else the shipped ones
        fs = sorted(glob.glob(str(base / "kv450_final" / "prod*" / "prod_c*.npz")))
        if fs:
            break
    else:
        sys.exit("no KV450 production chains (runs/kv450_final/prod*/prod_c*.npz) found: run scripts/run_kv450_mcmc.py prod first or restore the shipped runs/")
    return np.concatenate([np.load(f)["samples"][1000::20] for f in fs])

def work(i):
    from kidspipe import api, params
    from kidspipe.likelihood import cosebis as LC
    from kidspipe.inference.mcmc import blocked_metropolis
    m = api.build_model(linear_backend="emulator", nthreads=1) if "nthreads" in api.build_model.__code__.co_varnames else api.build_model(linear_backend="emulator")
    cm, like = LC.build_cosebis(NM, True, model=m, nthreads=1)
    P0 = LC.CosebisPosterior(cm, like)
    BOX = np.load(paths.emulator_artefact())["box"]
    class _P:   # truncate to the emulator training box (no CAMB fallback: avoids 1-2 s/eval and thread oversubscription)
        n_out = 0
        def log_post(self, t):
            if np.any(t[:3] < BOX[:, 0]) or np.any(t[:3] > BOX[:, 1]):
                _P.n_out += 1
                return -np.inf
            return P0.log_post(t)
    P = _P()
    S = xi_chain(); rng = np.random.default_rng(100 + i)
    cov0 = np.cov(S.T)
    x0 = S[rng.integers(len(S))]
    sched = [(np.concatenate([api.block_indices("slow"), api.block_indices("medium")]), 1),
             (api.block_indices("medium"), 2), (api.block_indices("fast"), 6)]
    t0 = time.time()
    WALL = float(os.environ.get("COSEBIS_WALL", 2400)); SEG = 500
    segs, lps, accs, cov = [], [], [], cov0
    x = x0; burn = BURN; done = 0
    while True:
        r = blocked_metropolis(P.log_post, x, sched, n_cycles=SEG, n_burn_cycles=burn, cov0=cov, bounds=params.BOUNDS, seed=7 + 1000 * i + done)
        segs.append(r.samples); lps.append(r.log_post); accs.append(np.atleast_1d(r.accept).tolist())
        x = r.samples[-1]; cov = r.cov; burn = 0; done += SEG
        fb = dict(m.linear.stats) if hasattr(m.linear, "stats") else {}
        print(f"chain {i} cycles {done} t={time.time()-t0:.0f}s fallback={fb.get('fallback')} box_rejects={_P.n_out}", flush=True)
        np.savez(OUT / f"chain_{TAG}_c{i}.npz", samples=np.concatenate(segs), log_post=np.concatenate(lps), accept=np.array(accs), time=time.time() - t0, x0=x0, cycles=done)
        if time.time() - t0 > WALL or done >= CYC:
            break
    samples = np.concatenate(segs)
    der = api.DerivedRecorder(m)
    d = np.array([[v for v in (lambda q: (q["sigma8"], q["omega_m"], q["S8"]))(der.lookup(s))] for s in samples[::5]])
    fb = dict(m.linear.stats) if hasattr(m.linear, "stats") else {}
    np.savez(OUT / f"chain_{TAG}_c{i}.npz", samples=samples, log_post=np.concatenate(lps), derived5=d, accept=np.array(accs), time=time.time() - t0, x0=x0, cycles=done, box_rejects=_P.n_out)
    return i, time.time() - t0, done, int(_P.n_out), {k: float(v) for k, v in fb.items()}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    from kidspipe import api  # prebuild operator cache once
    from kidspipe.likelihood import cosebis as LC
    LC.build_cosebis(NM, True, model=api.build_model(linear_backend="emulator"))
    with Pool(NCH) as p:
        for r in p.imap_unordered(work, range(NCH)):
            print(r, flush=True)
