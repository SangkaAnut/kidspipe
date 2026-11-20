"""Exact-CAMB re-evaluation helpers.  usage: run_kv450_exact.py map  |  run_kv450_exact.py chain TAG [N]
 map   : re-evaluate the top MAP candidates (runs/kv450_final/map_starts.json) with linear_backend='camb' (all else equal)
 chain : re-evaluate N thinned samples of the production chains with exact CAMB, report delta ln L distribution
"""
import json
import sys
import numpy as np
import multiprocessing as mp
import run_kv450_common as C

_exact = None
_emu = None


def _init():
    global _exact, _emu
    _exact = C.make_post(linear="camb", threads=1)
    _emu = C.make_post(linear="emulator", threads=1)


def _one(th):
    th = np.asarray(th)
    return float(_emu.log_like(th)), float(_exact.log_like(th)), float(_emu.log_post(th)), float(_exact.log_post(th))


def main():
    if sys.argv[1] == "map":
        R = json.load(open(C.OUT / "map_starts.json"))["results"]
        top = [r["theta"] for r in R[:3]]
        with mp.get_context("spawn").Pool(3, initializer=_init) as pool:
            out = pool.map(_one, top)
        res = [dict(name=r["name"], chi2_emu=-2 * o[0], chi2_exact=-2 * o[1], dchi2=-2 * (o[0] - o[1]),
                    theta=r["theta"]) for r, o in zip(R, out)]
        for r in res:
            print(r["name"], f"emu {r['chi2_emu']:.4f} exact {r['chi2_exact']:.4f} d {r['dchi2']:+.4f}")
        C.jdump(C.OUT / "map_exact_check_powell.json", res)
    else:
        tag = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        import glob
        S = np.concatenate([np.load(f)["samples"] for f in sorted([f for t in tag.split(",") for f in glob.glob(str(C.OUT / t / "prod_c*.npz"))])])
        rng = np.random.default_rng(7)
        pick = S[rng.choice(len(S), n, replace=False)]
        with mp.get_context("spawn").Pool(4, initializer=_init) as pool:
            out = np.array(pool.map(_one, pick.tolist(), chunksize=4))
        d = out[:, 0] - out[:, 1]
        print(f"n={n}  delta lnL (emu - exact): mean {d.mean():+.5f} sd {d.std():.5f} max|.| {np.abs(d).max():.5f}  "
              f"in-box {C.in_emu_box(pick).mean():.4f}")
        C.jdump(C.OUT / f"exact_check_{tag.replace(',', '_')}.json", dict(samples=pick, lnl_emu=out[:, 0], lnl_exact=out[:, 1], dlnl=d,
                                                       in_box=C.in_emu_box(pick)))


if __name__ == "__main__":
    main()
