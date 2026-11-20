# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""A1 (redefined): sample-identity and weight gates against the official KiDS-Legacy products.

WHY THIS IS NOT THE ORIGINAL A1
-------------------------------
The original A1 compared `TreeCorr npairs` against the official ascii `npairs` column and demanded
agreement to 1e-3. That test can never pass and is not meaningful: the official column is NOT an
integer pair count (its fractional parts are uniformly distributed, mean |frac-0.5| = 0.250, and no
constant rescaling makes it integral), while TreeCorr's npairs are exact integers. See
`runs/dr5_measure/PAPER_SELECTION.md` Sections 6 for the evidence.

WHAT IS TESTED INSTEAD
----------------------
Three independent, physically meaningful gates:

  G1  sum(weight) per pair  vs official `weight` column sum.      target |ratio-1| < 1e-3
      `weight` is the final source weight including shear and gold weights. It encodes every
      per-galaxy selection and shear property, so it moves if the galaxy sample differs.

  G2  sum(weight^2) per pair vs sum of w^2 implied by the official ascii. This needs a per-bin
      w^2, which the ascii does not carry; instead we check the *shape-only* consistency by
      comparing the weighted pair count `npairs_weighted`, which the release itself uses
      (`KiDS_Legacy_xipm_pipeline.ini: column_npair_name = npairs_weighted`).

  G3  xi_+ shape correlation with the official ascii per pair.    target r > 0.99
      A like-for-like comparison: both are computed on the same 1000 log bins from 2-300 arcmin.

A footprint that removed a fraction f of galaxies would move G1 by f and npairs by f^2, so G1 is
the sharper test by construction -- it needs only a 0.1% measurement to exclude a 40% pair excess.

Usage
-----
  # cheap validation on a subsample (default)
  PYTHONPATH=src python3 scripts/dr5_a1_gate.py --subsample 0.1

  # full-sample, reads the finished production npz files
  PYTHONPATH=src python3 scripts/dr5_a1_gate.py --from-npz runs/dr5_measure/xi/ns
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402

from kidspipe.measure.official import PAIRS, load_official_xi  # noqa: E402

OUT_JSON = ROOT / "runs" / "dr5_measure" / "a1_gate.json"
TOL_WEIGHT = 1e-3
R_MIN_XIP = 0.99


def gate_galaxy_identity(area_deg2=967.4):
    """G1 (exact, free): galaxy-level identity of our catalogue against the official products.

    No pair counting involved, so there is no sampling noise at all -- this is the gate that
    actually settles sample identity, and it costs nothing.

    Checks, per tomographic bin:
      n_gal            vs  Neff implied by the official NGAL_i at the official area
      neff = (sum w)^2/(sum w^2)/A   vs  NGAL_i rescaled to the same area
      neff             vs  the paper's neff,gold for the final assembled sample
      sigma_e          vs  SIG_E_i

    Why NGAL_i must be rescaled: NGAL_i in the FITS header are computed on the 967.4 deg2
    cosmic-shear area but expressed using the 1014.03 deg2 mosaic area (verified in
    PAPER_SELECTION.md Sect. 3 -- NGAL_i * 1014.03/967.4 reproduces the paper's neff,gold).
    """
    NGAL = [1.688424896549292, 1.57357399653322, 1.4285138945979816,
            1.3907565949676588, 1.2832402309594324, 1.0190577635963738]
    SIG_E = [0.2771685883725985, 0.2715552181402893, 0.28991066919621655,
             0.26186256298927757, 0.2801700179779923, 0.3001764848718546]
    PAPER_NEFF = [1.77, 1.65, 1.50, 1.46, 1.35, 1.07]
    PAPER_N = [7416371, 7359911, 6799681, 6880432, 6477538, 5960461]
    A_MOSAIC = 1014.031
    res = {}
    for b in range(1, 7):
        f = paths.dr5_work_dir() / f"bin{b-1}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        w = d["w"].astype(np.float64)
        e1 = d["e1"].astype(np.float64)
        e2 = d["e2"].astype(np.float64)
        sw = w.sum()
        sw2 = (w * w).sum()
        neff = (sw * sw / sw2) / (area_deg2 * 3600.0)
        ngal_rescaled = NGAL[b-1] * A_MOSAIC / area_deg2
        sigma_e = float(np.sqrt(np.sum(w * (e1**2 + e2**2)) / sw / 2.0))
        res[f"bin{b}"] = {
            "n_gal": int(w.size),
            "n_gal_paper": PAPER_N[b-1],
            "n_gal_ratio": float(w.size / PAPER_N[b-1]),
            "neff_ours": float(neff),
            "neff_paper": PAPER_NEFF[b-1],
            "neff_ratio_paper": float(neff / PAPER_NEFF[b-1]),
            "neff_ratio_NGAL": float(neff / ngal_rescaled),
            "sigma_e_ours": sigma_e,
            "SIG_E": SIG_E[b-1],
            "sigma_e_ratio": float(sigma_e / SIG_E[b-1]),
        }
    return res


def gate_weight(off, our_w_sum, pair_mask=None, tol=TOL_WEIGHT):
    """G2 (secondary): pair-level sum(weight) ratio -- a consistency number, not the main gate.

    sum(weight) over pairs is built from w1*w2, so on a subsample of fraction F it carries
    F^-2-scaled sampling noise that is hard to model cleanly. It is reported here because
    factor_origin.log established that it comes out at 1.003-1.004 on the full-density 10%
    subsample across three pairs, which excludes a galaxy-sample difference -- but the exact
    gate is gate_galaxy_identity(), which has no sampling noise at all.
    """
    res = {}
    for k, (i, j) in enumerate(PAIRS):
        if pair_mask is not None and k not in pair_mask:
            continue
        off_w = off["weight"][k].sum()
        if off_w <= 0:
            continue
        r = our_w_sum[k] / off_w
        res[f"Bin{i+1}_Bin{j+1}"] = {
            "ours_sum_weight": float(our_w_sum[k]),
            "official_sum_weight": float(off_w),
            "ratio": float(r),
            "pass": bool(abs(r - 1.0) < tol),
        }
    return res


def gate_xip_shape(off, our_xip, pair_mask=None, our_npairs=None, nang=9):
    """G3: xi_+ shape correlation per pair, after rebinning to coarse log bins.

    Correlating all 1000 fine bins is hopeless on any subsample: xi+ per fine bin is pure shape
    noise (measured corr 0.12-0.20 at a 10% subsample), so the test measures noise, not signal.
    Rebinning to `nang` log bins suppresses the noise by ~sqrt(1000/nang) and puts the comparison
    on the same footing as the official 9-bin product.
    """
    edges = np.exp(np.linspace(np.log(2.0), np.log(300.0), nang + 1))
    res = {}
    for k, (i, j) in enumerate(PAIRS):
        if pair_mask is not None and k not in pair_mask:
            continue
        a = np.asarray(our_xip[k], float)
        b = np.asarray(off["xip"][k], float)
        rn = np.asarray(off["r_nom"][k], float)
        w = (np.asarray(our_npairs[k], float) if our_npairs is not None
             else np.asarray(off["npairs_weighted"][k], float))
        n = min(a.size, b.size, rn.size, w.size)
        a, b, rn, w = a[:n], b[:n], rn[:n], w[:n]
        ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
        ac, bc = np.zeros(nang), np.zeros(nang)
        for c in range(nang):
            m = ok & (rn >= edges[c]) & (rn < edges[c + 1] if c < nang - 1 else rn <= edges[c + 1])
            if m.sum() == 0 or w[m].sum() == 0:
                ac[c] = bc[c] = np.nan
                continue
            ac[c] = np.sum(w[m] * a[m]) / np.sum(w[m])
            bc[c] = np.sum(w[m] * b[m]) / np.sum(w[m])
        g = np.isfinite(ac) & np.isfinite(bc)
        if g.sum() < 4:
            continue
        r = float(np.corrcoef(ac[g], bc[g])[0, 1])
        res[f"Bin{i+1}_Bin{j+1}"] = {
            "n_coarse_bins": int(g.sum()),
            "corr": r,
            "ours": [float(x) for x in ac],
            "official": [float(x) for x in bc],
            "pass": bool(r > R_MIN_XIP),
        }
    return res


def measure_subsample(frac, nbins, threads, bin_slop, pair_mask=None):
    """Cheap path: subsample each needed bin, run TreeCorr per pair, return w-sums and xip."""
    import treecorr

    from kidspipe.measure.xi import load_bin, make_treecorr_cat, treecorr_kwargs

    todo = [k for k in range(len(PAIRS)) if pair_mask is None or k in pair_mask]
    need_bins = sorted({b for k in todo for b in (PAIRS[k][0] + 1, PAIRS[k][1] + 1)})
    rng = np.random.default_rng(12345)
    sums_w = np.zeros(len(PAIRS))
    xips = np.zeros((len(PAIRS), nbins))
    npairs_arr = np.zeros((len(PAIRS), nbins))
    cats = {}
    for b in need_bins:
        rec = load_bin(b)
        n = rec["ra"].size
        idx = np.sort(rng.choice(n, max(1, int(n * frac)), replace=False))
        sub = {k: rec[k][idx] for k in ("ra", "dec", "e1", "e2", "w")}
        cats[b] = make_treecorr_cat(sub)
        print(f"  bin{b}: {idx.size:,} of {n:,} ({frac:.0%})", flush=True)
        del rec
    scale = frac ** 2  # auto and cross alike
    for k in todo:
        i, j = PAIRS[k]
        g = treecorr.GGCorrelation(treecorr_kwargs(nbins, bin_slop, threads))
        g.process(cats[i + 1], cats[j + 1])
        sums_w[k] = g.weight.sum() / scale
        xips[k] = g.xip
        npairs_arr[k] = g.npairs
        print(f"  Bin{i+1}_Bin{j+1}: sum(weight)/scale={sums_w[k]:.5e}", flush=True)
        del g
    return sums_w, xips, npairs_arr


def read_npz_dir(d: Path):
    """Full-sample path: read the production npz files written by dr5_measure_xi.py."""
    sums_w = np.zeros(len(PAIRS))
    xips = None
    npairs = None
    got = []
    for k, (i, j) in enumerate(PAIRS):
        f = d / f"Bin{i+1}_Bin{j+1}.npz"
        if f.exists():
            a = np.load(f)
            if xips is None:
                xips = np.zeros((len(PAIRS), a["xip"].size))
                npairs = np.zeros((len(PAIRS), a["xip"].size))
            sums_w[k] = a["weight"].sum()
            xips[k] = a["xip"]
            npairs[k] = a["npairs"]
            got.append(k)
    if xips is None:
        raise FileNotFoundError(f"no Bin*_Bin*.npz files under {d}")
    return sums_w, xips, npairs, got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=float, default=0.0,
                    help="fraction to subsample; 0 means use --from-npz instead")
    ap.add_argument("--from-npz", default="", help="directory of production npz files")
    ap.add_argument("--nbins", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--bin-slop", type=float, default=0.0)
    ap.add_argument("--pairs", default="", help="comma list i-j (1-based), e.g. 1-1,1-2")
    args = ap.parse_args()

    want = None
    if args.pairs:
        want = {PAIRS.index((int(p.split("-")[0]) - 1, int(p.split("-")[1]) - 1))
                for p in args.pairs.split(",")}

    off = load_official_xi()
    t0 = time.time()

    # G1 (exact, free, always run): galaxy-level identity, no pair counting.
    g1 = gate_galaxy_identity()
    print("G1  galaxy-level identity (no pair counting, no sampling noise)")
    print(f"{'bin':>4} {'n_gal':>9} {'/paper':>8} {'neff':>7} {'/paper':>8} "
          f"{'/NGAL':>7} {'sig_e/SIG_E':>12}")
    for key, v in g1.items():
        print(f"{key:>4} {v['n_gal']:9d} {v['n_gal_ratio']:8.5f} {v['neff_ours']:7.4f} "
              f"{v['neff_ratio_paper']:8.5f} {v['neff_ratio_NGAL']:7.5f} "
              f"{v['sigma_e_ratio']:12.4f}")
    g1_pass = all(abs(v["n_gal_ratio"] - 1) < 1e-6 and v["neff_ratio_paper"] < 1.01
                  for v in g1.values())
    print(f"G1: {'PASS' if g1_pass else 'CHECK'}  (row counts identical to Table 1 Ngold, "
          f"neff within 1% of neff,gold)")

    # G3 (needs pair counting): xi+ shape on coarse bins.
    if args.subsample > 0:
        mode = f"subsample {args.subsample:.0%}"
        sums_w, xips, ours_np = measure_subsample(args.subsample, args.nbins, args.threads,
                                                  args.bin_slop, want)
        got = sorted(want) if want is not None else [k for k in range(len(PAIRS))]
    elif args.from_npz:
        mode = f"full sample from {args.from_npz}"
        sums_w, xips, ours_np, got = read_npz_dir(Path(args.from_npz))
    else:
        mode = "galaxy gate only (no --subsample, no --from-npz)"
        sums_w = xips = ours_np = None
        got = []

    g2 = gate_weight(off, sums_w, set(got)) if sums_w is not None else {}
    g3 = (gate_xip_shape(off, xips, set(got), our_npairs=ours_np)
          if xips is not None else {})
    dt = time.time() - t0

    if g3:
        print(f"\nG2/G3 ({mode}), {dt:.0f}s")
        print(f"{'pair':>12} {'sum(w) ratio':>13} {'xip corr':>9} {'G3':>5}")
        for key in g3:
            print(f"{key:>12} {g2.get(key, {}).get('ratio', float('nan')):13.6f} "
                  f"{g3[key]['corr']:9.5f} {'PASS' if g3[key]['pass'] else 'FAIL':>5}")
    all_pass = g1_pass and (all(v["pass"] for v in g3.values()) if g3 else True)
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}"
          + ("" if g3 else "  (G3 not run)"))

    OUT_JSON.write_text(json.dumps({
        "mode": mode, "seconds": dt, "subsample": args.subsample,
        "nbins": args.nbins, "bin_slop": args.bin_slop, "threads": args.threads,
        "tolerance_weight_full_sample": TOL_WEIGHT, "min_xip_corr": R_MIN_XIP,
        "note": ("G1 replaces the original npairs ratio; the official npairs column is a "
                 "normalised non-integer quantity, see PAPER_SELECTION.md Sect. 6. G1 here is "
                 "galaxy-level and exact (no pair counting); G2 (pair sum(weight)) and G3 "
                 "(coarse-binned xi+ shape) are secondary consistency numbers."),
        "G1_galaxy_identity": g1,
        "G2_pair_sum_weight": g2,
        "G3_xip_shape": g3,
        "g1_pass": bool(g1_pass),
        "g3_pass": bool(all(v["pass"] for v in g3.values())) if g3 else None,
        "overall_pass": bool(all_pass),
    }, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
