"""OFFICIAL-style unordered measurement of the DR5 tomographic xi+/- and its evaluation against the release.

Pairing follows the official TreeCorr use: one GGCorrelation per bin pair, auto pairs as UNORDERED pairs via
gg.process(cat) on the whole bin catalogue (no chunking), cross pairs gg.process(cat1, cat2); 1000 log bins 2-300 arcmin,
sep_units arcmin, angle_slop = TreeCorr default, catalogue e1,e2 as is (no flip), catalogue weight.  Official release:
TreeCorr 4.2.3, bin_slop 1.5 (Wright+25 Sect. 2.5.1 footnote 1).

  python scripts/dr5_measure_xi_unordered.py run  [--pairs autos|all|I-J,...] [--bin-slop 1.5] [--threads N] [--force]
  python scripts/dr5_measure_xi_unordered.py eval [--pairs ...] [--bin-slop 1.5]            (reads our npz)
  python scripts/dr5_measure_xi_unordered.py eval --source theirs --pairs autos              (preview from their xi)

Paths (kidspipe.paths): inputs bin0..5.npz from $KIDSPIPE_DR5_WORK (written by dr5_extract.py), official release files from
$KIDSPIPE_OFFICIAL; outputs under <runs>/dr5_measure/xi_unordered/ with <runs> = $KIDSPIPE_RUNS (default <root>/runs):
b<slop>/Bin<I>_Bin<J>.npz, manifest.json, eval.json, EVAL.md.  --threads defaults to os.cpu_count().  Run one process per pair
on a 12-16 GB machine (a TreeCorr cross pair needs about 5 GB and a long-lived process retains memory across pairs).
Optional inputs that degrade gracefully when absent: the earlier pipeline's xi (``$KIDSPIPE_THEIRS``, default
/mnt/d/HI-gamma-correlation/...; the ours-vs-theirs comparison is skipped) and a1_salvage.json (the one-constant A_p test is skipped).
Evaluation functions take (pair, ours_dict, ordered): all weighting is scale invariant, so an ordered (2x npairs)
chunk-assembled npz gives the same gates; `ordered` only normalises the npairs ratio against theirs.
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
from kidspipe.measure import official as of  # noqa: E402
from kidspipe.measure.a1check import (A1B_THRESH, A3_MAX_ABS_OVER_SIGMA, a1b_noise, a3_pass,  # noqa: E402
                                      boxcar_ratio, curves_z)

OUTROOT = paths.runs_dir() / "dr5_measure" / "xi_unordered"
# earlier pipeline's xi (external, optional): only used for the informational ours-vs-theirs comparison
THEIRS = Path(os.environ.get("KIDSPIPE_THEIRS", "/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial"))
SALV = paths.find_result("dr5_measure/a1_salvage/a1_salvage.json")     # input (shipped copy is found if you have none of your own)
# m_gold per bin, Wright+25 Table 1: same list as scripts/dr5_adopt_01.py (M); xi_corr = xi_raw / ((1+m_i)(1+m_j))
M = [-0.023, -0.016, -0.011, 0.020, 0.030, 0.045]
N_ANG = 9
E9 = np.exp(np.linspace(np.log(2.0), np.log(300.0), N_ANG + 1))


def _load_chunked():
    spec = importlib.util.spec_from_file_location("drv", ROOT / "scripts/dr5_measure_xi_chunked.py")
    drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
    return drv


def parse_pairs(s):
    """'autos' | 'all' | 'I-J,...' (1-based) -> list of (I, J), I <= J."""
    if s == "autos":
        return [(i, i) for i in range(1, 7)]
    if s == "all":
        return [(i + 1, j + 1) for i, j in of.PAIRS]
    out = []
    for t in s.split(","):
        i, j = (int(x) for x in t.split("-"))
        out.append((min(i, j), max(i, j)))
    return list(dict.fromkeys(out))


def pname(p):
    return f"Bin{p[0]}_Bin{p[1]}"


def pair_index(p):
    """Index of 1-based pair (I, J) in official.PAIRS (= pair-major row order of the FITS products)."""
    return of.PAIRS.index((p[0] - 1, p[1] - 1))


# ------------------------------------------------------------------------------------------------ run
def atomic_write_text(f, text):
    tmp = f.with_name(f.name + ".tmp"); tmp.write_text(text); os.replace(tmp, f)


def pair_file_ok(f):
    """True if f loads as a complete pair npz (guards resume against a truncated file)."""
    try:
        with np.load(f) as d:
            return all(np.asarray(d[k]).size > 0 for k in ("xip", "xim", "npairs", "weight", "r_nom"))
    except Exception:
        return False


def _git_head():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def do_run(args):
    import treecorr
    from kidspipe.measure import xi as xim
    from kidspipe.measure.xi import corr_to_arrays, make_treecorr_cat, treecorr_kwargs
    drv = _load_chunked()
    pairs = parse_pairs(args.pairs)
    out = OUTROOT / (f"b{args.bin_slop:g}" + (f"_sub{args.subsample}" if args.subsample else ""))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kw = treecorr_kwargs(bin_slop=args.bin_slop, num_threads=args.threads)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    rows = man.get("pairs", {})
    top = {"pairing": "unordered (auto: gg.process(cat); cross: gg.process(cat1, cat2)), whole-catalogue, no chunking",
           "binning": "Log, 1000 bins, 2-300 arcmin, sep_units='arcmin'", "weight": "catalogue w", "flip": "none",
           "bin_slop": args.bin_slop, "angle_slop": f"TreeCorr default ({treecorr.GGCorrelation(kw).config.get('angle_slop')})",
           "subsample": args.subsample or None, "git_head": _git_head()}
    calls = man.get("calls", []) + [{"argv": sys.argv[1:], "treecorr": treecorr.__version__, "threads": args.threads,
                                     "start": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu_count": os.cpu_count()}]

    def write_man():
        atomic_write_text(mf, json.dumps({**top, "pairs": rows, "calls": calls, "last_call_seconds": time.time() - t0}, indent=2))
    print(f"threads={args.threads} (cpu_count={os.cpu_count()}) treecorr={treecorr.__version__} pairs={[pname(p) for p in pairs]}", flush=True)
    for p in pairs:
        f = out / f"{pname(p)}.npz"
        if f.exists() and not args.force:
            if pair_file_ok(f):
                if pname(p) not in rows:        # finished earlier but not in the manifest: rebuild the row from the file
                    d = np.load(f)
                    rows[pname(p)] = {"seconds": float(d["seconds"]), "npairs": float(d["npairs"].sum()), "weight": float(d["weight"].sum()),
                                      "threads": float(d["threads"]), "treecorr": str(d["treecorr_version"]),
                                      "nobj1": float(d["nobj1"]), "nobj2": float(d["nobj2"]), "recovered_from_npz": True}
                    write_man()
                print(f"[{time.time()-t0:7.0f}s] {pname(p)} exists, skipped", flush=True); continue
            print(f"[{time.time()-t0:7.0f}s] {pname(p)} exists but is unreadable/incomplete: re-measuring", flush=True)
        cats, src = {}, {}
        for b in sorted(set(p)):
            rec = xim.load_bin(b)
            bf = xim.WORK / f"bin{b-1}.npz"
            src[b] = {"file": str(bf), "size": bf.stat().st_size, "mtime": bf.stat().st_mtime} if bf.exists() else {"file": str(bf)}
            src[b]["radec_dtype"] = str(rec["ra"].dtype)    # on-disk precision (TreeCorr converts to float64 anyway)
            if args.subsample:
                sel = np.random.default_rng(args.seed + b).choice(rec["ra"].size, args.subsample, replace=False)
                rec = {k: rec[k][np.sort(sel)] for k in ("ra", "dec", "e1", "e2", "w")}
            cats[b] = make_treecorr_cat(rec)
            print(f"  loaded bin{b}: {cats[b].nobj:,} galaxies", flush=True); del rec
        t1 = time.time()
        gg = treecorr.GGCorrelation(kw)
        gg.process(cats[p[0]]) if p[0] == p[1] else gg.process(cats[p[0]], cats[p[1]])
        arr = corr_to_arrays(gg)
        arr.update(seconds=np.float64(time.time() - t1), treecorr_version=np.array(treecorr.__version__),
                   pairing=np.array("unordered"), bin_slop=np.float64(args.bin_slop), threads=np.float64(args.threads),
                   nobj1=np.float64(cats[p[0]].nobj), nobj2=np.float64(cats[p[1]].nobj))
        arr.update(drv.slop_record(args.bin_slop, None))
        arr.update(angle_slop_effective=np.float64(getattr(gg, "angle_slop", np.nan)), shear_convention=np.array("e1,e2 as stored, no flip"),
                   radec_dtype=np.array(",".join(src[b]["radec_dtype"] for b in sorted(set(p)))))
        tmp = f.with_name(f.stem + ".tmp.npz")      # atomic: a killed write never leaves a final-named partial file
        np.savez(tmp, **arr); os.replace(tmp, f)
        rows[pname(p)] = {"seconds": float(arr["seconds"]), "npairs": float(arr["npairs"].sum()), "weight": float(arr["weight"].sum()),
                          "threads": int(args.threads), "treecorr": treecorr.__version__, "nobj1": int(cats[p[0]].nobj), "nobj2": int(cats[p[1]].nobj),
                          "catalogues": {str(b): v for b, v in src.items()}}
        write_man()
        print(f"[{time.time()-t0:7.0f}s] {pname(p)} {arr['seconds']:.1f}s npairs={arr['npairs'].sum():.4e} "
              f"w={arr['weight'].sum():.4e}", flush=True)
        del gg, cats, arr
    write_man()
    print(f"done {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ evaluation
def load_context():
    """Official products + COSEBIs filters, loaded once.  Row order of the FITS files is asserted pair-major."""
    from kidspipe.theory.cosebis import CosebisFilters
    fx, cb = of.load_official_xipm_fits(), of.load_official_cosebis()
    assert np.array_equal(fx["pair"].ravel(), np.repeat(np.arange(of.N_PAIR), N_ANG))   # loader order == file order
    from astropy.io import fits
    for fn, ext, n in (("KiDS_Legacy_xipm.fits", "xiP", N_ANG), ("KiDS_Legacy_cosebis.fits", "En", of.N_MODES)):
        with fits.open(of.OFFICIAL / fn) as h:      # covariance rows follow FILE order: it must equal the loader's order
            t = h[ext].data
            pidx = {q: k for k, q in enumerate(of.PAIRS)}
            fp = np.array([pidx[(int(a) - 1, int(b) - 1)] for a, b in zip(t["BIN1"], t["BIN2"])])
            assert np.array_equal(fp, np.repeat(np.arange(of.N_PAIR), n)), f"{fn} not pair-major"
    sd = None
    if SALV.exists():
        amp = json.loads(SALV.read_text())["step2_amplitude"]
        sd = {k: np.asarray(amp[f"{k}_raw"]["A_pair_err_frac"], float) for k in ("xip", "xim")}
    return {"asc": of.load_official_xi(), "fx": fx, "cb": cb, "sd_frac": sd,
            "filt": CosebisFilters(2.0, 300.0, source="mpmath", dps=80)}


def pair_blocks(ctx, p):
    """Official data/covariance rows for 1-based pair p: 9-bin xi+/xi- values + sigma, E_n, B_n cov blocks."""
    K = pair_index(p); cov = ctx["fx"]["cov"]; n = of.N_PAIR * N_ANG
    ie = K * of.N_MODES + np.arange(of.N_MODES)
    return {"K": K, "val9": {"xip": ctx["fx"]["xiP"][K], "xim": ctx["fx"]["xiM"][K]},
            "sig9": {"xip": np.sqrt(np.diag(cov)[K * N_ANG:(K + 1) * N_ANG]),
                     "xim": np.sqrt(np.diag(cov)[n + K * N_ANG:n + (K + 1) * N_ANG])},
            "En": ctx["cb"]["En"][K], "Bn": ctx["cb"]["Bn"][K],
            "CE": ctx["cb"]["cov"][np.ix_(ie, ie)], "CB": ctx["cb"]["cov"][np.ix_(ie + 420, ie + 420)]}


def rebin9(r, w, y):
    """Weighted (own npairs) rebin of fine bins onto the 9 official bins; scale invariant in w."""
    o = np.zeros(N_ANG)
    for c in range(N_ANG):
        m = (r >= E9[c]) & (r < (E9[c + 1] if c < N_ANG - 1 else 300.01))
        o[c] = np.sum(w[m] * y[m]) / np.sum(w[m])
    return o


def mfac(p):
    return (1 + M[p[0] - 1]) * (1 + M[p[1] - 1])


def evaluate_pair(pair, ours, ordered, ctx, theirs=None):
    """All gates for one 1-based pair.  ours: dict with r_nom, xip, xim (RAW, m-uncorrected), npairs, weight;
    ordered: True if npairs counts each pair twice (chunk-assembled ordered file)."""
    K = pair_index(pair); blk = pair_blocks(ctx, pair)
    asc = {k: ctx["asc"][k][K:K + 1] for k in ("xip", "xim", "sigma_xip", "sigma_xim", "npairs_weighted")}
    th_, w_ = np.asarray(ours["r_nom"], float), np.asarray(ours["npairs"], float)
    raw = {k: np.asarray(ours[k], float) for k in ("xip", "xim")}
    mc = {k: raw[k] / mfac(pair) for k in raw}
    R = {"pair": pname(pair), "ordered": bool(ordered), "m_factor": mfac(pair)}
    # A1 (A1b-noise per pair: boxcar/white vs threshold; const-A_p test needs several pairs -> evaluate_a1_const)
    z, A = curves_z(raw["xip"][None], raw["xim"][None], asc); rat = boxcar_ratio(z)
    R["A1"] = {"A_p_xip": float(A["xip"][0]), "A_p_xim": float(A["xim"][0]),
               "boxcar_over_white": [float(rat[0]), float(rat[1])],
               "pass": bool(rat.max() <= A1B_THRESH["boxcar_over_white_max"]), "threshold": A1B_THRESH["boxcar_over_white_max"]}
    # A3: 9-bin max|Delta|/sigma_official, weighting by OUR npairs (same convention as dr5_pilot_bin6_exact.py)
    R["A3"] = {}
    for k in ("xip", "xim"):
        d = (rebin9(th_, w_, mc[k]) - blk["val9"][k]) / blk["sig9"][k]
        R["A3"][k] = {"d_over_sigma": d.tolist(), "max": float(np.abs(d).max()), "pass": a3_pass(float(np.abs(d).max()))}
    R["A3"]["threshold"] = A3_MAX_ABS_OVER_SIGMA
    R["A3"]["pass"] = bool(R["A3"]["xip"]["pass"] and R["A3"]["xim"]["pass"])
    # B1 / B3
    from kidspipe.measure.xi2cosebis import e_from_xi_binned
    E, B = e_from_xi_binned(th_, mc["xip"][None], mc["xim"][None], ctx["filt"], nmodes=20, scheme="edges", unit="rad")
    E, B = E[0], B[0]; dE = E - blk["En"]; CE, CB = blk["CE"], blk["CB"]; sE = np.sqrt(np.diag(CE))
    R["B1"] = {"En_minus_official_over_sigma": (dE / sE).tolist(), "max_En_6": float(np.abs(dE / sE)[:6].max()),
               "max_En_20": float(np.abs(dE / sE).max())}
    R["Bn"] = {}
    for nm in (6, 20):
        R["B1"][f"chi2_En_{nm}"] = float(dE[:nm] @ np.linalg.solve(CE[:nm, :nm], dE[:nm]))
        c = float(B[:nm] @ np.linalg.solve(CB[:nm, :nm], B[:nm])); co = float(blk["Bn"][:nm] @ np.linalg.solve(CB[:nm, :nm], blk["Bn"][:nm]))
        R["Bn"][str(nm)] = {"chi2": c, "pte": float(chi2dist.sf(c, nm)), "chi2_official": co, "pte_official": float(chi2dist.sf(co, nm))}
    if theirs is not None:
        R["vs_theirs"] = compare_theirs(pair, ours, ordered, theirs, ctx, asc)
    return R


def compare_theirs(pair, ours, ordered, theirs, ctx, asc):
    """theirs: dict with r_nom, xip, xim (raw), npairs.  Fine-bin comparison + their own A3/E_n (not recursive)."""
    out = {}
    blk, th_ = pair_blocks(ctx, pair), np.asarray(ours["r_nom"], float)
    for k in ("xip", "xim"):
        a, b = np.asarray(ours[k], float), np.asarray(theirs[k], float)
        z = (a - b) / asc["sigma_" + k][0]          # per fine bin, in the official ascii sigma (xi crosses zero: no relative diff)
        d9 = (rebin9(th_, np.asarray(ours["npairs"], float), a - b) / mfac(pair)) / blk["sig9"][k]
        out[k] = {"z_rms": float(np.sqrt(np.mean(z ** 2))), "z_max": float(np.abs(z).max()),
                  "z_lag1": float(np.corrcoef(z[:-1], z[1:])[0, 1]), "max9_over_sigma_official": float(np.abs(d9).max()),
                  "amp_ratio_weighted": float(np.sum(ours["weight"] * a) / np.sum(ours["weight"] * b))}
    na, nb = np.asarray(ours["npairs"], float) / (2.0 if ordered else 1.0), np.asarray(theirs["npairs"], float)
    out["npairs_bin_rel_rms"] = float(np.sqrt(np.mean(((na - nb) / np.maximum(nb, 1.0)) ** 2)))
    out["npairs_ratio_unordered"] = float(np.sum(ours["npairs"]) / (2.0 if ordered else 1.0) / np.sum(theirs["npairs"]))
    t = evaluate_pair(pair, theirs, False, ctx)
    out["theirs_A3_xip"], out["theirs_A3_xim"] = t["A3"]["xip"]["max"], t["A3"]["xim"]["max"]
    out["theirs_En_max_20"], out["theirs_En_max_6"] = t["B1"]["max_En_20"], t["B1"]["max_En_6"]
    out["theirs_A1_boxcar_over_white"] = t["A1"]["boxcar_over_white"]
    return out


def evaluate_a1_const(pairs, oursd, ctx):
    """A1b-noise over the available pairs, incl. the one-constant A_p test (pte) with the SALV bootstrap errors.
    Needs >= 2 pairs; the pte has dof n_pairs-1."""
    if ctx["sd_frac"] is None or len(pairs) < 2:
        return None
    ks = [pair_index(p) for p in pairs]
    asc = {k: ctx["asc"][k][ks] for k in ("xip", "xim", "sigma_xip", "sigma_xim", "npairs_weighted")}
    sd = {k: ctx["sd_frac"][k][ks] for k in ("xip", "xim")}
    a = a1b_noise(np.array([oursd[p]["xip"] for p in pairs]), np.array([oursd[p]["xim"] for p in pairs]), asc, sd)
    a["pairs"] = [pname(p) for p in pairs]
    return a


def load_theirs(pair):
    f = THEIRS / f"xipm_bin{pair[0]}_bin{pair[1]}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {"r_nom": np.asarray(d["r_nom"], float), "xip": np.asarray(d["xip_raw"], float), "xim": np.asarray(d["xim_raw"], float),
            "npairs": np.asarray(d["npairs"], float), "weight": np.asarray(d["weight"], float)}


def load_ours(path):
    d = np.load(path)
    o = {k: np.asarray(d[k], float) for k in ("r_nom", "xip", "xim", "npairs", "weight")}
    ordered = ("pairing" in d.files and str(d["pairing"]) == "ordered") or ("fold" in d.files)
    return o, bool(ordered)


def fmt(R):
    g = lambda b: "PASS" if b else "FAIL"
    b6 = R["Bn"]["6"]; b20 = R["Bn"]["20"]
    s = (f"| {R['pair']} | {max(R['A1']['boxcar_over_white']):.3f} {g(R['A1']['pass'])} (boxcar only) | "
         f"{R['A3']['xip']['max']:.4f} / {R['A3']['xim']['max']:.4f} {g(R['A3']['pass'])} | "
         f"{R['B1']['max_En_6']:.3f} / {R['B1']['max_En_20']:.3f} | {R['B1']['chi2_En_6']:.3f} / {R['B1']['chi2_En_20']:.3f} | "
         f"{b6['chi2']:.1f} (off {b6['chi2_official']:.1f}) / {b20['chi2']:.1f} (off {b20['chi2_official']:.1f}) |")
    v = R.get("vs_theirs")
    if v:
        s += (f" {v['xip']['z_rms']:.3f}, {v['xip']['max9_over_sigma_official']:.3f} / {v['xim']['z_rms']:.3f}, {v['xim']['max9_over_sigma_official']:.3f} | {v['npairs_ratio_unordered']:.4f} | "
              f"{v['theirs_A3_xip']:.4f} / {v['theirs_A3_xim']:.4f} | {v['theirs_En_max_20']:.3f} |")
    return s


def _jdefault(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def do_eval(args):
    ctx = load_context()
    if args.source == "theirs":
        out = OUTROOT / "theirs_eval"
    else:
        out = OUTROOT / (f"b{args.bin_slop:g}" + (f"_sub{args.subsample}" if args.subsample else ""))
    res, oursd, missing = {}, {}, []
    requested = parse_pairs(args.pairs)
    for p in requested:
        if args.source == "theirs":
            o, ordered = load_theirs(p), False
            if o is None:
                missing.append(pname(p)); print(f"WARNING: {pname(p)} missing in theirs", flush=True); continue
        else:
            f = out / f"{pname(p)}.npz"
            if not f.exists() or not pair_file_ok(f):
                missing.append(pname(p)); print(f"WARNING: {pname(p)} missing/unreadable in {out}", flush=True); continue
            o, ordered = load_ours(f)
        oursd[p] = o
        res[pname(p)] = evaluate_pair(p, o, ordered, ctx, None if args.source == "theirs" else load_theirs(p))
        print(fmt(res[pname(p)]), flush=True)
    if not res:
        sys.exit(f"ERROR: no pair file found for {args.pairs} in {out}; nothing written")   # before any output file is touched
    out.mkdir(parents=True, exist_ok=True)
    a1c = evaluate_a1_const(list(oursd), oursd, ctx)
    complete = not missing
    npair_gate = 21
    partial = f"partial set: {len(oursd)} of {npair_gate} pairs; gate calibrated on 21 pairs / 42 curves"
    if a1c is not None:
        a1c["verdict"] = ("INCOMPLETE" if not complete else ("PASS" if a1c["pass"] else "FAIL")) + (f" ({partial})" if len(oursd) < npair_gate else "")
    (out / "eval.json").write_text(json.dumps({"bin_slop": args.bin_slop, "source": args.source, "pairs": res, "A1_const": a1c,
                                               "requested": [pname(p) for p in requested], "missing": missing, "complete": complete,
                                               "thresholds": {"A1B_THRESH": A1B_THRESH, "A3": A3_MAX_ABS_OVER_SIGMA}},
                                              indent=1, default=_jdefault))
    hdr = ("| pair | A1 boxcar max boxcar/white (<= %.2f; const-A_p test is below, not per pair) | A3 max|d|/sig xi+ / xi- (< %.2f) | E_n-off max/sig 6m / 20m | chi2 E_n 6 / 20 | "
           "B_n null chi2 6 (official) / 20 (official) |" % (A1B_THRESH["boxcar_over_white_max"], A3_MAX_ABS_OVER_SIGMA))
    sep = "|---|---|---|---|---|---|"
    tv = any("vs_theirs" in r for r in res.values())
    if tv:
        hdr += " ours-vs-theirs fine z rms, 9-bin max/sigma_off: xi+ / xi- | npairs ratio | theirs A3 xi+ / xi- | theirs E_n20 max |"; sep += "---|---|---|---|"
    lines = [f"# Unordered pair evaluation ({len(res)} pairs), source={args.source}, bin_slop={args.bin_slop}", "", hdr, sep] + [fmt(r) for r in res.values()]
    if a1c:
        lines += ["", f"A1 one-constant A_p test over {len(a1c['pairs'])} pairs: xi+ pte {a1c['xip']['const_pte']:.3f} (C {a1c['xip']['C']:.3f}), "
                  f"xi- pte {a1c['xim']['const_pte']:.3f} (C {a1c['xim']['C']:.3f}); need pte > {A1B_THRESH['const_pte_min']}; "
                  f"A1 overall {a1c['verdict']}"]
    if missing:
        lines += ["", f"**INCOMPLETE: requested pairs missing: {', '.join(missing)}**"]
    lines += ["", "A3 = 9-bin max|Delta|/sigma_official after m-correction (1+m_i)(1+m_j) (m from dr5_adopt_01.py M, Wright+25 Table 1)."]
    lines += ["E_n and B_n are compared with the official values for information only: neither has a numeric pass/fail threshold.",
              "ours-vs-theirs is a code-equivalence check (theirs: the earlier pipeline, also TreeCorr 5.1.4 at bin_slop 1.5 on the same "
              "extract), not independent evidence of agreement with the release (TreeCorr 4.2.3); the release comparison is A3.",
              "Gate power (verify21/REFUTER.md): only A3 sees a common amplitude error (~1 %: xi*1.01 fails 2/21 pairs, xi*1.02 13/21); "
              "A1 boxcar and the const-A_p test do not."]
    (out / "EVAL.md").write_text("\n".join(lines) + "\n")
    print("wrote", out / "eval.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("--pairs", default="autos"); ap.add_argument("--bin-slop", type=float, default=1.5)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="TreeCorr threads (default: os.cpu_count())"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--source", choices=["ours", "theirs"], default="ours")
    ap.add_argument("--subsample", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(); (do_run if a.cmd == "run" else do_eval)(a)
