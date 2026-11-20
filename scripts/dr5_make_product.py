"""Assemble the DR5 production product (PI option (a), 2026-11-02) and compute the spec's full-vector checks on it.

Product = verified TreeCorr 5.1.4, bin_slop 1.5, angle_slop default (0.1), unordered pairs, float64 positions, all 21 pairs
(runs/dr5_measure/xi_unordered/b1.5/Bin{I}_Bin{J}.npz).  The raw xi is m-UNcorrected; the product also carries the m-corrected xi,
the 9-bin rebinned m-corrected xi and E_n/B_n (modes 1-20), all with exactly the evaluator's conventions
(scripts/dr5_measure_xi_unordered.py: mfac, rebin9, e_from_xi_binned scheme='edges', unit='rad').

  python scripts/dr5_make_product.py make      # product npz + README + fullvector_checks.json + FULLVECTOR.md
  python scripts/dr5_make_product.py stepd     # optional step D (model chi2 of our E_n 1-6), merged into fullvector_checks.json

Outputs: <runs>/dr5_measure/product/{dr5_xipm_cosebis_binslop1p5.npz, README.md, fullvector_checks.json, FULLVECTOR.md}
with <runs> = $KIDSPIPE_RUNS (default <root>/runs).
Read-only inputs: the pair npz files, the official release ($KIDSPIPE_OFFICIAL), dr5_measure/adoption/adoption.json and
dr5_measure/b1_integration.json, and (for a direct joint-chi2 comparison) the earlier pipeline's xi ($KIDSPIPE_THEIRS; optional,
skipped when absent).  Each input is looked up with kidspipe.paths.find_result: <runs>/... first, then the copy shipped in
<root>/runs/..., so a Colab session with KIDSPIPE_RUNS on Drive still finds the shipped adoption.json / b1_integration.json
(regenerate them with dr5_b1_integration.py; adoption.json needs the earlier pipeline's files and is shipped instead).
"""
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2dist

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kidspipe import paths  # noqa: E402
from kidspipe.measure import official as of  # noqa: E402
from kidspipe.measure.a1check import A3_MAX_ABS_OVER_SIGMA, a3_pass, curves_z  # noqa: E402
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402

_spec = importlib.util.spec_from_file_location("unord", ROOT / "scripts/dr5_measure_xi_unordered.py")
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)

SRC = paths.find_result("dr5_measure/xi_unordered/b1.5")          # input: the 21 pair files (yours, else the shipped copy)
OUT = paths.runs_dir() / "dr5_measure" / "product"                # output directory
PRODUCT = OUT / "dr5_xipm_cosebis_binslop1p5.npz"                 # where `make` writes the product
ADOPT = paths.find_result("dr5_measure/adoption/adoption.json")       # input (shipped; needs the earlier pipeline to regenerate)
B1JSON = paths.find_result("dr5_measure/b1_integration.json")         # input (dr5_b1_integration.py, or shipped)


def product_path():
    """The product to READ (yours if you ran `make`, else the shipped one); evaluated at call time."""
    return paths.find_result("dr5_measure/product/dr5_xipm_cosebis_binslop1p5.npz")
PAIRS1 = [(i + 1, j + 1) for i, j in of.PAIRS]          # official pair order, 1-based
NM = of.N_MODES
PUB_PTE = {6: 0.0635, 20: 0.0902}                       # published B_n null PTEs (126 / 420 points)


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ assemble
def assemble(ctx):
    filt = ctx["filt"]
    A = {k: [] for k in ("r_nom", "xip_raw", "xim_raw", "npairs", "weight", "xip", "xim", "xip9", "xim9", "En", "Bn")}
    names, sizes, shas, tcv, angs, dts, flat = [], [], [], set(), set(), set(), set()
    for p in PAIRS1:
        f = SRC / f"{u.pname(p)}.npz"
        d = np.load(f)
        assert str(d["pairing"]) == "unordered" and float(d["bin_slop"]) == 1.5, f
        tcv.add(str(d["treecorr_version"])); angs.add(float(d["angle_slop_effective"])); dts.update(str(d["radec_dtype"]).split(","))
        r = np.asarray(d["r_nom"], float)
        assert r.shape == (of.N_THETA_FINE,) and np.allclose(r, of.fine_bin_centres(), rtol=1e-10, atol=0), f
        raw = {k: np.asarray(d[k], float) for k in ("xip", "xim")}
        mf = u.mfac(p)
        mc = {k: raw[k] / mf for k in raw}
        w = np.asarray(d["npairs"], float)
        E, B = e_from_xi_binned(r, mc["xip"][None], mc["xim"][None], filt, nmodes=NM, scheme="edges", unit="rad")
        A["r_nom"].append(r); A["xip_raw"].append(raw["xip"]); A["xim_raw"].append(raw["xim"])
        A["npairs"].append(w); A["weight"].append(np.asarray(d["weight"], float))
        A["xip"].append(mc["xip"]); A["xim"].append(mc["xim"])
        A["xip9"].append(u.rebin9(r, w, mc["xip"])); A["xim9"].append(u.rebin9(r, w, mc["xim"]))
        A["En"].append(E[0]); A["Bn"].append(B[0])
        names.append(f.name); sizes.append(f.stat().st_size); shas.append(sha256(f))
    assert len(tcv) == 1 and len(angs) == 1 and len(dts) == 1, (tcv, angs, dts)
    A = {k: np.array(v) for k, v in A.items()}
    prov = {"bin_slop": 1.5, "angle_slop_effective": angs.pop(), "treecorr_version": tcv.pop(),
            "position_dtype": dts.pop(), "pairing": "unordered", "binning": "Log, 1000 bins, 2-300 arcmin (sep_units arcmin)",
            "m_gold": u.M, "m_source": "Wright+25 (A&A 703, A158) Table 1, same list as scripts/dr5_adopt_01.py",
            "m_correction": "xi_corr = xi_raw / ((1+m_i)(1+m_j))", "rebin9": "npairs-weighted over fine bins, 9 log bins 2-300 arcmin",
            "cosebis": "kidspipe.measure.xi2cosebis.e_from_xi_binned(scheme='edges', unit='rad'), filters mpmath 2-300 arcmin, 20 modes, on m-corrected xi",
            "release_treecorr": "4.2.3 (release used bin_slop 1.5); this product is TreeCorr 5.1.4",
            "producer": "scripts/dr5_make_product.py", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    return A, prov, names, sizes, shas


def save_product(A, prov, names, sizes, shas):
    OUT.mkdir(parents=True, exist_ok=True)
    mf = np.array([u.mfac(p) for p in PAIRS1])
    tmp = PRODUCT.with_name(PRODUCT.stem + ".tmp.npz")
    np.savez(tmp, pair_ids=np.array(PAIRS1, int), m_factor=mf, m_gold=np.array(u.M), r_nom=A["r_nom"], xip_raw=A["xip_raw"],
             xim_raw=A["xim_raw"], xip=A["xip"], xim=A["xim"], npairs=A["npairs"], weight=A["weight"],
             theta9_edges=u.E9, xip9=A["xip9"], xim9=A["xim9"], En=A["En"], Bn=A["Bn"],
             bin_slop=np.float64(prov["bin_slop"]), angle_slop_effective=np.float64(prov["angle_slop_effective"]),
             treecorr_version=np.array(prov["treecorr_version"]), position_dtype=np.array(prov["position_dtype"]),
             source_files=np.array(names), source_sizes=np.array(sizes, np.int64), source_sha256=np.array(shas),
             provenance_json=np.array(json.dumps(prov)))
    tmp.replace(PRODUCT)


def validate_product(d, fx=None):
    """Structural + identity checks on a loaded product (dict-like).  Raises AssertionError on any failure.
    A pair swapped or shuffled in the fine arrays is caught by (i) m-correction identity, (ii) the stored 9-bin arrays being
    the rebin of the fine arrays and (iii) every pair's 9-bin xi agreeing with ITS OWN official FITS row to < the A3 threshold."""
    fx = fx or of.load_official_xipm_fits()
    n = of.N_PAIR
    assert np.array_equal(d["pair_ids"], np.array(PAIRS1)), "pair order"
    for k in ("r_nom", "xip_raw", "xim_raw", "xip", "xim", "npairs", "weight"):
        assert d[k].shape == (n, of.N_THETA_FINE), (k, d[k].shape)
    for k in ("xip9", "xim9"):
        assert d[k].shape == (n, 9), k
    assert d["En"].shape == (n, NM) and d["Bn"].shape == (n, NM)
    assert d["m_factor"].shape == (n,)
    assert np.allclose(d["m_factor"], [(1 + d["m_gold"][i - 1]) * (1 + d["m_gold"][j - 1]) for i, j in PAIRS1], rtol=1e-14)
    for k in ("xip", "xim"):
        assert np.allclose(d[k], d[k + "_raw"] / d["m_factor"][:, None], rtol=1e-13, atol=0), f"{k} != raw / m_factor"
    for k in range(n):
        assert str(d["source_files"][k]) == f"Bin{PAIRS1[k][0]}_Bin{PAIRS1[k][1]}.npz", f"source file name pair {k}"
        for s in ("xip", "xim"):
            re9 = u.rebin9(d["r_nom"][k], d["npairs"][k], d[s][k])
            assert np.allclose(re9, d[s + "9"][k], rtol=1e-12, atol=0), f"{s}9 not the rebin of the fine xi (pair {k})"
            sig = np.sqrt(np.diag(fx["cov"]))
            sg = sig[k * 9:(k + 1) * 9] if s == "xip" else sig[n * 9 + k * 9:n * 9 + (k + 1) * 9]
            ref = fx["xiP"][k] if s == "xip" else fx["xiM"][k]
            dm = np.abs((re9 - ref) / sg).max()
            assert a3_pass(float(dm)), f"pair {PAIRS1[k]} {s}: max|d|/sigma = {dm:.3f} vs its official FITS row (wrong pair?)"


def validate_cosebis(d, filt):
    """The stored E_n, B_n must be what the evaluator's integrator gives from the stored m-corrected fine xi."""
    for k in range(of.N_PAIR):
        E, B = e_from_xi_binned(d["r_nom"][k], d["xip"][k][None], d["xim"][k][None], filt, nmodes=NM, scheme="edges", unit="rad")
        assert np.allclose(E[0], d["En"][k], rtol=1e-10, atol=0), f"En of pair {k} is not the integral of its xi"
        assert np.allclose(B[0], d["Bn"][k], rtol=1e-10, atol=0), f"Bn of pair {k} is not the integral of its xi"


def validate_sources(d, srcdir):
    """Product fine arrays and recorded hashes must match the pair files they were built from."""
    for k, name in enumerate(str(s) for s in d["source_files"]):
        f = Path(srcdir) / name
        assert f.stat().st_size == int(d["source_sizes"][k]), f"{name}: size"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == str(d["source_sha256"][k]), f"{name}: sha256"
        with np.load(f) as s:
            for key, pk in (("xip", "xip_raw"), ("xim", "xim_raw"), ("npairs", "npairs"), ("weight", "weight")):
                assert np.array_equal(np.asarray(s[key], float), d[pk][k]), f"{name}: {key} differs from the product"


# ------------------------------------------------------------------------------------------------ checks
def vec378(x9):
    """(21, 9) per-pair arrays -> pair-major 189 vector."""
    return np.asarray(x9, float).reshape(-1)


def a3_block(d, fx, x9p, x9m):
    cov, out = fx["cov"], {}
    dv = {"xip": vec378(x9p) - fx["xiP"].reshape(-1), "xim": vec378(x9m) - fx["xiM"].reshape(-1)}
    for j, k in enumerate(("xip", "xim")):
        idx = np.arange(j * 189, (j + 1) * 189); C = cov[np.ix_(idx, idx)]
        c = float(dv[k] @ np.linalg.solve(C, dv[k])); mx = float(np.abs(dv[k] / np.sqrt(np.diag(C))).max())
        out[k] = {"chi2": c, "dof": 189, "chi2_per_dof": c / 189, "max_abs_over_sigma": mx, "pass": a3_pass(mx)}
    full = np.concatenate([dv["xip"], dv["xim"]])
    c = float(full @ np.linalg.solve(cov, full))
    out["joint_378"] = {"chi2": c, "dof": 378, "chi2_per_dof": c / 378,
                        "max_abs_over_sigma": float(np.abs(full / np.sqrt(np.diag(cov))).max())}
    return out


def cosebis_block(E, B, cb):
    """B2 (E_n, B_n difference vs official) and B3 (B_n null, ours and official), modes 1-6 / 1-20, official COVMAT blocks."""
    cov = cb["cov"]; res = {"B2": {}, "B3": {}}
    for nm in (6, 20):
        idx = np.array([p * NM + m for p in range(of.N_PAIR) for m in range(nm)])
        CE, CB = cov[np.ix_(idx, idx)], cov[np.ix_(idx + 420, idx + 420)]     # B_n block offset 21*20 = 420
        dE, dB = (E - cb["En"]).ravel()[idx], (B - cb["Bn"]).ravel()[idx]
        r = {}
        for lab, dd, C in (("En", dE, CE), ("Bn", dB, CB)):
            c = float(dd @ np.linalg.solve(C, dd))
            r[lab] = {"chi2": c, "dof": len(idx), "chi2_per_dof": c / len(idx), "max_abs_over_sigma": float(np.abs(dd / np.sqrt(np.diag(C))).max())}
        c = float(dB @ np.linalg.solve(CE, dB))     # B_n diff with the E_n block: NOT valid; only to reproduce adoption.json B1 Bn (bug there)
        r["Bn_with_En_block_adoption_bug"] = {"chi2": c, "max_abs_over_sigma": float(np.abs(dB / np.sqrt(np.diag(CE))).max())}
        res["B2"][f"{nm}m"] = r
        bo, bu = cb["Bn"].ravel()[idx], B.ravel()[idx]
        co, cu = float(bo @ np.linalg.solve(CB, bo)), float(bu @ np.linalg.solve(CB, bu))
        res["B3"][f"{nm}m"] = {"dof": len(idx), "chi2_official": co, "pte_official": float(chi2dist.sf(co, len(idx))),
                               "chi2_ours": cu, "pte_ours": float(chi2dist.sf(cu, len(idx))), "published_pte": PUB_PTE[nm]}
    return res


def a2_block(d, ctx):
    """A2: per pair and overall median/max |Delta xi|/sigma_xi over the 1000 fine bins against the official ascii.
    The official ascii is m-UNcorrected, so OUR RAW xi is compared.  Two amplitude treatments:
    'A=1'   : no rescaling (raw ours - ascii);
    'A_p'   : ours - A_p * ascii with A_p the per-pair, per-statistic ratio of weighted sums (A1b convention, curves_z)."""
    asc = ctx["asc"]
    z, A = curves_z(d["xip_raw"], d["xim_raw"], asc)           # (42, 1000): rows 0-20 xi+, 21-41 xi-
    out = {"A_p": {k: A[k].tolist() for k in A}, "per_pair": {}, "overall": {}}
    zs = {"A=1": np.concatenate([(d["xip_raw"] - asc["xip"]) / asc["sigma_xip"], (d["xim_raw"] - asc["xim"]) / asc["sigma_xim"]]),
          "A_p": z}
    for ti, (stat, sl) in enumerate((("xip", slice(0, 21)), ("xim", slice(21, 42)))):
        for tr, zz in zs.items():
            a = np.abs(zz[sl])
            out["overall"][f"{stat}|{tr}"] = {"median": float(np.median(a)), "max": float(a.max())}
            for k, p in enumerate(PAIRS1):
                out["per_pair"].setdefault(u.pname(p), {})[f"{stat}|{tr}"] = {"median": float(np.median(a[k])), "max": float(a[k].max())}
    return out


def load_theirs_arrays():
    """Earlier pipeline's xi (read-only) -> (21,1000) raw xi/m_factor arrays, official pair order, for a direct joint comparison."""
    T = {"xip": [], "xim": [], "npairs": [], "r": []}
    for p in PAIRS1:
        t = u.load_theirs(p)
        if t is None:
            return None
        T["xip"].append(t["xip"] / u.mfac(p)); T["xim"].append(t["xim"] / u.mfac(p)); T["npairs"].append(t["npairs"]); T["r"].append(t["r_nom"])
    return {k: np.array(v) for k, v in T.items()}


def compare_adoption(ours, adopt):
    """Every full-vector number vs adoption.json (theirs).  Returns rows and the list of differences above 1e-6 relative."""
    rows = []
    def add(name, a, b):
        rows.append({"quantity": name, "ours": a, "theirs": b, "abs_diff": a - b, "rel_diff": (a - b) / abs(b) if b else float("nan")})
    for s, key in (("xip", "A3_xiP"), ("xim", "A3_xiM")):
        for f in ("chi2", "max_abs_over_sigma"):
            add(f"A3 {s} {f}", ours["A3"][s][f], adopt[key][f])
    for nm in ("6m", "20m"):
        for lab in ("En", "Bn"):
            for f in ("chi2", "max_abs_over_sigma"):
                src = ours["B2"][nm]["Bn_with_En_block_adoption_bug"] if lab == "Bn" else ours["B2"][nm][lab]
                add(f"B2/B1 {lab} {nm} {f}" + (" [ours recomputed with the En block, as adoption did]" if lab == "Bn" else ""), src[f],
                    adopt["B1_En_Bn_vs_official"][f"{lab}_{nm}"][f])
        for f in ("chi2_official", "pte_official", "chi2_ours", "pte_ours"):
            add(f"B3 {nm} {f}", ours["B3"][nm][f], adopt["B3_Bn_null"][nm][f])
    return rows


def run_make():
    t0 = time.time()
    for need in (SRC, ADOPT, B1JSON):
        if not need.exists():
            sys.exit(f"missing input {need}: run the earlier chain steps first (dr5_measure_xi_unordered.py run, dr5_b1_integration.py; "
                     f"adoption.json ships with the repository)")
    if SRC != paths.runs_dir() / "dr5_measure" / "xi_unordered" / "b1.5":
        print(f"NOTE: no pair files under {paths.runs_dir()}; using the shipped copy {SRC}", flush=True)
    ctx = u.load_context()
    A, prov, names, sizes, shas = assemble(ctx)
    save_product(A, prov, names, sizes, shas)
    d = dict(np.load(PRODUCT)); fx, cb = ctx["fx"], ctx["cb"]
    validate_product(d, fx)
    res = {"product": str(PRODUCT), "provenance": prov, "m_values_source": prov["m_source"], "thresholds": {"A3_max_abs_over_sigma": A3_MAX_ABS_OVER_SIGMA}}
    res["A2"] = a2_block(d, ctx)
    res["A3"] = a3_block(d, fx, d["xip9"], d["xim9"])
    res.update(cosebis_block(d["En"], d["Bn"], cb))
    for nm, pub in PUB_PTE.items():
        res["B3"][f"{nm}m"]["official_reproduces_published"] = bool(abs(res["B3"][f"{nm}m"]["pte_official"] - pub) < 5e-5)
    assert all(res["B3"][k]["official_reproduces_published"] for k in res["B3"]), "official B_n null does not give the published PTE: wrong covariance block (bug)"
    b1 = json.loads(B1JSON.read_text())
    res["B1_note"] = {"source": str(B1JSON), "trial": "meanr|edges|rad", "what": "official ascii xi (no m-correction) -> our E_n vs official",
                      "chi2_En_126": b1["trials"]["meanr|edges|rad"]["chi2_Emode_126_nomod"],
                      "chi2_En_420": b1["trials"]["meanr|edges|rad"]["chi2_Emode_420_nomod"],
                      "max_abs_dE_over_sigma": b1["trials"]["meanr|edges|rad"]["max_abs_dE_over_sigma"],
                      "note": "quoted, not recomputed; the adopted integration test is also adoption.json B1 (their xi, m-corrected)"}
    adopt = json.loads(ADOPT.read_text())
    rows = compare_adoption(res, adopt)
    res["vs_adoption"] = {"rows": rows, "max_rel_diff": float(np.nanmax([abs(r["rel_diff"]) for r in rows])),
                          "n_rel_diff_gt_1e-6": [r["quantity"] for r in rows if abs(r["rel_diff"]) > 1e-6]}
    th = load_theirs_arrays()
    if th is not None:                       # direct joint-378 chi2 for theirs (adoption.json only holds per-statistic blocks)
        x9 = {s: np.array([u.rebin9(th["r"][k], th["npairs"][k], th[s][k]) for k in range(21)]) for s in ("xip", "xim")}
        res["A3_theirs_direct"] = a3_block(d, fx, x9["xip"], x9["xim"])
        res["A3_joint_378_ours_minus_theirs_chi2"] = res["A3"]["joint_378"]["chi2"] - res["A3_theirs_direct"]["joint_378"]["chi2"]
        res["max_abs_fine_xi_diff_ours_minus_theirs_m_corrected"] = float(max(np.abs(d["xip"] - th["xip"]).max(), np.abs(d["xim"] - th["xim"]).max()))
    res["runtime_s"] = time.time() - t0
    (OUT / "fullvector_checks.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res)
    write_readme(prov)
    print(f"wrote product + checks in {res['runtime_s']:.0f}s; max rel diff vs adoption {res['vs_adoption']['max_rel_diff']:.3g}")


def write_md(r):
    a2, a3 = r["A2"], r["A3"]
    L = ["# FULLVECTOR: spec full-vector checks on OUR bin_slop 1.5 xi (product option (a))", "",
         "Script `scripts/dr5_make_product.py`; numbers in `fullvector_checks.json`. Our xi = verified TreeCorr 5.1.4, bin_slop 1.5, angle_slop 0.1, unordered, float64 positions. "
         f"m-values (Wright+25 Table 1, as in `scripts/dr5_adopt_01.py`): {u.M}. The release used TreeCorr 4.2.3; agreement with 'theirs' below is code equivalence, not independent release validation.", "",
         "## A2: fine-bin |Delta xi|/sigma_xi against the official ascii (1000 bins, official `sigma_xip/xim`)", "",
         "**Amplitude handling.** The official fine ascii is m-uncorrected and has an unexplained scale (RESUME: A_p ~1.17 for ours/ascii, ascii amplitude ~0.85) and is a blinded realisation of the same sample, "
         "not a truth reference; the numbers are for information, no gate. Our RAW (m-uncorrected) xi is compared. Two treatments: `A=1` (no rescaling) and `A_p` (per-pair, per-statistic ratio-of-weighted-sums scale, the A1b convention, "
         "`ours - A_p * ascii`). The scale, not a shape difference, dominates the `A=1` numbers.", "",
         "| statistic | treatment | overall median | overall max |", "|---|---|---|---|"]
    for k, v in a2["overall"].items():
        s, t = k.split("|"); L.append(f"| {s} | {t} | {v['median']:.3f} | {v['max']:.3f} |")
    L += ["", "Per pair (median / max): xi+ A=1 | xi+ A_p | xi- A=1 | xi- A_p", "", "| pair | A_p xi+ | A_p xi- | xi+ A=1 | xi+ A_p | xi- A=1 | xi- A_p |", "|---|---|---|---|---|---|---|"]
    for k, p in enumerate(PAIRS1):
        n = u.pname(p); q = a2["per_pair"][n]; f = lambda x: f"{x['median']:.2f} / {x['max']:.2f}"
        L.append(f"| {n} | {a2['A_p']['xip'][k]:.3f} | {a2['A_p']['xim'][k]:.3f} | {f(q['xip|A=1'])} | {f(q['xip|A_p'])} | {f(q['xim|A=1'])} | {f(q['xim|A_p'])} |")
    j = a3["joint_378"]
    L += ["", "## A3 full vector (9-bin m-corrected xi vs `KiDS_Legacy_xipm.fits`, COVMAT 378x378)", "",
          f"- chi2(ours - official) = {j['chi2']:.4f} / 378 = {j['chi2_per_dof']:.2e}; max|d|/sigma = {j['max_abs_over_sigma']:.4f}.",
          f"- Per statistic (189 points each, covariance sub-block): xi+ chi2 {a3['xip']['chi2']:.4f} (/189 = {a3['xip']['chi2_per_dof']:.2e}), max {a3['xip']['max_abs_over_sigma']:.4f}; "
          f"xi- chi2 {a3['xim']['chi2']:.4f}, max {a3['xim']['max_abs_over_sigma']:.4f}.",
          f"- GATE (adopted 2026-10-29): max|Delta|/sigma < {A3_MAX_ABS_OVER_SIGMA}: xi+ {'PASS' if a3['xip']['pass'] else 'FAIL'}, xi- {'PASS' if a3['xim']['pass'] else 'FAIL'}. "
          "chi2/dof < 1 against the official covariance is NOT a gate (RESUME bug 6: the covariance contains cosmic variance that cancels between measurements of the same galaxies, so chi2 is nearly blind to smooth errors, and 10 % amplitude error gives only ~12/378).", "",
          "## B2: E_n and B_n difference vs official (official COVMAT 840x840 = [E_n | B_n], B block offset 420)", "",
          "| modes | points | chi2 E_n | max/sigma E_n | chi2 B_n | max/sigma B_n |", "|---|---|---|---|---|---|"]
    for nm in ("6m", "20m"):
        b = r["B2"][nm]
        L.append(f"| {nm[:-1]} | {b['En']['dof']} | {b['En']['chi2']:.4f} | {b['En']['max_abs_over_sigma']:.4f} | {b['Bn']['chi2']:.4f} | {b['Bn']['max_abs_over_sigma']:.4f} |")
    L += ["", "E_n/B_n comparisons have no numeric gate in the spec; shown for information (as in the evaluator).", "",
          "## B3: B_n null (chi2 vs zero, B_n covariance block)", "", "| modes | points | official chi2 | official PTE (published) | ours chi2 | ours PTE |", "|---|---|---|---|---|---|"]
    for nm in ("6m", "20m"):
        b = r["B3"][nm]
        L.append(f"| {nm[:-1]} | {b['dof']} | {b['chi2_official']:.3f} | {b['pte_official']:.4f} ({b['published_pte']}) | {b['chi2_ours']:.3f} | {b['pte_ours']:.4f} |")
    ok = all(r["B3"][k]["official_reproduces_published"] for k in r["B3"])
    L += ["", f"Official reproduces the published PTEs 0.0635 / 0.0902: {'yes' if ok else 'NO (bug)'}.", "",
          "## B1 (integration check on official input), quoted, not recomputed", "",
          f"`b1_integration.json`, trial `meanr|edges|rad` (official ascii xi, no m-correction -> our E_n vs official): chi2 = {r['B1_note']['chi2_En_126']:.4f}/126, "
          f"{r['B1_note']['chi2_En_420']:.4f}/420, max|dE|/sigma = {r['B1_note']['max_abs_dE_over_sigma']:.3f}. (The adopted integration test, their m-corrected xi, is in `adoption.json` B1.)", "",
          "## Comparison with `adoption/adoption.json` (theirs)", ""]
    v = r["vs_adoption"]
    L += ["**Latent bug in `scripts/dr5_adopt_01.py` B1 (found here, not edited):** its B_n-vs-official chi2 and max/sigma use the E_n covariance block (`cov_cb[np.ix_(idx, idx)]`, no +420 offset) "
          "for B_n; the same mistake its B3 section documents. adoption.json `B1_En_Bn_vs_official.Bn_6m/Bn_20m` are therefore not valid B_n numbers (0.0169/126, 0.0644/420, max 0.0299, 0.0520). "
          "With the correct B_n block our B2 is "
          f"{r['B2']['6m']['Bn']['chi2']:.4f}/126, {r['B2']['20m']['Bn']['chi2']:.4f}/420, max {r['B2']['6m']['Bn']['max_abs_over_sigma']:.4f}, {r['B2']['20m']['Bn']['max_abs_over_sigma']:.4f}. "
          "The Bn rows in the table below are recomputed with the E_n block to compare like with like. The A3, E_n and B3 numbers in adoption.json are unaffected.", ""]
    L += [f"Max relative difference over {len(v['rows'])} numbers: {v['max_rel_diff']:.3g}. Entries above 1e-6 relative: {v['n_rel_diff_gt_1e-6'] or 'none'}.", "",
          "| quantity | ours | theirs (adoption.json) | abs diff | rel diff |", "|---|---|---|---|---|"]
    L += [f"| {x['quantity']} | {x['ours']:.6g} | {x['theirs']:.6g} | {x['abs_diff']:.2e} | {x['rel_diff']:.2e} |" for x in v["rows"]]
    if "A3_theirs_direct" in r:
        L += ["", f"Direct joint-378 A3 chi2 (adoption.json has per-statistic blocks only): ours {j['chi2']:.6f}, theirs (recomputed here from their xi, same code) {r['A3_theirs_direct']['joint_378']['chi2']:.6f}; "
              f"max |m-corrected fine xi ours - theirs| = {r['max_abs_fine_xi_diff_ours_minus_theirs_m_corrected']:.2e}."]
    L += ["", "Ours = theirs here is code equivalence (both TreeCorr 5.1.4, bin_slop 1.5, same extract), not independent release validation (release: TreeCorr 4.2.3). "
          "Only A3 is amplitude-sensitive (~1 %: xi*1.01 fails 2/21 pairs). Exact (bin_slop 0) was not run for all pairs."]
    if "D" in r:
        d = r["D"]
        L += ["", "## Step D (optional): model chi2 of our E_n, modes 1-6", "",
              f"`LegacyCosebisModel` at official best-posterior chain row {d['row']}: chi2 official data vector {d['chi2_official_data']:.3f} (quoted 127.23), "
              f"chi2 our E_n {d['chi2_ours']:.3f} (126 points, official En covariance block). Difference {d['chi2_ours'] - d['chi2_official_data']:+.3f}. Runtime {d['runtime_s']:.0f} s."]
    (OUT / "FULLVECTOR.md").write_text("\n".join(L) + "\n")


def write_readme(prov):
    (OUT / "README.md").write_text(f"""# DR5 cosmic-shear xi+/- and COSEBIs product (bin_slop 1.5)

File: `dr5_xipm_cosebis_binslop1p5.npz` (produced by `scripts/dr5_make_product.py`, checks in `FULLVECTOR.md`).
Measured from the KiDS-Legacy gold catalogue (our extract), TreeCorr {prov['treecorr_version']}, bin_slop 1.5, angle_slop {prov['angle_slop_effective']} (TreeCorr default),
unordered pairs (auto: one catalogue; cross: two), catalogue `weight`, e1,e2 as stored (no flip), positions {prov['position_dtype']}.
The exact (bin_slop 0) measurement was NOT run for all pairs; this is the PI-chosen option (a).

## Ordering and units
- 21 tomographic pairs in official order `(1,1),(1,2),...,(1,6),(2,2),...,(6,6)` (`pair_ids`, 1-based; same as the KiDS-Legacy FITS, pair-major).
- Fine arrays are `(21, 1000)`: 1000 log bins 2-300 arcmin. `r_nom` is the geometric bin centre in **arcmin**. xi is dimensionless.
- `xip_raw, xim_raw`: m-UNcorrected xi+/xi-. `npairs`: unordered pair counts; `weight`: sum of w_i w_j.
- `m_factor = (1+m_i)(1+m_j)`, m_gold = {prov['m_gold']} (Wright+25 Table 1). `xip, xim` = raw / m_factor: the multiplicative shear-calibration correction applied
  to both statistics (shear scales as 1+m per galaxy, xi as the product). Use `xip, xim` for science and the *_raw arrays only to compare with m-uncorrected references (e.g. the official fine ascii).
- `xip9, xim9` `(21, 9)`: m-corrected xi rebinned to the 9 official bins (`theta9_edges`, log 2-300 arcmin) with npairs weights; compare with `KiDS_Legacy_xipm.fits`.
- `En, Bn` `(21, 20)`: COSEBIs modes 1-20 from the m-corrected fine xi (scheme `edges`, theta in radians, filters for 2-300 arcmin). Compare with `KiDS_Legacy_cosebis.fits`.
- Provenance: `bin_slop`, `angle_slop_effective`, `treecorr_version`, `position_dtype`, `source_files`, `source_sizes`, `source_sha256`, `provenance_json`.

## Caveats
- The official release used TreeCorr 4.2.3 at bin_slop 1.5; this run is TreeCorr 5.1.4. Agreement with the release rests on the 9-bin comparison (max |Delta|/sigma < 0.1, see FULLVECTOR.md);
  equality with the earlier pipeline's xi is code equivalence (same library version, settings and extract), not independent validation.
- Only the 9-bin comparison is sensitive to a common amplitude error, with resolution of order 1 %.
- bin_slop 1.5 differs from the exact estimate (largest effect on xi-, E_n up to ~0.4 sigma over 20 modes in Bin6_Bin6); see `runs/dr5_measure/angle_slop/ANGLE_SLOP.md`.
- The official fine ascii has an unexplained amplitude (ours/ascii ~1.17) and is not a truth reference.
- E_n/B_n comparisons with the release have no numeric gate. Errors are not included: use the official covariances (COVMAT of the FITS files).
- Covariance of E_n for a mode set: pair-major, mode fastest; the B_n block of the 840x840 COVMAT is offset by 420.
""")


# ------------------------------------------------------------------------------------------------ step D
def run_stepd():
    t0 = time.time()
    from kidspipe.legacy import LegacyCosebisLikelihood, LegacyCosebisModel, chain_samples, load_chain
    d = np.load(product_path())
    chain = load_chain()
    post = np.where(np.isfinite(chain[:, 43]), chain[:, 43], -np.inf)
    row = int(np.argmax(post)); s = chain_samples(chain, [row])[0]
    model = LegacyCosebisModel(nthreads=min(8, os.cpu_count() or 1), cache_dir=str(paths.runs_dir() / "legacy" / "cache"), cache_size=1)
    like = LegacyCosebisLikelihood()
    mu = model.mu(s)
    cb = of.load_official_cosebis()
    assert np.allclose(like.d, cb["En"][:, :6].ravel(), rtol=1e-6, atol=0), "likelihood data vector != official En"
    c_off = like.chi2(mu)
    like.d = np.asarray(d["En"][:, :6], float).ravel()           # same covariance, our data vector
    c_us = like.chi2(mu)
    f = OUT / "fullvector_checks.json"
    src_checks = f if f.exists() else paths.find_result("dr5_measure/product/fullvector_checks.json")
    r = json.loads(src_checks.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    r["D"] = {"row": row, "chi2_official_data": c_off, "chi2_ours": c_us, "chi2_chain_quoted": float(s["chi2_off"]), "dof": 126, "runtime_s": time.time() - t0}
    f.write_text(json.dumps(r, indent=1, default=float)); write_md(r)
    print(f"D: row {row} chi2 official data {c_off:.3f}, ours {c_us:.3f}, chain {s['chi2_off']:.3f}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    {"make": run_make, "stepd": run_stepd}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
