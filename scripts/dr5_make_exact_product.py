"""Exact-run product (the PI's "additional part", sessions 2026-11-09 .. 2026-11-11) and its comparison with the bin_slop 1.5 product.

Exact = TreeCorr bin_slop 0, angle_slop 0, 1000 log bins 2-300 arcmin, float64 positions, 16-chunk decomposition
(scripts/dr5_measure_xi_chunked.py; chunk files runs/dr5_measure/xi_chunked/chunks/, assembled by `--assemble --nchunks 16`
into runs/dr5_measure/xi_chunked/assembled/ns/).  Gate decision (PI, 2026-11-02): the exact run is NOT gated; it is judged by
its difference from the bin_slop 1.5 product, reported per pair, with no pass/fail.

Conventions handled here (the driver is left unchanged):
* Pair counts: the chunked driver correlates a catalogue with itself for auto pairs, so its assembled npairs/weight are ORDERED
  counts (2x the unordered convention of the bin_slop 1.5 product) and varxi is 0.5x.  Auto pairs are converted here
  (npairs/2, weight/2, varxi*2).  xi+/xi- are unchanged (weights cancel; both orderings land in the same bin at bin_slop 0).
* meanr: the driver's assembly copies meanr from the FIRST chunk file only (RESUME bug 17).  Here meanr is recomputed from the
  chunk files as the weight-weighted mean sum(fold*w*meanr)/sum(fold*w).

  python scripts/dr5_make_exact_product.py        # product npz + README + exact_vs_b1p5.json + EXACT_VS_B1P5.md

Outputs: <runs>/dr5_measure/product_exact/ (written to temporary names, validated, then renamed together), <runs> = $KIDSPIPE_RUNS
(default <root>/runs).
Read-only inputs: the chunk and assembled files under <runs>/dr5_measure/xi_chunked/ (your own run: the chunk files are not
shipped), the bin_slop 1.5 product (yours, else the copy shipped in <root>/runs; kidspipe.paths.find_result), the release
($KIDSPIPE_OFFICIAL).  The step-D model chi2 uses min(8, os.cpu_count()) threads.
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
from kidspipe.measure.xi2cosebis import e_from_xi_binned  # noqa: E402

_spec = importlib.util.spec_from_file_location("mkprod", ROOT / "scripts/dr5_make_product.py")
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)
u = mk.u
_spec = importlib.util.spec_from_file_location("chunked", ROOT / "scripts/dr5_measure_xi_chunked.py")
ch = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ch)

RUNS = paths.runs_dir()
CHUNKS = RUNS / "dr5_measure" / "xi_chunked" / "chunks"
ASM = RUNS / "dr5_measure" / "xi_chunked" / "assembled" / "ns"
OUT = RUNS / "dr5_measure" / "product_exact"
PRODUCT = OUT / "dr5_xipm_cosebis_exact.npz"
B15 = mk.product_path()                 # input: your bin_slop 1.5 product, else the shipped one


def _rel(p):
    """Path relative to the repository root when inside it (the default layout), else the full path."""
    try:
        return Path(p).relative_to(ROOT)
    except ValueError:
        return Path(p)
PAIRS1 = mk.PAIRS1
NM = of.N_MODES
NCH = 16
EDGE_BINS = 5          # outermost fine bins (theta ~ 292.6-300 arcmin) for the edge decomposition
EDGES = np.exp(np.linspace(np.log(2.0), np.log(300.0), of.N_THETA_FINE + 1))


def sha256(f):
    return mk.sha256(f)


def pkey(p):
    return f"{p[0]}-{p[1]}"


# ------------------------------------------------------------------------------------------------ inputs
def check_chunks():
    """Every pair complete, every chunk file exact (bin_slop 0, angle_slop 0), assembled file newer than its chunks.
    Also returns the weight-weighted meanr per pair accumulated over the chunk files (the driver's assembled meanr is not)."""
    rows, meanr = {}, {}
    for p in PAIRS1:
        same = p[0] == p[1]
        d = CHUNKS / u.pname(p)
        names = sorted(f.name for f in d.glob("c*.npz"))
        expect = sorted(f"c{a:02d}_{b:02d}.npz" for a, b in ch.chunk_pairs(NCH, same))
        assert names == expect, f"{u.pname(p)}: chunk files differ from chunk_pairs({NCH})"
        slops, newest = set(), 0.0
        swr, sw = np.zeros(of.N_THETA_FINE), np.zeros(of.N_THETA_FINE)
        for n in names:
            with np.load(d / n) as z:
                slops.add(ch.chunk_slops(z))
                fold, w = float(z["fold"]), np.asarray(z["weight"], float)
                swr += fold * w * np.asarray(z["meanr"], float); sw += fold * w
            newest = max(newest, (d / n).stat().st_mtime)
        assert slops == {(0.0, 0.0)}, f"{u.pname(p)}: chunk settings {slops}, expected exact (0, 0)"
        assert np.all(sw > 0), f"{u.pname(p)}: a fine bin has zero weight"
        af = ASM / f"{u.pname(p)}.npz"
        assert af.stat().st_mtime >= newest, f"{af.name} is older than its chunk files (stale assembly)"
        rows[pkey(p)] = {"chunk_files": len(names)}
        meanr[pkey(p)] = swr / sw
    idx = json.loads((ASM / "index.json").read_text())
    assert idx["nchunks"] == NCH and Path(idx["source"]) == CHUNKS, idx
    return rows, meanr


def assemble(ctx, meanr):
    filt = ctx["filt"]
    A = {k: [] for k in ("r_nom", "meanr", "xip_raw", "xim_raw", "varxi", "npairs", "weight", "xip", "xim", "xip9", "xim9", "En", "Bn")}
    names, sizes, shas = [], [], []
    for p in PAIRS1:
        f = ASM / f"{u.pname(p)}.npz"
        d = np.load(f)
        assert float(d["angle_slop"]) == 0.0, f
        r = np.asarray(d["r_nom"], float)
        assert r.shape == (of.N_THETA_FINE,) and np.allclose(r, of.fine_bin_centres(), rtol=1e-10, atol=0), f
        conv = 0.5 if p[0] == p[1] else 1.0                       # ordered -> unordered for auto pairs
        npairs, weight = conv * np.asarray(d["npairs"], float), conv * np.asarray(d["weight"], float)
        varxi = np.asarray(d["varxi"], float) / conv
        raw = {k: np.asarray(d[k], float) for k in ("xip", "xim")}
        mf = u.mfac(p)
        mc = {k: raw[k] / mf for k in raw}
        E, B = e_from_xi_binned(r, mc["xip"][None], mc["xim"][None], filt, nmodes=NM, scheme="edges", unit="rad")
        A["r_nom"].append(r); A["meanr"].append(meanr[pkey(p)])
        A["xip_raw"].append(raw["xip"]); A["xim_raw"].append(raw["xim"]); A["varxi"].append(varxi)
        A["npairs"].append(npairs); A["weight"].append(weight)
        A["xip"].append(mc["xip"]); A["xim"].append(mc["xim"])
        A["xip9"].append(u.rebin9(r, npairs, mc["xip"])); A["xim9"].append(u.rebin9(r, npairs, mc["xim"]))
        A["En"].append(E[0]); A["Bn"].append(B[0])
        names.append(f.name); sizes.append(f.stat().st_size); shas.append(sha256(f))
    A = {k: np.array(v) for k, v in A.items()}
    import treecorr
    prov = {"bin_slop": 0.0, "angle_slop": 0.0, "nchunks": NCH,
            "treecorr_version_installed": treecorr.__version__,
            "treecorr_note": "chunk files do not record the version; TreeCorr 5.1.4 was installed for all three sessions (unchanged; no pip actions)",
            "position_dtype": "float64 (driver load_bin reads the float64 extract; checked 2026-11-09)",
            "pairing": "unordered (auto pairs: the driver's ordered counts halved, varxi doubled; xi unchanged)",
            "meanr": "weight-weighted mean over the chunk files, sum(fold*w*meanr)/sum(fold*w) (the driver's assembled meanr is the first chunk's only)",
            "binning": "Log, 1000 bins, 2-300 arcmin (sep_units arcmin)",
            "sessions": {"S1": "2026-11-09 10:49:51 .. 2026-11-10 00:29:04", "S2": "2026-11-10 00:44:30 .. 13:37:56 (WSL restart ~08:50, resumed 08:56:23)",
                         "S3": "2026-11-10 19:55:18 .. 2026-11-11 08:48:42"},
            "m_gold": u.M, "m_correction": "xi_corr = xi_raw / ((1+m_i)(1+m_j))", "rebin9": "npairs-weighted over fine bins, 9 log bins 2-300 arcmin",
            "cosebis": "kidspipe.measure.xi2cosebis.e_from_xi_binned(scheme='edges', unit='rad'), filters mpmath 2-300 arcmin, 20 modes, on m-corrected xi",
            "gate": "none (PI 2026-11-02): differences from the bin_slop 1.5 product are reported, no pass/fail",
            "producer": "scripts/dr5_make_exact_product.py", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    return A, prov, names, sizes, shas


def save_tmp(A, prov, names, sizes, shas):
    """Write the product to a temporary name; it is renamed to PRODUCT only after validation (run())."""
    OUT.mkdir(parents=True, exist_ok=True)
    mf = np.array([u.mfac(p) for p in PAIRS1])
    tmp = PRODUCT.with_name(PRODUCT.stem + ".tmp.npz")
    np.savez(tmp, pair_ids=np.array(PAIRS1, int), m_factor=mf, m_gold=np.array(u.M), r_nom=A["r_nom"], meanr=A["meanr"],
             xip_raw=A["xip_raw"], xim_raw=A["xim_raw"], xip=A["xip"], xim=A["xim"], varxi=A["varxi"], npairs=A["npairs"],
             weight=A["weight"], theta9_edges=u.E9, xip9=A["xip9"], xim9=A["xim9"], En=A["En"], Bn=A["Bn"],
             bin_slop=np.float64(0.0), angle_slop=np.float64(0.0), pairing=np.array("unordered"),
             source_files=np.array(names), source_sizes=np.array(sizes, np.int64), source_sha256=np.array(shas),
             provenance_json=np.array(json.dumps(prov)))
    return tmp


# ------------------------------------------------------------------------------------------------ validation (no gate)
def validate_exact_product(d, b15, fx=None):
    """Structural and identity checks.  Raises AssertionError on any failure.  Deliberately NOT the A3 gate (the exact run is
    not gated); pair identity is checked by each pair's 9-bin xi being closest, in official sigma, to ITS OWN bin_slop 1.5 row."""
    fx = fx or of.load_official_xipm_fits()
    n = of.N_PAIR
    assert np.array_equal(d["pair_ids"], np.array(PAIRS1)), "pair order"
    for k in ("r_nom", "meanr", "xip_raw", "xim_raw", "xip", "xim", "varxi", "npairs", "weight"):
        assert d[k].shape == (n, of.N_THETA_FINE), (k, d[k].shape)
    for k in ("xip9", "xim9"):
        assert d[k].shape == (n, 9), k
    assert d["En"].shape == (n, NM) and d["Bn"].shape == (n, NM)
    assert float(d["bin_slop"]) == 0.0 and float(d["angle_slop"]) == 0.0 and str(d["pairing"]) == "unordered"
    assert np.array_equal(np.asarray(d["m_gold"], float), np.asarray(u.M, float)), "m_gold != the pipeline's m values (u.M)"
    assert np.allclose(d["m_factor"], [u.mfac(p) for p in PAIRS1], rtol=1e-14, atol=0), "m_factor != (1+m_i)(1+m_j) of u.M"
    assert np.array_equal(d["r_nom"], b15["r_nom"]), "fine binning differs from the bin_slop 1.5 product"
    assert np.all((d["meanr"] >= EDGES[:-1] * (1 - 1e-9)) & (d["meanr"] <= EDGES[1:] * (1 + 1e-9))), "meanr outside its fine bin"
    for k in ("xip", "xim"):
        assert np.allclose(d[k], d[k + "_raw"] / d["m_factor"][:, None], rtol=1e-13, atol=0), f"{k} != raw / m_factor"
    # unordered convention: total pair counts within 1e-4 of the bin_slop 1.5 (unordered) product for every pair, autos included
    ratio = d["npairs"].sum(1) / b15["npairs"].sum(1)
    assert np.all(np.abs(ratio - 1) < 1e-4), f"npairs total / bin_slop 1.5 = {ratio} (auto pairs not converted to unordered?)"
    sig = np.sqrt(np.diag(fx["cov"]))
    for k in range(n):
        assert str(d["source_files"][k]) == f"Bin{PAIRS1[k][0]}_Bin{PAIRS1[k][1]}.npz", f"source file name pair {k}"
        for s in ("xip", "xim"):
            re9 = u.rebin9(d["r_nom"][k], d["npairs"][k], d[s][k])
            assert np.allclose(re9, d[s + "9"][k], rtol=1e-12, atol=0), f"{s}9 not the rebin of the fine xi (pair {k})"
            off = 0 if s == "xip" else n * 9
            dist = [np.abs((re9 - b15[s + "9"][q]) / sig[off + q * 9: off + (q + 1) * 9]).max() for q in range(n)]
            assert int(np.argmin(dist)) == k, f"pair {PAIRS1[k]} {s}: closest bin_slop 1.5 row is pair {PAIRS1[int(np.argmin(dist))]} (mix-up?)"


def validate_cosebis(d, filt):
    mk.validate_cosebis(d, filt)


def validate_sources(d, srcdir=ASM):
    """Product arrays and recorded hashes must match the assembled files (with the auto-pair conversion)."""
    for k, name in enumerate(str(s) for s in d["source_files"]):
        f = Path(srcdir) / name
        assert f.stat().st_size == int(d["source_sizes"][k]), f"{name}: size"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == str(d["source_sha256"][k]), f"{name}: sha256"
        conv = 0.5 if PAIRS1[k][0] == PAIRS1[k][1] else 1.0
        with np.load(f) as s:
            for key, pk, c in (("xip", "xip_raw", 1.0), ("xim", "xim_raw", 1.0), ("npairs", "npairs", conv), ("weight", "weight", conv),
                               ("varxi", "varxi", 1.0 / conv)):
                assert np.array_equal(c * np.asarray(s[key], float), d[pk][k]), f"{name}: {key} differs from the product"


# ------------------------------------------------------------------------------------------------ comparison
def chi2_xi(dp9, dm9, cov):
    vp, vm = np.asarray(dp9, float).reshape(-1), np.asarray(dm9, float).reshape(-1)
    out = {}
    for j, (lab, v) in enumerate((("xip_189", vp), ("xim_189", vm))):
        idx = np.arange(j * 189, (j + 1) * 189); C = cov[np.ix_(idx, idx)]
        out["chi2_" + lab] = float(v @ np.linalg.solve(C, v))
        out["max_abs_over_sigma_" + lab[:3]] = float(np.abs(v / np.sqrt(np.diag(C))).max())
    full = np.concatenate([vp, vm])
    out["chi2_joint_378"] = float(full @ np.linalg.solve(cov, full))
    return out


def cos_index(nm):
    return np.array([p * NM + m for p in range(of.N_PAIR) for m in range(nm)])


def chi2_cos(dE, dB, cov):
    out = {}
    for nm in (6, 20):
        idx = cos_index(nm); CE, CB = cov[np.ix_(idx, idx)], cov[np.ix_(idx + 420, idx + 420)]
        e, b = np.asarray(dE).ravel()[idx], np.asarray(dB).ravel()[idx]
        out[f"E_{nm}m_{len(idx)}"] = float(e @ np.linalg.solve(CE, e))
        out[f"B_{nm}m_{len(idx)}"] = float(b @ np.linalg.solve(CB, b))
    return out


def bnull(B, cov):
    out = {}
    for nm in (6, 20):
        idx = cos_index(nm); CB = cov[np.ix_(idx + 420, idx + 420)]; b = np.asarray(B).ravel()[idx]
        c = float(b @ np.linalg.solve(CB, b))
        out[f"chi2_{nm}m_{len(idx)}"] = c; out[f"pte_{nm}m"] = float(chi2dist.sf(c, len(idx)))
    return out


def range_masks(r):
    """Angular ranges for the decomposition of the COSEBIs difference (they partition the 1000 fine bins)."""
    i = np.arange(r.size)
    edge = i >= r.size - EDGE_BINS
    return {"below_100": r < 100.0, "100_to_edge": (r >= 100.0) & ~edge, "edge": edge}


def filter_weighting(r, filt):
    """Response of E_n / B_n to a unit change of xi+ or xi- in one fine bin, for every bin (COSEBIs are linear in xi).
    Returned per mode: mean |response| over the EDGE_BINS outermost bins divided by the median |response| over all bins."""
    I, Z = np.eye(r.size), np.zeros((r.size, r.size))
    out = {}
    for lab, (a, b) in (("xip", (I, Z)), ("xim", (Z, I))):
        E, B = e_from_xi_binned(r, a, b, filt, nmodes=NM, scheme="edges", unit="rad")      # (1000, 20): row = bin, column = mode
        for nm, R in (("E", E), ("B", B)):
            A = np.abs(R)
            out[f"{nm}_from_{lab}"] = (A[-EDGE_BINS:].mean(0) / np.median(A, axis=0)).tolist()
    return out


def compare(d, b15, ctx):
    fx, cb, filt = ctx["fx"], ctx["cb"], ctx["filt"]
    n = of.N_PAIR
    sig = np.sqrt(np.diag(fx["cov"]))
    S9 = {"xip": sig[:n * 9].reshape(n, 9), "xim": sig[n * 9:].reshape(n, 9)}
    OFF9 = {"xip": fx["xiP"], "xim": fx["xiM"]}
    sE = np.sqrt(np.diag(cb["cov"]))[:420].reshape(n, NM); sB = np.sqrt(np.diag(cb["cov"]))[420:].reshape(n, NM)
    dE, dB = (d["En"] - b15["En"]) / sE, (d["Bn"] - b15["Bn"]) / sB
    per, arrays = {}, {"dxip9_over_sigma": [], "dxim9_over_sigma": [], "dEn_over_sigma": dE.tolist(), "dBn_over_sigma": dB.tolist(),
                        "dEn_edgezeroed_over_sigma": [], "dBn_edgezeroed_over_sigma": []}
    decomp = {}
    for k, p in enumerate(PAIRS1):
        row = {}
        r = d["r_nom"][k]
        for s in ("xip", "xim"):
            z = (d[s + "9"][k] - b15[s + "9"][k]) / S9[s][k]
            arrays[f"d{s}9_over_sigma"].append(z.tolist())
            row[f"{s}9_ex_b15"] = float(np.abs(z).max()); row[f"{s}9_ex_b15_bin"] = int(np.abs(z).argmax()) + 1
            row[f"{s}9_ex_off"] = float(np.abs((d[s + "9"][k] - OFF9[s][k]) / S9[s][k]).max())
            row[f"{s}9_b15_off"] = float(np.abs((b15[s + "9"][k] - OFF9[s][k]) / S9[s][k]).max())
            zf = (d[s + "_raw"][k] - b15[s + "_raw"][k]) / np.sqrt(d["varxi"][k])
            row[f"{s}_fine_median_abs_z"] = float(np.median(np.abs(zf))); row[f"{s}_fine_mean_z"] = float(zf.mean())
        for lab, Z in (("E", dE), ("B", dB)):
            row[f"{lab}6"] = float(np.abs(Z[k, :6]).max()); row[f"{lab}20"] = float(np.abs(Z[k]).max())
            row[f"{lab}20_mode"] = int(np.abs(Z[k]).argmax()) + 1
            row[f"{lab}20_signed_at_max"] = float(Z[k][np.abs(Z[k]).argmax()])
        # COSEBIs of the exact-minus-b1.5 difference restricted to angular ranges (linear, so the ranges add up exactly)
        ddp, ddm = d["xip"][k] - b15["xip"][k], d["xim"][k] - b15["xim"][k]
        parts = {}
        for name, m in range_masks(r).items():
            Ec, Bc = e_from_xi_binned(r, (ddp * m)[None], (ddm * m)[None], filt, nmodes=NM, scheme="edges", unit="rad")
            parts[name] = (Ec[0] / sE[k], Bc[0] / sB[k])
        assert np.allclose(sum(v[0] for v in parts.values()), dE[k], rtol=1e-9, atol=1e-12), f"decomposition does not add up, pair {p}"
        decomp[pkey(p)] = {name: {"E6": float(np.abs(e[:6]).max()), "E20": float(np.abs(e).max()), "B6": float(np.abs(b[:6]).max()),
                                  "B20": float(np.abs(b).max())} for name, (e, b) in parts.items()}
        zE2 = dE[k] - parts["edge"][0]; zB2 = dB[k] - parts["edge"][1]          # difference with the edge-bin part set to zero
        arrays["dEn_edgezeroed_over_sigma"].append(zE2.tolist()); arrays["dBn_edgezeroed_over_sigma"].append(zB2.tolist())
        row["E6_edgezeroed"] = float(np.abs(zE2[:6]).max()); row["E20_edgezeroed"] = float(np.abs(zE2).max())
        row["B6_edgezeroed"] = float(np.abs(zB2[:6]).max()); row["B20_edgezeroed"] = float(np.abs(zB2).max())
        row["npairs_ratio_to_b15"] = float(d["npairs"][k].sum() / b15["npairs"][k].sum())
        row["meanr_rel_to_rnom_maxabs"] = float(np.abs(d["meanr"][k] / r - 1).max())
        per[pkey(p)] = row
    full = {"xi_ex_minus_b15": chi2_xi(d["xip9"] - b15["xip9"], d["xim9"] - b15["xim9"], fx["cov"]),
            "xi_ex_minus_off": chi2_xi(d["xip9"] - fx["xiP"], d["xim9"] - fx["xiM"], fx["cov"]),
            "xi_b15_minus_off": chi2_xi(b15["xip9"] - fx["xiP"], b15["xim9"] - fx["xiM"], fx["cov"]),
            "cosebis_ex_minus_b15": chi2_cos(d["En"] - b15["En"], d["Bn"] - b15["Bn"], cb["cov"]),
            "cosebis_ex_minus_off": chi2_cos(d["En"] - cb["En"], d["Bn"] - cb["Bn"], cb["cov"]),
            "cosebis_b15_minus_off": chi2_cos(b15["En"] - cb["En"], b15["Bn"] - cb["Bn"], cb["cov"]),
            "Bnull_exact": bnull(d["Bn"], cb["cov"]), "Bnull_b15": bnull(b15["Bn"], cb["cov"]), "Bnull_official": bnull(cb["Bn"], cb["cov"])}
    for nm, pub in mk.PUB_PTE.items():                                # index/offset sanity: the release's own B_n null PTE
        assert abs(full["Bnull_official"][f"pte_{nm}m"] - pub) < 5e-5, f"official B_n null PTE {nm} modes does not reproduce {pub}"
    fw = filter_weighting(d["r_nom"][0], filt)
    return per, full, arrays, decomp, fw


def stepd(d, b15):
    """Model chi2 of E_n modes 1-6 at the official best-posterior chain row (as dr5_make_product.py stepd): official data, the
    bin_slop 1.5 product and the exact product, all with the same model vector and the official covariance."""
    t0 = time.time()
    from kidspipe.legacy import LegacyCosebisLikelihood, LegacyCosebisModel, chain_samples, load_chain
    chain = load_chain()
    post = np.where(np.isfinite(chain[:, 43]), chain[:, 43], -np.inf)
    row = int(np.argmax(post)); s = chain_samples(chain, [row])[0]
    model = LegacyCosebisModel(nthreads=min(8, os.cpu_count() or 1), cache_dir=str(RUNS / "legacy" / "cache"), cache_size=1)
    like = LegacyCosebisLikelihood()
    mu = model.mu(s)
    cb = of.load_official_cosebis()
    assert np.allclose(like.d, cb["En"][:, :6].ravel(), rtol=1e-6, atol=0), "likelihood data vector != official En"
    out = {"row": row, "dof": 126, "chi2_chain_quoted": float(s["chi2_off"]), "chi2_official_data": float(like.chi2(mu))}
    for lab, E in (("chi2_b15", b15["En"]), ("chi2_exact", d["En"])):
        like.d = np.asarray(E[:, :6], float).ravel(); out[lab] = float(like.chi2(mu))
    out["runtime_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------------------ outputs
def rng(per, key):
    v = [r[key] for r in per.values()]
    return min(v), max(v)


def edge_counts(per):
    """How the largest COSEBIs difference (modes 1-20, and 1-6) changes when its edge-bin part is set to zero."""
    out = {}
    for lab in ("E", "B"):
        for nm in ("20", "6"):
            full = np.array([r[f"{lab}{nm}"] for r in per.values()]); ez = np.array([r[f"{lab}{nm}_edgezeroed"] for r in per.values()])
            top = np.argsort(full)[::-1][:5]
            out[f"{lab}{nm}"] = {"n_halved": int(np.sum(ez < 0.5 * full)), "n_rises": int(np.sum(ez > full)),
                                 "rises": [(list(per)[i], float(full[i]), float(ez[i])) for i in np.where(ez > full)[0]],
                                 "top5": [(list(per)[i], float(full[i]), float(ez[i])) for i in top]}
    return out


def md_row(p, r):
    return (f"| {p} | {r['xip9_ex_b15']:.3f} | {r['xim9_ex_b15']:.3f} | {r['xip9_ex_off']:.3f} | {r['xim9_ex_off']:.3f} | "
            f"{r['xip9_b15_off']:.3f} | {r['xim9_b15_off']:.3f} | {r['E6']:.3f} | {r['E20']:.3f} | {r['B6']:.3f} | {r['B20']:.3f} | "
            f"{r['E20_edgezeroed']:.3f} | {r['B20_edgezeroed']:.3f} |")


def write_md(res, path):
    per, full, ec, dec, fw = res["per_pair"], res["fullvector"], res["edge"], res["decomposition"], res["filter_weighting"]
    L = ["# Exact run vs bin_slop 1.5 product (all 21 pairs)", "",
         f"Product: `{_rel(PRODUCT)}` (bin_slop 0, angle_slop 0, unordered convention). Reference: `{_rel(B15)}`.",
         "PI decision (2026-11-02): the exact run is not gated; the differences below are reported, with no pass/fail.",
         "All differences are m-corrected and in units of the release's own sigma (diagonal of the official covariance): 9-bin xi",
         "(npairs-weighted rebin, each run with its own npairs) and COSEBIs modes 1-20.", "",
         "## Per pair", "",
         f"The last two columns set the exact-minus-b1.5 xi difference to zero in the outermost {EDGE_BINS} fine bins (theta 292.6-300",
         "arcmin) before computing the COSEBIs difference; the bin_slop 1.5 values are kept there. They show how much of the COSEBIs",
         "difference comes from those bins; they are not COSEBIs of a truncated xi.", "",
         "| pair | xi+ ex-b1.5 | xi- ex-b1.5 | xi+ ex-off | xi- ex-off | xi+ b1.5-off | xi- b1.5-off | E n<=6 | E n<=20 | B n<=6 | B n<=20 | E n<=20, edge diff zeroed | B n<=20, edge diff zeroed |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [md_row(p, r) for p, r in per.items()]

    def R(k, f=".3f"):
        a, b = rng(per, k); return f"{a:{f}} to {b:{f}}"
    L += ["", f"Ranges over 21 pairs: xi+ exact-b1.5 {R('xip9_ex_b15')}, xi- {R('xim9_ex_b15')}; exact-official xi+ {R('xip9_ex_off')}, "
          f"xi- {R('xim9_ex_off')}; b1.5-official xi+ {R('xip9_b15_off')}, xi- {R('xim9_b15_off')}; E_n n<=6 {R('E6')}, n<=20 {R('E20')}; "
          f"B_n n<=6 {R('B6')}, n<=20 {R('B20')}; with the edge-bin difference zeroed E_n n<=20 {R('E20_edgezeroed')}, B_n n<=20 {R('B20_edgezeroed')}.",
          f"Fine bins: median |exact-b1.5|/sqrt(varxi) {R('xip_fine_median_abs_z')} (xi+), {R('xim_fine_median_abs_z')} (xi-); mean signed "
          f"{R('xip_fine_mean_z', '+.4f')} (xi+), {R('xim_fine_mean_z', '+.4f')} (xi-). Total pair counts exact/b1.5 {R('npairs_ratio_to_b15', '.7f')}. "
          f"max |meanr / r_nom - 1| = {rng(per, 'meanr_rel_to_rnom_maxabs')[1]:.1e} (fine-bin half-width in ln theta: {np.log(150) / 2000:.1e}).", "",
          "## Full vector (official covariance)", "",
          "chi2 of a difference uses the release covariance only as a yardstick: the exact-minus-b1.5 difference is not a draw from it,",
          "so chi2/dof << 1 is expected; it says how large the shift is relative to the statistical errors.", "",
          "| difference | xi+ (189) | xi- (189) | joint (378) | E 1-6 (126) | E 1-20 (420) | B 1-6 (126) | B 1-20 (420) |", "|---|---|---|---|---|---|---|---|"]
    for lab, xk, ck in (("exact - b1.5", "xi_ex_minus_b15", "cosebis_ex_minus_b15"), ("exact - official", "xi_ex_minus_off", "cosebis_ex_minus_off"),
                        ("b1.5 - official", "xi_b15_minus_off", "cosebis_b15_minus_off")):
        x, c = full[xk], full[ck]
        L.append(f"| {lab} | {x['chi2_xip_189']:.3f} | {x['chi2_xim_189']:.3f} | {x['chi2_joint_378']:.3f} | {c['E_6m_126']:.3f} | "
                 f"{c['E_20m_420']:.3f} | {c['B_6m_126']:.3f} | {c['B_20m_420']:.3f} |")
    L += ["", "B_n null test (chi2 of B_n with the official B covariance; not a gate here):", "",
          "| B_n of | modes 1-6: chi2 (126 points) | PTE | modes 1-20: chi2 (420 points) | PTE |", "|---|---|---|---|---|"]
    for lab, k in (("exact", "Bnull_exact"), ("bin_slop 1.5 product", "Bnull_b15"), ("official release", "Bnull_official")):
        b = full[k]
        L.append(f"| {lab} | {b['chi2_6m_126']:.2f} | {b['pte_6m']:.4f} | {b['chi2_20m_420']:.2f} | {b['pte_20m']:.4f} |")
    L += ["", f"(The official row reproduces the published PTEs {mk.PUB_PTE[6]} / {mk.PUB_PTE[20]}; that checks the covariance indexing.)", ""]
    D = res["D"]
    L += ["## Model chi2 of E_n modes 1-6 (step D)", "",
          f"`LegacyCosebisModel` at the official best-posterior chain row {D['row']}, official E_n covariance, 126 points: official data "
          f"{D['chi2_official_data']:.3f} (chain quotes {D['chi2_chain_quoted']:.2f}), bin_slop 1.5 product {D['chi2_b15']:.3f} "
          f"({D['chi2_b15'] - D['chi2_official_data']:+.3f}), exact {D['chi2_exact']:.3f} ({D['chi2_exact'] - D['chi2_official_data']:+.3f}).", ""]
    # angular decomposition summary
    def drng(part, key):
        v = [dec[p][part][key] for p in dec]; return f"{min(v):.3f} to {max(v):.3f}"
    fwE = np.array(fw["E_from_xip"] + fw["E_from_xim"]); fwB = np.array(fw["B_from_xip"] + fw["B_from_xim"])
    L += ["## Where the COSEBIs difference comes from (angular decomposition, all 21 pairs)", "",
          "COSEBIs are linear in xi, so the exact-minus-b1.5 difference splits exactly into contributions from angular ranges",
          f"(the code asserts the parts add up). Per pair, max over modes 1-20 (and 1-6) of |contribution| / sigma:", "",
          "| range | E n<=6 | E n<=20 | B n<=6 | B n<=20 |", "|---|---|---|---|---|"]
    for part, lab in (("below_100", "2-100 arcmin"), ("100_to_edge", f"100 arcmin to the edge bins"), ("edge", f"outermost {EDGE_BINS} fine bins (292.6-300 arcmin)")):
        L.append(f"| {lab} | {drng(part, 'E6')} | {drng(part, 'E20')} | {drng(part, 'B6')} | {drng(part, 'B20')} |")
    L += ["", f"Filter weighting: per unit change of xi in one fine bin, the mean |response| of a mode to the outermost {EDGE_BINS} bins is "
          f"{fwE.min():.0f} to {fwE.max():.0f} times its median response over all 1000 bins, over modes 1-20 and both xi+ and xi- "
          f"(the same for B_n, {fwB.min():.0f} to {fwB.max():.0f}: E_n and B_n use the same two filters and differ only in sign).", ""]
    xmax = max(rng(per, "xip9_ex_b15")[1], rng(per, "xim9_ex_b15")[1])
    fx_ = full["xi_ex_minus_b15"]["chi2_joint_378"]; fb_ = full["xi_b15_minus_off"]["chi2_joint_378"]
    top = lambda k: ", ".join(f"{p} {a:.2f}->{b:.2f}" for p, a, b in ec[k]["top5"])
    rises = lambda k: ", ".join(f"{p} {a:.2f}->{b:.2f}" for p, a, b in ec[k]["rises"]) or "none"
    sgn = [per[p]["E20_signed_at_max"] for p in per]
    mx_ = lambda part, keys: max(dec[p][part][k] for p in dec for k in keys)
    L += ["## Reading", "",
          f"- COSEBIs differences come from large angles: below 100 arcmin the difference contributes at most {mx_('below_100', ('E20', 'B20')):.3f}"
          f" sigma to any mode in any pair; 100 arcmin to the edge bins up to {mx_('100_to_edge', ('E20',)):.2f} (E_n) and"
          f" {mx_('100_to_edge', ('B20',)):.2f} (B_n); the outermost {EDGE_BINS} fine bins up to {mx_('edge', ('E20',)):.2f} (E_n) and"
          f" {mx_('edge', ('B20',)):.2f} (B_n).",
          f"- xi: exact differs from bin_slop 1.5 by at most {xmax:.3f} sigma per 9-bin point (joint chi2 of the difference {fx_:.2f} over 378"
          f" points). The bin_slop 1.5 product is much closer to the release (joint {fb_:.3f}), which used bin_slop 1.5, so the exact-vs-official",
          "  difference is essentially this exact-vs-b1.5 shift.",
          "- Averaged over all 1000 fine bins the signed exact-minus-b1.5 difference is zero (mean z above), i.e. no overall shift of xi.",
          "  That does not exclude local differences: see the next point.",
          f"- COSEBIs: zeroing the exact-minus-b1.5 difference in the outermost {EDGE_BINS} fine bins at least halves max|dE_n| (modes 1-20)"
          f" in {ec['E20']['n_halved']} of 21 pairs and max|dB_n| in {ec['B20']['n_halved']}. Five largest, full -> edge difference zeroed: E_n {top('E20')};"
          f" B_n {top('B20')}. It raises the value in {ec['E20']['n_rises']} (E) and {ec['B20']['n_rises']} (B) pairs, where the edge part partly cancels"
          f" the rest: E_n {rises('E20')}; B_n {rises('B20')}. For modes 1-6 it raises max|dE_n| in {ec['E6']['n_rises']} pairs ({rises('E6')}).",
          f"  The sign of the largest E_n difference varies between pairs ({sum(s > 0 for s in sgn)} positive, {sum(s < 0 for s in sgn)} negative).",
          "  The large weight of the edge bins in the COSEBI filters (above) is why a small re-binning difference there shows up in E_n/B_n.",
          "- The B_n null of the exact measurement is listed for information; no gate applies.", "",
          "Arrays for plots (per pair, 9-bin and per-mode differences in sigma, with and without the edge-bin part) are in `exact_vs_b1p5.json`."]
    path.write_text("\n".join(L) + "\n")


def write_readme(prov, path):
    t = f"""# DR5 exact xi+/xi- and COSEBIs (bin_slop 0)

`dr5_xipm_cosebis_exact.npz`: the exact TreeCorr measurement of all 21 KiDS-DR5 tomographic pairs, built by
`scripts/dr5_make_exact_product.py` from `runs/dr5_measure/xi_chunked/assembled/ns/` (chunked driver, {prov['nchunks']} RA strips).
Settings: bin_slop 0, angle_slop 0, Log 1000 bins 2-300 arcmin, float64 positions, TreeCorr {prov['treecorr_version_installed']}.
Same keys and conventions as the bin_slop 1.5 product (`../product/README.md`), plus `varxi` and `meanr`:

- `pair_ids` (21, 2) 1-based in official order; `m_factor` = (1+m_i)(1+m_j); `r_nom` (21, 1000) arcmin.
- `xip_raw, xim_raw` m-uncorrected; `xip, xim` = raw / m_factor; `xip9, xim9` npairs-weighted rebin to the 9 official bins.
- `En, Bn` (21, 20) from the m-corrected fine xi (`e_from_xi_binned`, scheme 'edges', unit 'rad').
- `npairs, weight, varxi` in the UNORDERED convention (as in the bin_slop 1.5 product). The chunked driver counts ordered pairs for
  auto pairs; those were halved (npairs, weight) and varxi doubled here. xi is unaffected.
- `meanr` (21, 1000): weight-weighted mean separation per fine bin, accumulated over the chunk files here. (The driver's
  assembled files carry the first chunk's meanr only; do not use that one.)
- `source_files/sizes/sha256`: the assembled files the product was built from.

Status: not gated (PI, 2026-11-02). Its difference from the bin_slop 1.5 product, per pair and as full vectors, is in
`EXACT_VS_B1P5.md` / `exact_vs_b1p5.json`. The release itself used bin_slop 1.5, so for comparisons with the release the
bin_slop 1.5 product is the like-for-like measurement; the exact product removes the bin_slop approximation.
Sessions: S1 {prov['sessions']['S1']}; S2 {prov['sessions']['S2']}; S3 {prov['sessions']['S3']}. Interim checks:
`runs/dr5_measure/xi_chunked/SESSION1_CHECK.md`, `SESSION2_CHECK.md`, `SESSION3_CHECK.md`.
"""
    path.write_text(t)


def run():
    t0 = time.time()
    ctx = u.load_context()
    chunks, meanr = check_chunks()
    A, prov, names, sizes, shas = assemble(ctx, meanr)
    tmp = save_tmp(A, prov, names, sizes, shas)
    outs = {PRODUCT: tmp, OUT / "exact_vs_b1p5.json": OUT / "exact_vs_b1p5.json.tmp",
            OUT / "EXACT_VS_B1P5.md": OUT / "EXACT_VS_B1P5.md.tmp", OUT / "README.md": OUT / "README.md.tmp"}
    try:
        d = dict(np.load(tmp)); b15 = dict(np.load(B15))
        validate_exact_product(d, b15, ctx["fx"])
        validate_cosebis(d, ctx["filt"])
        validate_sources(d)
        per, full, arrays, decomp, fw = compare(d, b15, ctx)
        res = {"product": str(PRODUCT), "reference": str(B15), "provenance": prov, "chunk_check": chunks, "edge_bins": EDGE_BINS,
               "per_pair": per, "fullvector": full, "edge": edge_counts(per), "decomposition": decomp, "filter_weighting": fw,
               "arrays": arrays}
        res["D"] = stepd(d, b15)
        res["runtime_s"] = time.time() - t0
        outs[OUT / "exact_vs_b1p5.json"].write_text(json.dumps(res, indent=1, default=float))
        write_md(res, outs[OUT / "EXACT_VS_B1P5.md"]); write_readme(prov, outs[OUT / "README.md"])
    except BaseException:
        for t in outs.values():
            t.unlink(missing_ok=True)
        raise
    for final, t in outs.items():                      # publish only after everything validated and was written
        t.replace(final)
    print(f"wrote {PRODUCT.name}, EXACT_VS_B1P5.md, exact_vs_b1p5.json, README.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
