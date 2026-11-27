#!/usr/bin/env python3
"""Step 0: extract the KiDS-Legacy shear catalogue into compact per-bin NPZ files.

Reads only the 7 columns we need from the 16.4 GB FITS binary table, in chunks,
and writes ONE npz per tomographic bin on the Linux filesystem (~/kids_dr5_work).

Also records the selection diagnostics needed for the n_eff / sigma_e gate.

Usage (from any directory; the repository root is found from this file's location)::

  python scripts/dr5_extract.py [--catalogue P] [--out D] [--chunk N] [--force]
  python scripts/dr5_extract.py --out D --uncompressed-dir U [--from-npz] [--bins 1,6]

Defaults come from ``kidspipe.paths``: ``--catalogue`` = ``$KIDSPIPE_DR5_CATALOGUE`` (else
``<dr5_work>/KiDS_Legacy_NS_unblind_final.fits``), ``--out`` = ``$KIDSPIPE_DR5_WORK`` (else ``~/kids_dr5_work``).
The catalogue must be the UNCOMPRESSED 16.4 GB ``.fits`` on a local disk (cfitsio would inflate a ``.fits.gz`` entirely
in memory); download once from ``paths.CATALOGUE_URL`` (7.2 GB gz) and decompress with ``gunzip`` / ``pigz -d``.

The step is skipped when all six ``bin*.npz`` and ``extract_info.json`` exist and their row counts match the catalogue,
unless ``--force``. Every output is written under a temporary name and renamed (``os.replace``), so an interrupted run
never leaves a truncated bin file.
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

CAT = paths.dr5_catalogue()
OUT = paths.dr5_work_dir()
COLS = ["RAJ2000", "DECJ2000", "PATCH", "TOMOBIN", "e1", "e2", "weight"]
CHUNK = 5_000_000
NBIN = 6


def _fail(msg: str) -> None:
    print(f"dr5_extract: {msg}", file=sys.stderr)
    sys.exit(2)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_uncompressed_bin(
    uncompressed_dir: Path,
    b: int,
    arrays: dict[str, np.ndarray],
    *,
    force: bool = False,
) -> Path:
    """Write uncompressed ``bin{b}/<col>.npy`` arrays atomically for memory-mapped loading."""
    bdir = Path(uncompressed_dir) / f"bin{int(b)}"
    bdir.mkdir(parents=True, exist_ok=True)
    for col, arr in arrays.items():
        target = bdir / f"{col}.npy"
        if target.is_file() and not force:
            continue
        tmp = bdir / f".{col}.npy.part"
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(arr))
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return bdir


def convert_npz_to_uncompressed(
    npz_dir: Path,
    uncompressed_dir: Path,
    *,
    bins: list[int] | None = None,
    force: bool = False,
) -> list[Path]:
    """Stream existing ``bin*.npz`` columns into ``bin*/<col>.npy`` without duplicating full bins in RAM."""
    npz_dir = Path(npz_dir)
    uncompressed_dir = Path(uncompressed_dir)
    indices = list(range(NBIN)) if bins is None else [int(b) for b in bins]
    out_dirs: list[Path] = []
    for b in indices:
        src = npz_dir / f"bin{b}.npz"
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found; run scripts/dr5_extract.py first")
        with np.load(src) as z:
            cols = list(z.files)
            nrows = int(z["ra"].shape[0]) if "ra" in cols else -1
            for col in cols:
                bdir = write_uncompressed_bin(uncompressed_dir, b, {col: z[col]}, force=force)
        out_dirs.append(bdir)
        print(f"bin{b}: converted n={nrows:,} ({','.join(cols)}) -> {bdir}", flush=True)
    return out_dirs


def _parse_bins_arg(bins_arg: str) -> list[int] | None:
    """Parse 1-based comma-separated tomographic bin IDs into 0-based file indices."""
    s = bins_arg.strip()
    if not s:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        b = int(tok)
        if not (1 <= b <= NBIN):
            raise ValueError(f"--bins values must be in 1..{NBIN}, got {b}")
        if (b - 1) not in out:
            out.append(b - 1)
    return out or None


def _already_done(out: Path, nrows: int) -> bool:
    """True if all 6 bin files and extract_info.json exist and are consistent with the catalogue row count."""
    info_path = out / "extract_info.json"
    if not info_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text())
        if int(info["nrows_total"]) != int(nrows):
            return False
        for b in range(NBIN):
            p = out / f"bin{b}.npz"
            if not p.is_file():
                return False
            with np.load(p) as z:
                if int(z["ra"].shape[0]) != int(info["bins"][str(b)]["n_gal"]):
                    return False
    except Exception:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", type=Path, default=CAT, help=f"uncompressed KiDS-Legacy shear catalogue (default {CAT})")
    ap.add_argument("--out", type=Path, default=OUT, help=f"directory for bin0..5.npz + extract_info.json (default {OUT})")
    ap.add_argument("--uncompressed-dir", type=Path, default=None,
                    help="also write uncompressed bin{k}/{ra,dec,e1,e2,w,north}.npy for mmap loading")
    ap.add_argument("--from-npz", action="store_true",
                    help="convert existing bin*.npz in --out to --uncompressed-dir without reading FITS")
    ap.add_argument("--bins", default="",
                    help="comma-separated 1-based tomographic bins to convert with --from-npz (default all 6)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help=f"rows read per slice (default {CHUNK:,})")
    ap.add_argument("--force", action="store_true", help="re-extract even if complete outputs already exist")
    args = ap.parse_args()
    cat = args.catalogue.expanduser()
    out = args.out.expanduser()
    udir = args.uncompressed_dir.expanduser() if args.uncompressed_dir is not None else None
    chunk = int(args.chunk)
    bins_sel = _parse_bins_arg(args.bins)

    if args.from_npz:
        if udir is None:
            _fail("--from-npz requires --uncompressed-dir")
        convert_npz_to_uncompressed(out, udir, bins=bins_sel, force=args.force)
        return

    if udir is not None and not cat.is_file() and (out / "bin0.npz").is_file():
        convert_npz_to_uncompressed(out, udir, bins=bins_sel, force=args.force)
        return

    if str(cat).endswith(".gz"):
        _fail(f"{cat} is gzip-compressed. cfitsio would inflate the whole 16.4 GB table in memory. Decompress first, e.g.\n"
              f"  gunzip -k {cat}      (or: pigz -dk {cat})\nand pass the .fits file with --catalogue.")
    if not cat.is_file():
        _fail(f"catalogue not found: {cat}\n"
              f"Download the public KiDS-Legacy shear catalogue (7.2 GB gzip, 16.4 GB once decompressed; see "
              f"https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php):\n"
              f"  curl -L -o {cat}.gz {paths.CATALOGUE_URL}\n"
              f"  gunzip {cat}.gz\n"
              f"or point --catalogue / $KIDSPIPE_DR5_CATALOGUE at an existing copy. Keep it on a local disk, not a Drive mount.")
    try:
        import fitsio
    except ImportError:
        _fail("the python package 'fitsio' is required: pip install fitsio  (or pip install -e '.[measure]')")

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = fitsio.FITS(str(cat))
    nrows = f[1].get_nrows()
    print(f"rows={nrows:,}", flush=True)
    if not args.force and _already_done(out, nrows):
        if udir is not None:
            convert_npz_to_uncompressed(out, udir, bins=bins_sel, force=False)
            return
        print(f"all {NBIN} bin files and extract_info.json already present in {out} with matching row counts; "
              f"nothing to do (use --force to redo)", flush=True)
        return

    # accumulate per bin; 40.9M x ~20 B ~ 0.8 GB total -- fine
    ra = [[] for _ in range(NBIN)]
    dec = [[] for _ in range(NBIN)]
    e1 = [[] for _ in range(NBIN)]
    e2 = [[] for _ in range(NBIN)]
    w = [[] for _ in range(NBIN)]
    north = [[] for _ in range(NBIN)]  # bool: True == North

    tomo_hist = np.zeros(16, dtype=np.int64)
    n_bad_tomo = 0

    for start in range(0, nrows, chunk):
        stop = min(start + chunk, nrows)
        d = f[1].read(rows=np.arange(start, stop), columns=COLS)
        tomo = d["TOMOBIN"].astype(np.int16)
        tomo_hist += np.bincount(tomo[tomo >= 0], minlength=16)[:16]
        sel = (tomo >= 1) & (tomo <= NBIN)
        n_bad_tomo += int((~sel).sum())
        for b in range(1, NBIN + 1):
            m = tomo == b
            if not m.any():
                continue
            ra[b - 1].append(d["RAJ2000"][m])
            dec[b - 1].append(d["DECJ2000"][m])
            e1[b - 1].append(d["e1"][m])
            e2[b - 1].append(d["e2"][m])
            w[b - 1].append(d["weight"][m])
            patch = d["PATCH"][m]
            north[b - 1].append(np.char.startswith(patch.astype("U8"), "N"))
        print(f"  read {stop:,}/{nrows:,}  {time.time()-t0:.0f}s", flush=True)

    info = {
        "catalogue": str(cat),
        "nrows_total": int(nrows),
        "nrows_used": int(tomo_hist[1:NBIN + 1].sum()),
        "nrows_rejected_bad_tomobin": int(n_bad_tomo),
        "tomobin_hist": {int(k): int(v) for k, v in enumerate(tomo_hist) if v},
        "selection": "all rows with 1 <= TOMOBIN <= 6; no further cuts "
                     "(e1,e2 are c-corrected/recalibrated, weight is the final "
                     "shear+gold weight as shipped)",
        "columns": COLS,
        "bins": {},
    }

    for b in range(NBIN):
        r = np.concatenate(ra[b]).astype(np.float64)
        dd = np.concatenate(dec[b]).astype(np.float64)
        # keep the catalogue's float64: float32 ra/dec (0.055" steps) moved fine-bin xi by ~0.1 sigma (RESUME bug 15)
        ee1 = np.concatenate(e1[b]).astype(np.float64)
        ee2 = np.concatenate(e2[b]).astype(np.float64)
        ww = np.concatenate(w[b]).astype(np.float64)
        nn = np.concatenate(north[b])

        good = np.isfinite(r) & np.isfinite(dd) & np.isfinite(ee1) & np.isfinite(ee2) & np.isfinite(ww) & (ww > 0)
        n_drop = int((~good).sum())
        r, dd, ee1, ee2, ww, nn = r[good], dd[good], ee1[good], ee2[good], ww[good], nn[good]

        sw = float(np.sum(ww, dtype=np.float64))
        sw2 = float(np.sum(ww.astype(np.float64) ** 2))
        # ellipticity dispersion: weighted rms of sqrt(e1^2+e2^2)/sqrt(2) per component
        e2sum = ee1.astype(np.float64) ** 2 + ee2.astype(np.float64) ** 2
        sig_e = float(np.sqrt(np.sum(ww.astype(np.float64) * e2sum) / sw / 2.0))

        tmp = out / f".bin{b}.npz.part"
        with open(tmp, "wb") as fh:
            np.savez(fh, ra=r, dec=dd, e1=ee1, e2=ee2, w=ww, north=nn)
        os.replace(tmp, out / f"bin{b}.npz")
        if udir is not None:
            write_uncompressed_bin(
                udir, b, {"ra": r, "dec": dd, "e1": ee1, "e2": ee2, "w": ww, "north": nn},
                force=args.force,
            )
        info["bins"][str(b)] = {
            "n_gal": int(r.size),
            "n_dropped_nonfinite_or_nonpos_weight": n_drop,
            "n_north": int(nn.sum()),
            "n_south": int((~nn).sum()),
            "sum_w": sw,
            "sum_w2": sw2,
            "n_eff": sw ** 2 / sw2,
            "sigma_e": sig_e,
        }
        print(f"bin{b}: n={r.size:,} n_eff={sw**2/sw2:.3e} sigma_e={sig_e:.4f} "
              f"dropped={n_drop}", flush=True)
        del r, dd, ee1, ee2, ww, nn

    _atomic_write_text(out / "extract_info.json", json.dumps(info, indent=2))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
