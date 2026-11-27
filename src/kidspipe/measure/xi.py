"""Shear two-point measurement from the compact per-bin catalogues.

Primary estimator: TreeCorr (the code the official release used).
Independent cross-check: a KD-tree pair counter in :mod:`kidspipe.measure.paircount`.

Conventions
-----------
The catalogue's e1/e2 are Lensfit ellipticities whose components are defined with respect
to the local (RA, Dec) axes ("sky" convention, e1 along +RA).  TreeCorr by default expects
shears in the (North, East)-ish convention, so a sign flip on one component may be needed.
xi_+ is invariant under a global flip of both components, xi_- is not invariant under a
flip of a single component; the convention is therefore fixed by comparing xi_- with the
official ascii files (see scripts/dr5_convention.py) and never assumed here.

GOTCHA: with spherical coordinates TreeCorr's ``min_sep``/``max_sep`` are interpreted in the
catalogue's native units unless ``sep_units`` is given.  Omitting ``sep_units`` here makes
every bin empty (npairs == 0) with no warning at all, so it is always passed explicitly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TMIN, TMAX, NBINS_LOG = 2.0, 300.0, 1000
from .. import paths

#: per-bin catalogue extract written by ``scripts/dr5_extract.py`` (``$KIDSPIPE_DR5_WORK``, default ``~/kids_dr5_work``)
WORK = paths.dr5_work_dir()
BIN_REQUIRED_COLS = ("ra", "dec", "e1", "e2", "w")
BIN_OPTIONAL_COLS = ("north",)


def load_bin(
    b: int,
    work: Path = WORK,
    mmap: bool = False,
    columns: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Load the compact catalogue for 1-based tomographic bin ``b``.

    ``dr5_extract.py`` writes ``bin0.npz`` for tomographic bin 1 (zero-based file names), so
    the index is shifted here; a 1-based->1-based mapping silently measured the wrong bin.
    When ``mmap=True``, reads uncompressed ``bin{b-1}/<col>.npy`` via ``np.load(..., mmap_mode='r')``.
    """
    idx = int(b) - 1
    if mmap:
        d = Path(work) / f"bin{idx}"
        if not d.is_dir():
            raise FileNotFoundError(
                f"{d} (uncompressed tomographic bin {b}) not found; "
                f"run scripts/dr5_extract.py --uncompressed-dir {Path(work)}"
            )
        if columns is None:
            cols = list(BIN_REQUIRED_COLS) + [
                col for col in BIN_OPTIONAL_COLS if (d / f"{col}.npy").is_file()
            ]
        else:
            cols = list(columns)
        out = {}
        for col in cols:
            p = d / f"{col}.npy"
            if not p.is_file():
                raise FileNotFoundError(
                    f"{p} (column {col!r} of tomographic bin {b}) not found; "
                    f"run scripts/dr5_extract.py --uncompressed-dir {Path(work)}"
                )
            out[col] = np.load(p, mmap_mode="r")
        return out

    f = Path(work) / f"bin{idx}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} (tomographic bin {b}) not found; run scripts/dr5_extract.py")
    with np.load(f) as d:
        cols = list(d.files) if columns is None else list(columns)
        return {k: d[k] for k in cols}


def make_treecorr_cat(rec: dict, flip_g1: bool = False, flip_g2: bool = False,
                      g1: np.ndarray | None = None, g2: np.ndarray | None = None,
                      ra_units: str = "deg"):
    import treecorr
    g1 = rec["e1"] if g1 is None else g1
    g2 = rec["e2"] if g2 is None else g2
    return treecorr.Catalog(
        ra=np.asarray(rec["ra"], np.float64), dec=np.asarray(rec["dec"], np.float64),
        ra_units=ra_units, dec_units=ra_units,
        g1=np.asarray(g1, np.float64), g2=np.asarray(g2, np.float64),
        w=np.asarray(rec["w"], np.float64),
        flip_g1=flip_g1, flip_g2=flip_g2,
    )


def treecorr_kwargs(nbins: int = NBINS_LOG, bin_slop: float = 0.05,
                    num_threads: int = 16, angle_slop: float | None = None) -> dict:
    """TreeCorr configuration.  ``angle_slop=None`` (default) leaves TreeCorr's own default
    (0.1 rad in TreeCorr 5, which approximates the pair projection angle and so perturbs xi_-);
    a float is passed through as ``angle_slop`` (0.0 = exact projection angles)."""
    kw = dict(nbins=int(nbins), bin_type="Log", min_sep=TMIN, max_sep=TMAX,
              sep_units="arcmin", bin_slop=float(bin_slop),
              num_threads=int(num_threads), verbose=0)
    if angle_slop is not None:
        kw["angle_slop"] = float(angle_slop)
    return kw


def run_pair(cat1, cat2, nbins: int = NBINS_LOG, bin_slop: float = 0.05,
             num_threads: int = 16, angle_slop: float | None = None):
    """GG correlation for one tomographic pair; returns the finalized corr object."""
    import treecorr
    gg = treecorr.GGCorrelation(treecorr_kwargs(nbins, bin_slop, num_threads, angle_slop))
    gg.process(cat1, cat2)
    return gg


def corr_to_arrays(gg) -> dict:
    return {
        "r_nom": np.asarray(gg.rnom, float),
        "meanr": np.asarray(gg.meanr, float),
        "xip": np.asarray(gg.xip, float),
        "xim": np.asarray(gg.xim, float),
        "xip_im": np.asarray(gg.xip_im, float),
        "xim_im": np.asarray(gg.xim_im, float),
        "varxi": np.asarray(gg.varxip, float),
        "npairs": np.asarray(gg.npairs, float),
        "weight": np.asarray(gg.weight, float),
    }
