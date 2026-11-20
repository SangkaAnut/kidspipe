"""Where kidspipe finds its inputs and writes its outputs.

Every location has an environment-variable override, so the same checkout runs on a workstation, in
Google Colab (outputs on Google Drive) or on a cluster:

==========================  ====================================================  =========================================
variable                    meaning                                               default
==========================  ====================================================  =========================================
``KIDSPIPE_ROOT``           repository root                                       the checkout this package lives in
``KIDSPIPE_DATA``           shipped input data (KV450, KiDS-Legacy, COSEBIs, ...)  ``<root>/data``
``KIDSPIPE_RUNS``           run outputs                                           ``<root>/runs``
``KIDSPIPE_OFFICIAL``       official KiDS-Legacy release files used by the DR5     ``<data>/kids_legacy``
                            checks (``KiDS_Legacy_xipm.fits``,
                            ``KiDS_Legacy_cosebis.fits``, ``cosmosis_xipm/``)
``KIDSPIPE_DR5_WORK``       per-bin extract of the DR5 shear catalogue             ``~/kids_dr5_work``
                            (``bin0.npz`` .. ``bin5.npz``, about 1.6 GB)
``KIDSPIPE_DR5_CATALOGUE``  the DR5 shear catalogue                                ``<dr5_work>/KiDS_Legacy_NS_unblind_final.fits``
                            ``KiDS_Legacy_NS_unblind_final.fits`` (16.4 GB)
``KIDSPIPE_EMULATOR``       linear-transfer emulator artefact                      ``<data>/emulator/transfer_emulator_v1.npz``
                                                                                  (CAMB 1.x) or ``..._v1_camb2.npz`` (CAMB 2.x)
``KIDSPIPE_PLOTS``          figures written by the scripts                         ``<root>/plots``
==========================  ====================================================  =========================================

Functions are evaluated at call time, so a variable set after import still takes effect for later calls (module-level
constants such as ``kidspipe.measure.official.OFFICIAL`` are fixed at import).
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["root", "data_dir", "runs_dir", "shipped_runs_dir", "find_result", "official_dir", "dr5_work_dir", "dr5_catalogue", "emulator_artefact", "emulator_artefact_target", "plots_dir",
           "CATALOGUE_NAME", "CATALOGUE_URL", "RELEASE_URL"]

#: file name of the KiDS-Legacy (DR5) shear catalogue
CATALOGUE_NAME = "KiDS_Legacy_NS_unblind_final.fits"
#: public download (gzip, about 7.2 GB) from the KiDS DR5 weak-lensing page https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php
CATALOGUE_URL = "https://kids.strw.leidenuniv.nl/DR5/data_files/KiDS_Legacy_NS_unblind_final.fits.gz"
#: KiDS-Legacy cosmic-shear data release (data vectors, n(z), covariances, chains; about 131 MB)
RELEASE_URL = "https://kids.strw.leidenuniv.nl/sci_data/KiDS_Legacy_cosmic_shear_data_release.tar.gz"


def _env(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def root() -> Path:
    """Repository root: ``$KIDSPIPE_ROOT``, else the checkout that contains this package (src layout), else the
    current directory if it looks like a checkout."""
    env = _env("KIDSPIPE_ROOT")
    if env:
        return env
    here = Path(__file__).resolve().parents[2]                  # <root>/src/kidspipe/paths.py
    if (here / "pyproject.toml").is_file() or (here / "data").is_dir():
        return here
    cwd = Path.cwd()
    if (cwd / "data").is_dir() and (cwd / "src" / "kidspipe").is_dir():
        return cwd
    return here


def data_dir() -> Path:
    return _env("KIDSPIPE_DATA") or root() / "data"


def runs_dir() -> Path:
    return _env("KIDSPIPE_RUNS") or root() / "runs"


def shipped_runs_dir() -> Path:
    """``<root>/runs``: the run results that ship with the repository (reference values), independent of
    ``$KIDSPIPE_RUNS``."""
    return root() / "runs"


def find_result(relpath: str | Path) -> Path:
    """An existing result used as an INPUT: ``<runs>/<relpath>`` if present (a result you produced), else the copy
    shipped in ``<root>/runs/<relpath>``. Returns the ``<runs>`` path when neither exists, so error messages name the
    location the result would be written to."""
    mine = runs_dir() / relpath
    if mine.exists():
        return mine
    shipped = shipped_runs_dir() / relpath
    return shipped if shipped.exists() else mine


def official_dir() -> Path:
    return _env("KIDSPIPE_OFFICIAL") or data_dir() / "kids_legacy"


def dr5_work_dir() -> Path:
    return _env("KIDSPIPE_DR5_WORK") or Path.home() / "kids_dr5_work"


def dr5_catalogue() -> Path:
    return _env("KIDSPIPE_DR5_CATALOGUE") or dr5_work_dir() / CATALOGUE_NAME


def plots_dir() -> Path:
    return _env("KIDSPIPE_PLOTS") or root() / "plots"


def _camb_major() -> int | None:
    """Major version of the installed CAMB, read from the package metadata (does not import camb)."""
    try:
        from importlib.metadata import version
        return int(version("camb").split(".")[0])
    except Exception:
        return None


def emulator_artefact_target() -> Path:
    """The artefact matching the installed CAMB major version (where ``scripts/train_emulator.py fit`` writes):
    ``transfer_emulator_v1.npz`` (trained with CAMB 1.5.8) or ``transfer_emulator_v1_camb2.npz`` (CAMB 2.0.5);
    ``$KIDSPIPE_EMULATOR`` overrides."""
    name = "transfer_emulator_v1_camb2.npz" if (_camb_major() or 1) >= 2 else "transfer_emulator_v1.npz"
    return _env("KIDSPIPE_EMULATOR") or data_dir() / "emulator" / name


def emulator_artefact() -> Path:
    """The artefact the pipeline loads: ``$KIDSPIPE_EMULATOR``, else the one trained with the installed CAMB major
    version (:func:`emulator_artefact_target`), else ``transfer_emulator_v1.npz``."""
    t = emulator_artefact_target()
    if t.is_file() or _env("KIDSPIPE_EMULATOR"):
        return t
    return data_dir() / "emulator" / "transfer_emulator_v1.npz"
