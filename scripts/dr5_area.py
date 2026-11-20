# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Converged sky-area measurement for the extracted catalogue, and the area-vs-density test.

The occupancy area must be quoted at a resolution where pixels are still well populated
(occupancy >> 1 galaxy/pixel), otherwise Poisson noise inflates the count of occupied pixels.
Here the scan is extended and the converged value adopted, with the implied area from the
pair-count ratio for comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from kidspipe import paths  # noqa: E402

import numpy as np

MAIN = paths.dr5_work_dir()
RATIO = 1.41698


def area_deg2(ra_deg, dec_deg, nside_per_deg):
    ra = np.asarray(ra_deg, float) % 360.0
    s = np.sin(np.deg2rad(np.asarray(dec_deg, float)))
    n_ra = int(360.0 * nside_per_deg)
    n_s = int(2.0 * nside_per_deg)
    i_ra = np.minimum((ra / 360.0 * n_ra).astype(np.int64), n_ra - 1)
    i_s = np.minimum(((s + 1.0) / 2.0 * n_s).astype(np.int64), n_s - 1)
    n_occ = np.unique(i_ra.astype(np.int64) * n_s + i_s).size
    pix_area = (360.0 / n_ra) * (2.0 / n_s) * (180.0 / np.pi)
    return n_occ * pix_area, n_occ, pix_area


def main() -> None:
    m = dict(np.load(MAIN / "bin0.npz"))
    ra, dec = m["ra"], m["dec"]
    n = ra.size
    print(f"{n:,} galaxies, RA {ra.min():.2f}..{ra.max():.2f}, Dec {dec.min():.2f}..{dec.max():.2f}")
    print("\n  nside/deg   pixel_deg2   occ_pix    area_deg2   gal/pix    n_gal/area")
    best = None
    for nside in [20, 40, 60, 80, 120, 160, 200]:
        a, nocc, pa = area_deg2(ra, dec, nside)
        occ = n / nocc
        print(f"  {nside:9d}  {pa:11.5f}  {nocc:8d}  {a:11.1f}  {occ:9.1f}  {n/a:10.3f}")
        # adopt the finest scan with mean occupancy still >= 10 (Poisson-safe)
        if occ >= 10:
            best = (nside, a, occ)
    nside, a, occ = best
    dens = n / a
    print(f"\nadopted: nside/deg={nside}, area={a:.1f} deg2, occupancy={occ:.1f} gal/pix")
    print(f"number density = {dens:.3f} gal/arcmin^2  (official NGAL_1 = 1.6884)")
    print()
    print("Test A -- pure AREA difference (same density as official, larger footprint):")
    print(f"  predict npairs ratio = A_ours/A_off, with A_off = n/NGAL_1 = {n/1.688424896549292/3600:.1f} deg2")
    print(f"  -> predicted ratio = {a/(n/1.688424896549292/3600):.4f}   (measured {RATIO:.4f})")
    print()
    print("Test B -- pure DENSITY difference (same area as official):")
    print(f"  predict npairs ratio = (dens/NGAL_1)^2 = {(dens/1.688424896549292)**2:.4f}   (measured {RATIO:.4f})")


if __name__ == "__main__":
    main()
