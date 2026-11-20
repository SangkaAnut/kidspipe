"""KV450 redshift distributions and photo-z shifts (spec s3).

Conventions
-----------
* ``z`` ascending, dimensionless.  ``n_i(z)`` has unit ``1`` and is normalised so that
  ``trapz(n_i, z) = 1`` on its own node set (spec s3.1).
* The shipped ``Nz_DIR_z*.asc`` files tabulate the histogram height against the *lower* bin edge
  (0.00, 0.05, ... 5.95).  Loading adds half a bin width (``+0.025``) to obtain bin centres,
  prepends the node ``(z=0, n=0)`` and renormalises with the trapezoid rule, giving 121 nodes.
* Photo-z shift (spec s3.2)::

      n'_i(z) = n_i(z - dz_i)   (linear interpolation, 0 outside the table)
      n'_i   <- n'_i / trapz(n'_i, z)

  Positive ``dz`` moves the distribution to higher redshift; mass pushed below ``z = 0`` or
  beyond the last node is truncated and then redistributed by the renormalisation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "NZ_FILES",
    "default_nz_dir",
    "load_nz",
    "normalise_nz",
    "shift_nz",
    "shift_nz_batch",
    "NZ",
]

#: Order of the five tomographic bins (low to high redshift).
NZ_FILES = (
    "Nz_DIR_z0.1t0.3.asc",
    "Nz_DIR_z0.3t0.5.asc",
    "Nz_DIR_z0.5t0.7.asc",
    "Nz_DIR_z0.7t0.9.asc",
    "Nz_DIR_z0.9t1.2.asc",
)


def default_nz_dir() -> Path:
    """Locate ``data/kv450/nofz`` (env ``KIDSPIPE_DATA`` overrides the data root)."""
    env = os.environ.get("KIDSPIPE_DATA")
    if env:
        p = Path(env) / "kv450" / "nofz"
        if not p.is_dir():
            raise FileNotFoundError(f"KIDSPIPE_DATA={env!r} is set but {p} does not exist")
        return p
    for parent in Path(__file__).resolve().parents:
        p = parent / "data" / "kv450" / "nofz"
        if p.is_dir():
            return p
    raise FileNotFoundError("cannot find data/kv450/nofz; set KIDSPIPE_DATA")


def normalise_nz(z: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Normalise ``n`` (shape ``(..., nz)``) to unit trapezoid integral over ``z``.

    ``n <- n / trapz(n, z)``, unit of ``n``: 1.
    """
    n = np.asarray(n, dtype=np.float64)
    norm = np.asarray(np.trapz(n, z, axis=-1))
    if not (np.all(np.isfinite(norm)) and np.all(norm > 0.0)):
        raise ValueError("n(z) has a non-finite or non-positive trapezoid integral and cannot be normalised "
                         "(photo-z shift pushed the whole distribution out of the table, or NaN input)")
    return n / norm[..., None]


def shift_nz(z: np.ndarray, n: np.ndarray, dz: float, renormalise: bool = True) -> np.ndarray:
    """Photo-z shift of one distribution: ``n'(z) = n(z - dz)`` (spec s3.2).

    Linear interpolation, value 0 outside ``[z[0], z[-1]]``, then (optionally) trapezoid
    renormalisation ``n' <- n'/trapz(n', z)``.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.interp(z - dz, z, n, left=0.0, right=0.0)
    return normalise_nz(z, out) if renormalise else out


def shift_nz_batch(z: np.ndarray, n: np.ndarray, dz: np.ndarray, renormalise: bool = True) -> np.ndarray:
    """Shift all tomographic bins at once.

    Parameters
    ----------
    z : (nz,) ascending redshift nodes.
    n : (nbin, nz) normalised distributions.
    dz : (nbin,) additive shifts (bin ``i`` is shifted by ``dz[i]``).

    Returns
    -------
    (nbin, nz) array ``n_i(z - dz_i)`` renormalised to unit trapezoid integral.
    """
    z = np.asarray(z, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    dz = np.broadcast_to(np.asarray(dz, dtype=np.float64), (n.shape[0],))
    # vectorised linear interpolation (all bins share the node set)
    x = z[None, :] - dz[:, None]
    idx = np.searchsorted(z, x, side="right") - 1
    inside = (x >= z[0]) & (x <= z[-1])
    idx = np.clip(idx, 0, z.size - 2)
    z0 = z[idx]
    t = (x - z0) / (z[idx + 1] - z0)
    rows = np.arange(n.shape[0])[:, None]
    out = n[rows, idx] * (1.0 - t) + n[rows, idx + 1] * t
    out = np.where(inside, out, 0.0)
    return normalise_nz(z, out) if renormalise else out


def load_nz(data_dir: str | os.PathLike | None = None) -> "NZ":
    """Load the five KV450 DIR n(z) files (spec s3.1) into an :class:`NZ`.

    Steps: ``z <- z + (z[1]-z[0])/2`` (lower edge -> bin centre), prepend ``(0, 0)``,
    ``n <- n / trapz(n, z)``.  Result: ``z`` with 121 nodes, ``n`` of shape ``(5, 121)``.
    """
    d = Path(data_dir) if data_dir is not None else default_nz_dir()
    zs, ns = [], []
    for name in NZ_FILES:
        a = np.loadtxt(d / name)
        zs.append(a[:, 0])
        ns.append(a[:, 1])
    z_edge = zs[0]
    for zz in zs[1:]:
        if not np.allclose(zz, z_edge, atol=1e-12):
            raise ValueError("n(z) files do not share a z grid")
    z = z_edge + 0.5 * (z_edge[1] - z_edge[0])
    if z[0] > 1e-8:
        z = np.concatenate([[0.0], z])
        n = np.concatenate([np.zeros((len(ns), 1)), np.array(ns)], axis=1)
    else:
        n = np.array(ns)
    return NZ(z=z, n=normalise_nz(z, n))


@dataclass
class NZ:
    """Tomographic source distributions on a common node set.

    Attributes
    ----------
    z : (nz,) redshift nodes (ascending, first node 0 for KV450).
    n : (nbin, nz) normalised distributions, ``trapz(n[i], z) = 1``.
    """

    z: np.ndarray
    n: np.ndarray

    @property
    def nbin(self) -> int:
        return self.n.shape[0]

    def shifted(self, dz) -> np.ndarray:
        """Return ``(nbin, nz)`` shifted+renormalised distributions for ``dz`` of shape ``(nbin,)``."""
        return shift_nz_batch(self.z, self.n, dz)

    def mean_z(self, n: np.ndarray | None = None) -> np.ndarray:
        """Mean redshift ``int z n dz`` of each bin (of ``n`` if given)."""
        n = self.n if n is None else n
        return np.trapz(n * self.z, self.z, axis=-1)
