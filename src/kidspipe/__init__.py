"""kidspipe: clean-room fast KiDS/KV450 cosmic-shear (xi+/-) inference pipeline.

Mathematics: ``docs/MASTER_MATH_SPEC.md`` (cited as "spec sN"); layout: ``docs/DESIGN.md``.
Sub-packages are imported lazily by the user (``kidspipe.theory``, ``kidspipe.inference``, ...),
so importing the top-level package is cheap and does not pull in CAMB or numba.
"""

__version__ = "0.1.0"

from . import params  # noqa: F401  (light-weight: numpy/scipy only)

__all__ = ["params", "__version__"]
