"""Measurement layer: catalogue handling, shear two-point estimators, xi->COSEBIs, comparisons.

Nothing here touches the validated theory/likelihood code paths; this package only adds
measurement-side code (see docs/DR5_MEASUREMENT_REPORT.md).
"""
from __future__ import annotations

__all__ = ["official", "xi", "paircount", "xi2cosebis"]
