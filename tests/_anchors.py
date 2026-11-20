"""Spec anchors that depend on the CAMB version.

The spec's anchor values (sigma8 = 0.779557, P_nl = 350.03, ...) were produced with CAMB 1.x. CAMB 2.0 changed
defaults (accuracy target, BBN table, recombination fit, the mnu -> omega_nu h^2 mapping), which moves these values
by up to a few 1e-4 (docs/COMPATIBILITY.md). Under CAMB >= 2 the tests compare against values measured with
CAMB 2.0.5 by the same code paths (data/validation/anchors_camb2.json, benchmarks/measure_camb2_anchors.py),
with the SAME tolerances. Under CAMB 1.x the spec values are used unchanged.
"""
import json
from pathlib import Path

import camb

CAMB_MAJOR = int(camb.__version__.split(".")[0])
_FILE = Path(__file__).resolve().parents[1] / "data" / "validation" / "anchors_camb2.json"


def anchor(key: str, camb1_value):
    """``camb1_value`` (the spec anchor) under CAMB 1.x; the CAMB-2 reference value ``key`` under CAMB >= 2."""
    if CAMB_MAJOR < 2:
        return camb1_value
    return json.loads(_FILE.read_text())["values"][key]
