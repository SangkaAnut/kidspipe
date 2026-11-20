"""pytest configuration shared by all tests.

* Wall-clock assertions are marked ``@pytest.mark.timing`` and skipped unless ``KIDSPIPE_TIMING_TESTS=1`` or the machine has at
  least 8 cores (a 2-vCPU Colab runtime is too slow and too noisy for them).  No tolerance is relaxed: the tests either run
  with their original limits or are skipped.
* ``src`` is put on ``sys.path`` (pyproject also sets ``pythonpath``), so the tests run from any working directory.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _timing_enabled() -> bool:
    return os.environ.get("KIDSPIPE_TIMING_TESTS") == "1" or (os.cpu_count() or 1) >= 8


def pytest_collection_modifyitems(config, items):
    if _timing_enabled():
        return
    skip = pytest.mark.skip(reason="wall-clock test: set KIDSPIPE_TIMING_TESTS=1 or run on a machine with >= 8 cores")
    for item in items:
        if "timing" in item.keywords:
            item.add_marker(skip)
