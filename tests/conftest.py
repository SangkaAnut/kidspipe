"""pytest configuration shared by all tests.

* Wall-clock assertions are marked ``@pytest.mark.timing``. Their limits were set on a 22-core workstation; they run by
  default only on machines with at least 16 cores (Colab runtimes have 2-8 slower vCPUs and are skipped).
  ``KIDSPIPE_TIMING_TESTS=1`` forces them on, ``KIDSPIPE_TIMING_TESTS=0`` off.  No tolerance is relaxed: the tests either run
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
    env = os.environ.get("KIDSPIPE_TIMING_TESTS")
    if env in ("0", "1"):
        return env == "1"
    return (os.cpu_count() or 1) >= 16


def pytest_collection_modifyitems(config, items):
    if _timing_enabled():
        return
    skip = pytest.mark.skip(reason="wall-clock test: set KIDSPIPE_TIMING_TESTS=1 or run on a machine with >= 16 cores")
    for item in items:
        if "timing" in item.keywords:
            item.add_marker(skip)
