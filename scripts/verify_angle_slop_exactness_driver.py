# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Run the production driver main() with OUTROOT redirected to a scratch dir (argv: outroot, then driver args)."""
import sys, os, importlib.util
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","5")
out=Path(sys.argv[1]); sys.argv=["drv"]+sys.argv[2:]
spec=importlib.util.spec_from_file_location("drv",str(Path(__file__).resolve().parent/"dr5_measure_xi_chunked.py")); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.OUTROOT=out
m.main()
