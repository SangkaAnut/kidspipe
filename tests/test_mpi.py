"""MPI layer (kidspipe.mpi): the Pool adapter, launch guards, and real `mpirun` runs (skipped without mpi4py / mpirun)."""
import multiprocessing
import os
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from kidspipe import mpi

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
HAVE_MPI = mpi.available() and shutil.which("mpirun") is not None
needs_mpi = pytest.mark.skipif(not HAVE_MPI, reason="mpi4py or mpirun not available")


def _sq(x):
    return x * x


def test_pool_adapter_on_local_executor():
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:   # no fork in a threaded process
        p = mpi.Pool(ex, 2)
        assert p.map(_sq, range(10)) == [x * x for x in range(10)]                 # ordered
        assert sorted(p.imap_unordered(_sq, range(10))) == [x * x for x in range(10)]
        assert p.map(_sq, []) == [] and p.size == p._processes == 2
        p.close()                                                                   # not owned: executor stays usable
        assert ex.submit(_sq, 3).result() == 9


def test_launch_guards(monkeypatch):
    for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS"):
        monkeypatch.delenv(v, raising=False)
    assert mpi.launched_ranks() == 1
    mpi.refuse_unflagged_launch()                                                   # plain run: fine
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert mpi.launched_ranks() == 4
    with pytest.raises(SystemExit, match="without --mpi"):
        mpi.refuse_unflagged_launch()


def _mpirun(n, args, timeout=240, env_extra=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.update(env_extra or {})
    return subprocess.run(["mpirun", "--oversubscribe", "-n", str(n), PY, *args], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=timeout)


TOY = textwrap.dedent("""
    import os, sys
    from kidspipe import mpi
    STATE = {}
    def init(tag):
        STATE["tag"] = tag
        import numba
        STATE["numba_threads"] = numba.get_num_threads()
    def work(x):
        if x == -1:
            raise ValueError("boom")
        return (x * x, STATE["tag"], mpi.rank(), STATE["numba_threads"])
    if __name__ == "__main__":
        nw = mpi.require_workers(int(sys.argv[1]))
        with mpi.executor(init, ("ok",), threads=1) as ex:
            if ex is None:
                sys.exit(0)
            pool = mpi.Pool(ex, nw)
            res = pool.map(work, range(40))
            assert [r[0] for r in res] == [x * x for x in range(40)], "order"
            assert {r[1] for r in res} == {"ok"}, "initializer"
            assert 0 not in {r[2] for r in res}, "root computed a task"
            assert {r[3] for r in res} == {1}, "worker numba threads"
            assert sorted(r[0] for r in pool.imap_unordered(work, range(40))) == [x * x for x in range(40)]
            try:
                pool.map(work, [1, -1, 2])
                raise SystemExit("exception not propagated")
            except ValueError:
                pass
            print("TOY_OK workers", sorted({r[2] for r in res}))
""")


@needs_mpi
def test_executor_under_mpirun(tmp_path):
    f = tmp_path / "toy.py"
    f.write_text(TOY)
    r = _mpirun(4, [str(f), "2"], env_extra={"NUMBA_NUM_THREADS": "4"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TOY_OK workers [1, 2, 3]" in r.stdout


@needs_mpi
def test_too_few_ranks_is_refused(tmp_path):
    f = tmp_path / "toy.py"
    f.write_text(TOY)
    r = _mpirun(2, [str(f), "2"])
    assert r.returncode != 0 and "needs at least 3 ranks" in (r.stdout + r.stderr)


FAILFAST = textwrap.dedent("""
    import sys, time
    from kidspipe import mpi
    def work(x):
        if x == 0:
            raise ValueError("boom")
        time.sleep(1.0)
        return x
    if __name__ == "__main__":
        nw = mpi.require_workers(2)
        t0 = time.time()
        try:
            with mpi.executor() as ex:
                if ex is None:
                    sys.exit(0)
                for r in mpi.Pool(ex, nw).imap_unordered(work, range(60)):   # 60 x 1 s on 3 workers if not cancelled
                    pass
        except ValueError:
            print(f"FAILFAST_OK {time.time() - t0:.1f}")
""")


@needs_mpi
def test_root_error_cancels_queue(tmp_path):
    f = tmp_path / "failfast.py"
    f.write_text(FAILFAST)
    r = _mpirun(4, [str(f)])
    assert r.returncode == 0, r.stdout + r.stderr
    line = [s for s in r.stdout.splitlines() if s.startswith("FAILFAST_OK")]
    assert line and float(line[0].split()[1]) < 10.0, r.stdout       # ~20 s if the whole queue still ran


@needs_mpi
def test_initializer_failure_is_reported(tmp_path):
    f = tmp_path / "badinit.py"
    f.write_text(textwrap.dedent("""
        import sys
        from kidspipe import mpi
        def init():
            raise RuntimeError("no model here")
        def work(x):
            return x
        if __name__ == "__main__":
            nw = mpi.require_workers(2)
            with mpi.executor(init) as ex:
                if ex is None:
                    sys.exit(0)
                mpi.Pool(ex, nw).map(work, range(4))
    """))
    r = _mpirun(3, [str(f)])
    assert r.returncode != 0 and "worker initializer failed" in r.stderr and "no model here" in r.stderr


@needs_mpi
def test_cli_guards():
    r = _mpirun(2, ["-m", "kidspipe", "loglike", "--quiet"])                         # mpirun without --mpi
    assert r.returncode != 0 and "without --mpi" in (r.stdout + r.stderr)
    r = _mpirun(2, ["-m", "kidspipe", "map", "--mpi", "--starts", "1", "--quiet"])   # --mpi with only 1 worker
    assert r.returncode != 0 and "needs at least 3 ranks" in (r.stdout + r.stderr)


@needs_mpi
def test_dr5_driver_guards():
    r = _mpirun(2, ["scripts/dr5_measure_xi_chunked.py", "--status"])               # every rank would act
    assert r.returncode != 0 and "without --mpi" in (r.stdout + r.stderr)
    r = subprocess.run([PY, "scripts/dr5_measure_xi_chunked.py", "--status", "--mpi"], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode != 0 and "run --status / --assemble without mpirun" in (r.stdout + r.stderr)


def test_atomic_chunk_write(tmp_path):
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("chk", ROOT / "scripts/dr5_measure_xi_chunked.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    f = tmp_path / "c00_01.npz"
    m.write_chunk_atomic(f, {"xip": np.arange(3.0)})
    assert np.array_equal(np.load(f)["xip"], np.arange(3.0))
    assert [p.name for p in tmp_path.iterdir()] == ["c00_01.npz"]                    # no leftover temporary file
    assert not list(tmp_path.glob("c*.part")) and ".c00_01.npz.part" not in [p.name for p in tmp_path.glob("c*.npz")]
