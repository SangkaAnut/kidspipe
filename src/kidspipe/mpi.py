"""MPI support (mpi4py.futures), shared by the CLI samplers, the DR5 chunked measurement driver and the batch scripts.

Model: launch the SAME command on every rank with ``mpirun -n N``.  Rank 0 (the root) runs the program; ranks 1..N-1
are workers that only execute tasks sent by the root (dynamic scheduling: a free worker takes the next task).  Each
worker runs ``initializer(*initargs)`` once before its first task, e.g. to build its own model / likelihood.

    with mpi.executor(initializer, initargs) as ex:
        if ex is None:          # a worker rank: the root has finished; nothing else to do
            return
        pool = mpi.Pool(ex, mpi.world_size() - 1)
        results = pool.map(func, items)            # ordered
        for r in pool.imap_unordered(func, items): # as completed
            ...

Thread discipline: every rank is a separate process, so N ranks x T threads must not exceed the cores.  Workers call
:func:`limit_threads` (numba + BLAS/OpenMP via threadpoolctl) in their initializer; launch with
``OMP_NUM_THREADS=<T>`` as well so that libraries that read the variable at import get it.

mpi4py is optional: :func:`available` is False without it and nothing here is imported by the serial code paths.
"""
from __future__ import annotations

import contextlib
import os
from typing import Callable, Iterable, Iterator, Optional

__all__ = ["available", "world_size", "rank", "executor", "Pool", "limit_threads", "require_workers"]


def available() -> bool:
    try:
        import mpi4py  # noqa: F401
        return True
    except ImportError:
        return False


def _comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def world_size() -> int:
    """Number of MPI ranks (1 when mpi4py is missing or the program was not launched with mpirun)."""
    if not available():
        return 1
    return _comm().Get_size()


def rank() -> int:
    if not available():
        return 0
    return _comm().Get_rank()


def launched_ranks() -> int:
    """Rank count announced by the launcher (Open MPI / MPICH / Slurm env), without importing mpi4py; 1 if not launched."""
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return 1


def refuse_unflagged_launch(flag: str = "--mpi") -> None:
    """Stop a program started with `mpirun -n >1` WITHOUT its MPI flag: every rank would run the whole job and write the
    same outputs."""
    n = launched_ranks()
    if n > 1:
        raise SystemExit(f"started under mpirun with {n} ranks but without {flag}: every rank would run the whole job; "
                         f"add {flag} or run without mpirun")


def require_workers(minimum: int = 2) -> int:
    """Number of worker ranks (world size - 1); raises SystemExit with a clear message if fewer than ``minimum``."""
    if not available():
        raise SystemExit("--mpi needs mpi4py (not installed)")
    n = world_size() - 1
    if n < minimum:
        raise SystemExit(f"--mpi needs at least {minimum + 1} ranks (1 root + {minimum} workers); launch with "
                         f"`mpirun -n {minimum + 1} ...` or more (this run has {n + 1})")
    return n


def limit_threads(n: int) -> None:
    """Bound the threads of this process: numba, and BLAS/OpenMP pools via threadpoolctl (both optional)."""
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)              # for libraries loaded after this point
    try:
        import numba
        numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    except Exception:                                   # numba missing, or NUMBA_NUM_THREADS < n: keep its own limit
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass


def _init_with_threads(nthreads, initializer, initargs):
    try:
        limit_threads(nthreads)
        if initializer is not None:
            initializer(*initargs)
    except BaseException:                               # mpi4py reports only "initializer failed" on the root
        import sys
        import traceback
        print(f"[rank {rank()}] worker initializer failed:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise


@contextlib.contextmanager
def executor(initializer: Optional[Callable] = None, initargs: tuple = (), threads: int = 1):
    """Executor over the worker ranks.  Yields an ``MPIPoolExecutor`` on the root and ``None`` on the workers.

    On a worker rank the body is entered only after the root has shut the executor down, so the caller must return
    immediately when it receives ``None``.  ``threads`` bounds each worker's numba/BLAS threads before ``initializer``.

    If the body raises on the root, queued tasks are cancelled first (MPICommExecutor's own exit waits for the whole
    queue), so the error surfaces after the tasks already running finish, not after every queued one.
    """
    from mpi4py.futures import MPICommExecutor
    with MPICommExecutor(_comm(), root=0, initializer=_init_with_threads,
                         initargs=(threads, initializer, tuple(initargs))) as ex:
        try:
            yield ex
        except BaseException:
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=True)
            raise


class Pool:
    """``map`` / ``imap_unordered`` adapter over an executor (MPI or a local ``concurrent.futures`` one).

    ``size`` and ``_processes`` give the number of workers (dynesty and nautilus read them)."""

    def __init__(self, ex, nworkers: int, owned: bool = False):
        self._ex = ex
        self.size = self._processes = int(nworkers)
        self._owned = owned

    def map(self, func: Callable, iterable: Iterable, chunksize: Optional[int] = None) -> list:
        items = list(iterable)
        if not items:
            return []
        cs = chunksize or max(1, len(items) // (4 * self.size))
        return list(self._ex.map(func, items, chunksize=cs))

    def imap_unordered(self, func: Callable, iterable: Iterable) -> Iterator:
        from concurrent.futures import as_completed
        futs = [self._ex.submit(func, x) for x in iterable]
        try:
            for f in as_completed(futs):
                yield f.result()
        finally:                                        # error or abandoned iterator: drop what has not started
            for f in futs:
                f.cancel()

    def submit(self, *a, **k):
        return self._ex.submit(*a, **k)

    def close(self) -> None:
        if self._owned:
            self._ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
