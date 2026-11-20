# DR5 chunked driver under MPI: equivalence check (2026-11-20)

**What was run.** `scripts/dr5_measure_xi_chunked.py` under MPI with 1 root and 2 worker ranks of 8 threads each, writing to a scratch output root. That root held 130 of the 136 stored exact Bin6_Bin6 chunk files, so the driver recomputed the remaining 6 (c00_00, c00_01, c00_02, c00_07, c05_06, c03_15) and skipped the rest:

```
mpirun -n 3 -x PYTHONPATH -x OMP_NUM_THREADS=8 python3 scripts/dr5_measure_xi_chunked.py --pairs 6-6 --all --mpi \
       --nchunks 16 --threads 8 --bin-slop 0 --angle-slop 0 --outroot <scratch>
```

Wall time 599 s. The diagonal chunk c00_00 alone took 595 s on worker rank 1, while rank 2 did the other five in 170 s. Peak memory 1.2 GB on the root (`mpi_run.log`, `manifest_mpi.json`). No `.part` file was left.

**Comparison** with `../chunks/Bin6_Bin6/` (stored exact run: 20 threads, one process), made with `scripts/dr5_mpi_chunk_compare.py`; output in `compare.json`:

| chunk | result |
|---|---|
| c00_02, c00_07, c03_15 | bit-identical (no pairs in range) |
| c00_01 | npairs exact; max relative difference 1.3e-10 (xip) |
| c05_06 | npairs exact; 5.7e-11 |
| c00_00 | npairs exact; 2.9e-10 (xim); weight 3.3e-13 |

In every file the keys, `r_nom`, `fold`, `bin_slop` and `angle_slop` are identical. The remaining differences come from summation order, because the thread count differs (8 against 20). `xip_im` of the auto pair is zero in exact arithmetic. Both files hold only round-off there (|value| < 1e-17 against xi+ of order 1e-3), so its relative difference is not meaningful and is flagged `roundoff_only` when the reference is below 1e-15.

Full test suite after the MPI work: 400 passed, 0 failed (`pytest_full_20261120_tail.log`).
