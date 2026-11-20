# Exact run, session 3: timings and integrity (2026-11-11); whole-run summary

Session 3: pairs 1-4 1-6 2-2 2-3 3-4 3-6 6-6, same settings as sessions 1-2 (bin_slop 0, angle_slop 0, 16 chunks, 20 threads,
float64, one process per pair; `session3.sh`, `session3.log`). Started 2026-11-10 19:55:18, finished 2026-11-11 08:48:42
(12.89 h). All PAIR_DONE exit 0; no warning/error/traceback in the log; no interruption.

| pair | s | h | peak RSS GB |
|---|---|---|---|
| 1-4 | 8309 | 2.31 | 4.89 |
| 1-6 | 6584 | 1.83 | 4.58 |
| 2-2 | 6785 | 1.88 | 2.55 |
| 2-3 | 7080 | 1.97 | 4.85 |
| 3-4 | 6689 | 1.86 | 4.69 |
| 3-6 | 6278 | 1.74 | 4.38 |
| 6-6 | 4678 | 1.30 | 2.09 |

Predicted 13.0 h (refit on 14 pairs); measured 12.89 h.

**Whole run:** compute S1 13.65 h + S2 12.78 h + S3 12.89 h = 39.3 h for 4656 chunk-pairs (estimate before the run: 40-55 h).
Peak RSS 5.05 GB (1-2). One interruption (WSL restart during S2, pair 4-4), resumed without loss beyond one chunk-pair.

## Integrity (independent audit agent, all keys force-read)

Session 3: 1552 files (2 autos x 136 + 5 crosses x 256; corrected from a miscount of 1416, the per-pair audit covered every file), names equal to `chunk_pairs(16, same)`, settings exact, fold correct, r_nom equal to the product,
finite where weight > 0, meanr inside its bin, mtimes inside each pair's log window. Whole run: `chunks/` holds exactly the 21
pair directories and 4656 files (all 74964 bytes, same 12 keys, bin_slop 0 and angle_slop 0 in every file); no session-1 or
session-2 file was modified after its session ended.

## Assembly and product

`scripts/dr5_measure_xi_chunked.py --assemble --nchunks 16` wrote `assembled/ns/` (21 pairs, 2026-11-11 08:51). The exact
product (`runs/dr5_measure/product_exact/`, built by `scripts/dr5_make_exact_product.py`) converts auto pairs to the unordered
convention and carries the 21-pair comparison with the bin_slop 1.5 product (`EXACT_VS_B1P5.md`). Every number in it was
recomputed by an independent agent directly from the chunk files with its own code: per-pair metrics agree to <= 2e-14
relative, every full-vector chi2 and PTE to all printed digits.
