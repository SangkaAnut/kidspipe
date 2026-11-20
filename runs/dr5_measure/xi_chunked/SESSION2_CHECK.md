# Exact run, session 2: timings and interim check (2026-11-10)

Session 2: pairs 1-2 2-5 3-3 3-5 4-4 4-5 4-6, same settings as session 1 (bin_slop 0, angle_slop 0, 16 chunks, 20 threads,
float64, one process per pair). Started 2026-11-10 00:44:30 (`session2.sh`, `session2.log`). A WSL restart at ~08:50 killed
pair 4-4 after 58/136 chunk-pairs; resumed 08:56:23 with `session2_resume.sh` (4-4 4-5 4-6, `session2_resume.log`), which
skipped the existing files. Finished 13:37:56. All PAIR_DONE exit 0; no warning/error/traceback in either log.

## Timings

| pair | s | h | peak RSS GB |
|---|---|---|---|
| 1-2 | 7950 | 2.21 | 5.05 |
| 2-5 | 6687 | 1.86 | 4.74 |
| 3-3 | 5772 | 1.60 | 2.37 |
| 3-5 | 6728 | 1.87 | 4.55 |
| 4-4 | 1976 + 4324 = 6300 | 1.75 | 1.92 (resume) |
| 4-5 | 6514 | 1.81 | 4.58 |
| 4-6 | 6055 | 1.68 | 4.40 |

Compute 12.78 h (predicted 13.48 h); wall 12.9 h incl. ~6 min down and the lost part of c04_04. Refit on all 14 pairs:
auto 3.34e8, cross 2.94e8 ordered pairs/s (scatter 3.7 %). Prediction for S3 (1-4 1-6 2-2 2-3 3-4 3-6 6-6): 13.0 h.

## Integrity (independent audit agent, all keys force-read)

1552 files (2 autos x 136 + 5 crosses x 256; corrected 2026-11-11 from a miscount of 1456, the per-pair audit covered every file), names equal to `chunk_pairs(16, same)`, every file 74964 bytes with the same key set, bin_slop 0, angle_slop 0,
fold correct, r_nom equal to the product, finite where weight > 0, meanr inside its bin. Bin4_Bin4: 58 files before the kill
(exactly rows a=0..3, last c03_15 at 08:49:44 = last log line), 78 after (rows a>=4, first c04_04 at 09:02:57 = 08:56:23 +
394 s); no file has an mtime in 08:49:45-08:56:23; nothing duplicated or missing; no partial file. chunks/ holds exactly the
14 completed pairs; no session-1 file modified after 00:29:04.

## Exact minus bin_slop 1.5 product (official sigma; 9-bin max for xi, max over modes for COSEBIs)

Computed in the main session and independently by agents; numbers agree.

| pair | xi+ ex-b1.5 | xi- ex-b1.5 | xi+ ex-off | xi- ex-off | xi+ b1.5-off | xi- b1.5-off | E n<=6 | E n<=20 | B n<=6 | B n<=20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 0.071 | 0.103 | 0.070 | 0.101 | 0.010 | 0.010 | 0.151 | 0.240 | 0.103 | 0.181 |
| 2-5 | 0.059 | 0.113 | 0.057 | 0.125 | 0.023 | 0.012 | 0.191 | 0.301 | 0.103 | 0.188 |
| 3-3 | 0.084 | 0.125 | 0.076 | 0.129 | 0.008 | 0.014 | 0.306 | 0.605 | 0.065 | 0.144 |
| 3-5 | 0.056 | 0.057 | 0.068 | 0.091 | 0.024 | 0.034 | 0.083 | 0.116 | 0.016 | 0.098 |
| 4-4 | 0.063 | 0.087 | 0.059 | 0.097 | 0.030 | 0.031 | 0.064 | 0.064 | 0.256 | 0.369 |
| 4-5 | 0.049 | 0.101 | 0.077 | 0.135 | 0.042 | 0.035 | 0.256 | 0.576 | 0.042 | 0.163 |
| 4-6 | 0.088 | 0.114 | 0.064 | 0.107 | 0.036 | 0.018 | 0.069 | 0.102 | 0.120 | 0.292 |

xi: same scale as session 1 (0.044-0.125). Fine-bin differences: median |z| 0.26-0.42, mean z ~1e-3 (no overall shift; not a test of noise-likeness); cross npairs
ratio 1.000005, autos 2.00001 and varxi 0.5 (ordered counting, see SESSION1_CHECK.md); cross varxi ratio within 1e-3 of 1.

## Where the larger E_n/B_n differences come from

E_n differences in 3-3 (0.61) and 4-5 (0.58) exceed session 1 (max 0.43) and the Bin6_Bin6 pilot (0.44). Decomposition
(COSEBIs are linear in xi, so contributions by angular range add exactly):
- Everything below 100 arcmin contributes at most ~0.03-0.04 sigma to any mode; the difference comes from 100-300 arcmin.
- Within that, the outermost 5 of the 1000 fine bins (theta ~ 290-300 arcmin) dominate: 3-3 n=20 -0.55 of -0.61; 4-5 -0.48
  of -0.54; 2-4 +0.44 of +0.43.
- Over all 14 pairs, dropping the last 5 fine bins reduces max|dE| (n<=20) from 0.06-0.61 to 0.03-0.22 and max|dB| (n<=20)
  from 0.07-0.53 to 0.06-0.17 (in a few pairs the edge partly cancels the rest and the value rises slightly). Numbers:
  `session2_check_edge_decomp.json`.
- The fine-bin differences above 100 arcmin average to zero (mean z +-0.006 over 219 bins vs +-0.068 expected for noise), and
  the bin-6 xi difference is not coherent.

(Superseded by the 21-pair decomposition and filter-weighting numbers in `product_exact/EXACT_VS_B1P5.md`.)
Reading: the COSEBI filters weight xi strongly at theta_max, so the re-binning that bin_slop 1.5 causes in the last
few fine bins is amplified into E_n/B_n at the 0.3-0.6 sigma level for some pairs. This is a property of the bin_slop 1.5
approximation (used by the release), not a defect of the exact run. Per the PI decision it is reported, not gated.

Numbers: `session2_check_xi9.json`, `session2_check_cosebis.json` (main-session computation); agents' scripts in the session
scratchpad `s2check/` (audit2/, xi2/, cosebis2/).
