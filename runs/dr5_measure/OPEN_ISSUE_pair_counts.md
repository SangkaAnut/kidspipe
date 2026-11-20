# DR5 measurement — open issue: 1.417x pair-count excess

> **RESOLVED — superseded by `PAPER_SELECTION.md`. Read that first.**
>
> The diagnosis in this note ("the release measured a smaller footprint than the file we hold") is
> **wrong**. Our per-bin row counts are digit-for-digit the `Ngold` column of Table 1 in Wright et
> al. 2025 (arXiv:2503.19441), i.e. the file IS the published KiDS-Legacy gold sample, and
> `sum(weight)` matches the official ascii to 0.3–0.4% on Bin1_1 / Bin2_2 / Bin1_2.
> The 1.406 factor is in the official `npairs` **column definition**, which is continuous, not an
> integer pair count (mean |frac−0.5| = 0.250). There is no footprint to find and no cut to apply.
> Also note: the "ratio drifts 1.394→1.440" claim below was an artefact of rebinning a stored probe
> into 20 coarse bins; on 1000 bins the ratio is flat.
>
> Kept for the audit trail only.

Status: A1 (pair counts) FAILS by a constant factor. Shape of xi_+ matches.

## Measured  (CORRECTED -- see note below)
- Our npairs / official npairs = **1.40614** (tomographic bin 2 vs official Bin2_Bin2,
  1000 log bins, median, spread 1.3915-1.4271)
- At 20 coarse bins: 1.41635, and the ratio is NOT flat: it rises 1.394 -> 1.440 from
  2 to 300 arcmin (~3.3%), so it is not a pure density normalisation.
- implied surviving fraction f = 1/sqrt(1.406) = 0.843
- xi_+ shape correlation with official: r = 0.99886; xi_-: r = 0.98701
- median ratio of xi_+: ~1.15 -> xi amplitude is only a few per cent high while npairs is ~41% high

### Correction
An earlier version of this note quoted 1.41698 against official Bin1_Bin1. Both saved probes
were produced before the `load_bin()` off-by-one fix and therefore contain TOMOGRAPHIC BIN 2,
so that pairing was mismatched. Recomputed with the correct pairing by
`scripts/dr5_fix_ratio.py` (output: `runs/dr5_measure/fix_ratio.log`): 1.40614.
The 0.77% difference does not change any conclusion, but the wrong number must not be quoted.

## Area (occupancy on a (RA, sinDec) equal-area grid)
| nside/deg | area deg2 | gal/pix |
|---|---|---|
| 40  | 1466.9 | 181.1 |
| 60  | 1438.6 |  82.0 |
| 80  | 1365.3 |  48.6 |
| 120 | 1343.4 |  22.0 |
| 160 | 1310.2 |  12.7 |
| 200 | 1278.8 |   8.3 |  (not converged; occupancy too low)

Adopted ~1310 deg2 (still drifting by a few % -> treat as 1280-1320).
Number density: 7,416,371 / 1310 deg2 = 1.572 gal/arcmin2 vs official NGAL_1 = 1.6884.

## Two hypotheses tested, neither fits
- Pure AREA difference (same density): predicts ratio A_ours/A_off = 1310/1014 = 1.29 (measured 1.417)
- Pure DENSITY difference: our density is LOWER than official (0.93x), which would predict
  npairs ratio < 1, not 1.417
- Joint (n_gal^2/A): (7.416/6.164)^2 x (1014/1310) = 1.12 (measured 1.417)

So neither pure-area nor pure-density, and no per-column cut gives a uniform 84%:
| cut | surviving fraction |
|---|---|
| SG2DPHOT != 1 (drop stars) | 1.000 (stars are 0.02% of rows) |
| model_SNratio > 5 | 1.000 |
| Z_B > 0.1 | 1.000 (Z_B is binned in 32 values, per-bin ranges) |
| gold/shear_weight_only > 0 | 1.000 |
| MAG_AUTO < 25 | 1.000 |
| n_exposures_used == 5 | 0.457-0.502 (bimodal, wrong shape and too aggressive) |
| model_SNratio > 10 | 0.92..0.56 (wrong shape) |

## Duplicates
5.0% of rows (355,533 of 7,045,183 positions) share a rounded (RA,Dec) to 1e-5 deg,
max multiplicity 5, consistent with overlapping tiles. Too small to explain 41.7%,
but it does inflate small-separation pairs and should be investigated separately.

## What would settle it
1. The official DR5 footprint mask / patch list (the `PATCH` column here is only North/South,
   so sub-patch boundaries are not available to us).
2. The exact selection described in Wright et al. 2025 (DR5 paper) sect. 3 -- which cut,
   applied to which catalogue version, gives 1014 deg2 / 1.6884 arcmin^-2.
3. Cross-check TreeCorr's npairs definition for auto-pairs against the official file
   (e.g. whether it counts ordered pairs) -- would give a factor, not 1.417, but worth ruling out.

Nothing under /mnt/d/HI-gamma-correlation/ was modified.
