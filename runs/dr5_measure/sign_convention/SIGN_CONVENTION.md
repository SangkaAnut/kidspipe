# Shear sign convention, KiDS-Legacy DR5 catalogue (task B)

Script `scripts/dr5_sign_convention.py`; every number below is in `runs/dr5_measure/sign_convention/sign.json`
(keys `parity`, `pilot`, `ggl`, `official_search`) or `ggl_{a,b,c,d}.npz`; plot `plots/dr5_sign_ggl.png`. Run on 2026-10-30,
TreeCorr 5.1.4, seed 20261030.

## Verdict (one line)
Use the catalogue `e1, e2` **as is** with TreeCorr (`g1=e1, g2=e2`, `flip_g1=False, flip_g2=False`, `ra, dec` in degrees):
gamma_t > 0 at 8.6 sigma and gamma_x consistent with 0; the three other sign choices give gamma_t ~ 0 (two) or gamma_t < 0 at
-8.6 sigma (both flipped). xi+/xi-/COSEBIs real parts are unaffected by the global sign; xi- needs the parity right, and it is.

## 1. Analytic content (checked numerically)
Complex shear g = g1 + i g2 in a frame (x,y); phi = direction of the separation vector in that frame.
xi+ = <g_i g_j*> (no phi), xi- + i xi_x = <g_i g_j e^{-4 i phi}> (TreeCorr: xim, xim_im), gamma_t + i gamma_x = -<g e^{-2 i phi}>.

| transformation | Re xi+ | Re xi- | Im xi+ (cross-catalogue) | xi_x = Im xi- | gamma_t | gamma_x |
|---|---|---|---|---|---|---|
| g -> -g (e1,e2 both) | invariant | invariant | invariant | invariant | **sign flips** | sign flips |
| e2 -> -e2 (g -> g*) | invariant | **changes** (-> ~0 in expectation, see below) | sign flips | changes | -> ~0 in expectation | changes |
| e1 -> -e1 (= g -> -g* = e2->-e2 plus global sign) | invariant | **changes** (same as e2->-e2) | sign flips | changes | -> ~0 in expectation (sign-reversed vs e2->-e2) | changes |
| phi -> -phi (RA direction / position-angle handedness) alone | invariant | **changes** | invariant | changes | -> ~0 | changes |
| phi -> -phi **and** e2 -> -e2 together (true mirror of sky) | invariant | invariant | sign flips | sign flips | invariant | sign flips |

Why: xi+ contains no phi and conj(g) gives the same product -> invariant under every one. For g -> g* the product
g_i* g_j* e^{-4i phi} = (g_i g_j e^{+4 i phi})*, and for an isotropic field <g_i g_j> = xi- e^{+4 i phi}, so the pair average is
~ xi- e^{8 i phi}, whose azimuthal average is 0: **xi- (and gamma_t) of a wrong-parity catalogue vanish in expectation**; they do
not flip sign. Only (e1,e2) -> (-e1,-e2) leaves xi- exactly alone, and only that and its parity partner are separable by xi-:
the two classes are {(e1,e2), (-e1,-e2)} and {(e1,-e2), (-e1,e2)}. xi+ and xi- (any sign of g) cannot tell g from -g.

Numerical check (`parity` stage, 199,264 real bin-6 galaxies in a 6x6 deg box, TreeCorr bin_slop=0, 2-100 arcmin, plus a
cross-correlation of two random halves for Im xi+; synthetic tangential-shear field, 300 lenses, 200,000 sources):
- xi+ identical in all 6 transformations to <= 4.6e-4 of the std of the curve (the residual is TreeCorr's spherical geometry,
  not physics); g->-g reproduces xi+ and xi- to 1e-13.
- e2->-e2 or e1->-e1 or "mirror RA only": xi- changes by 2.3-2.4 std of the curve; "mirror RA and e2->-e2": xi- changes by 0.17 std
  (should be 0 for an exact isometry; this is a TreeCorr spherical-projection residual of ~0.15 std, see Caveats), xim_im sign-flips.
- Cross-catalogue Im xi+ sign-flips under e2->-e2, e1->-e1 and mirror+e2->-e2 (invariant for mirror alone / g->-g).
- Synthetic gamma_t / injected gamma_t: identity 1.000, g->-g -1.000, e2->-e2 +0.068, e1->-e1 -0.068, mirror+e2->-e2 0.9999,
  mirror alone 0.068; |gamma_x|/|gamma_t| <= 0.02 everywhere (identity: 0.0048).
- In the full-catalogue pilot (theirs, 1000 bins, Bin6_Bin6): e2-flipped run `outputs/xipm/e2flip/xipm_bin6_bin6.npz` has xi+ amplitude
  0.9997 and xi- amplitude **-0.057** (corr -0.10) relative to the fiducial: xi- collapses to ~0, as predicted.

TreeCorr's frame (`treecorr_frame_single_pair_gt`): for ra/dec input, a source NE of a lens needs g2 > 0 to give gamma_t > 0
(NW needs g2 < 0): TreeCorr's frame is x = WEST, y = NORTH (East to the left, the standard sky-image frame; tangential shear
-Re(g e^{-2 i phi}), phi from West towards North).

**`flip_g2=True` in tests/test_measure.py.** `PairCounter` measures phi = atan2(dDec, dRA cos Dec): frame x = EAST, y = NORTH, the
mirror image of TreeCorr's. A given (e1,e2) in PairCounter's frame is therefore (e1,-e2) in TreeCorr's. Measured (real 2x2 deg patch,
19,819 galaxies, `paircounter_vs_treecorr_real_2x2deg`): PairCounter(e1,e2) vs TreeCorr(e1,e2,flip_g2=True): xi+ 0.005 std, xi- 0.14 std,
xix = -xim_im (0.07 std; +xim_im disagrees at 2.3 std); vs TreeCorr with no flip: xi+ still 0.005 std (invariant), xi- 14 std off,
xix 6.3-6.7 std off. Per statistic: xi+ unaffected; xi- changes (so the flag is required for agreement); xix/xim_im sign-labelled
opposite and also needs the flag; gamma_t would change. **Consequence: `PairCounter` fed the raw catalogue e1,e2 is the wrong parity
for this catalogue (xi- ~ 0). It must be fed (e1, -e2). The production run uses TreeCorr without flips, so the pilot/production are right.**

## 2. Official convention (not stated)
Searched the release in `/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_cosmic_shear_data_release_unpacked/...`:
`README.txt` (only line 37: "2PCF measurements from treecorr"), the four `*_pipeline.ini`, `values/priors.ini`, all HDU headers of
`KiDS_Legacy_xipm.fits` and `KiDS_Legacy_cosebis.fits`, the ascii header. **There is no `flip_g1`, `flip_g2`, e1/e2 sign or position-angle
statement anywhere in the release** (`xipm/KiDS_Legacy_xipm_pipeline.ini:56-57` only name the columns `meanr`, `npairs_weighted`; the
release ships no TreeCorr config). Not available locally: the catalogue readme and the TreeCorr configs of the original measurement.
The HI project asserts "g1=e1, g2=e2, no flip, as in Wright+25" (`HI-gamma-correlation/pipeline/01_lensing_validation/REPORT.md:93`,
`code/run_finalize.py:66-67`) but itself marks it as "validated empirically ... by the xi- match and the e2-flip variant", not quoted.
Our evidence is empirical: (i) ours (TreeCorr, no flip) vs theirs (no flip) vs the official ascii (blinded same sample): Bin6_Bin6 xi-
amplitude 0.996 and Pearson 0.99993 (ours vs theirs, 20 coarse bins), ascii vs theirs corr 0.9957 (xi-), 0.9995 (xi+); (ii) theirs
e2-flipped destroys xi- (above). The HI project's separate HEALPix/NaMaster run found q,u = (-e1, e2) with C_gE > 0
(`outputs/sign_convention.json`: `healpix_shear_convention`), consistent with x=West, y=North (not re-derived here).

## 3. Physical test: galaxy-galaxy lensing (decisive for the sign)
Lenses: 300,000 random bin-1 galaxies (`bin0.npz`); sources: all 5,960,461 bin-6 galaxies with catalogue weight `w`; randoms: 600,000
random bin-6 positions (follows the source footprint; this subtracts additive systematics); TreeCorr NGCorrelation, 15 log bins 1-100
arcmin, bin_slop 0.1; 50 kmeans patches built on the lenses; jackknife covariance of (lens-source) - (random-source) via
`estimate_multi_cov`; chi2 with the Hartlap factor (50-15-2)/(50-1); 15 dof. No m correction (it is a ~4% rescale, sign-blind).

| variant | gamma_t chi2/15 (PTE) | gamma_x chi2/15 (PTE) | amplitude of a 1/theta template (gamma_t) | S/N |
|---|---|---|---|---|
| a: catalogue e1,e2 as is | **88.8 (1.6e-12)** | 14.9 (0.46) | **+2.15e-3 +- 0.25e-3** | **+8.6** |
| b: flip_g2 | 7.7 (0.93) | 6.5 (0.97) | -0.44e-3 +- 0.29e-3 | -1.5 |
| c: flip_g1 | 7.7 (0.93) | 6.5 (0.97) | +0.44e-3 +- 0.29e-3 | +1.5 |
| d: flip_g1 + flip_g2 | 88.8 (1.6e-12) | 14.9 (0.46) | -2.15e-3 +- 0.25e-3 | -8.6 |

gamma_t(a) per bin: 0.12e-3 at 1.2 arcmin, 0.84-0.86e-3 at 1.6-3.0 arcmin (max 0.86e-3), falling to 0.05e-3 at 90 arcmin; positive in all 15 bins
(0.2-5.4 sigma, the 5.4 sigma bin at ~14 arcmin), diagonal S/N 11.7. gamma_x(a): |value/sigma| <= 2.8, one bin at -2.8 sigma (4 arcmin), amplitude -0.28e-3 (consistent with 0
by chi2). b and c are exact negatives of each other and d of a (linearity, max |b+c| = 1.7e-18) and are consistent with zero, as the analytic
section predicts for a wrong parity. The random-point signal itself is small but nonzero (gamma_t ~ +0.3e-3 at 1-3 arcmin, rms 0.56 of the
rms jackknife error); without random subtraction (a) is positive too (`gt_no_random_subtraction`), so the conclusion does not depend on it.

## 4. Recommendation
(i) xi+/xi- and COSEBIs: use TreeCorr with `g1=e1, g2=e2`, **no flips**. The global sign is irrelevant for Re xi+ and Re xi-; parity matters for
xi- (and so E/B split) and that is fixed by (a) matching the official xi-. Do not feed `PairCounter` the raw e2 (use -e2).
(ii) Cross-correlation with an external field (HI, galaxy positions; /mnt/d/HI-gamma-correlation): build the TreeCorr catalogue with
`treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2, w=weight, flip_g1=False, flip_g2=False)`; process with
`NGCorrelation` / `KGCorrelation` (external field as `k`, or lens positions). Then `xi` = gamma_t, which is **positive for tangential
alignment around over-densities** (verified here), and `xi_im` = gamma_x (null test). Give TreeCorr the true (ra, dec) of each external
cell (do not mirror the map); divide by (1+m) per source bin and subtract the random-point (or randomised-field) signal. If shear
maps are built on a grid instead, use q = -e1, u = +e2 (HEALPix basis) -- the HI project's own result, not re-derived here.

## 5. Established / not established
Established: items above, in the stated range (bin-1 lenses x bin-6 sources, 1-100 arcmin; one random-point realisation, 600k randoms; 50 jackknife patches).
Not established: (1) the official TreeCorr configuration (no source in the release); (2) other lens/source bins and the stability of the
8.6 sigma to the lens sample, but the test is sign-only and the sign is robust per-bin; (3) the mirror-exactness residual of TreeCorr xi-
on the sphere (~0.15 std, unexplained, outside this task; does not affect any conclusion, since the wrong-parity effect is 2-14 std);
(4) the pilot alone says nothing about the global sign; only the parity class. (5) The sample variance of gamma_t from the jackknife with
50 patches and a 30x30 joint covariance is noisy; chi2 values carry this uncertainty (S/N 8.6 is the more robust number).

---
## Verifier note (2026-10-30, non-blocking; details in VERIFY.md)
Sign verdict independently CONFIRMED (new lens draw, footprint-uniform randoms, 40 patches: gamma_t S/N +15.6 as is, +-1.0 for single flips, -15.6 both flipped; amplitude differs from the S/N 8.6 above, so quote the sign, not the amplitude).
Correction to Caveat (3) / Section 1: the ~0.15-0.17 std "mirror residual" of TreeCorr xi- is not a mirror effect. ra->-ra and ra->360-ra mirror the same data exactly (1e-13 std),
while a pure rotation about the pole (ra+1, +42, -100 deg, no mirror) changes xi- (bin_slop 0, box ra 198-204, dec -4..2) by 0.09-0.20 std with identical npairs. It is an unexplained RA-offset dependence of TreeCorr xi- at the ~8e-6 level, relevant to how tight an xi- comparison can be.
