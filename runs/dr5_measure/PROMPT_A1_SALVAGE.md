# PROMPT FOR A NEW SESSION — try to save A1

Paste everything below the line into a fresh session. It is self-contained.

---

We are in `/mnt/d/cloning_kids_Kappa_pipeline` (no git). Interpreter:
`/home/anut/anaconda3/bin/python3` (py3.12, numpy 1.26.4 — do NOT upgrade). Always run with
`PYTHONSAFEPATH= PYTHONPATH=src` because the scripts import sibling modules.

Load the skill `kids-shear-two-point-measurement` first — it has the trap list, the COSEBIs
conventions and the covariance-block gotcha.

## The situation

Check A1 ("their measured xi+ reproduces the official release xi+, fine 1000 bins") currently
FAILS and I do not accept the failure as final. It compares the prior project's measured xi
(`/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/xipm_bin*_bin*.npz`,
1000 log bins, 2–300 arcmin) against the official release's *fine ascii*
(`data/.../cosmosis_xipm/XI_KiDS_Legacy_NScomb_nBins_6_Bin*_Bin*.ascii`).

Result of `scripts/dr5_adopt_01.py` (`runs/dr5_measure/adoption/ADOPTION_01.md`):
- A1 corr min 0.9772 / median 0.9875 / max 0.9957 → script prints "SHAPE PROBLEM".
- A3 the same their-xi, **rebinned to the 9 official bins, vs the official `KiDS_Legacy_xipm.fits`**:
  xiP chi2 0.0303/189, max 0.0528σ; xiM 0.0316/189, max 0.0438σ. Clean pass.
- B1 their xi through OUR COSEBIs integrator vs official `KiDS_Legacy_cosebis.fits`: En/Bn, 6 and
  20 modes, all chi2/dof ~ 0.0002, max 0.052σ. Clean pass.
- B3 B_n null vs the **Bn** covariance block: 6m pte 0.0648 vs published 0.0635; 20m 0.0930 vs
  0.0902. Clean pass.

So the 9-bin and COSEBIs routes agree to <=0.05σ; only the fine-ascii comparison fails. My working
hypothesis was "the fine ascii is bad ground truth". **The evidence for that is weaker than I
first claimed and you should re-test it before relying on it:**

1. I measured ascii/FITS per-pair ratios of 0.80–0.95 by rebinning the ascii with a *median* inside
   each theta bin. That is not the release's own rebinning (`KiDS_Legacy_xipm_pipeline.ini` sets
   `column_npair_name = npairs_weighted`, and `kidspipe.measure.official.weighted_rebin` implements
   the CosmoSIS `bin_xi` module's `weighted_binning=1`). Using the correct weighting the ascii is
   still off the FITS, but by a different (smaller) amount — measure it, do not quote my 0.80–0.95.
2. Baseline I did measure correctly, with `weighted_rebin`: **the official ascii rebinned to the 9
   official bins disagrees with the official FITS at chi2/dof = 3.34 (xiP) and 3.51 (xiM), max
   |d|/sig = 8.0, median ratio 0.879 / 0.858.** i.e. the release is internally inconsistent at
   ~8σ between its own two data products. That is the single most important number here. Verify it
   independently, then decide what it means.
3. Correlation is scale-blind, so "corr 0.977" alone never justified the words "shape problem". I
   have not yet established whether the residual of their-xi/ascii after removing a per-bin
   amplitude is (a) structured in theta, (b) structured in pair, or (c) pure noise.

The prior project (`PRIOR_VALIDATION_01.md` §6.4, and their `REPORT.md` item 2) flags the fine
ascii as coming from a different, probably **blinded**, realisation, with a per-tomographic-bin
amplitude pattern (0.893, 0.913, 0.917, 0.932, 0.949, 0.971) and Δχ² = 21.6, and says confirming
with A. Wright is still a TODO. My separable fit to the 9-bin ratios gave per-bin factors 0.929,
0.953, 0.967, 0.968, 0.995, 1.012 and an overall 0.913 — for xiP and xiM almost identically, which
is at least consistent with a per-bin calibration/blinding offset rather than a shape error. But
the two patterns are NOT the same numbers, so one of the two measurements is wrong. Find out which.

## What I want from you

**Decide, on evidence, whether A1 can be made into a meaningful, passing check — and if so, make it
one.** Concretely:

1. Reproduce the three numbers above from scratch in your own script
   (`scripts/dr5_a1_salvage.py`, write outputs to `runs/dr5_measure/a1_salvage/`):
   the ascii-vs-FITS chi2/dof with `weighted_rebin`, the their-xi-vs-ascii residual structure, and
   the their-xi-vs-FITS chi2 (should come out ~0.0002 as in A3).
2. Test the amplitude hypothesis properly, not with a median: fit
   `their_xi(theta) = A_pair * B_bin_i * B_bin_j * ascii(theta) + ...` over the 1000 bins and see
   whether the residual is then white. Report the fitted `B_bin`, its errors, and whether it matches
   the prior project's (0.893…0.971) or my (0.929…1.012) or neither. A per-tomographic-bin factor
   appearing in *both* xi+ and xi- with the same value is the signature of a calibration/blinding
   offset; a different value in each is not.
3. Test the alternative: maybe their xi is the odd one out and the ascii is fine. Rebin their xi to
   9 bins with the *same* weighting and compare to the FITS (that is A3, which passes). Also
   integrate the **ascii** through our COSEBIs integrator and compare to the official COSEBIs FITS.
   If the ascii passes the COSEBIs route but their xi also passes it while ascii vs their-xi fails,
   the fine-bin disagreement is confined to scales the COSEBIs weight down — say so quantitatively
   (which theta range carries the chi2).
4. Then write A1 as whatever the evidence supports, in one of these forms, and say which you chose
   and why:
   - **A1a** their xi vs the official 9-bin FITS data vector (same instrument as A3, different
     source) — this is basically A3 and may be redundant; if so say that plainly rather than
     shipping two names for one check.
   - **A1b** their xi vs the ascii, but as a *shape-only, amplitude-marginalised* test: fit one
     scale per pair, then chi2/dof on the residual with the official covariance. This is honest and
     is the check I actually wanted (shape agreement, calibration separate).
   - **A1c** if the ascii is genuinely unusable: retire A1 with a documented reason and a
     quantitative statement of the release's internal inconsistency, and move the sample-identity
     job entirely onto the galaxy-level gate (`scripts/dr5_a1_gate.py`, already PASS: row counts =
     Table 1 `Ngold`, n_eff within 0.4%).
5. Update `runs/dr5_measure/RESUME.md` (the A1 row and the "Bugs already found" list) and write a
   short `runs/dr5_measure/a1_salvage/A1_SALVAGE.md`. If you conclude the ascii should be retired,
   that decision is for the PI to sign off — present the evidence and a recommendation, do not
   silently rewrite the gate.

## Do not

- Do not start the ~49 h production pair-count run. It is not needed for any of the above.
- Do not re-litigate: the sample has no missing selection cut (PAPER_SELECTION.md), the sign
  convention is `flip_g1=False, flip_g2=False` (ξ− decides it at 7.7σ), the m-correction is
  `ξ_corr = ξ_raw/[(1+m_i)(1+m_j)]` with Wright+25 Table 1 values, and `process(cat)` counts
  unordered pairs while `process(cat,cat)` counts ordered.
- Do not compare raw `npairs` across projects: ours are ordered integers, theirs are unordered, and
  the official column is neither (not integral). Use `npairs_weighted` or nothing.
- Do not use `np.allclose` on values ~1e10 to test integrality (`floor(x)==x`).
- Do not use the En covariance block for a Bn null test: `COVMAT` is [En(420)|Bn(420)]; Bn needs
  `+21*20`. Gate: a Bn null test that does not give pte 0.0635/0.0902 is on the wrong block.
- Do not put unrelated study notes into `artifacts/` or `runs/` — this project's notes go in
  `runs/dr5_measure/`.

## Files to read first

1. `runs/dr5_measure/adoption/ADOPTION_01.md` — the result table and the numbers above.
2. `runs/dr5_measure/PRIOR_VALIDATION_01.md` — §2 estimator convention, §6.4 the ascii caveat.
3. `runs/dr5_measure/RESUME.md` — status table, settled decisions, cost warning.
4. `scripts/dr5_adopt_01.py` — A1/A3/B1/B3 as they stand (~200 lines).
5. `src/kidspipe/measure/official.py` — loaders; note `weighted_rebin` is the CosmoSIS-matching
   rebin and is currently unused by `dr5_adopt_01.py` (which uses an npairs-weighted mean of its
   own — check whether those differ).
6. `src/kidspipe/measure/xi2cosebis.py` — the binned integrator (scheme="edges", unit="rad").

Report back with: the three reproduced numbers, the amplitude-fit result, which A1 form you
implemented (or why none), and the two-line PI decision needed.
