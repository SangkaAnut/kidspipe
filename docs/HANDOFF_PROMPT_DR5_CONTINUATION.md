# Handoff prompt: continue the KiDS-DR5 shear measurement — resolve the selection via the DR5 paper

> Paste everything below the line into the new session. It is self-contained.

---

## 0. Who you are and the one decision already made

You are continuing a scientific software project at `/mnt/d/cloning_kids_Kappa_pipeline`
(note the capital K; not a git repo). A previous session built the measurement layer and ran
the cheap validation: the COSEBIs integration check (B1) and the two-estimator cross-check (A4)
**pass**, but the pair-count check (A1) fails by a constant factor **1.406**, which blocks
A2/A3/B2/B3.

The user has decided the resolution: **option (b) — recover the exact galaxy selection used by
the official release from the DR5 paper, apply it, and re-measure.** Do not guess cuts, and do
not start the 21-pair production run until A1 is fixed.

Read, in this order:
1. `runs/dr5_measure/RESUME.md` — status table, open question, what is already ruled out.
2. `runs/dr5_measure/OPEN_ISSUE_pair_counts.md` — the full diagnostic evidence.
3. `docs/HANDOFF_PROMPT_DR5_MEASUREMENT.md` — the original task spec (steps, checks, deliverables).
4. `docs/interface_notes/cosebis.md` and `docs/KIDS_LEGACY_VALIDATION.md` — the validated
   theory/likelihood side, which you must not modify.
5. Skill `kids-shear-two-point-measurement` (`skill_view`) — every trap already paid for.

## 1. The problem in one paragraph

Our catalogue file `/mnt/d/HI-gamma-correlation/KiDs_DR5_lensing/KiDS_Legacy_NS_unblind_final.fits`
(40,894,394 rows, TOMOBIN 1..6, no other cuts applied) covers ~1310 deg² at 1.572 gal/arcmin⁻²
in tomographic bin 1. The official release products imply **1014 deg² at 1.6884 gal/arcmin⁻²**
(`NGAL_1` header of `KiDS_Legacy_cosebis.fits` / `KiDS_Legacy_xipm.fits`). Consequently our
pair counts are **1.406×** the official ascii values while the xi_+ shape matches at
r = 0.999. Same signal, more sky, slightly lower density: the release measured a smaller
footprint than the file we hold. The missing ingredient is a selection/footprint cut, not an
estimator error and not a weighting error.

Already ruled out by measurement (do not redo):
- per-column cuts, all tested: stars (`SG2DPHOT==1`) are 0.02% of rows; `model_SNratio>5`,
  `Z_B>0.1`, `gold_weight_only>0`, `shear_weight_only>0`, `MAG_AUTO<25` keep ~100%;
  `n_exposures_used==5` keeps 0.46–0.50 with the wrong bin dependence; `model_SNratio>10`
  keeps 0.92→0.56 across bins (wrong shape).
- pure area difference predicts 1.29; pure density difference predicts <1; joint n²/A predicts 1.12.
- the ratio drifts with separation (1.394 → 1.440 over 2–300 arcmin), so it is not a pure
  normalisation.
- 5.0% of rows share a rounded sky position (max multiplicity 5, consistent with overlapping
  tiles) — too small to explain 1.406, but it does contaminate the smallest separations and
  must be handled if the official pipeline applied a de-duplication.

## 2. Your task, in order

1. **Get the selection from the paper.** Wright et al. 2025, A&A 686, A170 is the KiDS DR5 data
   release paper; the cosmic shear analysis is Wright et al. 2025, A&A 703, A158
   (the original spec cites arXiv:2503.19441 for the latter). **Look up the DR5 paper's arXiv
   id yourself — do not trust an id you have not verified.** Extract, with section/table
   numbers:
   - the footprint definition (which tiles/patches, any mask, the mask apodisation scale),
   - the effective area quote and compare it with 1014 deg²,
   - every cut applied to the shear catalogue (star/galaxy, S/N, number of exposures,
     magnitude, photo-z range, weight flags),
   - whether the catalogue is de-duplicated across overlapping tiles,
   - whether `weight` is used as shipped or re-derived.
   Use the arXiv source/LaTeX if the PDF text is ambiguous; `web_extract` on `arxiv.org/abs/`
   and `arxiv.org/pdf/` is cheapest. Record quotes verbatim into
   `runs/dr5_measure/PAPER_SELECTION.md` with section numbers.
2. **Implement the selection** as a new function in `src/kidspipe/measure/catalogue.py`
   (create it), with the cut values as named constants carrying the citation. Re-extract the
   compact per-bin NPZ files to a **separate** directory (do not clobber `~/kids_dr5_work/`,
   which is the known-good run) — e.g. `~/kids_dr5_work/sel/`.
3. **Re-run the selection gate** on the new sample: `n_eff/NGAL_i` must give the same implied
   area in all six bins **and** that area must come out near 1014 deg², and
   `sigma_e` must match `SIG_E_i` to a few per cent. This is a hard gate: if it does not land
   on 1014 deg², the selection is still wrong — iterate on the paper, do not proceed.
4. **Re-check A1** on one auto pair and one cross pair (tomo bin 1, coarse 20 bins is enough):
   the npairs ratio must come out 1.000 ± a few 1e-3, per bin. Only then continue.
5. **Settle the shear sign convention on real data.** The toy-field test is degenerate between
   `flip_g1=0,flip_g2=1` and `flip_g1=1,flip_g2=0`; only correlation with the official ascii
   xi_− breaks the tie. Run both, adopt the one giving xi_− correlation ≈ +1 with a positive
   slope, and record the numbers.
6. **Then the 21-pair production run.** ~2.5–3.0e8 pairs/s → 3.06e13 pairs ≈ 30–50 h.
   `bin_slop` is not a lever and coarse bins do not help. Get the user's explicit go-ahead on
   the budget first; prefer the resumable driver `scripts/dr5_measure_xi.py` (writes one npz
   per pair, so an interrupted run restarts where it stopped).
7. **Then** A2, A3, B2, B3, plots, report (numbers 1–6 of the original spec, section 5).
8. **Remember the second deliverable.** The original task requires
   `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md`: a self-contained prompt the user pastes into Claude
   Code to audit the work independently. It must list every claim with the file holding the
   number and the script that produced it, ask Claude to re-run a cheap subset with its own
   short scripts, and ask it to hunt the specific failure modes named in the original spec
   section 6 (sign flip, double/missing m-correction, `weight` vs `shear_weight_only`, TOMOBIN
   mapping, arcmin vs radians in the COSEBIs integral, trapezoid vs bin-averaged integration,
   patch combination, bin_slop effects, tests that cannot fail, report numbers that do not
   match the files). State the user's standing preference: if Claude delegates, use Sonnet 5.5
   subagents at medium effort; report findings ranked by severity. Keep it under ~150 lines.

## 3. Non-negotiable engineering constraints

- Interpreter: `/home/anut/anaconda3/bin/python3` (py3.12.4, numpy 1.26.4, astropy 6.1.0,
  numba 0.59.1, treecorr 5.1.4, fitsio 1.4.2). **Do not upgrade numpy** — numba 0.59 needs
  <1.27 and the emulator artefact is tied to camb 1.5.8. Do not use other conda envs.
- Baseline test suite: `cd /mnt/d/cloning_kids_Kappa_pipeline && PYTHONPATH=src $PY -m pytest -q -m "not slow"`
  → 287 pass; `tests/test_emulator.py::test_shared_model_threads_last_derived_is_thread_local`
  is a load-sensitive flake that passes in isolation. Re-run it alone before reporting a regression.
- ≤16 cores, keep RSS under ~10 GB, never load the full catalogue into memory.
- Never modify anything under `/mnt/d/HI-gamma-correlation/` (read-only data release), and do not
  change validated code in `src/kidspipe/theory`, `likelihood`, `legacy` except additively.
- Write results to `runs/dr5_measure/` (JSON/NPZ/logs) and plots to `plots/dr5_*.png`; every
  number in any report must trace to a file a named script wrote.
- The conda shell hook prints a harmless libmamba crash report in every non-interactive
  subshell; it does not affect exit codes. Ignore it.

## 4. State you inherit (verified, do not redo)

| item | value |
|---|---|
| catalogue rows | 40,894,394; TOMOBIN 1..6 only; 0 rejected; 0 non-finite weights |
| per-bin counts | 7,416,371 / 7,359,911 / 6,799,681 / 6,880,432 / 6,477,538 / 5,960,461 |
| compact cache | `~/kids_dr5_work/bin{0..5}.npz` (bin0 = tomographic bin **1**) + `extra/` (12 columns) |
| binning | TreeCorr Log, 1000 bins, 2–300 arcmin; `r_nom` = geometric centre (agreement 5e-13) |
| B1 | **PASS** — official ascii xi → E_n: χ²/126 = 0.098; `runs/dr5_measure/b1_integration.json` |
| A4 | **PASS** — independent counter vs TreeCorr: npairs ratio exactly 0.500000 (ordered vs unordered), xi_+ r = 1.00000, xi_− r = 0.99999; `runs/dr5_measure/a4_estimator.json` |
| A1 | **FAIL** — 1.406× (tomo bin 2 vs official Bin2_Bin2) |

Settled conventions (evidence in `b1_integration.json`, do not re-litigate):
- COSEBIs integral uses **theta in radians** in `dθ θ` (arcmin gives χ²/dof ~ 7e14).
- bin weight `w_k = (b_{k+1}² − b_k²)/2`; `centre` identical; composite trapezoid wrong (χ²/126 = 4.84).
- `meanr` vs `r_nom` indistinguishable (~1e-5).
- `theory/cosebis.E_from_xi` **raises** on binned data (centres span only 2.005–299.25); use
  `kidspipe.measure.xi2cosebis.e_from_xi_binned`.

Bugs already fixed — do not reintroduce:
1. `load_bin()` had a 1-based/0-based off-by-one and read tomographic bin b+1. This is why an
   earlier headline ratio was quoted as 1.417 against the wrong official pair (correct: 1.406).
2. `paircount.py` computed xi_− as `gA * conj(gB)`, whose phases cancel and silently reproduce
   xi_+. Correct form is `gA * gB` (net e^{−4iφ}).
3. TreeCorr needs `sep_units='arcmin'` or every bin is silently empty with no warning.

## 5. Definition of done for this continuation

- [ ] `runs/dr5_measure/PAPER_SELECTION.md` with the cited selection and the section numbers.
- [ ] Selection implemented and the gate lands on ~1014 deg² with per-bin consistency.
- [ ] A1 passes (ratio 1.000 ± few 1e-3) on a coarse test before any long run.
- [ ] Sign convention fixed on real data, with the numbers that decided it.
- [ ] (with user's budget approval) A2, A3, B2, B3 computed and reported honestly.
- [ ] Plots `dr5_01` … `dr5_06` produced and actually inspected.
- [ ] `docs/DR5_MEASUREMENT_REPORT.md`.
- [ ] `docs/PROMPT_FOR_CLAUDE_CROSSCHECK.md` — do not let this slide; it is a required deliverable.
- [ ] `tests/test_measure.py` fast and synthetic (no 16 GB file), full suite still ≥287 pass.
- [ ] Nothing under `/mnt/d/HI-gamma-correlation/` modified.
