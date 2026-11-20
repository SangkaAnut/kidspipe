# AUDIT_REFEREE: adversarial read of the DR5 product, report and plots (2026-11-02)

Scope: `docs/DR5_MEASUREMENT_REPORT.md`, `product/README.md`, `product/FULLVECTOR.md`, `plots/dr5_01..06`, `tests/test_dr5_product.py`, `tests/test_measure_unordered.py`, spec sections 4-7. Read-only; nothing was fixed. All seven `dr5_0*.png` were opened and viewed. Not re-run: `test_measure.py`, the full suite, any xi measurement.

What I ran: the two test files (11 passed, 19 s, 1 RuntimeWarning "invalid value in divide" in `test_eval_scale_invariant_and_ordered_normalisation`); three mutation probes of `validate_product` (below); a comparison of product `xip_raw/xim_raw` with the 21 source `b1.5/*.npz` (max difference 0, so raw is bitwise the source).

## Problems, by severity

### MAJOR

1. **Figure 1 caption does not match the plot (report 4.1 and Figure 1).** The section is titled "Fine-bin xi+/- against the release" and the caption says "ours filled, official ascii open with sigma bars ... strip shows Delta/sigma". The plot (title and side text) shows the 9 rebinned bins, m-corrected, against the official **FITS** (error bars = sqrt(diag COVMAT)), not the 1000-bin ascii. Nine points per panel is what is drawn; the fine-bin A2 is not plotted anywhere. The spec's plot 1 is "21 panels of theta*xi with the ratio or difference in sigma"; the substitution is not disclosed, and "Ours and the ascii overlap" is false as written (the overlap is with the FITS). Fine-bin A2 exists only as the FULLVECTOR table.

2. **Product test is blind to E_n and B_n, and to anything not tied to the 9-bin xi.** Probe: multiplying `En` by 3 and zeroing `Bn` still passes `mk.validate_product`, and no test touches En/Bn values, `r_nom`, `npairs`/`weight` content, or the stored hashes against the files. The product's COSEBIs half (a headline deliverable) is therefore untested by `tests/test_dr5_product.py`; only shape is asserted. It can fail on xi (see item 8 for what it does catch), but the report's "tests" row gives no hint of this limit.

3. **A4 "PASS" is conditional and is stated as a pass.** The independent counter agrees with TreeCorr on xi- only after flipping g2 (3.97 / 3.21 sigma without; Figure 6 shows this clearly). The product uses TreeCorr with g2 as stored. So the estimator cross-check validates the product's xi- convention only if the "mirror frame" explanation (report 3.4) is right; that explanation is asserted from the helper's coordinate convention, not shown by an independent test on the product. Also the check is on 0.25 % of galaxies, theta <= 60 arcmin, bin_slop 0 (not 1.5), so it says nothing about the bin_slop 1.5 pair counting that is the product, nor about 60-300 arcmin where xi- is largest. The report does state the size limits; the table status "PASS to 0.05 sigma" still reads stronger than "PASS with a flip of one input".

4. **"PASS" labels on ungated checks.** B1 is "PASS with caveat" in the table, yet the spec gives no target and the report's own Figure 4 shows B1 residuals up to -1.6 sigma and rms 1.2 sigma at n = 1 (E_n scale ratio 0.907). B2 is correctly "ungated"; B1 and B3 are not consistently so. B3 "PASS" tests only that the official and our chi2 agree (the official value is reproduced by construction); neither tests the null itself (PTE 0.06-0.09 is marginal, and Bin6_Bin6 6-mode PTE 0.004 is mentioned only in 4.4). Recommend "no gate; consistent" for B1/B2/B3.

### MEDIUM

5. **A1 and A3 thresholds were fixed after seeing the data, and the report says so, but the summary table still shows plain PASS.** A3's 0.1 is, in the report's own words, "a round number about twice the observed maximum" with no null simulation deriving it. The report discloses this in 1.2 only; the Figure 1 side-text and the table say "PASS" without the qualifier. The A3 gate is also the only amplitude check at 1 % resolution, and the release used a different TreeCorr version (disclosed repeatedly, good).

6. **Plot 1 strip axes are fixed at +-0.1 (the gate).** Any residual above 0.1 would be clipped at the axis edge rather than shown. It does not matter at present (max 0.053), but the range was chosen to make the pass look flat; an auto-ranged or log strip would show how far inside the gate the values are. Similarly Plot 2 bottom panels use a +-2e-5 window, which is appropriate only because the ratio is exactly 1.

7. **Spec section 7 definition of done, status.**
   - A1-A4, B1-B3 numbers: present (A1 and A3 redefined; disclosed).
   - Plots: produced; spec's `dr5_03_En_Bn_vs_official.png` is split into `03` and `03b` (harmless); plot 1 is 9-bin (item 1).
   - Report and cross-check prompt: both exist (prompt timestamp 15:57, report 16:01; I did not check that the prompt's claims match the final report).
   - `tests/test_measure.py` passes and the full suite (300 or more tests): the report says the full suite was not re-run, and the only log (`pytest_after.log`) shows 1 failure. **This item is not met.** I could not verify the report's "68 passed in 34 s" (my two files: 11 passed in 19 s; `test_measure.py` not run by me).
   - "Nothing under /mnt/d/HI-gamma-correlation/ was modified": the report states the directory is read-only but contains no statement that it was verified unmodified. Missing.
   - Also missing from the spec: N/S patch results (spec 3.5), xi_x in the product, Step C, bin_slop <= 0.05 (all disclosed as not done or PI-decided).

8. **What the product test does catch (probed).** Swapping xi+/xi- arrays: caught (3.24 sigma on pair 1,1); a global 1.5 % amplitude error: caught; shuffled pairs, reversed order, missing m for one pair: caught by the test file. A global error below ~1 % is not caught (by design of A3), and the test does not compare against the source files, so a product built from the wrong run (e.g. the `b1.5_f32` files) would pass if it were within 0.1 sigma of the FITS. The report itself says the gates cannot tell float32 from float64 positions; the test inherits this. The provenance check (sha256 vs files) is not asserted.

9. **The tests skip silently.** `pytestmark skipif` on a missing product or release means a CI machine without the data reports success with zero checks. Here nothing was skipped (11 passed), but "tests pass" in the report should say they need the product and release files.

### MINOR

10. **Plot 6 title is cut at the right edge** ("... patch ~2 deg across)" runs to the border) and the figure is very wide (4 columns); the right-hand panels are small. The third column's y range (+-6) is fine. The caption in the report is accurate (3.97, 3.21 visible as the largest orange squares).

11. **Plot 2 top-right note** says "see RESUME" (internal reference in a figure) and sits in the empty middle of a 0.7-1.4 axis; the y axis spans 0.7-1.4 with all data on the two edges, so the plot conveys two numbers per panel. Caption is accurate. The "0.695 to 1.416" range quoted in the caption comes from inline numbers (listed in 5.9, good).

12. **Plot 4 compares unlike inputs on one axis** (B1: m-uncorrected ascii; B2: m-corrected product). The title says so ("different xi inputs ... neither gated"), so the m-treatment is consistent with the labels, but the left/middle panels share axes that invite reading B1 as a failure and B2 as success of the same integrator. The report's 4.3 says it. No m inconsistency found: Plots 1, 3, 3b use m-corrected ours (labelled), Plot 2 and Plot 6 raw (appropriate), the npz carries both.

13. **Plot 5 heatmap y tick labels** show only odd modes (1,3,...,19), the dotted line is at 6.5, not "n=6" as the title says. The chi2 curves have no y axis. Cosmetic.

14. **README typo**: "positions float64 float64". The README says weights agree "ours/theirs" nowhere, fine; its caveats match the report.

15. **Threads**: 20 used vs spec cap 16, disclosed in 1.2 (no sign-off recorded).

16. **Inline numbers**: whole-pair ratios, xi_x statistics, 22,748 s total and the 68-test count are listed as inline-computed, not stored. I did not recompute them (outside my read-only scope); they are the least traceable numbers in the report.

## Items checked and found consistent

- FULLVECTOR numbers quoted in the report (A3 0.0528/0.0438, chi2 0.0609/378, B2, B3, step D 127.516 vs 127.234, A_p ranges 1.13-1.32 / 1.14-1.24 / 1.95) match `FULLVECTOR.md`.
- Ours = theirs is described as code equivalence everywhere I looked; release TreeCorr 4.2.3 stated; "additional part" mentioned only as unspecified; only A3 described as amplitude-sensitive (~1 %).
- m-correction: product `xip = raw / ((1+m_i)(1+m_j))`, E_n/B_n from the corrected xi, plots labelled accordingly; raw arrays bitwise equal to the source run. No double or missing correction found.
- Plot units: theta in arcmin, theta*xi in 1e-4 arcmin, E_n in 1e-10, sigma from the correct COVMAT block (B_n offset 420), all as labelled.

## Verdict

Product and numbers look internally consistent. Fix before release: the Figure 1 caption/section mismatch (1), the test coverage gap for E_n/B_n and source files (2, 8), the conditional A4 pass (3), PASS labels on ungated checks (4), and the missing definition-of-done items (7: full suite, HI-gamma-unmodified statement).
