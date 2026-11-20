# VERIFY (synthesis of INTEGRITY, RECOMPUTE, A1CONST_BN, REFUTER)

Method: read the four records plus their structured summaries; independently checked the only cross-record claims that were
disputable (file locations/labels, manifest git_head, completion of the chain) with ls/grep/head on b1.5/ and RESUME.md.
I did not re-run any measurement. Numbers below are taken from the records, except where marked "checked".

## Overall: CONFIRMED_WITH_CAVEATS

No verifier found a numeric discrepancy. The three claimed measurement results (A1 boxcar 21/21, A3 21/21 max 0.053, A1 const-A_p
pte 0.889/0.496) were reproduced. The caveats concern the power and independence of the gates, not the data.

## Check table

| check | by | result |
|---|---|---|
| 21 files load, finite, 1000 bins, bin_slop 1.5, TreeCorr 5.1.4, unordered, float64 radec, nobj = extract | INTEGRITY | pass |
| mtimes after extract, no tmp files, manifest sums consistent, 4 pre-OOM crosses intact | INTEGRITY | pass |
| Ours vs theirs: npairs bitwise 21/21, max abs dxi ~1e-11 (<=3.8e-11) | INTEGRITY, REFUTER | pass (float32 e/w in theirs explains 1e-11) |
| A3 and A1 boxcar recomputed from FITS with no evaluator code, 21 pairs | RECOMPUTE | identical to eval.json (diff 0.0); max A3 0.0528 Bin5_Bin6 xi+; max boxcar 0.787 |
| Pair-to-row mapping sensitivity (wrong rows 1.6-20 sigma, swap xi+/- 3-7 sigma, no m-corr 0.12-0.92) | RECOMPUTE | gate is discriminating; Bin1_Bin1 needs the m-correction to pass |
| A1 const-A_p: chi2 12.72/19.40, pte 0.889/0.496 | A1CONST_BN, REFUTER | reproduced |
| Full-covariance bootstrap GLS pte 0.574 / 0.653; dropping 2 weak pairs 0.815 / 0.392 | A1CONST_BN | still pass |
| B_n null chi2 vs official (6 pairs shown), E_n max diff | A1CONST_BN | match EVAL.md; Bin6_Bin6 B_n 6-mode pte 0.004 is also in official data |
| Gate power: A3 fails xi x1.01 on 2/21, x1.02 on 13/21, x1.05 on 21/21; passes x1.005 | REFUTER | A3 resolves ~1% amplitude |
| float32 / float64 mixing, OOM resume | REFUTER, INTEGRITY | none found; atomic writes; Bin1_Bin6 re-measured fresh |
| Chain completion (checked) | me | run_f64_cross.log tail: ALL_EVAL_EXIT=0, CHAIN_DONE; EVAL.md/eval.json written after manifest |

## Where records disagree / who is right

No record contradicts another. Two things looked like possible disagreements and are not:
- INTEGRITY "manifest git_head null" (checked: b1.5/manifest.json line 9 is null). True, cosmetic provenance gap.
- RECOMPUTE flagged Bin1_Bin1 xi- A_p = 1.95; A1CONST_BN shows its sd is 0.89, so it carries almost no weight in the const test.
  Consistent, not a fault.

## Discrepancies

| # | item | severity |
|---|---|---|
| 1 | EVAL.md title (and the title string in do_eval, scripts/dr5_measure_xi_unordered.py line ~348) says "Unordered auto-pair evaluation" but it covers all 21 pairs (checked) | cosmetic |
| 2 | run_f64_all.log name is stale: holds autos + only 4 crosses; remaining 11 are in run_f64_cross.log | cosmetic |
| 3 | RESUME.md gate-table row for autos still says "15 cross pairs running" (checked, line 40); also "A1 const-A_p over 6 pairs (partial set)" is superseded by the 21-pair result | cosmetic |
| 4 | manifest git_head null | cosmetic |
| 5 | "ours == theirs" is not independent evidence: same TreeCorr 5.1.4 and same extract; official release used TreeCorr 4.2.3; vs_theirs gates run through the same evaluate_pair | caveat |
| 6 | A1 boxcar is blind to a common amplitude error (xi x1.10 passes 21/21) and to an xi- sign flip; const-A_p ignores a common scale. Only A3 has power (~1%) | caveat |
| 7 | Gates do not detect float32 vs float64 positions (float32 autos also pass, A3 0.063); only ours-vs-theirs does | caveat |
| 8 | B_n "matches official" and E_n have no numeric pass/fail threshold in code (chi2 differences up to ~0.4) | caveat |
| 9 | A3 0.1 and A1 1.03 thresholds were set after seeing results (justified by null sims); m_gold is the project's Table 1 list, not read from the release | caveat |
| 10 | A_p chi2 uses precomputed sd_frac from a1_salvage.json (bootstrap on "theirs"); fresh 3000-rep bootstrap reproduces it (ratio 1.005/1.012). Test ignores cross-pair correlation; full-covariance version also passes | caveat |
| 11 | e2 sign convention not validated on the full catalogue: 150k subsample, flipping e2 changes xi- by up to 0.036 vs scale 0.021, xi+ unchanged. A3 against official (which would flag a flip) passes at 0.053, which constrains this indirectly | caveat |

No blocking discrepancies.

## Corrections needed (not applied)

EVAL.md (b1.5/EVAL.md):
1. Retitle to "Unordered evaluation, all 21 pairs" (and fix the title string in do_eval so a regeneration does it too).
2. State explicitly that B_n/E_n are compared to official by eye, not by a threshold.
3. Add a one-line note that ours-vs-theirs is a code-equivalence check, not an independent validation.

RESUME.md:
4. Gate-table autos row: replace "15 cross pairs running" with the completed 21-pair result (A3 max 0.053 Bin5_Bin6 xi+; boxcar max 0.787; const-A_p pte 0.889/0.496) and drop "partial set".
5. Note run_f64_all.log covers autos + 4 crosses only; the other 11 crosses are in run_f64_cross.log.
6. Record that manifest git_head is null (no commit pinned).
7. Add the gate-power note: only A3 is amplitude-sensitive (~1%).

## Open questions for the PI

1. Should the e2 sign convention be checked on the full catalogue (or against a released-data subsample cross-check)? Flipping e2 would change xi- only, and A1 boxcar cannot see it.
2. Is an independent TreeCorr 4.2.3 run on one small pair (subsample) wanted to bound the 4.2.3 vs 5.1.4 difference?
3. Should B_n/E_n get explicit numeric thresholds (e.g. |chi2 - official| < 1) before they are cited as gates?
4. Should the post-hoc thresholds (A3 0.1, A1 1.03) be frozen with the null-sim justification written into RESUME?
5. Pin a git commit for the runner so provenance is reproducible.

VERDICT: CONFIRMED_WITH_CAVEATS
Discrepancies: #1-#4 cosmetic (labels/provenance), #5-#11 caveats (gate power and independence); none blocking.
