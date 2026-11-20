# Critic synthesis of the angle_slop study verification (2026-10-30)

Inputs: the six verifier records in this directory (RECOMPUTE, EXACTNESS, SOUTHPATCH, PROVENANCE, REFUTER, COST_TESTS), ANGLE_SLOP.md and the JSON files it cites.
Spot checks I did myself (reading existing outputs, no new TreeCorr runs):
1. Recompute's `recompute_numbers.json` exact block gives max|A3 xi+| 0.07479, max|A3 xi-| 0.15627 (150-300' bin), max|dE| over 6 modes 0.2380, over 20 modes 0.4844. These match C2 and `exact_bin6_eval_b0_a0.json`.
2. `fullpair_vs_exact.json` 1.5:0.1 gives xi+ 0.0507, xi- 0.0930, E6 0.1944, E20 0.4367. These match C3.
3. `verify/southpatch/analysis.json` 1.5:0.1 gives 0.0646 / 0.1018 / 0.1315 / 0.2416. These match the Southpatch record.
4. Cost_tests' core-second ratio, 1857*8/(6995*14) = 0.152, is correct arithmetic. Whether it is meaningful depends on the 8-thread run being uncontended, which is not known.
Not independently re-run: any TreeCorr measurement, the T_- kernel, the TreeCorr 4.2.3 numerics.

## Verdicts

| claim | verdict |
|---|---|
| C1 (0,0) = brute force and same cost as (0,default) | PARTIAL (accuracy confirmed, cost equality only on patches) |
| C2 exact vs official A3/E/B numbers | CONFIRMED |
| C3 (1.5,0.1) vs exact and vs official | CONFIRMED |
| C4 official behaves like bin_slop ~1.5, xi- failure is the official approximation | PARTIAL (first half supported, second half is inference) |
| C5 patch forecasts under-predict by 2-2.5x; mid-cost not established | PARTIAL (direction and second half confirmed, factor overstated as a general number) |
| C6 production cost table | PARTIAL (arithmetic exact, inputs shaky) |

### C1: PARTIAL
- Accuracy: Exactness finds (0,0) vs TreeCorr `brute=True` at 3.5e-17 (xi+) and 4.9e-17 (xi-), with identical npairs.
- Against an independent spherical numpy brute force the residual is 1.6e-9 / 1.4e-9 (30k galaxies), also with identical npairs. So "~1e-14 vs brute" holds only against TreeCorr's own brute force. The 1e-9 residual is a small TreeCorr projection effect and is scientifically negligible.
- Cost on patches: (0,0) vs (0,0.1) is 0.996 on patch A and 0.944 on patch S (uncontended). The southern patch gives 756 s vs 761 s.
- Cost on the full pair: exact took 6995 s and the pilot (0,default) took 5905 s, so exact was 1.18x slower (median per chunk 1.21). The exact run was contended and the pilot's load is unknown, so the 1.18 cannot be attributed to angle_slop=0 or to load. "Costs the same" is therefore supported at about 0-6% on patches and OPEN at full-pair scale.
- Binning caveat (Exactness): the driver uses TreeCorr's default Euclidean chord metric, so bins are in chord units. This is about a 3e-4 relative offset at 300'. It is an existing production property, applies equally to the official-like run, and is not a regression.
- The pytest fix (tests/test_measure.py:818-841) swapped the reference to TreeCorr brute. It is legitimate but weaker: the remaining independent check only bounds xi+ at atol 5e-5 and does not check xi-. The suite passes (57 in test_measure, 356 in the fast suite).

### C2: CONFIRMED
- Recompute used its own FITS parsing and rebin, and reproduced 0.0748 / 0.1563 / E6 0.238 / E20 0.484, B_n chi2 19.47/6.
- The pilot agrees to 3e-5 / 6e-4 / 4e-4 / 1e-3 sigma.
- The covariance block and units were checked. The m correction is applied identically to all three variants.
- The xi- number is robust to the rebin weighting for the exact run: 0.1563 with npairs, 0.1560 with weight, 0.1563 with official npw.
- Gaps: the T_- kernel was verified only end-to-end (the external run reproduces official E_n to 0.04 sigma), and m_6 = 0.045 was taken from the repo.
- Note that B_n chi2 is essentially the same for exact (19.47), slop 1.5 (19.14) and official (19.24). The B-mode test cannot discriminate between the variants.

### C3: CONFIRMED
- Exact minus slop 1.5 is 0.0511 / 0.0929 / 0.1944 / 0.4367. Slop 1.5 vs official is 0.0456 / 0.0888 / 0.071.
- The slop-1.5 xi- number depends on the rebin weighting: 0.0888 (npairs), 0.0926 (official npw), 0.0958 (weight). All stay below 0.1, but the margin is thin and the gate verdict for 1.5:0.1 flips at about 0.10.
- Refuter's decomposition: the exact-minus-official E_n curve is explained by the exact-minus-slop15 curve with correlation 0.979, slope 1.04, and residual rms 0.034 against 0.326 (R^2 0.989).

### C4: PARTIAL
- Supported: Provenance read Wright+25 (arXiv:2503.19441) Sect 2.5.1 footnote 1 via alphaXiv: "TreeCorr 4.2.3, binslop = 1.5", 1000 log bins 2'-300'. Joachimi+21 gives the same for KiDS-1000. The "not established" statements in ANGLE_SLOP.md and PILOT_BIN6/VERIFY.md are stale on this point. Refuter called the citation second-hand because the paper is not on disk. Provenance queried the paper text directly through a tool, so I treat it as documented. I did not re-query it myself.
- Rebin convention, COSEBI integration, weights/m, ordering bookkeeping and TreeCorr 4 vs 5 are each ruled out as the main cause, by Refuter's argument that slop15 and "theirs" use the same code and match official.
- Not supported:
  - "Not an error in our measurement" is inference. Matching slop 1.5 shows mimicry, not uniqueness; only 1.5 was run at full pair.
  - What is documented is only bin_slop 1.5. Ordered/unordered, metric, angle_slop (absent in 4.2.3, added in 5.0), patches and the continuous official npairs are undocumented. "Theirs" is the earlier agent's own TreeCorr 5.1.4 run with default angle_slop 0.1, not an official file. Its npairs are integer while the official ones are continuous, which is unexplained.
  - A residual 0.04-0.06 sigma in the 150-300' xi- bin remains for slop 1.5 (0.063) and theirs (0.044).
  - The bin_slop 1.5 approximation may be biased rather than merely different. The 4-subsample test shows a positive xi- shift in 4/4 subsamples at t~2.3-2.4, with overlapping subsamples, so it is suggestive and not proven. The exact run is the less approximate one, but "official is biased by 0.16 sigma in xi-" has not been shown.
- A 4.2.3 vs 5.1.4 numerical difference is not excluded. The statement can stand as "the official release is documented as bin_slop 1.5 and our slop-1.5 run reproduces it to <=0.09 sigma; our exact run differs from it by 0.16 sigma in xi-".

### C5: PARTIAL
- Direction confirmed: the full-pair E_n deviation of 1.5:0.1 (0.194 / 0.437) exceeds the study's patch values (0.078 / 0.191 worst).
- Southpatch, a new independent 571,938-galaxy patch at ra 10-24 / dec -36..-27, gives 0.132 / 0.242 for E6/E20. This exceeds every study patch (A 0.108, B 0.094, S 0.191 at E20) and reduces the full-pair/patch ratio to 1.5 (E6) and 1.8 (E20). xi+ and xi- on the full pair are at or below the patch values (0.8-0.9 of South).
- So "2-2.5x" is a ratio against specific small patches and should be stated as 1.5-2.5x for E_n, with no under-prediction for xi+/xi-.
- The E_n deviation is smooth and coherent in n (no sign changes, lag-1 autocorrelation 0.98, correlation 0.93 with the full-pair profile). Its amplitude and even its sign differ between patches (A and B have opposite signs in the first modes). Patch forecasts of E_n are therefore unreliable in sign and amplitude.
- Likely cause (Southpatch): the patch-to-sigma_official conversion mixes noise scaling with a signal-proportional part and assumes independent fine-bin noise. This is not proven.
- Mid-cost settings are not established. Confirmed: no full-pair run exists for 0.5:0.1, 1:0.1, or any 1.5 with angle_slop < 0.1.

### C6: PARTIAL
- Arithmetic: all 10 table entries recompute exactly as 49 h x patch ratio. The 49 h base traces to CHUNKING.md (4.29e13 binned pairs, 2.41e8 pairs/s from one Bin1 chunk-pair, 9.763e14 total pairs reproduced). Starred entries use the S-patch ratio, which differs from A (1.5:0.1 is 0.316 on S vs 0.229 on A).
- The 49 h base is an extrapolation from one chunk-pair, not a measurement (RESUME.md "Measured" is wrong). Cost_tests' best estimate for (0,0) is about 45 h at 16 threads (range 37-60) and about 38 h at 20 threads (31-55). Bin6_Bin6 is only 3.6% of pairs, so about 96% of the work remains.
- 1.5:0.1: the table says 11.2 h and PILOT_BIN6.md:70 says about 15 h. The full-pair slop15 run (1857 s at 8 threads) gives 7.5-8.8 h in core-seconds. The 20% subsample gave only a 2.1x speed-up, against 3.2x at the full pair, so the speed-up depends on size. Take 8-15 h as the honest range for 1.5:0.1.
- Unflagged error in the recommendation: setting 0.5:0 is identical to (0,0) (max |dxi| 3e-17, same npairs) and not cheaper (827 s vs 756 s on the South patch). angle_slop=0 disables the bin_slop saving. The ANGLE_SLOP.md recommendation "a full Bin6_Bin6 run at 0.5:0, ~1 h" is a dead end. The study's own S-grid already shows 33.9 s vs 33.5 s. Any mid-cost candidate needs angle_slop > 0.
- Cost_tests found 11 record overstatements or contradictions (list in COST_TESTS.md section (c)): for example RESUME.md:66 "bin_slop is not a lever", the bug-13 text "exact for xi+ only", and RESUME.md:4 vs :113.

## Disagreements and how they resolve

| issue | resolution |
|---|---|
| Is official bin_slop 1.5 established? ANGLE_SLOP.md and Refuter say no; Provenance says documented in Wright+25 footnote 1. | Documented (Provenance read the paper text). The other settings are undocumented. |
| Same cost? Patches 0.94-1.0 vs full pair 1.18x. | OPEN. The exact run was contended and the pilot's load is unknown. A clean paired timing of one chunk pair at (0,0) and (0,0.1) would settle it. |
| 1.5:0.1 hours: 11.2 h (patch A), about 15 h (PILOT note), 7.5-8.8 h (full-pair core-seconds). | OPEN, 8-15 h. The full-pair number is unverified for contention. |
| E_n under-prediction factor: 2-2.5x (study) vs 1.5-1.8x (Southpatch). | Both are one-realisation ratios. State as 1.5-2.5x. |
| Is the xi- gap bin_slop approximation or noise? | Recompute and Refuter show the E_n gap is explained by exact-minus-slop15. The bias-versus-noise question stays open (t~2.3, single realisation). |

## Still missing

- No full-pair run at any mid-cost setting (0.5:0.1, 1:0.1, 1.5 with angle_slop 0.001-0.003). The real mid-cost candidate 0.5:0.1 was not run on any new data.
- Only Bin6_Bin6 (and patch pair 1-6) tested. Cross pairs, and especially low-redshift auto pairs where the angular clustering signal differs, are untested at full scale.
- No signal-only check at full-pair scale. No second full-survey realisation exists, so the full-pair E_n deviation is a single realisation.
- No paired clean timing exists for (0,0) vs (0,0.1) at full-pair scale. Chunk-pair timings for queue4 were not done. 20-thread scaling on the shared 22-core machine is an assumption.
- The TreeCorr 4.2.3 numerics were never run; 4.2.3 is not installed and cannot be installed under the rules. The ordered/unordered, patch and metric choices of the official run are undocumented.
- The T_- kernel has not been verified by independent quadrature. m_6 is unchecked against the release.
- The 0.04-0.06 sigma xi- residual in the 150-300' bin, even for slop 1.5 and theirs, is unexplained.
- Records to correct (all from COST_TESTS): RESUME.md lines 4, 38, 60-66, 91-96, 106-113; PILOT_BIN6.md lines 5, 60, 70; scripts/dr5_measure_xi_chunked.py:152-155 help text; ANGLE_SLOP.md headline 1, the table "cost the same" wording, and the recommendation text on 0.5:0.

## Recommendation

Choices and hours (all hours are estimates, full 21 pairs, 16 threads unless noted; Bin6_Bin6 about 3.6% is already done):

| setting | hours | evidence | what it means for A3 vs official |
|---|---|---|---|
| (0,0) exact, or equivalently (0,0.1) | about 45 h (37-60); about 38 h at 20 threads (31-55). ANGLE_SLOP.md says 49 h. | Equal to brute force to 5e-17 (TreeCorr brute) and 1.6e-9 (independent brute). Full-pair measured on Bin6_Bin6. (0,0.1) differs by <=1e-3 sigma on Bin6_Bin6; cost 1.18x unresolved. | On Bin6_Bin6: xi+ 0.075 pass, xi- 0.156 FAIL, E20 0.48. Expect similar failures on other pairs wherever the official bin_slop effect matters. |
| (1.5,0.1) mimic | 8-15 h (table 11.2) | Full-pair measured on Bin6_Bin6: 0.046 / 0.089 vs official, E 0.071. Differs from exact by E20 0.44, xi- 0.09. | A3 would pass, with xi- margin of about 0.01 on 6-6 (0.089 to 0.096 depending on weighting). This reproduces the official approximation, so it is a reproduction test, not an accuracy test. |
| Both: exact for the science run, plus (1.5,0.1) once as the A3 cross-check | about 45 h + 8-15 h | The two together separate "our pipeline vs official processing" from "our measurement vs truth". Not yet tested beyond Bin6_Bin6. | A3 passes for the mimic run; the exact run is also reported against the official as a stated bin_slop difference. |
| Mid-cost: 0.5:0.1 (~28 h), 1:0.1 (~18 h), 1.5 with angle_slop 0.001-0.003 (~26-42 h) | Hours from patch ratios only | Patch forecasts for E_n failed to bound the full pair (Southpatch exceeded all study patches). 1:0.1 and 0.5:0.1 E20 forecasts 0.26 / 0.25 are already above 0.1. 1.5 with 0.001-0.002 look acceptable only on patches. | Unjustified without a full-pair run (about 0.3-1.5 h each for Bin6_Bin6). Not recommended as production settings on current evidence. |
| 0.5:0 | same as (0,0) | Identical to (0,0), no saving. | Not a candidate. |

Only (0,0)/(0,0.1) and (1.5,0.1) have full-pair accuracy evidence. Of these, only (0,0) has an accuracy argument independent of the official release.

What the A3 gate means then:
- A3 (9-bin max|Delta|/sigma_official < 0.1) compares us with the official release, and on the evidence it measures the bin_slop difference more than pipeline correctness. The exact run fails at xi- 0.156, but slop 1.5 through our code (0.089) and the external reproduction (0.044) agree with official to under 0.1 sigma. So apart from the TreeCorr setting, the pipeline reproduces the official numbers on Bin6_Bin6.
- If the exact measurement is believed more accurate, the PI has three options:
  (a) Keep A3 vs official as the gate: then production must be the (1.5,0.1) mimic (8-15 h), with margin about 0.01 on xi- for 6-6, unknown for other pairs. Exact then cannot be the accepted result.
  (b) Re-define the gate: compare the (1.5,0.1) run against official for pipeline validation (A3 as written), and accept the exact run as the science product with its difference from official stated and bounded by the slop-1.5-vs-exact difference (up to about 0.09 sigma xi-, 0.44 sigma E20 on 6-6). Cost is about 45 h plus 8-15 h.
  (c) Keep exact only and relax A3 to a documented threshold (about 0.16 xi-, about 0.5 for E20), justified by the Wright+25 bin_slop 1.5 footnote. This leaves no separate evidence that the pipeline is otherwise right, apart from the slop-1.5 and theirs numbers already in hand.
- Evidence the PI should weigh: the exact-vs-official gap is not noise-like. It is smooth and coherent in E_n (R^2 0.99 with exact-minus-slop15), and the B_n null chi2 is identical for all variants. The sign of the xi- bias (4/4 subsamples at t~2.3) is suggestive that slop 1.5 biases xi- high, but this is not established, so "official is biased" should not yet be written into the record.
