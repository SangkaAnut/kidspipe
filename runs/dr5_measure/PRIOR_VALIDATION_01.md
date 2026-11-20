# Prior validated work: `pipeline/01_lensing_validation`

Study only — no heavy job run. This folder **already does what our `runs/dr5_measure` plans to do**,
and it passes against the official KiDS-Legacy data vector. This note records what it contains, what
it settles, and what it changes about our plan.

Path: `/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/`
Report: `REPORT.md` (status COMPLETE, 30 Sep 2026), final PDF `outputs/report_01_lensing_validation_20260930_0711.pdf`

---

## 1. What it measured, and how well it agrees

| quantity | agreement with official release | source |
|---|---|---|
| ξ+ and ξ−, all 21 pairs, 9 bins (378 pts) | **Δχ² = 0.0607; max \|Δ\|/σ = 0.0511** (ξ+ 0.0510, ξ− 0.0437); median 0.0063 | `outputs/compare_results.json → fiducial.xi` |
| Band powers C_E, C_B, 8 bands (336 pts) | Δχ² = 0.0563; max \|Δ\|/σ_E = 0.0450, _B = 0.0751 | `… → fiducial.bandpowers` |
| COSEBIs E_n, B_n, all 21 pairs | max \|ΔE\|/σ = 0.046, max \|ΔB\|/σ = 0.060 | `outputs/null_tests.json → cosebis_vs_release` |
| B_n null, 6 modes | **p = 0.0648** (release 0.0635) | `null_tests.json → Bn_6modes` |
| B_n null, 20 modes | p = 0.0928 (release 0.0902) | `Bn_20modes` |

`fiducial.pass = True` in `compare_results.json`.

Its own B_n recipe was validated first by applying it to the *released* B_n and reproducing the
published p-values exactly (6-6: 0.037/0.0038; all pairs 0.090/0.064). That is the right way to
validate a null test, and it is a check our project has not done.

## 2. The estimator convention — this explains our "1.406 excess"

Both projects read the same catalogue and produce the same per-bin counts (task 00's
`checksum_report.json` has `N=7416371, sumw=82612776.94913222` for bin 1; ours is bit-identical,
and `tests/test_extracts.py` asserts the six N values against Wright+25 Table 1).

The difference is one line of API:

| project | auto-pair call | npairs (Bin1_Bin1) |
|---|---|---|
| 01_lensing_validation | `gg.process(cats[i])` | 1.209e12 |
| ours (`measure/xi.py`) | `gg.process(cat1, cat2)` with `cat2 = cat1` | 2.418e12 |

Measured directly (`scripts/dr5_pair_convention_test.py`):

```
process(cat)        npairs=8.759383e+06  weight=1.086257e+09
process(cat, cat)   npairs=1.751877e+07  weight=2.172514e+09
npairs ratio (cat)/(cat,cat) = 0.500000000000
xip max |diff|               = 7.795e-17
xip corr                     = 1.000000000000
```

So `process(cat)` counts **unordered** pairs, `process(cat, cat)` counts **ordered**, and the ξ
values are identical to machine precision.

**Consequence for our A1.** Our npairs are exactly 2× theirs, and their convention is the one
validated against the official data vector. I earlier compared our npairs to the official ascii
`npairs` column and concluded the official column was "normalised and non-integer". That reading
still stands on its own evidence (the official values really are not integers and no rescaling makes
them so), but it is now clear that **the raw-npairs comparison was the wrong instrument entirely**:
three different conventions are in play (ours ordered, theirs unordered, the official column
neither), so a ratio between any two of them measures bookkeeping, not agreement.

Numbers, so this is auditable rather than asserted:
- ours/official, Bin1_Bin1 = 1.4142; theirs/official = 0.7084.
- 0.7084 × 2 = 1.4168 ≈ 1.4142, i.e. the two projects differ by exactly the factor 2 and agree on
  the physics.
- Our `sum(weight)`/official = 1.0032–1.0042 (three pairs); **their `weight`/official = 0.5014 for
  Bin1_Bin1**, since `process(cat)` also halves the weight sum. Neither is a discrepancy.

The ξ comparisons are unaffected by any of this, and those are the ones that pass at 0.051σ.

## 3. Sign convention — already settled

From `outputs/test_treecorr_bruteforce.json` and `outputs/sign_convention.json`:

- The brute-force O(N²) estimator uses `x = East, y = North`, `γ = e1 + i e2`, and computes
  `ξ+ = ⟨Re(g'_a conj g'_b)⟩`, `ξ− = ⟨Re(g'_a g'_b)⟩` — the correct `gA * gB` form we also arrived at.
- **TreeCorr with `(g1,g2) = (e1,e2)` matches that frame only after `e2 → −e2`** (ξ+ to 1.9e-12,
  ξ− to 3.3e-12 on a synthetic field). On real data (3000 galaxies) the match is 3.7e-11.
- But the **production** run uses `g1=e1, g2=e2` with **no flip**, as Wright+25 do, and this is what
  matches the official ξ− to 0.044σ. The `e2flip` variant is off by 7.7σ in ξ− (and ξ+ barely moves,
  0.15σ), which is exactly the diagnostic pattern.
- HEALPix/NaMaster: **(q, u) = (−e1, +e2)**, with `C^{gE} > 0` for density(bin1)×shear(bin5) fixing
  the overall sign, which EE/BB alone cannot.

**So our handoff step 5 ("settle the shear sign convention on real data") is answered:**
`flip_g1=False, flip_g2=False` is correct, and the tie is broken by ξ− at 7.7σ, not by a toy field.

## 4. The m-correction — settled

`REPORT.md` and `code/config.py`: the released ξ± **are m-corrected**. Convention is
`ξ_corr = ξ_raw / [(1+m_i)(1+m_j)]` with `m_gold` per bin from Wright+25 Table 1
(−0.023, −0.016, −0.011, +0.020, +0.030, +0.045). Dropping the correction costs
Δχ² = 6.40 over 378 points (max 0.92σ) — small but real, and their fiducial includes it.

Its `config.py` also records that the band-power `theta_min/theta_max` in the release ini are the
*ramp centres* (2.568, 233.64), whereas the code needs the outer edges (2.0, 300.0); using the ini
values directly degrades C_E by up to 0.9σ. That kind of detail is worth not rediscovering.

## 5. bin_slop — measured on our machine, does NOT transfer as a speed lever

Their production run used `bin_slop=1.5` at 1.5–1.7e9 pairs/s. Our `scripts/dr5_binslop_test.py`
on the same data and TreeCorr 5.1.4 measured:

| bin_slop | time (10% sample) | speedup | npairs ratio | max abs Δξ+ / σ_own |
|---|---|---|---|---|
| 0.00 | 88.2 s | 1.0× | 1.0 | 0 |
| 0.05 | 91.0 s | 1.0× | 0.99999994 | 0.42 |
| 0.75 | 66.9 s | 1.3× | 0.99999895 | 2.11 |
| 1.50 | 49.4 s | 1.8× | 0.99999454 | 3.10 |
| 3.00 | 28.2 s | 3.1× | 0.99997504 | 4.02 |

So on this box `bin_slop=1.5` buys only **1.8×**, not 7×, and it shifts ξ+ by up to 3σ of the
shape-noise error. Their 12× better throughput must come from elsewhere (they cap at 8 threads, so
not thread count) — most likely the whole sky is denser in their run, or their timing reflects a
different TreeCorr build/I-O state. **Their npairs-per-second cannot be reproduced here**, so their
1.5e9 pairs/s is not a basis for our budget. Our measured 2.4–2.7e8 pairs/s at slop=0 stands, and
the ~49 h estimate stands.

The `bin_slop=1.5 → 0.75` convergence shift they report is 0.096σ *in the official covariance*, which
is a much weaker test than my per-bin σ_own — both can be true, and it means bin_slop=1.5 is probably
acceptable for the final data vector even though it visibly perturbs individual fine bins.

## 6. What this changes about our plan

1. **The production run is largely redundant.** All 21 ξ± pairs, all 21 COSEBIs E/B pairs and all
   band powers already exist in `outputs/xipm/fiducial/` and already pass against the official data
   vector at ≤0.05σ. Our A2, A3, B2, B3 answers can be read off their files instead of spending
   ~49 h.
2. **A1 should be redefined again.** The right statement is ξ-based (their 0.051σ), and any npairs
   statement must first pin the ordered/unordered convention. Comparing raw npairs to the official
   `npairs` column, as our first A1 did, cannot work.
3. **Our handoff steps 5 (sign) and the m-correction question are already answered** with evidence.
4. **Their fine ascii caveat matches our own finding.** `REPORT.md` item 2: rebinned, the released
   fine `cosmosis_xipm` files differ from the released xiP/xiM by an almost exactly separable
   per-bin amplitude (0.893, 0.913, 0.917, 0.932, 0.949, 0.971), giving Δχ² = 21.6. They flag the
   files as coming from a different (probably blinded) realisation and say a TODO is to confirm with
   A. Wright. **Our project is treating that same ascii as the reference for A1/A2.** It should not
   be treated as a peer-reviewed truth without that caveat.

## 7. Honest caveats on using this work

- It is a different project tree (`pipeline/`, not `cloning_kids_Kappa_pipeline`) with its own
  `REPORT.md` status; it is not in our test suite (287 tests) and our `docs/` does not reference it.
- Its `config.py` records a plausible TreeCorr version mismatch: the report says v4.2.3, the npz
  files say `treecorr_version = 5.1.4`. Worth noting but the ξ agreement is what matters.
- `run_all.log` shows the DRY-RUN diagnostics, the sign-convention table and the final compare all
  ran; I did not re-execute anything, in line with "study, no heavy job".
- Their per-pair npz `npairs` are unordered; do not mix them with ours without the factor 2.

## 8. Files worth reading first

1. `REPORT.md` — the summary table and the caveats.
2. `code/config.py` — every external number with its citation, m values, bin_slop, variants.
3. `code/run_compare.py` — how the pass/fail is computed (Δχ² with the matching covariance sub-block).
4. `code/xipm.py` — 62 lines; the whole estimator, including `process(cats[i])` vs `process(a,b)`.
5. `tests/test_treecorr_bruteforce.py` + `outputs/test_treecorr_bruteforce.json` — the frame test.
6. `outputs/null_tests.json` — COSEBIs and B_n, with the release reproduction.
