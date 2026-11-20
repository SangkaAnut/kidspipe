# KiDS-DR5 / KiDS-Legacy: the official selection, verbatim from the papers

Written to answer step 1 of `docs/HANDOFF_PROMPT_DR5_CONTINUATION.md`. **Result: the premise of
the handoff is wrong.** The catalogue we hold is already the complete, final official sample.
There is no missing selection cut. Evidence below, then the consequence.

Papers, arXiv ids verified this session by fetching them (not taken from the handoff):

| paper | arXiv | journal | role |
|---|---|---|---|
| Wright, Kuijken, Hildebrandt, Radovich, Bilicki et al. | **2503.19439** | A&A **686**, A170 (2024) | KiDS-DR5 data release |
| Wright, Stölzner, Asgari et al. | **2503.19441** (v2) | A&A **703**, A158 (2025) | KiDS-Legacy cosmic shear |

The handoff said "look up the DR5 paper's arXiv id yourself" — done. The handoff's own citation of
`arXiv:2503.19441` for the cosmic-shear paper is correct; `2503.19439` is the DR5 release paper.

---

## 1. What the cosmic-shear paper says the sample *is* (the decisive passage)

Wright et al. 2025 (2503.19441), **Table 1**, "Properties of the six KiDS-Legacy tomographic bins":

```
 Tomo.  photo-z            Ngold     neff,gold  σε,gold   ⟨z⟩est   δz        mgold
  Bin   Selection                   [arcmin−2]          [×10−2]          [×10−2]
   1    0.10 < zB ≤ 0.42   7 416 371   1.77      0.28   0.335   −2.6±0.2±1.0  −2.3±0.6
   2    0.42 < zB ≤ 0.58   7 359 911   1.65      0.27   0.477    1.4±0.1±1.0  −1.6±0.6
   3    0.58 < zB ≤ 0.71   6 799 681   1.50      0.29   0.587   −0.2±0.2±1.0  −1.1±0.7
   4    0.71 < zB ≤ 0.90   6 880 432   1.46      0.26   0.789    0.8±0.1±1.0   2.0±0.7
   5    0.90 < zB ≤ 1.14   6 477 538   1.35      0.28   0.940   −1.1±0.2±1.0   3.0±0.8
   6    1.14 < zB ≤ 2.00   5 960 461   1.07      0.30   1.224   −5.4±0.4±1.0   4.5±0.9
  1–6   0.10 < zB ≤ 2.00  40 894 394   8.79      0.28   0.680      –            –
```

> "All values are computed for the final gold-selected sample of sources, and with the appropriate
> shape-measurement and gold weights applied. Where relevant values are computed using an
> **effective area of 967.4 deg²**."

**The seven `Ngold` values in that table are, digit for digit, the row counts of the file we hold**
(40,894,394 total; per-bin 7,416,371 / 7,359,911 / 6,799,681 / 6,880,432 / 6,477,538 / 5,960,461 —
see `runs/dr5_measure/paper_gate.json`). So the file we hold *is* the published KiDS-Legacy gold
sample. Any additional cut would break that identity.

Also: "The fiducial KiDS-Legacy sample consists of 41 409 179 sources drawn from 967.4 deg² of sky,
with an effective number density of neff = 8.87 arcmin⁻²" (Sect. 9, Summary) — i.e. 41.4M before the
`0.1 < zB ≤ 2.0` limit, 40.9M after. Our file is the *after* number, the fiducial sample.

## 2. The selection was applied before the file was written (2503.19439, Sect. 7.2, Table 14)

Table 14, "Statistics of the additional selections used for KiDS-Legacy sample definition" —
successive cuts, `N^k_s,run` = sources remaining:

| Selection | Eq. | remaining |
|---|---|---|
| MASK | 29 | 100 744 685 |
| FLAG GAAP | 30 | 100 659 493 |
| ASTEROIDS | 31 | 99 708 082 |
| UNMEASURED | 32 | 79 511 178 |
| BLENDING | 33 | 73 950 871 |
| FITCLASS | 34 | 63 574 740 |
| BINARY | 35 | 63 560 519 |
| MAG AUTO | 37 | 62 809 455 |
| RESOLUTION | 38 | 50 150 788 |
| WEIGHT | 39 | **43 205 363** |
| Astrometric masking | – | 41 409 179 |
| 0.1 < zB ≤ 2.0 | – | **40 950 607** |

The cut definitions (verbatim, Sect. 7.2):

- **Eq. 29** `MASK ≤ 1` — initial mask.
- **Eq. 30** `Nflag = Σ_x Flag_GAAP,x == 0` over `x ∈ ugri1i2ZYJHKs` (all ten GAaP bands succeed).
- **Eq. 31** `(g − r ≤ 1.5) or (i1 − r ≤ 1.5) or (i2 − r ≤ 1.5)` — asteroid/transient rejection.
- **Eq. 32** `(PSF_Q11 ≠ 0.0) and (PSF_Q22 ≠ 0.0)` — lensfit produced no shape.
- **Eq. 33** `contamination_radius > 4.25 pix`.
- **Eq. 34** `fitclass ∉ {−10, −7, −4, −3, −1, 1, 2}`.
- **Eq. 35** `log10(2 R_s) ≥ (24.9 − r_GAaP)/3.5`   **Eq. 36** `r_GAaP > 21.5`.
- **Eq. 37** `MAG_AUTO > 20.0`.
- **Eq. 38** `R < 0.9`, with `R ≡ r²_PSF/(r²_PSF + r²_ab)`.
- **Eq. 39** `AlphaRecalC_weight > 0`.
- then astrometric masking, then `0.1 < zB ≤ 2.0`.

**None of the columns in these equations exist in our file.** The readme lists 51 columns; there is
no `MASK`, no `GAAP_Flag_*`, no `contamination_radius`, no `fitclass`, no `autocal_scalelength_pixels`
in the sense needed, and `R` requires `re`, `r_PSF` and `q` per source. The cuts are unapplicable by
construction — because they were already applied upstream. This is exactly why the previous
session's per-column scan found every cut keeping "~100%": the columns present are the *post*-cut
quantities (`e`, `weight`, `TOMOBIN`, `Z_B`).

## 3. Footprint / effective area (2503.19439, Sect. 6.4–6.5, Table 12)

Fiducial mask definition, **Eq. 14**: `MASK ≤ 1`. Mosaic-mask areas, **Table 12** (6″ masks):

| Selection Name | Bits | North | South | Total |
|---|---|---|---|---|
| WCS | 14 | 681.098 | 649.883 | 1 330.981 |
| WCS+theli | 1,2,3,14 | 583.059 | 562.348 | 1 145.407 |
| WCS+AW+theli | 1,2,3,4…8,14 | 544.850 | 529.992 | 1 074.842 |
| **WCS+THELI+AW+NIR (fiducial)** | **1–14** | **517.342** | **496.689** | **1 014.031** |
| WCS+TH.Cons.+AW+NIR | 0–14 | 468.845 | 452.350 | 921.195 |

> "After removal of area that does not have overlap with the VISTA ZYJHKs-bands, the survey area is
> further reduced to the 'fiducial' survey area of **1014.0 deg²** ('WCS+THELI+AW+NIR', bits {1-14})."

So **1014.0 deg² is the DR5 *imaging* footprint**, not the cosmic-shear footprint. The cosmic shear
analysis then applies a **further astrometric mask** (Sect. 4.3 / Appendix E of 2503.19441):

> "For our first statistic … we chose to mask all areas of the survey where our first statistic
> exceeds the minimum PSF FWHM: that is, where δr_int > 0″.6. This corresponds to approximately
> 0.7 deg² of removed data over the full KiDS-Legacy footprint."
> "…we chose to investigate … the impact of various masks (defined by thresholds in δr_exp) …
> our fiducial analysis [uses] δr_exp ∈ [0″.20, 0″.25]."
> "The masking of systematic astrometric residuals causes a further down-selection of 4.16% of the
> survey sample via on-sky masks, leading to a final KiDS-Legacy sample that consists of 41 409 179
> sources drawn from **967.4 deg²**."

**1014.0 → 967.4 deg².** The handoff's gate ("that area must come out near 1014 deg²") is therefore
the wrong target by construction: it compares a pre-cosmic-shear area against post-cut per-bin
number densities.

## 4. Weight

`weight` in our file = "Final source weight, including shear and gold weights" (readme). The paper's
`neff,gold` uses exactly those weights, and the FITS `NGAL_*` header values are those densities
rescaled from 967.4 deg² to the 1014.03 deg² mosaic area — verified to four decimals in all six bins
(`paper_gate.json`). No re-derivation is needed or possible.

## 5. De-duplication across overlapping tiles

The catalogue is a **mosaic** catalogue ("Mosaic catalogues (the KiDS-Legacy catalogue for
weak-lensing science)", Sect. 8), built on a single predefined WCS with the WCS tiling cuts
(`MASK = 16384`) knitting overlapping tiles "without source duplication". So duplicate sources are
already removed upstream. Residual close pairs are physically distinct objects: measured 1.23% of
bin-1 galaxies have a neighbour within 3″ (max multiplicity 6), median separation 2.6″ — far too
numerous and too widely separated to be tile duplicates, and ~1% cannot produce a 40% pair excess.

---

## 6. Consequence: the selection hypothesis is dead; the pair count is not a raw count

Applying the paper's selection is impossible (columns absent) and unnecessary (counts already match).
Rather than stop at that, I tested what the 1.406 factor actually is
(`runs/dr5_measure/factor_origin.log`, `paper_gate.json`):

1. **`sum(weight)` agrees with the official ascii to 0.3%** (ratio 1.00324 for Bin2_Bin2, and flat in
   separation: 1.0034 → 1.0031 from 2′ to 300′, `factor_origin.log`). Since `weight` carries the full
   per-galaxy selection+shear information, a **different galaxy sample would change this number**.
   It does not change. So the sample is the same sample.
2. **A sample/footprint difference is arithmetically excluded.** Pairs scale as N², so an excess of
   1.406 in npairs would require `sum(weight)` to be high by √1.406 = 1.1858. It is high by 1.0032.
   Discrepancy 18%.
3. **The official `npairs` column is not an integer pair count.** Our TreeCorr `npairs` are exact
   integers; the official values are continuous (mean |frac − 0.5| = 0.250, i.e. uniformly
   distributed fractional parts, and no constant rescaling makes them integral — best-fit residual
   27%). The column has been divided by a normalisation (area, or n̄², or pairs-of-pairs).
4. **`npairs_weighted` is used by the official modelling, not `npairs`.** `KiDS_Legacy_xipm_pipeline.ini`
   line 57: `column_npair_name = npairs_weighted`.
5. **The excess is flat and universal**: measured independently on a fresh 10% subsample with
   `bin_slop=0` (`runs/dr5_measure/factor_origin.log`): Bin1_Bin1 npairs ratio **1.41515**,
   Bin2_Bin2 **1.40641**, Bin1_Bin2 **1.41159**. An auto pair and a cross pair, and the same
   factor within 0.6%. A footprint or galaxy-sample difference cannot do that: removing sky
   scales an auto pair and a cross pair differently only through edge effects, but more
   importantly (point 2) it would have to show up in `sum(weight)`, which is flat at 1.003.

   The same run gives `sum(weight)` ratios 1.00421 / 1.00325 / 1.00371 — all three pairs agree
   with the official ascii to 0.4%. **That is the check worth having**: `weight` encodes every
   per-galaxy selection and shear property, so it is the most sensitive available probe of whether
   we are measuring the same galaxies. We are.

**Therefore A1 as specified cannot pass, and its failure is not a selection error.** It compares a raw
integer pair count to a normalised column. The correct comparison is `sum(weight)`, which passes at
1.003 ± 0.004.

## 7. What is *not* yet explained, stated plainly

**RESOLVED 2026-10-29** (see `sigma_e/SIGMA_E.md`, verified in `sigma_e/VERIFY.md`): Joachimi+21 Eq. C.9 with w^2 weighting, per component, divided by (1+m_gold), reproduces SIG_E_i to <0.07 % in all six bins. The text below is the original (superseded) statement.

The per-bin `SIG_E_i` and `NGAL_i` are consistent with the paper (neff ratio 0.996–1.000 against the
paper's `neff,gold`; see `paper_gate.json`), but my raw `sigma_e` from `w` and `e` comes out 0.4–7%
*above* `SIG_E_i` (bin 1: 0.2760 vs 0.2772, ratio 0.996; bins 4–6: 0.260–0.322 vs 0.262–0.300,
ratios 1.061–1.072). The paper states "σ_ε values include the appropriate correction for
multiplicative shear biases (i.e. the m_gold column)". Applying `(1+m)` in the direction I guessed
makes bins 4–6 worse (1.09–1.12), so I have **not** adopted a convention here — this needs the exact
Eq. C.9 form from Joachimi et al. 2021 before it can be called a pass. Recorded as open, not as
passing.

## 8. Recommendation (needs the user's decision; I have not started the long run)

A1 should be **redefined** rather than fixed: compare `sum(weight)` and `sum(weight²)` per pair
(target 1e-3), and compare xi_+/− against the official ascii (which is a genuine like-for-like test
and was already found to agree in shape). The 21-pair production run is still the only way to do
A2/A3/B2/B3, and its budget (30–50 h) is unchanged — but it is no longer motivated by a selection
bug, because there is no selection bug.

## 9. Files this analysis rests on

| file | what it holds | script |
|---|---|---|
| `paper_gate.json` | per-bin n, sum_w, sigma_e, SIG_E, neff vs paper neff | inline gate, this session |
| `factor_origin.log` | npairs + sum(weight) ratios for Bin1_1, Bin2_2, Bin1_2 | `scripts/` inline (10% subsample, slop=0) |
| `amplitude_check_INCONCLUSIVE.md` | why the 9-bin amplitude test is not usable | `scripts/dr5_amplitude_check.py` |
| `PAPER_SELECTION.md` | this file | – |

Verified this session, no file under `/mnt/d/HI-gamma-correlation/` modified.
