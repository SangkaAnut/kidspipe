# sigma_e gate: resolved (verdict PASS, 0.07 % in all six bins)

Produced by `scripts/dr5_sigma_e.py` (cached npz + `NZ_SOURCE` header of `KiDS_Legacy_xipm.fits` only).
All numbers are in `runs/dr5_measure/sigma_e/sigma_e.json` (360 variants x 6 bins, ranked list under `ranked`).

## 1. Definition (source text)

Joachimi et al. 2021, A&A 647, A124 (arXiv:2007.01844v2), Appendix C.3 "Shape noise estimates", p. 37:

> Eq. C.8: <eps^obs_it eps^obs_jt + eps^obs_i× eps^obs_j×> = sigma_eps^2 delta_ij + xi_+(|theta_i - theta_j|)
>
> "the shape noise can therefore be estimated from our weighted and calibrated ellipticity distribution as
> **sigma_eps^2 = K^-1(0) sum_i w_i^2 [ (eps^obs_i1)^2 + (eps^obs_i2)^2 ]**   (C.9)
> ... In evaluating K(0) we use the limit Delta_ij(0) = delta_ij. The variances of the ellipticities are calculated after the additive bias correction has been applied."

with (Eq. C.5) K(theta) = sum Ŵ_i Ŵ_j Delta_ij ≈ (1+<m>)_I (1+<m>)_J sum w_i w_j Delta_ij, Ŵ_i = w_i (1+m̂_i).
Hence K(0) = (1+<m>)^2 sum w_i^2, and

    sigma_eps^2 (total, e1^2+e2^2) = sum w^2 (e1^2+e2^2) / [ (1+<m>)^2 sum w^2 ].

Dividing by 2 gives the per-component value (the one quoted: 0.28, 0.27, ... in Wright+25 Table 1).
Note the correction is a DIVISION by (1+m) (calibrated ellipticity = e/(1+m)); this is the direction the earlier attempt got wrong.
The weights enter squared and the normalisation is sum w^2, not sum w. The earlier "raw" 0.2760 (bin 1) was
sum w|e|^2/sum w, which is a different estimator.

## 2. Where SIG_E_i and NGAL_i live

- `.../KiDS_Legacy_cosmic_shear_data_release/data/KiDS_Legacy_xipm.fits`, extension `NZ_SOURCE`, keywords `SIG_E_1..6`, `NGAL_1..6`
  (identical in `KiDS_Legacy_cosebis.fits` and `KiDS_Legacy_bandpowers.fits`). The .ini files and the catalogue readme contain no sigma_e or m values.
- SIG_E = 0.27717, 0.27156, 0.28991, 0.26186, 0.28017, 0.30018. NGAL = 1.68842, 1.57357, 1.42851, 1.39076, 1.28324, 1.01906.
- m_gold: only in Wright+25 Table 1 (A&A 703, A158, arXiv:2503.19441), x1e-2: -2.3, -1.6, -1.1, +2.0, +3.0, +4.5 (errors 0.6-0.9). No m file in the release; no per-galaxy m column in the catalogue (51 columns, none is m), so only per-bin m can be applied. A per-galaxy application is not testable here.
- Table 1 sigma_e (rounded to 0.01): 0.28, 0.27, 0.29, 0.26, 0.28, 0.30 (agrees with SIG_E rounded).

## 3. Ratio to SIG_E_i (bins 1..6), weight = catalogue `weight`, no extra c subtraction

| variant | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---|---|---|---|---|
| C.9 without m | 0.9770 | 0.9839 | 0.9885 | 1.0197 | 1.0294 | 1.0443 |
| **C.9 with 1/(1+m)** | **1.0000** | **0.9999** | **0.9995** | **0.9997** | **0.9994** | **0.9993** |
| C.9 with x(1+m) (wrong direction) | 0.9546 | 0.9682 | 0.9777 | 1.0401 | 1.0603 | 1.0913 |
| sum w|e|^2/sum w /2, no m (earlier "raw") | 0.9958 | 1.0183 | 1.0145 | 1.0702 | 1.0617 | 1.0722 |
| sum w|e|^2/sum w /2, with 1/(1+m) | 1.0192 | 1.0349 | 1.0258 | 1.0493 | 1.0308 | 1.0261 |
| sum w^2|e|^2 N/(sum w)^2 /2, 1/(1+m) | 1.097 | 1.132 | 1.141 | 1.164 | 1.175 | 1.265 |
| unweighted |e|^2/2, no m | 1.028 | 1.071 | 1.050 | 1.132 | 1.094 | 1.080 |
| C.9 no m, shear_weight_only | 0.9881 | 0.9908 | 0.9931 | 1.0241 | 1.0303 | 1.0464 |
| C.9 no m, gold_weight_only | 1.0111 | 1.0617 | 1.0435 | 1.1276 | 1.093 | 1.0797 |

Other findings from `ranked` in the json:
- `weight` and `shear_weight_only*gold_weight_only` give identical results (weight = shear x gold), so they are the same variant.
- c subtraction (none / per bin / per patch N,S) changes nothing at the 4th decimal (catalogue e is already c-corrected), so it is irrelevant.
- Best non-C.9-with-1/(1+m) variant: `shear_weight_only` + C.9 + 1/(1+m), max deviation 1.1 %. Everything else is worse than 4 %.
- Only a single variant family reproduces all six bins: C.9, weight = `weight` (equivalently shear x gold), 1/(1+m_gold) per bin; max |ratio-1| = 0.0007.

## 4. Residual and consistency checks

- Solving for the m that makes C.9 exact gives (1+m)_implied - 1 = -0.02297, -0.01609, -0.01145, +0.01972, +0.02938, +0.04430
  vs Table 1 -0.023, -0.016, -0.011, +0.020, +0.030, +0.045. Differences 0.00003-0.0007, all far inside the quoted m errors (0.006-0.009).
  The residual 0.01-0.07 % is a slight drift in bins 4-6; plausible explanations are m rounding in Table 1 (bins 1-5 differ by less than the 0.0005 rounding; bin 6 differs by 0.0007, marginally more) and
  the exact m used in the pipeline being unrounded. Not established beyond that; the sign of the drift (ratio <1 in bins 3-6) is consistent with true m slightly smaller than tabulated.
- n_eff is unchanged: n_eff(weight, 967.4 deg2) = 1.7698, 1.6494, 1.4974, 1.4578, 1.3451, 1.0682 arcmin^-2, vs NGAL x 1014.03/967.4 = 1.7698, 1.6494, 1.4974, 1.4578, 1.3451, 1.0682 (RESUME decision 3 confirmed to 5 digits).
  Joachimi Eq. C.12 would include (1+m) in n_eff; we did not apply it (it agrees to 1e-5 without), so the Legacy n_eff apparently omits it. That is an observation, not tested further.

## 5. Verdict: PASS

sigma_e^2 = sum w^2 (e1^2+e2^2) / [2 (1+m_gold)^2 sum w^2], with `weight` and per-bin Table-1 m_gold, reproduces SIG_E_i in all six bins to 0.07 % (max). The match is not merely empirical: the formula is Eq. C.9 of the cited methodology paper with K(0) from Eq. C.5, and the m values are the paper's own. The per-galaxy variant of (1+m) could not be tested (no per-galaxy m in the file).

## 6. Recommendation for the gate

Close the sigma_e gate as PASS with threshold ratio within 1e-3 of 1 in every bin (observed max 7e-4), define the check as the Eq. C.9 estimator above (sum w^2 weighting, per-component = total/2, division by (1+m_gold)^2 inside the square root), and record the m_gold values and the SIG_E/NGAL header location. Retire the earlier "0.4-7 % high" note in PAPER_SELECTION.md Sect. 7 and RESUME.md (the lead updates these records): it came from using sum w|e|^2/sum w and from applying (1+m) in the wrong direction. Keep n_eff on the 967.4 deg2 logic unchanged.

## Verifier note (independent re-derivation, see VERIFY.md)
Result CONFIRMED (Eq. C.9, per-component, /(1+m_gold), `weight`). One correction: the statement that the residual is explained by Table-1 rounding holds for bins 1-4 only. In bin 5 the implied m (0.02938) differs from 0.030 by 0.00062 and in bin 6 by 0.0007, both beyond the 0.0005 rounding, so the 0.06-0.07% residual in bins 5-6 is unexplained (immaterial). Also, n_eff needs no (1+m) because it cancels in Eq. C.12 for a per-bin constant m.
