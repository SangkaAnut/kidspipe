# Independent verification of the sigma_e result

Verifier script: `scripts/dr5_sigma_e_verify.py` (does not import `dr5_sigma_e.py`); output `runs/dr5_measure/sigma_e/verify.json`.
Sources opened by me this session: Joachimi+21 (arXiv:2007.01844v2) App. C pp. 36-38; Wright+25 (arXiv:2503.19441v2) Table 1 and Sect. 3.5; the FITS header `NZ_SOURCE` of `KiDS_Legacy_xipm.fits`; the catalogue readme.

## Per-claim verdicts

1. **Eq. C.9 and K(0) quoted correctly: CONFIRMED.** Source text: sigma^2 = K^-1(0) sum_i w_i^2 [(e_i1)^2+(e_i2)^2] (C.9), K(theta) ~ (1+<m>)_I(1+<m>)_J sum w_i w_j Delta_ij (C.5), Delta_ij(0)=delta_ij, so K(0)=(1+<m>)^2 sum w^2. Eq. C.8 defines sigma_eps via <e_t e_t + e_x e_x>, i.e. the TOTAL (both components). "Variances are computed after additive bias correction" also confirmed.
2. **Per-component = total/2: CONFIRMED.** Wright+25 Table 1 notes: "sigma_eps values are the ellipticity dispersion averaged over both components" and "include the appropriate correction for multiplicative shear biases (i.e. the m_gold column)", computed with Eqs. C.9 and C.12, area 967.4 deg2.
3. **Division by (1+m) (not multiplication): CONFIRMED numerically.** Readme: e1,e2 are "c-corrected and recalibrated ... no m corr", so m has not been applied in the file; the C.1 model e_obs=(1+m)[...] implies dividing. My ratios to SIG_E_i (bins 1-6): 1.00003, 0.99991, 0.99954, 0.99972, 0.99940, 0.99933 (division); no m: 0.9770, 0.9839, 0.9886, 1.0197, 1.0294, 1.0443; multiply: 0.9546, 0.9682, 0.9777, 1.0401, 1.0603, 1.0913. These agree with the colleague's table to 4 digits (bin 5 and 6 division: mine 0.9994 / 0.99933, theirs 0.9994 / 0.9993).
4. **Alternatives rejected: CONFIRMED.** sum w|e|^2/sum w /2 with 1/(1+m): 1.019, 1.035, 1.026, 1.049, 1.031, 1.026. Unweighted with 1/(1+m): 1.052, 1.089, 1.062, 1.110, 1.062, 1.033. Only C.9 / (1+m) is within 0.1% in all bins.
5. **SIG_E_i / NGAL_i values and location: CONFIRMED.** I re-read header `NZ_SOURCE` of `KiDS_Legacy_xipm.fits`: SIG_E_1..6 = 0.27717, 0.27156, 0.28991, 0.26186, 0.28017, 0.30018 (identical to sigma_e.json).
6. **m_gold values: CONFIRMED against the paper itself.** Wright+25 Table 1 last column ("m_gold [x1e-2]"): -2.3, -1.6, -1.1, +2.0, +3.0, +4.5 (errors 0.6-0.9). NB the preceding column (delta_z: -2.6, 1.4, -0.2, 0.8, -1.1, -5.4) is NOT m; the colleague used the correct column. Counts N_gold in Table 1 equal the cached per-bin n (7 416 371 ... 5 960 461).
7. **Bin indexing, weight column: CONFIRMED.** I asserted TOMOBIN==b+1 for every row of extra/bin{b}.npz and that extra `weight` equals bin{b}.npz `w` (rtol 1e-6, same row order). `weight` is the final shear x gold weight per readme.
8. **c-term handling: CONFIRMED irrelevant.** Weighted mean of e1,e2 in every bin is 0.0000 (rounded to 4 d.p.), and re-subtracting a weighted-mean c per patch (N/S) and component leaves ratios unchanged to 5 digits. Float32-input vs float64 accumulation also identical. No double counting of (1+m): m enters once, in the denominator.
9. **n_eff / Eq. C.12 concern (colleague left open): RESOLVED, not an issue.** In C.12 n_eff = (sum What)^2/(A sum What^2); for a per-bin constant m the (1+m)^2 cancels exactly, so n_eff with and without m are identical. This explains why n_eff matches NGAL without any m. (Analytic argument; I did not add a separate numeric run.)

## ERRORS FOUND (minor, none changes the verdict)

- **E1. Rounding explanation is wrong for bin 5.** SIGMA_E.md Sect. 4 and the summary say bins 1-5 differ from the implied m by less than the 0.0005 Table-1 rounding. My implied m (from verify.json): -0.02297, -0.01609, -0.01145, +0.01972, +0.02938, +0.04430. Differences to Table 1: 0.00003, 0.00009, 0.00045, 0.00028, **0.00062**, **0.00070**. Bins 5 and 6 exceed 0.0005, and the implied values would round to 2.9 and 4.4 (x1e-2), not 3.0 and 4.5. So the residual in bins 5 and 6 is NOT explained by table rounding. The cause is not established (candidates, untested: m averaged differently over the two components or unrounded in the pipeline, small difference in the weight/sample used by the release). Size: 0.06-0.07% in sigma_e, about 0.1 of the quoted m error (0.8-0.9e-2), so immaterial for the gate.
- **E2.** Wording "sigma_e matches in all six bins to 0.07%" is correct (max 0.067% in bin 6), but the claim that the formula is fully explained is slightly overstated given E1.

## Verdict

Gate PASS is CONFIRMED: Eq. C.9 per component with 1/(1+m_gold), catalogue `weight`, reproduces SIG_E_i to within 0.07% in all bins. Recommend the gate tolerance be stated as 1e-3 and the rounding explanation for bins 5-6 be dropped from the records (residual unexplained, harmless).
