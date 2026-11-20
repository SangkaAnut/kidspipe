# KiDS-Legacy COSEBIs pipeline: independent adversarial review

Reviewer ran its own scripts (scratchpad `a.py`, `b.py`, `c.py`); no source file was modified (no clear bug found).
Test sample: 20 official-chain rows chosen independently of the implementer (seed 7: 10 weighted-random, 10 uniform-random
among rows with finite derived columns). Metric: diff = chi2_ours - chi2_off, chi2_off = -2(post - prior).

## Verdict
The implementation reproduces the official likelihood on an independent sample: base mean diff -0.30, std 0.63, max |diff| 1.67
(implementer: -0.15 / 0.62 / 2.95 on its 50). Every convention the implementer chose is the best of the alternatives I tried,
and most alternatives are rejected by 3-100 sigma-equivalents in chi2. Residual 0.6 std is plausibly CosmoPower-vs-CAMB error
(unproven, as the implementer says). No blocking issue found.

## Findings, by severity
1. (Medium, documentation) The implementer says the IA formula is "only weakly discriminated" (IA off changes chi2 by ~1). That
   is true for IA off, but the specific functional form is strongly discriminated (table below): dropping f_r (+47), dropping
   (M/Mpiv)^beta (+19), using log10 M ratio instead of M ratio (+16), (1-f_r) (+26), flipped sign (+6.8). The formula
   A_i = a f_r,i (M_i/M_piv)^beta (paper Eq. 11, text of arXiv:2503.19441 read) is confirmed empirically, not only by the paper.
   Doc claim in `docs/interface_notes/legacy.md` / report should be strengthened.
2. (Low) C1 rho_crit = 0.0138731 (limber.py, inherited from KV450). Using 0.0134 changes chi2 by only -0.03 mean (std 0.66, max 1.77
   vs 1.67): data cannot discriminate; official value unknown to us. Irrelevant for validation, but note for parameter inference
   (0.0134 vs 0.0139 is a 3.5% rescaling of A, degenerate with `a` whose prior is wide).
3. (Low) Residual max |diff| 1.67 (my sample) / 2.95 (implementer's, omch2~0.25 edge). Unexplained but consistent with emulator
   error; the 50-node ell grid is now tested (below): no effect. Not attributable to any convention toggled here.
4. (Info) 1016 of 18720 chain rows have NaN derived columns (S_8, A_S, Omega_m...) but zero posterior weight; the implementer
   correctly excludes them.
5. (Info) Nothing in the model depends on the chain's derived sigma_8; sigma_8 not used as a check by me.

## (a) FITS ordering and covariance sub-block (official best sample, row of max post)
chi2 ours 127.234 vs official 127.287. Alternatives with the same model vector:
| variant | chi2 |
|---|---|
| as implemented (En rows 1-6 per pair, En cov sub-block) | 127.23 |
| covariance from the Bn block instead of En block | 197.3 |
| model in mode-major ordering | 4978 |
| pair order reversed | 4295 |
| diagonal covariance only | 150.0 |
Header check: En table has 420 rows = 21 pairs x 20 ANGBIN, ordered pair-major / ANGBIN fastest, COVMAT header STRT_0=0 (En), STRT_1=420 (Bn).
The loader raises on any deviation. Selection is correct: En, modes 1-6, 126 of 840.

## (b) Convention toggles (20 independent samples, diff mean / std / max|diff|)
| toggle | mean | std | max | reading |
|---|---|---|---|---|
| **base** | -0.30 | 0.63 | 1.67 | |
| dz sign reversed n(z+bias) | +13.8 | 6.3 | 23.2 | rejected |
| n(z) nodes at Z_LOW instead of Z_MID | +3.4 | 3.9 | 11.8 | rejected (Z_MID confirmed) |
| linear instead of cubic n(z) shift | -0.41 | 0.63 | 1.74 | cubic marginally better (0.12 in mean), weak |
| IA without f_r | +47.1 | 37.7 | 157 | rejected |
| IA without (M/Mpiv)^beta | +18.9 | 5.2 | 25.5 | rejected |
| IA with log10M/13.5 in place of 10^(logM-13.5) | +16.4 | 4.7 | 22.9 | rejected |
| IA with (1-f_r) | +26.4 | 20.6 | 82.8 | rejected |
| IA sign flipped | +6.8 | 5.8 | 15.9 | rejected |
| C1 rho_crit 0.0134 | -0.33 | 0.66 | 1.77 | not discriminated |
| Omega_m in IA/NLA excluding neutrinos | -0.30 | 0.63 | 1.68 | not discriminated (nu mass tiny) |
| ell grid 50 nodes (official project_2d) | -0.30 | 0.63 | 1.67 | no effect |
| l_max 1e4/theta_max instead of 1e5 | -0.29 | 0.63 | 1.65 | no effect |
| CAMB kmax 20 h/Mpc | -0.30 | 0.63 | 1.68 | no effect |
| no baryon feedback (mead2020) | +0.07 | 1.61 | 4.8 | std rises 0.63 -> 1.61: feedback mapping supported |
| mnu = 0 | +0.08 | 1.34 | 3.2 | rejected-ish (std doubles) |
| neutrino hierarchy degenerate vs normal | 0.00 | 0.63 | 1.67 | not discriminated (identical to 1e-3) |
Note the "no feedback" and "mnu 0" toggles move the mean little but double the scatter, so mean-only metrics hide them; std is the discriminator.
Not tested: the exact log T_AGN mapping beyond on/off (e.g. an offset), HMcode-2020 vs another Mead version of feedback.

## (c) COSEBIs filters, independent checks (2-300 arcmin, modes 1-6, mpmath roots)
- int e^{2z} T_n dz and int e^{4z} T_n dz over z in [0, ln 150]: |value| <= 5e-10 and 1e-5 (relative to the scale e^{4Z} ~ 5e8, i.e. zero).
- Orthonormality int dtheta T_n T_m = (tmax - tmin) delta_nm: diagonal 1.0000, off-diagonal <= 5e-14 (own Gauss-Legendre quadrature).
- T_- recomputed from T_+ by my own quadrature: agrees to 5e-14.
- E_n route comparison for pair (3,5) at the best sample: xi_+/- by direct Bessel sums over a dense l grid (l 1..1e5), then
  E_n = 1/2 int dtheta theta (T+ xi+ + T- xi-); versus the C_ell operator M @ C: relative differences 5e-5, 5e-6, -1e-5, -5e-6, 6e-5,
  -5.5e-4 for modes 1-6 (mode 6 is the smallest value). B_n/E_n < 4e-4 for this pure-E input (consistency of T+/T- and constraints).
- Absolute normalisation (theta-linear convention) is verified only through the match to the official data chi2, which any error would wreck.

## (d) Existing test suite (KV450 unchanged)
Full `pytest tests`: 300 passed, 1 warning (18.5 min). KV450 tests unchanged and passing; legacy tests included.

## Not independently checked
Emulator-vs-CAMB residuals (no emulator access); official `bk_corrected` growth convention (D from P(kmin) is empirical, matched to 0.05 in
the best-sample chi2); Limber vs the official 50-ell-node interpolation scheme beyond the ell50 toggle (no effect).
