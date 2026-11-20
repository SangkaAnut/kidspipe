# KiDS-Legacy COSEBIs validation (src/kidspipe/legacy, tests/test_legacy.py, scripts/legacy_validate.py)

Model: CAMB 1.5.8 (`mead2020_feedback`, HMCode_logT_AGN, mnu 0.06 normal, kmax 10 h/Mpc, 200 k, log-log extension to 500 h/Mpc,
150-node z grid) -> Limber (`theory/limber.py`, 2000 z nodes, per-bin IA amplitude array; backward compatible) -> E_n
(`theory/cosebis.py` operator, filters regenerated with mpmath for 2-300 arcmin, modes 1-6, l in [1,1e5]) -> 126-vector
(21 pairs BIN1<=BIN2 pair-major, mode fastest = FITS En order, verified at load). Likelihood: Cholesky chi2 with the 126x126 sub-block of
the En part of COVMAT. No m-bias division (m is in the covariance), no c-term.

## Decisions and evidence (50 official-chain samples: 20 highest post + 30 weighted random, seed 1; chi2_off = -2(post-prior))
Official ln L = post - prior exactly equals -chi2/2 (no constant): our chi2 matches chi2_off with mean difference -0.15, so no offset needed.
| variant | mean diff | std | max abs | corr |
|---|---|---|---|---|
| **base** (below) | -0.149 | 0.620 | 2.95 | 0.979 |
| dz sign reversed: n(z+bias) | +15.2 | 4.5 | 24.5 | 0.34 |
| linear instead of cubic n(z) shift | -0.253 | 0.635 | 3.00 | 0.978 |
| clip negative cubic n(z) | -0.155 | 0.619 | 2.95 | 0.980 |
| IA off | +0.99 | 2.33 | 6.5 | 0.74 |
| single global NLA amplitude a (no f_r, M) | +1215 | 265 | | |
| ell grid 100 -> 200 nodes | 0.000 | 0.620 | 2.95 | 0.979 |  (50-node grid not run)
| lmax 1e5 vs 1e4/theta_max | +0.01 | 0.620 | | |
| kmax 10 vs 20 h/Mpc | -0.002 | 0.619 | | |
| HMcode-2020 without feedback (log T_AGN ignored) | -0.35 | 1.40 | 5.4 | 0.90 |
Base: n(z) = n(z - bias_i) (positive bias moves n to higher z), cubic spline on Z_MID nodes, renormalised;
A_i = a f_r,i (10^logM_i / 10^13.5)^beta (paper Eq. 11); F_i = -A_i C1rho_crit Omega_m / D(z), D = sqrt(P(kmin,z)/P(kmin,0)).
Chain column "A_S" is ln(1e10 A_s), not A_s*1e9. chain bias/log10_M columns are the physical correlated values.
sigma_8 of CAMB vs official: mean +5.6e-4 relative (std 3e-4), the residual is emulator-vs-CAMB error. Residual chi2 scatter (0.6) does not
correlate with sigma_8 mismatch and is largest at the prior-box edges (omch2~0.25), consistent with CosmoPower emulator error.
Official best sample (post -69.66): chi2_off 127.287, ours 127.234.
Run: `python scripts/legacy_validate.py --variants base,dz_neg,... --out runs/legacy/validate.json` (~75 s per variant, 4 threads).
