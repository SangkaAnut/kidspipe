# COSEBIs interface note (spec s14.1)

Files: `src/kidspipe/theory/cosebis.py` (filters, W_n, E_n), `src/kidspipe/likelihood/cosebis.py` (FITS data, chi2,
model/posterior helpers), `tests/test_cosebis.py`, `scripts/cosebis_validate.py` (numbers + `plots/cosebis_0*.png`,
`plots/cosebis_numbers.json`).  One additive hook in `theory/model.py`: `CosmicShearModel.cl(theta) -> (ell, C[5,5,nell])`
(same stage caches as `mu`, which is unchanged).

## Definitions implemented
- `z = ln(theta/theta_min)`, `T_+n = N_n prod (z - r_{n,i})` from `data/cosebis/roots_norms/{Root,Normalization}_<tmin>-<tmax>.table`
  (parsers tolerate blanks/comments, check n+1 roots per line, keep the 50-digit norm strings).
- `T_-n = T_+n + int_0^z 4 T_+n(y)[e^{2(y-z)} - 3 e^{4(y-z)}] dy`: 96-node Gauss-Legendre on [0,z] (integrand = polynomial x exponential;
  agrees with an mpmath adaptive quad to <1e-11 absolute).
- `regenerate_roots_norms(nmax, tmin, tmax, dps)`: mpmath, exact moments, linear system from
  int e^{2z}T = int e^{4z}T = 0 and orthogonality to lower modes (weight dtheta = tmin e^z dz), norm
  int dtheta T^2 = tmax - tmin (arcmin), positive leading coefficient.  Roots via `mp.polyroots`.
- `W_n(l) = int dtheta theta T_+n J_0(l theta)` (theta in RADIANS; T is dimensionless): Gauss-Legendre (10 pts) panels in theta,
  24 log-spaced base panels subdivided to <= 6 rad of Bessel phase; l grouped in octave tiers so T_+n is evaluated once per tier. Error ~1e-9 (test vs adaptive quad).
- **Units**: E_n uses theta in radians in `dtheta theta`; (the 299.5 of the orthonormality is in arcmin because T is a function of theta/theta_min).  Confirmed by the data: with radians chi2 = 71 for 75 points.

## E_n routes
1. `cosebis_operator(ell, filters, nmodes, lmin=1, lmax=1e4/theta_max_rad)` -> M[n,k]; `E_from_cl(C, M)` = `C @ M.T`.
   `E_n = 1/(2pi) int_{lmin}^{lmax} dl l W_n(l) C(l)`; l integral by composite Gauss-Legendre cells of width min(0.1 l, 24) with 5 nodes (W_n oscillates with period 2pi/theta_max = 72 in l, so
   a log-spaced l grid would under-sample it: this is why the integral is not done on the 400-node model grid).  C(l) between the model nodes: natural cubic spline in ln l of *linear* C (kcap-style, linear in C so it folds into M);
   outside the model grid C_0 (l/l_0)^{+1}, C_end (l/l_end)^{-3} (as hankel.py).  M is cached in memory and as npz in `data/cosebis/cache/` (20-mode operator: 2.2 s uncached, ~0.1 s from cache).
2. `E_from_xi(theta_rad, xip, xim, filters, nmodes)` -> (E_n, B_n): `1/2 int dz theta^2 [T_+ xi_+ +- T_- xi_-]`, xi from `HankelBinner.xi_theta` (FFT route) splined in ln theta.
   B_n of a pure-E C_ell is a test of the T_-/T_+ relation (|B|/|E| < 1e-4).  Theory B_n = 0 (`CosebisModel.B_theory`).

l range: kcap uses l in [1, 1e4/theta_max_rad = 1.146e5].  Convergence of E_n(C_l) towards E_n(xi) (n<=5, MAP): lmax=1e4: 0.012 sigma, 3e4: 9e-4 sigma, 6e4: 2.2e-4 sigma, 1.146e5: 2.0e-4 sigma,
5e5: 1.8e-4 sigma.  So the kcap range is converged (floor 2e-4 sigma is FFT/spline noise); lmin 0.1 vs 1 changes E_n by 6e-9 sigma.

## Data / likelihood
- FITS `En`: 300 rows pair-major (BIN1<=BIN2), mode (ANGBIN) fastest.  This equals `data.PAIRS` order with (j,i)=(BIN2-1,BIN1-1): verified at load. `ANG` column = mode index (unit label 'arcmin' is a mislabel).
- `n_modes` N (default 5, up to 20): rows with mode <= N, vector (15 pairs x N modes), covariance sub-block.
- chi2 = |L^-1 r|^2 with scaled `cov/1e-20`, `r/1e-10`.  COVMAT symmetric exactly, positive definite (min eig 2.3e-22; condition number 7.2e3 for N=5, 2.9e4 for N=20).
- **m-bias**: E_n is linear in xi so E_obs = (1+m_a)(1+m_b) E_true.  Default `apply_m=True` divides the data by (1+m_a)(1+m_b) with `params.M_BIAS` (kcap xi+/- convention); `apply_m=False` uses data as shipped.
  Whether the shipped E_n are already m-corrected is not documented; the chi2 difference is 1.3 (N=5) and 5.8 (N=20) at the MAP, smaller than the 5-mode noise.
- **delta_c, A_c do not apply** (additive xi c-term / delta_c^2 offsets are not part of the COSEBIs model here); `CosebisModel.mu(theta)` ignores those entries of theta.  Photo-z shifts dz, A_IA, A_baryon, cosmology enter via C_ell.
- `CosebisModel(model, n_modes)` : theta(14) -> E (15*n_modes,);  `CosebisLikelihood(n_modes, apply_m)`; `CosebisPosterior(model, like).log_post(theta)` (params.log_prior + COSEBIs ln L).

## Results (emulator model, xi+/- MAP `runs/kv450_final/map_best.json`)
| quantity | value |
|---|---|
| chi2, 5 modes (75 pts), m-corrected / raw | 71.15 / 69.87 (null 206.5 / 207.3) |
| chi2, 20 modes (300 pts), m-corrected / raw | 293.5 / 287.7 (null 428.9 / 425.1) |
| E(C_l) vs E(xi), max over 20 modes, 15 pairs | 2e-4 sigma (n<=5: 1.8e-4 relative to the entry, 8e-5 of peak) |
| T_+ vs tabulated (0.5-100) | 1e-16 relative to peak |
| T_- vs tabulated (0.5-100) | 4e-9 (n=1) ... 1.4e-5 (n=20) of peak: table is the less accurate side; ours matches mpmath to 1e-14 |
| mpmath regeneration vs table, n<=10 | roots 1.7e-8 .. 5e-8 abs (8-digit rounding of the table), norms 0 .. 1e-15 rel |
| orthonormality (0.5-300, 20 modes) | 8.4e-5 abs of 299.5 = 2.8e-7 rel (limited by 8-digit table roots) |
| int e^{2z}T, int e^{4z}T / int|T|e^{4z} | 5e-12, 1.4e-6 (n up to 20, 8-digit roots) |
| posterior samples (200): chi2 5 modes | 68.2 / 74.7 / 88.8 (min/median/max); 20 modes 290.3 / 296.8 / 312.1 |
Caveat: with only 8 significant digits in the root tables, modes n >~ 12 are accurate only to ~1e-4-1e-3 in B_n (pure-E test) -- irrelevant for n<=5; mpmath regeneration gives full precision if needed (`CosebisFilters(source="mpmath")`, nmax <~ 12 at dps 80-100).
