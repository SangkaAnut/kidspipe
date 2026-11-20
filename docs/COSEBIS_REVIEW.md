# COSEBIs independent review

Reviewer did not write the code. Clean-room rule respected (no reference/, docs/study/, docs/spec_checks). Scripts are
in the session scratchpad (rev_a.py, rev_b.py, rev_c.py); everything below was recomputed from the spec s14.1 only.
No source file was modified (no clear bug found).

## Verdict
No bug found. Filters, E_n (both routes), units, 1/2pi, FITS ordering and covariance structure all verified
independently. Remaining issues are conventions/limitations, listed by severity.

## Checks
(a) Roots/norms n=1..3 regenerated with my own mpmath code (mp.quad moments, not closed-form; dps 40) from the
defining conditions. vs staged tables: roots differ 1.7e-8 / 3.5e-8 / 3.4e-8 absolute (table rounding to 8 digits),
norms identical to printed precision. vs `regenerate_roots_norms`: roots identical, norms agree to 1e-39.

(b) E_n for pair index 3 (bins 4,1) at map_best.json, modes 1-5, fully separate route: model.cl -> my own log-spline of
C -> my own Gauss-Legendre l-quadrature (J0, J4, panels sized to the Bessel period) of xi+/-(theta) at 64 z-nodes ->
1/2 int dtheta theta (T+ xi+ + T- xi-), theta in radians.
- mine (lmax 1e4/theta_max = 1.146e5): 2.4479e-10, 2.5987e-10, 1.8263e-10, 1.2592e-10, 7.904e-11
- package E_from_cl:                  2.4480e-10, 2.5987e-10, 1.8263e-10, 1.2592e-10, 7.904e-11
- difference / peak: 1.7e-5, 1.3e-5, 3e-6, 6e-7, 4.6e-6 (about 1e-5 of the 1-sigma error of 1.5e-10 x 1e-1; negligible).
This also independently validates T_-n and the T_- recursion, because the package C_ell route uses only T_+ (via
W_n) while my route uses T_+ and T_-.

(c) Ordering. FITS rows: pair-major, mode fastest, (BIN1,BIN2) = (1,1),(1,2),...,(1,5),(2,2),... maps to
data.PAIRS (j,i) with j=BIN2-1, i=BIN1-1. Cross-pair labelling is irrelevant for E_n because xi is symmetric under bin
swap. Discriminating tests of the ordering at the best fit (5 modes, 75 pts): correct 71.2; column-major pair order
145.9; mode-major vector 278.5; 200 random pair permutations min 187.6, median 388. So the ordering is
unambiguously right. Data E_1 rises with source redshift (auto (1,1) -0.9e-10 noisy, (2,2) 4.9, (3,3) 4.0, (4,4) 8.2,
(5,5) 14.5, all x1e-10). Diagonal sigma(E_1) pattern: autos 2.5, 1.6, 1.3, 2.0, 2.6 (shot noise) vs crosses 1.0-1.7;
correlations: (1,1)-(1,2) 0.05, (1,1)-(2,2) 0.02, (2,3)-(2,4) 0.38, (4,5)-(5,5) 0.56 (pairs sharing a bin correlate
more), within-pair mode 1-2 correlation 0.84. NZ_SOURCE in the FITS has mean z 0.394/0.488/0.668/0.831/0.998;
pipeline n(z) 0.395/0.488/0.668/0.831/0.998 (agreement to 1e-3, consistent with the pipeline half-bin shift; shapes
identical to 3e-17 on the common grid). This closes implementer's open item.

(d) Units. theta in radians in all integrals, theta in arcmin only inside z=ln(theta/theta_min); T dimensionless;
1/2pi in E_n = 1/(2pi) int l W C confirmed by agreement with the xi route (b) (xi = 1/(2pi) int l C J). Truncation: my
lmax=1e6 vs kcap lmax=1.146e5 changes E_1..5 by (2.0, 0.8, 0.9, -0.8, 1.2)e-5 of peak, i.e. ~1e-4 sigma. Irrelevant.
Emulator ell grid is 0.1..5e5 (400 nodes), so no extrapolation is used below 5e5.

(e) m-bias. Spec silent. E_n is linear in xi so (1+m_a)(1+m_b) applies identically to xi+/-. Fit of a free overall
amplitude of the model to 20-mode data: A = 0.996 +- 0.086 (m-corrected) and 1.003 +- 0.086 (raw), chi2 293.5 vs
287.7 (the 5.8 difference comes from shape of the correlated residuals, ~2.4 sigma-equivalent at most for 300 dof, not
decisive). The data cannot distinguish the conventions; default apply_m=True is a reasonable but unverified choice.

## Tests
tests/test_cosebis.py: 19 passed (6.7 s). tests/test_e2e.py after the cl() hook: 31 tests passed with no failure
before my 580 s timeout; full completion NOT confirmed (it takes > 9 min here).

## Findings ranked by severity
1. (Medium, convention) m-bias treatment unverifiable from the shipped data; shifts chi2 by 1.3 (5 modes) / 5.8 (20
   modes). Report both when quoting COSEBIs results.
2. (Low) Tabulated roots have 8 digits only: orthonormality/B_n degrade for n > 10 (documented by the implementer);
   modes 1-5 unaffected. Use source='mpmath' (n <= ~12) if B_n at high n matters.
3. (Low) Tabulated T_- files are the less accurate side (1e-8..1.4e-5 of peak), already noted; package uses its own
   quadrature, fine.
4. (Low) B_n theory is a zero placeholder; no B-mode null test against the data B_n is implemented (the FITS has only
   E_n).
5. (Info) delta_c / A_c ignored by CosebisModel (stated). The 5-mode chi2 71/75 at the xi+/- best fit is a consistency
   check, not an independent fit; the same data are not independent of xi+/-.
6. (Info) Operator cache npz files under data/cosebis/cache/ pollute the tree.
7. (Info) Files in cache are keyed on ell grid and filter parameters only, not on a package version; if W_n code
   changes, stale caches would be silently reused (delete the cache dir).
