# nz_limber: interface notes (theory/nz.py, theory/limber.py)

## Deviations from DESIGN.md
* `cl_shear` has the plain-array signature requested by the task, not the `(lin, pk_nl_ext, k_ext, ...)` one in
  DESIGN.md:
  `cl_shear(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl, z_nz, nz_shifted, A_IA, ell, fidelity='fast', n_nodes=2000)
  -> C (nbin, nbin, nell)` (symmetric, all pairs filled). The caller must pass `chi_tab` in Mpc/h (= h * CAMB
  comoving_radial_distance in Mpc), `z_tab` ascending (6000 nodes in 0..6 in the spec), `pk_nl` already extended
  to 500 h/Mpc, `z_pk[0] == 0` exactly (needed for D(z), spec s5).
* Cached/warm interface: `Limber(ell, fidelity, n_nodes, interp_order=3, kernel_oversample=4)`;
  `.set_power(chi_tab, z_tab, omega_m, z_pk, k_h, pk_nl, z_max=None)` builds Q[m, l] (cosmology-dependent,
  ~5 ms warm, ~0.15 s on the very first numba call without the on-disk cache);
  `.set_nz_grid(z_nz)`; `.cl(nz_shifted, A_IA, z_nz=None)` -> (nbin, nbin, nell), ~1 ms (dz and A_IA change),
  ~0.4 ms when only A_IA changes (W is cached on an exact n(z) match).
* `NZ` is a dataclass (`z`, `n`, `shifted(dz)`, `mean_z()`); `load_nz(data_dir=None)` looks in
  `data/kv450/nofz` (lower case, as in this repo; `KIDSPIPE_DATA` overrides the data root).
* ell must be strictly ascending (the numba look-up walks a pointer along ln k).
* Integration upper limit is `z_max = min(z_tab[-1], z_pk[-1])` (= 6); n(z) is zero above 5.975 so this equals
  the spec's chi(5.975). A ValueError is raised if n(z) has support above z_max.

## Numerical choices (documented, beyond the spec text)
* Integration is a trapezoid in z (dchi = (dchi/dz) dz, dchi/dz from a cubic spline of the chi table).
* A_i, B_i (-> W_i) are accumulated on a 4x finer z grid than the Limber nodes and sub-sampled. Reason: the
  piecewise-linear n(z) has kinks every 0.05 in z, and a coarse trapezoid leaves a 3e-4 relative error in W at
  high z. With it, node convergence beats the spec numbers (toy P: 500 nodes 2e-4, 1000 5e-5, 2000 2.6e-5,
  4000 8e-6 vs 12000; CAMB P_nl: 2000 nodes < 1.5e-4).
* ln P_nl: linear in z, 4-point Lagrange cubic in ln k (interp_order=1 gives bilinear, differs by < 1e-3);
  exactly 0 outside [k_h[0], k_h[-1]] and outside [z_pk[0], z_pk[-1]] (spec D13).
* D(z) (NLA) is a cubic spline in z through sqrt(P_nl(k_h[0], z)/P_nl(k_h[0], 0)).
* `fidelity='kcap'` is only kcap-*like*: P (not ln P) is interpolated and W is divided by Z_i = int n dz of the
  sampled n; it differs from 'fast' by ~1e-4 here. It does not reproduce GSL Akima/bicubic splines or QAG noise.

## Observations
* VALIDATION_TARGETS quotes trapz(n) = 0.9945, 0.9981, 0.9985, 0.9994, 0.9998 for the RAW tabulation (lower edges,
  no prepend). The algorithm of spec s3.1 (shift by +0.025, prepend (0,0), trapz over 121 nodes) divides by a
  slightly different number (0.9973 for bin 1). Both are tested; after loading trapz = 1 to 1e-12.
* Agreement of cl_shear with the spec Takahashi anchors (CAMB 1.5.8, no IA, no shift): max |rel dev| = 7.6e-4 over
  4 pairs x l in {10, 100, 1000, 1e4}.
* tests/test_likelihood.py::test_covariance_pair_order_consistent_with_noise_structure (not my module) failed in a
  full-suite run at one point (0.134 < 0.12); it belongs to the likelihood agent.

## Review pass (independent adversarial check)
* Independent reference (Gauss-Legendre per n(z) segment, chi straight from CAMB, W from its definition at every
  outer node, bicubic spline of ln P, D(z) from the CAMB P_nl at the smallest k) agrees with `cl_shear` to 1.9e-4
  with A_IA = 1.5 and dz != 0 (5 bins, l = 10, 200, 3000); without IA 1.2e-4 (residual: linear-in-z vs cubic
  P interpolation). Now a test (`test_camb_with_ia_and_photoz_shift_vs_brute_force`, slow, ~1 s).
* Timings re-measured: cold first `set_power` 0.13 s, warm `set_power` 2.9 ms, `cl` 1.0 ms (dz + A_IA), 0.3 ms (A_IA only).
* Added input validation (previously silent): `shift_nz`/`NZ.shifted`/`normalise_nz` raise ValueError when the
  shifted distribution has zero or non-finite integral (was NaN rows); `set_power` raises on descending `k_h`
  (was C == 0 silently), non-finite or non-positive `pk_nl` (was clamped to 1e-300 / 0), non-monotone or
  non-zero-start `z_tab`, `z_max > z_tab[-1]` (chi extrapolation); `Limber(ell=...)` raises for ell <= -1/2 or
  non-finite (was NaN); `KIDSPIPE_DATA` pointing at a missing directory raises instead of silently falling back.
