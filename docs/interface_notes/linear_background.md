# linear_background (theory/linear.py, theory/nonlinear.py CAMB path)

Python 3.12, CAMB 1.5.8, 22-core WSL box (other agents were running concurrently, so absolute timings
are noisy by up to ~2x; ratios are what matter).

## Interface deviations / decisions
* `LinearResult` keeps the DESIGN positional order `(z, k_h, pk_lin, chi, z_chi, omega_m, sigma8_0, h)` and adds
  keyword extras: `pk_cb, ez, omnuh2, omega_b, omega_c, S8, n_s, ombh2, omch2, ln_1e10_A_s, mnu` and a private
  `_state` handle (CAMB run + A_s/n_s) used by the CAMB non-linear backend. Properties: `A_s, omega_nu_frac, k_mpc`.
  `omega_b`, `omega_c` are the dimensionless Omega_b = ombh2/h^2, Omega_c = omch2/h^2 (physical densities are `ombh2/omch2`).
* Cache key is `(omch2, ombh2, h, mnu)`, **not** including n_s: the n_s dependence is also an exact multiplicative
  tilt `(k_Mpc/0.05)^(n_s-n_ref)` (verified 1e-15 on P, identical k nodes). `LinearBackend(key_on_ns=True)` gives the
  strict DESIGN key. sigma8 for a changed n_s is obtained exactly by re-running only CAMB's transfer->power step
  (~0.3 ms). A model stage-1 cache keyed on n_s (DESIGN) is therefore pessimistic; `compute()` is cheap for any
  (n_s, A_s) once (omch2, ombh2, h) is cached (0.05 ms A_s-only, 0.6 ms with n_s change).
* `compute` always returns fresh `pk_lin`/`pk_cb` arrays; `z, k_h, chi, z_chi, ez` are shared **read-only** arrays.
* `z_grid` must start at 0 (sigma8_0 is read at z=0). `Z_GRID_SPEC` (150 nodes), `Z_GRID_REDUCED` (41 nodes:
  dz=0.05 on [0,1), 0.1 on [1,2), 0.4 on [2,6]) are exported. Cubic interpolation of ln P in z from the 41 nodes
  reproduces the 150-node values to <2e-4 for z<=3 (tested).
* `get_pnl(lin_or_params, A_baryon, backend="hmcode", **kw) -> pk_nl (nz, nk)` on the *same* (z, k_h) grid as `lin`
  (CAMB native nodes, k_h <= 21/h). `BACKENDS = {"camb": _pnl_camb}`; the default backend name stays "hmcode" per
  DESIGN, so `get_pnl(lin, A)` raises `KeyError` until the hmcode agent registers `BACKENDS["hmcode"]` in the
  reserved SECTION 4 of nonlinear.py. `lin_or_params` may also be a dict (`omch2, ombh2, h, n_s, ln_1e10_A_s[, mnu]`).
* `extend_high_k(k_h, pk, khmax=500., nnew=200)` accepts (nz,nk) or (nk,), returns `(k_ext, pk_ext)` with 199 new
  nodes (393 for the CAMB grid, spec s2.6 reproduced: slopes -2.568 lin; -2.162 (z=0) .. -1.680 (z=6) nl;
  P_nl/P_lin at last node 61.62 / 31.47 / 15.46 for z=0/2/6).
* Thread safety: CAMB's thread count is a process-global (`camb.config.ThreadNum`); the backend sets it before every
  CAMB call. Per-entry `RLock` protects the in-place re-powering of a cached CAMB result. Not fork-/MPI-aware: each
  process owns its own cache.
* The CAMB non-linear backend mutates the cached CAMBdata (A_s, n_s, NonLinear model are loaded in place via
  `results.Params` + `results.calc_power_spectra()`); linear arrays are unaffected (separate arrays are stored).

## Input validation (added in review)
`compute()` raises `ValueError` (before touching CAMB or the cache) for non-finite arguments, `omch2<=0`, `ombh2<=0`,
`h` outside [0.1, 3], `mnu<0`, `|n_s|>10`, or a `ln_1e10_A_s` whose A_s under/overflows; non-scalar arguments also raise.
Reasons (all reproduced before the fix): NaN `omch2/ombh2` corrupted the CAMB heap and negative `omch2`/NaN `h` crashed the
interpreter (segfault); NaN/negative `mnu` was silently treated as 0; a NaN/inf `A_s` on the first call of a cosmology was
stored as its reference amplitude so every later valid call returned NaN. `_run_camb` additionally refuses to cache a
non-finite or non-positive CAMB spectrum, and `compute` raises if the rescaled spectrum is non-finite.
The linear P(k=0.1005) anchor of the test cosmology is the *linear-in-k* `np.interp` between CAMB nodes (matches to 1e-14).

## CAMB usage investigation (what is cheapest)
Setup used: `CAMBparams.set_cosmology(H0, ombh2, omch2, omk=0, mnu=0.06, num_massive_neutrinos=1)`,
`set_dark_energy(w=-1, fluid)`, `Reion.Reionization=False`, `WantCls=False`, `set_matter_power(redshifts desc.,
kmax=20, nonlinear=False)` (-> `Transfer.high_precision=True`).

1. **Thread count.** `CAMBparams.num_threads` does *not exist* in CAMB 1.5.8 (assigning it silently does nothing; the
   OpenMP count comes from `OMP_NUM_THREADS` at import or from `camb.config.ThreadNum`, which can be changed at
   run time). Measured `camb.get_results` (150 z, kmax 20), quiet machine: 1 thr 2.04 s, 2: 1.0 s, 4: 0.53 s,
   8: 0.42 s, 12: 0.34 s, 22: 0.32 s. Whole `compute` under load (3 trials, min): 3.3 / 2.0 / 1.0 / 0.65 / 0.46 / 0.34 s.
   So the transfer stage scales well to ~4-8 threads. The **halofit/HMcode stage is where 22 threads hurts**:
   `get_pnl(..., 'camb')` (150 z) takes 0.99 s (1 thr), 0.63 (2), 0.51 (4), 0.51 (8) and **3.6 s (22 threads)**.
   Default `nthreads = min(4, ncpu)`; configurable via `LinearBackend(nthreads=...)`.
2. **Number of z nodes** has no measurable effect on the transfer run (150 vs 41 vs 2 nodes: 0.53 s each) but the
   halofit stage is linear in nz (150 z: 0.50 s; 41 z: 0.13 s at 4 threads).
3. **kmax**: run time grows with kmax (4 thr): 5 -> 0.40 s, 10 -> 0.45, 20 -> 0.53, 40 -> 0.67 s; nk = 188/191/194/197.
4. **Accuracy knobs** (speed vs sigma8 vs defaults): `Transfer.high_precision=False` 0.31 s (1.7x faster) but nk 130
   and sigma8 -1.4e-4 (k-sampling coarser, P interpolated differs ~2e-3) - kept True per spec s2.1 (exposed as
   `high_precision=`). `DoLateRadTruncation=False` 0.93 s and 1.6e-3 shift (not used). AccuracyBoost 1.5 -> 1.0 s, 2 -> 1.5 s,
   0.8 -> 0.46 s (nk 152) - not used. neutrino_hierarchy 'normal' vs default: identical to all digits at 0.06 eV.
5. **Extraction of P(k,z)**: `results.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)` returns the
   *native* (z ascending, nk=194) table in (Mpc/h)^3 and h/Mpc in **0.6-1.8 ms** with no interpolation; `var1=var2='delta_nonu'`
   gives the CDM+baryon spectrum in the same time (verified against fc^2 Pcc + 2 fc fb Pcb + fb^2 Pbb from
   `delta_cdm/delta_baryon`, <1e-6). `get_matter_power_spectrum` (resampling spline, 2-4 ms, warns when maxkh > kmax and
   deviates from native nodes) and `get_matter_power_interpolator` (builds a RectBivariateSpline, ~4 ms) are not needed.
   ln_1e10_A_s / n_s changes need no new transfer evolution: `results.Params.InitPower.set_params(...)` +
   `results.calc_power_spectra()` (0.2-0.5 ms) gives the exact new `get_sigma8_0()` and halofit spectra.
   (`r.Params` is the live state; the original `CAMBparams` object passed to `get_results` is a copy and changing it does nothing.)
6. **Distances**: `comoving_radial_distance(z, tol)` on 6000 z in ~1-2 ms (independent of `tol`); agrees with scipy quad of
   CAMB's own `1/H` to ~1e-14; `h_of_z` 0.3 ms. chi is multiplied by h to give Mpc/h.
7. **Memory**: ~4 MB per cached CAMBdata+arrays at 150 z x 194 k.

## Validation achieved (tests/test_linear.py, tests/test_nonlinear.py: 47 tests, ~35 s total)
* spec s2.2 sanity point: sigma8 = 0.779557, omega_m = 0.250296, omnuh2 = 6.44867e-4, nk 194, k in [7.1429e-5, 30.0].
* spec s4.1 test cosmology: sigma8 0.709092, S8 0.647692, chi(0.5) = 1345.58, chi(1) = 2384.96 (rel 1e-6), P_lin(k=0.1005) rel 1e-3.
* A_s and n_s rescaling vs fresh CAMB: max |dP/P| ~1e-15 (limit 1e-6); sigma8 exact.
* chi vs scipy quad of 1/E (E from the returned `ez` spline): 2e-7 (z<=1.9), 1e-5 at z=6 (limited by the 150-node spline).
* P_nl 'mead' vs stored `pk_mead.txt` (A=2, eta0=0.74): max 0.64 %, rms 0.3 % (and eta0=0.603 deviates >5 %: confirms hazard 2).
* P_nl(z=0,k=1.00298) = 350.03 and P_lin = 69.57 at (A=3.0, eta0=0.62) reproduced to 1e-4 (spec s4.3).
* CAMB-backed `get_pnl` with rescaled A_s/n_s/A_baryon == independent fresh CAMB 'mead' run to < 1e-8.

## Open questions
* VALIDATION L3 says "pk_mead anchor P(0,0.00968)=22860.8 is linear"; it is not: HMcode is -0.36 % below P_lin there
  (22878 nonlinear vs 22943 linear in our run). Linear-regime comparison with the file is therefore done for k<4e-3 (agree to <1e-3).
* kcap uses CAMB's *own* 5 %-padded kmax=21/Mpc; we follow (30.0 h/Mpc at h=0.7), `k_h[-1] = 21/h`.
