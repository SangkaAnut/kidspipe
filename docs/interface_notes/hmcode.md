# Interface notes: `kidspipe.theory.hmcode`

## Public signature (deviation from DESIGN.md: extra optional `pk_cb`, keyword options)

```python
hmcode_pnl(z, k_h, pk_lin, omega_m, omega_nu_frac, h, n_s, A_baryon, eta0, pk_cb=None, *,
           ms=24, dl=0.04, mmin=1.0, mmax=1e18, kmin_nl=0.0, omega_rad_h2=3.6e-5,
           dolag=1.0, tol_win=1e-10) -> pk_nl[nz, nk]
hmcode_pnl_diag(...same...) -> (pk_nl, diag[nz,10])   # sigma8(z), sigma_v, sigma_v100, f, neff, alpha, delta_c, Delta_v, eta, k*
pk_cb_approx(z, k_h, pk_lin, omega_m, omega_nu_frac) -> pk_cb   # fallback only
```

Inputs (all float64, units h/Mpc and (Mpc/h)^3):

| arg | meaning |
|---|---|
| `z` (nz,) | redshifts, any order, each row independent (a reduced z grid is simply a shorter array; the result for a z does not depend on the other z's) |
| `k_h` (nk,) | strictly increasing; use CAMB's **native** nodes (~195). P_nl is returned on the same nodes. The first/last node set the sigma(R) integration range (above the last node ln Delta^2 is continued as a power law, see module docstring) |
| `pk_lin` (nz,nk) | linear **total-matter** P (CAMB `delta_tot`, `get_linear_matter_power_spectrum()` default) |
| `pk_cb` (nz,nk) | linear **CDM+baryon** P (CAMB `var1=var2='delta_nonu'`). **Needed by the model**: sigma(R), neff of the halo model use the cold spectrum. With m_nu=0.06 eV, using `pk_lin` instead of `pk_cb` biases P_nl by ~1 % at high k. If omitted, `pk_cb_approx` (fit, ratio error <=1.4e-3) is used. `LinearResult` in DESIGN.md has no `pk_cb` field: **please add `pk_cb` (and `n_s`, `omnuh2` or `f_nu`) to `LinearBackend.compute`** - it costs one extra `get_linear_matter_power_spectrum` call on data CAMB already has (A_s rescaling applies equally to pk_cb) |
| `omega_m` | late-time total matter Omega_m = (omch2+ombh2+omnuh2)/h^2 (CAMB `omegam`) |
| `omega_nu_frac` | f_nu = omnuh2/(omch2+ombh2+omnuh2) = omnuh2/(omega_m h^2) |
| `h` | only enters the (tiny) radiation term in E(z) |
| `n_s` | only for the clip range of n_eff, [n_s-4, n_s] |
| `A_baryon`, `eta0` | halo concentration amplitude and window parameter; `nonlinear.get_pnl` should pass `eta0 = 0.98 - 0.12*A_baryon` |

`get_pnl(lin, A_baryon, backend="hmcode")` therefore is
`hmcode_pnl(lin.z, lin.k_h, lin.pk_lin, lin.omega_m, f_nu, lin.h, n_s, A, 0.98-0.12*A, lin.pk_cb)`.

## Performance
* After JIT: **54 ms** single thread for 150 z x 194 k (default `ms=24`), 35 ms with `ms=16`; 14 ms with 4 threads; reduced 25-z grid 9.5 ms.
* Parallel over z with `numba.prange`. To control threading use `numba.set_num_threads(n)` (global) or `NUMBA_NUM_THREADS`.
* First call in a fresh checkout compiles for ~11 s; `cache=True` stores the machine code in `src/kidspipe/theory/__pycache__` (directory must be writable), later processes start in <1 s.
* z-independent operators (dense ln k grid, R x k window matrix) are cached per k-grid (`lru_cache`, 6 grids); the first call for a new k grid costs ~10 ms extra.

## Accuracy (details in `benchmarks/hmcode_vs_camb.py`, `tests/test_hmcode.py`)
* Stored array `data/validation/pk_mead.txt` (A=2.0, eta0=0.74, 1e-4<k<30, all z): **max 0.80 %, rms 0.29 %** (CAMB 1.5.8 itself: 0.82 % / 0.30 %); target 1 %. (eta0=0.603 as in the repo scripts gives >5 %: confirmed.)
* vs CAMB 1.5.8 `mead`, same linear input, A in {2, 2.5, 3.13}, eta0=0.98-0.12A:
  * KV450 start, Planck-like, high-Om, Om>1, ns=0.7/lnAs=5, massless nu: **<= 1.3e-3 for z<2, <= 2.3e-3 for 2<=z<=6**.
  * After the review fix below, all benchmark cases incl. the extreme corners are within the targets: omch2=0.01 (Om=0.049, ns=1.3, lnAs=5): 2.6e-3 (z<2, worst at k~0.1, z=0) / 9e-4 (z>=2); low-amplitude red corner (omch2=0.05, ombh2=0.02, h=0.65, ns=0.8, lnAs=2.5, A=3.13): 1.2e-3 / 3.7e-4 (the earlier "no root for nu=1" explanation was wrong; the residual was the continuation slope). Random draws over the whole KV450 prior box (25 draws, incl. sigma8 ~ 0.1-3): worst 3.0e-3 (z<2; Om~0.1, sigma8~0.5) and 1.2e-3 (z>=2).
* Numerical convergence (internal): ms=24/dl=0.04 vs ms=48/dl=0.02: < 3e-4.

## Choices where the spec is silent (all documented in the module docstring)
1. **No hard switch at Min_kh_nonlinear=0.005**: CAMB 1.5.8 output shows the smooth 2-halo damping continuing below 0.005 h/Mpc (ratio 0.9995 at k=0.003); `kmin_nl=0.005` reproduces the literal spec switch (changes P by <= 1e-3 near 0.005).
2. sigma(R): exact integral on a dense ln k grid (4-point Lagrange in ln Delta^2), power-law continuation above the last CAMB node (slope between the last 2 nodes; a 3-node LS slope was tried and is 3-5x worse vs CAMB at the last k nodes in the low-Om corners, see 'Review fixes' below) up to 5e6 h/Mpc. Truncating at kmax instead is *wrong* (errors up to 40 %).
3. Growth factor: exact flat matter+Lambda hypergeometric solution (no ODE, no radiation, neutrinos as matter); Dolag factor = 1 (w=-1).
4. Mass/radius integration: uniform ln R nodes, Jacobian dnu/dlnR = -nu dln(sigma)/dlnR with 4th-order differences (exponentially convergent, unlike trapezoid-in-nu which converges slowly and noisily at high k). Range M in [1, 1e18] Msun/h; masses outside are dropped, the nonlinear radius is searched inside this range.
5. n_eff = -3 - dln(sigma_cb^2)/dlnR at nu=1 (sigma = delta_c), cubic-Hermite interpolation of the root; clipped to [n_s-4, n_s]; alpha clipped [0.5, 2]. The curvature n_cur of Mead (2015) is not used by HMcode-2016 and is not computed.
6. sigma_8(z), sigma_v, sigma_v100 from **total**; sigma(R), n_eff, formation redshift from **cold** P (confirmed against CAMB: using total everywhere gives ~1 % error).
7. NFW window evaluated via non-oscillatory auxiliary functions f,g (tables, abs. error <= 5e-6 vs scipy Ci/Si).
8. Omega_m(z) uses E^2 with a radiation term Omega_r h^2 = 3.6e-5 (effect <= 3e-4 at z=6).

## Open questions
* Whether `get_pnl` wants a CAMB-based fallback (`backend="camb"`) when `pk_cb` is not available - not implemented here (hmcode is CAMB-free).
* For z>6 the model is fine numerically (growth table to a=1e-5) but untested against CAMB.

## Review fixes (independent review pass)
* **Continuation slope**: above the last native k node ln Delta^2 was continued with the least-squares slope of the last 3 nodes; CAMB's last nodes are jittery, which produced errors up to 3.2e-3 (z<2) and 4.6e-3 (z>=2) in low-Om corners. The slope of the last two nodes matches CAMB 3-6x better (z>=2 worst case 4.6e-3 -> 9e-4) and is now used. Consequence: the result is only meant for CAMB's native node set (a very fine resampled grid makes a two-point slope noisy; a 100-node resampling is tested to stay within 3e-3).
* **Input validation**: NaN/inf/non-positive P, negative z, A_baryon <= 0, omega_m <= 0, f_nu outside [0,1), h <= 0, bad ms/dl/mass range etc. previously returned silent garbage (fastmath kernels) or a cryptic `SystemError`; now `ValueError`.
* z is made C-contiguous before the numba call (one compiled signature).
* Tests: the z grid of the CAMB comparison now includes z=0.25, 0.75, 1.9 (the old grid hid the worst z of the extreme corners); the 4e-3 exception for the low-amplitude corner is gone (spec 3e-3 applies everywhere); z>=2 is guarded at 4e-3; new tests for validation, determinism / thread independence, halo-model parameters (delta_c, Delta_v, eta, alpha, neff) vs independent scipy quadrature, and k-grid resampling.
* Verified, not changed: timing claims (single thread 52-55 ms, 4 threads 14 ms, 22 threads 7.5 ms, 25 z 9.5 ms), sigma_8 / sigma_v quadratures, benchmark script runs.
