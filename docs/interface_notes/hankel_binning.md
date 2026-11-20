# hankel_binning (src/kidspipe/theory/hankel.py, tests/test_hankel.py)

## Interface
`HankelBinner(ell, theta_edges_arcmin=None, fidelity="fast"|"kcap")`
* `xi_binned(C)` -> `(xip, xim)`, each `(15, 9)`; `C` is `(5,5,nell)` (only `C[j,i]`, j>=i, pair order
  `PAIRS = [(j,i) for i in range(5) for j in range(i,5)]`) or already `(15,nell)`.
* `ell` is fixed at construction (the fast operator is built for exactly that grid, ~0.15 s for the 400-node
  kcap grid; build once, reuse). Any strictly increasing positive grid with >= 4 nodes works (not necessarily log-uniform).
* Extras: `xi_binned_fft(C, binning="kcap"|"exact")`, `xi_theta(C)` (unbinned FFT result on the 1024-node theta grid),
  `M_plus`, `M_minus` (9 x nell), `W_kcap`, `W_exact` (9 x 1024), module functions `cl_to_xi_fft`, `hankel_kernel`,
  `ClInterpolator`, `bin_operator`, `bessel_bin_average`, `bessel_antiderivative`.

No deviation from DESIGN.md. Notes:
* `fidelity="fast"` = continuous definition (spec s7.1) + exact bin average (s8 (E)) as a dense operator.
  `fidelity="kcap"` = FFT (s7.2) + node-trapezoid (s8 (F)). The "exact-binning FFT" mix is available only via `xi_binned_fft`.
* Fast operator, C off the grid: local cubic Lagrange in ln l between nodes (linear in C), tails C = C_0 (l/l_0)^{+1}
  (integrated from l=0) and C = C_{-1}(l/l_{-1})^{-3} (integrated to l = 5e6, end of the FFT grid), i.e. the same
  extrapolation as the FFT route. Tail columns of M carry large entries (up to ~5e4 for the smallest bin) but C there is ~1e-16.
* kernel uses kcap's 8-digit `ln 2 = 0.69314718` by default (`LN2_KCAP`); changing to the exact value moves xi by < 1e-7 relative.

## Independent review (hankel_binning reviewer pass)
Public signatures unchanged. Behaviour changes (all strictly stricter, no change to any valid-input number):
* `HankelBinner`, `ClInterpolator`, `bin_operator` now validate `ell` (finite, strictly increasing, positive; NaN used to be accepted for
  `fidelity="kcap"`) and the theta edges (finite, positive, strictly increasing, >= 2 entries; descending/negative/duplicate edges used to be
  accepted silently by the fast operator and produced garbage).
* `xi_binned`, `xi_binned_fft`, `xi_theta` raise `ValueError` if the result is not finite (NaN/inf in C_ell used to give NaN rows silently).
  Same policy as `limber.py` (which raises on non-finite P).
* `xi_binned_fft(binning=...)` raises for anything but `"kcap"`/`"exact"` (a typo used to fall through to `"exact"`).
* `HankelBinner.ell` is a private read-only copy (in-place edits of the caller's array used to desynchronise the precomputed operator);
  `THETA_EDGES_ARCMIN` is read-only.
* Deviation from the spec text (recorded, no numerical effect for the KiDS grid): spec s7.2 step 1 reads only the end nodes of the input ell grid
  (assumes log-uniform); the code uses the actual nodes. Identical for a log-uniform grid; correct for any other grid.
* Measured by the reviewer (independent of the implementer's tests): analytic Gaussian C(l) (Weber closed form for mu=0 and 4) reproduced by the
  fast operator to 2e-6 of the peak xi_+ (limited by the cubic interpolation of an extremely steep Gaussian); the faithful (F) route matches
  the spec's own numpy transcription (`docs/spec_checks`) to 6e-10 (xi_+) / 2e-13 of peak (xi_-) in theta and 3e-15 binned; end-to-end
  CAMB(Takahashi)+Limber+`kcap` route vs `data/validation` anchors: max 2.8e-3 (xi_- bin 0 of pair 1-1; Limber-dominated), typically <= 3e-4;
  fast route vs the same anchors differs by the documented node-trapezoid bias (up to 0.67 % in kept bins, 1.2 % in the cut bin 8).
  Fast operator vs brute-force quadrature of a cubic-spline-in-ln-ln interpolant of the real Limber C_l: 1e-6 (xi_+), <= 4e-5 (xi_-, bin 0).
* The l > 5e5 tail exponent of the fast operator is invisible in wide bins (the tail ripple averages out, < 4e-4 even for C ~ l^-1) but is
  visible in 0.1 %-wide bins; `test_fast_operator_upper_tail_slope` now pins it with narrow bins (a mutant with exponent -2 or -3.05 passed
  the previous suite).

* Timing caveat (reviewer): the 8-20 us per call of the fast operator was reproduced (7.7 us with 2 BLAS threads, 10 us with 8, 35-40 us with 1 thread);
  on a machine that is oversubscribed by other multi-threaded jobs the (15 x 400) @ (400 x 18) product can degrade to 0.3 ms with the default
  thread count, so pin BLAS threads (spec D37) when benchmarking. The finite-output check costs ~2 us.

## Measured accuracy (400-node kcap ell grid 0.1..5e5; see `pytest -s tests/test_hankel.py`)
* FFT machinery with exactly-sampled smooth C vs direct quadrature: 1e-8 .. 1e-9 (mu=0 and 4, 1' and 10'); spec s7.2 targets (1e-6, 4e-6) met.
* FFT incl. kcap log-linear interpolation on the 400-node grid vs direct quadrature of the same interpolant: ~1e-6 relative to xi_+
  (3e-4 at 1' for pure power laws l^-1 whose l*C does not decay at the end of the FFT grid -- wrap-around -- irrelevant for real C_l).
* Fast operator is converged in its quadrature to 3e-12 (panel width 1 oscillation, GL-8; halving gives the same).
* Fast operator vs FFT route with *exact* binning, on a cosmic-shear-like spectrum (ln-ln spline through the spec C_l anchors,
  15 pairs): max |dxi/xi| = 3.4e-5 (xi+), 3.8e-5 (xi-); Delta chi2 (KV450 195-pt cuts, analytic covariance) = 3e-8.
* Fast operator vs full kcap route (FFT + node trapezoid): max |dxi/xi| = 7.3e-3 (xi+), 6.7e-3 (xi-) -- this is the documented node-trapezoid
  binning bias of spec s8, not an operator error; Delta chi2 = 1.2e-3 on that spectrum (spec s15.3 quotes 0.04 for the real pipeline output).
* Timing (one core): fast operator 11-18 us per call for all 15 pairs x 2 signs x 9 bins; FFT+trapezoid route 0.5-0.6 ms; operator build 0.15 s.

## Open questions
* None blocking. The anchors in `data/validation` give only four C_l points per pair, so the xi anchors can be compared only at ~20 % (sanity of
  normalisation/units); the precise end-to-end comparison (5e-3) needs the Limber module.
