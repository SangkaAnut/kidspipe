# kidspipe - design contract

Pure-Python (numpy/scipy/numba + CAMB) re-implementation of the KV450 cosmic-shear
(xi+/-) inference pipeline. The single source of truth for the MATHEMATICS is
`docs/MASTER_MATH_SPEC.md` (cited "spec sN"); validation material is in `docs/VALIDATION_TARGETS.md`.
Implementers were told NOT to read the kcap source code (<https://github.com/KiDS-WL/kcap>; a local copy `reference/kcap`, not shipped in the repository, is what `docs/study/` and the comparison documents quote) or `docs/study/`
("clean room" in this sense: the implementation was written from the spec); the data files
they need are already copied to `data/`. Developed with Python 3.12, numpy 1.26, scipy 1.13, numba 0.59, camb 1.5.8 (Python 3.10-3.13, numpy 2.x and CAMB 2.x are supported too, see `COMPATIBILITY.md`),
dynesty, nautilus-sampler, pytest. No CosmoSIS, no datablock, no MPI requirement (optional `--mpi` through mpi4py, see USAGE.md "MPI").

## Conventions (global)
* Units: k in h/Mpc, P in (Mpc/h)^3, chi in Mpc/h, theta in radians internally (arcmin at I/O boundaries).
* Redshift arrays ascending. float64 everywhere. No global mutable state except explicit caches.
* Every public function has a docstring giving formula + units; every module has pytest tests in `tests/`.
* Mode switch `fidelity`: `"kcap"` reproduces the spec's (F) behaviour/quirks (for validation) and
  `"fast"` uses the spec's (E) exact/fast replacements. Default `"fast"`.
* Parameter vector order (14 varied) = `PARAM_NAMES`:
  `omch2, ombh2, h, n_s, ln_1e10_A_s, A_baryon, A_IA, delta_c, A_c, dz1, dz2, dz3, dz4, dz5`.
  Fixed: Omega_k=0, w=-1, mnu=0.06 (1 massive nu), N_eff/T_CMB CAMB defaults.

## Package layout (`src/kidspipe/`)
```
params.py            PARAM_NAMES, bounds, priors (flat + truncated Gaussian), log_prior, prior_transform
theory/linear.py     LinearBackend: CAMB wrapper + caching (background, P_lin(z,k), sigma8, Omega_m)
theory/hmcode.py     numba HMcode-2016 ("mead") non-linear P(k,z) from P_lin (stand-alone, no CAMB)
theory/nonlinear.py  get_pnl(lin, A_baryon, backend="hmcode"|"camb") + extrapolate_high_k (spec s2.6)
theory/nz.py         load KV450 n(z), photo-z shift (spec s3)
theory/limber.py     kernels W_i, N_i, NLA IA factor F(z), Limber C_ell for 15 pairs (spec s4-s6)
theory/hankel.py     C_ell -> xi+/- (FFT Hankel, spec s7) and theta binning (spec s8), plus the fast
                     linear-operator form C_ell(grid) -> binned xi
theory/model.py      CosmicShearModel: staged, cached param vector -> 195-vector mu (spec s12)
likelihood/data.py   load data vector / covariance / c-term, ordering, cuts (spec s10)
likelihood/gauss.py  Gaussian chi2 via Cholesky, m-bias, c-term, delta_c (spec s9, s11)
inference/posterior.py  Posterior(model, likelihood): log_post(theta), log_like(theta), counters
inference/maxlike.py    MAP finder (multi-start L-BFGS-B / Nelder-Mead) + Hessian covariance
inference/mcmc.py       adaptive Metropolis with fast/slow parameter blocks; affine-invariant ensemble
inference/nested.py     nautilus / dynesty wrappers using prior_transform
inference/chains.py     weighted-sample utilities: S8, marginal stats, PJ-HPD, writing chains
cli.py               `python -m kidspipe {check,loglike,map,mcmc,nested,bench} ...`
```

## Interfaces (all numpy arrays; deviations must be recorded in `docs/INTERFACE_NOTES.md`)

```python
# params.py
PARAM_NAMES: tuple[str, ...]; BOUNDS: np.ndarray (14,2); START: np.ndarray (14,)
def log_prior(theta) -> float                 # -inf outside bounds; spec s1.3
def prior_transform(u) -> np.ndarray          # unit cube -> theta, exact inverse CDFs
def to_dict(theta) -> dict

# theory/linear.py
@dataclass LinearResult: z(nz,), k_h(nk,), pk_lin(nz,nk), chi(nzb,) [Mpc/h], z_chi(nzb,), omega_m, sigma8_0, h
class LinearBackend:
    def __init__(self, z_grid=None, kmax_per_mpc=20.0, nthreads=None, cache_size=8)
    def compute(self, omch2, ombh2, h, n_s, ln_1e10_A_s, mnu=0.06) -> LinearResult
        # transfer functions cached on (omch2, ombh2, h, n_s); A_s change = exact rescale of pk_lin.

# theory/hmcode.py
def hmcode_pnl(z, k_h, pk_lin, omega_m, omega_nu_frac, h, n_s, A_baryon, eta0, ...) -> pk_nl(nz,nk)

# theory/nonlinear.py
def get_pnl(lin: LinearResult, A_baryon, backend="hmcode") -> (pk_nl)   # eta0 = 0.98 - 0.12 A_baryon
def extend_high_k(k_h, pk, khmax=500.0, nnew=200) -> (k_ext, pk_ext)  # spec s2.6

# theory/nz.py
class NZ: z(nz,), n(nbin,nz) ; def shifted(dz(5,)) -> (nbin,nz)

# theory/limber.py
def cl_shear(lin, pk_nl_ext, k_ext, nz_shifted, z_nz, A_IA, ell, fidelity) -> C(5,5,nell) (i>=j filled symmetrically)

# theory/hankel.py
class HankelBinner: __init__(ell, theta_edges_arcmin, ...); def xi_binned(C_ij_ell) -> (xip(15,9), xim(15,9))

# theory/model.py
class CosmicShearModel:
    def __init__(self, fidelity="fast", pnl_backend="hmcode", nthreads=None)
    def mu(self, theta) -> (195,) PURE binned theory in data ordering (NO delta_c^2 / A_c^2 c terms: GaussianLikelihood adds them,
                                    see docs/interface_notes/params_data_likelihood.md; corrected in the independent review)
    def timing(self) -> dict   # per-stage cumulative seconds
    # caches: stage1 (omch2,ombh2,h,n_s) transfer; stage2 (+ln As, A_baryon) P_nl; stage3 (+dz, A_IA) C_ell.

# likelihood/data.py + gauss.py
class KV450Data: d(195,), cov(195,195), cterm_plus/minus, ordering helpers
class GaussianLikelihood: __init__(data); def chi2(mu, A_c, delta_c) ; def log_like(mu) -> float
```

## Validation ladder (acceptance criteria)
1. Unit: Hankel vs direct quadrature (<=1e-6 rel); Limber vs high-res reference (<=1e-3 / spec s15.4);
   Cholesky chi2 vs explicit inverse (<=1e-10 rel).
2. HMcode (numba) vs `data/validation/pk_mead.txt` (rtol 1e-2 on A=2, eta0=0.74 per VALIDATION_TARGETS s1.2) and
   vs CAMB 'mead' (<=3e-3, z<2).
3. End-to-end vs spec anchors `data/validation/anchors_test_cosmology.json` (rel 5e-3 on C_ell and xi).
4. chi2 on KV450 data at the amplitude optimum ~188.7 (spec s15.2) and ML chi2 <~ 185 after full 14-param MAP.
5. Efficiency: report ms/likelihood evaluation, cold and warm (fast-parameter-only moves), vs the
   unoptimised faithful path; tolerance of the fast path vs the faithful path: |Delta chi2| < 0.1.
