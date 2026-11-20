# 07 - Tests, benchmarks, performance profile and software quality of kcap

Subsystem scope: `tests/`, `tests/benchmarks/`, `tests/test_pofk/`, `modules/test_*.py`, repository layout, build and dependency state, per-likelihood-evaluation cost structure, and the numerical validation targets that exist in the repository.

Reference checkout: `/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap` (read-only). All paths below are relative to that directory unless absolute.
Method: source reading only. kcap was not built or executed. [verified] An adversarial fact-check pass re-read every cited file and additionally ran a few stand-alone probes (installed CAMB 1.5.8 micro-benchmarks, a CMake 4.2.3 configure test on the top-level CMakeLists, gcc 15 syntax-only compiles of the projection C files); corrections are marked [CORRECTED], additions [verified], and a log is at the end of the file. Where a number is an estimate (not measured) it is marked ESTIMATE. A few small numerical checks (grid arithmetic, a bin-averaging error estimate) were done with stand-alone numpy and are marked CHECK.

Important caveat up front: **the repository does not contain stored numerical reference outputs of the kcap pipeline** (no stored C_ell, xi_pm or P(k) produced by kcap). The "benchmarks" directory holds a plotting script, not data. The only stored numerical arrays that can serve as validation targets are listed in section 1.4 and each has a limitation. A rebuild must generate its own validation targets from independent codes (CAMB, pyccl) using the recipes in 1.3.

---

## 1. Purpose and role in the pipeline; inputs and outputs

### 1.1 What the pipeline does (just enough context for this subsystem)

kcap is a CosmoSIS pipeline for KiDS cosmic shear (KV450 / KiDS-1000 3x2pt). One likelihood evaluation is the chain (module order of the fiducial run, `runs/config/KV450_fiducial.ini:11`):

```
sample_ln_As -> one_parameter_hmcode -> camb -> extrapolate_power -> load_nz ->
source_photoz_bias -> linear_alignment -> projection -> add_intrinsic -> cl2xi ->
derived_parameters -> mini_2pt_like
```

The minimal configuration used by the CCL test is `runs/config/KV450_no_sys.ini:11`:
`sample_ln_As one_parameter_hmcode camb extrapolate_power load_nz projection cl2xi derived_parameters mini_2pt_like` (no IA, no photo-z shifts).

Datablock hand-offs (section / key -> shape, units):

| Step | Writes | Shape / units |
|---|---|---|
| `sample_ln_As` (`utils/sample_ln_As.py:8-10`) | `cosmological_parameters/A_s = exp(ln_1e10_A_s)*1e-10` | scalar |
| `one_parameter_hmcode` (`utils/one_parameter_hmcode.py:11-16` [CORRECTED: was 25-30]) | `halo_model_parameters/eta0 = a_0 + a_1*A` (a_0=0.98, a_1=-0.12 in the ini) | scalar |
| `camb` (`cosmosis-standard-library/boltzmann/pycamb/camb_interface.py`) | `matter_power_lin`, `matter_power_nl` grids `z`(nz), `k_h`(nk) [h/Mpc], `p_k`(nz,nk) [(Mpc/h)^3]; `distances/{z,a,D_A,D_C,D_M,D_L,MU,H,rs_DV,F_AP}` (n=`background_nz`) in Mpc [verified: `H` is CAMB `h_of_z`, i.e. Mpc^-1 not km/s/Mpc (CAMB results.py h_of_z docstring; camb_interface.py:390), and `MU[0]` is NaN at z=0 (line 388)]; `growth_parameters/*`; derived `cosmological_parameters/{sigma_8,S_8,sigma_12,omega_m,...}` | see 3.1 |
| `extrapolate_power` | replaces both power grids with k extended to 500 h/Mpc | nk -> nk+199 |
| `load_nz` | `nz_KV450_5bin/{z, nbin, nz, bin_1..bin_5}` | z has len(file)+1 samples (see QUIRK Q9) |
| `projection` | `shear_cl/{ell(400), nbin, bin_i_j (i>=j)}` C_ell (convergence-like, no ell-prefactor) | dimensionless |
| `cl2xi` (nicaea) | `shear_xi_plus`, `shear_xi_minus` `{theta(1024) [rad], bin_i_j(1024)}` | dimensionless |
| `mini_2pt_like` | `data_vector/<like>_CHI2`, `likelihoods/<like>_LIKE` (+ theory vector if `keep_theory_vector`) | scalar / 195-vector |

### 1.2 What the test suite is meant to do, and what it actually does

README (`README.md:78` [CORRECTED: was :60]) says "run the tests (todo...)". There is no test runner, no CI configuration (no `.github`, no `.travis.yml` anywhere in the repo or in the vendored standard library), no assert-based regression suite for the end-to-end pipeline. Each file is a script with `if __name__ == "__main__"`.

| File | What it checks | Pass/fail criterion | Runnable standalone? |
|---|---|---|---|
| `tests/test_pipeline_against_CCL.py` | kcap `KV450_no_sys.ini` vs `pyccl` (P_lin, P_nl at z=0 for k<5 h/Mpc; C_ell; xi+/-) | **None.** It only prints max fractional differences (lines 67-68, 96-102) and plots with y-limits +-5% (P) and +-10% (C_ell, xi). | Needs a built kcap + pyccl, run from repo root (`PIPELINE_FILE="runs/config/KV450_no_sys.ini"`, line 14). |
| `tests/test_pipeline_against_CCL_and_MP.py` | same, plus MontePython CLASS likelihoods (`K1K_COSEBIs`, `K1K_BandPowers`) | None; plots with dotted lines at 1% and 5% (lines 265-266). | **No.** Hard-coded `PATH_TO_MONTEPYTHON = "/home/fkoehlin/soft/montepython_private/"` (line 16), shells out to MontePython (line 120-124). COSEBIs/BandPowers panels are placeholders: they plot `x/x - 1` (lines 324, 356, "TODO: add also output from CosmoSIS!"). |
| `tests/test_hmcode.py::test_meadcb` | CSL Fortran `meadcb` vs CAMB's built-in `halofit_version="mead"` | `np.allclose(..., rtol=3e-3)` for z<2 and `rtol=1.5e-2` for z>=2 (lines 89-90); the test itself prints "WARNING: Non-linear powerspectrum tolerance is set to 3e-3 for z < 2 and 1.5e-2 for z > 2." (line 91) [CORRECTED: quote]. [verified] The `__main__` block (lines 228-230) runs only `test_camb(plot=True)`; `test_meadcb` is commented out. Config: w=-0.9, A_s=2.1e-9, kmax=20, `nonlinear=pk` for the CAMB reference and `nonlinear=none` + `meadcb` for the Fortran side, default HMCode baryon parameters (no `halo_model_parameters`). | needs compiled `mead_interface.so`; `CSL_PATH` env or `../cosmosis-standard-library` (line 13), i.e. must be run from `tests/`. |
| `tests/test_hmcode.py::test_camb` | kcap's CosmoSIS CAMB interface (+ `one_parameter_hmcode` with a_0=1.0, a_1=-0.5) vs direct CAMB call | `np.allclose` with **default** rtol=1e-5, atol=1e-8 for both linear and nonlinear P (lines 186-187). This is the strictest test in the repo and is the one that pins the CAMB interface. [verified] It does not compare raw nodes: direct-CAMB P is turned into a `RectBivariateSpline` in (z, ln k) of ln P (lines 33-39) and evaluated at the cosmosis grid; because the grids coincide (z = linspace(0,4,100), identical k nodes) this reproduces the nodes to spline round-off. It uses `camb.CAMBparams(WantCls=False)`, which is exactly what the CosmoSIS interface sets in `mode=transfer` (`camb_interface.py:52`), explaining the comment at lines 151-152. The test also fixes w=-0.9, A_s=2.1e-9, do_reionization=F on the CosmoSIS side only (the direct-CAMB side keeps CAMB's default reionization), eta_baryon = 1 - 0.5*A = 0.0 for A=2.0. | needs cosmosis + camb |
| `tests/test_kcap_modules.py` | `utils/camb_background.py` vs CSL pycamb distances; vs old Fortran CAMB (Jan15) | `np.allclose` default tol for D_A vs pycamb (line 69); `rtol=5e-5` vs old CAMB for massless-nu case (line 95; message: "The agreement should be much better than this!") | needs compiled old `camb.so` |
| `tests/test_correlated_priors.py` | `utils/correlated_priors.py`: output = chol(cov) @ p | `assert_almost_equal` (7 decimals). Re-implements the module line for line, so it only guards against typos. Writes `tests/parameter_covariance.txt` into the repo tree (line 18 [CORRECTED: was 111], relative path `tests/...` so must run from repo root). [verified] Imports `dict_to_datablock` from the deprecated `kcap/cosmosis_utils.py` via `sys.path.append("./kcap")`, and its `__main__` calls `test_parameters()` with `plot=True`, so it ends in a blocking `plt.show()`. | yes (cosmosis only) |
| `tests/test_pofk/test_hmcode.py::test_hmcode_no_camb` | cosmosis `meadcb` output (`cosmosis_output/matter_power_nl/*.txt`, **not in the repo**) vs stored CAMB-mead arrays | `np.allclose(..., rtol=1e-2)` (line 97). The `__main__` block calls `test_hmcode(plot=True)` instead; `test_hmcode_no_camb()` is commented out (line 111). [verified] In `test_hmcode()` the `plt.show()` at line 85 sits outside `if plot:`; the plotting branch reads `cosmosis_camb_output/` which is also not in the repo. | **No**: needs a prior run of `hmcode.ini` to create `cosmosis_output/`. |
| `modules/test_likelihoods.py`, `modules/test_module.py` | toy modules (identity-covariance Gaussian likelihood; `sleep(0.001)` module) for exercising CosmoSIS fast/slow sampling | none; they are test fixtures for sampler timing, not tests | yes |
| `tests/benchmarks/KV450_ccl.py` | **Plotting script only.** Reads an existing CosmoSIS `test`-sampler output directory `../../runs/output/KV450/output_no_sys/` (hard-coded relative path, lines 68-91), recomputes CCL curves and saves 4 PDFs. No reference numbers stored. | n/a | needs a previous run |

Conclusion: the validation philosophy is "plot kcap against an independent code and eyeball". Accuracy statements must be taken from the plots' axis limits: P(k) within a few percent (y-range +-5%), C_ell and xi within +-10% (y-range), MontePython C_ell within +-1..5% (dotted guides). No quantitative accuracy number is recorded anywhere in the repo.

### 1.3 Recipes for validation targets that the repo defines (use these cosmologies)

The repo's de-facto validation cosmology (`tests/test_pipeline_against_CCL.py:17-25`, identical in `..._and_MP.py:48-56` and in the MontePython `COSEBIs_Benchmark.param`):

```
omch2 = 0.1, ombh2 = 0.022, h0 = 0.7, n_s = 0.96, ln_1e10_A_s = 3.0 (A_s = e^3 * 1e-10 = 2.0086e-9),
omega_k = 0, w = -1, mnu = 0.06 (one massive nu; [CORRECTED] num_massive_neutrinos is NOT taken from a values file: the test passes `values=values_ini` built from `param_dict` (test_pipeline_against_CCL.py:29-32), which replaces the ini's values file, so CAMB's own default num_massive_neutrinos=1 applies),
halo_model_parameters/A = 2.0   (-> HMCode eta0 = 0.98 - 0.12*2 = 0.74 when mead is used)
n(z): the five files data/KV450/nofz/Nz_DIR_z0.1t0.3.asc ... z0.9t1.2.asc
```

How the CCL comparison is built (formulas you need to reproduce a CCL target):

* `ccl.Cosmology(Omega_c=omega_c, Omega_b=omega_b, Omega_k, h, n_s, A_s, Neff=n_eff (taken from the CAMB output; [CORRECTED] CAMB 1.5.8 gives N_eff=3.044, older CAMB gave 3.046, whereas the MontePython twin hard-codes total N_eff=3.046), m_nu=0.06, w0=w)` (`tests/test_pipeline_against_CCL.py:36-45`). No `wa`. CCL default non-linear model = halofit; the matching kcap ini sets `halofit_version=takahashi` (`KV450_no_sys.ini:136`). **The HMCode parameters in this ini are therefore irrelevant for this comparison.**
* n(z) tracers: `z += (z[1]-z[0])/2` (histogram bin lower edge -> centre), `ccl.WeakLensingTracer(dndz=(z, nofz))` (lines 52-55). Matches kcap's `histogram = True` in `load_nz`.
* CCL C_ell for shear tracers includes the factor `(l-1) l (l+1) (l+2) / (l+1/2)^4` relative to the convergence spectrum. kcap's `shear_cl` does **not** include it. For the xi comparison the test divides it out (line 84): `cl_for_xi /= (l-1)l(l+1)(l+2)/(l+1/2)^4`, evaluated on `ell_for_xi = [2,3,...,99] U logspace(2, log10(ell[-1]), 300)` (line 82), then `ccl.correlation(..., corr_type="L+"/"L-")`. For the direct C_ell comparison (line 95) the factor is *not* divided out, so a deliberate 2.3% offset exists at l=10 (factor = 0.977) and 0.27% at l=30 [CORRECTED: was 0.25%]. CHECK: (9*10*11*12)/(10.5^4) = 0.9774 and 1 - (29*30*31*32)/(30.5^4) = 0.00269. Note the sign: kcap/CCL = 1/0.977 = +2.3% at l=10, not -2.3%.
* Comparison masks: `ell in (10, 10000)`, `theta in (0.05, 10.0)` degrees (lines 89-91); P(k) at z=0 only (`1.0` is the scale factor argument of `ccl.linear_matter_power(cosmo, k*h, 1.0)*h**3`), k < 5 h/Mpc.
* Bug (line 102): the "xi_minus" printout takes `max(abs(frac_diff_xi_plus[...]))`, i.e. repeats xi_plus. The `..._and_MP.py` version (line 191) is correct. [verified] Neither script saves a figure in the plain CCL test (the `savefig` calls are commented out, lines 128/156/179/204), while the `_and_MP` script saves PDFs (lines 218-369) and also prints the line-192 label "theta>0.05 deg" for xi_minus.

### 1.4 Stored numerical data usable as validation targets

| File | Content and exact format | Provenance | Use / limits |
|---|---|---|---|
| `tests/test_pofk/camb_output/k_h.txt` | 223 lines, one float per line, `%.18e`; k in h/Mpc, strictly increasing from 1.0359655789216049e-05 to 154.51060485839844. Non-uniform: 47 pts in [1e-5,1e-3), 50 in [1e-3,0.1), 102 in [0.1,1) (uniform dk = 0.005317 h/Mpc from k=0.1005 up to ~0.62, then opening to dk = 0.032, 0.033, 0.119, 0.197 [verified]), only 11 in [1,10), 13 in [10,200). d ln k between 0.0086 and 0.219; for k>=1 the nodes are exactly log-spaced with d ln k = 0.21901 [verified]. | CAMB native k nodes | the k nodes CAMB returned for that run. [verified] These nodes are NOT reproducible with the installed CAMB 1.5.8: the same call returns 215 nodes ending at k_h = 30.90 (= 1.05*kmax/h = 21.0 Mpc^-1), see 3.1. |
| `.../z.txt` | 100 lines, ascending z = linspace(0, 4, 100) (step 0.040404...) | `np.linspace(0,4,100)` as in `tests/test_pofk/test_hmcode.py:36` | |
| `.../pk_mead.txt` | 100 rows (z index) x 223 columns (k index), whitespace separated, `%.18e`. Non-linear total-matter P(k,z) in (Mpc/h)^3. Row 0 = z=0. Anchors (CHECK, read from file): P(z=0,k=0.00968)=22860.84; P(0,0.1005)=5674.30; P(0,1.00298)=355.118; P(0,8.96295)=9.08861; P(0,99.707)=0.0365878; P(z=1.616,k=1.00298)=35.2375 (row 40). | CAMB HMCode-2016 (`halofit_version='mead'`, `HMCode_A_baryon=2.0`; the script `tests/test_pofk/test_hmcode.py:37` and `values.ini` say `HMCode_eta_baryon=0.603`, **but see the [CORRECTED] note below**), cosmology of `tests/test_pofk/values.ini`: ombh2=0.02228089, omch2=0.1181838, h=0.679565, A_s=2.170229e-9, n_s=0.9679701, tau=0.0739, mnu=0.06, 1 massive nu, w=-1, kmax=20. **No script writes these files**; the generating call is `test_hmcode()` (lines 12-45) but it never saves. CAMB version and `WantCls` setting used are unknown (tests/test_hmcode.py:151-152 states WantCls changes linear P by 2e-3). [CORRECTED, verified by re-running CAMB 1.5.8 with this cosmology, z=linspace(0,4,100), kmax=20, spline-interpolated onto the stored k nodes]: with `HMCode_A_baryon=2.0, HMCode_eta_baryon=0.603` (what the script and values.ini state) CAMB 1.5.8 `mead` differs from the stored array by up to 20% (13-15% at k=1-10 h/Mpc, e.g. P(0,k=1.003)=388.4 vs stored 355.1); with `HMCode_eta_baryon=0.74` (= 0.98 - 0.12*2.0, i.e. the `one_parameter_hmcode` relation) the maximum deviation over all z and k<30 h/Mpc is 0.79% (rms 0.28%, z=0 row max 0.65%; the 0.07% linear-regime offset P(0,0.00968)=22876.7 vs 22860.8 is a CAMB-version effect). So the stored array is a HMCode-2016 result for (A=2.0, eta0=0.74), not (2.0, 0.603); do not use the 0.603 provenance when regenerating targets. | Best available stored target for the non-linear (HMCode) P(k,z) module. Linear P is not stored. Tolerance used by the repo: 1e-2. |
| `data/KiDS1000/Flinc/data_vector/xipm_binned_theory_cut.ascii` | 195 lines, one float per line, `%.18e` (e.g. first 5 values 2.629082258331515e-05, 1.842286426540398e-05, 1.158101799744017e-05, 6.544860273620414e-06, 3.497612027770311e-06). 195 = 15 pairs x (7 xi+ + 6 xi-). Ordering: not documented here (consumed by the cosebis `lib2pcfs_likelihood.so`, `runs/config/Flinc/xipm_Flinc_complextheory_no_noise.ini` sets `theta_min_plus=0.5, theta_max_plus=72.40262468, nTheta_plus=7, theta_min_minus=4.21716333, theta_max_minus=300.0, nTheta_minus=6`). | theory vector for the Flinc mock cosmology, cosmology of `runs/config/Flinc/Flinc_values.ini` (omch2=0.1157, ombh2=0.0225, h=0.6898, n_s=0.969, ln_1e10_A_s=3.086619, A_baryon=3.13, A_IA=0); the code that produced it is not in the repo (n(z) is the Flinc FITS, bin-averaged xi from the COSEBIs C++ code, not `mini_2pt_like`) | Cross-check only for the full chain on Flinc n(z). Not directly comparable to KV450 ini. |
| `data/KV450/data_vector/KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat` | 18 rows x 16 columns, `%.4e`. Column 0 = theta [arcmin] (rows 0-8 = xi+ angles, rows 9-17 = same 9 angles for xi-: 0.78613, 1.5996, 3.2535, 6.623, 13.493, 27.385, 55.541, 112.61, 223.72). Columns 1..15 = the 15 tomographic pairs in "montepython" order (see 2.9 [verified: section added]). Rows 0-8 hold xi+, rows 9-17 hold xi-. | real KV450 measurement (m-corrected) | data, not theory |
| `data/KV450/covariance/cov_analytic_montepython_mcorr.txt` | 270x270 text matrix = 15 pairs x 18 angles, montepython ordering | analytic covariance | likelihood input |
| `data/KV450/systematics/KV450_ALL_c12_treecorr.out` | columns 3,4 (0-based) are the c-term corrections to xi+ and xi- (read at `utils/mini_2pt_like.py:104`). [verified] 9 rows x 10 columns (one row per angular bin: theta [arcmin] 0.7134 ... in col 0, 5.38e-7 / 1.04e-9 in cols 3/4 for the first row); the same 9-vector (cut to 7 / 6 entries) is added to every tomographic pair | | |
| `modules/scale_cuts/test_files/twoPoint_*.fits` | 2pt FITS (CosmoSIS "twopoint" format): `COVMAT` image, tables `xiP/xiM/gT/PeeE/PneE` with columns `BIN1,BIN2,ANGBIN,VALUE,ANG`, and `NZ_SOURCE/NZ_LENS` tables (`Z_LOW,Z_MID,Z_HIGH,BIN1..`). `twoPoint_KV450.fits`: COVMAT 270x270, xiP and xiM 135 rows each (15 pairs x 9 angles), NZ_SOURCE 119 rows. `twoPoint_PneE+PeeE.fits`: COVMAT 200x200, PneE 80 rows (GPF x GEF, lens), PeeE 120 rows, NZ tables 2000 rows. `twoPoint_gT+xiP+xiM.fits`: COVMAT 360x360, gT 90, xiP 135, xiM 135. `twoPoint_fakeCTerm.fits`: `2d_cterm, cterm_cos, cterm_sin` 135 rows each. | test fixtures for `modules/scale_cuts`, used as MontePython BandPowers data path (`tests/..._and_MP.py:24`) | data and format fixtures, not theory |
| `montepython/K1K_*/K1K_*.data.benchmark{,_hmcode}` | MontePython likelihood `.data` configuration files (not numbers). Benchmark settings: `theta_min=0.5, theta_max=300, nbins=5 [verified: number of COSEBIs modes, not tomographic bins], nzbins=5, nzmax=500, type_redshift_interp='slinear', ell_min=0.1, ell_max=5e5, nells=400, k_max_h_by_Mpc=500, method_non_linear_Pk='halofit'` (`benchmark_hmcode` variant: `'hmcode'` and `filename_corrmat_D_z='DIR_cov.asc'`). | | They document the CLASS-side settings that were meant to be numerically comparable with kcap. |
| `montepython/K1K_COSEBIs/INPUT/COSEBIs_Benchmark.param` | benchmark cosmology: omega_cdm=0.1, ln10^10A_s=3.0, omega_b=0.022, n_s=0.96, h=0.7, Omega_k=0, N_eff=2.0328 + 1 ncdm with m=0.06 eV, T_ncdm=0.71611 (so total N_eff 3.046), `k_pivot=0.05`, sBBN Y_p. | | |

The MontePython wrapper writes `Cls_tot.txt` and `<name>_theory.ascii` into the data folder at test time (`tests/..._and_MP.py:130-155`); those outputs are not stored.

---

## 2. Mathematical description (formulas that matter for tests, cost model and quality metrics)

This subsystem owns the validation arithmetic and the glue formulas; the physics modules are described by other reports, but the formulas below are needed to reproduce kcap numbers and to understand where the tests do and do not constrain them.

### 2.1 Limber projection as implemented (the quantity the CCL test checks)

Flat universe, comoving distance chi in Mpc/h (the module multiplies `distances/d_m` by h0, `project_2d.py:541`):

```
C_ij(l) = P_ij * Integral_{chi_lo}^{chi_hi} dchi  W_i(chi) W_j(chi) / chi^2 * P_nl( k = (l + 1/2)/chi , chi )
```

with `P_ij = (1.5 * Omega_m * (100/c)^2)^2`, `c = 299792.4580 km/s` (`project_2d.py:75-79, 134-147`; `limber.c:134-135` for k, `limber.c:171` the (chi/f_K)^2 factor which is 1 for K=0). Units: P in (Mpc/h)^3, k in h/Mpc, chi in Mpc/h, so (100 [h km/s/Mpc] / c [km/s])^2 has units (h/Mpc)^2. The result is dimensionless. This is the convergence (kappa-kappa) spectrum; there is **no** (l+2)!/(l-2)! shear prefactor.

Lensing efficiency (kernel.c:53-60 and 90-156):

```
n(chi) = n(z) |dz/dchi| = -(1+z)^2 n(z) da/dchi            (da/dchi from the Akima derivative of a(chi))
W_i(chi) = N_i * (chi / a(chi)) * Integral_{chi}^{chi_max} dchi' n_i(chi') (chi' - chi)/chi'
N_i = 1 / Integral_0^{chi_max} n_i(chi') dchi'
```

The 1.5 Omega_m H0^2 is **not** in W; it is in `P_ij` (kernel.c comment at line ~88). chi_max = max of the distance table (= chi(z=background_zmax)), `NGLT=1024` (kernel.c:21).

Reduced-kernel trick (kernel.c:60-84, limber.c:151): with growth D(chi) = sqrt(P_nl(k0,z)/P_nl(k0,0)) at the first grid k above `k_growth = 1e-3` h/Mpc (`limber.py:118-127`), `Wr = W D / sqrt(chi)` and the integrand is `Wr_i Wr_j * P /(chi D^2)`. D cancels analytically, giving `W_i W_j P /chi^2` exactly as above. It affects only numerics (it is spline-interpolated), and sets `Wr=0` for chi outside (0, chi of last growth sample). `ext_order=0` is passed (the literal `0` is the 6th argument of `lib.limber_integral`, `limber.py:175-176` [CORRECTED: was 175]), so the extended-Limber branch (limber.c:157-164) is **not used**.

[verified] Further behaviour of this stage that the original text missed:

* **Hard k cutoff / silent zeros.** `ext_limber_integrand` returns 0 whenever the bicubic spline lookup fails (`limber.c:141-146`), i.e. when `k=(l+1/2)/chi` falls outside the P grid (k > 500 h/Mpc after `extrapolate_power`, or k below the CAMB minimum ~1e-5 h/Mpc) or chi is outside the P/kernel grid. For l=5e5 every chi < 1000 Mpc/h is therefore cut, and the integral limits are `chimin = max(kernel minima) = 0`, `chimax = min(kernel maxima) = chi_max` (`limber.c:499-502, 554-555`). MontePython does the same with an explicit `k_max_h_by_Mpc=500` cut. A rebuild must apply the same truncation to match kcap at l >~ 1e4.
* **Curvature is never passed to the Limber stage.** `limber.py:160,169` defaults `K=0.0` and `project_2d.py:169` calls `limber.limber(...)` without K, so `f_K(K,chi)=chi` in `limber.c:135,171` even if `cosmological_parameters/K` exists; only the kernel code (`kernel.c:225`) reads K from the block.
* **W and N kernels are normalised differently.** `W` is divided by the numerically integrated `int n(chi)dchi` (1024-pt GL, `kernel.c:127`); the `N(chi)` kernels used for GI/II (`get_named_nchi_spline`, `kernel.c:177-200`) are only multiplied by `-(1+z)^2 da/dchi` on the n(z) nodes and are not renormalised, so the shear-IA and IA-IA terms inherit the z-space `trapz` normalisation of `load_nz`/`photoz_bias`. The mismatch is a numerical-integration effect, size not measured.
* **n(z) spline range.** `n_of_chi` returns 0 beyond the last n(z) sample (`kernel.c:43-47`, `zmax = z[-1]`); for KV450 that is z=5.975 (see 3.1 nz row), below the distance-table maximum z=6.
* **MontePython normalisation cross-check.** In the MP twin the lensing kernel `g` already carries a factor `2 r (1+z)` (`K1K_COSEBIs/__init__.py:403-419`), so `(9/16) Omega_m^2 * 2^2 = (9/4) Omega_m^2 = (3/2 Omega_m)^2`, the same constant as kcap's `P_ij`.

### 2.2 Hankel transform used by `cl2xi` (nicaea)

```
xi_+(theta) = 1/(2 pi) Integral dl l C(l) J_0(l theta)
xi_-(theta) = 1/(2 pi) Integral dl l C(l) J_4(l theta)
```

Implemented as an FFTLog convolution with bias q=0, mu=0 (xi+) and mu=4 (xi-), norm = 2 pi (maths.c:1918-1956, 1959-1997). Fixed grid, independent of the data: N=1024 points, `l_min=1e-4`, `l_max=5e6` (maths.h:34-35, 38), `dlnl = ln(l_max/l_min)/(N-1) = 0.024081416...`, `lnrc = 0.5 ln(l_max l_min) = 3.1073040...`, `nc = N/2+1 = 513`; sample l_i = exp(lnrc + (i - nc) dlnl) for i=0..1023 (**QUIRK Q3**: first/last l are 9.645e-5 and 4.8226e6, not 1e-4/5e6 because of the i-nc offset; CHECK); output theta_j = 1/l_{N-1-j}, theta in [2.0736e-7, 1.0368e4] rad, log-spaced with the same dlnl, `theta_j = exp(logtheta_min + j dlnl)`, `logtheta_min = (nc-N+1) dlnl - lnrc`. The input spectrum is sampled on that l grid by `lP[i] = l * C(l)` where C(l) is obtained from the 400 projection nodes by **linear interpolation of ln C in ln l** with extrapolation of slope +1 in ln C below `ell_min=0.1` and slope -3 above `ell_max=5e5` (`cl_to_xi_interface.c:105,288-292`; `sm2_interpol` maths.c:1618-1650 and the constants `1.0, -3.0` at line 289). If all C are negative the sign is flipped; if signs are mixed it interpolates C itself linearly in ln l (lines 303-317, 337-339), and [CORRECTED: the original said "no extrapolation"] the extrapolation is then *constant* (C(l<0.1)=C(0.1), C(l>5e5)=C(5e5)), not absent and not an error: `sm2_interpol` guards the "no extrapolation" case with `fabs(lower)<EPS`, but `#define EPS 1.0-5` (maths.c:1617) expands to `1.0-5 = -4.0`, so the guard is never true and `f[0] + 0*(x-a)` / `f[n-1] + 0*(x-b)` is returned. Because the FFTLog input is `l*C(l)` out to l=4.8e6, a flat C above 5e5 is a very different tail from the slope -3 used in the all-positive case. Also: entries with |C|<=1e-30 are replaced by +-1e-30 only if all other entries share one sign (lines 276-285); exact zeros in an otherwise positive spectrum therefore push the pair into the mixed-sign branch (zeros count as non-positive at line 271). Convolution kernel for J_mu, bias q:

```
K(k) = 2^q * Gamma((1+mu+q)/2 + i k/2) / Gamma((1+mu-q)/2 - i k/2) * exp(i k ln 2)      (hankel_kernel_mu, maths.c:2016-2041)
k_m = 2 pi m / (dlnl N),  m = 0..N/2
```

Output `xi[N-1-i] = lP[i] / (t N norm)` with `t = exp((nc - i) dlnl - lnrc)` (maths.c:1995-1998). Complex gamma is recomputed for every bin pair and for both mu (CHECK: 513 x 2 x 2 x 15 = 30,780 `cdgamma` calls per likelihood evaluation).

[verified] Numerical re-implementation of exactly this algorithm in numpy/scipy (same N, l_min, l_max, nc, kernel `2^q Gamma(a1)/Gamma(a2) e^{ik ln2}`, `xi[N-1-i] = lP[i]/(t N 2pi)` with FFTW's unnormalised c2r replaced by `irfft` and the explicit `N` dropped) reproduces the direct integral `(1/2pi) Int l C(l) J_mu(l theta) dl` for a smooth test spectrum to 1e-10 (theta=1', 10') and 8e-7 (100') for xi+ (mu=0) and to <1e-9 for xi- (mu=4). So the `1/(2pi)` normalisation, the order mu=4 for xi-, the output theta = 1/l_{N-1-j} and the factor `l*C` are all right; no 2pi or (2l+1) factor is missing or doubled in the Hankel step. Note kcap's xi is the flat-sky continuous-l integral `l*C(l)/(2pi)`, not the discrete spherical sum `sum (2l+1) C_l d^l_{mm'}(theta)/(4pi)`; the size of that approximation at large theta was not tested in this pass.

### 2.3 Likelihood arithmetic (`utils/mini_2pt_like.py`)

```
chi2 = (x - mu)^T C^{-1} (x - mu),   lnL = -chi2/2   (no log-determinant, no Hartlap factor)
C^{-1} computed once in setup with np.linalg.inv (line 99).
```

[verified] chi2 is evaluated as `np.einsum('i,ij,j', d, inv_cov, d)` (line 240); `like = -0.5*chi2` (line 242); outputs go to `data_vector/<like>_CHI2` and `likelihoods/<like>_LIKE` (lines 244-245). The covariance is the *analytic* KV450 covariance file `cov_analytic_montepython_mcorr.txt` (270x270, symmetric, checked), reduced to 195x195 by `np.delete` of the cut rows/columns in setup (lines 84-95), so the likelihood does not depend on the nuisance parameters through the covariance.

Data vector x: for each pair (i,j) the xi+ and xi- values with angular indices in `cut_xi_plus` / `cut_xi_minus` removed. KV450 fiducial: `cut_xi_plus = 7 8`, `cut_xi_minus = 0 1 2` (ini lines 192-193), so 7 xi+ and 6 xi- per pair -> 15 x 13 = **195** data points; covariance 270x270 reduced to 195x195 (lines 84-95). Multiplicative shear bias correction applied to the data, not the theory (lines 140-141): `x_ij /= (1+m_i)(1+m_j)` with m = (-0.0128, -0.0104, -0.0114, 0.0072, 0.0061) in the fiducial ini (line 197). [verified] The `no_sys` ini comments `m_correction` out (zeros). The data file is named `..._xipm_mcor_5bin.dat` and the covariance `..._mcorr.txt`; whether "mcor" means the file is *already* m-corrected (in which case this division double-counts m) cannot be decided from the repository (see Open Question O9).

Additive c-term (lines 186-191 for order `xi_pm-bin-theta`, 225-230 for `montepython`):

```
xi+_theory += delta_c            (order xi_pm-bin-theta)      <-- QUIRK Q5
xi+_theory += delta_c**2         (order montepython; this is what the fiducial ini uses)
xi+_theory += A_c^2 * xi+_c     ;   xi-_theory += A_c^2 * xi-_c      (if xi_pm_c_file given)
xi- constant offset is commented out in both branches.
```

### 2.4 Angular binning of the theory (`get_theory_point`, lines 11-33)

`angular_binning_mode = integrate` (both fiducial inis). For each of the 9 angular bins with edges (arcmin) `0.5 1.01777898 2.0717481 4.21716333 8.58428037 17.4738002 35.56893304 72.40262468 147.37973879 300.0` (ratio 2.03556 = 600^(1/9)), converted to radians:

```
mask  = (edge_k <= theta_j) & (theta_j < edge_{k+1})        # nicaea grid points only
xi_bin_k = trapz(theta*xi over masked points) / trapz(theta over masked points)
```

So the bin average is weighted by theta (the area measure), uses only the 29 or 30 Hankel nodes inside the bin, and the trapezoid does **not** extend to the bin edges (**QUIRK Q4**). CHECK (stand-alone numpy): for a pure power law xi ~ theta^p the ratio to the exact weighted bin average is `+2.9e-3,-2.3e-4,+3.5e-3,+3.6e-4,+4.1e-3,+9.5e-4,+4.6e-3,+1.5e-3,+5.2e-3` for p=-0.8 (bins 0..8), i.e. a systematic bias of up to ~0.5% that alternates with the node count (29 vs 30 nodes per bin; first node lies 0.7-2.1% above the lower edge). For p=-1.0 the same pattern with up to 0.62%. A rebuild that wants to reproduce kcap digit-for-digit must reproduce this masking; a rebuild that wants to be correct should integrate over the exact edges.

### 2.5 Derived quantities and small formulas

```
A_s       = exp(ln_1e10_A_s) * 1e-10                       (sample_ln_As.py:9)
eta_0     = a_0 + a_1 * A_baryon;  a_0=0.98, a_1=-0.12     (one_parameter_hmcode.py:14 [CORRECTED: was 28]; test_hmcode uses a_0=1.0, a_1=-0.5)
S8        = sigma_8 * (Omega_m/0.3)^alpha, alpha default 0.5 (derived_parameters.py:27-30) [CORRECTED: was 60-63]
S_star    = A_s/2e-9 * (Omega_m/0.3)^alpha                 (derived_parameters.py:36) [CORRECTED: was line 69]
camb S_8  = sigma_8 * sqrt(omega_m/0.3)                    (camb_interface.py:421)
sigma_12  = sigma(R=12 Mpc, no h, z=0)                     (camb_interface.py:418)
```

### 2.6 Extrapolation of P(k) (`boltzmann/extrapolate/extrapolate_power.py`)

Per redshift row: fit a straight line to `(ln k, ln P)` of the last `npoint=3` CAMB nodes (`np.polyfit(...,1)`, line 24 [CORRECTED: was 22]), append `nmax=200` points uniformly spaced in ln k from the last node to ln(500) (the first, equal to the last node, is dropped, so 199 new points; lines 25-27), P = exp(line). Low-k extrapolation is disabled by default (`kmin=1e10` so `xmin < x.min()` is false; line 58 [CORRECTED: was 59]; `nmin` default 50 is therefore unused). [verified] With CAMB 1.5.8, kmax=20 (1/Mpc), h=0.7 the last CAMB node is k_h = 30.0 (CAMB internally uses 1.05*kmax), so the 3-node fit sits at k = 19.4, 24.1, 30.0 h/Mpc (d ln k = 0.219) and the appended grid has d ln k = ln(500/30)/199 = 0.0141 (an order of magnitude finer than CAMB's 0.219 at high k); everything between 30 and 500 h/Mpc is a pure power law in k per redshift. Applies to `matter_power_nl`, `matter_power_lin` (and optional extra sections via the `sections` option, which is empty in the KV450 inis). The grid is rewritten with `block.replace_grid(section, "z", z, "k_h", k, "P_k", P_out.T)` (line 53; key case "P_k" vs "p_k" is normalised by the datablock).

### 2.7 Cost model used in section 7

Counts per likelihood evaluation, fiducial KV450 (5 bins, 400 ell, 150 z):

```
N_pair_auto = 15 ;  N_pair_cross = 25
Limber integrals: shear-shear 15 ; with IA (fiducial) + shear-intrinsic 25 + intrinsic-intrinsic 15 = 55
  (project_2d.py:704-705: autocorrelation spectra loop j<=i; cross spectra loop all j)
kernel-integrand evaluations: bins * (1024 chi nodes) * (1024 GL nodes) = 5 * 1,048,576 = 5.24e6 per W (kernel.c:95-153)
Hankel: 15 pairs * (1 rfft + 2 irfft of length 1024) + 30,780 complex-gamma evaluations
```

[verified] The 30,780 count is 513 frequencies x 2 `cdgamma` per kernel call x 2 kernels (mu=0,4) x 15 pairs (`maths.c:1959-1966, 2026-2027`). Extra per-evaluation set-up not listed above: three `GSLSpline2d` bicubic initialisations of the P grid (150 z x ~350 k) in the fiducial run (one per power type: `matter_power_nl`, `matter_intrinsic_power`, `intrinsic_power`; `limber.py:111-116`), and `get_BAO`/distance vectors over 6000 z nodes.

### 2.8 [verified] How `add_intrinsic` assembles the observable (`cosmosis-standard-library/shear/add_intrinsic/add_intrinsic.py:77-93`)

```
shear_cl_gg[bin_i_j] = shear_cl[bin_i_j]                       (copy of the pure-lensing GG term)
shear_cl[bin_i_j]   += A_i A_j * shear_cl_ii[bin_i_j]            (II)
                       + A_j * shear_cl_gi[bin_i_j]              (GI, kernels W_i N_j)
                       + A_i * shear_cl_gi[bin_j_i]              (IG, transposed)
```

with `perbin=False` (the default and the fiducial setting) giving A_i = 1. The module therefore *overwrites* `shear_cl` in place, and `cl2xi` reads the sum. The IA amplitude lives only inside `P_GI` and `P_II` (section 3.1, IA row), so C_ell = C_GG + A (C_GI + C_IG) + A^2 C_II is an exact polynomial in the amplitude `intrinsic_alignment_parameters/A`. The bk_corrected model has no redshift or luminosity scaling (no (1+z)^eta, no L/L0 factor). The same module lines 94-99 add galaxy-intrinsic terms for position-shear, switched off by `position-shear=False` in the KV450 inis.

### 2.9 [verified] Pair ordering used by the data, covariance and likelihood ("montepython" layout)

Data-file column `1+bin_idx` for the pair (i,j), i>=j (0-based tomographic bins), is `bin_idx = j*(2*n_bin - j + 1)//2 + (i - j)` (`mini_2pt_like.py:126`), i.e. the 15 columns are ordered (0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(4,1),(2,2),(3,2),(4,2),(3,3),(4,3),(4,4) as (higher,lower) = bins (i,j) with the *lower* index as the outer loop. Within each column the first `n_theta=9` rows are xi+ and the next 9 xi-; the covariance uses the same layout, one block of 18 per pair (`idx = pair*2*n_theta + {0, n_theta} + theta_idx`, lines 86-90), whereas the alternative `xi_pm-bin-theta` layout is all xi+ pairs followed by all xi- pairs (lines 74-78). `execute` (lines 203-205) loops `for i: for j in range(i, n_bin)` and reads `data_vectors[j][i]`, which reproduces the same order. The theory datablock keys are `bin_{j+1}_{i+1}` with j>=i (lines 207-210). The cut indices are applied *inside* every pair block, which yields 7 xi+ then 6 xi- per pair (13 per pair, 195 total).

---

## 3. Numerical choices

### 3.1 Grids and tolerances (with sources)

| Item | Value | Source |
|---|---|---|
| CAMB redshifts, fiducial | `z = concat(linspace(0, 2, 100, endpoint=False), linspace(2, 6, 50))`, i.e. 0.02 step to 1.98, then 0.0816 step; 150 total, passed descending | `KV450_fiducial.ini:116-120`, `camb_interface.py:314-324` |
| CAMB redshifts, no_sys | `linspace(0, 6, 150)` step 0.0403 | `KV450_no_sys.ini:130-132` |
| CAMB `kmax` | 20.0 (fiducial, tests) vs 10.0 (no_sys). Units of CAMB `Transfer.kmax` are 1/Mpc, not h/Mpc (camb/model.py:690). Stored benchmark with kmax=20 reaches k_h=154.5 = 105.0 Mpc^-1 (unexplained, see Open Questions). [verified] CAMB 1.5.8 (installed) returns the grid up to `1.05*kmax/h`: kmax=20, h=0.7 -> last k_h=30.00 (194 nodes for the 150-z fiducial request, 10 nodes in [1,10) h/Mpc, max d ln k = 0.2167 for k>=1); kmax=10 -> 15.00 (191 nodes); the test_pofk cosmology (h=0.6796) -> 30.90 (215 nodes with the default `WantCls=True` of `camb.CAMBparams()`, 197 with `WantCls=False`). The 154.5 of the stored file (5.25 x kmax) therefore came from a different CAMB version. Note `set_matter_power` logs a warning when `kmax<20 and zmax>4` (model.py:711-712), which the `no_sys` ini (kmax=10, zmax=6) triggers. | `KV450_fiducial.ini:115`; benchmark file |
| CAMB forced flags | `WantTransfer=True`, `Transfer.high_precision=True` (inside `set_matter_power`, installed camb 1.5.8 model.py:701), `WantScalars=True`, `DoLensing=False`, reionization off (`do_reionization=F`), `lmax=2500` only used through `set_for_lmax` | `camb_interface.py:42-133`, ini |
| Distances | `background_nz = 6000` (fiducial) or 150 (no_sys), z in [0, 6]; 7 vectors + BAO written every call; [verified] `chi_max` for Omega_m=0.25, h=0.7 is 6132 Mpc/h (z=6), so the 1024 kernel nodes are 6.0 Mpc/h apart | `camb_interface.py:378-394` |
| nonlinear model | fiducial: key typo `halofit_verion=mead` (line 124 of `KV450_fiducial.ini`; same typo in `BOSS_KV450.ini`, `Flinc/xipm_Flinc*.ini`). The typo key is ignored, so CAMB's constructor default applies. In the CAMB installed here (1.5.8) `halofit_default = 'mead2020'` (camb/nonlinear.py:24), whose docstring says `A_baryon`/`eta_baryon` are used only by mead2015/mead2016. The 3x2pt production ini spells it correctly: `halofit_version = mead` (`runs/3x2pt/.../pipeline.ini:51`). `KV450_no_sys.ini:136` uses `takahashi`. | see Q1 |
| `extrapolate_power` | `kmax=500`, `npoint=3`, `nmax=200` | ini; script defaults lines 57-61 |
| `projection` | `ell = logspace(log10 0.1, log10 5e5, 400)` (Delta ln l = 0.03866); `limber_rel_tol=1e-3`, `limber_abs_tol=0` | `project_2d.py:412-417`, ini |
| Limber integrator | GSL `qag` GAUSS61 with limit 4096; on failure falls back to 4096-point fixed Gauss-Legendre; one stderr warning per process; OpenMP over ell (`schedule(dynamic)`), per-thread accelerators | `limber.c:15,419-445,528-585` |
| Splines | all 1-D splines are GSL **Akima** (`utils.c:23`), 2-D P is GSL **bicubic** on `(chi, ln k)` of **P itself (not ln P)** (`limber.py:115`); da/dchi is the Akima derivative (piecewise quadratic) | |
| kernel quadrature | 1024-pt fixed Gauss-Legendre inner integral for each of 1024 uniformly spaced chi nodes on [0, chi_max]; normalisation also 1024-pt GL | `kernel.c:21,95-153` |
| P_ij constant | `c = 299792.4580` km/s | `project_2d.py:76,139` |
| Hankel | N=1024, l in [1e-4, 5e6], FFTW `FFTW_ESTIMATE` plans | `maths.h:34-38`, `maths.c:1899-1900` |
| nz histogram | `histogram=True`: `z += dz/2`; if `z[0] > 1e-8` prepend z=0 with n=0; renormalise `trapz(n, z) = 1`. [verified] Each `Nz_DIR_*.asc` has 120 rows (z lower edge 0.000...5.950, step 0.05; raw `sum(n)*0.05 = 1`, raw trapz = 0.9945-0.9998), so after the shift z = 0.025...5.975 plus the prepended 0 -> 121 samples. **The n(z) are not compact**: they are non-zero up to z = 4.45 (bin 1) / 4.65 (bins 2-5); fractions of the distribution above z=1.2 / 2 / 3: bin1 5.2% / 2.6% / 0.36%, bin2 3.2% / 1.8% / 0.58%, bin3 6.5% / 3.1% / 0.80%, bin4 6.6% / 1.7% / 0.39%, bin5 18.1% / 2.0% / 0.27% | `load_nz.py:58-64,67-72,84-86` (real file line numbers) |
| photo-z shift | additive: `n_new(z) = interp(z - bias)`, `interp1d(kind='linear')` in both KV450 inis (module default and 3x2pt production ini: cubic), `fill_value=0`, renormalise with `trapz` | `photoz_bias.py:51-60` |
| IA | `bk_corrected`: `F(z) = -A * C1_RHOCRIT * Omega_m / D(z)`, `P_GI = F P_nl`, `P_II = F^2 P_nl`, `D(z) = sqrt(P_nl(k_min,z)/P_nl(k_min,0))` (D from the smallest k of the grid); `C1_RHOCRIT = 5e-14 h^-2 Msun^-1 Mpc^3 / Msun[kg] * Mpc[m]^3 * rho_crit0` with G=6.67384e-11, Msun=1.9891e30 kg, Mpc=3.0857e22 m; [verified] numerically `C1_RHOCRIT = 0.013873`; `ksmall = argmin(k)` is the first CAMB node (k~1e-5 h/Mpc) and the module needs an exact `z==0` entry in the grid (`np.where(z_nl == 0)`), so `zmin=0` is mandatory | `linear_alignments.py:16-27 (C1) and 79-106 (bridle_king_corrected)` [CORRECTED: was 15-27,67-87]; interface `linear_alignments_interface.py:54-56` |
| MontePython twin | trapezoid on a uniform 500-point z grid (`nzmax=500`), k=(l+1/2)/chi, P set to 0 above k_max=500 h/Mpc, prefactor `(9/16) Omega_m^2 (h/2997.9)^4` (2997.9 vs exact 2997.92458: +3.3e-5 relative error in C_ell; [verified] the missing factor 4 relative to kcap's (3/2 Omega_m)^2 is inside `g`, which carries `2 r (1+z)`, `__init__.py:416`); units are Mpc (not Mpc/h), hence the h^4 | `montepython/K1K_COSEBIs/__init__.py:372-419, 538-560` |

### 3.2 Test tolerances that define "agreement" in this repo

* CAMB interface vs direct CAMB: rtol 1e-5 (default `allclose`), P_lin and P_nl, 100 z x all k. Requires `WantCls=False` in the direct call to match (otherwise 2e-3 difference, per the comment at `tests/test_hmcode.py:151`).
* meadcb Fortran vs CAMB mead: 3e-3 (z<2), 1.5e-2 (z>=2).
* Stored CAMB mead vs cosmosis meadcb: 1e-2.
* Distances: CAMB-python vs `camb_background.py`: default allclose; vs Fortran CAMB 2015: 5e-5.
* CCL/MontePython: visual (+-5%, +-10%).

### 3.3 Defaults versus the fiducial ini (traps)

* `camb_interface.py:106`: default `zmax=3.01` (ini sets 6.0); default `kmax=1.2` (line 129; ini 10 or 20); default `lmax=2600`; `nonlinear` default `both` (transfer modes); `do_reionization` default True (ini F).
* Accuracy knobs are read with CamelCase keys (`AccuracyBoost`, `lSampleBoost`, `lAccuracyBoost`, `DoLateRadTruncation`, line 97). The commented-out ini lines use `accuracy_boost = 2.0`, `high_accuracy_default = T` (`KV450_fiducial.ini:112-113`); those would be silently ignored if uncommented.
* `mini_2pt_like.py:101`: `constant_c_offset` default **True** (the no_sys ini overrides to F; the fiducial sets T).
* `project_2d` default `limber_rel_tol=1e-3`.
* [verified] `camb_interface.py:107`: default `nz=150`; `nonlinear` is not set in the KV450 inis so it is `both` (`NonLinear_both`; the 3x2pt production ini sets `nonlinear = pk`); `neutrino_hierarchy` is not set (CAMB default `degenerate`, one massive eigenstate with mnu=0.06, `num_massive_neutrinos=1` from the values file), while the 3x2pt production ini sets `neutrino_hierarchy = normal`.
* [verified] The 3x2pt production ini (`runs/3x2pt/.../config/pipeline.ini`) differs from the KV450 xi_pm inis in more than the halofit key: `ell_min=1, ell_max=3e4, n_ell=400` (no 5e5), no `cl2xi` (band powers `libbandpower.so` instead), `fast-shear-shear-ia`, `interpolation = cubic`, `output_deltaz = T`, correlated n(z) priors via `correlated_priors.py` + `SOM_cov_multiplied.asc`. The Flinc xi inis use `n_ell = 200` (not 400) and `angular_binning_mode = none` (`xipm_Flinc_complextheory_no_noise.ini:213-214, 259`). The ell_max=5e5 / 400-node setting is therefore specific to the xi_pm Hankel route.

---

## 4. Quirks, approximations, and deviations from the textbook formula

* **Q1 (QUIRK, reproducibility-critical): `halofit_verion` typo.** `runs/config/KV450_fiducial.ini:124`, `BOSS_KV450.ini`, `Flinc/xipm_Flinc.ini`, `xipm_Flinc_complextheory_no_noise.ini`, `..._s_star.ini` (5 files, `grep -rn halofit_verion`). The option is never read (`camb_interface.py:94` looks for `halofit_version`), so these runs used whatever `camb.nonlinear.Halofit()` defaults to in the installed CAMB. That is a moving target: mead2020 in CAMB 1.5.8. The 2022-era default is not determinable from the repo (Open Question O1). The HMCode baryon parameters `A` and `eta0` that the pipeline carefully computes (`one_parameter_hmcode`) only matter for `mead`/`mead2016`/`mead2015`.
* **Q2: convergence-like C_ell.** `shear_cl` is C_kappa-kappa (Limber, k=(l+1/2)/chi) with no (l+2)!/(l-2)! factor and no exact (non-Limber) treatment; xi is built from it directly. CCL's shear C_ell differ by a factor `(l-1)l(l+1)(l+2)/(l+1/2)^4` (0.977 at l=10).
* **Q3: Hankel grid offset.** Hankel nodes run from l=9.645e-5 to 4.8226e6, theta from 2.0736e-7 to 1.0368e4 rad (CHECK), not the nominal 1e-4/5e6 and 3e-7/0.12 constants (`theta_min`, `theta_max` macros in maths.h:31-32 are defined but unused by the code in `tpstat_via_hankel`). Only theta between about 1 arcmin and a few hundred arcmin is numerically valid (module.yaml says so).
* **Q4: bin averaging with edge-excluded trapezoid on the nicaea nodes** (section 2.4): up to ~0.5% bias, alternating.
* **Q5: c-term inconsistency.** `delta_c` enters as `+delta_c` in order `xi_pm-bin-theta` (line 187) but as `+delta_c**2` in order `montepython` (line 226). The fiducial uses `montepython`. A sampler prior `delta_c ~ N(0, 2e-4)` (`KV450_priors.ini`) therefore acts on delta_c^2, producing a tiny positive offset (<= 3.6e-7 over the +-6e-4 prior box, ~4e-8 at the 1-sigma prior width); whether delta_c is meant to be squared is undocumented.
* **Q6: dead computation in `get_theory_point`.** An `InterpolatedUnivariateSpline` is built for every call even in `integrate` mode where it is unused (line 12). 30 spline fits of 1024 points per evaluation.
* **Q7: random draw every evaluation.** With `keep_theory_vector = T` (both fiducial inis) every likelihood call does `np.random.multivariate_normal(mu, cov)` (line 251), an O(n^3) SVD of the 195x195 covariance, and writes the full 195x195 inverse covariance into the datablock (line 250). Besides cost this makes the block contents non-deterministic.
* **Q8: `fast_slow=T` in `KV450_no_sys.ini:16` has no effect:** every varied parameter in `KV450_no_sys_values.ini` is cosmological (feeds CAMB), nothing is fast.
* **Q9: `load_nz` reports `nz` = file length (120) but writes a `z` array of 121 samples** (prepended zero), and `z += dz` modifies the input array in place (a view into the first file's array `data[0]` when several files are given [CORRECTED: was "data_full"; with several files `data_full` is a fresh `np.concatenate` copy, so only the single-file case aliases `data_full`]; harmless either way because `setup` runs once).
* **Q10: Akima spline for n(z) histograms.** Piecewise-constant histogram values at bin centres are interpolated by an Akima spline (`get_named_w_spline`). Different from CCL (CCL interpolates dndz with its own scheme) and from MontePython `slinear`. This is one of the plausible sources of the few-percent differences in the CCL plots.
* **Q11: P is bicubic-interpolated in linear space** over a CAMB k grid that is very coarse at high k (11 nodes between 1 and 10 h/Mpc, d ln k up to 0.22). ESTIMATE: for P ~ k^-2 the cubic-spline relative error is ~ (5/384) h^4 f''''/f = (5/384)(0.22)^4 * 16 ~ 5e-4, below the 1e-3 Limber tolerance but not negligible for xi-.
* **Q12: Limber tolerance 1e-3 relative per C_ell node** with an adaptive integrator => C_ell is only a piecewise-smooth function of the parameters, with noise up to 1e-3 that is irrelevant to nested sampling but harmful for finite-difference derivatives, Fisher matrices or HMC.
* **Q13: curvature.** `camb_interface.py:385` writes `D_M = comoving_radial_distance`, which equals the transverse comoving distance only for omega_k = 0, and nothing writes `cosmological_parameters/K`, which `kernel.c` reads with default 0. The MontePython likelihood README warns "only valid for flat" (`montepython/K1K_COSEBIs/README.md:18`). [verified] Even if a `K` were supplied, the Limber stage would still run with K=0 (section 2.1). Whether the non-flat KiDS-1000 analysis used another path is not visible in this repo (Open Question O4).
* **Q14: cosmology-independent work is redone every call:** FFTLog kernel (complex gamma), 1024x1024 kernel quadrature structure, reduced kernels (per pair, not per bin; `limber.py:173-174` called inside each `limber()` call; `get_reduced_kernel` is also defined twice, lines 130 and 133), ell grid, theta masks, the covariance inverse is the only thing cached.
* **Q15: dead/broken branches in the CAMB interface:** line 183 uses undefined `include_helium_fullreion` (NameError when `use_optical_depth=False` and `do_reionization=True`), line 200 uses undefined `name.de_equation_of_state` (NameError for `use_tabulated_w=True`). Version check at line 44 is a component-wise (not lexicographic) comparison, `not M>1 and not m>0 and not v>9`: it is False for 1.5.8 (correct outcome) but also silently skips the warning for e.g. 0.3.5 [CORRECTED: wording, was "inverted boolean logic"]. CAMB warnings are non-fatal prints.
* **Q16: `tests/test_pipeline_against_CCL.py` prints the wrong quantity for xi_minus** (line 102), and the C_ell panel is not corrected for Q2.

---

## 5. Strengths of this subsystem

* Modular, config-driven pipeline: every physics step is a swappable CosmoSIS module; the same inis drive samplers (MultiNest, PolyChord/dyPolyChord, emcee, metropolis) and the `test` sampler for single evaluations.
* The repository does include independent cross-checks against CCL and an independent CLASS-based re-implementation (MontePython likelihoods, `montepython/K1K_*`), so two Limber codes exist; this is the right idea for validation.
* The strictest test (`test_camb`, rtol 1e-5) pins the CAMB interface to direct CAMB output, including HMCode parameters. Good model for the rebuild's own regression tests.
* Compiled hot paths are in C with OpenMP (Limber, kernel), with a robust integrator fallback (QAG61 -> fixed GL4096) and per-thread GSL workspaces.
* The data products needed to re-run KV450 are bundled (n(z), data vector, covariance, c-terms, m-corrections, 2pt FITS fixtures).
* Parameter handling is thoughtful: sampling in ln(1e10 A_s), S8 and correlated priors (`correlated_priors.py` Cholesky transform) for decorrelated sampling.
* Production-quality settings are visible in `runs/3x2pt/.../pipeline.ini` (proper `halofit_version`, `fast-shear-shear-ia`, `nz=150` + 100 low-z points, 6000 distance nodes), i.e. a record of what the real analysis used.

---

## 6. Weaknesses / risks

### 6.1 Correctness / accuracy
* No automated pass/fail end-to-end test, no stored regression numbers, no CI (section 1.2). Accuracy of the pipeline relative to CCL is only available as plots. The tests cannot detect a 1% regression.
* Silent misconfiguration (Q1) in five shipped configs; unrecognised options are never reported by the CosmoSIS option reader. [CORRECTED] Upstream CosmoSIS 3.12.1 *does* print a non-fatal `**** WARNING: Parameter 'halofit_verion' in the [camb] section never used!` (`cosmosis/runtime/module.py:136-183`, `access_check_report`), so the typo is detectable in a current CosmoSIS if the start-up log is read; whether the 2022 `kcap`-branch fork had this check is unknown. A rebuild should make unknown options a hard error.
* Five-way inconsistent defaults between ini files (KV450_fiducial vs no_sys: kmax 20/10, nz grid, `background_nz` 6000/150, halofit mead(typo)/takahashi, `photoz_bias` interpolation linear/cubic [CORRECTED: the photoz_bias interpolation is `linear` in *both* KV450 inis (and `no_sys` does not even run `source_photoz_bias`); `cubic` is the module default and the 3x2pt production ini value]). The Flinc inis add a further variant (`n_ell=200`).
* Systematics Q4 (up to ~0.5% in binned xi), Q5, Q10, Q11 are not covered by any test.
* The CCL comparison of C_ell is polluted by the known factor of Q2 and the xi_minus printout bug (Q16).

### 6.2 Maintainability / quality
* Single (shallow) git commit visible: `git rev-parse --is-shallow-repository` returns `true`, HEAD `36b40c55394caf26021408e71ff74538bfa3d51e`, dated 2022-06-13 14:45 +0200 ("Use plain 3x2pt as example, instead of joint run with Planck"). **Last maintenance activity ~4.4 years before today (2026-10-25).** No tags, no changelog. README itself marks `kcap/` (Python package) and `montepython/` as deprecated.
* No LICENSE file at the repository root (only third-party licences under `cosmosis-standard-library`: `shear/cl_to_xi_wigner/LICENSE.txt`, `structure/cosmic_emu/LICENSE.md`). Legally "all rights reserved" by default; reuse of code needs clarification with the authors. This is another reason to build a clean-room implementation.
* Vendored code as git subtrees: `cosmosis-standard-library` (123 MB, includes two old Fortran CAMB copies `boltzmann/camb/camb_Jan15` and `boltzmann/camb_Nov13`, CLASS 2.4.1, MGCAMB, ISiTGR, Planck 2018 likelihood code, 48 MB of likelihoods), `cosebis` (37 MB, ~20 MB of `Tplus_minus/*.table` files [CORRECTED: was ~25 MB] and a 6.6 MB `cosebis/modules/besselzero04.h`), 16 MB of Planck data. Repo is 186 MB without `.git`, 1777 files [CORRECTED: 1777 both by `find` and `git ls-files`], the large majority of which (old CAMB/CLASS copies, likelihood code, Planck data, COSEBIs tables) is not used by the cosmic-shear chain.
* Dead and legacy code: `kcap/` package (README: "Deprecated"), `montepython/` copies, `runs/*` generators for specific past analyses, `utils/run_kcap.py` (1387 lines) config generator, `modules/P_gm_approx/pgm_model_old.py`, `COSEBIs` cluster paths such as `/Users/marika_asgary/...` (`cosebis/demos/*.ini`), `/disk09/ma/...`. Python 2 idioms remain (`from builtins import range`, `xrange` in the MP likelihood, `unicode` at `project_2d.py:459`, bundled `enum34.py`, 790 lines).
* Hard-coded or dangling paths: `/home/fkoehlin/soft/montepython_private/` (`tests/test_pipeline_against_CCL_and_MP.py:16`), `PIPELINE_FILE = "runs/config/KV450_no_sys.ini"` (relative, requires repo-root cwd), `../kcap`, `../utils` relative paths in `tests/test_hmcode.py` (requires cwd = `tests/`), inis reference `%(KCAP_PATH)s/examples/kv450_data/...` (no `examples/` directory exists), `./../kcap_boss_module/...` and `%(KCAP_PATH)s/HMx/lib/...` (external repos not included), `runs/*` is git-ignored (`.gitignore` has `runs/*` with exceptions), `.gitignore` ends with a `tests/test_pofk` rule although the files in that directory are tracked (`git ls-files` lists them), so new files there would be silently ignored.
* Hard-coded constants: `c = 299792.4580` (twice), `2997.9` in MontePython, `5e-14` and G/Msun/Mpc values for C1, `k_growth = 1e-3`, `NGLT=1024`, `LIMBER_FIXED_TABLE_SIZE=4096`, `LIMBER_NTHREAD_MAX=128` (hard `exit(1)` if more threads, `limber.c:518-520`), Hankel l/theta macros, `0.3` in S8.
* Documentation: README covers installation only; no docs for module data model, no description of the validation status, a stray reference to a "todo" tests step. `build.py` has an `install_requirements()` that reads a nonexistent `requirements.txt` (line 85), and `check_cosmosis` returns a `COSMOSIS_SRC_DIR` that `build()` discards (line 67).

### 6.3 Portability and build reproducibility (why a fresh clone attempt can fail)
Evidence-based list; each is a hypothesis about the user's failed attempt, ranked by how strongly the repo supports it. (The local machine has Python 3.12.4, numpy 1.26.4, scipy 1.13.1, astropy 6.1.0, camb 1.5.8, **upstream cosmosis 3.12.1**, numba 0.59.1, gcc/gfortran/cmake/mpif90 present, no pyccl.)

1. **Wrong CosmoSIS.** `README.md:9` [CORRECTED: was :11] and `build.py:33` require the Bitbucket fork `https://bitbucket.org/tilmantroester/cosmosis.git@kcap#egg=cosmosis-standalone` (pip-installable, "doesn't rely on environmental variables"; Bitbucket removed Mercurial and has been pruning old repositories; availability not verifiable from here). `build.py:7-27` only checks `cosmosis.__version__ >= 0.0.10`; upstream cosmosis 3.12.1 passes this check, so the fork is **never installed**, and the C/Fortran modules are then built against the wrong layout. Evidence of the mismatch: kcap C sources include `"cosmosis/datablock/c_datablock.h"` (`structure/projection/src/kernel.c:1`, `utils.c:5`) while upstream 3.12.1's `config/compilers.mk:25` gives `-I${COSMOSIS_SRC_DIR}/` with `COSMOSIS_SRC_DIR` = the `cosmosis` package directory, where the header lives as `datablock/c_datablock.h`; kcap's `FindCosmoSIS.cmake` sets `COSMOSIS_DIR=dirname(cosmosis.__file__)` whereas `build.py:29` computes the parent of that directory. The two disagree. [verified] Experiment: `gcc -fsyntax-only -std=c99 -I<site-packages>/cosmosis/ kernel.c` (the include path that `-I${COSMOSIS_SRC_DIR}/` produces with `COSMOSIS_SRC_DIR` = package dir) fails with `fatal error: cosmosis/datablock/c_datablock.h: No such file or directory`; with `-I<site-packages>/` (the parent) the file compiles. The same holds for `limber.c`, `utils.c`, `interp2d.c` (gcc 15.2, `-fopenmp`). `FindCosmoSIS.cmake:53,73` takes `os.path.dirname(cosmosis.__file__)` as `COSMOSIS_DIR`, and `structure/projection/CMakeLists.txt` forwards it as `COSMOSIS_SRC_DIR` to the Makefile, which includes `${COSMOSIS_SRC_DIR}/config/compilers.mk` (that file exists in upstream 3.12.1 `cosmosis/config/`).
1b. **[verified] Modern CMake rejects the top-level project outright.** `CMakeLists.txt:1` (and the CSL and cosebis sub-projects) use `cmake_minimum_required(VERSION 3.2)` plus `cmake_policy(SET CMP0002 OLD)`. With the installed CMake 4.2.3 a minimal file with those two lines fails at configure time with "Compatibility with CMake < 3.5 has been removed from CMake" (suggested workaround `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`). `build.py:71` runs plain `cmake ..` with no such flag, so on any current distribution that ships CMake >= 4 the documented `python build.py` stops before compiling anything. This is a strong candidate for the user's failed clone attempt, independent of the CosmoSIS-fork issue. (Also: `FindCosmoSIS.cmake` uses `find_package(PythonInterp)`, deprecated since CMake 3.12 and gated by policy CMP0148.)
2. **Python-3.12 / NumPy >= 1.24 incompatibilities in the pure-Python code:** `import pkg_resources` in `build.py:1` and `cosmosis-standard-library/setup.py:2` (setuptools is not present in fresh 3.12 venvs, and `pkg_resources` is deprecated), `distutils` in the CSL `setup.py:3-5` (removed in Python 3.12), `np.int` at `structure/projection/limber.py:8` (removed in NumPy 1.24; with the installed NumPy 1.26.4 `project_2d.py` will fail at import), `np.trapz` (deprecated in NumPy 2.0 in favour of `np.trapezoid`; used in `load_nz.py:85`, `photoz_bias.py:60`, `mini_2pt_like.py:22,26` [CORRECTED: the original said "removed"]), `unicode` name at `project_2d.py:459`.
3. **Too much to build:** `cosmosis-standard-library/CMakeLists.txt` builds CAMB Fortran (2015), halofit, halofit_takahashi, cosmic_emu, meadcb, projection, nicaea, **JLA and Planck 2018 (clik)** as dependencies of the `csl` target, plus `cosebis`. For a cosmic-shear chain only `projection`, `cl_to_xi_nicaea` (and `meadcb`) are needed. Any failure in the unrelated parts (clik needs LAPACK/CFITSIO, old Fortran with modern gfortran>=10 argument-mismatch errors; no `-fallow-argument-mismatch` or `-std=legacy` anywhere in the repo, CHECK by grep) breaks `make`. [verified] Local state: gcc/gfortran 15.2.0, `/usr/include/gsl` present, but no `fftw3.h`, so the nicaea module (`shear/cl_to_xi_nicaea/maths.h:10` includes `<fftw3.h>`) would need FFTW from the ExternalProject; `grep` over all 72 Makefile/`.mk`/CMakeLists files finds no `-fallow-argument-mismatch`/`-std=legacy`.
4. **Network fetches of ancient sources:** `BuildGSL.cmake:14` `ftp://ftp.gnu.org/gnu/gsl/gsl-2.4.tar.gz` (FTP, MD5 only), `BuildFFTW.cmake:14` `http://www.fftw.org/fftw-3.3.7.tar.gz`, `BuildCFITSIO.cmake:16` cfitsio 3.450. They are only used if the system library is missing; FTP/HTTP-only URLs often fail behind proxies/WSL.
5. **Unpinned environment:** `conda_env.yaml` pins nothing (`python=3`, numpy, scipy, astropy, `future`, `pyyaml`; pip: `emcee`, `mpi4py`). The README asks for `pip install camb` (any version) and separately for a cluster build of CAMB from `git clone --recursive cmbant/CAMB` + `python setup.py build_cluster` (this command no longer exists in modern CAMB). The pipeline's numbers depend on the CAMB version (HMCode default, `WantCls` behaviour, Q1) yet no version is recorded. Samplers (MultiNest, PolyChord, dyPolyChord) are not in the environment file.
6. **WSL / drvfs:** the checkout lives on `/mnt/d` (NTFS via drvfs: all files appear `rwxrwxrwx`, slow metadata, no real symlinks); autotools/ExternalProject builds of GSL/FFTW and the Fortran make steps on that file system are slow and fragile. A native ext4 build tree is advisable if anything is built.
7. **Absent external repos:** BOSS module (`kcap_boss_module`), HMx, `examples/` and the KiDS-1000 data release are required by several shipped configs but not included, so the "3x2pt" example in `runs/3x2pt` cannot run from this repository alone.

### 6.4 Performance
Dominated by two things that are both cosmology-dependent but structurally wasteful (sections 2.7 and 7): the CAMB non-linear P(k,z) call and the kernel/Limber stage; the Hankel and likelihood stages are small but contain avoidable redundancy. No profile or timing data are stored in the repo; every runtime figure here is an ESTIMATE from operation counts.

---

## 7. Concrete efficiency opportunities

Baseline per-evaluation cost model (ESTIMATE; to be replaced by a profile as the first task of the rebuild, because nothing is measured in the repo):

| Stage | Work (exact counts where given) | Estimated share on one core |
|---|---|---|
| CAMB transfer + HMCode, kmax=20, 150 z, high_precision forced, thermal history recomputed | background + recombination + perturbation evolution over ~0.3-1e3 k modes + HMCode per z | 50-80% (typically 1-3 s with OpenMP; unmeasured). [verified, measured with CAMB 1.5.8 on this 22-logical-core WSL2 box, h=0.7 benchmark cosmology, 150 z as in the fiducial ini, `WantCls=False`, reionization off]: `get_results` (transfer) 0.28 s (22 threads) / 0.55 s (4) / 2.05 s (1); the lazy HMCode `mead` + linear P extraction (`get_nonlinear_matter_power_spectrum` + `get_linear_matter_power_spectrum`, kmax=20) 3.4 s (22 threads) / 0.46 s (4) / 0.95 s (1) -- i.e. **with all 22 OpenMP threads HMCode is ~7x slower than with 4**; Takahashi halofit at kmax=10 (the `no_sys` ini): 0.20 s + 0.18 s (22 thr) / 1.75 s + 0.13 s (1). Totals: mead/kmax=20 ~1.0 s with `OMP_NUM_THREADS=4`, ~3.0 s single-threaded, ~3.7 s at 22 threads; takahashi/kmax=10 ~0.6 s at 4 threads |
| `get_BAO`, 7 distance vectors with 6000 z, sigma8/sigma12/fsigma8 | Python calls into CAMB | 1-3% (tens of ms) [verified, measured: distances 0.007-0.016 s, `get_BAO` 0.01-0.06 s, sigma8/fsigma8/sigmaR <0.01 s] |
| extrapolate_power | 2 sections x 150 z x `np.polyfit` + exp + dstack | ~10-30 ms |
| kernels (5 bins) | 5 x 1,048,576 integrand evals x 3 Akima evals (+ norm) = 5.2e6 integrand evals | 10-25% (~0.5-1.5 s serial; OpenMP over chi nodes only if built with OMP) |
| Limber | no_sys: 15 x 400 = 6,000 adaptive GK61 integrals (each 61-250 evals of 2 Akima + growth + bicubic); fiducial (IA on): 55 x 400 = 22,000 | 5-15% no_sys, 15-35% fiducial |
| Hankel (nicaea) | 15 pairs; 30,780 complex-gamma calls; FFTs of 1024 | 1-3% (~20-60 ms) |
| `mini_2pt_like` | 30 unused spline fits; SVD of 195x195 (`multivariate_normal`); block copy of 195x195 | 1-3% (~5-15 ms) |

Opportunities, ordered by expected payoff per implementation effort:

1. **Fast-slow factorisation of the parameter space (largest sampling-level win, exact).** KV450 fiducial has 14 sampled parameters: 6 slow (omch2, ombh2, h0, n_s, ln_1e10_A_s, halo A_baryon) and 8 fast (A_IA, delta_c, A_c, 5 n(z) shifts) (`runs/config/KV450_values.ini`). Everything after CAMB depends on the cosmology only through P(k,z) and chi(z). Cache `Q(l, chi_j) = P_nl((l+1/2)/chi_j, chi_j)` on a fixed (ell, chi) grid per slow point. Then for any fast point: the kernels change (n(z) shifts), `C_ij(l) = sum_j w_j W_i(chi_j) W_j(chi_j) Q(l,chi_j)/chi_j^2` is an `einsum` (400 x 300 x 15 ~ 1.8e6 MACs, ~1-3 ms), IA enters as an exactly polynomial amplitude `C = C_GG + A C_GI + A^2 C_II` (three cached terms), m-bias and c-terms are analytic. Expected speed-up of a fast-only step: 100-1000x versus the full call; with 8 of 14 parameters fast and a fast-slow sampler (PolyChord's oversampling) the effective CAMB-call count drops by a factor ~2-5 (CosmoSIS `fast_slow` machinery already exists but kcap only turns it on, vacuously, in `KV450_no_sys.ini`). [verified caveat] Stock CosmoSIS fast-slow re-runs every module from the first one that consumes a fast parameter (here `source_photoz_bias` / `linear_alignment`), so with the fiducial module order a 'fast' step still repeats kernels, all 55 Limber integrals, the Hankel stage and the likelihood; the speed-ups quoted here require the restructuring described (cached Q(l, chi) and an einsum), not just switching `fast_slow=T` on. Upstream 3.12.1 also raises an error when a varied parameter is never used (`pipeline.py:211-212`). The 14-parameter split itself checks out against `KV450_values.ini`/`KV450_priors.ini`: flat priors on all 14 (A_IA in [-6, 6]; A_baryon [2.0, 3.13]), plus Gaussian priors delta_c ~ N(0, 2e-4), A_c ~ N(1.01, 0.13) (start 1.13), n(z) shifts ~ N(0, [0.039, 0.023, 0.026, 0.012, 0.011]) (independent here; the 3x2pt ini uses a correlated covariance instead); the A_IA prior is commented out (flat).
2. **Replace the nested kernel integral by a cumulative-sum formula (exact, ~10^3 x on this stage).** `W(chi) = N * (chi/a) * [ A(chi) - chi * B(chi) ]` with `A(chi) = Integral_chi^inf n(chi')dchi'`, `B(chi) = Integral_chi^inf n(chi')/chi' dchi'`. In redshift space use the n(z) nodes directly: `W_i(chi_j) = N_i (chi_j/a_j) sum_k n_i(z_k)(1 - chi_j/chi_k) Delta_z` [CORRECTED: the original dropped the `chi/a` and `N` factors] as a (n_chi x n_z) matrix-vector product with the histogram weights (121 z nodes): 5.2e6 -> 1024 x 121 x 5 = 6.2e5 multiply-adds [CORRECTED: was ~3e4; only ~1.8e5 with 300 chi nodes], still <1 ms, trivially vectorised in numpy/numba; and it removes the Akima-derivative noise by using the analytic `dz/dchi = H(z)/c` from the background. Validate against the kcap kernel to <1e-3.
3. **Restrict the Limber integral to the kernel support and use fixed Gauss-Legendre in z or chi** (no adaptivity). [CORRECTED] The original text claimed that source support ends near z~2 and that ~60% of the chi range is wasted on zero kernels. That is wrong for the shipped n(z): the DIR n(z) files are non-zero out to z=4.45-4.65 (up to 3% of the weight above z=2, 18% of bin 5 above z=1.2; see the 3.1 nz row) and the lensing kernel W(chi) is non-zero for all chi below the last source, i.e. up to chi(z=4.65)=5624 of 6132 Mpc/h (92% of the range; Omega_m=0.25, h=0.7 benchmark cosmology, CAMB 1.5.8); only the last ~8% (z>4.65) is genuinely empty. The kcap Limber range [0, chi_max] is therefore nearly all used; truncating the n(z) at z~2 would change the kernels at the 1e-2 level for the high-z tails and must not be done without a convergence test. With 200-400 GL nodes and a log-P bicubic (or tensor Chebyshev) interpolant, the matrix form above makes each full C_ell set ~1-5 ms: 20-100x versus ~0.5-2 s. Tolerance control: halve the nodes and require |dC|/C < 1e-4 across l in [10, 1e4].
4. **Fold the three IA integrals into one** (kcap already ships this as `FastShearShearIA`/`fast-shear-shear-ia`, `project_2d.py:313-323`, used in the 3x2pt production ini but not in `KV450_fiducial.ini`): `W_eff = W + F(chi) n(chi)/pref`, reducing 55 -> 15 Limber integrals (3.7x on that stage for the fiducial). The factorisation `P_II = F^2 P`, `P_GI = F P` of `bk_corrected` makes this exact.
5. **Cache cosmology-independent Hankel pieces.** Precompute the FFTLog kernels `K_mu(k_m)` (mu = 0, 4) once (they depend only on N and dlnl); batch the 15 pairs into one `rfft`/`irfft` of shape (15,1024); the whole `cl2xi` stage falls from ~tens of ms to <1 ms (30-100x). An even cheaper variant: a precomputed linear operator from C_ell on the l nodes to the 195 binned data points (bin-averaged `J_0`/`J_4` integrals), cost 195 x 400 MACs; this also removes Q4 (exact bin edges) but changes kcap's numbers by up to ~0.5% (so offer both: `mode="kcap"` for bit-faithful comparison and `mode="exact"`).
6. **Reduce the l-grid.** ell_max=5e5 and 400 nodes are chosen so the FFTLog window covers 1e-4..5e6. For theta >= 0.5 arcmin (l ~ 7e3) contributions above l~1e5 are negligible after the Bessel suppression and the C_ell(l) ~ l^-1 fall-off. Using ~120-150 log-spaced nodes in [1, 3e5] with a cubic spline in (ln l, ln C) keeps the interpolation error below ~1e-4 (d^2 lnC/dlnl^2 ~ 0.3 at Delta ln l = 0.13 gives 6e-4 for linear interpolation; cubic is far lower, ESTIMATE) and gives 2.5-3x on projection. [verified arithmetic] 120-150 nodes over [1, 3e5] is Delta ln l = 0.105-0.084, not 0.13 (0.13 corresponds to ~97 nodes); the linear-interpolation error 0.3*Delta^2/8 is then 4e-4 to 2.6e-4. Caveat: the FFTLog needs l*C(l) up to l=4.8e6 and kcap supplies it by a slope -3 extrapolation above l=5e5 (all-positive spectra) -- a rebuild that stops at 3e5 must supply an equivalent tail or its xi at theta<~1 arcmin will differ. The high-k part of P (to k=500 h/Mpc) is then not needed either: `extrapolate_power` (a log-log linear extrapolation of the last 3 CAMB nodes) can be dropped, kmax_effective ~ 30-50 h/Mpc.
7. **CAMB: cut accuracy and redundant output where the error is controlled** (needs measurement in the rebuild): (a) `nz` 150 -> ~40-60 nodes (log-spaced in a or Chebyshev; interpolate in z with a spline in (z, ln k); HMCode cost scales with the number of z); expected 2-3x on the HMCode part; (b) `kmax` 20 (1/Mpc) -> 10 as the no_sys ini already uses (error on xi at theta>=0.5' should be well below 1e-3 because k>10 h/Mpc enters mostly through the extrapolated tail) [verified caveat: CAMB 1.5.8 logs "Using kmax=10 with Halofit non-linear models may give inaccurate results" for kmax<20 with z_max>4, and the no_sys ini uses takahashi not HMCode, so it does not demonstrate that HMCode survives kmax=10; this reduction must be tested against a kmax>=20 reference]; (c) keep `high_precision=True` only if a convergence test shows >1e-3 impact; (d) skip `get_BAO`, `luminosity_distance`, `MU`, `sigma_12`, fsigma8 (not used by cosmic shear): ~1-3%; (e) avoid computing both `matter_power_lin` and extrapolating it when IA is `bk_corrected` (nonlinear only). Compare each reduction against a converged reference (`AccuracyBoost=2`, `kmax=40`, 300 z) with the stored `pk_mead.txt` as one anchor.
8. **Emulator for P_nl(k,z)** (CosmoPower-style or a BACCO/EuclidEmulator2-class model; for baryonic HMCode parameter A, include it as an extra input): ms-level evaluation versus ~1 s CAMB: 100-1000x, with typical emulator errors 0.5-1% at k<10 h/Mpc which must be budgeted against the repo's own 1e-2/3e-3 test tolerances. Keep CAMB as the reference/fallback and validation oracle.
9. **numba / vectorisation of the small stuff:** `mini_2pt_like` (drop `keep_theory_vector` randomness and block copies unless asked; Cholesky once; one matrix-vector `chi2 = |L^-1 d|^2`), `extrapolate_power` (a single vectorised least-squares over all z rows, ~10x), `photoz_bias` (precomputed interpolation matrix for shifts; a linear-interp shift is a sparse matrix), `linear_alignment` (broadcast instead of Python loops).
10. **Parallelism.** Limber and kernel loops are OpenMP across ell / chi nodes; `LIMBER_NTHREAD_MAX=128` exits if exceeded. For a Python rebuild use numba `prange` over ell or over pairs, and MPI at the sampler level (one process per live point), setting `OMP_NUM_THREADS` so CAMB threads x MPI ranks does not oversubscribe. [verified, measured] On the 22-logical-core test machine CAMB's HMCode pass is 7x *slower* with the default 22 threads (3.4 s) than with `OMP_NUM_THREADS=4` (0.46 s) and 3.6x slower than single-threaded (0.95 s), so a rebuild should pin the CAMB thread count (4 here) before anything else; this is the cheapest 3x on the dominant stage.

Combined plausible outcome (ESTIMATE): full cold evaluation 3-6x faster than kcap through CAMB-side trimming + O(N) kernels + folded IA; warm (fast-parameter-only) evaluation of order milliseconds; emulator-backed evaluation ~10 ms total.

Testing improvements to adopt in the rebuild: store regression numbers (C_ell, binned xi, chi2 for the benchmark cosmology of 1.3), assert tolerances, run pyccl comparison with an explicit (l-1)l(l+1)(l+2)/(l+1/2)^4 correction (or comparing in the convergence convention), and pin dependency versions in a lock file with the CAMB version recorded in the output header.

---

## 8. Open questions

* **O1.** What did CAMB's default `Halofit()` resolve to in the 2022 environment where the shipped fiducial inis (Q1 typo) were run (HMCode-2016 or something else)? In CAMB 1.5.8 it is `mead2020`, which ignores the baryon parameters. The KV450/KiDS-1000 papers describe HMCode-2016 with A_baryon; the production 3x2pt ini spells the key correctly (`mead`).
* **O2.** Which CAMB version and which `WantCls` setting produced `tests/test_pofk/camb_output/*`? Why does a request `kmax=20` return k_h up to 154.51 h/Mpc = 105.0 Mpc^-1 (5.25 x kmax)? (Possibly an internal CAMB rule for non-linear models; not derivable from the repo.) [verified, partially resolved] CAMB 1.5.8 does not reproduce that grid (30.90 h/Mpc, 215 or 197 nodes); the *values* are reproduced by CAMB 1.5.8 only for HMCode-2016 with `HMCode_eta_baryon=0.74`, not the 0.603 stated in `test_hmcode.py:37`/`values.ini` (section 1.4). The CAMB version that wrote the file remains unknown; the k-grid difference suggests a CAMB older than the installed one.
* **O3.** Ordering/format of the 195-vector in `data/KiDS1000/Flinc/data_vector/xipm_binned_theory_cut.ascii` and which cosmology/n(z) generated it (the COSEBIs `lib2pcfs_likelihood.so` code is C++ in `cosebis/`; not read here).
* **O4.** How the non-flat ("beyond flat LCDM", Tröster et al. 2021) analysis supplied `cosmological_parameters/K` and transverse distances to `project_2d`, given `camb_interface.py:385` and the absence of a `K` writer; may live in an external config/branch.
* **O5.** Whether the fork `bitbucket.org/tilmantroester/cosmosis@kcap` is still retrievable and how its header layout and `COSMOSIS_SRC_DIR` differ from upstream 3.x (cannot be checked offline). The mismatch hypothesis in 6.3(1) is based on the installed upstream 3.12.1 headers only.
* **O6.** Real runtime of one evaluation for the fiducial KV450 and 3x2pt inis on the authors' hardware (not in repo); needed to calibrate the cost model in section 7.
* **O7.** Whether `NGLT=1024` kernel resolution (Delta chi ~ 5-6 Mpc/h) and the Akima derivative of a(chi) with only 150 background nodes (no_sys ini) produce measurable differences from the 6000-node fiducial; this would show up as a difference between the two inis at the 1e-3 level (not tested anywhere in the repo).
* **O8.** Exact CCL version/settings used when the plots in `tests/benchmarks` were made (CCL changed its default non-linear model and neutrino treatment over versions).
* **O9. [verified] Is the m-correction applied twice?** `mini_2pt_like.py:140-141` divides the data by (1+m_i)(1+m_j) using the five m values of the ini, while the data file is called `KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat` and the covariance `cov_analytic_montepython_mcorr.txt`. The repository contains nothing that says whether "mcor" already includes this correction; resolving it needs the KV450 release notes (Hildebrandt et al. 2020) or the MontePython KV450 likelihood.
* **O10. [verified] Why does the shipped `KV450_fiducial.ini` still contain a `[cosebis]` section pointing to `%(KCAP_PATH)s/examples/kv450_data/...` (no `examples/` directory)?** It is not in the module list, so it is harmless, but it shows that the configuration was cut from a larger repository whose data are not included.
* **O11. [verified] Cause of the HMCode multi-thread slowdown.** The 7x slowdown of CAMB's HMCode pass at 22 OpenMP threads vs 4 (section 7, cost table) is measured, but its cause (oversubscription on WSL2, small per-z work chunks, or hyper-threads) was not isolated.

---

## 9. Fact-check log (adversarial pass)

Method: every cited line/number was re-read in `reference/kcap`; numerical claims were recomputed (node grids, binning bias, stored-array statistics, kernel and Hankel constants) and the installed CAMB 1.5.8, CMake 4.2.3 and gcc 15.2 were used for small probes. Items below are the changes made to sections 1-8 (each also carries an inline [CORRECTED]/[verified] tag).

Corrections to the original text: README line numbers (:60 -> :78, :11 -> :9); `one_parameter_hmcode.py` and `derived_parameters.py` line cites; test_correlated_priors write line (111 -> 18); test_hmcode warning quote; `num_massive_neutrinos` "from the values file" (the CCL test replaces the values file); N_eff 3.046 vs CAMB 1.5.8's 3.044; the l=30 offset (0.25% -> 0.27%); provenance of `pk_mead.txt` (eta 0.603 -> reproduced only by 0.74); `limber.py` ext_order line; mixed-sign branch of the Hankel input (constant, not absent, extrapolation; `EPS` macro bug); `extrapolate_power` line cites (22 -> 24, 59 -> 58); `linear_alignments.py` line cites; `load_nz` aliasing remark; version-check wording; `photoz_bias` interpolation inconsistency claim; file count and COSEBIs table size; `np.trapz` "removed" -> "deprecated"; the claim that KV450 n(z) are compact below z~1.2-2 (they extend to z=4.65) and the dependent "60% wasted chi range"; the O(n_chi n_z) operation count of the cumulative kernel (3e4 -> 6.2e5) and its missing chi/a and N factors; the dangling "see 2.9".

Additions: Limber hard k cutoff and silent zeros; K never passed to Limber; W/N normalisation mismatch; MontePython factor-2 bookkeeping; `add_intrinsic` formula (2.8); pair ordering (2.9); CAMB `kmax` x 1.05 rule and 1.5.8 grid; Mpc^-1 units of `H` and NaN `MU[0]`; measured CAMB timings and the HMCode OpenMP pathology; CMake 4 configure failure; include-path experiment; FFTW header missing locally; production-vs-KV450 ini differences; open questions O9-O11.

Items that could not be resolved from the repository: see O1-O11 (CAMB version behind the stored `pk_mead.txt`; whether the KV450 data are already m-corrected; behaviour of the unavailable CosmoSIS fork; real kcap runtime, since kcap was not built).
