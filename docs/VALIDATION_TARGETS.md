# VALIDATION TARGETS for a clean-room KiDS cosmic-shear pipeline

Scope: every numerical reference file in `reference/kcap` that can be used to validate a re-implementation (paths relative to `reference/kcap`, "[K]"), what each can and cannot validate, the fiducial / best-fit parameter values documented in the repository, and the expected chi2.

**Headline facts**

1. **The repository stores no output of the kcap pipeline** (no C_l, xi+-, P(k) produced by kcap, no chain, no best-fit chi2, no `.fits` model vector) [07 s0, s1.4]. All "tests" are plotting scripts with no assertions, except `tests/test_hmcode.py` and `tests/test_correlated_priors.py` [07 s1.2]. Validation therefore has to combine (a) *inputs and data* shipped here, (b) *two stored numerical arrays* (one P_nl(k,z) array, one 195-point Flinc theory vector), (c) *consistency identities* (tables that must be orthonormal etc.), (d) *independent codes* (CAMB, CCL, MontePython/CLASS) driven with the repo's recipes, and (e) the regression numbers produced in this pass by an independent implementation of `MASTER_MATH_SPEC.md` (section 4).
2. **Expected chi2:** the repo documents none. For the KV450 cosmic-shear data (195 points, 14 parameters) the published maximum-likelihood chi2 is **180.6** (external: Hildebrandt et al. 2020, Table 6; the repo lacks it). My spec implementation gives **188.7** with only the amplitude varied (4 cosmology parameters frozen, no nuisance tuning) and **269.3** for the null model; see section 4 and 5.
3. Items marked **(computed)** were computed in this pass directly from the files; **(spec-impl)** = produced by `docs/spec_checks/kv450_spec_crosscheck.py` (an independent implementation of `MASTER_MATH_SPEC.md`, using python CAMB 1.5.8, NOT kcap output; expect ~1e-3 level differences w.r.t. kcap because it uses linear n(z) interpolation and trapezoid Limber); **(ext)** = taken from published papers, not from the repo, and flagged for verification by the reader.

Index: 1 inventory by validation stage - 2 parameter values documented in the repo - 3 stored numerical references (detail) - 4 regression anchors from the spec implementation - 5 expected chi2 - 6 validation ladder with tolerances - 7 what is missing - 8 hazards when using these files.

---------------------------------------------------------------------------
## 1. Inventory of reference files

### 1.1 KV450 cosmic shear (the target of `MASTER_MATH_SPEC.md`)

| file | format / shape (computed) | content | validates |
|---|---|---|---|
| `data/KV450/data_vector/KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat` | text, 18 x 16, `%.4e` | col 0 theta [arcmin] (pair-weighted means 0.78613, 1.5996, 3.2535, 6.623, 13.493, 27.385, 55.541, 112.61, 223.72); rows 0-8 xi+, rows 9-17 xi-; cols 1..15 = pairs in montepython order | data loader, pair order `col = 1 + i*(2n-i+1)/2 + (j-i)`, cuts, m-division |
| `data/KV450/covariance/cov_analytic_montepython_mcorr.txt` | text, 270 x 270 (sha256-12 f7c111c59487) | analytic covariance, blocks [xi+(9), xi-(9)] per pair; symmetric; min eigenvalue 4.7e-14 | covariance cut (195x195, cond 2.38e5), inversion, chi2 |
| `data/KV450/systematics/KV450_ALL_c12_treecorr.out` | text, 9 x 10, no header | TreeCorr columns R_nom (0.7134..210.27), meanR, meanlogR, xip(c), xim(c), ... ; usecols 3,4 are the c-term amplitudes (xip: 5.38e-7, 4.98e-7, 3.98e-7, 1.92e-7, 2.27e-8, -1.93e-8, 1.07e-8, 4.78e-9, 2.80e-9) | c-term addition |
| `data/KV450/nofz/Nz_DIR_z0.1t0.3.asc` ... `z0.9t1.2.asc` | text, 120 x 2 each | z (lower edge 0..5.95, step 0.05), n(z); raw sum*dz = 1; trapz = 0.9945, 0.9981, 0.9985, 0.9994, 0.9998 | n(z) loading (half-bin shift, prepended zero, trapz renormalisation) |
| `data/KV450/nofz/DIR_cov.asc` | text 5 x 5 | sqrt(diag) = 0.0377, 0.0225, 0.0247, 0.0119, 0.0101 | prior widths (KV450 rounds to 0.039, 0.023, 0.026, 0.012, 0.011) |
| `data/KV450/nofz/SOM_cov.asc`, `SOM_cov_multiplied.asc`, `id_cov.asc` | 5 x 5 | SOM covariance (x4 version: sqrt(diag) 0.0096, 0.0114, 0.0116, 0.0084, 0.0097) | KiDS-1000-style correlated dz prior (K1000 uses its own copy, sqrt(diag) 0.0106, 0.0113, 0.0118, 0.0087, 0.0098) |
| `data/KV450/combined/COSEBIs_En_nModes20.fits` | FITS: COVMAT 300x300, En 300 rows (BIN1,BIN2,ANGBIN,VALUE,ANG), NZ_LENS 2000 rows, NZ_SOURCE 120 rows | E_n data (15 pairs x 20 modes, first value E_1(1,1) = -8.60e-11) and covariance (min eig 2.3e-22) | COSEBIs implementation; the header labels QUANT2='GBF' and unit 'arcmin' for ANG (=mode index) are as shipped |
| `modules/scale_cuts/test_files/twoPoint_KV450.fits` | FITS: COVMAT 270x270, xiP, xiM (135 rows each), NZ_SOURCE 119 rows | xiP/xiM VALUE columns equal the text data file exactly for all 15 pairs (computed [05]); ANG = nominal geometric centres 0.713..210.3 | **data only.** The COVMAT pair blocks do NOT follow its own BIN1/BIN2 order (first two pairs match the text covariance; the rest need a permutation [05 s1.4, Q18]) - do not use it as a chi2 reference |

### 1.2 Stored numerical arrays that act as targets

| file | content | status as a target |
|---|---|---|
| `tests/test_pofk/camb_output/{k_h.txt, z.txt, pk_mead.txt}` | k_h: 223 values, 1.036e-5..154.5 h/Mpc, `%.18e`; z: 100 values linspace(0,4,100); `pk_mead.txt`: 100 x 223, non-linear P(k,z) [(Mpc/h)^3] | **Best stored target for the non-linear P(k).** Cosmology = `tests/test_pofk/values.ini`: ombh2=0.02228089, omch2=0.1181838, h=0.679565, A_s=2.170229e-9, n_s=0.9679701, tau=0.0739, mnu=0.06 (1 massive nu), w=-1, A_baryon=2.0. Anchors read from the file: P(z=0,k=0.00968)=22860.84; P(0,0.1005)=5674.30; P(0,1.00298)=355.118; P(0,8.96295)=9.08861; P(0,99.707)=0.0365878; P(z=1.616,k=1.00298)=35.2375. **Verified in this pass (CAMB 1.5.8):** HMcode-2016 (`mead`) with A=2.0 and **eta=0.74** reproduces the array to max 0.82 % / rms 0.30 % for 1.1e-4<k<30 h/Mpc, all z (with k restricted to CAMB's native range); with the **eta=0.603 stated in `test_hmcode.py:37` and `values.ini` it deviates by up to 20 %**. So the array is HMcode-2016 for (A, eta0) = (2.0, 0.98-0.12*2.0). k >30 h/Mpc cannot be reproduced with CAMB 1.5.8 (it ends at 30.9); the file was written by an older CAMB. Repo tolerance: rtol 1e-2. No linear P stored. |
| `data/KiDS1000/Flinc/data_vector/xipm_binned_theory_cut.ascii` | 195 values, `%.18e`; layout [xi+ for 15 pairs x 7 bins | xi- for 15 pairs x 6 bins] (computed from the monotonic block structure) | Theory vector of the Flinc mock cosmology (`runs/config/Flinc/Flinc_values.ini`: omch2=0.1157, ombh2=0.0225, h=0.6898, n_s=0.969, ln_1e10_A_s=3.086619, A_baryon=3.13, A_IA=0; n(z)=`data/KiDS1000/Flinc/nofz/nofz.fits` ext `NZ_flinc`, 2001 rows). Produced by the COSEBIs C++ `xipm_binned` route, theta bins 0.5'-72.4' (7) and 4.2'-300' (6). First values 2.6291e-5, 1.8423e-5, 1.1581e-5, 6.545e-6, 3.498e-6, 1.891e-6, 1.012e-6 |
| `data/KiDS1000/Flinc/data_vector/xipm_binned_data.ascii` | 195 values, same layout | Flinc "measurement" (first values 8.945e-5, 7.623e-5, 6.511e-6). |
| `data/KiDS1000/Flinc/covariance/cov_{complexMask,complexTheory,simpleMask,theory2}_GG_obs_cut.ascii` | 195 x 195 each, symmetric, positive definite, cond 2.47e5 / 2.44e5 / 4.3e5 / 3.96e5 | four covariance variants for the same data vector | covariance handling |
| `data/KiDS1000/mocks/3x2pt_sample_logA_parameter_covariance.txt` | 20 x 20 | parameter covariance of the 3x2pt sample (20 varied parameters) | proposal covariance for MCMC |

**Computed in this pass:** chi2(data - theory) for the Flinc vectors = **201.4** (complexMask), **209.5** (complexTheory), **235.4** (theory2), **265.2** (simpleMask) for 195 points; chi2(data vs 0) = 1152, 1496, 815, 1122. The near-dof values prove that the three files share one ordering and that the layout above is right; they are targets for a covariance-handling test, not for a physics test.

### 1.3 Mathematical identity tables (exact, no pipeline needed)

| file | what it is | check (computed in [04] and in this pass) |
|---|---|---|
| `cosebis/TLogsRootsAndNorms/Root_0.50-300.00.table`, `Normalization_0.50-300.00.table` | roots r_{n,i} (line n: `n r_1 ... r_{n+1}`, 8 sig. digits) and norms (50 digits); 20 modes | regenerate with `docs/study/04_verification/cosebis_roots_norms_from_scratch.py` (n=1: roots 5.0883097, 6.2059688, norm 0.37596958385); conditions int e^{2z}T = int e^{4z}T = 0; Gram matrix of T_n in dtheta = (theta_max-theta_min) delta_nm (299.5 for 0.5'-300') |
| `cosebis/Tplus_minus/T{plus,minus}<n>_0.50-100.00.table` | T_+n, T_-n on 10^4 log points, theta [arcmin] (first value T_+1(0.5) = 9.10105643) | T_- relation `T_-n = T_+n + int_0^z 4T_+n(y)[e^{2(y-z)}-3e^{4(y-z)}]dy` |
| `cosebis/Benjamins_Bandpower/bandpower_filter_output_0_100.0_140.29.dat` | 1000 x 4: [l, unapodised E, apodised E, apodised B] weights of the first band (100-140.29, theta 0.5'-300', Delta_x=0.5) | `(W_0+W_4)/2` (E) and `(W_0-W_4)/2` (B) from the analytic g and cos^2 apodisation: agreement 1.3e-4 of peak (0.867 at l=126.5); absolute offset ~1.5e-5 at l <= 16 unresolved [04 s2.6] |
| `modules/P_gm_approx/parameter_gridfit_{b2,g2,g3}.dat` | 3 x 6 polynomial coefficients | tabulated N_X(k) values in [06 s3.9] (e.g. N_b2(k=1)=5.112e-2 at n_s=0.97, omch2=0.12, h=0.7) |

### 1.4 KiDS-1000 3x2pt material (`runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/`)

| file | content |
|---|---|
| `data/KiDS/bp_KIDS1000_BlindC_with_m_bias_..._SOM_Fid.fits` | extensions COVMAT 200x200 (STRT_0=0 PneE, STRT_1=80 PeeE; cond 4.6e5), PneE (80 rows = 2 lens x 5 source x 8 bands), PeeE (120 rows = 15 x 8), NZ_LENS (79 rows), NZ_SOURCE (119 rows; SIG_E_1..5 = 0.2696, 0.2579, 0.2726, 0.2539, 0.2703; NGAL_1..5 = 0.5527, 1.0601, 1.6626, 1.1292, 1.1755 arcmin^-2). ANG = geometric band centres 118.442, 166.156, 233.093, 326.994, 458.724, 643.521, 902.764, 1266.443 (label says arcmin, they are multipoles) |
| `data/KiDS/SOM_cov_multiplied.asc` | 5x5 dz covariance (diag 7.6e-5..1.4e-4) |
| `data/BOSS/BOSS.DR12.{lowz,highz}.3xiwedges_{measurements,covmat}.txt` | measurements (32 x 7: s = 2.5..157.5 Mpc/h, 3 wedges x (xi,sigma)); covariance 96x96 wedge-major |
| `config/{pipeline,values,priors}.ini` | the **only generated, committed ground-truth configuration** of the KiDS-1000 3x2pt run (module order, grids, priors, scale cuts, sampler) |
| expected data-vector sizes (computed [05 s2.5]) | PeeE 120 + PneE 22 = **142** (cond 3.3e4); BOSS 2 x 84 = 168 (Hartlap 0.958476) |

### 1.5 Planck material (`data/Planck/COM_Likelihood_Data-baseline_R3.00/plc_3.0/`)
clik directories (`_mdb` metadata, `plik_lite_v22_TTTEEE` lmin/lmax 30/2508, commander 2-29, simall 2-29, lensing `smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8[_CMBmarged]`) plus `_external/bf_lite_plikTTTEEE_v22b_lowl_simall.minimum.theory_cl`, a best-fit theory C_l file that can validate a CMB code's C_l (cosmology not documented in the repo). Not needed for cosmic shear.

### 1.6 Test scripts that define tolerances and recipes (not data, but the repo's acceptance criteria)

| script | criterion in the repo | what it tells you |
|---|---|---|
| `tests/test_hmcode.py::test_camb` | `np.allclose` rtol 1e-5 between kcap's CAMB interface and direct CAMB (WantCls=False, same A/eta; a0=1.0, a1=-0.5, so eta0=0) | strictest test; pins the CAMB interface |
| `tests/test_hmcode.py::test_meadcb` | rtol 3e-3 (z<2), 1.5e-2 (z>=2) for stand-alone `meadcb` vs CAMB `mead`, dark-matter-only | accuracy of the stand-alone HMcode |
| `tests/test_pofk/test_hmcode.py::test_hmcode_no_camb` | rtol 1e-2 vs `pk_mead.txt` | tolerance of the stored array |
| `tests/test_kcap_modules.py` | rtol 1e-5/5e-5 distances vs CAMB | distance table |
| `tests/test_pipeline_against_CCL.py` | **none** (prints max fractional differences; masks 10<l<1e4, 0.05<theta<10 deg; plots +-5 % P, +-10 % C_l/xi) | recipe, below |
| `tests/test_pipeline_against_CCL_and_MP.py` | none; MontePython COSEBIs/BandPowers panels are placeholders | needs private MontePython path |

CCL recipe (works with any version): cosmology (`omch2=0.1, ombh2=0.022, h0=0.7, n_s=0.96, ln_1e10_A_s=3.0, mnu=0.06, omega_k=0, w=-1`), `halo_model_parameters/A=2.0`; the KV450_no_sys ini uses `takahashi` halofit, kmax=10, 150-node uniform z grid, no IA/photo-z, so compare with CCL halofit. CCL's shear C_l contain (l-1)l(l+1)(l+2)/(l+1/2)^4 (0.977 at l=10) which kcap's C_l do not - divide it out [07 s1.3].

---------------------------------------------------------------------------
## 2. Fiducial and best-fit parameter values documented in the repository

### 2.1 Cosmologies used as de-facto references

| use | omch2 | ombh2 | h | n_s | amplitude | other | source |
|---|---|---|---|---|---|---|---|
| CCL/MontePython benchmark and `KV450_no_sys` test | 0.1 | 0.022 | 0.7 | 0.96 | ln_1e10_A_s = 3.0 (A_s = 2.0086e-9) | mnu=0.06 (1 massive), A_baryon=2.0 (eta0=0.74 if mead), omega_k=0, w=-1; MontePython twin: N_eff=3.046 | `tests/test_pipeline_against_CCL.py:17-25`, `montepython/K1K_COSEBIs/INPUT/COSEBIs_Benchmark.param` |
| `tests/test_pofk` (stored P_nl) | 0.1181838 | 0.02228089 | 0.679565 | 0.9679701 | A_s=2.170229e-9, tau=0.0739 | mnu=0.06, A_baryon=2.0 | `tests/test_pofk/values.ini` |
| Flinc mock | 0.1157 | 0.0225 | 0.6898 | 0.969 | ln_1e10_A_s = 3.086619 | A_baryon=3.13, A_IA=0 | `runs/config/Flinc/Flinc_values.ini` |
| KV450 start point (sampler start, not a best fit) | 0.10 | 0.022 | 0.7 | 1.09 | 3.09 | A_baryon 3.0, A_IA 0, delta_c 0, A_c 1.13, dz 0 | `runs/config/KV450_values.ini` |
| KV450 no-sys | same ranges, A_baryon=3.13 fixed, A_IA=0, delta_c=0, A_c=1, dz=0, ln A_s lower bound 1.5 | | `KV450_no_sys_values.ini` |
| KiDS-1000 start (S_8 sampled) | 0.13 | 0.0225 | 0.7 | 0.97 | S_8_input = 0.7458 | A_baryon 2.6, A_IA 0.8, c1=c2=0, Ac=1.13 | `utils/run_kcap.py:850-891`, `runs/3x2pt/.../values.ini` |
| 3x2pt dz prior means (p_i, with bias = L p, L = chol(SOM_cov_multiplied)) | p = (0, -0.18125, -1.11014, -1.39501, 1.26543) -> shifts (0.000, -0.002, -0.013, -0.011, +0.006) (computed with the K1000 matrix [05 s2.7]) | | | | | | `runs/3x2pt/.../priors.ini` |
| 3x2pt bias start | b1 2.1 / 2.3, b2 0.2 / 0.5, gamma3 0.9 / 0.1, a_vir 3.8 / 3.0 | | | | | | `run_kcap.py` |
| BOSS-only / Planck-only inis | see [06 s3.7-3.8] | | | | | | `runs/config/BOSS_*.ini`, `Planck_only*.ini` |
| Planck 5-sigma boxes for joint runs | omch2 [0.11337, 0.127031], ombh2 [0.021616, 0.023104], h [0.642529, 0.715019], n_s [0.934281, 0.986698], S_8_input [0.722929, 0.913414], tau [0.015071, 0.093819], a_planck [0.987908, 1.013081] | | | | | | `runs/3x2pt/3x2pt_configs.py:52-60` |

### 2.2 Best-fit values / chi2

**The repository documents no best-fit parameter vector and no chi2 for any run.** Its only "best-fit"-like content is (i) `utils/maxlike/` (scripts, no outputs), (ii) the MAP *start points* of the 3x2pt maximum-likelihood scripts, which are the 36 highest-posterior samples of a MultiNest chain that is not in the repo [01 s2.4], (iii) the Planck `bf_lite_...theory_cl`.

External numbers that a re-implementation can be compared against, **taken from the papers named in the README, not from the repo - verify before using as acceptance criteria**:

| analysis | quantity | value (ext) |
|---|---|---|
| KV450 cosmic shear xi+- (Hildebrandt et al. 2020, A&A 633, A69; arXiv:1812.06076), posterior means (Table 4) | omch2, ln 10^10 A_s, ombh2, n_s, h | 0.118 (+0.038/-0.066), 3.158 (+1.154/-1.426), 0.022 (+0.003/-0.004), 1.021 (+0.149/-0.141), 0.745 (+0.073/-0.043) |
| | A_IA, B(=A_baryon), delta_c, A_c | 0.981 (+0.694/-0.678), 2.484 (+0.189/-0.475), 0.000 (+-0.0002), 1.022 (+0.129/-0.125) |
| | dz_1..dz_5 | -0.007 (0.034), -0.010 (0.020), 0.013 (0.020), 0.001 (0.012), -0.001 (0.011) |
| | Omega_m, sigma8, **S8** | 0.256 (+0.064/-0.123), 0.836 (+0.132/-0.218), **0.737 (+0.040/-0.036)** |
| | **max-likelihood chi2** (Table 6) | **180.6** for N_data = 195, N_par = 14 (the paper quotes 181 degrees of freedom) |
| | setups of interest in Table 6 | no-baryons S8=0.722, chi2 181.3; no-systematics S8=0.710, chi2 180.8 (5 parameters); all-xip (225 pts) chi2 198.7; IA-Gauss S8=0.740; nu0 S8=0.740 |
| KiDS-1000 cosmic shear (Asgari et al. 2021, A&A 645, A104; arXiv:2007.15633), Table 3 | chi2 / DoF / S8 (MAP + PJ-HPD) | COSEBIs 82.2 / 75-4.5 / 0.759 (+0.024/-0.021); band powers 152.1 / 120-4.5 / 0.760 (+0.016/-0.038); 2PCFs (xi+-) 260.3 / 225-4.5 / 0.764 (+0.018/-0.017) |
| KiDS-1000 3x2pt (Heymans et al. 2021; arXiv:2007.15632) | S8 | 0.766 (+0.020/-0.014) (from memory of the paper; not re-read in this pass) |

---------------------------------------------------------------------------
## 3. Details of the stored arrays worth knowing before using them

* `pk_mead.txt` rows are z-ascending (row 0 = z = 0). k_h.txt is non-uniform: 47 nodes in [1e-5,1e-3), 50 in [1e-3,0.1), 102 in [0.1,1), 11 in [1,10), 13 in [10,200); for k >= 1 the nodes are exactly log-spaced with dln k = 0.21901. When comparing, interpolate **your** P onto these nodes (cubic in ln k, ln P) and restrict to CAMB's own k range (k_min = 5e-5 Mpc^-1 / h); otherwise extrapolation artefacts dominate (observed deviation 567 % when the range is not restricted).
* Flinc vectors: ordering [xi+ pair-major (15 x 7) | xi- pair-major (15 x 6)], i.e. the `xi_pm-bin-theta` layout; the pair order inside is not documented (monotonic increase of amplitude with pair index suggests higher redshift pairs later). The chi2 values in section 1.2 are insensitive to the pair order only if data and theory share it.
* `twoPoint_KV450.fits`: see the warning in 1.1 (covariance pair order).
* All n(z) files use the *lower bin edge* as z. Using them as bin centres moves every bin by 0.025 and is a 0.5 % change in chi that is easy to miss.
* `COSEBIs_En_nModes20.fits` covariance entries are ~1e-20; use scaled linear algebra (it is positive definite, min eig 2.3e-22).

---------------------------------------------------------------------------
## 4. Regression anchors produced by the spec implementation (spec-impl)

These are NOT kcap outputs. They are self-consistent numbers from `docs/spec_checks/kv450_spec_crosscheck.py` (python CAMB 1.5.8, HMcode-2016 `mead` with eta0 = 0.98-0.12 A, 2000 trapezoid nodes in z, nicaea-style FFT Hankel, node-trapezoid binning). Use them to detect gross errors (tolerance guide: 5e-3 relative on C_l and xi; the (E) Limber is converged to 1.5e-4 relative for 10<l<2e4 against a 12000-node reference, the remaining difference to kcap is dominated by linear vs Akima n(z) and QAG 1e-3 noise).

Full numbers: `docs/spec_checks/anchors_test_cosmology.json`.

### 4.1 Test cosmology (omch2=0.1, ombh2=0.022, h=0.7, n_s=0.96, ln_1e10_A_s=3.0, mnu=0.06, A_baryon=2.0, no IA, no photo-z shift)

Background / linear (CAMB 1.5.8): omega_m = 0.250296; sigma8(z=0) = 0.709092; S8 = 0.647692; nk = 194; k_h range [7.1429e-5, 30.0]; chi(z=0.5) = 1345.58 Mpc/h; chi(z=1) = 2384.96 Mpc/h; P_lin(z=0,k=0.1005) = 4233.95; P_nl(z=0,k=1.003): **231.09** (mead, A=2.0), **263.03** (takahashi).

C_l (convergence-like, `bin_i_j`, mead A=2.0 | takahashi):

| pair | l=10 | l=100 | l=1000 | l=10000 |
|---|---|---|---|---|
| 1-1 | 5.0009e-9 \| 5.0055e-9 | 3.6449e-10 \| 3.7260e-10 | 1.5282e-11 \| 1.9198e-11 | 2.0784e-13 \| 3.0896e-13 |
| 3-3 | 1.5618e-8 \| 1.5619e-8 | 1.4974e-9 \| 1.5075e-9 | 6.3246e-11 \| 7.5807e-11 | 1.0244e-12 \| 1.5175e-12 |
| 5-5 | 2.9716e-8 \| 2.9703e-8 | 4.1816e-9 \| 4.2070e-9 | 1.6107e-10 \| 1.8585e-10 | 3.3574e-12 \| 4.8939e-12 |
| 5-1 | 1.0596e-8 \| 1.0599e-8 | 1.0528e-9 \| 1.0636e-9 | 4.2875e-11 \| 5.1422e-11 | 7.3925e-13 \| 1.0861e-12 |

Binned xi (9 bins of section 8 of the spec; pair 1-1, mead A=2.0):
xi+ = 8.8479e-6, 6.4264e-6, 4.3044e-6, 2.6509e-6, 1.5683e-6, 9.0806e-7, 5.0902e-7, 2.5264e-7, 1.0456e-7;
xi- = 7.4397e-7, 8.3471e-7, 8.5505e-7, 7.7560e-7, 6.1326e-7, 4.1991e-7, 2.6495e-7, 1.6581e-7, 1.0512e-7.
(Takahashi pair 1-1: xi+ = 1.1165e-5, 7.7519e-6, 4.9090e-6, 2.8466e-6, 1.6073e-6, 9.0767e-7, 5.0535e-7, 2.5185e-7, 1.0465e-7.)
Pair 5-5 xi+ (mead): 1.0479e-4, 6.9943e-5, 4.3413e-5, 2.6183e-5, 1.6055e-5, 9.3209e-6, 4.7146e-6, 1.8552e-6, 5.2918e-7.

chi2 against the KV450 data at this point (nominal systematics: A_c=1, delta_c=0, dz=0, A_IA=0, m-correction on): **196.06** (mead A=2.0), **192.28** (takahashi).

### 4.2 Amplitude scan and minimum (omch2=0.12, ombh2=0.0222, h=0.70, n_s=0.97, A_baryon=3.13)

| ln_1e10_A_s | 2.60 | 2.80 | 3.00 | 3.20 | 3.40 | 3.60 |
|---|---|---|---|---|---|---|
| sigma8 | 0.6573 | 0.7264 | 0.8028 | 0.8873 | 0.9806 | 1.0837 |
| chi2 | 192.39 | 188.77 | 199.43 | 239.69 | 336.76 | 537.64 |

Minimum chi2 = 188.67 at ln A_s10 = 2.774 (sigma8 0.7172, S8 0.7070). Sensitivities: spec section 15.3.

### 4.3 Reference sanity point for the CAMB stage (reports [02] A, reproduced here)
(omch2, ombh2, H0, n_s, ln10As) = (0.10, 0.022, 70, 1.09, 3.09), mnu=0.06, mead A=3.0 eta0=0.62, kmax=20, 150 z: omega_m = 0.250296, omnuh2 = 6.44867e-4, N_eff = 3.044, sigma8 = 0.779557, nk = 194, k_h in [7.1429e-5, 30.0]; P_nl(z=0,k=1.00298) = 350.03 and P_lin(z=0,k=1.00298) = 69.57 (this pass, linear interpolation in k between CAMB nodes; [02] gives the first six quantities, not these two).

---------------------------------------------------------------------------
## 5. Expected chi2

| situation | chi2 | N | basis |
|---|---|---|---|
| KV450 data, null model | 269.3 | 195 | computed |
| KV450 data, LCDM, **only ln A_s varied** (others frozen), A_IA=0, nominal systematics | **188.7** | 195 | spec-impl |
| KV450 data, published ML fit with 14 free parameters | **180.6** | 195 (dof 181) | ext (Table 6) |
| KV450 "no-systematics" published | 180.8 | 195 (5 params) | ext |
| Flinc data vs shipped Flinc theory, 4 covariances | 201.4 / 209.5 / 235.4 / 265.2 | 195 | computed |
| KiDS-1000 published (not reproducible from the repo: data not included) | xi+-: 260.3 (225 pts); band powers 152.1 (120); COSEBIs 82.2 (75) | | ext |

Acceptance rule for the KV450 clone: with all 14 parameters free, the best chi2 must be 180-182 (the published value), S8 posterior 0.737 +0.040/-0.036; with the amplitude-only scan the minimum must lie within 2 of 188.7 if HMcode-2016 (A=3.13) is used, with the S8 at the minimum 0.707 +-0.005. A model run with `mead2020` or Takahashi shifts S8 by +0.3 % / -0.8 % and chi2 by <0.2 (spec s15.3), so the chi2 alone cannot distinguish non-linear models.

---------------------------------------------------------------------------
## 6. Validation ladder (what to test, in order, with tolerance)

| level | target | reference | tolerance |
|---|---|---|---|
| L0 | data IO: shapes, pair order, cuts -> 195 points; covariance 195x195 positive definite, cond 2.4e5; Flinc chi2 values of 1.2 | files of 1.1, 1.2 | exact shapes; chi2 to 1e-6 |
| L1 | n(z) loading: 121 nodes, z_1=0, z_2=0.025, trapz norm 1 | `Nz_DIR_*.asc` | 1e-12 |
| L2 | background: chi(z), omega_m, omnuh2 | CAMB 1.5.8 values in 4.3 | 1e-6 |
| L3 | linear P and sigma8 | 4.3 (0.779557), `pk_mead` anchors for the linear regime (P(0,0.00968)=22860.8; ~0.07 % CAMB-version offset) | 1e-3 |
| L4 | non-linear P (HMcode-2016): own implementation vs CAMB 'mead' | `tests/test_pofk/camb_output/pk_mead.txt` (A=2.0, eta0=0.74) and `test_hmcode.py` | 1e-2 stored array; 3e-3 (z<2) / 1.5e-2 (z>=2) vs CAMB |
| L5 | high-k extension: 393 nodes, slopes (-2.57 lin; -2.16 nl at z=0) | [02 s3.4] | 1e-3 |
| L6 | kernels: int n dchi, W(0)=0, W(chi_max)=0, mean-z, Z=1 | analytic | 1e-3 |
| L7 | C_l vs CCL (divide CCL by (l-1)l(l+1)(l+2)/(l+1/2)^4) with Takahashi halofit, no IA/photo-z | recipe 1.6 and anchors 4.1 | 1e-2 for 10<l<1e4 (repo plots +-10 %); 5e-3 vs spec-impl |
| L8 | Hankel: direct quadrature of (1/2pi) int l C J_mu | `docs/study/04_verification/nicaea_logfft_hankel_check.py` | 1e-6 at 1'..10' |
| L9 | binned xi: 4.1 anchors; exact vs node-trapezoid bin average | | 5e-3 / 0.7 % (documented bias) |
| L10 | chi2 at the reference points and 4.2 scan | 4.2, 5 | 0.5 absolute near the minimum |
| L11 | free-parameter fit: ML chi2 and S8 | 180.6, 0.737 (+0.040/-0.036) (ext) | chi2 within ~2; S8 within 0.5 sigma (posterior mean depends on sampler and priors) |
| L12 | COSEBIs identity: T tables, E_n from W_n vs from xi | section 1.3 | 1e-7 (tables), 1e-4 (E_n routes) |
| L13 | band-power weights | `Benjamins_Bandpower` file | 2e-3 relative of peak |
| L14 | correlated dz prior: bias = chol(C) p | `tests/test_correlated_priors.py` | 7 decimals |

---------------------------------------------------------------------------
## 7. What is NOT in the repository (cannot be validated against the repo)

* Any stored kcap output (C_l, xi, chain, MAP, chi2, evidence).
* Real KiDS-1000 xi+-, COSEBIs data and covariances other than the blind-C band-power FITS (`Cat_to_Obs_K1000_P1`), `deltaz.asc`, other blinds.
* The BOSS `wedges` model (`../kcap_boss_module`), window and r-band files, HMx and ReACT, `examples/kv450_data/...`, `utils/sample_bsigma8S8.py`, `lib2pcfs_likelihood.so` source, `libcosebis_likelihood.so`.
* CosmoSIS fork (`bitbucket.org/tilmantroester/cosmosis@kcap`), CAMB version used for publication.
* The coefficient fitting data for `parameter_gridfit_*` (P_gm approximation) [06 Q7].
* LICENSE file (no licence at the repo root; another reason to re-implement clean-room) [07 s6.2].

---------------------------------------------------------------------------
## 8. Hazards when using these files

1. `twoPoint_KV450.fits` covariance pair order is inconsistent (1.1).
2. `pk_mead.txt` provenance in the scripts (eta=0.603) is wrong; the array matches eta0=0.74 (1.2).
3. `KV450_fiducial.ini` silently ignores `halofit_verion`; a CAMB run with the default model is mead2020 and does not use A_baryon (spec D1). Do not generate targets with an un-pinned CAMB.
4. CCL comparisons need the shear/convergence factor and the same halofit.
5. `data/KV450/nofz/*.asc` are lower-edge histograms.
6. m-bias: the data file name contains "mcor"; kcap divides by (1+m)(1+m) again (possible double correction, open question); the effect is +-0.2 % in S8.
7. The 3-number parameter-file grammar is `lo start hi` although CosmoSIS docstrings say otherwise.
8. `tests/test_pofk/camb_output/k_h.txt` extends to 154.5 h/Mpc, which an up-to-date CAMB does not return for kmax=20 (it stops at 30.9).
