# 06. The 3x2pt, galaxy-bias, BOSS, Planck and extended-cosmology subsystem of kcap

Scope: `modules/P_gm_approx/`, `utils/mini_BOSS_like.py`, `utils/galaxy_bias.py`, `utils/cobaya_planck_lensing.py`,
`runs/3x2pt/`, `runs/extended_cosmologies/`, `runs/config/BOSS_*.ini`, `runs/config/Planck_only*.ini`,
`cosmosis-standard-library/bias/`, and the parts of `utils/run_kcap.py`, `project_2d`, `scale_cuts`, `add_intrinsic`,
`add_magnification`, `bandpower`, `camb_interface` and `planck_interface` that decide what the 3x2pt data vector is.

All paths are relative to `/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap` (read-only reference). Line numbers were taken from that tree.
Nothing was executed from kcap. The only numbers computed by me are the small sanity tables in section 3.9, which I produced with a
fresh numpy re-implementation of the formulas in section 2.

**Adversarial fact-check pass (second reader).** Every formula, constant, path and line citation below was re-read against the source. Corrections are made
inline and tagged "[corrected]"; additions are tagged "[verified]"; section 9 is a log of all changes plus items that could not be resolved. In this pass
nothing from kcap was executed, but two kinds of independent checks were run: (i) standalone CAMB 1.5.8 calls that replicate the settings in
`camb_interface.py` / `sigma8toAs.py` (CAMB is a third-party library, not kcap code), and (ii) numpy/astropy reads of the data files that ARE checked in under
`runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/data/` (see 1.1).

Conventions in this document:
- "QUIRK" marks a deviation from the textbook formula or a surprising implementation choice.
- "NOT IN REPO" marks something the subsystem needs that is not in this checkout, so it cannot be specified from source.
- "INFERRED" marks a statement derived from reading code or configs without seeing the data file or the external module.

---------------------------------------------------------------------------------------------------

## 1. Purpose, role, inputs and outputs

### 1.1 What "3x2pt" means in kcap (important, it is not the usual DES-style 3x2pt)

In the KiDS-1000 setup (`utils/run_kcap.py`, run types in `K1000Pipeline.pipelines`, lines 110-172) the three probes are:

1. Cosmic shear: E-mode band powers `PeeE` of the KiDS-1000 source sample (5 tomographic bins, 15 bin pairs).
2. Galaxy-galaxy lensing (GGL): E-mode band powers `PneE` between 2 BOSS lens bins (low-z, high-z) and the 5 KiDS source bins.
3. Clustering: NOT angular w(theta). It is the BOSS DR12 configuration-space redshift-space clustering wedges xi_w(s), 3 wedges, 2 redshift bins
   (z_eff = 0.38 and 0.61), modelled by an external module (`wedges`) that is NOT IN REPO. (The BOSS DATA and COVARIANCE files themselves ARE in the checkout, see the
   [verified] data-inventory paragraph after the likelihood-structure list below.)

The run-type names map as: `EE` = cosmic shear only, `nE` = GGL only, `w` = BOSS wedges only, `EE_nE` = shear + GGL, `EE_nE_w` = full 3x2pt,
`EE_w`, `nE_w` = pairs, `xipm`, `cosebis` = alternative shear statistics, `EE_fR` = f(R) modified-gravity shear-only (run_kcap.py:110-172).

Likelihood structure (INFERRED from code): the total log-likelihood is a plain sum of independent terms,
- `2x2pt_like` (utils/mini_like.py): Gaussian chi2 of the concatenated PeeE+PneE vector with the full KiDS covariance (shear and GGL are correlated
  inside this one covariance, which lives in the KiDS fits file) [corrected: the earlier "NOT IN REPO" is wrong; the fits file is checked in, see below],
- `xi_wedges_like` (utils/mini_BOSS_like.py): two independent Gaussian chi2 terms, one per BOSS redshift bin,
- optionally `PLANCK2018` (and `planck_lensing_like`, `pantheon_like`).
There is no covariance between KiDS lensing and BOSS wedges, and none between KiDS and Planck. The only coupling is the shared cosmological
parameters (and, for BOSS and GGL, the shared bias parameters `b1, b2, gamma3` per lens bin).

[verified] Likelihood list: `set_sampling_config` builds `likelihoods` from the `like_name` of every module left in the pipeline (run_kcap.py:776), so with Planck
enabled only `PLANCK2018` is summed even though `save_separate_likelihoods=True` also writes `PLANCK18_LIKE_n` into the block (those extra keys are not in the
list and are not double counted). The checked-in generated ini has `likelihoods = xi_wedges_like 2x2pt_like`.

[verified] Data inventory actually present in the checkout (the earlier text said all of this was external):
- `runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/{config,data}`: a generated `pipeline.ini`, `values.ini`, `priors.ini` for the production
  blind-C EE_nE_w run (this is the ground truth for what `run_kcap.py` emitted, and its module list matches 1.2 exactly), the KiDS-1000 blind-C band-power fits file,
  `SOM_cov_multiplied.asc` (5x5 n(z) covariance, diagonal about 7.6e-5..1.4e-4) and the four BOSS Sanchez et al. text files.
- KiDS fits file (astropy read): extensions `COVMAT` (200x200 float64, symmetric, header STRT_0=0 NAME_0=PneE, STRT_1=80 NAME_1=PeeE), `PneE` (80 rows = 2 lens x 5 source x 8 bands,
  QUANT1=GPF, QUANT2=GEF, KERNEL_1=NZ_LENS, KERNEL_2=NZ_SOURCE), `PeeE` (120 rows = 15 pairs x 8 bands), `NZ_LENS` (79 rows, dz=0.01, columns Z_LOW Z_MID Z_HIGH BIN1 BIN2),
  `NZ_SOURCE` (119 rows, dz=0.05, 5 bins, last Z_HIGH = 5.95, i.e. last Z_MID = 5.925). The ANG column of PneE and PeeE is [118.44199, 166.15641, 233.09262, 326.99411, 458.72386, 643.52102, 902.76383, 1266.44273]
  (TUNIT arcmin is just the generic label; these are multipoles) = exactly the geometric band centres of 2.5. Pair order in the file is lens-major for PneE
  ((1,1)..(1,5),(2,1)..(2,5)) and (1,1),(1,2)..(1,5),(2,2).. for PeeE, with the 8 bands contiguous, which is the order `scale_cuts.execute` builds the theory vector in.
- BOSS files: `BOSS.DR12.{lowz,highz}.3xiwedges_measurements.txt` have 32 data rows (s = 2.5, 7.5, ..., 157.5 Mpc/h, i.e. 5 Mpc/h bins) x 7 columns; the covariance files are 96x96 = 3 wedges x 32 s-bins,
  symmetric, wedge-major (sqrt(diag) reshaped to (3,32) matches the quoted sigma columns to 0.1 per cent for lowz and to 0.9 per cent for highz).
- Also in the checkout: `data/KV450/nofz/*` (the older KV450 DIR n(z) files and `DIR_cov.asc`, the run_kcap.py default for `--dz-covariance-file`), `data/BOSS/nofz/nOfZ_hist_BOSSA_tomo{0,1}.dat` (default `--lens-nz-files`),
  and `data/Planck/COM_Likelihood_Data-baseline_R3.00/plc_3.0/{low_l,hi_l,lensing}` (clik directories; the `_mdb` files give lmin/lmax 2/29 for commander and simall, 30/2508 for plik_lite_v22_TTTEEE).
- Still NOT in the checkout: the `wedges` module (`../kcap_boss_module`), `BOSS.DR12_windows.txt`, `BOSS.DR12_rbands.txt`, `../Cat_to_Obs_K1000_P1`, `deltaz.asc`, and the external packages (`plancklensing`, ReACT, HMx).
- The highz BOSS file header says "for 0.4 < z < 0.6" (lowz: "0.2 < z < 0.5"), whereas run_kcap.py sets `z_eff = 0.38 0.61` and the lens n(z) bin 2 spans 0.5-0.75. I could not resolve from the repo
  whether this header is just a stale label or whether the file is the Sanchez et al. middle bin; the pipeline simply trusts the file order.
- Lens n(z) (computed from `NZ_LENS`): bin 1 non-zero for Z_MID in [0.205, 0.495], trapz norm 1.0005, mean z 0.385; bin 2 non-zero for Z_MID in [0.505, 0.745], norm 1.0001, mean z 0.586.
  The two bins are sharp-edged histograms that meet exactly at z = 0.5 = `z_sep`.
- Shipped driver scripts are experiment logs, not a single recipe: `runs/3x2pt/3x2pt_configs.py` has `blinds = []` and `run_types = ["EE","EE_nE","EE_w"]` in its first loop (so it generates nothing, and never EE_nE_w),
  and its second loop is hard-wired to run type EE with the `fix_As_ns` config; `runs/extended_cosmologies/3x2pt_ext_cosmo_configs.py` has `run_types = ["w"]` and `configs = nuLCDM_configs`,
  i.e. as shipped it only generates BOSS-wedge-only nuLCDM chains. The README points to `runs/3x2pt/data/cosmology/multinest_blindA_EE_nE_w_Planck`, which is not in this checkout.

### 1.2 Module chain for the full 3x2pt run (EE_nE_w), in execution order

The module order is the insertion order of the dict built in `create_config` (run_kcap.py:417-771 [corrected from 406-770; `create_config` itself starts at 358], joined at line 777). Modules that appear in
`default_config_cuts` (97-108) and are not re-enabled are removed. For `EE_nE_w` the effective chain is:

`sample_S8 -> sigma8toAs -> correlated_dz_priors -> one_parameter_hmcode -> camb -> wedges -> approximate_P_gm -> extrapolate_power -> load_nz_fits ->
source_photoz_bias -> linear_alignment -> magnification_alphas -> projection -> add_intrinsic -> add_magnification ->
bandpower_shear_e -> bandpower_ggl -> scale_cuts -> BOSS_like -> 2x2pt_like`.

(`sigma8toAs`, dict line 430, runs before CAMB: it performs its own small CAMB run to convert sigma_8 into A_s.) `consistency` is NOT in the
kcap-generated pipeline (no dict entry), which matters in section 4 (Q1). `planck_like` / `planck_lensing_like` are cut by default and are re-enabled
with `--enable-modules` (run_kcap.py:97-104, 991).

[verified] The chain above is identical to `modules =` in the checked-in generated ini `runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline.ini`.
That ini also shows run-time values that differ from today's code defaults: `live_points = 1000`, `update_interval = 100` (code: 250 default / 500 in the driver script, and 20), `efficiency 0.3`,
`tolerance 1.0e-2`, `wedges timeout = 600.0`, `omch2 = 0.051 0.13 0.2` (BOSS cap), `nofz_shifts p_1..p_5` start values 0.0, -0.18125, -1.11014, -1.39501, 1.26543 with matching Gaussian priors of width 1,
`shear_c_bias c1 = c2 = 0.0, Ac = 1.13` (fixed) and no `[consistency]`/`[planck]` section.

[verified] Other run types (code, run_kcap.py:110-172). The `w` (BOSS-only) type cuts only `bandpower_shear_e`, the ggl modules, `extrapolate_power, load_nz_fits, source_photoz_bias, linear_alignment, projection,
scale_cuts, 2x2pt_like`; it does NOT cut `correlated_dz_priors` or `one_parameter_hmcode`, so its chain is `sample_S8 -> sigma8toAs -> correlated_dz_priors -> one_parameter_hmcode -> camb -> wedges -> BOSS_like`
[corrected: section 2.10's last paragraph listed it without the two cheap modules]. `nE` keeps `approximate_P_gm`, magnification and IA but cuts the wedges and cosmic-shear modules; `EE` cuts all ggl modules and
fixes the bias parameters; `nE_magnification` and `EE_nE_w_magnification` merely re-enable modules that `nE` / `EE_nE_w` already contain (their "uncut" lists are redundant).
The older hand-written inis (`BOSS_only.ini`, `BOSS_KV450.ini`, `Planck_only.ini`) DO contain a `consistency` module, unlike anything run_kcap.py generates.

### 1.3 Inputs and outputs per component (datablock section/key, shape, unit)

| Component | Reads | Writes |
|---|---|---|
| `camb_interface.py` (mode transfer) | `cosmological_parameters/{ombh2, omch2, h0 or cosmomc_theta, n_s, A_s, omega_k, w, wa, mnu, ...}`, `halo_model_parameters/{A, eta0, logT_AGN}` | `matter_power_lin/{z,k_h,p_k}`, `matter_power_nl/{z,k_h,p_k}`: p_k shape (nz, nk), k in h/Mpc, P in (Mpc/h)^3 (camb_interface.py:396-404); `distances/{z,a,D_A,D_C,D_M,D_L,H,...}` (378-390), distances in Mpc and `H` in Mpc^-1 (= H/c; CAMB `h_of_z`, camb/results.py:1479) [corrected: was "km/s/Mpc"]; also `MU`, `rs_DV`, `F_AP` (the latter two from an extra `get_BAO` background run over the 6000 z nodes, not used by KiDS); `cosmological_parameters/{sigma_8,S_8,sigma_12,omega_m,omega_lambda,...}` (407-421) |
| `approximate_P_gm` | `matter_power_lin`, `matter_power_nl`, `cosmological_parameters/{omch2,h0,n_s}`, `bias_parameters/{b1_bin_1,b1_bin_2,b2_bin_1,b2_bin_2,gamma3_bin_1,gamma3_bin_2}` (+ `gamma2_bin_*` only if `local_lag_g2=F`) (p_gm_approx_interface.py:31-51) | `matter_galaxy_power/{z,k_h,p_k}`, p_k shape (nz,nk), (Mpc/h)^3 (67-72) |
| `extrapolate_power` | `matter_power_nl`, `matter_power_lin` | same sections, k extended up to 500 h/Mpc (but NOT `matter_galaxy_power`, see Q5) |
| `load_nz_fits` | KiDS twopoint fits extensions `NZ_SOURCE`, `NZ_LENS` | `nz_source/{nbin,nz,z,bin_i}`, `nz_lens/{...}` (sections are named `NZ_SOURCE`/`NZ_LENS`, case-insensitive; `nz` is the NUMBER of z samples, not the n(z) array; `z` = Z_MID with a zero node prepended at z=0 if Z_MID[0]>1e-8, `upsampling` default 1 so no step-function up-sampling); each bin normalised with trapz (load_nz_fits.py:72-76) |
| `projection` | `distances`, `nz_*`, `matter_power_nl`, `matter_galaxy_power`, `galaxy_intrinsic_power`, `cosmological_parameters/{omega_m,h0}`, `galaxy_luminosity_function/alpha_binned` | `galaxy_shear_cl/{ell, nbin_a=2, nbin_b=5, bin_i_j}` (i = lens, j = source; shape (400,)); `magnification_shear_cl/...`; `galaxy_intrinsic_cl/...`; `shear_cl` (fast-IA cosmic shear) |
| `add_intrinsic` (position-shear only) | `galaxy_intrinsic_cl`, `intrinsic_alignment_parameters` | adds into `galaxy_shear_cl` (add_intrinsic.py:94-99) |
| `add_magnification` | `magnification_shear_cl` | adds into `galaxy_shear_cl` (add_magnification.py:34-38) |
| `bandpower_ggl` (C++, cosebis/BandPower_interface.cc) | `galaxy_shear_cl/{ell, bin_i_j}` | `bandpower_galaxy_shear/{bin_i_j (nBands=8), l_min_vec, l_max_vec, ...}` (565-577) |
| `scale_cuts` | data+cov from KiDS fits, theory sections | `theory_data_covariance/{data, theory, covariance, inv_covariance}` (scale_cuts.py:105-108,198) |
| `2x2pt_like` | `theory_data_covariance` | `data_vector/2x2pt_like_CHI2`, `likelihoods/2x2pt_like_LIKE` (mini_like.py:24-25) |
| `wedges` (external) | cosmology, bias params incl. `a_vir_bin_*` | `xi_wedges/bin_1`, `xi_wedges/bin_2` (each (n_wedge=3, n_points=28)), `lss_parameters/{rs_DV_bin_i, F_AP_bin_i, fsigma_8_bin_i}` (BOSS_*.ini extra_output) |
| `BOSS_like` (mini_BOSS_like.py) | `xi_wedges/bin_b` | `data_vector/<name>_CHI2`, `likelihoods/<name>_LIKE` (87-88); with `keep_theory_vector` also theory/data/cov/inv_cov/simulation per bin (77-83) |
| `planck_like` (C, clik) | `cmb_cl/{TT,EE,BB,TE,TB,EB}` and `planck/<clik extra parameter names>` | `likelihoods/PLANCK2018_LIKE`, optional `PLANCK18_LIKE_<n>` (planck_interface.c:349-381) |
| `planck_lensing_like` | `cmb_cl/pp` | `likelihoods/<like_name>_LIKE`, `data_vector/<like_name>_CHI2` (cobaya_planck_lensing.py:28-29) |

---------------------------------------------------------------------------------------------------

## 2. Mathematical description

Notation: chi = radial comoving distance in Mpc/h (the projection module multiplies CAMB's Mpc by h0, project_2d.py:541); a = 1/(1+z);
k in h/Mpc; Omega_m = `cosmological_parameters/omega_m` as written by CAMB (includes massive neutrinos, camb_interface.py:363, 421);
c = 299792.4580 km/s (project_2d.py:76,139).

### 2.1 Galaxy-matter power spectrum approximation (`P_gm_approx`)

For each grid point (z_i, k_j) of the CAMB non-linear grid:

    P_gm(k, z) = b1_s * P_nl(k, z)  +  F_s(k; cosmology) * P_lin(k, z)^2          (a)

where s = "low" (parameters `*_bin_1`) if z < z_sep, else "high" (`*_bin_2`), with z_sep = 0.5 set in run_kcap.py:511 (no default in code,
`options.get_double(option_section, "z_sep")`, p_gm_approx_interface.py:23). The comparison is strict `z < z_sep` (line 64), so z = 0.5 belongs to the high bin.

    F_s(k) = Theta(k - 0.01) * [ b2_s * N_b2(k) - gamma2_s * N_g2(k) - gamma3_s * N_g3(k) ]          (b)

Theta is implemented as `np.where(k < 1e-2, 0.0, ...)` (p_gm_fit.py:33). k is in h/Mpc, so the cut is k < 0.01 h/Mpc gives zero.
The minus signs are intentional ("g2/3 F terms negative", p_gm_fit.py:33).

Tidal bias: with the default `local_lag_g2 = True` (p_gm_approx_interface.py:20, not overridden by run_kcap), the second-order tidal bias is tied to b1 by local Lagrangian bias:

    gamma2_s = -(2/7) (b1_s - 1)          (c)

If `local_lag_g2=F`, `gamma2_bin_{1,2}` are read from `bias_parameters`. gamma3 is always read from `bias_parameters/gamma3_bin_{1,2}` (there is no gamma3 tie-in in this module,
even though the `wedges` module has its own `local_lag_g3` switch, set False in run_kcap.py:501, i.e. gamma3 is free).

Fit functions: for X in {b2, g2, g3}, with x = n_s and y = omch2/h0 (note: omega_c h^2 divided by h, i.e. NOT omega_c):

    N_X(k) = exp( A_X(x,y) * (ln k)^2 + B_X(x,y) * ln k + C_X(x,y) )                (d)
    A_X, B_X, C_X = poly2(x, y; row 0, 1, 2 of coefficient file for X)
    poly2(x,y;c) = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2                      (e)

File layout: `parameter_gridfit_{b2,g2,g3}.dat` have 3 rows of 6 numbers; row 0 multiplies (ln k)^2, row 1 multiplies ln k, row 2 is the constant
(header comment: "rows: decreasing order in k-fit polynomials"; p_gm_fit.py:25-31 uses `logk**(2-i)`). The coefficient values are reproduced in section 3.9.

Units: N_X carries units (Mpc/h)^-3 (because P_lin^2 has (Mpc/h)^6 and P_gm has (Mpc/h)^3); the fit is therefore dimensional and k must be in h/Mpc.

[verified] Table in 3.9 re-computed independently from the .dat files (numpy, all 24 numbers agree to the printed 4 digits). The fit is evaluated on the full CAMB k grid, whose maximum is NOT 20 h/Mpc but
about 21/h h/Mpc (25.6-32.8 for h in [0.82, 0.64]; see 3.1 [corrected]), so the last decade of the fit is further outside any plausible fitting range than the text of Q7 implies:
for n_s = 0.97, omch2 = 0.12, h = 0.7: N_b2 = 1.143e3, 1.919e3, 2.469e3; N_g2 = 4.283e4, 9.125e4, 1.325e5; N_g3 = 6.064e3, 1.277e4, 1.845e4 at k = 25, 30, 32.8 h/Mpc.
[verified] Implementation detail: `polyval_2d_order2` calls `np.meshgrid` on scalars, so each coefficient is a (1,1) array and `nonlin_term[j] = np.exp(...)` relies on numpy dropping the leading unit axis on assignment;
it works (shape (nk,) results) but is fragile. [verified] The prototype `pgm_model_old.py` hard-codes b1=1.8, b2=-0.1171, g2=0.2 (the tied value is commented out), g3=1.0, omch2=0.11818 and reads PT products
from `/share/splinter/joachimi/kids/kids1000/models/` (line 4), so the fit cannot be regenerated from the repo.

Meaning (INFERRED, not documented in repo): the terms proportional to b2, gamma2, gamma3 are the one-loop-order contributions of second- and third-order
Lagrangian/Eulerian bias operators to P_gm. They are modelled as N_X(k) P_lin^2, i.e. all redshift dependence is carried by P_lin(k,z)^2 and the fit
depends on cosmology only through (n_s, omega_c h^2/h). The prototype `pgm_model_old.py` (lines 57-75) shows the fit was checked against a
perturbation-theory calculation `galaxy_matter_power_spectrum_pt` / `matter_matter_power_spectrum_pt` (outputs of the external BOSS module), at the first
redshift bin only (`zbin=0`, line 80), and that "use pk_nl (halofit) for KiDS modelling" in the b1 term (line 73 comment).

Grid mismatch fallback: if P_lin and P_nl have different shapes, P_lin is resampled onto the (k, z) grid of P_nl with a cubic `RectBivariateSpline(z, ln k, ln P)`,
evaluated with `grid=True` (p_gm_fit.py:5-11, interface line 37-38). With the CAMB interface both come from the same transfer calculation so this branch is normally not taken (INFERRED).

### 2.2 Limber projection used for GGL (flat-sky, no curvature)

For a lens bin i (position field, kernel N) and source bin j (shear, kernel W):

    C^{g gamma}_{ij}(l) = A_L * Integral_0^{chi_max} dchi  n_i(chi) W_j(chi) / chi^2  * P_gm( k=(l+1/2)/chi, z(chi) )          (f)

    A_L = [ (3/2) (100 km/s/Mpc)^2 / c^2 * Omega_m ]^1         (prefactor_power = 1 for PositionShear, project_2d.py:255-260, 134-147)

Kernels (kernel.c):
- Lens kernel: n_i(chi) = n_i(z) * dz/dchi = -(1+z)^2 n_i(z) da/dchi (kernel.c:177-200). The n(z) are already normalised (trapz) by `load_nz_fits`; kcap does
  not renormalise. [verified] This N-kernel spline lives on the (non-uniform) chi values chi(z_i) of the n(z) nodes (Z_MID, spacing 0.01 for the lens bins, with a zero node prepended at z=0) and is an Akima
  spline through the histogram heights; da/dchi is the Akima derivative of the 6000-node a(chi) spline. For the lens bins the integrand therefore is non-zero only for z in about [0.2, 0.75]
  while the adaptive integrator is run over the whole spline overlap chi in [0, chi(z=0.785)].
- Source lensing kernel: W_j(chi) = norm_j * (chi/a(chi)) * Integral_chi^{chi_max} dchi' n_j(chi') (chi'-chi)/chi', with norm_j = 1/Integral_0^{chi_max} n_j(chi) dchi
  (kernel.c:53-58, 90-170, line 153). Evaluated at 1024 equally spaced chi nodes in [0, chi_max] (NGLT = 1024, line 21; delta_chi = chi_max/1023).
- The factor 1.5 (H0/c)^2 Omega_m is NOT inside W; it is the prefactor A_L (kernel.c:86-89 comment).

Limber evaluation point is k = nu/chi with nu = l + 1/2 (limber.c:134). Integrand actually coded (limber.c:108-173, with K = 0 so f_K(chi) = chi):

    integrand = [Wr_X Wr_Y] * P(chi, ln k) / (chi * D(chi)^2) * (chi/f_K(chi))^2 ,   Wr = W * D / sqrt(chi)

so algebraically integrand = W_X W_Y P / chi^2. The growth D(chi) cancels except through interpolation (see Q4).

Each output is `ell`-sampled at n_ell log-spaced multipoles (project_2d.py:413-415): np.logspace(log10(ell_min), log10(ell_max), n_ell).

### 2.3 Magnification contribution to GGL

    C^{kappa_g gamma}_{ij}(l) = A_L^2 * 2 (alpha_i - 1) * Integral dchi  W_i^{lens}(chi) W_j^{src}(chi) / chi^2 * P_nl(k=(l+1/2)/chi, z)          (g)

MagnificationShear: kernels "W W", P = `matter_power_nl`, `prefactor_power = 2`, `magnification_prefactors = (1,)` (project_2d.py:277-283, 143-152).
`W_i^{lens}` is the lensing kernel built from the LENS n(z) (not the N kernel). alpha_i = `galaxy_luminosity_function/alpha_binned[i]`, i = lens bin index (0-based);
written by `magnification_alphas.py:13` [corrected from :55; the file has 17 lines] as `galaxy_luminosity_function/alpha_binned`; default values `--magnification-alphas 1.8 2.62` (run_kcap.py:971). So 2(alpha-1) = 1.6 (low-z lens) and 3.24 (high-z lens).
`add_magnification` simply adds `magnification_shear_cl/bin_i_j` to `galaxy_shear_cl/bin_i_j` for all (lens i, source j) (add_magnification.py:35-38).

### 2.4 Intrinsic alignment contamination of GGL

`linear_alignment` with `method = bk_corrected` and `do_galaxy_intrinsic = True` (run_kcap.py:545-548):

    F(z) = - A_IA * C1_RHOCRIT * Omega_m / D(z),     D(z) = sqrt( P_nl(z, k_min) / P_nl(0, k_min) )         (h)
    P_II = F^2 P_nl,   P_GI(matter-intrinsic) = F P_nl     (linear_alignments.py:79-106)
    P_gI(k,z) = P_GI(k,z) * P_gm(k,z) / P_nl(k,z) = F(z) P_gm(k,z)  for k <= 0.999 k_max(P_gm)   (linear_alignments_interface.py:70-77)

[verified] C1_RHOCRIT (linear_alignments.py:16-46) = C1 rho_crit,0 with C1 = 5e-14 h^-2 Msun^-1 Mpc^3, Msun = 1.9891e30 kg, Mpc = 3.0857e22 m, G = 6.67384e-11 and H = 100 h km/s/Mpc, which evaluates to 0.0138731
(dimensionless). The NLA amplitude has no extra redshift or luminosity scaling: F(z) = -A C1_RHOCRIT Omega_m / D(z) only. [verified] D(z) here uses `ksmall = argmin(k_nl)` = the smallest k of the CAMB grid (about 7.1e-5 h/Mpc
for h = 0.7), not the 1e-3 h/Mpc of the projection module. [verified] In `linear_alignments_interface.py` the P_gI grid is cut to `k <= 0.999 * k[-1]` of P_gm, which drops the LAST CAMB k node (and, because linear_alignment runs after
extrapolate_power, the remaining 199 extrapolated nodes of P_GI are discarded as well). Then
`position-intrinsic` Limber: kernels "N N", `prefactor_power = 0`, P = `galaxy_intrinsic_power` (project_2d.py:262-268):

    C^{g I}_{ij}(l) = Integral dchi n_i(chi) n_j(chi) / chi^2 * P_gI(k=(l+1/2)/chi, z)          (i)

`add_intrinsic` with `position-shear=True`, `shear-shear=False` (run_kcap.py:570-573) adds it with weight A[j] = 1 (perbin False), i.e.
the IA amplitude A_IA enters only through F. Total GGL theory before band-power conversion:

    C^{gE}_{ij}(l) = C^{g gamma}_{ij} + C^{kappa_g gamma}_{ij} + C^{g I}_{ij}          (j)

### 2.5 Band powers for GGL (`bandpower_ggl`)

Configuration: `type = ggl`, `Response_function_type = tophat`, `Analytic = 1`, `l_min = 100`, `l_max = 1500`, `nBands = 8`, `Apodise = 1`, `Delta_x = 0.5`,
`theta_min = 0.5` arcmin, `theta_max = 300` arcmin (run_kcap.py:658-678, 1109-1115).
Band edges (BandPower_interface.cc:153-160): l_b = l_min * (l_max/l_min)^(b/nBands), b = 0..8:

    100, 140.285, 196.799, 276.080, 387.298, 543.322, 762.199, 1069.251, 1500    (computed from the formula)

Band power (BandPower.h:16-52; BandPower.cc:211-230, 343-376; BandPower_interface.cc:345-354 for the GGL case, Bessel order 2):

    BP^b_{ij} = (1/N_b) * Integral dl  l  W^b_2(l)  C^{gE}_{ij}(l),     N_b = ln( l_max,b / l_min,b ),
    W^b_2(l) = Integral_{theta_min}^{theta_max} dtheta  theta  T(theta)  J_2(l theta)  g^b_2(theta),
    g^b_2(theta) = Integral dl' l' S^b(l') J_2(l' theta),   S^b = 1 inside [l_min,b, l_max,b] else 0.

T(theta) is a cos^2 apodisation in ln theta of width Delta_x = 0.5 centred on ln theta_min and ln theta_max (BandPower.cc:515-545). The l-integral runs from LLOW = 0.1 to
LHIGH = 1e4 [verified: so the 400-point projection grid between 1e4 and 3e4, about 43 of the 400 nodes, is never used] (BandPower_interface.cc:22-24) in sub-intervals delimited by the extrema of W on a 10000-point log grid, each integrated with 20-point Gauss-Legendre (`gaussianIntegrate_gsl(...,20)`, BandPower.cc:346-354, 476-509).
C(l) is interpolated with a cubic spline in ln l (function_cosebis.cc:775: `gsl_interp_cspline`) from the projection's `ell` grid and log-linearly extrapolated outside (see Q16).
[verified] The formulas above match the code: for `real = false` (which is how BandPower_interface.cc always initialises the class, lines 322-386) `BandPower::value` returns `result/Ni`
without any 2 pi (the 2 pi/Ni branch, BandPower.cc:358-359, is only for the unused real-space variant), and `BandPower_W::integrant` (BandPower_W.cc:213) is theta * T(theta) * J_n(l theta) * g(theta) with
the analytic g of `Theta_g_tophat` (the copy that is actually used is `BandPower_g::Theta_g_tophat`, BandPower_g.cc:268-304, called from `BandPower_g::get` when Analytic is true; BandPower_W.cc:469-504 is an unused duplicate; for n = 2, which equals the closed form of the integral over l of l J_2(l theta) because int x J_2 dx = -x J_1 - 2 J_0: g = -(1/theta^2)[l_u theta J_1(l_u theta) - l_l theta J_1(l_l theta) + 2 J_0(l_u theta) - 2 J_0(l_l theta)]).
[verified] Delta_x = 0.5 is a width in ln(theta) (BandPower.cc:515-545: T = cos^2[(pi/2)(x - x_l - Delta_x/2)/Delta_x] on [ln theta_min - Delta_x/2, ln theta_min + Delta_x/2], 1 in between,
mirrored at theta_max); the comment "apodisation length in arcmins" in run_kcap.py:651,672 is wrong. The g/W integrals are done on theta_min e^{-Delta_x/2} .. theta_max e^{+Delta_x/2}.
[verified] The ini switch `Analytic = 1` is effectively inert: `BandPower::initialize` calls `set_W()` with the header default `Analytic = true` (BandPower.h:106) and the later `set_W(Analytic)` in `calBP`
is a no-op because `WSet` is already true, so setting `Analytic = 0` would not change anything (`setAnalytic` is commented out, BandPower.cc:65). [verified] The W(l) tables (NLBINS = 1000 log-spaced points
between 0.1 and 1e4, cubic-spline interpolated) are computed on first use and cached under `Output_FolderName`; no cache is shipped (`cosebis/BandPower_outputs` does not exist in the checkout), so the first
likelihood call of a fresh install pays for 100000-point extremum searches and Bessel integrals for every (band, ell, order in {0,2,4}).
The band-power "ell" assigned to each band for scale cuts is the geometric centre 10^(0.5(log10 l_min + log10 l_max)) (scale_cuts.py:140-144):

    118.442, 166.156, 233.093, 326.994, 458.724, 643.521, 902.764, 1266.443          (computed)

For cosmic shear E-modes (not GGL) the code uses BP_E = 1/2 (BP_0 + BP_4) with Bessel orders 0 and 4 (BandPower_interface.cc:531-544).

### 2.6 KiDS GGL and shear likelihood

    chi2 = (d - m)^T C^{-1} (d - m),   ln L = -chi2/2         (scale_cuts.py:87-89; mini_like.py:19-25)

with d, C from the KiDS twopoint fits (COVMAT extension), after scale cuts, and no Hartlap factor, no log-determinant. The theory vector is the concatenation of the
spectra in the order PeeE-section list of scale_cuts.py:117-123 (xiP, xiM, PneE, PeeE, En), restricted to `use_stats`. For `EE_nE_w`, `used_stats = [PneE, PeeE]` (run_kcap.py:1156), and the
spectra are concatenated in the sectionNameList order, i.e. PneE first, then PeeE (scale_cuts.py:117-123, 125-185; the data vector order in the fits must match or the chi2 is wrong)
[verified: the data side keeps the FITS extension order (`choose_data_sets`, twopoint.py:746-765), and in the shipped file COVMAT starts are PneE at 0 and PeeE at 80, so the orders agree.
Theory pair order from `scale_cuts.execute` (GGL: lens-major, i.e. bin_1_1..bin_1_5, bin_2_1..; shear: bin_{j+1}_{i+1} for j >= i) matches the file's BIN1/BIN2 ordering listed in 1.1.]

Scale cuts for the production 3x2pt (run_kcap.py:1156-1171 plus `nE_scale_cuts` in runs/3x2pt/3x2pt_configs.py:35-44):
- GGL pairs removed altogether: `cut_pair_PneE = 1+1 2+1 2+2 2+3` (lens+source).
- GGL global range `keep_ang_PneE = 100 1070`; per pair overrides: lens 1 -> `100 300`, lens 2 -> `100 600`.
- Cosmic shear `keep_ang_PeeE = 100 1500`.
Consequence: lens 1 keeps band centres 118, 166, 233 (3 points) for sources 2-5, lens 2 keeps 118...459 (5 points) for sources 4-5, i.e. 12 + 10 = 22 GGL points; cosmic shear keeps all 8 bands x 15 pairs = 120 points
[verified by re-running exactly this mask logic (per-pair inclusive cut, then global strict cut) on the ANG/BIN1/BIN2 columns of the checked-in fits file: kept per pair (1,2)=(1,3)=(1,4)=(1,5)=3, (2,4)=(2,5)=5, all others 0; GGL 22, EE 120,
data vector length 142; the file's ANG values equal the geometric centres, so Q17's worry does not materialise for this file; the 200x200 COVMAT is cut to 142x142].
Note the global GGL cut `100 1070` is therefore redundant for the production config (every surviving point has centre <= 459) and only matters for pairs without a per-pair override.
The global cut uses strict inequalities `angle > min & angle < max` (twopoint.py:735), while the per-pair cut removes values `< min | > max` (twopoint.py:711-712), i.e. inclusive keep. For centres well inside the ranges this is moot.

### 2.7 BOSS wedges likelihood (`mini_BOSS_like.py`)

Data files: whitespace columns `s, w1, sigma1, w2, sigma2, w3, sigma3` (n_wedge = (ncols-1)//2 = 3; mini_BOSS_like.py:20-22; also create_BOSS_data_file header in run_kcap.py:936). Only the value columns are used (index 2j+1);
the quoted errors are ignored. Data vector for one z-bin: [w1(all s), w2(all s), w3(all s)] (wedge-major).
Covariance files are square with the same ordering (since the same index mask is applied to both, lines 40-49) [verified: the shipped covariances are 96x96 = 3 x 32 and wedge-major, because sqrt(diag) reshaped to (3,32) reproduces the sigma columns of the data files to within 0.1% (lowz) / 0.9% (highz); the quoted sigma columns are still ignored by the likelihood].
[verified] Condition numbers of the cut 84x84 covariances: 2.4e3 (lowz), 1.1e3 (highz), both inverted directly with `np.linalg.inv` once at setup (mini_BOSS_like.py:51).

Cut: keep s-bin indices `points_range[0] .. points_range[1]-1` (default `[4, 32]`, mini_BOSS_like.py:36-40; set explicitly to `4 32` in the inis and run_kcap.py:1090). The full file has n_points_full = 32 rows [corrected: the earlier text inferred 60 from the mock s-axis `2.5 + 5*arange(60)` (run_kcap.py:1094) and called the files NOT IN REPO; the shipped files have 32 rows, s = 2.5 ... 157.5 Mpc/h in 5 Mpc/h steps]. With `points_range = 4 32` the upper cut list `range(32, 32)` is EMPTY, so only the first 4 s-bins of each wedge are deleted (indices 0-3 of each 32-block); the kept range is s = 22.5 ... 157.5 Mpc/h, 28 points per wedge, 84 per z-bin, 168 in total. The mock s-axis slice `[points_range[0]:points_range[1]]` is consistent with 32 rows only because the mock `points_range` is [0, 32].

Hartlap-corrected inverse covariance per bin (lines 51-57):

    Psi = ((nu - p - 1)/nu) * C^{-1},   nu = n_mocks - 1,  p = dim(C) = 84,  n_mocks = 2048 (default)
    factor = (2048 - 84 - 2)/2047 = 0.958476

(identical to the standard Hartlap factor (N - p - 2)/(N - 1)) [verified numerically: 1962/2047 = 0.9584758; both the module default `n_mocks=2048` (mini_BOSS_like.py:26) and the explicit run_kcap.py:747 value agree.
Whether 2048 is the number of mocks that actually produced the shipped Sanchez et al. covariances cannot be checked from the repo.]

    chi2 = sum_{b=1,2} (x_b - mu_b)^T Psi_b (x_b - mu_b),   ln L = -chi2/2          (lines 63-88)

mu_b = `xi_wedges/bin_b` flattened with `np.concatenate` over the 3 wedge rows (line 71), so the theory module must output a (3, 28) array already cut to the same `points_range`.

The theory side (`wedges` module, `kcap_boss_module/python_interface/cosmosis_module.py`) is NOT IN REPO. What is known from configuration only:
`window_file BOSS.DR12_windows.txt`, `bands_file BOSS.DR12_rbands.txt`, `bands_range 20 160` (Mpc/h), `points_range 4 32`, `num_ell 3`, `z_eff 0.38 0.61`, `use_growth F`, `local_lag_g2 T`, `local_lag_g3 F`,
`output_section_name xi_wedges` and optional `timeout 600.0` s (run_kcap.py:488-501 and the driver scripts). `compute_lss_parameters T` [corrected] is set only in the hand-written `BOSS_only.ini` / `BOSS_KV450.ini` (module named `boss`
there, with `verbose F`), NOT by run_kcap.py nor in the checked-in generated ini, so the kcap-generated chains do not necessarily write `lss_parameters/*`; the module's default is unknown (module not in repo). Free parameters per z bin: `b1, b2, gamma3, a_vir` (priors in section 3.7).
The data come from Sanchez et al. 2017 (path `.../boss/Sanchez_etal_2017/`, runs/3x2pt/3x2pt_configs.py:22-25). Any further description of the model
(perturbation theory, Lagrangian bias basis, RSD treatment) would be from the literature, not from this repo, and is therefore not asserted here.

### 2.8 Planck

(a) CMB likelihood (`planck_interface.c`). CAMB writes lensed total spectra with `r.get_total_cls(raw_cl=False, CMB_unit="muK")`, so cmb_cl/{TT,EE,BB,TE} = l(l+1) C_l/(2 pi) in muK^2, array index 0 <-> l = 2 (camb_interface.py:425-431).
The clik interface converts back to raw C_l (lines 175-179):

    C_l = [ l(l+1)/(2 pi) C_l ]_stored / ( l(l+1)/(2 pi) ),   l = 2..lmax_clik

C_0 = C_1 = 0 (153-155). The vector passed to `clik_compute` is the concatenation [TT(0..lmax_TT), EE(...), BB, TE, TB, EB (only those with lmax>0), then nuisance parameters]; nuisance names are
obtained from clik and read from datablock section `planck` (line 188), e.g. `A_planck`.

Three clik files are used (Planck_only.ini:120-122; run_kcap.py:758-760): low-l TT (Commander, `commander_dx12_v3_2_29.clik`), low-l EE (`simall_100x143_offlike5_EE_Aplanck_B.clik`),
high-l TT,TE,EE `plik_lite_v22_TTTEEE.clik`. The names indicate l <= 29 for the low-l files (filename suffix "_29") [verified from the clik `_mdb` files in data/Planck: commander lkl_type gibbs_gauss lmin 2 lmax 29 free_calib A_planck; simall lmin 2 lmax 29 nell 28 free_calib A_planck;
plik_lite_v22_TTTEEE lkl_type plik_cmbonly lmin 30 lmax 2508, so CAMB's lmax = 2650 comfortably covers it]. Their log-likelihoods are summed (349-360) and stored as `PLANCK2018_LIKE` (381) (and individually as `PLANCK18_LIKE_n` if `save_separate_likelihoods=T`).
The returned number is already ln L from clik (the interface does not transform it).

Prior on the calibration: `[planck] a_planck = gaussian 1.0 0.0025` (Planck_only_priors.ini; same via `--set-priors planck a_planck "gaussian 1.0 0.0025"` in the 3x2pt configs). Free range `[0.9, 1.00061, 1.1]` in Planck_only_values.ini:21 [corrected from :144; the whole file has only 68 lines]; in the joint runs `[0.98791, 1.00061, 1.01308]`.
[verified] `planck_interface.c` can also evaluate native clik lensing likelihoods through `lensing_N` options (`run_clik_cosmosis_lensing`, summed into `PLANCK2018_LIKE` and stored as `PLANCK18_LENSING_LIKE_N`, lines 364-376 of the
file), a path that run_kcap.py does not use (it uses the cobaya wrapper instead), although the clik_lensing data directories are in `data/Planck/.../lensing`.

(b) CMB lensing (`cobaya_planck_lensing.py`). CAMB writes `cmb_cl/PP = C_l^{phiphi} * l(l+1)/(2 pi)` where `C_l^{phiphi}` is returned with `raw_cl=True` (camb_interface.py:438-439). The module prepends two zeros and passes to the external cobaya package
`plancklensing.PlanckLensingMarged().log_likelihood({"pp": l(l+1) * stored}, A_Planck=1.0)` (cobaya_planck_lensing.py:21-26), i.e. pp = [l(l+1)]^2 C_l^{phiphi}/(2 pi), the usual CAMB-normalised lensing-potential spectrum; the result is used as ln L directly and `CHI2 = -2 ln L` (28-29).
The package itself is NOT IN REPO (pip package from `CobayaSampler/planck_lensing_external`, referenced in the comment at line 6).

(c) How Planck is combined with the lensing probes. Simply additive in ln L plus a shared cosmology; the nuisance `a_planck` is Planck-only. The KiDS side uses `h0` and `S_8_input` as sampled parameters while the Planck-only config samples `cosmomc_theta` (camb_interface.py:280-284); for joint runs the Planck 5-sigma boxes below are imposed as hard flat bounds on the cosmological parameters (runs/3x2pt/3x2pt_configs.py:52-60):

    omch2 [0.113370, 0.12, 0.127031], ombh2 [0.021616, 0.0225, 0.023104], h0 [0.642529, 0.7, 0.715019],
    n_s [0.934281, 0.97, 0.986698], S_8_input [0.722929, 0.7458, 0.913414], tau [0.015071, 0.0543, 0.093819], a_planck [0.987908, 1.00061, 1.013081]

with the comment "5 sigma ranges, with S8 and n_s having a 7 sigma lower range and h having a 7 sigma upper range". This is a real truncation of the prior volume: the joint posterior cannot leave these boxes.

### 2.9 Cosmological parameterisation shared by all runs (needed to reproduce numbers)

- Sampled: `S_8_input`, `omch2`, `ombh2`, `h0`, `n_s` (run_kcap.py:850-860). A_s is then derived by `sigma8toAs.py`:

      sigma8_input = S_8_input / (Omega_m,pre / 0.3)^0.5,     Omega_m,pre = (omch2 + ombh2)/h0^2        (sample_S8.py:23-33, alpha = 0.5)
      A_s = 2.1e-9 * (sigma8_input / sigma8_fid)^2,   sigma8_fid from a CAMB run with As = 2.1e-9, no reionisation, kmax = 1.2, k_per_logint = 5,
            PK_redshifts = [0], same (h, ombh2, omch2, omk, mnu, w, wa, n_s)        (sigma8toAs.py:28-56)

  The Omega_m used in the `sample_S8` step excludes massive neutrinos; the derived `S_8 = sigma_8 sqrt(Omega_m/0.3)` after CAMB uses CAMB's `omegam` which includes them (camb_interface.py:421). See Q2.
  [verified] CAMB 1.5.8 `omegam` = omegab + omegac + omeganu: for ombh2 = 0.0225, omch2 = 0.12, h = 0.7, mnu = 0.06 it is 0.29213 versus 0.29082 without neutrinos (omeganu = 0.001316).
  [verified] sample_S8.py:23-33: if `cosmological_parameters/omega_m` is already in the block it is used instead of (omch2+ombh2)/h^2; in the kcap pipeline sample_S8 runs before CAMB, so the fallback is what is used.
- `ln_1e10_A_s` route (`sample_ln_As.py`): A_s = exp(ln_1e10_A_s) * 1e-10 (used in BOSS_*.ini and Planck_only.ini).
- Fixed in the fiducial 3x2pt: `omega_k = 0`, `w = -1`, `mnu = 0.06` with `neutrino_hierarchy = normal` in `camb` (run_kcap.py:857-859, 458). `wa` is not set in `create_values`, so it defaults to 0 (camb_interface.py:205).

### 2.10 Extended cosmologies supported

All are enabled purely by freeing a parameter via `--set-parameters`, in runs/extended_cosmologies/3x2pt_ext_cosmo_configs.py:80-106 and maxlike scripts.

1. wCDM (`wLCDM`): free `w` in [-3.0, -0.33], start -1.0. CAMB `DarkEnergyFluid` with `set_params(w=w, wa=wa, cs2=cs2_de default 1.0)` (camb_interface.py:192-209), fluid model unless `use_ppf_w=T` (not set anywhere in these runs). CAMB's definition (in CAMB source, `dark_energy.py` line 43 in the locally installed 1.5.8): w(a) = w + (1-a) wa. Hence
   H(a)^2/H0^2 = Omega_m a^-3 + Omega_r a^-4 + Omega_k a^-2 + Omega_DE a^{-3(1 + w0 + wa)} exp(-3 wa (1 - a)) (standard CPL integral, quoted from the model definition not from kcap code).
   w0wa: `wa` is read from `cosmological_parameters/wa` by CAMB (camb_interface.py:205) and by `sigma8toAs.py:23`, so the plumbing exists, but NO config in runs/ frees `wa` (no `wa` string in runs/**/*.py). For the Planck wCDM case the box is `w [-3.0, -1.0, -0.33]` with `h0 [0.64, 0.7, 0.82]` (3x2pt_ext_cosmo_configs.py:69-75).
2. Massive neutrinos (`nuLCDM`): free `mnu` in [0.0, 3.0] eV, start 0.06 (line 91). Passed as `mnu` to `CAMBparams.set_cosmology` together with `neutrino_hierarchy = normal` (camb_interface.py:83, 267, 297-300); `omnuh2`/`omega_nu` inputs are ignored with a warning (274-277). Alternatives tried and commented out: folded prior (`sample_folded_prior.py`: mnu = fold + |mnu_folded - fold|) and wrapped parameter (lines 84-91); `sample_negative_mnu.py` (mnu = |mnu_proxy|).
3. Curvature (`oLCDM`, `oCDM`): free `omega_k` in [-0.4, 0.4], start 0.0 (line 80). Passed to CAMB as `omk` (camb_interface.py:265) with `h0` supplied, so Omega_Lambda = 1 - Omega_m - Omega_k (350). Distances: `D_M` and `D_C` are both `comoving_radial_distance` (384-385). See Q1: the projection treats space as flat.
4. Modified gravity f(R) (`fRCDM`, run type `EE_fR`): `reaction`+`hmcode`+`multiply_reaction` modules (external ReACT/HMx, NOT IN REPO), `log10_fR0` in [-8, -2], `S_8_input` widened to [0.45, 0.7458, 1.3], `camb nonlinear=none` (run_kcap.py:168-172; ext configs 102-106). Not described further here (belongs to the non-linear power spectrum subsystem).
5. Baryon feedback variants: one-parameter HMcode2016 (`eta0 = 0.98 - 0.12 A`, A in [2.0, 2.6, 3.13], utils/one_parameter_hmcode.py:11-16; run_kcap.py:447-450) versus `HMCode2020_wide`: `halofit_version = mead2020_feedback`, `logT_AGN` in [7.3, 7.8, 8.3], `halo_model_parameters/A` removed (ext configs 93-100).
6. Additional external data in the extended-data scripts (3x2pt_ext_data_configs.py): Planck lensing (requires `tau = 0.055` fixed and `lens_potential_accuracy = 1`, lines 62-69), Pantheon (`supernova_params/M` free in [-22,-19,-17] or fixed at -19, lines 71-75), Riess+2019 H0 prior `h0 ~ N(0.7403, 0.0142)` (line 77).

For BOSS-only extended chains (`w` run type) the pipeline reduces to `sample_S8, sigma8toAs, camb, wedges, BOSS_like` (the lensing modules are cut, run_kcap.py:140-144) [corrected: also `correlated_dz_priors` and `one_parameter_hmcode`, which are not cut for `w`; see 1.2], with `omch2 <= 0.2` and `n_s <= 1.1` imposed whenever the wedges module is on (`wedges_param_range`, lines 64-65).
[verified] Other details of the extended scripts: fRCDM uses `log10_fR0` [-8, -7, -2] (start -7), wrapped via `--sampler-config wrapped_params cosmological_parameters--log10_fr0`, and extra derived `sigma_8_LCDM`, `S_8_LCDM`
(3x2pt_ext_cosmo_configs.py:102-106); the Planck+wCDM box has `omch2 [0.11312, 0.12, 0.12674]`, `ombh2 [0.021654, 0.0225, 0.023132]`, `n_s [0.94376, 0.97, 0.98703]`, `S_8_input [0.62806, 0.7458, 0.92690]`, `tau [0.014360, 0.0543, 0.093506]`
(lines 69-76), different from the LCDM Planck box in 2.8. As shipped (`run_types = ["w"]`, `configs = nuLCDM_configs`) only the BOSS-only nuLCDM chain would be generated.

---------------------------------------------------------------------------------------------------

## 3. Numerical choices

### 3.1 CAMB settings used by the 3x2pt pipeline (run_kcap.py:452-467)

mode `transfer`; `nonlinear = pk` (CAMB's non-linear matter P(k); `lens` is not enabled); `halofit_version = mead` (HMcode 2016, default argument at run_kcap.py:970); `neutrino_hierarchy = normal`; `do_reionization = F`; `kmax = 20.0` in CAMB's native units of 1/Mpc, NOT h/Mpc [corrected: the earlier text said h/Mpc throughout;
CAMB's `Transfer.kmax` is "k, not k/h" (camb/model.py set_matter_power docstring) and I confirmed with CAMB 1.5.8 that the returned grid has 197 k nodes, from 7.14e-5 h/Mpc (h = 0.7; = 5e-5 Mpc^-1 / h) up to 21.0/h h/Mpc (i.e. CAMB extends kmax = 20 to 21 Mpc^-1), so k_h,max = 32.81 / 30.00 / 25.61 for h = 0.64 / 0.70 / 0.82, the same for linear and non-linear];
redshift grid for P(k): `zmin = 0` (default 0, camb_interface.py:105), `zmid = 2.0`, `nz_mid = 100`, `zmax = 6.0`, `nz = 150` => z = {linspace(0, 2, 100, endpoint=False)} U {linspace(2, 6, 50)}, 150 points, spacing 0.02 below z=2 and 4/49 above (camb_interface.py:314-322; reversed to decreasing for CAMB);
background: `background_zmin = 0`, `background_zmax = 6`, `background_nz = 6000` linearly spaced (378).
The number of k points is not set by kcap (`k_per_logint` unset, `transfer_params` contains only kmax; 127-129), so it is whatever CAMB's default gives; [corrected: determinable] `p.set_matter_power(redshifts=z, nonlinear=..., kmax=20)` (camb_interface.py:324)
passes `k_per_logint=None`, which CAMB 1.5.8 turns into `Transfer.k_per_logint = 0` (its optimised default spacing) and it also forces `WantTransfer = True` and `Transfer.high_precision = True` (camb/model.py:684-720). Measured with CAMB 1.5.8 using the kcap settings: 197 k nodes for every h tested.
[verified] This high_precision = True contrasts with `sigma8toAs.py:37`, which sets `high_precision = False`, `k_per_logint = 5`, `kmax = 1.2`; for the fiducial LCDM point that low-precision run gives sigma_8 smaller by 7.0e-4 (relative) than the main run at the same A_s
(0.81838 vs 0.81895 with A_s = 2.1e-9, h = 0.7, ombh2 = 0.0225, omch2 = 0.12, n_s = 0.97, mnu = 0.06 normal), i.e. the realised sigma_8 is about 0.07 per cent above the sampled target (see Q2).
[verified] CAMB 1.5.8 timing on this machine for the kcap settings (150 z, 197 k, HMcode 2016): `get_results` 0.39 s; extracting the linear spectrum 0.00 s; extracting the non-linear spectrum 2.8 s (HMcode at 150 redshifts is the dominant cost;
`halofit_version = takahashi` would take 0.38 s); the sigma8toAs-style run 0.07 s; `get_BAO` on 6000 redshifts 0.05 s; distances 0.04 s.
`z_sep = 0.5` is an exact grid node (z[25] = 0.5 exactly in double precision, I checked; re-checked in this pass, `np.linspace(0,2,100,endpoint=False)[25] == 0.5`, z[24] = 0.48, and CAMB returns the z array ascending with 150 entries), so grid z <= 0.48 use the low-z bias and z >= 0.5 the high-z bias.
Dark energy fluid c_s^2 = 1 (default `cs2_de`).

### 3.2 Power spectrum treatment before projection

- `extrapolate_power`: kmax = 500 h/Mpc, `kmin` default 1e10 (so no extension to low k), `nmax` = 200 (default) nodes linearly spaced in ln k from the current k_max to 500, of which the first (= the old k_max) is dropped, so 199 NEW points are added [corrected from "200 new points", extrapolate_power.py:25-27]; `npoint = 3` (default), least-squares straight-line (`np.polyfit(..., 1)`) fit of ln P versus ln k through the last 3 grid points, per redshift (extrapolate_power.py:12-31, 56-67), i.e. a pure power law P ~ k^n with n the local slope at k_max. [verified] The extension happens AFTER `approximate_P_gm` in the chain. Applied to `matter_power_nl` and `matter_power_lin` only; `sections` option is empty because the line `#"sections" : "matter_galaxy_power"` is commented out in run_kcap.py:516 and `("extrapolate_power","sections")` is commented out at line 72.
- Therefore `matter_galaxy_power` stays on the CAMB k grid (k_h,max = 21/h, i.e. 25.6-32.8 h/Mpc, NOT 20 h/Mpc [corrected]) (see Q5).
- In `limber.py:111-116`, the power is turned into a bicubic GSL spline `GSLSpline2d(chi, ln k, P.T, BICUBIC)` of P itself (not ln P), with chi(z) from the distance spline. k values outside the grid make `gsl_spline2d_eval_e` fail and the integrand returns 0 (limber.c:141-146).
  [verified] Orientation (resolves Open question 10): `GSLSpline2d(x=chi, y=ln k, Z=p.T)` flattens a (nk, nz) C-ordered array, which is exactly GSL's `za[j*nx + i]` layout with nx = len(chi) = nz, ny = nk
  (gsl_wrappers.py:127-136); chi must be strictly increasing, which holds because CAMB returns z ascending. The spline is `gsl_interp2d_bicubic` (limber.py:109,115), the 1-D splines are Akima (gsl_wrappers.py:68 default; utils.c:23).
  Because the spline is bicubic in (chi, ln k) of P itself and the P_gm grid has a step between the z nodes 0.48 and 0.50 (3.1), the interpolation cells touching the step are the ones in z in [0.46, 0.54];
  from the checked-in NZ_LENS, 13.9 per cent of the (unit-normalised) lens bin 1 and 12.8 per cent of lens bin 2 n(z) mass lie in 0.48 < z < 0.52 (the edge bins at Z_MID 0.485/0.495 and 0.505/0.515), so the
  step is not an academic corner of the integral (the effect itself was not computed; see Q4).
- Growth: `D(chi) = sqrt( P(z,k_g) / P(0,k_g) )` with `k_g` the first grid k above 1e-3 h/Mpc (limber.py:118-128). Here P is whichever 3D spectrum is being projected, so for GGL it is P_gm itself.

### 3.3 Limber integration (limber.c, project_2d.py)

- ell grid: `ell_min`, `ell_max`, `n_ell` log-spaced. 3x2pt: 1.0, 3.0e4, 400 (run_kcap.py:556-558). BOSS_KV450.ini (cosmic shear xi_pm only): 0.1, 5.0e5, 400 (BOSS_KV450.ini:187-189).
- Integrator: GSL `qag` with Gauss-Kronrod 61 rule, workspace 4096 intervals, relative tolerance 1e-3 (`limber_rel_tol` default), absolute tolerance 0 (project_2d.py:416-417; limber.c:15, 429); on failure falls back to fixed 4096-point Gauss-Legendre (limber.c:435-443, with a one-time stderr warning). Integration range = overlap of the two kernel splines' chi support (limber.c:554-555).
- Kernel splines: GSL Akima (`spline_from_arrays` uses `gsl_interp_akima`, utils.c:23).
- Kernel normalisation integral and the nested W integral use 1024-point fixed Gauss-Legendre (`NGLT`, kernel.c:21, 127, 153).
- Reduced-kernel splines also Akima on the same nodes (kernel.c:60-84). [verified] With chi_max = chi(z = 6) of order 5.8e3 Mpc/h the 1024 W-kernel nodes are about 5.7 Mpc/h apart; the reduced kernel is set to exactly 0 at chi = 0 and at chi >= the last node of the growth spline (kernel.c:73-79), so the Limber integral can never see chi = 0.
  [verified] `limber.limber()` passes `ext_order = 0` and `K = 0.0` unconditionally (limber.py:160-176); the extended-Limber and curvature branches of limber.c are dead code for kcap.
- OpenMP loop over ell (limber.c:577). Spectra are computed for every (i,j) of the requested pairs: 2x5 = 10 for GGL, 10 for magnification-shear, 10 for position-intrinsic, plus 15 for the fast shear+IA, all x 400 ell.

### 3.4 Distances and constants

`chi = d_m * h0` where `d_m` = CAMB `comoving_radial_distance` (Mpc) (project_2d.py:530-541 and camb_interface.py:385), `chi_max = max(chi)` (z = 6), `a(chi)` and `chi(z)` Akima splines on the 6000-point background grid. c = 299792.4580 km/s in the prefactor (project_2d.py:76,139), the same value as in consistency.py:88.

### 3.5 Band powers

Defined in 2.5. Numerical constants: NLBINS = 1000 (weight table points), LLOW = 0.1, LHIGH = 1e4 (BandPower_interface.cc:22-24); Fourier integration limit finding on 10000 log points (BandPower.cc:478-486); 20-point Gauss-Legendre per sub-interval (BandPower.cc:351); band edges as in 2.5; `Analytic = 1`: analytic top-hat g (BandPower_interface.cc:255-272). The weight files are cached on disk in `Output_FolderName = COSEBIS_PATH/BandPower_outputs/` (run_kcap.py:656,677).

### 3.6 Scale cuts and data assembly

See 2.6. The KiDS twopoint fits file (`bp_KIDS1000_Blind{A,B,C}_with_m_bias_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_goldclasses_Flag_SOM_Fid.fits`, `fits_iterative_covariance/` variant for the production chains, runs/3x2pt/3x2pt_configs.py:8) lives outside this repo (`../Cat_to_Obs_K1000_P1`). The `COVMAT` extension is loaded with `covmat_name='COVMAT'` (scale_cuts.py:59); `np.linalg.inv` of the cut covariance is computed once at setup (89).

### 3.7 Priors and parameter ranges (flat unless stated)

3x2pt fiducial (`create_values`, run_kcap.py:850-891):

| parameter | [min, start, max] |
|---|---|
| omch2 | [0.051, 0.13, 0.255] (BOSS-containing runs: max 0.2) |
| ombh2 | [0.019, 0.0225, 0.026] |
| h0 | [0.64, 0.7, 0.82] |
| n_s | [0.84, 0.97, 1.1] |
| S_8_input | [0.1, 0.7458, 1.3] |
| halo_model_parameters/A | [2.0, 2.6, 3.13] |
| intrinsic_alignment_parameters/A | [-6.0, 0.8, 6.0] |
| nofz_shifts/p_1..5 | [-5, mean_i, 5] with Gaussian N(mean_i, 1) priors (runs/3x2pt/3x2pt_configs.py:98-101; decorrelated by Cholesky of SOM_cov, lines 14-19) |
| bias b1_bin_1 / b1_bin_2 | [0.5, 2.1, 9.0] / [0.5, 2.3, 9.0] |
| bias b2_bin_1 / b2_bin_2 | [-4, 0.2, 8] / [-4, 0.5, 8] |
| bias gamma3_bin_1 / gamma3_bin_2 | [-8, 0.9, 8] / [-8, 0.1, 8] |
| bias a_vir_bin_1 / a_vir_bin_2 | [0, 3.8, 12] / [0, 3.0, 12] |
| shear_c_bias c1, c2, Ac | fixed in all 3x2pt run types (`c_term_values` in fix_values) at the middle of their ranges [verified]: ranges c1, c2 [-6e-4, 0, 6e-4] -> 0.0; Ac [0.62, 1.13, 1.4] -> 1.13 (run_kcap.py:871-873; confirmed in the checked-in values.ini). In the 3x2pt band-power likelihood they have no effect: BandPower_interface.cc:16 defines `cterm_section = "shear_c_bias"` but never reads it, and no module in the 3x2pt chain does. |

The older BOSS-only and BOSS+KV450 configs (runs/config/BOSS_only_values.ini, BOSS_KV450_values.ini) use a different parameterisation: `omch2 [0.01,0.10,0.2]`, `ombh2 [0.019,0.022,0.026]`, `cosmomc_theta [0.9,1.04,1.2]` (CAMB receives theta/100), `n_s [0.5,0.96,1.1]`, `ln_1e10_A_s [1.5,3.09,4.0]`, and bias `b1 [0.5,1.9,4.0]`, `b2 [-4,0,8]`, `gamma3 [-8,1,8]`, `a_vir [0,2,12]` for both bins; comments say BOSS "doesn't like" omch2 > 0.18, n_s > 1.1, A_s > 7e-9 (values files lines 3-21, 19-20).
Nuisance Gaussian priors in BOSS_KV450_priors.ini: `nofz_shifts bias_i ~ N(0, [0.039,0.023,0.026,0.012,0.011])`, `A_c ~ N(1.01, 0.13)`, `delta_c ~ N(0, 2e-4)`.

Sampler settings in the driver scripts (runs/3x2pt/3x2pt_configs.py:28-31): `multinest_efficiency 0.3`, `nested_sampling_tolerance 1e-2`, `live_points 500`. [corrected] The one production-run ini that is actually checked in (blind C, EE_nE_w, data_iterated_cov) records
`live_points = 1000`, `efficiency = 0.3`, `tolerance = 1.0e-2`, `update_interval = 100`, `resume = T`, `max_iterations = 1000000`, so "500 live points" describes the current script / "medium" chains (the ext-cosmo script labels 500 as "_medium", 250 as "_fast"), not that run; run_kcap.py's own defaults are 250 / 0.1 / 0.8 and update_interval 20. The BOSS_*.ini defaults are `live_points 1000`, `efficiency 0.3`, `tolerance 0.5`. Wedge module timeout 600 s (`--set-keys wedges timeout 600.0`).

### 3.8 Planck numerics

`Planck_only.ini` (CMB-only chain): camb `mode = cmb`, `lmax = 2650`, `lens_potential_accuracy = 1.0`, `nonlinear = lens`, `do_lensing = T`, `do_tensors = F`, `neutrino_hierarchy = normal`, `theta_H0_range = 20 100` (Planck_only.ini:104-115); sampled `ombh2 [0.005, 0.0223773, 0.1]`, `omch2 [0.001, 0.1201035, 0.99]`, `cosmomc_theta [0.5, 1.04092, 10.0]` (units: 100 theta_MC), `tau [0.02, 0.0543, 0.2]`, `n_s [0.8, 0.9659, 1.2]`, `ln_1e10_A_s [1.61, 3.045, 3.91]`, `mnu = 0.06` fixed (Planck_only_values.ini:4-17 [corrected from 127-141; the file has 68 lines]). Planck-only nuisance values fixed at the listed foreground numbers (lines 22-68 [corrected from 145-190]) are irrelevant for `plik_lite` (the `_mdb` of the commander/simall/plik_lite files lists only `A_planck` as `free_calib`, which supports the earlier inference [verified]).
Joint runs add: `camb mode cmb`, `lmax 2650`, `nonlinear both`, `do_lensing T`, `do_reionization T` (3x2pt_configs.py:46-51). [verified] These joint-run settings do not set `lens_potential_accuracy`, so `set_for_lmax(lmax=2650)` runs with lens_potential_accuracy = 0 (CAMB default: max_eta_k = 2.5*(lmax + 150) = 7000 instead of >= 18000, nonlinear lensing only through `NonLinear_both`), a lower-accuracy lensed-C_l setting than `Planck_only.ini` (1.0); only the Planck-lensing add-on in 3x2pt_ext_data_configs.py:68 sets it to 1. [verified] Every kcap CAMB call, in any mode, goes through `p.set_matter_power(redshifts=z, kmax=...)`
(camb_interface.py:324), which re-enables the transfer-function calculation: in `Planck_only.ini` (mode = cmb, zmax and nz at their module defaults 3.01 and 150, kmax = 1.2, nonlinear = lens) CAMB therefore also computes linear matter power at 150 redshifts every call and `camb_interface` writes `matter_power_lin` that nothing uses. CMB lensing add-on uses `lens_potential_accuracy = 1`, `tau = 0.055` fixed.

### 3.9 Reference tables for validating a re-implementation (computed by me with a fresh numpy implementation of 2.1)

Coefficient files (rows A, B, C = coefficients of (ln k)^2, ln k, 1; columns c0..c5 of poly2(n_s, omch2/h)):

    b2: A: 0.5628861256 -0.9553686702 -2.547460246 0.33526972 2.080580728 2.054045174
        B: 5.031424975 -2.711370504 -5.521082325 0.8855493505 5.784427695 3.903627908
        C: -0.4528361048 -2.208331776 -5.693046324 0.2434621343 1.485884066 4.122002559
    g2: A: -0.02602659138 -0.01114663864 0.8813154952 0.02261063623 0.09054636111 -1.032800833
        B: 3.451785031 -0.3408714858 1.691399268 0.04791497226 -0.189591498 -2.465691848
        C: 0.2647921464 -1.856465283 -3.680857178 0.5062176066 1.50060236 2.39213751
    g3: A: -0.01041776093 -0.003976297741 0.8138485744 0.01032723946 0.109961338 -1.006772233
        B: 3.432653559 -0.3187241028 1.17065563 0.01304505358 -0.0551902014 -2.060797267
        C: -1.808048506 -1.56151633 -3.08409227 0.4880724973 1.499536316 1.928633333

Test values for n_s = 0.97, omch2 = 0.12, h = 0.7 (y = 0.171429), k in h/Mpc, N_X in (Mpc/h)^-3:

    k:    0.01       0.05       0.1        0.3        1.0       3.0     10.0     20.0
    N_b2: 1.794e-9   1.057e-6   1.454e-5   7.937e-4   5.112e-2  1.874   77.99    601.6
    N_g2: 6.420e-7   3.250e-5   2.134e-4   5.340e-3   0.2539    11.67   1080.6   17160
    N_g3: 1.842e-7   7.791e-6   4.761e-5   1.070e-3   4.568e-2  1.919   162.6    2464.6

Other test numbers: Hartlap factor 0.958476 (p = 84, n_mocks 2048); GGL band centres in 2.5; z grid has 150 nodes with z[25] = 0.5.

---------------------------------------------------------------------------------------------------

## 4. Quirks, approximations and deviations from textbook

Q1. QUIRK, curvature is not implemented consistently in this snapshot (affects all `oLCDM`/`oCDM` results). The C Limber integrand and the W kernel are written for general K via `f_K` (limber.c:171; kernel.c:57,153; utils.c:10-19), but (i) the Python `limber()` call never passes K (`Spectrum.compute`, project_2d.py:169, so `config.K = 0.0` by default, limber.py:160-169); (ii) the W kernel reads `cosmological_parameters/K` with default 0 (kernel.c:225); (iii) nothing in the kcap-built pipeline writes `K` (the `consistency` module that would compute `K = -h^2 omega_k/c^2` is not in the module dictionary, run_kcap.py:406-770; consistency.py:88; camb_interface.py writes no K); (iv) `d_m` is filled with CAMB's `comoving_radial_distance` rather than the transverse comoving distance (camb_interface.py:384-385). Net effect: for Omega_k != 0 the projection uses flat-space geometry with the radial distance. The correction is of order (sqrt(|Omega_k|) chi H0/c)^2/6; for Omega_k = 0.4 and chi ~ 2000 Mpc/h that is about 0.42^2/6 = 3 per cent in chi. Even if K were supplied there would be a unit problem (K from consistency in Mpc^-2 vs chi in Mpc/h). Whether the published KiDS-1000 oLCDM chains used a different build is unknown (Open question 4). [verified] The geometry statements were re-checked: `K` is read at kernel.c:225 (default 0.0), `limber()` never sets it (limber.py:169 default K = 0.0), `consistency.py:88` is the only writer of `K` in the tree (`K = -hubble^2 omega_k / c^2` with hubble in km/s/Mpc, so K is in Mpc^-2) and the hand-written BOSS_only / BOSS_KV450 / Planck_only inis include `consistency` while run_kcap.py never does. The 3 per cent estimate re-computes as sqrt(0.4)*2000/2997.92 = 0.4219, squared/6 = 2.97 per cent.

Q2. QUIRK, S_8 sampling vs derived S_8 for massive neutrinos. The sampled `S_8_input` converts to sigma_8 with Omega_m,pre = (omch2 + ombh2)/h^2 (excludes nu; sample_S8.py:23-33), then A_s is set so that CAMB's sigma_8 reproduces it (sigma8toAs.py), but the derived S_8 = sigma_8 sqrt(Omega_m/0.3) written after CAMB uses CAMB's omegam which includes massive neutrinos (camb_interface.py:421). For nuLCDM with mnu up to 3 eV (Omega_nu h^2 ~ mnu/93.14 = 0.032) the two S_8 differ by a factor sqrt(1 + 0.032/0.14) ~ 1.11. Related: `sigma8toAs.py` calls `set_cosmology(... mnu=mnu)` without `neutrino_hierarchy`, while the main CAMB call uses `normal` (camb_interface.py:83) and `DoLateRadTruncation=True`, `high_precision=False` (sigma8toAs.py:36-38); for mnu = 0.06 the effect on sigma_8 is expected to be small but is not quantified in the repo. [verified and quantified, CAMB 1.5.8, h = 0.7, ombh2 = 0.0225, omch2 = 0.12, n_s = 0.97, target sigma_8 = 0.8, replicating sigma8toAs.py then the main call with `neutrino_hierarchy = 'normal'`, high_precision on, kmax 20 Mpc^-1 and no non-linear model): realised/target sigma_8 = 1.00067 (mnu 0.06), 1.00259 (0.3), 0.96535 (1.0), 0.84021 (3.0 eV). The reason is that `sigma8toAs.py:43` calls `set_cosmology(..., mnu=mnu)` without a hierarchy, which in CAMB means ONE massive eigenstate plus 2.03 massless species (num_nu_massive = 1), whereas the main run uses `normal` = three massive eigenstates (num_nu_massive = 3, massless 0): at fixed A_s the fiducial sigma_8 are 0.6755 (default) vs 0.6515 (normal) at mnu = 1 eV and 0.5718 vs 0.4797 at 3 eV. So for the nuLCDM extension with mnu up to 3 eV the sampled `S_8_input` is NOT the S_8 of the model that is evaluated (a 3.5 per cent sigma_8 error already at 1 eV), and the sigma8toAs fiducial run is never consistent with the main CAMB call beyond mnu ~ 0.3 eV. The remaining 0.07 per cent at mnu = 0.06 comes from low vs high precision (7.0e-4 for `high_precision = False, k_per_logint = 5, kmax = 1.2` against the main run's defaults), and it is present in every run, including the 3x2pt fiducial.

Q3. QUIRK, `sigma8toAs` requires a second full CAMB transfer run per likelihood call (kmax 1.2, k_per_logint 5). It is necessary because the sampling parameter is S_8 rather than A_s, and the dependence of sigma_8 on the other parameters is not analytic.

Q4. QUIRK, the "growth" used to build reduced kernels is derived from P_gm itself, which is piecewise in z. D(z) = sqrt(P_gm(z,k_g)/P_gm(0,k_g)) at k_g ~ 1e-3 h/Mpc; there F_s = 0 so D_gm(z)^2 = b1_s P_nl(z,k_g)/(b1_low P_nl(0,k_g)), which has a jump of factor b1_high/b1_low at z = 0.5. Analytically D cancels in the integrand (section 2.2), but the cancellation is only exact if the spline of Wr = W D/sqrt(chi) and the spline of D are evaluated consistently; between the nodes around z = 0.5 an Akima spline of a stepped D introduces a local mismatch. Whether this is numerically relevant for the lens bin edges (low-z bin ends at z = 0.5, high-z bin starts there; section 3.1) was not tested. It is a risk, not a demonstrated bug. [verified] Additional facts: in the shipped lens n(z) the two histograms end/start exactly at z = 0.5 (Z_MID 0.495 -> 0.505, heights 6.96 -> 0 for bin 1 and 0 -> 6.39 for bin 2), and bicubic interpolation of P_gm itself across the step between the CAMB z nodes 0.48 and 0.50 (3.2) touches 13.9 per cent / 12.8 per cent of the unit-normalised bin 1 / bin 2 n(z) mass, so the leakage of the high-z bias into bin 1 (and vice versa) is bounded by the b1_high/b1_low jump times roughly these fractions; it was not computed. D(z) at k_g is the CAMB linear-regime value times sqrt(b1_s) because the first CAMB k above 1e-3 h/Mpc is below the 0.01 cut of the fit term (limber.py:118-128, p_gm_fit.py:33).

Q5. QUIRK, hard k cut for P_gm. `matter_galaxy_power` is produced before `extrapolate_power` and `sections` is not set for it, so it stops at k = 21/h h/Mpc (25.6-32.8 h/Mpc; the text originally said 20 h/Mpc, which confused CAMB's 1/Mpc `kmax` with h/Mpc [corrected]), whereas `matter_power_nl` goes to 500 h/Mpc. For k beyond the grid the 2D spline evaluation fails and the integrand returns 0 (limber.c:141-146), i.e. the high-l tail of GGL, magnification-lens and IA terms (the IA term uses `k <= 0.999 k_max(P_gm)`, linear_alignments_interface.py:73-77) is truncated at l = k_max chi, with k_max = 21/h h/Mpc this is about 1.5e4-2.0e4 for chi = 600 Mpc/h and 3.1e4-3.9e4 at chi = 1200 Mpc/h [corrected from 1.2e4 and 2.4e4]; for the lens support (chi of about 570-1900 Mpc/h for z = 0.2-0.75, flat LCDM, h in [0.64, 0.82]) this is above the LHIGH = 1e4 limit of the band-power integral everywhere, so the cut is harmless for the band powers (it does truncate the projected C_l themselves above l about 1.5e4). The band-power integral uses l up to 1e4 (section 2.5). So the cut only affects the far tail of the window (INFERRED: negligible, unquantified).
The magnification-shear term uses P_nl and is not affected.

Q6. QUIRK, no positivity clipping in `p_gm_approx` while the band-power code log-extrapolates C_l (function_cosebis.cc:373, `exp(log(y2)+...)` [corrected from 372-373]) and returns 0 on NaN (line 272). With prior corners (b2 in [-4,8], gamma3 in [-8,8], b1 up to 9) nothing prevents the second term from driving P_gm negative at large k: N_g3 grows from 0.046 at k = 1 to 2.5e3 at k = 20 and 1.8e4 at k = 32.8 (table 3.9 and 2.1 [verified]) while P_lin^2 falls steeply, so the relative size of the correction to b1 P_nl at high k is not bounded by construction (not evaluated here). `utils/galaxy_bias.py` does clip (`p_k_gm[p_k_gm<0] = 0`, lines 46, 74) but is not wired into the kcap-generated pipeline. Negative C_l inside the table is interpolated fine by the cubic spline, but any extrapolation with y <= 0 gives NaN -> 0.

Q7. QUIRK, the polynomial fits are extrapolated far outside any documented validity. The fit is evaluated for all k >= 0.01 h/Mpc up to the CAMB k_max of 21/h h/Mpc (25.6-32.8 h/Mpc; originally written as 20 h/Mpc [corrected]) without limit, and the repo does not record the k-range or cosmology-range of the fitting grid (Open question 3). The coefficient tables are named `parameter_gridfit_*` (fit over a grid in (n_s, omega_c h^2/h) per k) but the grid bounds are not given.

Q8. QUIRK, cosmology dependence of the fit is only through (n_s, omch2/h) and P_lin. No dependence on w, wa, Omega_k, Omega_nu or on z beyond P_lin(z)^2 and the two-bin bias. In extended cosmologies this is an uncontrolled extrapolation of the GGL theory (INFERRED from p_gm_fit.py:21-37).

Q9. QUIRK, flat-sky Limber with nu = l + 1/2, no extended-Limber or reduced-shear correction. The `ext_order` argument is hard-wired to 0 (limber.py:176) and there is no (l+2)!/(l-2)! factor for spin-2 fields. For GGL (spin-2 shear x spin-0 position) the exact full-sky relation between the lensing potential and the shear introduces factors of order 1 + O(1/l^2); I expect these to be well below 1 per cent for the l > 100 band powers retained, but this is an expectation from the order of magnitude, not something verified in the repo. [verified] The exact flat-sky-to-full-sky ratio sqrt((l-1)l(l+1)(l+2))/(l+1/2)^2 = 1 - 1.25/(l(l+1)) + ... evaluates to 0.99988 at l = 100 (the lowest band edge) and is closer to 1 above, so the omitted factor is about 1e-4 for the retained band powers, far below the 1e-3 Limber tolerance; the first-order Limber (ext_order = 0) error is a separate and larger question that is not quantified here.

Q10. QUIRK, the pipeline writes the first (k_h grid) arrays twice (`block[output_section, "p_k"] = P_gm` and `block.put_grid(...)`, p_gm_approx_interface.py:67-72). Harmless.

Q11. [corrected: the original claim here was WRONG and is retracted] `Planck_settings` in all 3x2pt+Planck scripts set `camb mode = cmb` (runs/3x2pt/3x2pt_configs.py:46-51 and the extended scripts). The earlier text argued that `config['WantTransfer'] = mode in [transfer, all]` (camb_interface.py:53) plus the `if p.WantTransfer` guard (line 396) means no matter power spectra are written. That ignores line 324: every call does `p.set_matter_power(redshifts=z, nonlinear=..., kmax=...)` AFTER building the CAMBparams, and CAMB's `set_matter_power` unconditionally executes `self.WantTransfer = True` (camb/model.py:700 in 1.5.8; also sets `Transfer.high_precision = True`, `Transfer.kmax`, `Transfer.PK_redshifts`). The guard at line 396 tests `p.WantTransfer`, which is therefore True in EVERY mode (the `get_background` branch at line 340 is dead code as long as `extract_camb_params` runs `set_matter_power`). So `mode = cmb` WITH the kcap redshift/kmax settings does produce `matter_power_lin/nl` for the lensing part, and the joint scripts are consistent with this `camb_interface.py`; the settings `Planck_settings` also keep `nonlinear = both` so that both nonlinear lensing and nonlinear P(k) are on. The side effect is wasted work in CMB-only runs (3.8). Open question 4's second half (a differing local CAMB interface) is therefore not needed to explain the scripts.

Q12. QUIRK, `runs/config/Planck_only.ini` is internally inconsistent: `values = runs/config/Planck_only_lite_values.ini` (line 15) does not exist; the file shipped is `Planck_only_values.ini`. Also `[DEFAULT] boss_path` is commented out but unused. `likelihoods = PLANCK2018` (line 18) matches the C code output `PLANCK2018_LIKE`.

Q13. QUIRK, Gaussian BOSS covariance treatment. Hartlap correction is applied to BOSS (n_mocks 2048) but not to the KiDS covariance (mini_like.py has none). The BOSS "sigma" columns in the data files are ignored. `mini_BOSS_like.setup` takes `n_wedge` and `n_points_full` from the last data file read (lines 20-22), assuming all bins have identical shape. `points_range` read failure falls back to `[4, 32]` silently via `BlockNameNotFound` (lines 35-38).
When `keep_theory_vector` is true (run_kcap.py:750 sets it True for `BOSS_like`), every likelihood call draws `np.random.multivariate_normal(mu, cov)` (an 84x84 SVD) and writes five arrays per bin into the datablock (lines 77-83). It is only needed for mock generation and is wasted in sampling.

Q14. QUIRK, the b1 term uses the non-linear (HMcode/halofit) matter power spectrum including the baryon feedback parameter A, i.e. the lens-galaxy cross spectrum inherits baryonic suppression/boost with `b1` constant. This follows from formula (a) and the choice of `matter_power_nl`.

Q15. QUIRK, `local_lag_g2` coupling: gamma2 = -(2/7)(b1-1) is used in GGL via the module default and in the wedges module via `local_lag_g2 = T` (run_kcap.py:500). If someone sets `local_lag_g2 = F` for one but not the other, the GGL and BOSS bias models decouple silently (the "zero_ho" robustness test in 3x2pt_configs.py:171-179 sets only `--set-keys wedges local_lag_g2 F` plus zeros for `gamma2_bin_*`; it does NOT set `approximate_P_gm local_lag_g2`, which stays at its code default True, so GGL would still use gamma2 = -(2/7)(b1-1) while the wedges would use gamma2 = 0 [corrected: the earlier text said it sets both]; the entry is a commented-out config of a run type EE that has neither module, so it is moot as shipped).

Q16. QUIRK, band-power input extrapolation. `function_cosebis::extrapolate` for x below the table uses nodes [1] and [2] (not [0] and [1]) (function_cosebis.cc:366-367) and extrapolates ln y linearly in ln l; since the table starts at l = 1 and LLOW = 0.1 this affects only the window below l = 1.

Q17. QUIRK, scale-cut angles. Theory band-power "angles" are recomputed as geometric centres (scale_cuts.py:140-144), while the data angles come from the fits file. If the fits file stores effective l values that differ from geometric centres, theory and data would be cut by different criteria, and nothing checks that the cut vectors have equal length except the final dot product. [verified] For the shipped blind-C file the stored ANG values equal the geometric centres to all printed digits (118.44199, 166.15641, 233.09262, 326.99411, 458.72386, 643.52102, 902.76383, 1266.44273), so the two cuts coincide.

Q18. QUIRK, bug in `camb_interface.py:200-201` (`name.de_equation_of_state` should be `names.`); the tabulated-w option `use_tabulated_w` would raise NameError. Not used in kcap runs.

Q19. QUIRK / limitation, with `DarkEnergyFluid` and `wa != 0`, CAMB (version 1.5.8 installed locally, `dark_energy.py:96-100`) raises "fluid dark energy model does not support w crossing -1" when `w < -1 - 1e-6` or `1 + w + wa < 0`. The original CAMB used for the publication is unknown, but a w0-wa extension with kcap's default `use_ppf_w=False` would hit this for w < -1 with wa != 0 (the sampled wCDM range [-3,-0.33] with wa = 0 passes because the check is only evaluated when `wa` is non-zero). [verified] The exact condition is `if self.wa and (self.w < -1 - 1e-6 or 1 + self.w + self.wa < -1e-6)` (dark_energy.py:98-100), and the base-class `validate_params` (called from `set_params`) additionally raises `w + wa > 0` ("w>0 at high redshift"), which is irrelevant for the sampled box but would reject w0-wa points with w0 + wa > 0.

Q20. QUIRK, `run_kcap.py:591-593` pass `theta_max = xi_theta_min` to `bin_xi_plus/minus/ggl`. Those modules are removed from every 3x2pt run type; harmless here but would produce wrong real-space GGL if re-enabled. `cl2xi_ggl` (Nicaea `corr_type = 2`) is the real-space alternative and is likewise cut.

Q21. Approximation, no cross-covariance between lensing, GGL, BOSS clustering and Planck beyond what the KiDS fits covariance contains (INFERRED).

---------------------------------------------------------------------------------------------------

## 5. Strengths

- Compact, cheap GGL bias model: formula (a) is O(n_z n_k) (a handful of exponentials of a quadratic in ln k), needs only the CAMB linear and non-linear power already computed, has 3 bias parameters per lens bin (b1, b2, gamma3) with gamma2 tied by local-Lagrangian relation, and the same parameter set drives both the wedges and GGL, giving a consistent joint bias treatment.
- Clean separation of data-vector assembly (`scale_cuts` + `mini_like`) from theory; scale cuts expressed per statistic and per tomographic pair as plain ini keys; bandpowers avoid the numerical problems of Hankel transforms to real space and are insensitive to the small-scale part of C_l (J_2 window, apodised).
- Magnification bias, IA-GGL contamination and tidal/second-order bias are all included in the GGL model (formulas (g), (i)).
- Correct Hartlap treatment of the mock-based BOSS covariance, ln L with no determinant (constant covariance).
- Reproducible orchestration (`run_kcap.py` records git state, command line, config, values and priors per run; the run types make probe subsets simple to select).
- Planck and CMB lensing interfaces are thin wrappers on `clik` and on the cobaya lensing package; unit conversions are explicit (C_l and l(l+1)/2pi), and are consistent between CAMB output, the clik wrapper and cobaya (checked in 2.8).
- Extended cosmologies need no new code paths: a single CAMB call handles w, wa, omk, mnu; priors are changed from the command line.

## 6. Weaknesses and risks

Correctness:
- Q1 (curvature not implemented in the projection), Q2 (S_8 definition inconsistency with massive neutrinos), Q19 (CAMB fluid DE limits for wa). [corrected: Q11 (`mode=cmb` incompatible with the lensing pipeline) was retracted, see Q11; replaced by the verified Q2 finding that the S_8 -> A_s calibration run uses a different neutrino setup than the main run, which is a 3.5 per cent sigma_8 error at mnu = 1 eV and 16 per cent at 3 eV.]
- Q6/Q7/Q8: the `P_gm` approximation has no recorded validity domain, no clipping of negative values, and no w/Omega_k/Omega_nu dependence. Its accuracy against the PT model it imitates is not tested anywhere in the repo (only the prototype script comparing plots; `tests/` has no P_gm test, searched).
- The BOSS model that provides the clustering part and calibrates the P_gm fit is not in the repo (`../kcap_boss_module`), so the central 3x2pt result is not reproducible from this checkout.
- The KiDS data vector and covariance are NOT external for the production blind-C file [corrected]: the fits file is in `runs/3x2pt/data_iterated_cov/.../data/KiDS/`, and the GGL part is confirmed as 22 points (142 total) in 2.6. What remains external is `Cat_to_Obs_K1000_P1` (other blinds, other covariance variants, `deltaz.asc`).

Accuracy:
- Limber with 1e-3 relative tolerance and Akima-interpolated 1024-node kernels; P is interpolated with a bicubic spline of P (not ln P) over a CAMB k grid of 197 nodes [corrected from "unknown density"; measured with CAMB 1.5.8, 3.1].
- Step at z_sep in the 3D spectrum interacts with the growth-based reduced kernels (Q4).
- The hard truncation of P_gm at k = 21/h h/Mpc (25.6-32.8 h/Mpc) [corrected from 20 h/Mpc] (Q5); this lies above every k reached by the band-power integral for the lens support.

Maintainability and portability:
- Heavy dependency stack: modified CosmoSIS fork, compiled C/C++/Fortran modules (GSL, clik, MultiNest, Nicaea, ReACT, HMx), external BOSS module, cobaya lensing package, absolute relative paths such as `../Cat_to_Obs_K1000_P1` and `../kcap_boss_module`.
- Inconsistent or stale configs (Q12 missing values file; Q11; `run_kcap.py:1129` tests `xip_2D_c_term_file` twice).
- `limber.py` uses `np.int` (removed in recent numpy, limber.py:8) and relies on `ctypes` with raw GSL pointers; `camb_interface` has a version-dependent neutrino warning check (lines 43-45).
- Strings used as dictionary keys for physics options (`z_interpolation`, etc.) are validated only at setup.

Performance (structural reasoning; the CAMB numbers were measured in this pass with CAMB 1.5.8 on this machine, everything else was not):
- [verified] Measured CAMB cost split for the kcap settings: transfer functions 0.39 s, HMcode-2016 non-linear spectrum at 150 z x 197 k 2.8 s (takahashi: 0.38 s), the extra sigma_8 run 0.07 s, `get_BAO` over 6000 z 0.05 s; so the sigma8toAs run is about 2 per cent of the CAMB time and HMcode at 150 redshifts is about 85 per cent of it.
- Per likelihood call: two CAMB runs (main transfer with kmax=20 Mpc^-1, 150 z; plus the sigma_8 run), up to 45 Limber spectra x 400 ell x (adaptive GK61 integrand evaluations with two spline lookups), 3 band-power types, one BOSS PT/wedge evaluation with a 600 s timeout, and for Planck a full lensed CMB with lmax = 2650.
- MultiNest with 500 live points, efficiency 0.3, and tolerance 1e-2 means O(10^6) such calls per chain; fast/slow parameter ordering is not exploited (fast_slow=F in the sample ini files).

## 7. Concrete efficiency opportunities (rough, unmeasured estimates)

1. Vectorised, fixed-grid Limber (numpy/numba): evaluate P(k=(l+1/2)/chi_m, z_m) once on an (n_ell x n_chi) grid (e.g. n_ell ~ 100, n_chi ~ 300-500 Gauss-Legendre or Simpson nodes in chi) and obtain every bin-pair spectrum by matrix products with the stacked kernels. This replaces about 18000 adaptive 1-D integrals (45 spectra x 400 ell) per call, each needing hundreds of spline evaluations, by ~3e4-5e4 interpolations. Expected speed-up of the projection stage: 10-100x (estimate from operation counts, would need benchmarking). The 1e-3 tolerance of the reference can be matched with a modest grid because the lens kernel is a histogram only 0.3 wide in z.
2. Reduce the ell grid: the band powers only need C_l up to the point where the apodised J_2 window is negligible. The reference computes 400 points from 1 to 3e4 although cuts keep bands centred below ~1270; use ~100 log-spaced points to 1e4 (LHIGH is 1e4 anyway). Roughly 3-4x fewer integrals.
3. Linear decomposition of the GGL model in the bias parameters. By (a)-(c), C^{gE}_{ij} is affine in (b1, b2, gamma3) for fixed cosmology, nuisance n(z), IA and baryon parameters: P_gm = b1 [P_nl + (2/7) N_g2 P_lin^2] + b2 N_b2 P_lin^2 - gamma3 N_g3 P_lin^2 - (2/7) N_g2 P_lin^2. Four Limber/band-power evaluations per (lens, source) pair give the GGL vector for any bias values by a dot product. When only bias parameters change (e.g. in Gibbs-like moves, PolyChord fast parameters, profile likelihood), the GGL cost drops to near zero. In MultiNest all parameters change each call so the gain is limited to avoiding recomputation within a call.
4. Cache the cosmology-only products: N_X(k) depends on (n_s, omch2/h, k) only (not on z or on lens bin); the reference computes them twice (low-z and high-z) per call (p_gm_approx_interface.py:60-61). Compute once (the gain is tiny, micro-seconds, but it is free). Also compute P_gm only for z within the lens support (0.2-0.75) ~ 30 of 150 z nodes.
5. [corrected: the quantitative estimate below is wrong, measured 0.07 s for the sigma8toAs-style run versus 0.39 s + 2.8 s for the main CAMB call, i.e. about 2 per cent, not 25-40 per cent; the better reason to touch sigma8toAs is correctness, see Q2: pass the same `neutrino_hierarchy` (and ideally the same accuracy settings) as the main call] Remove the `sigma8toAs` second CAMB run: one can obtain sigma_8(A_s fid) from the transfer functions of the main run if the sampling parametrisation is changed to A_s (or ln A_s) in the inner loop and S_8 is only a derived parameter (but this changes the prior). Otherwise run the first CAMB with a cheap accuracy setting (already done) or replace it with a fitted/emulated sigma_8(omch2, ombh2, h, ...) interpolation. Possible saving: of order 25-40 per cent of CAMB wall time (estimate) [corrected: about 2 per cent, see above].
6. CAMB: restrict `kmax` (in 1/Mpc, i.e. kmax = k_h h, not h/Mpc [corrected units]) to the largest k needed by GGL lenses (k = l_max/chi_min, ~ 1500/570 ~ 3 h/Mpc plus the tail), fewer z nodes (the reference computes 150; kernels require only z <= 1.5 plus coarse z up to 6 for the source kernel). Cost of CAMB's non-linear transfer scales roughly with the number of z nodes and k nodes; a reduction to ~50 z and kmax = 10 could give 2-3x (estimate). [verified] The dominant item is HMcode-2016 at 150 redshifts (2.8 s of 3.2 s measured); restricting the non-linear spectrum to z <= 2-3 (cosmic shear needs P_nl up to the source n(z) tail only at a negligible weight) or interpolating the non-linear boost between 20-40 redshifts targets the real cost, while GGL alone needs only z in [0.2, 0.785].
7. BOSS wedges: the module is slow enough to be given a 600 s timeout; it only depends on cosmology and (b1, b2, gamma3, a_vir) with an expensive PT part that depends on cosmology only. Caching the cosmology-dependent PT spectra and treating the bias terms as a linear/bilinear combination (bias expansion) is the standard route; can be done only with access to the module (NOT IN REPO).
8. `mini_BOSS_like`: set `keep_theory_vector=False` in sampling runs (saves the SVD and four arrays x 2 bins per call; ~ms), pre-factorise Psi via Cholesky (negligible).
9. Planck: [verified] in the CMB-only chain every call also computes the matter transfer functions for 150 redshifts (z up to 3.01, kmax 1.2) because `set_matter_power` is always called (3.8); set the module's `nz` to 1 for CMB-only runs. The lensed C_l with lmax = 2650 is the dominant joint cost. Use `lens_potential_accuracy`/accuracy boost reduction as in the Planck_only.ini (currently 1.0), or a neural emulator of C_l (CosmoPower-style) for TT/TE/EE/PP. For CMB-only chains, `plik_lite` makes the likelihood call itself negligible.
10. [verified additions] (a) Drop the `get_BAO` call and the `MU`/`rs_DV`/`F_AP` outputs from `camb_interface` for lensing-only runs (0.05 s, small). (b) Ship (or pre-generate) the band-power W(l) tables, otherwise the first call of a clean install performs a 100000-point extremum search per (band, l, Bessel order) (2.5). (c) The `Analytic` ini switch has no effect (2.5), so there is no faster/slower variant to select. (d) `keep_theory_vector=True` in run_kcap.py:750 is set for all BOSS runs including sampling, which is the only reason `np.random.multivariate_normal` runs every call (Q13). (e) Evaluate GGL only on the ~30 CAMB z nodes in the lens support and only for the 6 retained (lens, source) pairs (1,2), (1,3), (1,4), (1,5), (2,4), (2,5) [verified in 2.6]: the projection and band-power modules compute all 10 pairs and only `scale_cuts` discards the 4 cut pairs (1,1), (2,1), (2,2), (2,3), so 40 per cent of the GGL Limber and band-power work (and of the magnification and position-intrinsic work) is thrown away; moreover for lens 1 only the first 3 of 8 bands and for lens 2 only the first 5 survive the per-pair angular cuts.
11. Use of `fast_slow` ordering (BOSS_only.ini:`fast_slow=T`) with samplers that exploit it (PolyChord, Metropolis) so that bias and nuisance parameters (cheap) are moved without rerunning CAMB; MultiNest does not exploit it.

## 8. Open questions (could not be resolved from the repo)

1. The `wedges` module (`../kcap_boss_module/python_interface/cosmosis_module.py`) and the `CosmoMC_BOSS` window/rbands files (`BOSS.DR12_windows.txt`, `BOSS.DR12_rbands.txt`) are not in the checkout [corrected: the BOSS measurements and covariances ARE, under runs/3x2pt/data_iterated_cov/.../data/BOSS]. Hence: exact model equations for xi_w (PT scheme, RSD treatment, role of `a_vir`, `use_growth`, `num_ell`, window convolution), definition of the three wedges, `rs_DV`, `F_AP`, `fsigma_8` outputs, and how `timeout` behaves, are all unknown here.
2. [partly resolved] The blind-C KiDS-1000 fits file and `SOM_cov_multiplied.asc` are in the checkout, which fixes the data-vector length (22 GGL + 120 shear = 142), order, band angles and n(z) (1.1, 2.6). Still external: the other blinds and covariance variants, `deltaz.asc` (hence the numeric prior means of `nofz_shifts` cannot be re-derived, only read from the generated values.ini), `Cat_to_Obs_K1000_P1`.
3. How `parameter_gridfit_{b2,g2,g3}.dat` were obtained: fitting k-range, grid in (n_s, omega_c h^2/h), PT model used as target, redshift used, and accuracy of the fit versus the PT model. The file headers and `pgm_model_old.py` do not say.
4. Whether the published KiDS-1000 results used this exact `camb_interface.py` and `project_2d` build: no `K` is passed to the Limber code (Q1). [corrected: the `mode=cmb` argument (Q11) was retracted; set_matter_power re-enables the transfer calculation.] A differing local CAMB interface or a differently patched projection module may still have been used for the oLCDM chains.
5. [partly resolved] Number of CAMB k nodes at `kmax = 20` Mpc^-1: 197 with CAMB 1.5.8, k_h,max = 21/h (3.1); this may differ for the CAMB version used originally, which is itself unknown (installed CAMB here is 1.5.8, newer than the 1.0.x era hinted at by camb_interface.py:43-45; note that this version check is logically odd: `not M>1 and not m>0 and not v>9` fires only for 1.0.0-1.0.9, so for 0.x versions it never warns [verified]).
6. Whether CAMB's `get_sigma8` uses total matter or CDM+baryon fluctuations in the version used (CAMB's default for `get_linear_matter_power_spectrum` is `delta_tot`, not verified for `get_sigma8`); matters for the Q2 mismatch [verified: in the Q2 test the sigma8toAs-style and the main call both use the same `get_sigma8()`, so the 3.5 / 16 per cent mismatch is a hierarchy effect and is not caused by this ambiguity].
7. [resolved] `C1_RHOCRIT` = 0.0138731 (linear_alignments.py:16-46, 2.4).
8. Which version/data of the Planck lensing likelihood `PlanckLensingMarged` loads (package not in repo), and whether the Planck lensing marginalisation includes the CMB-lensing-only foreground model.
9. The size of the effect of the P_gm step at z_sep on the Limber integral (Q4) and of the k = 21/h h/Mpc cut (Q5; expected to be nil for the band powers, see Q5); neither is tested in `tests/` (tests exist for CCL/MontePython comparisons of cosmic shear, `tests/test_pipeline_against_CCL*.py`, but not for GGL or P_gm).
10. [resolved] Orientation convention of `GSLSpline2d(chi, ln k, p.T)` (limber.py:115): verified against `gsl_wrappers.py:127-136` and GSL's `za[j*nx+i]` layout, see 3.2.
11. [new] Whether the highz BOSS file (header "0.4 < z < 0.6") is the bin that the pipeline calls z_eff = 0.61 (1.1).
12. [new] The origin of `n_mocks = 2048` for the shipped Sanchez et al. covariances (2.7).
13. [new] The size of the P_gm-step leakage (Q4) was bounded (13.9 / 12.8 per cent of the n(z) mass lies in the interpolation cell) but not computed.

## 9. Fact-check log (adversarial second pass)

Method: every cited line was re-opened in the source; every formula was compared with the code (not with the literature); numerical claims were recomputed (numpy for the 3.9 table, band edges/centres, Hartlap factor, curvature estimate;
astropy/numpy for the shipped KiDS and BOSS files; standalone CAMB 1.5.8 for k-grid, timing and sigma_8 consistency). No kcap module was executed.

### 9.1 Confirmed without change (checked against source)
- Formulas (a)-(e) and the sign/tie conventions of the P_gm approximation (`p_gm_fit.py:21-37`, `p_gm_approx_interface.py:53-65`); coefficient files; strict `z < z_sep`; all 24 table values in 3.9.
- Limber integrand W_X W_Y P / chi^2 with nu = l + 1/2, kernel normalisations and prefactor `1.5 (100/c)^2 Omega_m` raised to `prefactor_power`, magnification factor `2(alpha_i - 1)` on the lens index, `GSLSpline` Akima / `GSLSpline2d` bicubic, GK61 `qag`, 1e-3 relative tolerance, 4096-interval workspace, 1024-node kernels, ell grid `logspace(0, log10 3e4, 400)`.
- IA: `bk_corrected` NLA formulas, `P_gI = P_GI P_gm / P_nl`, `A[j] = 1` weight in `add_intrinsic`; band-power edges (100, 140.285, ... 1500) and centres (118.442 ... 1266.443); BP_E = (BP_0 + BP_4)/2; W has no 2 pi; N_b = ln(l_max,b / l_min,b); cos^2 apodisation.
- Scale-cut logic and the 22 + 120 point count; Hartlap factor 0.958476; chi2 definitions with no determinant; Planck l(l+1)/2pi conversions; the P(k,z) grid (150 z, z[25] = 0.5 exactly, 4/49 spacing above z = 2); sample_S8 / sigma8toAs formulas; wCDM/CPL formula; the module order of EE_nE_w (identical to the checked-in generated ini).

### 9.2 Errors corrected (distinct statements)
1. KiDS fits file / COVMAT / NZ_LENS / NZ_SOURCE "NOT IN REPO": they are checked in (blind C); data vector 142 points confirmed (1.1, 2.6, 6, 8).
2. BOSS data and covariance "NOT IN REPO" and `n_points_full` "INFERRED 60": files are present, 32 rows, 96x96 covariance; with `points_range = 4 32` the upper cut list is empty (2.7).
3. CAMB `kmax = 20` is in 1/Mpc, not h/Mpc: the grid reaches 21/h h/Mpc (25.6-32.8), 197 nodes, same for linear and non-linear; consequences for the P_gm domain, Q5 thresholds (1.2e4 / 2.4e4 were wrong), Q7, section 6, Open Q9 (3.1, 3.2, Q5, Q7).
4. "k_per_logint not determinable": it is CAMB's default (`set_matter_power` passes None -> 0), and `set_matter_power` also forces `WantTransfer = True`, `high_precision = True` (3.1).
5. Q11 (mode = cmb gives no matter power) is wrong: retracted.
6. Q15: the "zero_ho" test does not set `local_lag_g2` in `approximate_P_gm`; also it is a commented-out config of a run type without P_gm.
7. `distances/H` is in Mpc^-1 (H/c), not km/s/Mpc (1.3).
8. `magnification_alphas.py:55` should be `:13`.
9. `extrapolate_power` adds 199 new nodes, not 200.
10. `compute_lss_parameters T` is set only in BOSS_only.ini / BOSS_KV450.ini, not by run_kcap.py.
11. Production sampler settings: the checked-in ini has `live_points = 1000`, `update_interval = 100`; "500 live points" is the current driver script.
12. `Planck_only_values.ini` line citations (127-141, 144, 145-190) were beyond the 68-line file; correct lines are 4-17, 21, 22-68.
13. `w` (BOSS-only) chain omitted `correlated_dz_priors` and `one_parameter_hmcode`.
14. `create_config` span is lines 417-771 (the dict) / 358 (the method), not 406-770.
15. Efficiency item 5: dropping sigma8toAs would save about 2 per cent of CAMB time (measured 0.07 s vs 0.39 + 2.8 s), not 25-40 per cent.
16. Q2 said the sigma8toAs/main CAMB mismatch is "expected to be small": it is small (0.07 per cent) only for mnu = 0.06; it is -3.5 per cent in sigma_8 at 1 eV and -16 per cent at 3 eV because sigma8toAs omits `neutrino_hierarchy` (default one massive eigenstate vs three).
17. Section 6 "22 points?" and the "performance (not measured)" statement, and Open Questions 1, 2, 4, 5, 7, 10 updated accordingly.
18. Q6 line citation `function_cosebis.cc:372-373` -> 373; Planck "ANG in arcmin" label in the fits file is generic (values are multipoles).

### 9.3 Additions (tagged [verified] inline)
- Data inventory (1.1) incl. highz-header observation, lens n(z) edges at z = 0.5, driver scripts' actual state, data/Planck clik lmax values.
- Fits-file ordering and angle facts; BOSS covariance ordering; condition numbers (1.1, 2.6, 2.7).
- CAMB timings (HMcode dominates), high-precision mismatch between main and sigma8toAs runs, extra `get_BAO` run, Planck-only wasted transfer calculation, joint-run `lens_potential_accuracy = 0` (2.8, 3.1, 3.8, 7).
- Band-power implementation facts: `Analytic` ini switch inert, Delta_x is a ln(theta) width (comment in run_kcap.py is wrong), W tables cached on disk and not shipped, ell > 1e4 unused, g-function copy used (2.5).
- C1_RHOCRIT = 0.0138731; NLA has no extra scalings; P_gI loses the last CAMB k node (2.4).
- Orientation of the 2-D spline resolved; P_gm step vs n(z) quantified (13.9 / 12.8 per cent of mass in the interpolation cell) (3.2, Q4).
- `polyval_2d_order2` scalar-meshgrid fragility; prototype hard-codes; fit evaluated up to k = 32.8 h/Mpc (2.1).
- `limber()` always passes `ext_order = 0`, `K = 0` (3.3); CAMB `omegam` includes neutrinos (numbers) (2.9); exact CAMB fluid-DE condition (Q19); extended-cosmology details (2.10); c-terms inert in the bandpower chain (3.7).
- Not used by kcap: `cosmosis-standard-library/bias/*` (binwise_bias, clerkin, constant_bias, no_bias are not referenced from run_kcap.py, runs/ or utils/); `utils/galaxy_bias.py` (modes `pk_gm`, `bias`, `linear_bias`) post-processes the external PT product `galaxy_matter_power_spectrum_pt` onto a z grid and clips negatives to 0, but no config in the checkout calls it. The generated ini has no `fast_slow` key at all (the sample inis set it to F or T explicitly).

### 9.4 Items I could not resolve
- Everything inside the `wedges` module and the BOSS window/band files (not in the checkout).
- Whether the highz BOSS file (header "0.4 < z < 0.6") corresponds to z_eff = 0.61.
- The origin (number of mocks) behind `n_mocks = 2048`, and the fitting domain/accuracy of `parameter_gridfit_*` (the PT products live in `/share/splinter/...`).
- The numerical size of the P_gm step leakage (Q4), of the first-order-Limber error for GGL, and of the HMcode-baryon coupling in b1 P_nl; none was evaluated.
- Whether the published oLCDM chains used a projection build that passed `K`; and whether results with other CAMB versions (before 1.0.10, or after 1.5.8) reproduce the k-grid and Q2 numbers (the Q2 numbers are CAMB 1.5.8 only).
- Whether CAMB's `get_sigma8` uses delta_tot or delta_nonu (it does not affect the Q2 inconsistency).
