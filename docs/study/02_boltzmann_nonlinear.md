# kcap study 02: Background, Boltzmann code, non-linear P(k), parameter transforms

Scope: the part of kcap that turns sampled parameters into (a) background
distances, (b) linear and non-linear 3D matter power spectra P(k,z), and
(c) derived quantities (sigma_8, sigma_12, S_8).

Reference tree: `/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap` (git HEAD
36b40c55, 2022-06-13). All paths below are relative to that root unless stated.
`CSL` = `cosmosis-standard-library`.

Evidence conventions used throughout:

* `file:line` = read directly in the kcap tree.
* `[CAMB-1.5.8 path:line]` = read in the locally installed pip CAMB 1.5.8
  (`/home/anut/.local/lib/python3.12/site-packages/camb/`). kcap does NOT pin a CAMB
  version (`conda_env.yaml`, `README.md:43` just say `pip install camb`), so
  anything that depends on library defaults is version dependent. I flag these.
* `[MEASURED]` = I ran stand-alone CAMB 1.5.8 calls (not kcap code) replicating
  kcap's settings, to read off grid sizes, timings, defaults. Machine: WSL2, 22
  logical cores. CAMB 1.5.8 is newer than what kcap was written against, so
  treat MEASURED numbers as indicative for a modern CAMB, not as kcap's historic output.
* `[LITERATURE-COPY]` = HMcode equations read from the HMcode Fortran copy that
  ships inside kcap (`CSL/structure/meadcb/mhmcamb.f90`), NOT from CAMB's own
  `halofit.f90`, which is a compiled binary here and was not readable.

**Adversarial verification pass (second reader).** Every file:line citation below was re-read against the
kcap tree and CAMB 1.5.8; corrections are made in place and additions are tagged `[verified]`. The main
[MEASURED] numbers (nk=194, k range, sigma_8 0.779557, sigma_12 0.753536, omega_m 0.250296, omnuh2, YHe,
S_8/A_s chain of 2.7, HMcode thread timings) were independently re-run and reproduce. Items still not resolvable
are listed in section 8 and in the new Appendix C.

Two pipeline variants matter and differ in details:

| Variant | Built by | Boltzmann chain |
|---|---|---|
| KV450 (fiducial) | `runs/config/KV450_fiducial.ini:11` | `sample_ln_As one_parameter_hmcode camb extrapolate_power ... derived_parameters` |
| KiDS-1000 3x2pt | `utils/run_kcap.py:406-...` (module config dict starts at :406; `sample_S8` :409, `sigma8toAs` :430, `one_parameter_hmcode` :447-450, `camb` :452-467, `extrapolate_power` :513-517; dict order = pipeline order, confirmed by the generated `runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline.ini:2`) | `sample_S8 -> sigma8toAs -> one_parameter_hmcode -> camb -> extrapolate_power -> ...` (`sample_ln_As` and `sample_S8_squared` are in the default cut list, `run_kcap.py:97-98`) |

---

## 1. Purpose, role, inputs and outputs

Role: the first physics stage. Everything downstream (Limber projection to C_ell,
Hankel transform to xi_pm, COSEBIs, band powers) consumes only the datablock
sections written here. In kcap's fiducial runs CAMB is used only in `mode=transfer`
(no CMB C_ell), so the Boltzmann solve is a pure matter-transfer-function calculation
(+ HMcode inside CAMB).

### 1.1 Modules in this subsystem

| Module | File | Job |
|---|---|---|
| `sample_ln_As` | `utils/sample_ln_As.py` | `A_s = exp(ln_1e10_A_s) * 1e-10` |
| `sample_S8` | `utils/sample_S8.py` | sampled `S_8_input` -> `sigma_8_input` |
| `sigma8toAs` | `utils/sigma8toAs.py` | `sigma_8_input` -> `A_s`, using a private low-accuracy CAMB run |
| `one_parameter_hmcode` | `utils/one_parameter_hmcode.py` | `eta0 = a_0 + a_1 * A` |
| `camb` | `CSL/boltzmann/pycamb/camb_interface.py` | main Boltzmann + HMcode + distances + sigma_8, S_8 |
| `extrapolate_power` | `CSL/boltzmann/extrapolate/extrapolate_power.py` | log-log linear extension of P(k) to k_h = 500 |
| `consistency` | `CSL/utility/consistency/*` | fill/validate density-parameter relations (see 2.9; mostly NOT used as a pre-filler) |
| `derived_parameters` | `utils/derived_parameters.py` | S8 (and S_star) from sigma_8, omega_m |
| `camb_background` | `utils/camb_background.py` | background-only distances (not in the fiducial runs) |
| `sample_negative_mnu`, `sample_folded_prior`, `sample_S_star` | `utils/` | exotic reparametrisations (2.10) |

Datablock key case: CosmoSIS lower-cases keys, so `D_A` is stored as `d_a`,
`P_k` as `p_k`, `S_8` as `s_8`. Names below are written in the form used in the source.

### 1.2 Inputs read by `camb` (section `cosmological_parameters` unless noted)

| Key | Required | Meaning / transform | Source |
|---|---|---|---|
| `ombh2`, `omch2` | yes | physical baryon / CDM densities | `camb_interface.py:297-298` |
| `h0` (little h) | one of h0 / `hubble` / `cosmomc_theta` | `H0 = 100*h0`; `hubble` (km/s/Mpc) wins if present; else `cosmomc_theta/100` is passed to CAMB as `cosmomc_theta` | `:263-284` |
| `n_s`, `A_s` | yes | primordial tilt, amplitude at pivot | `:216-221` |
| `omega_k` | optional | -> CAMB `omk` | `:263-272` |
| `w`, `wa`, `cs2_de` | optional | defaults -1, 0, 1 | `:204-207` |
| `mnu` | optional | -> CAMB `mnu` (eV, sum) | `:267` |
| `num_massive_neutrinos` / `massive_nu` | optional | -> CAMB `num_massive_neutrinos` | `:270-271` |
| `n_eff` | optional | -> CAMB `nnu` | `:268` |
| `standard_neutrino_neff`, `t_cmb`, `yhe`, `a_lens` | optional | -> `standard_neutrino_neff`, `TCMB`, `YHe`, `Alens` | `:263-272` |
| `tau` | only if reionisation on and `use_optical_depth=T` | | `:171-173` |
| `nrun`, `nrunrun`, `nt`, `ntrun`, `r` | optional | extra primordial params | `:212` |
| `halo_model_parameters/A` | optional | -> `HMCode_A_baryon` | `:225-228` |
| `halo_model_parameters/eta0` | optional | -> `HMCode_eta_baryon` | `:225-228` |
| `halo_model_parameters/logT_AGN` | optional | -> `HMCode_logT_AGN` | `:225-228` |
| `recfast/*` (11 keys) | optional | all default to CAMB `Recfast()` defaults | `:135-166` |

[verified] Code defaults that differ from the fiducial inis: `do_reionization` default is **True** and
`use_optical_depth` default is **True** (`camb_interface.py:85-86`), so any ini that does not set
`do_reionization = F` needs `cosmological_parameters/tau`; `nonlinear` default is `both` (`:61-63`);
`kmax` default 1.2 (`:129`); `zmax` default 3.01 (`:106`); `lmax` default 2600 (`:79`); `mode` default `all` (`:47`).
KV450 starting points: `halo_model_parameters/A` start 3.0 (range 2.0-3.13), `n_s` start 1.09, `ln_1e10_A_s` start 3.09,
`h0` start 0.7 (`KV450_values.ini:3-30`).

Fiducial KiDS values (`runs/config/KV450_values.ini`, `run_kcap.py:851-861`):

* KV450: sampled `omch2 in [0.01,0.99]`, `ombh2 in [0.019,0.026]`, `h0 in [0.64,0.82]`,
  `n_s in [0.7,1.3]`, `ln_1e10_A_s in [1.7,5.0]`, `halo_model_parameters/A in [2.0,3.13]`
  (all flat; `KV450_priors.ini` has no prior on A). Fixed: `omega_k=0, w=-1, wa=0,
  mnu=0.06, num_massive_neutrinos=1` (`KV450_values.ini:22-27`).
* KiDS-1000: sampled `omch2 in [0.051,0.255]`, `ombh2 in [0.019,0.026]`, `h0 in [0.64,0.82]`,
  `n_s in [0.84,1.1]`, `S_8_input in [0.1,1.3]`, `A in [2.0,3.13]` (start 2.6). Fixed:
  `omega_k=0, w=-1, mnu=0.06`. `num_massive_neutrinos` is NOT set (CAMB default 1 applies).

### 1.3 Outputs written by `camb` (shapes for the K1000 settings)

| Section / key | Shape / unit | Content | Source |
|---|---|---|---|
| `cosmological_parameters/{age,zstar,rstar,thetastar,DAstar,zdrag,rdrag,kd,thetad,zeq,keq,thetaeq,thetarseq}` | scalars | CAMB derived params (always overwritten) | `:345-347`; names `[CAMB-1.5.8 model.py:48]` |
| `cosmological_parameters/{h0,hubble,omnuh2,n_eff,num_nu_massless,num_nu_massive,massive_nu,massless_nu,omega_b,omega_c,omega_nu,omega_m,omega_lambda,ommh2}` | scalars | written only if absent from block; if present and `!np.isclose` only a warning | `:349-373` |
| `cosmological_parameters/cosmomc_theta` | scalar, = 100*theta_MC | only if absent | `:375-376` |
| `cosmological_parameters/sigma_8` | scalar | sigma_8 at the first (lowest-z) entry of the z grid | `:414` |
| `cosmological_parameters/sigma_12` | scalar | R = 12 Mpc (no h), same z index | `:418-419` |
| `cosmological_parameters/S_8` | scalar | `sigma_8 * sqrt(omega_m/0.3)` | `:421` |
| `distances/z` | (6000,) | `linspace(0,6,6000)` | `:378,381` |
| `distances/a` | (6000,) | `1/(1+z)` | `:382` |
| `distances/D_A`, `D_C`, `D_M`, `D_L` | (6000,) Mpc (NOT Mpc/h) | see 2.3. `D_M` is a copy of `D_C` | `:383-387` |
| `distances/MU` | (6000,) | `5 log10(D_L/Mpc)+25`; element 0 forced to NaN | `:388` |
| `distances/H` | (6000,) in Mpc^-1, i.e. H(z)/c | CAMB `h_of_z` [CAMB-1.5.8 results.py:1479-1486]; [MEASURED] h_of_z(0)=2.33495e-4 = 70/299792.458 | `:390` |
| `distances/rs_DV`, `F_AP` | (6000,) | from `get_BAO`; [MEASURED] element 0 is `inf` / 0 | `:392-394` |
| `matter_power_lin/{z,k_h,p_k}` | z (150,), k_h (nk,), p_k (150, nk) | linear P of delta_tot, z ascending, k in h/Mpc, P in (Mpc/h)^3 | `:398-399` |
| `matter_power_nl/{z,k_h,p_k}` | same shape | HMcode-corrected P | `:402-404` |
| `growth_parameters/{z,sigma_8,fsigma_8}` | (150,) | z ascending; sigma_8(z), f sigma_8(z) | `:406-412` |
| `cmb_cl/*` | only `mode=cmb/all` | lensed TT/EE/BB/TE in muK^2; PP, PT, PE scaled by l(l+1)/2pi | `:423-441` |

`matter_power_*` are stored with `put_grid(name, "z", z, "k_h", k, "p_k", P)` where
`P.shape == (nz, nk)`; the grid carries a sentinel key `_cosmosis_order_p_k` recording
(z, k_h) ordering so readers using `get_grid(sec, "k_h", "z", "p_k")` get the
transpose automatically (`cosmosis/datablock/cosmosis_py/block.py:1199-1231`, local CosmoSIS 3.12.1).

After `extrapolate_power`: same sections, `k_h` extended to 500 (nk -> nk+199), same z.

---

## 2. Mathematical description

### 2.1 Primordial spectrum and A_s

```
A_s = 1e-10 * exp(ln_1e10_A_s)                      (utils/sample_ln_As.py:8-10)
P_R(k) = A_s (k / k_p)^(n_s - 1)  [+ running terms if nrun, nrunrun set]
k_p = 0.05 Mpc^-1   (CAMB InitialPowerLaw default pivot_scalar; [CAMB-1.5.8], printed by probe; kcap only overrides if option pivot_scalar is given, camb_interface.py:81)
```
Tensors, vectors off (`do_tensors` default False, `:55-56`). Adiabatic initial conditions (`:65`).

Note `consistency_interface.py:49-50` offers a second route `A_s = 1e-10*exp(log1e10As)`
using a differently spelled key (`log1e10As`, no underscores); unused in fiducial runs.

### 2.2 Background

Dimensionless expansion rate (flat or curved, w0-wa dark energy; consistent with the
HMcode copy `mhmcamb.f90:1669-1708` and CAMB's `DarkEnergyFluid`/`PPF`):

```
E(z)^2 = H^2/H0^2 = (Om_c + Om_b)(1+z)^3 + Om_nu(z) + Om_gamma(1+z)^4 + Om_massless_nu(1+z)^4
         + Om_K (1+z)^2 + Om_DE f_DE(a)
f_DE(a) = a^(-3(1 + w0 + wa)) exp(-3 wa (1 - a)),   w(a) = w0 + wa (1 - a)   (mhmcamb.f90:1687-1708)
Om_i = omega_i / h^2 ,  Om_K = omk,  Om_DE = 1 - Om_K - Om_m - Om_rad (flat closure done inside CAMB)
```
`Om_nu(z)`: massive neutrinos treated by CAMB with the exact Fermi-Dirac integral
(density -> relativistic at early times, -> omnuh2/h^2 (1+z)^3 late).

Neutrino density from `mnu` ([CAMB-1.5.8 model.py:501-509]):

```
neutrino_mass_fac = 94.0641058 eV   [MEASURED constants.neutrino_mass_fac, scales as (TCMB/2.7255)^-3]
omnuh2 = mnu / neutrino_mass_fac * (standard_neutrino_neff/3)^0.75       (nnu >= standard_neutrino_neff, or hierarchy != degenerate)
omnuh2 = mnu / neutrino_mass_fac * (nnu/3)^0.75                          (otherwise)
```
[MEASURED] mnu=0.06, nnu=3.044 -> omnuh2 = 6.44867e-4. CAMB then converts omnuh2 into the
physical mass(es) with small velocity corrections internally (docstring `model.py:465-469`).
Reported `omega_m = (ombh2 + omch2 + omnuh2)/h^2` ([CAMB-1.5.8 model.py:572-574]), i.e. **includes
massive neutrinos**. `omega_cb` would exclude them.

Massless species: `num_nu_massless + num_nu_massive = N_eff`. [MEASURED] with mnu=0.06 and
default `num_massive_neutrinos=1`: eigenstates=1, degeneracy 1.014667, num_nu_massless=2.029333
(= 3.044 - 1.014667). BBN helium: `YHe` from CAMB's BBN interpolation table
(CAMB>=1.4: `PRIMAT_Yp_DH_ErrorMC_2021.dat`; before 1.4.0: `PArthENoPE_880.2_standard.dat`
[CAMB-1.5.8 bbn.py:12-13]); [MEASURED] YHe=0.24569 for ombh2=0.022.

### 2.3 Distances (Mpc, c = 299792.458 km/s)

```
chi(z)  = D_C(z) = c * int_0^z dz' / H(z')                       -> distances/D_C
D_M(z)  = S_k(chi) with S_k = sinh(sqrt(Ok) H0 chi/c)/(...) etc.  (CAMB internal; NOT written)
D_A(z)  = D_M/(1+z)                                               -> distances/D_A
D_L(z)  = (1+z) D_M                                               -> distances/D_L
H(z)/c  [Mpc^-1]                                                  -> distances/H
MU(z)   = 5 log10(D_L / Mpc) + 25,  MU[0] = NaN
```
QUIRK: `distances/D_M` is assigned `comoving_radial_distance` (= D_C), not the
transverse comoving distance (`camb_interface.py:384-385`). For Omega_k = 0 they coincide.
[MEASURED] omk=0.05, z=1: D_C=3232.39, true D_M = D_A*(1+z) = 3247.76 Mpc (0.5% off).
`project_2d.py:532` reads `distances/d_m` as comoving distance; downstream is therefore wrong
for omk != 0 unless that module corrects (not examined: other subsystem). Fiducial runs use omk=0.
[verified] `utils/camb_background.py:67` repeats the same `D_M = comoving_radial_distance` assignment. `project_2d.py:538-541` then
multiplies `chi_distance` by `h0` (Mpc -> Mpc/h), line 544 does the same for the optional `distances/CHISTAR`; it does not apply
S_k(chi) itself. Note this is a *transverse* vs *radial* distinction only for omk != 0; D_A and D_L written by kcap use CAMB's
curvature-correct values, so only the `D_M` key is mislabelled (how project_2d uses chi for omk != 0 was not examined).

`omega_lambda` written as `1 - omegam - omk` (`camb_interface.py:350`, comment "Doesn't include
radiation!"), i.e. exact Omega_Lambda + Omega_rad. Radiation is ~8.5e-5 (derived: ~4.15e-5/h^2).

`rs_DV, F_AP` are the CAMB BAO outputs `r_s(z_d)/D_V(z)` and `F_AP = (1+z) D_A H / c`
(`get_BAO`, [CAMB-1.5.8 results.py:259-270]); `get_BAO` calls `calc_background` on a COPY of the
params and recomputes the thermal history; cheap ([MEASURED] 10-50 ms for 6000 z).

### 2.4 Linear matter power spectrum

CAMB returns, for the **total matter** density contrast (`delta_tot`, CDM+baryons+massive
neutrinos; default `var1=var2=delta_tot`, [CAMB-1.5.8 results.py:731-778]):

```
P_lin(k, z) = (2 pi^2 / k^3) * P_R(k) * T_tot(k, z)^2        [units Mpc^3 internally]
P_h(k_h, z) = h^3 * P_lin(k = h k_h, z)   [(Mpc/h)^3],    k_h = k/h
```
with `T_tot` the transfer function of the total-matter density contrast per unit primordial
comoving curvature (normalised in CAMB's synchronous-gauge convention). A re-implementation must
reproduce CAMB's `T_tot` normalisation; the practical check is sigma_8 (2.6).

Verified numerically [MEASURED]: `get_sigma8()` (0.861756) is consistent with sigma_R of delta_tot
(0.861834, diff 9e-5, "numerical error" per CAMB docstring `results.py:808-810`) and differs from the
cold-matter value (delta_nonu: 0.865248). So `matter_power_lin` is **total matter**, and kcap's
sigma_8 is the total-matter sigma_8 (neutrinos included, scale-dependent suppression included).
At k_h~1, P_cb/P_tot = 1.0085 for mnu=0.06.
[verified] The 1.0085 was not reproduced for the Appendix A point (ombh2=0.022, omch2=0.10, h=0.7, mnu=0.06): CAMB 1.5.8 gives
P_cb/P_tot = 1.0106 at k_h=0.936, z=0. The value depends on omch2/f_nu; treat 1.0085 as cosmology-specific (cosmology not recorded).

### 2.5 Non-linear correction: CAMB HMcode

Selected through `camb.nonlinear.Halofit(halofit_version=..., HMCode_A_baryon=..., HMCode_eta_baryon=..., HMCode_logT_AGN=..., Min_kh_nonlinear=...)`
constructed directly with keyword arguments (`camb_interface.py:230-233`), NOT through
`.set_params(...)`. Consequence: fields not supplied take the compiled-library defaults
(`A=3.13`, `eta=0.603`, `logT_AGN=7.8`, `Min_kh_nonlinear=0.005` [MEASURED on 1.5.8 `Halofit()`]).

Version aliases ([CAMB-1.5.8 nonlinear.py:12-36]): `'mead' == 'mead2016'` (internal id 5, HMcode
Mead et al. 2016, arXiv:1602.02154), `'mead2015'`=8, `'mead2020'`=9, `'mead2020_feedback'`=10,
`'takahashi'`=4, `'original'`=1. Docstring (`nonlinear.py:75-77`): `A_baryon` and `eta_baryon`
are used **only** in `mead2015`/`mead2016`(`mead`); `logT_AGN` is used **only** in `mead2020_feedback`.

Baryon feedback parameterisation used by kcap:

```
mead / mead2016 :   A_baryon = halo_model_parameters/A
                    eta_baryon = eta0 = a_0 + a_1 * A,  a_0 = 0.98, a_1 = -0.12   (one_parameter_hmcode.py:12-15; fiducial KV450_fiducial.ini:104-105, run_kcap.py:447-450)
                    A = 3.13  ->  eta0 = 0.6044   (~ DM-only 0.603)
                    A = 2.0   ->  eta0 = 0.74
mead2020_feedback : logT_AGN (7.3..8.3 "wide" in runs/extended_cosmologies/3x2pt_ext_cosmo_configs.py:96-100); one_parameter_hmcode is cut there.
```
(The standalone HMcode copy states the same 1-parameter relation in a comment,
`mhmcamb.f90:148`; its own module uses `eta0 = 1.03 - 0.11 A` for `one_baryon_parameter`,
`CSL/structure/meadcb/module.yaml`. The test `tests/test_hmcode.py:118-168` uses yet another pair
(1.0, -0.5), for testing only.)

HMcode 2016 equations `[LITERATURE-COPY: CSL/structure/meadcb/mhmcamb.f90]` (lines in parentheses).
Do not assume CAMB's internals are identical in every numerical detail (see 2.5.3).

Notation: sigma(R,z) = rms linear fluctuation with a top-hat of comoving radius R (h^-1 Mpc),
`Delta2(k) = k^3 P_lin/(2 pi^2)`, `g(z) = D(z)/D(0)` scale-independent growth, `f_nu = omega_nu/omega_m`,
`Om_m(z) = Om_m (1+z)^3 / E^2(z)` (radiation neglected, `:1669-1721`), `rho_bar = 2.775e11 Om_m h^2 Msun Mpc^-3`
i.e. `2.775e11 * Om_m` in (Msun/h)/(Mpc/h)^3 (`:845-854`). Mass grid: 256 points log-spaced
M in [1, 1e18] Msun/h (`:490-492`). `M = (4 pi/3) rho_bar R^3` (`:601-611`).

```
Full:    P_HM(k)^alpha = P_2h(k)^alpha + P_1h(k)^alpha                         (halomod, :254-276)
alpha    = 3.24 * 1.85^neff, clipped to [0.5, 2]                                (:216-237)
P_2h     = P_lin(k) [1 - f tanh^2( k sigma_v / sqrt(f) )], floored at 0         (:910-933)
f        = 0.0095 * sigma_v100^1.37, clipped to [1e-3, 0.99]                    (:193-214)
sigma_v^2     = (1/3) int dlnk  Delta2_lin(k,z) / k^2                           (R=0 tophat; :1740-1860)
sigma_v100^2  = (1/3) int dlnk  Delta2_lin(k,z) / k^2 * W_th(100 k)^2           (R=100 h^-1 Mpc)
P_1h     = (1 - exp(-(k/k*)^2)) * (2 Delta_v k^3 / (3 pi)) * int dnu g(nu) R_v^3 |W(nu^eta k; R_v, c)|^2   (dimensionless Delta2 form; :935-991)
k*       = 0.584 / sigma_v                                                       (:154-171)
eta      = eta0 - 0.3 * sigma_8(z)         [eta0 = HMCode_eta_baryon]            (:132-152)
Delta_v  = 418 Om_m(z)^-0.352 (1 + 0.916 f_nu)                                   (:90-108)
delta_c  = [1.59 + 0.0314 ln sigma_8(z)] (1 + 0.262 f_nu) [1 + 0.0123 log10 Om_m(z)]   (:110-130)
nu       = delta_c / sigma(R,z);  g(nu) = 0.21616 [1 + (0.707 nu^2)^-0.3] exp(-0.707 nu^2/2)   (Sheth-Tormen in nu, :1648-1667)
R_v      = R / Delta_v^(1/3)
c(M,z)   = A_baryon * (1+z_f)/(1+z) * (g_w(z=10)/g_LCDM(z=10))^1.5               (Bullock + Dolag; :633-676)
           z_f from  g(z_f) = delta_c g(z)/sigma(0.01^(1/3) R, z)  (if RHS > g(z): z_f = z)   (:792-831)
neff     = -3 - dln sigma^2/dln R at R_nl (nu(R_nl)=1), clipped to [n_s - 4, n_s]  (:613-631)
W(k;R_v,c) = NFW transform, ks = k R_v/c:
           [cos ks (Ci((1+c)ks) - Ci(ks)) + sin ks (Si((1+c)ks) - Si(ks)) - sin(c ks)/((1+c) ks)] / [ln(1+c) - c/(1+c)], clipped to [0,1]   (:1597-1635)
sigma(R,z) = g(z) sigma(R,0) with sigma(R,0) from a 64-point table, R in [1e-4,1e3] log-spaced, cubic interpolation; cold-matter P for the halo model (:993-1057)
cold-matter ratio: P_cb(k) = P_tot(k) [Dcb/Dcbnu]^2 from Eisenstein-Hu 1999 approximations (Tcb_Tcbnu_ratio, :395-447)
```
`A_baryon` enters only through c(M,z); `eta0` only through the window-function mass dependence.

#### 2.5.0 [verified] Further details of the stand-alone HMcode copy (`mhmcamb.f90`)
* Linear power fed to it: `Delta2_lin = k^3 P_in/(2 pi^2)` taken at the input redshift of the table, then divided by `g(z)^2` to
  put it at z=0 (`fill_plintab_cosmosis`, `:326-386`, esp. `:374,382-386`); the cold-matter table is `Delta2 * Tcb_Tcbnu_ratio^2` (`:378`).
* **Growth factor g(z) is NOT an ODE solve.** It is the growth-index approximation
  `g(a) = exp( - int_a^1 Omega_m(a')^gamma da'/a' )`, `gamma = 0.55 + 0.02 (1+w)` if w<-1, `0.55 + 0.05 (1+w)` if w>-1, else 0.55
  (`growint`, `growint_integrand`, `:678-790`), tabulated and interpolated by `grow` (`:1723-1738`); `Omega_m(z)` ignores radiation
  and uses `Hubble2` of `:1669-1685`. The ~3e-3 agreement with CAMB's internal HMcode therefore includes this approximation.
* Which linear spectrum enters which quantity: `sigma_v`, `sigma_v100` (`dispint`, `:1740-1860`, integrand uses `p_lin(...,itype=0)`)
  and `sigma_8(z)` (`:501`, itype 0) use **total** matter; the `sigma(R)` table (`fill_sigtab`, `:993-1043`, itype 1) used for `nu`, `R_nl`,
  `neff`, `sigma(fM)` (concentration) and the halo mass function uses **cold** matter. `dispint` and `sigma` integrals use
  the substitution `k = 1/theta - 1` (h/Mpc) with 1e-3 relative tolerance (`acc=1d-3`, `:1756`, `:1201`).
* `find_pk` beyond the last CAMB k: `P(k) = P(kmax) (ln k/ln kmax)^2 (k/kmax)^(n_s-1)` (`:874-880`), i.e. NOT the kcap log-log
  extension of 3.4 (the stand-alone code never sees that, since it runs before `extrapolate_power`).
* The halo-model sigma(R) table is computed once at z=0 and rescaled `sigma(R,z)=g(z) sigma(R,0)` (`:1045-1057`); the concentration
  uses `f = 0.01^(1/3)` for the Bullock formation-time mass fraction (`:534`).
* Cosmic density `rho_bar = 2.775e11 * cosm%om_m` in (Msun/h)/(Mpc/h)^3 (`:845-854`); `Omega_m(z)` in Delta_v and delta_c is computed
  from the same `om_m` (whether `cosm%om_m` includes neutrinos was not checked).

#### 2.5.1 How the correction is applied to the CAMB spectrum
Not readable from the sources available: CAMB's compiled Fortran returns `matter_power_nl`
on the **same (z, k_h) grid** as the linear spectrum ([MEASURED] `k` arrays identical). Below
`Min_kh_nonlinear = 0.005 h/Mpc` the correction is documented as switched off
(`nonlinear.py:9`). Whether CAMB applies a ratio `P_HM/P_lin(cb)` to the total-matter linear
spectrum (the usual CAMB design) or replaces it is an open question (section 8).

#### 2.5.2 Accuracy claims inside kcap
`tests/test_hmcode.py:41-91`: stand-alone HMcode (`meadcb`) vs CAMB `mead` agree to rtol 3e-3 for
z<2 and 1.5e-2 for z>=2 (the authors left a printed WARNING about these tolerances). Also
`test_camb` (:115-189) shows CAMB `mead` reproduces the pipeline exactly (`np.allclose`, rtol 1e-5) when given the same
`A`, `eta` and when the CAMB parameter object is built with `WantCls=False` ("Setting WantCls
matters... causes 2e-3 differences in the linear power spectrum", `:151-152`; the earlier `:141-142` citation was wrong).
[verified] Caveats on these tests: (i) `test_meadcb` runs the stand-alone code with `feedback=False`, `one_baryon_parameter=False`
(`test_hmcode.py:60-61`), i.e. a **dark-matter-only** comparison (CAMB default A=3.13, eta=0.603), at w=-0.9, mnu=0.06, h=0.7,
A_s=2.1e-9, n_s=0.96, so the 3e-3 / 1.5e-2 tolerances say nothing about the baryon-feedback (A, eta0) path used in KiDS;
(ii) `test_camb` uses `a_0=1.0, a_1=-0.5` and A=2.0, hence **eta0 = 0.0** (`:118-120,142,168`) - a stress test of plumbing, not a physical value;
(iii) the CAMB reference in `test_camb` passes `halofit_version='mead'` via `NonLinearModel.set_params` (`:166-168`) while kcap constructs `Halofit(...)` directly (QUIRK 4.2).

#### 2.5.3 Caveat
The equations above describe Mead et al. 2016 via kcap's own HMcode copy. A bit-level
re-implementation of CAMB's `mead` requires CAMB's `halofit.f90` (grid/tolerance choices,
growth-factor source, treatment of massive neutrinos). That file is not in kcap.

### 2.6 sigma_8, sigma_12, S_8, f sigma_8

```
sigma^2(R, z) = int dlnk  Delta2_lin(k,z) W_th(kR)^2,    W_th(x) = 3 (sin x - x cos x) / x^3,   Delta2 = k^3 P_lin/(2 pi^2)
sigma_8(z)    = sigma(R = 8 h^-1 Mpc, z)   [CAMB get_sigma8, CAMB-internal integration; total matter]
sigma_12      = sigma(R = 12 Mpc, z=z_first); in h^-1 Mpc units R = 12*h  (h=0.7 -> 8.4 h^-1 Mpc)
                (call: get_sigmaR(R=12.0, z_indices=-1, hubble_units=False), camb_interface.py:418)
S_8           = sigma_8(z=0) * (Omega_m / 0.3)^0.5,   Omega_m = (ombh2+omch2+omnuh2)/h^2   (camb_interface.py:421)
f sigma_8     : CAMB's velocity-density definition (Planck 2015) [CAMB-1.5.8 results.py:871-882]
```
`z_indices=-1` and `sigma_8[0]` refer to the first element of the user redshift grid, which is
z=zmin (=0 in all fiducial inis). If `zmin>0` they would NOT be sigma_8(z=0) (QUIRK).
`get_sigma8()` returns decreasing-z order; kcap reverses it (`:407-408`) so that arrays are z-ascending,
matching `growth_parameters/z`.

`derived_parameters.py` (module `[derived_parameters] parameters = S8`) recomputes
`S8 = sigma_8 (omega_m/0.3)^alpha`, alpha=0.5 default (`:25-30`); `S_star = A_s/2e-9 (omega_m/0.3)^alpha`
(`:32-37`). In KV450 the ini lists `cosmological_parameters/S8` in `extra_output`; in KiDS-1000 the
`S_8` written by CAMB is used.

### 2.7 S_8 sampling: sample_S8 + sigma8toAs (KiDS-1000)

```
sample_S8:     sigma_8_input = S_8_input / (Omega_m,eff / 0.3)^alpha ,  alpha = 0.5 (default)
               Omega_m,eff = omega_m if present in block, else (omch2 + ombh2)/h0^2      (utils/sample_S8.py:23-34)
               S8_squared option: S_8_input := sqrt(S_8_2_input) first (:21-22)
sigma8toAs:    A_s = A_fid (sigma_8_input / sigma_8,fid)^2 ,  A_fid = 2.1e-9           (utils/sigma8toAs.py:28,54)
```
Exactness: P_lin is exactly proportional to A_s, so sigma_8^2 is too; the rescaling is exact for
the same CAMB configuration.

QUIRKS (all [MEASURED] on CAMB 1.5.8 for ombh2=0.0225, omch2=0.13, h=0.7, n_s=0.97, mnu=0.06, S_8_input=0.7458):

1. At the time `sample_S8` runs, `omega_m` does not exist (CAMB has not run), so the fallback
   `Omega_cb = (omch2+ombh2)/h^2 = 0.311224` (neutrinos excluded) is used. The final `S_8` (CAMB) uses
   `omega_m = 0.312541` (neutrinos included). Factor sqrt(0.312541/0.311224) = 1.00211.
2. `sigma8toAs` computes sigma_8,fid with a private low-accuracy CAMB run (kmax=1.2, `k_per_logint=5`,
   `high_precision=False`, no HMcode); the main run uses kmax=20 (internally 21), `high_precision=True`.
   Realised main-run sigma_8 is higher than the target by +7.19e-4 (relative) (0.732754 vs 0.732228);
   consistent with the low-accuracy run truncating the integral at k=1.26 Mpc^-1 (my interpretation).
3. Combined, output `S_8 = 0.747912` versus the sampled `S_8_input = 0.7458` (+0.28%). The sampled parameter
   is therefore a reparametrisation label, not exactly equal to the reported S_8. A_s found: 1.51833e-9.
4. `sigma8toAs` does not pass `neutrino_hierarchy`, `nnu`, `num_massive_neutrinos`, `TCMB`, `YHe`,
   `theta_H0_range`; it uses CAMB defaults. [MEASURED] no numerical effect for mnu=0.06 versus the main
   run's `neutrino_hierarchy='normal'` (identical sigma_8 to 1e-16), see 2.8 and 4.7.
5. It ignores `cs2_de` and uses `p.set_dark_energy(w, wa)` (CAMB default fluid model) (`:44`); it requires `h0`, `ombh2`, `omch2`,
   `omega_k`, `n_s` in the block; `omega_k` has no default (a missing key raises a CosmoSIS `BlockError`, not a Python KeyError) (`:19`).
   [verified] It also ignores `use_ppf_w`, `hubble`/`cosmomc_theta` (needs `h0`) and `t_cmb`/`yhe`/`n_eff`.
6. [verified] Which chains actually contain this step: all K1000 inis built by `run_kcap.py` with `S_8_input` sampling (default
   `sample_ln_As` and `sample_S8_squared` are in `default_config_cuts`, `run_kcap.py:97-98`). Several K1000 runs cut
   `sample_S8` and `sigma8toAs` and either fix `A_s = 2.1e-9` (`runs/3x2pt/3x2pt_configs.py:183-194`, the `--enable-modules sample_ln_As`
   line there is commented out) or sample `ln_1e10_A_s` via `--enable-modules sample_ln_As` (`runs/3x2pt/clustering_configs.py:64-69`;
   also `extended_cosmologies/*maxlike_configs2.py:89-90`, `methodology/create_test_priors_configs.py:130-131`, not individually read).
   The `sample_S8_squared` variant reads `S_8_2_input`, takes `sqrt` and writes `sigma_8_input` (`run_kcap.py:412-415`, `sample_S8.py:20-22`).
   Also: `sample_S8.py` has no write of `S_8`; the physical `S_8` is only produced later by camb.
   Order caveat: in the K1000 module list `correlated_dz_priors` sits between `sigma8toAs` and `one_parameter_hmcode` (harmless).

### 2.8 Neutrinos in the main run
`neutrino_hierarchy` is not set in KV450 inis (-> CAMB default 'degenerate') and is `normal` in
`run_kcap.py:458`; `theta_H0_range` is passed only through `more_config["cosmology_params"]`
(`:83`). `mnu` is the sum in eV. [MEASURED] CAMB 1.5.8, mnu=0.06: hierarchy 'normal' with
`num_massive_neutrinos` 1 or 3 gives ONE massive eigenstate with degeneracy 1.014667 and
2.029333 massless; 'degenerate' with `num_massive_neutrinos=3` gives three states of
degeneracy 3.044 each (a different physical model). So for the fiducial mnu=0.06 + default
num_massive_neutrinos=1 the 'normal' label is inert (my probe; other mnu values not probed).
`massless_nu`, `omega_nu`, `omnuh2` input keys are ignored with a warning (`:274-277`).

### 2.9 consistency module (parameter relations)

Fixed-point solver over this relation list (`consistency.py:55-89`). Names without prefix are in
`cosmological_parameters`; `h0` is little h.

```
omega_m = ommh2/h0^2      omega_b = ombh2/h0^2      omega_c = omch2/h0^2      omega_nu = omnuh2/h0^2
ommh2 = omega_m h0^2      ombh2 = omega_b h0^2      omch2 = omega_c h0^2      omnuh2 = omega_nu h0^2
omch2 = ommh2 - ombh2 - omnuh2     ommh2 = omch2 + ombh2 + omnuh2     omnuh2 = ommh2 - ombh2 - omch2
baryon_fraction = omega_b/omega_m = ombh2/ommh2   (and the two solved-for-numerator/denominator forms)
omega_m = omega_b + omega_c + omega_nu  (and the three rearrangements)
h0 = sqrt(ommh2/omega_m) = sqrt(ombh2/omega_b) = sqrt(omch2/omega_c) = hubble/100 ;  hubble = 100 h0
omega_lambda = 1 - omega_m - omega_k ; omega_m = 1 - omega_lambda - omega_k ; omega_k = 1 - omega_m - omega_lambda
K = -hubble^2 omega_k / c^2   (Mpc^-2, c=299792.458 km/s)
```
Algorithm: all parameters start NaN; known ones inserted; up to `len(relations)+1` sweeps; a
relation that yields a value for a NaN parameter fills it; if the parameter is already known it
must satisfy `numpy.allclose` (rtol 1e-5, atol 1e-8) else `OverSpecifiedModel` (exit 2); if some
stay NaN after all sweeps -> `UnderSpecifiedModel` (exit 1). Escalating defaults: first none, then
`omega_nu=0`, then `omega_nu=0, omega_k=0` (`consistency.py:94-97,127-165`). Relations used on the
first success are cached and replayed via `eval` thereafter (`:167-181`). Keys with a `___` separator
address other sections (`consistency_interface.py:21-28`). `relations_file` option allows custom relations.
[verified] Details: the relation `h0 = sqrt(omnuh2/omega_nu)` is deliberately commented out (ZeroDivisionError, `consistency.py:80-82`);
`eval` of each relation is done with the current parameter dict, NaN propagates (`:231-235`); the first-call loop tries `possible_defaults[:i]`
for i in 0..len-1 and finally all defaults (`:132-165`); the interface writes back **every** relation output to the block, overwriting
existing values with the (allclose-equal) computed ones (`consistency_interface.py:53-58`) and, separately, `A_s` from `log1e10As` (`:49-50`). Parameter checks
use `numpy.allclose` defaults (rtol 1e-5, atol 1e-8), the same tolerances as the `np.isclose` warning in camb's own write-back (`camb_interface.py:369`).
Note: omega_nu=0 default means that, in a pipeline where `consistency` runs BEFORE camb, neutrino
density from `mnu` is ignored by the consistency step (CAMB computes its own).

IMPORTANT: in the fiducial inis `consistency` is either not in the module list
(`KV450_fiducial.ini:11`, `run_kcap.py`) or placed AFTER `camb` (`BOSS_KV450.ini:12`,
`Planck_only.ini:13`, `BOSS_only.ini:12`). CAMB itself writes every quantity the relation graph
needs (2.1 outputs), so `consistency` degenerates to a validator there. A re-implementation need
only reproduce the CAMB side.

### 2.10 Other transforms
```
sample_negative_mnu:   mnu = |mnu_proxy|                                    (utils/sample_negative_mnu.py:8-9; reads cosmological_parameters/mnu_proxy)
sample_folded_prior:   X = fold + |X_folded - fold|, fold default 0         (utils/sample_folded_prior.py:16-18; reads <sec>/<name>_folded, option name="sec/name"; the earlier :51-57 citation was wrong, file has 22 lines)
sample_S_star:         ln_As = S_star * 3.0 / (omch2/0.1)^alpha; A_s = exp(ln_As)*1e-10   (use_ln_A_s=T default)
                       or A_s = S_star * 3e-9/(omch2/0.1)^alpha                          (utils/sample_S_star.py:12-20; the earlier :23-34 citation was wrong, file has 25 lines)
camb_background:       same distances as camb, with get_background(no_thermo=True), omega_k required,
                       default z grid linspace(0,10,1000) (utils/camb_background.py:28-30,59-72); w, wa names configurable.
```
`sample_S_star.py` defines `clean` instead of `cleanup` (:24, not :37): harmless with current CosmoSIS (unverified).
[verified] `S_star = A_s/2e-9 (Omega_m/0.3)^alpha` in `derived_parameters.py` (:36) is NOT the inverse of `sample_S_star`
(`ln_As = 3 S_star/(omch2/0.1)^alpha`, i.e. A_s normalised by omch2 not Omega_m, and pivot 3 in ln(1e10 A_s) vs 2e-9 -> ln(1e10 A_s)=3.0): two different "S_star" definitions exist.

---

## 3. Numerical choices

### 3.1 The CAMB parameter object, step by step (`extract_camb_params`, `camb_interface.py:250-326`)

Order of operations, with fiducial values:

1. `InitPower = InitialPowerLaw.set_params(ns, As, [nrun, nrunrun, nt, ntrun, r], [pivot_scalar, pivot_tensor])` (`:215-221`).
2. `Recomb = Recfast(...)` with all 11 knobs read from section `recfast` falling back to the `Recfast()` defaults (`:135-166`).
3. Reionisation object `TanhReionization` (`:168-190`). Fiducial: `do_reionization = F` -> `reion.Reionization = False` (`:187-189`). With reionisation off, `tau` is not needed or used. (A fresh `TanhReionization()` is created at `:169` and `:188`; attribute `Reionization=False` set on it.)
4. Dark energy: `DarkEnergyFluid` (since `use_ppf_w=F`), `set_params(w, wa, cs2)` (`:192-209`).
5. Non-linear: `Halofit(**{halofit_version?, Min_kh_nonlinear?}, HMCode_A_baryon?, HMCode_eta_baryon?, HMCode_logT_AGN?)` (`:224-233`, 2.5).
6. `CAMBparams(InitPower, Recomb, DarkEnergy, NonLinearModel, **config)` where `config` (`:51-68`) is:
   `WantCls = (mode in {cmb, all})`, `WantTransfer = (mode in {transfer, all})`, `WantScalars=True`,
   `WantTensors=do_tensors(F)`, `WantVectors=do_vectors(F)`, `WantDerivedParameters=True`,
   `Want_cl_2D_array=False`, `Want_CMB = WantCls`, `DoLensing = do_lensing (F)`,
   `NonLinear = 'NonLinear_' + nonlinear` with default **'both'** unless mode is background/thermal (-> 'none'),
   `scalar_initial_condition='initial_adiabatic'`, `want_zstar = want_zdrag = (mode in {thermal, cmb, all})`.
   In `mode=transfer`: `WantCls=False`, `want_zstar=want_zdrag=False`. K1000 sets `nonlinear='pk'` (`run_kcap.py:456`); KV450 relies on default 'both'.
7. `p.set_cosmology(ombh2, omch2, [theta_H0_range, neutrino_hierarchy], [TCMB, YHe, omk, H0, mnu, nnu, standard_neutrino_neff, num_massive_neutrinos, Alens])` (`:297-300`). CAMB default `H0` branch: error if H0<1. `cosmomc_theta`: solved by CAMB root-find within `theta_H0_range` (KV450 `10 150`; BOSS `20 100`).
8. Fix for CAMB < 1.0.10 (`:303-304`) is a no-op on modern CAMB (`omnuh2 != 0`, or nnu equals massless count).
9. `p.Reion = reion` (after set_cosmology on purpose, `:307-309`).
10. `p.set_for_lmax(lmax=2600 default [KV450 ini: 2500], [max_eta_k, lens_potential_accuracy, lens_margin, k_eta_fac, lens_k_eta_reference])` (`:79,311`). Sets `max_l` / `max_eta_k = 2.5*max_l` [CAMB-1.5.8 model.py:766-770]; irrelevant in transfer mode.
11. `p.set_accuracy(**accuracy_params)` (`:312`): with no options this **resets** `AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1, DoLateRadTruncation=True` [CAMB-1.5.8 model.py:268-290]. Options accepted by kcap: `AccuracyBoost, lSampleBoost, lAccuracyBoost, DoLateRadTruncation` only (`:96-97`); none set in any fiducial ini (`accuracy_boost = 2.0` appears only commented out and with a name that kcap would ignore).
12. Redshift grid and `p.set_matter_power(redshifts=z, nonlinear=NonLinear in {both,pk}, **transfer_params)` (`:314-324`).

### 3.2 Redshift grid for P(k,z)  (`:105-108, 314-323`)

Defaults zmin=0, zmax=3.01, nz=150. Fiducial (KV450 `ini:116-120`, K1000 `run_kcap.py:459-463`):

```
zmin=0, zmid=2, nz_mid=100, zmax=6, nz=150
z = concat( linspace(0, 2, 100, endpoint=False),     # dz = 0.02, 0 <= z <= 1.98
            linspace(2, 6, 50) )[::-1]                # dz = 4/49 = 0.081633, 2 <= z <= 6
total 150 points, passed in decreasing order. CAMB sorts descending and returns ascending in get_*_power_spectrum.
```
(The `Min nz` for the second segment is `nz - nz_mid`.) CAMB hard limit `max_transfer_redshifts` ([CAMB-1.5.8 model.py:721]); 150 passes. [verified] The limit value is 256 in CAMB 1.5.8 (`model.py:20`); it was not checked for older versions.

### 3.3 Wavenumber grid

kcap controls: `kmax` (default 1.2 in `camb_interface.py:129`; fiducial **20.0**; units **1/Mpc, not h/Mpc**), `k_per_logint` (optional, unset in fiducial -> 0 = CAMB's automatic spacing), `accurate_massive_neutrinos` (optional; unset -> False).
`set_matter_power` unconditionally sets `Transfer.high_precision = True` [CAMB-1.5.8 model.py:701].

[MEASURED] CAMB 1.5.8, kmax=20, h=0.7, WantCls=False:
* nk = **194** for h = 0.64, 0.70, 0.82 (grid is fixed in absolute k, so nk does not vary with h); nk = 191 for kmax=10.
* k_min = 5.0e-5 Mpc^-1 (k_h = 7.14e-5 at h=0.7).
* k_max = **1.05 * kmax = 21 Mpc^-1** (k_h = 30.0 at h=0.7; 32.81 at h=0.64; 25.61 at h=0.82).
* Spacing in ln k: 0.333 at the lowest step, ~0.0333 (30 points per e-fold) over most of the range, ~0.039 near k~1, and 0.217 at the last step.
* Linear and non-linear spectra are on the identical k grid.
* A different CAMB version may give a different nk (not tested).

CAMB warns if `kmax<5` or (`kmax<20` and `max z>4`) with Halofit models; fiducial avoids this by kmax=20, zmax=6.

### 3.4 Extrapolation to high k (`extrapolate_power.py`)

Parameters: `kmax=500` (h/Mpc, `KV450_fiducial.ini:183`, `run_kcap.py:515`), `kmin=1e10` default (never triggers),
`nmin=50` (unused by default), `nmax=200`, `npoint=3` (`:56-67`).
Applied to `matter_power_nl`, `matter_power_lin`, plus `sections` option (`:79-81`); skipped silently if section absent.

For every redshift independently (loop over `z` columns):
```
x = ln k_h ; y = ln P(k_h, z)
fit  y ~ a + b x  by np.polyfit(deg 1) on the LAST npoint=3 samples      (:22-24; the earlier :21-23 was off by one)
x_new = linspace(x_max, ln 500, nmax=200)[1:]   -> 199 new points, uniform in ln k   (:25-27)
P_new = exp(a + b x_new)
```
New nk = 194 + 199 = 393. Step in ln k of the added part = ln(500/k_h,max)/199 (0.0141 at h=0.7), a 15x finer step than the
last native step (0.217), so the grid is non-uniform at the junction. No extrapolation in z. Fitting is unweighted on three
points spaced by the native (coarse at the end) steps.
[MEASURED] logarithmic slope b of the last three points for a test cosmology: linear P: -2.568 (all z);
non-linear P: -2.16 (z=0), -2.17 (z=1), -2.12 (z=2), -1.68 (z=6). P_nl/P_lin at the last native k is 61.6 (z=0), 31.5 (z=2), 15.5 (z=6).
Output is written with `replace_grid(section, "z", z, "k_h", k, "P_k", P_out.T)`.
Low-k branch (`kmin < k_min`) fits the first `npoint` samples and prepends `nmin` points via `linspace(..., endpoint=False)`.
[verified] The branch test is `xmin < x.min()` with `kmin` default 1e10 (so `log kmin`=23.0 > any ln k_h) -> never taken; likewise the high-k
branch is skipped silently if `kmax <= k_h,max` (`:21`), and the nz>1/nk>1 assumption of `np.dstack(...).squeeze()` is at `:51`.
The fit uses P(k) of **every** z column independently, so the extrapolated slope differs per z (measured above); the same
treatment is applied to `matter_power_lin`, so the linear and non-linear spectra agree on the same 393-point grid.
Output key is written as `P_k` (upper case) but CosmoSIS lower-cases it, so downstream still sees `p_k`.

### 3.5 Background grid (`:105-125, 378`)
`zmin_background=0, zmax_background=6, nz_background=6000` in fiducial inis; z uniform: dz = 6/5999 = 1.0002e-3 (defaults would otherwise be zmin/zmax/nz of the P(k) grid, i.e. 150 points). The deprecated names `background_*` and `*_background` are both accepted (`:116-121`). `KV450_no_sys.ini` uses `background_nz=150`.

### 3.6 Hard-coded constants and conventions (by file)
* `camb_interface.py`: `lmax` default 2600 (`:79`), `zmax` default 3.01 (`:106`), `kmax` default 1.2 (`:129`), sigma_12 radius 12.0 Mpc (`:418`), S_8 pivot 0.3 (`:421`), CMB_unit "muK" (`:425`).
* `sigma8toAs.py`: `fid_As=2.1e-9` (`:28`), `kmax=1.2` and `k_per_logint=5` defaults (`:8-9`), `DoLateRadTruncation=True`, `high_precision=False`, `accurate_massive_neutrino_transfers=False`, `PK_redshifts=[0.0]`, reionisation off, `mnu` default 0.06, no non-linear (`:30-48`).
* `sample_S8.py`: `alpha=0.5` default, pivot 0.3 (`:10,33`). `derived_parameters.py`: same, plus 2e-9 for S_star (`:36`).
* `one_parameter_hmcode.py`: `a_0=0.98, a_1=-0.12` defaults (`:5-6`), and the same in every ini/run_kcap.
* `extrapolate_power.py`: 500, 1e10, 50, 3, 200 as above.
* `consistency.py`: c = 299792.458 km/s in the K relation (`:88`).

### 3.7 Defaults vs fiducial: the set of CAMB settings actually used for KiDS
(effective, K1000 = `run_kcap.py:452-466`)

| Setting | Value |
|---|---|
| mode | transfer (WantCls=False, WantTransfer=True, Want_CMB=False, DoLensing=False) |
| NonLinear | `pk` (K1000), `both` (KV450 default; lens part unused) |
| Non-linear model | Halofit class, `halofit_version` = **'mead'** in K1000 (`run_kcap.py:457, 970`); KV450 fiducial/BOSS_KV450/Flinc inis use a misspelled key (see QUIRK 4.1) |
| Baryon feedback | A_baryon = sampled A in [2.0,3.13], eta_baryon = 0.98 - 0.12 A |
| Reionisation | off |
| Dark energy | DarkEnergyFluid, w=-1, wa=0, cs2=1 (exact LCDM) |
| Neutrinos | mnu = 0.06 eV, `num_massive_neutrinos` = 1 (CAMB default in K1000; explicit 1 in KV450 / KV450_no_sys values; [verified] commented out in BOSS_KV450_values.ini:39 and BOSS_only_values.ini:38, hence CAMB default there), `nnu`/`standard_neutrino_neff` = CAMB default (3.044 in CAMB 1.5.8; I recall 3.046 in CAMB <=1.3, not verified here), `neutrino_hierarchy` 'normal' (K1000) or default 'degenerate' (KV450) |
| Accuracy | `set_accuracy()` defaults (all boosts 1, DoLateRadTruncation True); `high_precision=True`; `accurate_massive_neutrinos=False`; AccuratePolarization etc. at library defaults |
| k grid | kmax=20 /Mpc -> 194 pts, 5e-5..21 /Mpc, automatic spacing |
| z grid | 150 pts (100 below z=2, 50 in 2..6) |
| Distances | 6000 pts z in [0,6] |
| Extrapolation | log-log linear from last 3 points to k_h=500, +199 pts |
| TCMB, YHe | CAMB defaults: 2.7255 K, BBN-table YHe |
| Pivot | k_p = 0.05 /Mpc |

---

## 4. Quirks, approximations, deviations (QUIRK)

QUIRK 4.1 **Typo `halofit_verion = mead`** in `runs/config/KV450_fiducial.ini:124`, `BOSS_KV450.ini:136`,
`Flinc/xipm_Flinc.ini:160` ([verified] also `Flinc/xipm_Flinc_complextheory_no_noise.ini:161` and
`Flinc/xipm_Flinc_complextheory_no_noise_s_star.ini:162`; i.e. 5 inis carry the typo). kcap reads only `halofit_version` (`camb_interface.py:94`), so the line is
silently ignored and the library default model is used. On CAMB 1.5.8 `Halofit()` defaults to **mead2020**
(`halofit_default`, [CAMB-1.5.8 nonlinear.py:24]; confirmed by instantiation), which **ignores** `A_baryon`
and `eta_baryon` (docstring `nonlinear.py:75`). So with a modern CAMB the KV450 fiducial ini would run
HMcode-2020 without feedback and the sampled `halo_model_parameters/A` would be a flat, unconstrained nuisance
parameter. With the CAMB the authors used (default presumably 'mead') the typo is harmless. The CAMB default
changed across versions; which release introduced `mead2020` as default I could not verify from the files here.
`KV450_no_sys.ini:136` spells it correctly (`takahashi`). The K1000 generator passes `'mead'` explicitly
(`run_kcap.py:457`, default `--halofit-version mead` at `:970`) and the tests do too (`tests/test_hmcode.py:46,125`; the earlier `:123` was off by two).
[verified] In the same ini files the sampled `halo_model_parameters/A` still feeds `one_parameter_hmcode` (eta0) and `camb`
(`HMCode_A_baryon`/`HMCode_eta_baryon` are set on the `Halofit` object), so the chain runs without error even though the
physics is silently different on modern CAMB. CAMB 1.5.8 was confirmed by experiment: with `halofit_version` left at default
(`mead2020`), P_nl for (A=2.0, eta=0.74) and (A=3.13, eta=0.603) is bit-identical (max |ratio-1| = 0), whereas with `mead` the change (A=2.0, eta=0.74) vs (A=3.0, eta=0.62) moves P_nl by up to 31% (at the highest k, Appendix A point). The re-implementation should treat the
model as HMcode-2016 (mead) with A_baryon/eta0, as intended.

QUIRK 4.2 Direct `Halofit(**kw)` construction means unspecified fields keep library defaults, in particular
`A=3.13, eta=0.603` (DM-only) when `halo_model_parameters` is absent. Also, `eta0` is written into
`halo_model_parameters/eta0` by `one_parameter_hmcode` and `Min_kh_nonlinear` (default 0.005) is never overridden.

QUIRK 4.3 `halofit_version=mead` is an alias of `mead2016` (id 5), not `mead2015` (id 8). Documented HMcode-2016 includes neutrino
corrections (f_nu terms in Delta_v, delta_c) [LITERATURE-COPY, 2.5].

QUIRK 4.4 `sigma_8`, `sigma_12`, `S_8` are taken at the first redshift of the grid, not necessarily z=0 (2.6).
`sigma_8` = total-matter sigma_8; `S_8` uses Omega_m including neutrinos; `S_8_input` (sampler) uses Omega_cb (2.7).

QUIRK 4.5 `S_8_input -> A_s` pipeline accuracy mismatch (+0.07% in sigma_8, +0.28% in S_8; 2.7). It is a deterministic reparametrisation, so
posteriors are well-defined, but "S_8" of the chain's `S_8_input` column is not the physical S_8.

QUIRK 4.6 `D_M` assigned the comoving radial distance (wrong for omk != 0). `MU[0]=NaN`, `rs_DV[0]=inf`, `F_AP[0]=0` at z=0 ([MEASURED]).
`omega_lambda` excludes radiation.

QUIRK 4.7 `neutrino_hierarchy='normal'` is inert at mnu=0.06 with `num_massive_neutrinos` 1 (and 3), per measurement on CAMB 1.5.8 (2.8).

QUIRK 4.8 Bugs that only trigger off the fiducial path:
* `camb_interface.py:183` uses `include_helium_fullreion`, which is undefined (NameError) when `do_reionization=T` and `use_optical_depth=F`.
* `:200` uses `name.de_equation_of_state` (undefined; should be `names.`) for `use_tabulated_w=T`.
* `:127` collects `accurate_massive_neutrinos`, but `set_matter_power` expects `accurate_massive_neutrino_transfers` [CAMB-1.5.8 model.py:685] -> TypeError if the option is ever set.
* The CAMB-version guard at `:43-45` (`not M>1 and not m>0 and not v>9`) only warns for 1.0.0-1.0.9 or 0.x with m=0, not for general old versions.
* Only `camb.baseconfig.CAMBParamRangeError` is caught (`:334-336`) and mapped to return code 1; other CAMB errors propagate as exceptions.
* `extrapolate_power`: `np.dstack(...).squeeze()` breaks for nz==1 or a single k (`:51-53`).
* `consistency_interface.py:49`: key `log1e10As` vs the kcap `ln_1e10_A_s` naming.
* `sample_S_star.py:37`: function named `clean` not `cleanup`.
* `ConsistencyModule` docs in `module.yaml` list relations that differ from the code (e.g. docs have `omega_lambda = 1-omega_m-omega_k-omega_nu`; the code has no `-omega_nu`).

QUIRK 4.9 `WantCls=False` in transfer mode reduces CAMB's internal k/accuracy settings: kcap's own test notes
2e-3 differences in P_lin if the flag is not matched between a reference CAMB call and the pipeline (`tests/test_hmcode.py:151-152`; the earlier `:141-142` was wrong).
A re-implementation comparing against kcap numbers must use the same flags.

QUIRK 4.10 Extrapolation (3.4) is a pure power law from the last three native points; the physical P_nl slope between k_h=30 and 500 is not a
constant power law (HMcode 1-halo asymptote differs). The `module.yaml` itself states "this is not a great approximation"
(`extrapolate/module.yaml`, assumptions). The extension matters for high-ell/low-z contributions only (K1000 uses ell<=~1500; KV450 ell_max=5e5 in `projection`).

QUIRK 4.11 `D_A`, `D_C`, `D_L` are in Mpc while `k_h` and `P` use h-units; downstream code must multiply by h0 where needed (e.g. `project_2d.py:541` for chi, `:544` for CHISTAR; `:538` merely reads h0 - earlier citation corrected).

QUIRK 4.12 `[consistency]` is defined but not run in the KV450/K1000 fiducial chains (2.9).

---

## 5. Strengths

* Thin and transparent: ~480 lines wrapping CAMB's official Python API; all numerics delegated to a well-tested Boltzmann code.
* Grid design is sensible: finer z sampling at z<2 where the lensing kernel lives (dz=0.02), a fixed absolute k grid that does not vary with h.
* HMcode-2016 with a 1-parameter baryon-feedback relation (`eta0 = 0.98 - 0.12 A`) gives a cheap, well-defined nuisance model.
* The `sigma8toAs` trick exploits exact amplitude scaling (sigma_8^2 ∝ A_s), so S_8 is a sampled parameter at the cost of one cheap CAMB run (0.07-0.09 s [MEASURED]).
* Redundant bookkeeping outputs (`omega_m`, `h0`, `cosmomc_theta`, derived thermal parameters) make downstream modules independent of how the user specified the cosmology.
* Extensive option surface (recfast knobs, tabulated w, PPF, theta parametrisation).
* The repository contains tests comparing CAMB-HMcode with the stand-alone HMcode (`tests/test_hmcode.py`), a CCL comparison (`tests/test_pipeline_against_CCL*.py`), and a P(k) comparison folder.

---

## 6. Weaknesses and risks

Correctness
* Silent mis-configuration (QUIRK 4.1); no validation of option names: any misspelled ini key is ignored, with no warning from the module.
* Unpinned CAMB: results depend on CAMB version through (a) default Halofit model, (b) default `nnu` (3.046 vs 3.044), (c) BBN table for YHe, (d) k-grid/accuracy internals, (e) HMcode implementation. No lock file, `conda_env.yaml` lists no `camb`.
* `S_8_input` vs `S_8` mismatch and low-accuracy sigma_8 inversion (2.7).
* `D_M` wrong for omk != 0; `omega_lambda` excludes radiation; several never-exercised code paths have NameErrors (4.8).
* `sigma_8`-family quantities assume zmin = 0.
* The nonlinear spectrum for z in (4, 6] is produced with kmax=20 (CAMB warns for kmax<20 and z>4 only when kmax<20; kcap sits exactly at the limit). HMcode accuracy at z>2 is only ~1.5% against the standalone code per kcap's own test.
* Extrapolating P_nl by a 3-point power-law (4.10).

Accuracy
* HMcode-2016 is a dark-matter + simple baryon model; its accuracy limits (a few % at k>1 h/Mpc) are inherited. No emulator / HMcode-2020 in the main path (available only via extended configs).

Maintainability and portability
* Dependence on CosmoSIS's compiled datablock, inifile string conventions, environment variables (`CSL_PATH`, `KCAP_PATH`) and, for other modules, compiled Fortran/C. The subsystem in this study is pure Python, but needs `cosmosis` just to run.
* Mixed units across sections (Mpc vs Mpc/h) are undocumented in the code.
* Heavy use of undocumented side-effects (module order matters; `one_parameter_hmcode` must precede `camb`; consistency position is arbitrary).

Performance
* [MEASURED] CAMB 1.5.8, K1000-like settings on a 22-thread WSL2 box. Per-evaluation cost: `get_results` (transfer functions) 1.84 s with 1 OpenMP thread, 0.48-0.52 s with 4 threads, 0.23-0.35 s with the default 22 threads. HMcode over 150 redshifts (first call of `get_nonlinear_matter_power_spectrum`) 1.0 s (1 thread), 0.5 s (4 threads) but **4.1-4.5 s with the default 22 threads** (oversubscription; the default is 4x slower than 1 thread). Takahashi Halofit on the same grid: 0.2 s. HMcode time scales linearly with nz: 0.07 s (10 z), 0.32 s (50), 0.98 s (150) at 1 thread. `sigma8toAs`: 0.07-0.09 s. `get_BAO` on 6000 z: 10-50 ms.
* So in default threading the non-linear step dominates the Boltzmann step by ~10x.
* `extrapolate_power` doubles nk (194 -> 393) and every downstream P(k) interpolation/Limber integral.
* 6000-point background grid and unused outputs (`D_L, MU, rs_DV, F_AP`, derived thermal parameters) are computed on every call.

---

## 7. Efficiency opportunities (with estimates)

All estimates are from the [MEASURED] numbers above unless stated; re-measure on the target machine.

1. **Fix OpenMP threading** (zero code change; set `OMP_NUM_THREADS` to ~4-8, or use `threadpoolctl`/`omp_set_num_threads` around the HMcode call). HMcode step 4.3 s -> 0.5 s (about 8x); transfer step ~0.23-0.5 s either way. Whole per-sample cost of CAMB part: ~4.5 s -> ~1.0 s (about 4.5x). With many MPI/multinest workers use 1 thread per worker (HMcode 1.0 s, transfer 1.8 s).
2. **Cut the redshift grid of the non-linear power.** HMcode cost is linear in nz. The lensing kernels for KiDS bins (z <~ 1.2, tails to ~3) do not need 50 points between z=2 and 6, nor dz=0.02 resolution everywhere. Computing HMcode on ~30-40 z nodes (e.g. denser at low z, a up to ~3) and interpolating P(k,z) smoothly in ln a (or in z with splines on ln P) reduces HMcode cost 4x-5x with errors far below 0.1% (to be validated against the 150-point result). Cutting zmax to 3 and nz to ~80 alone gives about 1.9x.
3. **Re-implement HMcode (numpy/numba)** with: vectorised sigma(R,z) via a precomputed log-grid (k, R) matrix product with the top-hat window (the standalone code already tabulates sigma(R) on 64 points and uses a 256-point mass grid), the NFW window via vectorised Si/Ci (`scipy.special.sici`), and growth from a single ODE solve (note [verified]: the stand-alone Fortran copy does NOT solve an ODE, it uses the growth-index integral with gamma=0.55 [+0.02/0.05 (1+w)], `mhmcamb.f90:678-790`; CAMB's own internal growth source is unknown, section 8 item 2 - an ODE-based growth may differ from both; size not measured). CAMB's Fortran HMcode already costs only ~7 ms per z-slice on one thread [MEASURED], so a numpy/numba port will not beat it per slice; the benefit is control (batching over z, choosing nz, threading policy) and caching: for a change in `A_baryon` only (nuisance sweep), the linear transfer function, sigma(R), sigma_v, neff etc. are unchanged and only the concentration prefactor and eta change, so a baryon-only update can cost ~1 ms/slice.
4. **Avoid the second CAMB run in `sigma8toAs`** (0.07-0.09 s, ~5-20% of the Boltzmann cost): since the transfer function is independent of A_s, solve once with a fiducial amplitude, compute sigma_8 from that linear P, rescale A_s analytically, and apply to P_lin before HMcode. This also removes the +0.07% accuracy mismatch (2.7). (With CAMB as is, one needs a second `get_results`; with an own pipeline it's free.)
5. **Cache by parameter subsets** (CosmoSIS `fast_slow` is present but off, `BOSS_KV450.ini:22`). Parameters that don't change the linear spectrum or background (A_baryon, IA, photo-z shifts, shear c-terms, m-bias) don't require a Boltzmann call; with Boltzmann cached the per-sample cost is the HMcode (or its cached variant, item 3) plus projection.
6. **Drop unused outputs** in transfer mode: `D_L`, `MU`, `rs_DV`, `F_AP` (the `get_BAO` call re-runs `calc_background` incl. recombination), derived thermal parameters (not needed with reionisation off); reduce `background_nz` from 6000 to ~300-500 (cubic spline of chi(z)); saves ~10-60 ms/sample and shrinks the datablock.
7. **Skip or limit extrapolation**: extend only to the k actually probed. For ell_max <= 1500 and z_min ~ 0.01-0.05 the needed k is <~ 10-30 h/Mpc, which the CAMB grid (k_h up to 30) already covers; 199 extra points double the cost of the Limber integral's k interpolation (est. 1.3-2x on the P(k) interpolation step; actual share of projection time not measured).
8. **Lower kmax / k sampling where safe**: the nk=194 grid is already ~30 pts/e-fold; going to kmax=10 saves little (191 pts; HMcode 2.8 s vs 4.6 s in the default-thread test, but kcap's own tolerance concerns apply). Not recommended without accuracy tests (CAMB warns <20 for z>4).
9. **Replace CAMB's Boltzmann solve by an emulator / Taylor expansion** (e.g. a pre-trained surrogate for T_tot(k,z) in (omch2, ombh2, h, n_s, mnu)) for MCMC: 0.2-1.8 s -> sub-ms. Needs validation against CAMB at sub-0.1% (the data constraining power is ~1-2% on S_8).
10. **Precision of S_8 sampling**: using sigma8 from the same high-accuracy transfer (item 4) removes the mismatch for free.

---

## 8. Open questions (could not resolve from the available sources)

1. CAMB version used for the published KV450/KiDS-1000 chains. Not recorded in kcap (`conda_env.yaml`, README); determines the default Halofit model (QUIRK 4.1), `nnu` (3.046 vs 3.044), YHe table, and the HMcode source. The kcap commit is dated 2022-06-13; I believe CAMB 1.3.x was current, but this is not verified.
2. CAMB's internal HMcode (Fortran `halofit.f90`) was not readable (only the compiled library), so details such as: P_cb vs P_tot used as the halo-model input; how the HMcode result is converted into the returned `matter_power_nl` (ratio vs absolute); the growth-factor source (ODE vs transfer-function ratio); integration tolerances; and `Min_kh_nonlinear` handling are only known from the docstring and from the standalone copy.
3. Meaning of the "kmax*1.05" internal factor: observed (k_max=21 for kmax=20) but the source was not read; also how k-sampling depends on `max_eta_k`, `AccuracyBoost`, `WantCls`.
4. Whether `get_BAO` (which calls `calc_background` on `r` itself, `results.py:267-270`) can alter later `get_*_matter_power_spectrum` results; [MEASURED] on 1.5.8 the linear P is unchanged (np.allclose) after `get_BAO`, but the ordering in `camb_interface.py:392` vs `:398` is unguarded.
5. Whether `rs_DV` requires `want_zdrag=True` (in `mode=transfer` it is False). [MEASURED] `get_BAO` returned finite `rs_DV` for z>0 on 1.5.8, so probably computed independently; not verified against older CAMB.
6. `sample_S8.py:23` uses `(cosmo, "omega_m") in block`: tuple membership on a `DataBlock`; I did not verify its semantics in the CosmoSIS version used by kcap, hence whether `omega_m` can ever be picked up if present.
   [verified, resolved] `DataBlock.__contains__` accepts a `(section, name)` tuple (local CosmoSIS 3.12.1, `block.py:873-...`), so the branch works as written;
   in the K1000 chain no module before `sample_S8` writes `omega_m`, so the fallback `(omch2+ombh2)/h0^2` (Omega_cb, no neutrinos) is always the one used.
7. Which `mnu` values produce a two-eigenstate 'normal' hierarchy in CAMB (only 0.06 eV probed); matters for the extended-cosmology runs with `mnu in [0,3]` (`runs/extended_cosmologies/3x2pt_ext_cosmo_configs.py:91`).
   [verified, partly resolved] CAMB 1.5.8 `set_cosmology`: 'normal' with mnu=0.06 -> 1 eigenstate (degeneracy 1.014667, 2.029333 massless) for
   `num_massive_neutrinos` 1 or 3; 'normal' with mnu=0.3 -> **2** eigenstates (degeneracies 2.029333 and 1.014667, mass fractions 0.639 / 0.361, 0 massless),
   again independent of `num_massive_neutrinos` (1 or 3); 'degenerate' with `num_massive_neutrinos=3` -> 1 eigenstate of degeneracy 3.044 (any mnu tested 0.06);
   'inverted', 3, mnu=0.1 -> 1 massive eigenstate (deg 2.029) + 1.015 massless. So for K1000 (`normal`), the
   `num_massive_neutrinos` setting is irrelevant and the number of eigenstates is set by mnu; the exact mnu threshold between 1 and 2 eigenstates was not located.
8. The KiDS-1000 pipeline uses `nonlinear = pk` while the earlier inis use 'both': difference has no effect in transfer mode, but I did not confirm that CAMB does not use the `NonLinear_lens` setting to modify the matter-power k-range.
9. `HMcode` copy `structure/meadcb` is a separate module with `Tcb_Tcbnu` Eisenstein-Hu cold-matter ratio; it is used only for tests and the `EE_fR` ReACT path in kcap (not in the fiducial chain); its relation to CAMB's own handling of neutrinos is unclear.
10. Origins of the constants 0.98 and -0.12 (not cited in the kcap source; the Fortran comment calls it an "updated one-parameter relation").

---

## Appendix A. Validation anchors from my CAMB 1.5.8 runs (reproduction checks)

These are CAMB 1.5.8 numbers, useful as order-of-magnitude regression targets for a re-implementation. They are not kcap outputs.

A. KV450-like point: ombh2=0.022, omch2=0.10, H0=70, n_s=1.09, ln(1e10 A_s)=3.09, mnu=0.06, num_massive=1, w=-1, reionisation off, HMcode `mead` with A=3.0, eta0=0.62, kmax=20, z grid as in 3.2:
`omega_m=0.250296`, `omnuh2=6.44867e-4`, `N_eff=3.044`, `YHe=0.245695`, `age=14.1504 Gyr`, `sigma_8(z=0)=0.779557` (get_sigma8), 0.779749 (get_sigmaR at 8 h^-1 Mpc), `sigma_12=0.753536`, `nk=194`, `k_h in [7.1429e-5, 30.0]`.

B. K1000-like point: ombh2=0.0225, omch2=0.13, h=0.70, n_s=0.97, S_8_input=0.7458, mnu=0.06:
`Omega_cb=0.311224`, `sigma_8,input=0.732228`, low-accuracy `sigma_8(A_s=2.1e-9)=0.861138`, `A_s=1.51833e-9`, main-run `sigma_8=0.732754`, `omega_m=0.312541`, `S_8=0.747912`.

C. Same cosmology, A_s=2.1e-9, main-run settings: `sigma_8=0.861756` (get_sigma8), cold-matter 0.865248 at 8 h^-1 Mpc.

D. Timings (22-thread box): see section 6.

## Appendix B. Minimal spec for a faithful re-implementation of this subsystem

1. Parameters: sample (omch2, ombh2, h0, n_s, S_8_input or ln_1e10_A_s, A_baryon); fixed (omk=0, w=-1, wa=0, mnu=0.06, num_massive_neutrinos=1, N_eff=3.044 or 3.046 depending on target CAMB).
2. A_s: either exp(ln10A)*1e-10 or from S_8_input via sigma_8,input = S_8/(Omega_cb/0.3)^0.5 (Omega_cb = (omch2+ombh2)/h^2) and A_s = 2.1e-9 (sigma_8,input/sigma_8,fid)^2 (for exact CAMB reproduction use the low-accuracy sigma_8,fid of 2.7, otherwise the high-accuracy one).
3. Linear P_tot(k,z) on 150-point z-grid (3.2) and k grid, in h-units; sigma_8 on total matter, R=8 h^-1 Mpc.
4. HMcode 2016 (2.5) with A_baryon=A, eta0=0.98-0.12 A, applied to the CAMB-type P_lin; then log-log 3-point power-law extension of both P_lin and P_nl to k_h=500 with 199 extra points (3.4).
5. Outputs: distances on 6000 pts with D_A,D_C,D_L in Mpc and H in 1/Mpc; omega_m including neutrinos; sigma_8, sigma_12 (R=12 Mpc), S_8 = sigma_8 sqrt(omega_m/0.3).

## Appendix C. [verified] Behaviour and corrections added by the adversarial re-read

Behaviour the first pass missed (all read in source, or run on CAMB 1.5.8 where stated):

1. **Reionisation defaults**: `do_reionization` and `use_optical_depth` both default to True (`camb_interface.py:85-86`). The KiDS inis/`run_kcap.py:454` set
   `do_reionization = F`; any other ini inheriting the code default requires `cosmological_parameters/tau` (CosmoSIS BlockError otherwise).
2. **Error handling is narrow**: `CAMBParamRangeError` (a subclass of `CAMBError`, `baseconfig`) is the only exception turned into return code 1 (`:334-336`), and only
   around `extract_camb_params`; `get_results`/`get_background` (`:338-342`) are outside the `try`. Example verified: `DarkEnergyFluid` (kcap default, `use_ppf_w=F`)
   with w=-0.9, wa=-0.5 (w crosses -1) raises `CAMBError: fluid dark energy model does not support w crossing -1`, which propagates and aborts the sample;
   `DarkEnergyPPF` accepts the same parameters. Relevant for any extended-cosmology w0-wa run.
3. **`nonlinear` option subtleties**: option values are `none|pk|lens|both` (`:61-63`). kcap passes `nonlinear=(NonLinear in {both, pk})` to `set_matter_power`, and writes
   `matter_power_nl` iff `p.NonLinear is not camb.model.NonLinear_none` (`:402`; identity comparison on strings, works on 1.5.8 because the getter returns the module constant).
   With `nonlinear=lens` the state becomes `NonLinear_lens` (not `none`), so kcap still requests the non-linear spectrum; on 1.5.8 this call returns a P_nl that differs
   from the linear one (tested), i.e. no error but semantically unintended. With `nonlinear=none` (used by `EE_fR`, `run_kcap.py:172`, so ReACT/HMcode modules provide P_nl) no
   `matter_power_nl` section is written and `extrapolate_power` quietly handles only `matter_power_lin`.
4. **Dynamic attributes**: `p.omegal` and `p.ommh2` are ordinary Python attributes assigned onto the `CAMBparams` object (`:350-351`); CAMB 1.5.8 has no such properties (tested), so
   `omega_lambda` and `ommh2` are kcap's own definitions (`omega_lambda = 1 - omegam - omk`, radiation lumped in; `ommh2 = omegam h^2` = ombh2+omch2+omnuh2).
5. **Write-back policy**: derived and consistency scalars are written only if absent, otherwise compared with `np.isclose` and only a `warnings.warn` is emitted (`:366-373`);
   so a stale or inconsistent upstream value (e.g. an `omega_m` written by a previous module) silently wins downstream while CAMB used its own.
6. **Thermal outputs in transfer mode**: `get_derived_params()` is always called with `WantDerivedParameters=True` even though `want_zstar=want_zdrag=False` in `mode=transfer`
   (`:57,67-68,345-347`); `rs_DV`/`F_AP` come from a *second* background calculation on a copy of `p` (`results.py:259-270`).
7. **Parameter spelling**: `get_optional_params` silently ignores absent keys, and `h0` vs `hubble` precedence is: `hubble` (km/s/Mpc, mapped to CAMB `H0`) > `h0*100` > `cosmomc_theta/100` (`:263-284`).
   Mass/neutrino keys are likewise only forwarded if present, so a misspelled `mnu` falls back to CAMB's default (mnu=0.06, `set_cosmology` default).
8. **Option types**: `theta_H0_range` is forwarded exactly as read from the ini (`"10 150"`); CAMB only uses it when `cosmomc_theta`/`thetastar` is given (`model.py:481-482`), so it is inert
   in all fiducial KiDS runs (h0 sampled); CAMB 1.5.8's own default is (10, 100) (`model.py:435`) whereas KV450 sets 10-150 and BOSS/Planck inis 20-100. Whether CosmoSIS delivers it as a string or a list was not tested.
9. **Redshift grid vs `growth_parameters/z`**: `growth_parameters/z` is the `z` variable left over from the last `get_*_matter_power_spectrum` call (`:403` if non-linear is on, else `:398`);
   all are identical 150-point ascending grids.
10. **sigma_12 return type**: with scalar `R` and scalar `z_indices=-1`, `get_sigmaR` returns a Python/numpy scalar (`results.py:838-841`), so `cosmological_parameters/sigma_12` is a scalar,
    and `z_indices=-1` is the last entry of CAMB's descending-z list, i.e. the lowest redshift (docstring `results.py:812-814`: "-1 gives redshift 0").

Citation corrections made in the body above (old -> new): `run_kcap.py:852-863` -> `:851-861`; `test_hmcode.py:141-142` -> `:151-152`; `test_hmcode.py:123` -> `:125`;
`sample_folded_prior.py:51-57` -> `:16-18`; `sample_S_star.py:23-34` -> `:12-20` and `clean` `:37` -> `:24`; `extrapolate_power.py:21-23,24-27` -> `:22-24,25-27`;
`project_2d.py:538,544` -> `:541,544`. All other file:line citations in sections 1-5 were re-read and agree (within 1-2 lines for ranges).

Claims checked and CONFIRMED (no change needed): factor `2 pi^2/k^3` and `k^3/(2 pi^2)` in P<->Delta^2 (also in `mhmcamb.f90:374`); sigma_8 window `W=3(sin x - x cos x)/x^3` (`:1068`);
`Delta_v=418 Om_m(z)^-0.352 (1+0.916 f_nu)`, `delta_c=[1.59+0.0314 ln sigma_8(z)](1+0.262 f_nu)[1+0.0123 log10 Om_m(z)]`, `eta=eta0-0.3 sigma_8(z)`, `k*=0.584/sigma_v`,
`f=0.0095 sigma_v100^1.37` clipped [1e-3,0.99], `alpha=3.24 * 1.85^neff` clipped [0.5,2], `P_2h=P_lin[1-f tanh^2(k sigma_v/sqrt f)]`, `P_1h` prefactor `2 Delta_v k^3/(3 pi)` and
`(1-exp(-(k/k*)^2))`, Sheth-Tormen `0.21616[1+(0.707 nu^2)^-0.3]exp(-0.707 nu^2/2)`, NFW window and `ln(1+c)-c/(1+c)`, concentration `A (1+z_f)/(1+z) (g_w/g_LCDM)^1.5` at z=10,
256-point mass grid [1,1e18], 64-point sigma table R in [1e-4,1e3]; eta0 = 0.98-0.12 A numbers (A=3.13 -> 0.6044, A=2 -> 0.74); S_8, sigma_12 definitions; distances in Mpc (not Mpc/h),
`H` in Mpc^-1 (2.33495e-4 = 70/299792.458); 6000-point background grid, 150-point z grid (dz 0.02 below z=2); nk=194, k in [5e-5, 21] 1/Mpc; extrapolation 194+199=393 points;
Appendix A values for point A and the chain in 2.7 (omega_m 0.312541, A_s 1.51833e-9, S_8 0.747912); HMcode-vs-threads timings (1 thread 1.0 s, 4 threads 0.65 s, 22 threads 3.9 s).
No factors of 2pi, (2l+1) or (l+1/2) appear in this subsystem (they enter in the Limber/Hankel stages, other reports); the only 2pi factors here are the P<->Delta^2 conversion and the
`l(l+1)/2pi` scaling of PP/PT/PE in `mode=cmb` (`camb_interface.py:439-441`, applied to raw lens-potential Cl, with `ell` starting at 2).
