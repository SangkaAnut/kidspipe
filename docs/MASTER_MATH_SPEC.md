# MASTER MATHEMATICAL + NUMERICAL SPECIFICATION
## Cosmic-shear (xi+/-) pipeline of kcap (KV450 fiducial), from cosmological parameters to ln L

Status: consolidated from the seven verified study reports in `docs/study/` (cited as `[01]`..`[07]` with section / QUIRK ids), cross-checked against the reference source in `reference/kcap` (HEAD `36b40c55`, cited as `[K]/path:line`), and **executed**: an independent ~210-line numpy implementation of exactly this specification (`docs/spec_checks/kv450_spec_crosscheck.py`) reproduces a sensible KV450 fit (chi2 = 188.7 for 195 data points with only the amplitude varied; section 15). That script is a witness that the text below is complete enough to re-implement; it is NOT kcap code.

How to read: every step gives (F) the **kcap-faithful** algorithm (what kcap actually computes, including its quirks) and, where useful, (E) a cleaner **exact/fast** replacement with the measured size of the difference. "Decision points" (Dn, section 13) mark every non-obvious kcap choice.

Scope: the KV450 cosmic-shear run `runs/config/KV450_fiducial.ini` (module chain in section 0.2). The KiDS-1000 variants and the COSEBIs / band-power / 3x2pt / Planck extensions are in section 14 at lower detail.

---------------------------------------------------------------------------
## 0. Conventions

### 0.1 Units and symbols

| symbol | meaning | unit |
|---|---|---|
| h | little h = H0/100 | - |
| k | wavenumber, always k_h = k/h in the P(k) tables | h/Mpc |
| P(k,z) | 3-D matter power spectrum (total matter incl. massive nu) | (Mpc/h)^3 |
| chi | **radial** comoving distance, converted to Mpc/h in the projection stage: chi = h * D_C[Mpc] | Mpc/h |
| a | 1/(1+z) | - |
| Omega_m | `omega_m` = (ombh2 + omch2 + omnuh2)/h^2 (**includes** the massive-neutrino density) | - |
| s | lensing prefactor s = (3/2) Omega_m (100/c)^2, c = 299792.4580 km/s | (h/Mpc)^2 |
| nu | l + 1/2 | - |
| l | multipole, a non-integer float on a log grid | - |
| theta | angle; radians inside the code, arcmin for data/edges | rad / arcmin |

arcmin = pi/(180*60) rad. Natural log everywhere. `trapz` = composite trapezoid rule. All arrays z-ascending unless stated.

### 0.2 Module chain (KV450 fiducial) and the maths of each step

```
 params -> sample_ln_As -> one_parameter_hmcode -> camb -> extrapolate_power
        -> load_nz -> source_photoz_bias -> linear_alignment -> projection
        -> add_intrinsic -> cl2xi -> mini_2pt_like  =>  ln L
 (derived_parameters adds S8 only; it does not enter ln L)           [01 s1.2]
```

| step | section of this spec |
|---|---|
| A_s, eta0 | 2 |
| Background, P_lin, HMcode P_nl, k-extension | 3 |
| n(z), photo-z shift | 4 |
| kernels W, N | 5 |
| NLA intrinsic alignments | 6 |
| Limber C_l (GG+GI+II) | 7 |
| C_l -> xi+/- (Hankel) and theta binning | 8, 9 |
| m, c systematics, data vector, covariance, chi2 | 10, 11 |
| prior / posterior | 1, 12 |

---------------------------------------------------------------------------
## 1. Parameters, priors, posterior  [01 s2.1, 05 s3.2]

### 1.1 Varied parameters (KV450_values.ini, 3-number lines are `lo start hi`)

| parameter | lo | start | hi | prior |
|---|---|---|---|---|
| omch2 | 0.01 | 0.10 | 0.99 | flat |
| ombh2 | 0.019 | 0.022 | 0.026 | flat |
| h | 0.64 | 0.70 | 0.82 | flat |
| n_s | 0.7 | 1.09 | 1.3 | flat |
| ln(1e10 A_s) | 1.7 | 3.09 | 5.0 | flat |
| A_baryon (`halo_model_parameters/A`) | 2.0 | 3.0 | 3.13 | flat |
| A_IA | -6 | 0 | 6 | flat (Gaussian N(1.13,0.48) is commented out, `KV450_priors.ini:2`) |
| delta_c | -6e-4 | 0 | 6e-4 | N(0, 2e-4), truncated at the bounds (+-3 sigma) |
| A_c | 0.62 | 1.13 | 1.4 | N(1.01, 0.13), truncated (-3 / +3 sigma) |
| dz_1..dz_5 | -0.1 | 0 | 0.1 | independent N(0, sigma_i), sigma = (0.039, 0.023, 0.026, 0.012, 0.011); bounds are +-2.56, 4.3, 3.8, 8.3, 9.1 sigma |

14 varied parameters. Fixed: Omega_k = 0, w = -1, w_a = 0, m_nu = 0.06 eV (sum), `num_massive_neutrinos` = 1 (integer).
Fixed likelihood constants: m = (-0.0128, -0.0104, -0.0114, 0.0072, 0.0061).

The Gaussian widths 0.039.. are rounded sqrt(diag(DIR_cov.asc)) = (0.0377, 0.0225, 0.0247, 0.0119, 0.0101); the off-diagonals of `DIR_cov.asc` are NOT used in KV450 (independent shifts) [01 s2.1]. (Published ~90 % neighbouring-bin correlation is ignored; the KiDS-1000 setup uses the Cholesky map of section 14.4.)

### 1.2 Derived-parameter formulas

```
A_s   = exp(ln_1e10_A_s) * 1e-10                      (sample_ln_As.py:8-10)
eta0  = a0 + a1 * A_baryon,   a0 = 0.98, a1 = -0.12    (one_parameter_hmcode.py:25-30)   -> A=3.13: eta0 = 0.6044; A=2: 0.74
S8    = sigma8 * (Omega_m/0.3)^0.5   (CAMB writes S_8; derived_parameters writes S8; same formula)
```

### 1.3 Prior and posterior

Flat on [lo,hi]: ln pi = -ln(hi-lo). Gaussian N(mu,sig) on a varied parameter is a **truncated** normal on [lo,hi]:
`ln pi(x) = -(x-mu)^2/(2 sig^2) - ln(Phi(b)-Phi(a)) - 0.5 ln(2 pi sig^2)`, a=(lo-mu)/sig, b=(hi-mu)/sig; -inf outside [lo,hi] (CosmoSIS 3.12.1 semantics; the 2020 fork is unverified) [01 s2.1]. Posterior:

```
ln P(theta) = sum_i ln pi_i(theta_i) + ln L(theta),      ln L = -chi2/2   (no -0.5 ln|C|, no Hartlap)
```
A failing module (non-zero status, NaN) gives ln L = -inf. The normalisation of the priors only matters for evidences.

---------------------------------------------------------------------------
## 2. Boltzmann stage: configuration of CAMB  [02 s3, 01 s3.1]

kcap delegates background, linear transfer functions and (HMcode) non-linear P to python CAMB in `mode=transfer`. A clean-room code may use CAMB (recommended; it is a documented dependency, not kcap code) or any equivalent Boltzmann code, but must reproduce the outputs below. **Pin and record the CAMB version** (D2).

### 2.1 CAMB parameter object (effective settings)

| setting | value |
|---|---|
| ombh2, omch2, H0 = 100 h | sampled |
| mnu, num_massive_neutrinos | 0.06 eV, 1 |
| neutrino_hierarchy | not set in KV450 inis -> CAMB default (`degenerate`; one massive eigenstate, degeneracy 1.0147). ('normal' is used by the KiDS-1000 generator; identical at mnu=0.06, see D34.) |
| N_eff, T_CMB, YHe | CAMB defaults (N_eff = 3.044 in CAMB 1.5.8, 3.046 in older versions; T = 2.7255 K; YHe from the BBN table). omnuh2 = mnu/94.0641058 * (N_eff/3)^0.75 = 6.44867e-4 at 0.06 eV, N_eff=3.044. |
| Omega_k, w, w_a | 0, -1, 0 (`DarkEnergyFluid`, cs2 = 1) |
| initial power | `InitialPowerLaw(ns, As)`, adiabatic, pivot k_p = 0.05 Mpc^-1, no running, no tensors |
| reionisation | **off** (`do_reionization = F`; tau not needed) |
| WantCls | **False** (`WantTransfer=True`, `DoLensing=F`). Matters at 2e-3 in P_lin if mismatched [07 s1.2]. |
| accuracy | `set_accuracy()` defaults (all boosts 1, DoLateRadTruncation True); `set_matter_power` forces `Transfer.high_precision = True` |
| non-linear model | intended: `Halofit(halofit_version='mead', HMCode_A_baryon = A, HMCode_eta_baryon = eta0)` = HMcode-2016 (D1: the shipped ini mis-spells the key) |

### 2.2 Grids

| quantity | definition |
|---|---|
| P(k,z) redshifts | z = concat( linspace(0, 2, 100, endpoint=False), linspace(2, 6, 50) ); 150 nodes; spacing 0.02 below z=2 and 4/49 above; passed to CAMB descending, returned ascending |
| P(k,z) k range | `set_matter_power(kmax=20.0)`: **kmax is in 1/Mpc, not h/Mpc**; CAMB 1.5.8 pads by 5 %, so k_max = 21 Mpc^-1 = 21/h h/Mpc (30.0 h/Mpc at h=0.7; 32.8 at h=0.64; 25.6 at h=0.82); k_min = 5e-5 Mpc^-1 (7.14e-5 h/Mpc at h=0.7); CAMB's own k spacing, **194-197 nodes** (cosmology dependent) |
| distances | z_b = linspace(0, 6, 6000) |
| outputs used downstream | `matter_power_nl(z,k_h,p_k[nz,nk])`, `matter_power_lin`, `distances/{z,a,D_M}` (D_M in Mpc = comoving **radial** distance, valid only for Omega_k = 0, D5), `cosmological_parameters/omega_m` |

Measured with CAMB 1.5.8 (this pass): (omch2, ombh2, H0, ns, ln10As) = (0.10, 0.022, 70, 1.09, 3.09), mnu=0.06 -> nk=194, k_min=7.1429e-5, k_max=30.0 h/Mpc, sigma8(z=0)=0.779557, Omega_m=0.250296, omnuh2=6.44867e-4, N_eff=3.044.

### 2.3 Background formulas (for reference / validation)

```
E(z)^2 = (Om_c+Om_b)(1+z)^3 + Om_nu(z) + (Om_gamma + Om_nu_massless)(1+z)^4 + Om_K(1+z)^2 + Om_DE      (w=-1: Om_DE const)
chi(z) = (c/H0) * int_0^z dz'/E(z')           [Mpc]    -> kcap uses CAMB comoving_radial_distance
Om_m   = (ombh2 + omch2 + omnuh2)/h^2         (CAMB p.omegam)
```
Massive neutrinos use the exact Fermi-Dirac density (Om_nu(z) -> relativistic early, omnuh2/h^2 (1+z)^3 late). Distances are used in Mpc/h: chi = h * chi_Mpc.

### 2.4 Linear power spectrum

```
P_lin(k,z) = (2 pi^2 / k^3) * P_R(k) * T_tot(k,z)^2       [Mpc^3]  ;  P_R = A_s (k/0.05 Mpc^-1)^(n_s-1)
P_h(k_h,z) = h^3 P_lin(k = h k_h, z)                       [(Mpc/h)^3]
```
`T_tot` is CAMB's total-matter (delta_tot: CDM+baryons+massive nu) transfer function; reproducing CAMB's normalisation is the point of the sigma8 check. sigma^2(R,z) = int dlnk Delta^2(k,z) W_th(kR)^2, W_th = 3(sin x - x cos x)/x^3, Delta^2 = k^3 P/(2 pi^2); sigma8 = sigma(8 Mpc/h) (total matter; cold-matter value is 0.4 % higher). Nothing in the xi+- likelihood uses sigma8 except derived S8.

### 2.5 Non-linear power: HMcode-2016 with a one-parameter baryon model  [02 s2.5]

The intended physics is Mead et al. 2016 ("mead"/"mead2016", id 5) with A_baryon = A, eta_baryon = eta0. A bit-level port needs CAMB's `halofit.f90` (not in kcap, compiled binary). The equations below are the stand-alone Fortran copy shipped in kcap (`structure/meadcb/mhmcamb.f90`), which agrees with CAMB 'mead' to rtol 3e-3 (z<2) / 1.5e-2 (z>=2) **for dark matter only** (A=3.13, eta0=0.603) [07 s1.2]; the baryon path (A, eta0) was not covered by that test.

Notation: Delta2_lin(k,z)=k^3 P_lin/(2 pi^2); g(z)=D(z)/D(0) (the Fortran copy uses the growth-index integral g=exp(-int_a^1 Om_m(a')^gamma da'/a'), gamma=0.55, not an ODE); f_nu = omnuh2/(omega_m h^2); Om_m(z)=Om_m(1+z)^3/E^2; rho_bar = 2.775e11 Om_m (Msun/h)/(Mpc/h)^3; mass grid 256 points log-spaced in [1,1e18] Msun/h; M = (4pi/3) rho_bar R^3.

```
P_HM(k)^alpha = P_2h(k)^alpha + P_1h(k)^alpha
alpha  = 3.24 * 1.85^neff,  clipped to [0.5, 2]
P_2h   = P_lin(k) [1 - f tanh^2( k sigma_v / sqrt(f) )], floored at 0
f      = 0.0095 sigma_v100^1.37, clipped to [1e-3, 0.99]
sigma_v^2    = (1/3) int dlnk Delta2_lin(k,z) / k^2                      (R=0 top-hat)
sigma_v100^2 = (1/3) int dlnk Delta2_lin(k,z) / k^2 * W_th(100 k)^2      (R = 100 Mpc/h)
P_1h   = (1 - exp(-(k/k*)^2)) * (2 Delta_v k^3/(3 pi)) * int dnu g(nu) R_v^3 |W(nu^eta k; R_v, c)|^2   [dimensionless Delta^2 form]
k*     = 0.584 / sigma_v
eta    = eta0 - 0.3 sigma_8(z)                                            (eta0 = HMCode_eta_baryon = 0.98 - 0.12 A)
Delta_v = 418 Om_m(z)^-0.352 (1 + 0.916 f_nu)
delta_c = [1.59 + 0.0314 ln sigma_8(z)] (1 + 0.262 f_nu) [1 + 0.0123 log10 Om_m(z)]
nu      = delta_c / sigma(R,z);  g(nu) = 0.21616 [1 + (0.707 nu^2)^-0.3] exp(-0.707 nu^2/2)     (Sheth-Tormen in nu)
R_v     = R / Delta_v^(1/3)
c(M,z)  = A_baryon (1+z_f)/(1+z) * (g_w(z=10)/g_LCDM(z=10))^1.5          (Bullock + Dolag); z_f from g(z_f) = delta_c g(z)/sigma(0.01^(1/3) R, z)
neff    = -3 - dln sigma^2/dlnR at R_nl (nu(R_nl)=1), clipped to [n_s-4, n_s]
W(k;R_v,c) = NFW transform with ks = k R_v/c:
   [cos ks (Ci((1+c)ks) - Ci(ks)) + sin ks (Si((1+c)ks) - Si(ks)) - sin(c ks)/((1+c) ks)] / [ln(1+c) - c/(1+c)], clipped to [0,1]
```
The halo-model sigma(R) uses **cold** (CDM+baryon) P (Eisenstein-Hu Tcb/Tcbnu ratio); sigma_v, sigma_v100 and sigma_8(z) use **total** P. Below Min_kh_nonlinear = 0.005 h/Mpc the correction is switched off. A_baryon enters only through c(M,z); eta0 only through the window mass-dependence. Sensitivity (this pass, section 15.3): A = 3.13 -> 2.0 changes the amplitude-optimised S8 by +4.6 %.

### 2.6 High-k extension (`extrapolate_power`, applied to BOTH P_lin and P_nl)  [02 s3.4]

For each redshift row independently: fit y = ln P against x = ln k_h by `polyfit(deg 1)` on the **last 3** CAMB nodes; new nodes x_new = linspace(x_max, ln 500, 200)[1:] (199 nodes uniform in ln k from k_max,CAMB to 500 h/Mpc); P_new = exp(a + b x_new). No low-k extension. New nk = 393 (cosmology dependent +-3). Typical slopes: linear -2.57; non-linear -2.16 (z=0) ... -1.68 (z=6). P_nl/P_lin at the last CAMB node = 61.6 (z=0), 31.5 (z=2), 15.5 (z=6).

---------------------------------------------------------------------------
## 3. Redshift distributions and photo-z shifts  [03 s2.2-2.3, 05 s1.4]

### 3.1 Loading (KV450, `load_nz`, `histogram=True`)

Five files `data/KV450/nofz/Nz_DIR_z0.1t0.3, z0.3t0.5, z0.5t0.7, z0.7t0.9, z0.9t1.2 .asc`, each 120 rows x 2 columns (z = lower bin edge 0.00 ... 5.95 step 0.05; n(z) histogram height).

```
z_i  <- z_i + (z_1 - z_0)/2          (lower edge -> bin centre, +0.025)
prepend the node (z=0, n=0)           (always, since z_0 = 0.025 > 1e-8)   -> 121 nodes, z in {0, 0.025, ..., 5.975}
n_i  <- n_i / trapz(n_i, z)           (raw sum*dz = 1 exactly, but the trapz integral is 0.9945..0.9998, so a re-normalisation of 0.2-0.5 % occurs)
```
The scalar `nz` stored in the datablock (=120) differs from len(z)=121; nothing uses it. The tails are not compact: n > 0 up to z = 4.45 (bin 1) / 4.65 (bins 2-5); mass above z=1.2/2/3: bin 5: 18 %/2.0 %/0.27 % [07 s3.1].

### 3.2 Photo-z shift (`photoz_bias`, mode additive, interpolation linear in KV450)

```
n'_i(z) = n_i(z - dz_i)     via  interp1d(z, n_i, kind='linear', bounds_error=False, fill_value=0)(z - dz_i)
n'_i   <- n'_i / trapz(n'_i, z)
```
Positive dz moves the distribution to higher z. Mass shifted below z=0 or beyond 5.975 is truncated and then redistributed by the renormalisation. (KiDS-1000 uses `cubic` interpolation, which can produce slightly negative n(z): <= 7e-4 of the unit norm, [03 Q14]). Lens bins are never shifted.

---------------------------------------------------------------------------
## 4. Lensing and density kernels  [03 s2.4-2.5]

Flat universe. Let n_i(z) be the normalised, shifted histogram of section 3, z(chi) and a(chi)=1/(1+z) from the 6000-node table (chi in Mpc/h).

```
n_i(chi) = n_i(z) dz/dchi = n_i(z) (1+z)^2 (-da/dchi)                      [h/Mpc]     ("N kernel", NOT renormalised)

W_i(chi) = (chi/a(chi)) * [ int_chi^chi_max n_i(chi') (chi' - chi)/chi' dchi' ] / [ int_0^chi_max n_i(chi') dchi' ]       [Mpc/h]
         = (chi/a) * (1/Z_i) * int_{z(chi)}^{z_max} n_i(z') [1 - chi/chi(z')] dz'         (change of variable, exact)
chi_max = chi(z=6) ; Z_i -> 1 for a perfectly normalised n_i
```
The factor (3/2) Omega_m (100/c)^2 = s is **not** inside W; it multiplies each lensing kernel in the prefactor. The physical kernel is `s * W_i`.

(F) kcap numerics: a(chi) and chi(z) are GSL **Akima** splines through the 6000 nodes; n_i(z) is an Akima spline through the 121 nodes (set to 0 for z >= 5.975); W_i is evaluated on **1024 uniformly spaced chi nodes** chi_j = j chi_max/1023 using a fixed **1024-point Gauss-Legendre** rule for both the inner integral (over [chi_j, chi_max]) and the normalisation Z (over [0, chi_max]); the result is stored as an Akima spline. The N kernel is an Akima spline through the points (chi(z_n), n_i(z_n)(1+z_n)^2(-a'(chi_n))) at the n(z) nodes.
(E) recommended: the cumulative-sum form `W_i(chi) = (chi/a)[A_i(z) - chi B_i(z)]`, `A_i(z)=int_z^inf n_i dz'`, `B_i(z)=int_z^inf n_i/chi(z') dz'`, evaluated by reverse cumulative trapezoid on a dense z-grid: O(N) instead of O(1024^2) and free of Akima-derivative noise. Used in `docs/spec_checks`; agrees to better than the 1e-3 Limber tolerance (section 15.4).

---------------------------------------------------------------------------
## 5. Intrinsic alignments: NLA, "bk_corrected"  [03 s2.6]

Only an amplitude A_IA = `intrinsic_alignment_parameters/A` (same for all bins). No redshift power law, no luminosity scaling, no tidal torquing.

```
C1 rho_crit = 5e-14 h^-2 Msun^-1 Mpc^3 -> dimensionless 0.0138731   (Msun=1.9891e30 kg, Mpc=3.0857e22 m, G=6.67384e-11, H=100 h km/s/Mpc)
D(z)   = sqrt( P_nl(k_small, z) / P_nl(k_small, 0) ),   k_small = smallest k of the CAMB grid (~7.1e-5 h/Mpc)   [non-linear spectrum!]
F(z)   = - A_IA * 0.0138731 * Omega_m / D(z)
P_II(k,z) = F(z)^2 P_nl(k,z) ;   P_GI(k,z) = F(z) P_nl(k,z)
```
A_IA > 0 gives negative P_GI. The grid z must contain z=0 exactly. P_II, P_GI inherit the full (z,k) grid including the k<=500 extension.

---------------------------------------------------------------------------
## 6. Limber projection to C_l  [03 s2.5, 2.7, 2.8; 07 s2.1]

### 6.1 The observable convergence-like spectrum

With q_i(chi) = s W_i(chi) + F(z(chi)) n_i(chi):

```
C_ij(l) = int_0^chi_max  dchi  q_i(chi) q_j(chi) / chi^2  *  P_nl( k = (l+1/2)/chi , z(chi) )
```
which expands to GG + GI + IG + II exactly:
`C^GG = s^2 int W_i W_j P_nl/chi^2`, `C^GI_ij = s int W_i n_j P_GI/chi^2` (plus the transposed term), `C^II = int n_i n_j P_II/chi^2`. kcap's KV450 fiducial projects the three spectra separately (`shear-shear`, `shear-intrinsic`, `intrinsic-intrinsic`) and `add_intrinsic` sums
`C_ij = C^GG_ij + C^II_ij + C^GI_ij + C^GI_ji` (the "fast shear+IA kernel" `Q_i = W_i + F n_i/s` of the KiDS-1000 generator is the same algebra with 15 instead of 55 integrals, [03 s2.7.1]).

**This is the convergence spectrum**: no (l-1)l(l+1)(l+2)/(l+1/2)^4 (shear/convergence) factor (0.977 at l=10, [07 s1.3]); no 2pi, no (2l+1); C_l are dimensionless and *not* multiplied by l(l+1)/2pi. Auto-spectra are stored for j <= i only (`bin_i_j`, i>=j), 15 pairs.

### 6.2 kcap numerics (F)

* ell grid: `l = logspace(log10(0.1), log10(5e5), 400)` (non-integer, ratio 1.0394; nu = 0.6 at the first node). KiDS-1000 xi-less runs use 1..3e4.
* Reduced kernels: kcap integrates `Wr_X Wr_Y P /(chi D^2)` with `Wr = W D/sqrt(chi)` and D(chi) = sqrt(P(k_g,z)/P(k_g,0)) at the first grid k above k_g = 1e-3 h/Mpc; D cancels analytically (spline-noise only) [D14].
* The 3-D spectrum is interpolated by a GSL **bicubic** spline in (chi_j, ln k) of **P itself** (not ln P), chi_j = chi(z_j) on the CAMB z grid; **outside the rectangle (k < k_min or k > 500 h/Mpc, chi outside the table) the integrand is exactly 0** (no warning) [D13].
* Integration limits: [max of the two kernels' lower nodes, min of their upper nodes] = [0, chi_max] for W kernels; for the N kernel the upper limit is chi(z_last of n(z)) = chi(5.975).
* Quadrature: adaptive GSL QAG, GAUSS61, `epsabs = 0`, `epsrel = 1e-3`, limit 4096 sub-intervals; if it fails, a fixed 4096-point Gauss-Legendre. Extended-Limber correction hard-wired off (`ext_order = 0`); `K` is never passed to the integrator (flat only).
* Thus C_l carries ~1e-3 noise that is a non-smooth function of the parameters [D21].

### 6.3 Recommended exact/fast form (E)

Tabulate Q(l, chi_m) = P_nl((l+1/2)/chi_m, z_m) once per cosmology on a (400 x ~2000) grid (use ln P in (z, ln k), zero outside the table), then all 15 pairs are one contraction `C[i,j,l] = sum_m w_m q_i(m) q_j(m) Q(l,m)/chi_m^2`. Measured: 2000 trapezoid nodes in z reproduce a 12000-node reference to 1.5e-4 (10<l<2e4); 1000 nodes 6e-4; 500 nodes 3e-3 (this pass). Total kernels + Limber time 0.16 s in numpy (section 15.5).

---------------------------------------------------------------------------
## 7. C_l -> xi+/-: the nicaea FFTLog-style Hankel transform  [04 s2.1-2.3]

### 7.1 Definition

```
xi_+(theta) = (1/2pi) int_0^inf dl  l  C(l)  J_0(l theta)
xi_-(theta) = (1/2pi) int_0^inf dl  l  C(l)  J_4(l theta)
```
Flat-sky, continuous-l form fed by the convergence-like C_l of section 6 (D24). Only E-modes enter.

### 7.2 Algorithm (F), `cl_to_xi_nicaea` (`maths.c:1877-2008`), applied per tomographic pair

Fixed grid, independent of the input ell:
```
N = 1024, l_min = 1e-4, l_max = 5e6
Dl   = ln(l_max/l_min)/(N-1) = 0.0240814163
lnrc = 0.5 (ln l_max + ln l_min) ;  nc = N/2 + 1 = 513
l_i  = exp( lnrc + (i - nc) Dl ),  i = 0..1023     (l_0 = 9.645e-5, l_1023 = 4.8226e6: shifted by 1.5 steps, D23)
```
1. Interpolate C onto l_i. The 400 input nodes are assumed exactly log-uniform (only the end nodes are read).
   * all C > 0: ln C linear in ln l between nodes; **below** l=0.1 extrapolate with slope +1 (C ~ l), **above** l=5e5 with slope -3 (C ~ l^-3). Entries |C|<1e-30 are floored to 1e-30 only if all entries share a sign.
   * all C <= 0: same on ln(-C), sign restored afterwards.
   * mixed signs (or any exact zero): C linear in ln l with **constant** extrapolation (the intended "no extrapolation error" is dead because of `#define EPS 1.0-5`, [04 Q2]).
2. f_i = l_i C(l_i); F_m = rfft(f)_m (unnormalised, m=0..N/2); k_m = 2 pi m/(Dl N).
3. Kernel for order mu (mu=0 for xi+, mu=4 for xi-; bias q=0):
   `K_mu(k) = exp(i k ln2) * Gamma((1+mu)/2 + i k/2) / Gamma((1+mu)/2 - i k/2)`, with kcap's `ln2 = 0.69314718` (8 digits; effect <= 7e-8 rad, only matters for bit-matching). Force Im[conv_0] = Im[conv_{N/2}] = 0.
4. r = N * irfft(F K_mu); t_i = exp((nc - i) Dl - lnrc) = 1/l_i;  `xi[N-1-i] = r_i / (t_i N 2 pi) = l_i * irfft(F K)_i / (2 pi)`.
5. Output grid: `theta_j = 1/l_{N-1-j} = exp(ln theta_min + j Dl)`, theta in [2.0736e-7, 1.0368e4] rad = [7.1e-4, 3.6e7] arcmin; 265 nodes fall in [0.5', 300'] (j = 273..537).

A numpy transcription (function `hankel_xi` in `docs/spec_checks/kv450_spec_crosscheck.py`; also `docs/study/04_verification/nicaea_logfft_hankel_check.py`) reproduces a direct quadrature of the integral to ~1e-8 relative at 1' for mu=0 and 4e-6 at 10' for mu=4 [04 s2.2].

Validity: the module documents ~1' to a few hundred arcmin; the kept xi+ bins start at 0.5' (D25). Cost: 15 pairs x (1 rfft + 2 irfft of length 1024) -> ~5 ms in numpy; the complex Gamma kernel is cosmology-independent and can be cached.

(E) alternative: precompute a linear operator from C_l on a fine l grid to the 195 binned values (bin-averaged Bessel integrals); gives the same numbers to better than 1e-3 in chi2 terms (section 15.3).

---------------------------------------------------------------------------
## 8. Binning of the theory in theta  [05 s2.2, 04 s2.4b]

Bin edges (arcmin): `edges = exp(linspace(ln 0.5, ln 300, 10))` = 0.5, 1.01777898, 2.0717481, 4.21716333, 8.58428037, 17.4738002, 35.56893304, 72.40262468, 147.37973879, 300.0 (ratio 600^(1/9)=2.0356). Converted to radians.

(F) `mini_2pt_like` "integrate" mode, per pair and per sign:
```
for b in 0..8:  S_b = { j : edges[b] <= theta_j < edges[b+1] }        (nodes of the 1024-grid; 29,30,29,30,29,30,29,30,29 nodes)
   xi_bin[b] = trapz(theta_S * xi_S, theta_S) / trapz(theta_S, theta_S)        (theta-weighted, edges NOT included)
```
Only grid nodes inside the bin are used; the partial end intervals are dropped (bias up to ~0.6 % of a bin value, alternating with the 29/30 node parity; irrelevant to chi2: Delta chi2 = 0.04, section 15.3). The data theta of the data file are not used in this mode. (E): exact bin average `int theta xi dtheta / int theta dtheta` over the true edges from a cubic spline in ln theta.

---------------------------------------------------------------------------
## 9. Systematics at likelihood level  [05 s2.2, 01 s2.3]

Per tomographic pair (a,b):
```
mu+_ab[k] = B_k[xi+_ab] + delta_c^2 + A_c^2 * c+[k]
mu-_ab[k] = B_k[xi-_ab]            + A_c^2 * c-[k]
d_ab = x_ab / ((1+m_a)(1+m_b))                       (m-bias divides the DATA; covariance untouched; fixed m)
```
* `delta_c^2`: the fiducial ini uses `order_cov = montepython`, which **squares** delta_c and adds it to xi+ only (the `xi_pm-bin-theta` branch adds delta_c linearly, D29). The offset is <= 3.6e-7.
* c-term file `data/KV450/systematics/KV450_ALL_c12_treecorr.out`: 9 rows x 10 columns (TreeCorr output); `usecols=[3,4]` = (xi_p_c, xi_m_c): xi_p_c = 5.38e-7, 4.98e-7, 3.98e-7, 1.92e-7, 2.27e-8, -1.93e-8, 1.07e-8, 4.78e-9, 2.80e-9; xi_m_c = 1.04e-9, 2.43e-9, 7.22e-9, 9.91e-9, 2.80e-9, -5.23e-9, 2.34e-8, -1.54e-9, -2.63e-10. The **same** 9-vector is used for all 15 pairs and then cut like the data.
* No m uncertainty, no marginalisation, no extra m nuisance parameter (D28). Whether the data file (named "..._mcor_...") is already m-corrected is unresolved in the repo (O9); the experiment in section 15.3 shows the S8 effect of the (double) correction is only +-0.2 %.

---------------------------------------------------------------------------
## 10. Data vector, scale cuts, covariance  [05 s1.4, 2.2]

### 10.1 Files

| file | layout |
|---|---|
| `data/KV450/data_vector/KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat` | 18 rows x 16 columns, `%.4e`. Column 0 = theta [arcmin], pair-weighted means 0.78613, 1.5996, 3.2535, 6.623, 13.493, 27.385, 55.541, 112.61, 223.72 (rows 0-8 for xi+, rows 9-17 repeat them for xi-). Columns 1..15 = the 15 pairs in montepython order. Rows 0-8 = xi+, rows 9-17 = xi-. |
| `data/KV450/covariance/cov_analytic_montepython_mcorr.txt` | 270 x 270, analytic, symmetric, positive definite (min eig 4.7e-14); layout: for each pair in montepython order [xi+(9), xi-(9)] |

### 10.2 Ordering ("montepython")

Tomographic bins 0-based. Loop `for i in 0..4: for j in i..4:` pair (j,i) (higher bin index first): (1,1),(2,1),(3,1),(4,1),(5,1),(2,2),(3,2),(4,2),(5,2),(3,3),... in 1-based labels. Data-file column of pair (j,i) with j>=i is `1 + i*(2n-i+1)/2 + (j-i)`, n=5 (i.e. `(j*(2n-j+1))//2 + i-j` in the source's swapped naming). Theory is read from `shear_xi_plus/bin_{j+1}_{i+1}`. Each pair contributes `[xi+ (kept thetas), xi- (kept thetas)]`.

### 10.3 Cuts

`cut_xi_plus = 7 8` (0-based theta indices: drop bins 8,9 -> keep xi+ bins 1-7, theta <= 72.4'), `cut_xi_minus = 0 1 2` (drop bins 1-3 -> keep xi- bins 4-9, theta >= 4.2'). Applied identically to every pair, to the data, to the theory, to the c-term vectors and (as row/column deletions *before* inversion) to the covariance. Result: 15 x (7+6) = **195** points; 195x195 covariance, condition number 2.4e5. The cut covariance is inverted once (`np.linalg.inv`); no Hartlap or Sellentin correction (analytic covariance).

---------------------------------------------------------------------------
## 11. Likelihood  [05 s2.1]

```
r    = d - mu                                (195-vector, same ordering)
chi2 = r^T C^-1 r ;   ln L = -chi2 / 2
```
No ln|C|. `keep_theory_vector = T` in the fiducial ini additionally draws a random simulated vector every call and writes the inverse covariance to the datablock; it does not enter ln L (D38). Published reference: KV450 maximum-likelihood chi2 = 180.6 for 195 points and 14 parameters ([Hildebrandt et al. 2020, A&A 633, A69, Table 6], external literature, not in the repo).

---------------------------------------------------------------------------
## 12. Complete algorithm (pseudo-code, kcap-faithful)

```
INPUT theta = (omch2, ombh2, h, ns, lnAs10, A_b, A_IA, delta_c, A_c, dz[5])
 1  lnprior = sum of flat / truncated-Gaussian terms (section 1.3); if -inf: return -inf
 2  A_s = exp(lnAs10)*1e-10 ; eta0 = 0.98 - 0.12*A_b
 3  CAMB (section 2): z150, k grid (nk~194..197), P_lin, P_nl (HMcode-2016 with A_b, eta0), chi(z_b[6000]) in Mpc, omega_m
 4  extend P_lin and P_nl to k_h = 500 (section 2.6)
 5  n_i(z) <- load (section 3.1) ; shift by dz_i, renormalise (section 3.2)
 6  chi <- h*chi_Mpc ; build W_i, n_i(chi) (section 4) ; IA: F(z) (section 5)
 7  for ell in logspace(log10 0.1, log10 5e5, 400): C_ij(ell) = Limber(q_i, q_j, P_nl)  (section 6)   [15 pairs]
 8  for each pair: xi+, xi- on the 1024 theta grid via the FFT Hankel transform (section 7)
 9  bin to 9 theta bins (section 8); delete cut bins; add delta_c^2 and A_c^2 c-terms (section 9)
10  mu <- concatenate over pairs in montepython order ; chi2 = (d-mu)^T Cinv (d-mu)  (sections 10-11)
11  return lnprior - chi2/2
```
Array shapes: P `[150, ~194]` -> `[150, ~393]` after extension; C_l `[5,5,400]` (15 independent); xi `[15,1024]` per sign; mu `[195]`.
Fast/slow structure: A_IA, delta_c, A_c, dz_1..5 (8 parameters) never touch CAMB; A_baryon is read by `camb` (slow) [01 s7.3].

---------------------------------------------------------------------------
## 13. Decision points: where kcap makes a non-obvious choice

"Effect" values marked (M) were measured in this pass with the independent implementation (section 15.3): change of the amplitude-optimised S8 and of the minimum chi2 (195 points) at (omch2, ombh2, h, n_s) = (0.12, 0.0222, 0.70, 0.97), A_baryon = 3.13, vs baseline S8 = 0.7070, chi2 = 188.67.

| ID | topic | kcap choice | alternative / textbook | effect | source |
|---|---|---|---|---|---|
| D1 | Halofit model | key `halofit_verion` (typo) in `KV450_fiducial.ini:124` is **silently ignored**; CAMB's default applies (mead2020 in CAMB 1.5.8, which **ignores** A_baryon/eta0; intended mead = HMcode-2016). KiDS-1000 generator passes `mead` explicitly. | spell the key; pin `halofit_version='mead'` | mead2020: S8 +0.3 %, chi2 -0.2 (M); A_baryon becomes unconstrained | [01 Q1][02 4.1][07 Q1] `camb_interface.py:94` |
| D2 | CAMB version | unpinned (`conda_env.yaml`, README `pip install camb`) | pin and record in output | N_eff 3.044 vs 3.046, BBN YHe table, HMcode implementation, k-grid (194 vs 215 nodes) | [02 8.1][07 6.3] |
| D3 | kmax | `kmax=20` is **1/Mpc** (not h/Mpc); CAMB pads x1.05 -> k_h,max = 21/h, h-dependent | state kmax in h/Mpc | kmax 10 vs 20: Delta chi2 = 0.015 (M) (CAMB warns for HMcode, z>4) | [02 3.3][01 Q2] |
| D4 | z grid | 100 pts to z=2 (dz 0.02) + 50 to 6 | uniform / Chebyshev | 41-node grid: Delta chi2 < 1e-3 (M), HMcode time /3 | [02 3.2] |
| D5 | distances | `D_M = comoving_radial_distance` (flat only); chi = h D_M | transverse S_k(chi) | omega_k fixed 0: none; wrong for curvature | [02 Q4.6][06 Q1] |
| D6 | Omega_m | CAMB `omegam` **includes** massive nu | CDM+baryon only | 0.45 % in Omega_m (hence in s and F), ~0.2 % in S8 at 0.06 eV | [02 2.2][03 Q9] |
| D7 | sigma8 / S8 | total-matter sigma8 at first z-grid node; S8 = sigma8 (Omega_m/0.3)^0.5 (derived output only) | cold matter | none on ln L | [02 2.6] |
| D8 | sampled amplitude (KiDS-1000 only) | S_8_input with Omega_cb, then A_s via a **second low-accuracy CAMB** run | sample ln A_s (KV450) | +0.07 % sigma8, +0.28 % S8 offset; grows with m_nu | [02 2.7][06 Q2] |
| D9 | k > k_CAMB | 3-point log-log linear extension to 500 h/Mpc per z, for P_lin and P_nl | HMcode asymptote / emulator | entire l >~ 1e4 tail; not separately quantified | [02 3.4, Q4.10] |
| D10 | interpolation of P | GSL bicubic of **P** (linear) in (chi, ln k); 0 outside | spline in ln P | O(5e-4) ringing (estimate); my (E) uses ln P | [03 3.2][07 Q11] |
| D11 | Limber flavour | k=(l+1/2)/chi, first-order Limber, convergence spectrum (no F^2 factor) | exact full-sky / extended Limber | F^2 = 0.977 at l=10, 0.9999 at l>=100 (estimate: affects only the largest-theta bins) | [03 Q1][07 Q2] |
| D12 | extended Limber | `ext_order=0` hard-wired | ext_order 1 | unquantified | [03 s2.5] |
| D13 | P support | integrand = 0 if k outside [k_min, 500 h/Mpc] (silent) | error / extrapolation | truncates l > 500 chi | [03 Q15][07 s2.1] |
| D14 | reduced kernels | Wr = W D/sqrt(chi), D(chi) from the integrated P itself | D = 1 | spline noise ~1e-3 | [03 Q2] |
| D15 | W normalisation | divided by numerically integrated Akima n(chi) (1024-pt GL); N kernel uses trapz-in-z normalisation only | one consistent normalisation | ~1e-3 | [03 Q3] |
| D16 | n(z) histogram | `load_nz`: z+0.025, prepend (0,0), trapz normalise; Akima through bin centres (KiDS-1000 `load_nz_fits`: Z_MID, no half-bin shift) | true histogram | few 1e-3 | [03 Q4] |
| D17 | photo-z shift | additive n(z-dz), linear interp (K1000: cubic), zero fill, trapz renormalise | multiplicative / exact histogram shift | truncation of tails at z=0 | [03 2.3, Q14] |
| D18 | IA model | NLA, F = -A C1rho Omega_m / D(z), D from P_nl at the lowest k, no (1+z)^eta, no luminosity; single A for all bins; C1rho = 0.0138731 (G, Mpc rounded; textbook 5e-14 x 2.775e11 = 0.013877) | TATT, z/lum evolution | A_IA=+1: S8 +3.1 %; -1: -5.2 % (M); published z-evolution test: negligible | [03 2.6, Q9][Hildebrandt+20] |
| D19 | add_intrinsic | C = GG+II+GI(i,j)+GI(j,i); the intended `shear_cl_gg` copy is a **no-op** (`add_intrinsic.py:52` aliases `shear_cl`) | - | none on ln L (resolved vs [07 2.8]) | [03 Q6] source verified |
| D20 | ell grid | 400 log nodes, 0.1..5e5, floats | fewer nodes | 120 nodes 1..3e5: Delta chi2 = 0.01 (M) | [03 2.8] |
| D21 | Limber tolerance | QAG rel 1e-3, fallback GL4096 | fixed quadrature | 1e-3 noise in C_l | [03 Q17][07 Q12] |
| D22 | C_l interpolation / extrapolation inside Hankel | log-log linear; slope +1 below 0.1, -3 above 5e5; constant for mixed signs | cubic in ln C, physical tail | tail above 5e5 matters for xi+ at theta < 1' | [04 2.3, Q2] |
| D23 | Hankel grid | N=1024, 1e-4..5e6, `nc=N/2+1` shift, `ln2` 8 digits, no FFTLog bias | FFTLog with low-ringing kr | <= 1e-7 | [04 Q3, Q4] |
| D24 | xi definition | flat-sky continuous (1/2pi) int l C J_mu | discrete (2l+1)/4pi Wigner sum with shear C_l | tiny except theta > 100' | [07 s2.2] |
| D25 | validity window of the transform | used from 0.5' although module doc says ~1' | - | not measured | [05 Q9] |
| D26 | theta binning | node trapezoid inside [edge_b, edge_b+1), theta-weighted; data theta unused | exact edges, pair-count weighting | up to 0.6 % per bin; Delta chi2 = 0.04 (M) | [05 Q8][04 2.4] |
| D27 | binning bug (xipm run type only) | `theta_max = xi_theta_min` passed to `bin_xi_*` in `run_kcap.py:592,611,634` | fix | irrelevant to the fiducial | [04 Q10] |
| D28 | m-bias | divide the **data** by (1+m_a)(1+m_b), fixed m; no m uncertainty; possible double correction | multiply the theory; marginalise | removing: S8 +0.2 %, chi2 -3.2; doubling: S8 -0.2 %, +3.5 (M) | [05 Q5, Q6] |
| D29 | additive c | `delta_c**2` (montepython branch; linear in the other branch) to xi+ only; A_c^2 x c-file, same 9-vector for all pairs | A_c linear (C++ module) | c-file removal: Delta chi2 = 0.8 (M) | [05 Q4, Q7] |
| D30 | data ordering | "montepython" layout | xi_pm-bin-theta | must match covariance exactly | [05 2.2] |
| D31 | cuts | by 0-based theta index, same for all pairs | by angle per pair (scale_cuts) | identical for KV450-like presets | [05 Q20] |
| D32 | covariance | analytic, fixed, inverted once; no Hartlap; no ln det | cosmology-dependent / Sellentin | - | [05 2.8] |
| D33 | priors | truncated Gaussians; delta-z independent in KV450, Cholesky-correlated in K1000; A_IA flat | correlated; Gaussian A_IA | published: negligible on S8 | [01 2.1][Hildebrandt+20 setups 10,11] |
| D34 | neutrino hierarchy | KV450: CAMB default (degenerate, 1 eigenstate); K1000: `normal`; `sigma8toAs` omits it | - | identical at 0.06 eV; differs at >= 0.3 eV (verified: normal+1 eV -> 3 massive states, default -> 1) | [02 2.8][06 Q2] |
| D35 | baryon-feedback range | A_baryon in [2.0, 3.13]; eta0 = 0.98 - 0.12 A | wider / HMcode2020 | A=2.0: S8 +4.6 % (M) | [01 2.2] |
| D36 | spline library | GSL Akima / bicubic end conditions | scipy | reproducing kcap to < 1e-4 needs GSL | [03 8.2] |
| D38 | `keep_theory_vector=T` | RNG draw (195-dim) + inverse-covariance copy per call; no effect on ln L | F | 5-17 ms per call | [05 Q11] |
| D37 | threading | CAMB OpenMP default threads | pin threads | HMcode 4x slower at 22 threads (3.7 s vs 1.0 s at 4) (M) | [02 s6][07 s7] |

---------------------------------------------------------------------------
## 14. Extensions (lower detail)

### 14.1 COSEBIs  [04 s2.5]
```
E_n = (1/2pi) int dl l W_n(l) C_E(l),   W_n(l) = int_{tmin}^{tmax} dtheta theta T_+n(theta) J_0(l theta)
    = 1/2 int dtheta theta [ T_+n xi_+ + T_-n xi_- ],   B_n with a minus sign
```
`T_+n(z) = norm_n prod_{i=1}^{n+1}(z - r_{n,i})` in z = ln(theta/theta_min); roots/norms from `cosebis/TLogsRootsAndNorms/{Root,Normalization}_<tmin>-<tmax>.table` (names with 2 decimals; (0.50, 300.00) has 20 modes). Defining conditions: degree n+1 polynomial; int e^{2z} T = int e^{4z} T = 0; orthonormal in dtheta (int dtheta T_n T_m = (theta_max - theta_min) delta_nm, theta in arcmin); positive leading coefficient. `T_-n(z) = T_+n(z) + int_0^z dy 4 T_+n(y)[e^{2(y-z)} - 3 e^{4(y-z)}]`. Recipe to regenerate with mpmath: `docs/study/04_verification/cosebis_roots_norms_from_scratch.py`. kcap: n_max = 5, l in [1, 10^4/theta_max_rad] (= 1.146e5 for 300'), natural cubic spline in ln l of linear C with power-law extrapolation. Data: `data/KV450/combined/COSEBIs_En_nModes20.fits` (300 E_n values = 15 pairs x 20 modes + 300x300 COVMAT).

### 14.2 Band powers (KiDS-1000)  [04 s2.6]
```
BP_i = (1/N_i) int_{0.1}^{1e4} dl l W^i(l) C(l),  N_i = ln(lmax_i/lmin_i),
W_mu^i(l) = int dtheta theta T(theta) J_mu(l theta) g_mu^i(theta)
```
8 log bands in l = [100, 1500]; top-hat response; analytic g_0, g_2, g_4 (closed forms in [04 s2.6]); cos^2 apodisation T of log-width Delta_x = 0.5 about theta in [0.5', 300'] (integration range 0.39'..385'); E-mode shear BP_E = (BP_0 + BP_4)/2, GGL BP = BP_2. Effective angle for scale cuts = geometric band centre (118.4, 166.2, 233.1, 327.0, 458.7, 643.5, 902.8, 1266.4).

### 14.3 3x2pt (KiDS-1000 + BOSS + optional Planck)  [06]
* Sources: 5 KiDS bins (shear), 2 BOSS lens bins (z_eff 0.38, 0.61). "3x2pt" here = cosmic-shear band powers `PeeE` (120 pts) + GGL band powers `PneE` (22 pts kept after cuts) + BOSS DR12 xi wedges (168 pts); KiDS part 142 points, 142x142 covariance.
* P_gm = b1 P_nl + [b2 N_b2(k) - gamma2 N_g2(k) - gamma3 N_g3(k)] P_lin^2 (k >= 0.01 h/Mpc), gamma2 = -(2/7)(b1-1), N_X = exp(A_X (ln k)^2 + B_X ln k + C_X), A_X.. quadratic in (n_s, omch2/h) with coefficients from `modules/P_gm_approx/parameter_gridfit_*.dat`; low-z bias for z<0.5, high-z else.
* GGL: `C_gG = s int n_i W_j P_gm/chi^2 + int n_i n_j F P_gm/chi^2 + s^2 2(alpha_i-1) int W_i^lens W_j P_nl/chi^2` (position-shear + galaxy-intrinsic + magnification with alpha = 1.8, 2.62).
* BOSS: Gaussian chi2 per z-bin with Hartlap factor (n_mocks - p - 2)/(n_mocks - 1) = 0.958476 (p=84, n_mocks=2048); wedge model module is **not in the repo**.
* Planck: clik interface and cobaya lensing wrapper; additive in ln L.

### 14.4 KiDS-1000 differences vs KV450
Parameter set (omch2, ombh2, h, n_s, S_8_input, A_baryon, A_IA, p_1..5, + 8 bias parameters for 3x2pt) [01 s2.1]; photo-z shifts dz = L p with L = chol(SOM_cov_multiplied.asc) and p ~ N(L^-1 mu_dz, 1); cubic photo-z interpolation; `load_nz_fits` (Z_MID, no half-bin shift); fast shear+IA kernel; ell 1..3e4 (400 nodes); no Hankel transform (band powers); `consistency` module not used.

---------------------------------------------------------------------------
## 15. Evidence that this specification is complete and correct

### 15.1 Executable witness
`docs/spec_checks/kv450_spec_crosscheck.py` follows sections 1-11 using python CAMB 1.5.8 for the Boltzmann stage (linear n(z) interpolation, trapezoid Limber in the (E) form, `mead` HMcode with eta0 = 0.98-0.12A). Run: `OMP_NUM_THREADS=4 python3 kv450_spec_crosscheck.py --scan`.

### 15.2 Result
Amplitude scan at (omch2, ombh2, h, n_s) = (0.12, 0.0222, 0.70, 0.97), A_baryon=3.13, A_IA=0, dz=0, delta_c=0, A_c=1 (195 data points):

| ln(1e10 A_s) | sigma8 | S8 | chi2 |
|---|---|---|---|
| 2.60 | 0.6573 | 0.648 | 192.39 |
| 2.80 | 0.7264 | 0.716 | 188.77 |
| 3.00 | 0.8028 | 0.791 | 199.43 |
| 3.20 | 0.8873 | 0.875 | 239.69 |
| 3.40 | 0.9806 | 0.967 | 336.76 |
| 3.60 | 1.0837 | 1.068 | 537.64 |

Minimum chi2 = 188.67 at ln A_s10 = 2.774 (sigma8=0.717, S8 = 0.707 with Omega_m = 0.292). The null chi2 (data vs zero) is 269.3, so a model-free S/N is ~16 and the fit removes it with a single amplitude. Published KV450 maximum-likelihood chi2 = 180.6 (14 free parameters), posterior-mean S8 = 0.737 +0.040/-0.036 (external). The same pipeline therefore sits in the right place, which would not happen with a wrong ordering, cut, m-correction, c-term, covariance, Hankel transform or Limber prefactor.

### 15.3 Sensitivities (all other settings as baseline)
| variation | S8 at chi2 minimum | chi2 min |
|---|---|---|
| baseline | 0.7070 | 188.67 |
| A_IA = +1 | 0.7287 (+3.1 %) | 186.12 |
| A_IA = -1 | 0.6701 (-5.2 %) | 195.39 |
| no m-correction | 0.7085 (+0.2 %) | 185.48 |
| m-correction applied twice | 0.7055 (-0.2 %) | 192.18 |
| no c-term file | 0.7084 (+0.2 %) | 189.46 |
| A_baryon = 2.0 (eta0 = 0.74) | 0.7397 (+4.6 %) | 187.26 |
| halofit mead2020 | 0.7093 (+0.3 %) | 188.51 |
| halofit takahashi | 0.7015 (-0.8 %) | 188.54 |
| exact bin average (spline, true edges) | 0.7074 | 188.64 |
| 120-node ell grid (1..3e5) | 0.7070 | 188.66 |
| 41-node P(k,z) z grid | - | 188.668 vs 188.668 |
| kmax = 10 /Mpc | - | 188.653 |

### 15.4 Numerical convergence of the (E) Limber
Relative deviation of C_ij(l) vs a 12000-node reference, 10 < l < 2e4: 3.2e-3 (500 nodes), 6.1e-4 (1000), 1.5e-4 (2000), 5.4e-5 (4000).

### 15.5 Timing of this implementation (one cold evaluation)
| stage | 4 CAMB threads | 1 thread | 22 threads (default) |
|---|---|---|---|
| CAMB transfer + HMcode (150 z) | 1.00 s | 2.86 s | 3.7 s |
| n(z) shift + kernels + Limber (all 15 pairs, GG+GI+II folded) | 0.16 s | 0.15 s | 0.26 s |
| Hankel (15 pairs, both signs) + binning | < 0.01 s | | |
| likelihood | < 1 ms | | |

---------------------------------------------------------------------------
## 16. Open items that no amount of reading the repo can settle
1. CAMB version behind the published numbers (decides D1, N_eff, YHe) [01 s8, 02 s8].
2. Whether `*_mcor_*` data are already m-corrected (O9, [05 s8]).
3. GSL Akima/bicubic end conditions (needed for < 1e-4 agreement) [03 s8.2].
4. The `wedges` BOSS model, `deltaz.asc` and the other KiDS-1000 data products (not in the clone) [06 s8].
5. CosmoSIS fork semantics (`ins` default, prior normalisation, dypolychord) [01 s8].
