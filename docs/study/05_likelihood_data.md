# 05 - Likelihoods, data vectors, covariance, scale cuts, m-/c-biases (kcap)

Scope: `utils/mini_2pt_like.py`, `utils/mini_like.py`, `utils/mini_BOSS_like.py`, `utils/shear_c_bias.py`,
`utils/correlated_priors.py`, `modules/scale_cuts/*`, `cosmosis-standard-library/likelihood/2pt/`,
`cosmosis-standard-library/shear/shear_bias/`, and the data under `data/KV450`, `data/KiDS1000`,
`runs/3x2pt/data_iterated_cov/...`. All paths below are relative to
`/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap` ("KCAP/"). Nothing in kcap was executed. Numbers marked
"(computed)" come from reading data files with astropy/numpy or from arithmetic on constants quoted from source;
the micro-benchmarks in section 7 time my own stand-in snippets, not kcap code.

Conventions: `file:line` cites the clone at commit `36b40c5`. "pair" = tomographic bin pair. Angles are arcmin
unless stated; CosmoSIS theory arrays are radians.

Fact-check pass: this file was re-audited line by line against the source and data. Corrections were made in
place; additions are tagged "[verified]" (re-read of source or re-computation on the committed files). The most
consequential corrections are: the K1000 photo-z prior mean (2.7, Q16), the KV450 mean-z values (1.4),
the `twoPoint_KV450.fits` covariance block order (1.4, Q18), the gamma_t claim for Path B (1.1, 5), the NumPy
fragility of `mini_2pt_like.setup` (Q13b, now resolved), and several `file:line` cites.

---------------------------------------------------------------------------------------------------

## 1. Purpose, role, inputs and outputs

### 1.1 Role

This subsystem is the last stage of every kcap CosmoSIS pipeline. It turns theory arrays already in the
datablock into one number (ln L). It owns five jobs:

1. Load the measured data vector and covariance, and cut scales.
2. Reduce the theory to the same layout as the data (binning or interpolation in theta, or band-power
   selection).
3. Apply nuisance corrections that live at likelihood level (m-bias, c-terms).
4. Evaluate a constant-covariance Gaussian chi^2.
5. Write `*_CHI2` and `*_LIKE` into the datablock.

Four independent likelihood code paths exist.

| Path | Used by | Statistic | Cut/binning code |
|---|---|---|---|
| A. `utils/mini_2pt_like.py` | `runs/config/KV450_fiducial.ini`, `KV450_no_sys.ini`, `BOSS_KV450.ini` | xi+/xi- (KV450) | in the same file |
| B. `modules/scale_cuts/scale_cuts.py` then `utils/mini_like.py` | `runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline.ini` (KiDS-1000 band powers EE+nE; the file lives in `config/`) and the MontePython K1K likelihoods; also every config produced by `utils/run_kcap.py` (`2x2pt_like` = `mini_like.py`, `input_section_name = theory_data_covariance`, `run_kcap.py:752-754`) [verified] | band powers (PneE, PeeE), COSEBIs (En), xi+/- (xiP, xiM). **Not gamma_t**: the theory builder only knows these five statistics (`scale_cuts.py:117-123`); a FITS gT extension can be cut on the data side but never gets a theory vector, so `use_stats = gT` would end in an empty `np.concatenate` [verified] | `wrapper_twopoint.py` + `twopoint.py` |
| C. `utils/mini_BOSS_like.py` | the same 3x2pt pipeline | BOSS DR12 xi wedges (two z-bins) | in the same file |
| D. `cosmosis-standard-library/likelihood/2pt/2pt_like.py` (+ `2pt_like_marg.py`) | commented out in the KV450 ini (`KV450_fiducial.ini` lines "[2pt_like]" block); no committed pipeline uses it | generic twopoint FITS | `twopoint.py` in the same dir |

Path D relies on the CosmoSIS core class `GaussianLikelihood`. That core is NOT in the clone (`find` for
`gaussian_likelihood.py` returned nothing), so its exact chi^2/log-det bookkeeping is unverified against the
clone itself. [verified] A CosmoSIS install on this machine (`/mnt/d/CosmoSIS/cosmosis/gaussian_likelihood.py`,
version may differ from what kcap targeted) does: `chi2 = einsum('i,ij,j', d, inv_cov, d)`, `ln L = -chi2/2`,
plus `-0.5 log|C|` only if `include_norm = T` (default F) for a constant covariance, or always for a
non-constant covariance (e.g. `gaussian_covariance = T`). It also writes `_CHI2`, `_N`, `_LOG_DET`, `_NORM`,
`_theory`, `_data`, `_inverse_covariance`, `_covariance` and a `_simulation` (mean + chol(C) r) on every call
unless `likelihood_only = T`. The section names are `data_vector`, `likelihoods`, and
`shear_calibration_parameters` (`cosmosis/datablock/cosmosis_py/section_names.py`).
Also in this directory: `2pt_like_marg.py` (75 lines) subclasses the above and analytically marginalises a
point-mass gamma_t mode: it adds `sigma_a^2 t t^T` with `t_k = theta_k^-2` (theta in radians, only the
`gammat` extension) to C through a Sherman-Morrison update of the inverse,
`Psi -> Psi - (Psi t t^T Psi)/(1 + t^T Psi t)` with `t` pre-multiplied by `sigma_a`. It uses the raw
`covmat` inverse, so any `covariance_realizations` (Hartlap) or Sellentin setting is ignored [verified]. This
is the same linear-nuisance marginalisation proposed as item 6 of section 7.

`utils/shear_c_bias.py` is a fifth, effectively dead module (section 4, Q12).
`utils/correlated_priors.py` is a prior helper for the photo-z shifts (section 2.7).

### 1.2 Datablock contract (inputs read)

**Path A** (`mini_2pt_like.py:153-243`):

| Section | Key | Type | Meaning |
|---|---|---|---|
| `shear_xi_plus` | `nbin_a` | int | number of tomographic bins; overrides the ini `n_bin` for loops (l.156) |
| `shear_xi_plus` | `theta` | 1-D float, **radians** | theory theta grid (1024 nodes from the Nicaea Hankel module) |
| `shear_xi_plus` | `bin_{a}_{b}` | 1-D float, len 1024 | xi+ for pair (a>=b), 1-based, CosmoSIS convention "larger index first" |
| `shear_xi_minus` | `theta`, `bin_{a}_{b}` | same | xi- |
| `shear_c_bias` | `delta_c`, `A_c` | float | read only if `constant_c_offset` / `xi_pm_c_file` set |

Section names `shear_xi_plus/minus` come from `cl_to_xi_nicaea/cl_to_xi_interface.c:41-60,226-227`: default
output section `shear_xi` with `_plus`/`_minus` appended. The array key is `bin_i_j`, written via `name_out`
(l.343-344). The module.yaml in that directory still documents the old `xiplus_i_j` names (stale doc).

**Path B** (`scale_cuts.py:101-198`):

| Statistic | Section (ini-configurable) | angle key | value keys | order read |
|---|---|---|---|---|
| xiP | `shear_xi_plus` (default; K1000 ini uses `shear_xi_plus_binned` / `xi_binned_plus`) | `theta_bin_1_1` (arcmin, written by `cosebis/xipm_binned.cc`) | `bin_{j+1}_{i+1}`, j>=i | (1,1),(1,2)...(1,n),(2,2)... |
| xiM | `shear_xi_minus` | same | same | same |
| PeeE | `bandpower_e_cosmic_shear` (ini: `bandpower_shear_e`) | computed from `l_min_vec`, `l_max_vec` | `bin_{j+1}_{i+1}` | same |
| PneE | `bandpower_ggl` (ini: `bandpower_galaxy_shear`) | same | `bin_{i+1}_{j+1}` (i=lens, j=source) | lens-major |
| En | `cosebis` | `cosebis_n` | `bin_{j+1}_{i+1}` | same |

Band-power module outputs (`cosebis/BandPower_interface.cc:565-577`): `bin_i_j` (vector per band),
`l_min_vec`, `l_max_vec`, `b_modes`, `theta_min`, `theta_max`, `nbin_a`, `nbin_b`.

**Path C**: `xi_wedges / bin_{b}`, an array of shape (n_wedge, n_s) per BOSS z-bin b=1,2 (`mini_BOSS_like.py:71-72`).
The producer (`../kcap_boss_module/python_interface/cosmosis_module.py`) is NOT in the clone.

### 1.3 Datablock contract (outputs written)

| Path | Section / key | Content |
|---|---|---|
| A | `data_vector / {like_name}_CHI2` (float), `likelihoods / {like_name}_LIKE` (= -chi2/2) | l.244-245 |
| A, if `keep_theory_vector=T` | `data_vector / {like_name}_theory, _data, _inverse_covariance, _simulation, _theta_xi_plus, _theta_xi_minus` | l.247-253 |
| B (scale_cuts) | `{output_section_name}` (default `likelihood`; ini uses `theory_data_covariance`) keys `data` (1-D), `covariance` (n x n), `inv_covariance` (n x n), `theory` (1-D) | l.105-108, 198 |
| B (mini_like) | `data_vector / {like_name}_CHI2`, `likelihoods / {like_name}_LIKE` | `mini_like.py:24-25` |
| C | `data_vector / {like_name}_CHI2`, `likelihoods / {like_name}_LIKE`; optional per-bin theory/data/cov/inv_cov/simulation | `mini_BOSS_like.py:75-88` |

The CosmoSIS sampler sums every `likelihoods/*_LIKE` named in the ini `likelihoods =` line. The section name
constants `names.data_vector` and `names.likelihoods` live in the CosmoSIS core (not in the clone).

### 1.4 Data sets present in the clone (computed from the files)

**KV450 xi+/-** (`data/KV450/`)

| Item | Value |
|---|---|
| Tomographic bins | 5. DIR n(z) files `Nz_DIR_z0.1t0.3, 0.3t0.5, 0.5t0.7, 0.7t0.9, 0.9t1.2`. Mean z of the n(z) histograms: 0.369, 0.463, 0.643, 0.806, 0.973 if the first column (lower bin edge) is used as z, but the bin-centre mean z is **0.394, 0.488, 0.668, 0.831, 0.998** (+dz/2 = 0.025), which is what `load_nz` with `histogram = True` uses (`load_nz.py:58-64`: `z += dz/2`) [verified] |
| Pairs | 15 |
| Theta bins | 9 log-spaced bins, edges 0.5 to 300 arcmin (edges listed in `KV450_fiducial.ini:191`) |
| Raw data vector | 15 x 9 x 2 = 270 |
| After fiducial cuts | `cut_xi_plus = 7 8` and `cut_xi_minus = 0 1 2` give 7 xi+ and 6 xi- bins per pair: **195 points** |
| `data_vector/KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat` | text, shape (18, 16). Column 0 = theta (arcmin): 0.78613, 1.5996, 3.2535, 6.623, 13.493, 27.385, 55.541, 112.61, 223.72. These are pair-weighted mean thetas, not bin centres (consistent with the `meanR` column of the c-term file: 0.78628 vs 0.78613 for bin 1, but note the file itself does not say so) [verified]. Rows 0-8 = xi+, rows 9-17 = xi-. Columns 1..15 = the 15 pairs in "montepython" order (0,0),(0,1)..(0,4),(1,1)..(4,4) |
| `covariance/cov_analytic_montepython_mcorr.txt` | text (270,270), exactly symmetric, min eigenvalue 4.7e-14 (positive), first diagonal element (pair (1,1), xi+ theta bin 1) = 1.12e-8. Layout: for each pair in montepython order, [xi+ (9 thetas), xi- (9 thetas)] |
| After cuts | 195 x 195, cond number 2.4e5 (computed) |
| `systematics/KV450_ALL_c12_treecorr.out` | 9 rows x 10 columns, no header line (column names are inferred from the TreeCorr convention; column 8 "weight" is 0 in all rows [verified]), TreeCorr-style: R_nom, meanR, meanlogR, xip, xim, xip_im, xim_im, sigma, weight, npairs. R_nom = 0.7134, 1.4521, 2.9558, 6.0168, 12.247, 24.93, 50.75, 103.3, 210.27 (geometric bin centres). xip(c) (col 3) = 5.38e-7, 4.98e-7, 3.98e-7, 1.92e-7, 2.27e-8, -1.93e-8, 1.07e-8, 4.78e-9, 2.80e-9; xim(c) (col 4) = 1.04e-9, 2.43e-9, 7.22e-9, 9.91e-9, 2.80e-9, -5.23e-9, 2.34e-8, -1.54e-9, -2.63e-10 |
| `nofz/` | 5 x `Nz_DIR_*.asc` (120 rows, 2 columns: z lower bin edge 0.00..5.95 step 0.05, n(z); sum(n)*0.05 = 1 to 1e-7, whereas `np.trapz(n, z)` over the 120 samples is 0.9945, 0.9981, 0.9985, 0.9994, 0.9998, so `load_nz` re-normalises by trapz [verified]). `DIR_cov.asc` (5x5; sqrt diag = 0.0377, 0.0225, 0.0247, 0.0119, 0.0101), `SOM_cov.asc`, `SOM_cov_multiplied.asc` (5x5; sqrt diag = 0.0096, 0.0114, 0.0116, 0.0084, 0.0097; this is exactly 2x the sqrt diag of `SOM_cov.asc` = 0.00478, 0.00568, 0.00578, 0.00421, 0.00485, i.e. covariance x4 [verified]; the K1000-run copy has sqrt diag 0.01058, 0.01131, 0.01175, 0.00869, 0.00975 [verified]), `SOM_cov_multiplied_diagonal.asc`, `id_cov.asc` |
| `combined/COSEBIs_En_nModes20.fits` | En: 300 rows (15 pairs x 20 modes), COVMAT (300,300), NZ_SOURCE 120 rows, NZ_LENS 2000 rows |

**Test twopoint FITS** (`modules/scale_cuts/test_files/`)

| File | Content |
|---|---|
| `twoPoint_KV450.fits` | COVMAT (270,270) ordered [xiP(135) | xiM(135)]. xiP, xiM: 135 rows = 15 pairs x 9 theta. ANG = nominal geometric bin centres (0.71336, 1.4521, 2.9558, 6.0168, 12.247, 24.930, 50.747, 103.30, 210.27 arcmin), unit arcmin. QUANT1/2 = `G+R` / `G-R`. NZ_SOURCE 119 rows |
| `twoPoint_PneE+PeeE.fits` | COVMAT (200,200); PneE 80 rows (2 lens x 5 source x 8), PeeE 120 rows (15 x 8); ANG 118.44..1266.44 |
| `twoPoint_gT+xiP+xiM.fits` | COVMAT (360,360); gT 90 (2 x 5 x 9), xiP 135, xiM 135 |
| `twoPoint_fakeCTerm.fits` | extensions `2d_cterm`, `cterm_cos`, `cterm_sin`, 135 rows each |

In `twoPoint_KV450.fits` the xiP and xiM VALUE columns equal the text data file exactly for ALL 15 pairs and
9 thetas (max relative difference 0.0, computed) [verified]. The COVMAT is a different story. Only the first
two pair blocks, (1,1) and (1,2), agree with the text covariance (diagonal to better than 4%; the first
diagonal element 1.1197e-8 agrees to 6 digits, the 2nd xi+ bin is 2.90563e-9 vs 2.90548e-9, so the two are
slightly different versions). For the other 13 pairs the diagonal of the text covariance in natural
"montepython" pair order differs from the FITS COVMAT by factors 0.3 to 3.0 (identity ordering: median
|corr difference| 0.004 but max 0.52, median diagonal mismatch 37%). A linear-assignment match of the pair
blocks (cost = correlation difference + log-variance difference) recovers the text covariance to within 8% on
every diagonal element and 0.048 in correlation (median 0.9% and 0.0025) if the text-covariance pair index p is
mapped to the FITS pair (BIN1,BIN2) as p0:(1,1) p1:(1,2) p2:(1,4) p3:(2,3) p4:(3,4) p5:(1,3) p6:(1,5) p7:(2,4)
p8:(3,5) p9:(2,2) p10:(2,5) p11:(4,4) p12:(3,3) p13:(4,5) p14:(5,5) (p0-p9 are unambiguous; p10-p14 less so).
A shot-noise consistency test decides which file is internally consistent: the smallest-theta xi+ variance should
follow ln var = a_i + a_j (+ln 2 for auto pairs). With the natural labels the TEXT covariance fits with rms
residual 0.058 and gives a_i that track the known n_eff pattern (bin 3 least noisy, bin 1 most noisy), but the
FITS COVMAT with its own BIN1/BIN2 labels fits with rms 0.44. Conclusion: the committed
`twoPoint_KV450.fits` COVMAT has its pair blocks in a different order from its xiP/xiM labels and must not be
used as a reference without re-ordering; the text covariance with `order_cov = montepython` is the
self-consistent one [verified, computed]. `TwoPointFile.get_cov_start` re-orders covariance blocks only at the
level of whole extensions (via `NAME_i`/`STRT_i` header cards), never pairs inside an extension.

**KiDS-1000 band-power data used by the committed 3x2pt run**
(`runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/data/KiDS/bp_KIDS1000_BlindC_with_m_bias_..._SOM_Fid.fits`,
computed with astropy)

| Extension | Content |
|---|---|
| `COVMAT` | (200,200), STRT_0=0 NAME_0=PneE, STRT_1=80 NAME_1=PeeE. Exactly symmetric, min eigenvalue 2.8e-11, cond 4.6e5. Max abs correlation between the PneE and PeeE blocks = 0.21 |
| `PneE` | 80 rows. Columns BIN1 (lens 1-2), BIN2 (source 1-5), ANGBIN (1-8), VALUE, ANG [arcmin, TUNIT5]. QUANT1/2 = GPF/GEF, KERNEL_1/2 = NZ_LENS/NZ_SOURCE, WINDOWS=SAMPLE, N_ZBIN_1=2, N_ZBIN_2=5, N_ANG=8. Pair order (1,1)..(1,5),(2,1)..(2,5), each with 8 bands |
| `PeeE` | 120 rows, 15 pairs (1,1),(1,2)..(1,5),(2,2)..(5,5) x 8 bands, GEF/GEF |
| `ANG` (both) | (header `TUNIT5 = arcmin`, but these are multipoles ell, not angles: the unit label is wrong; likewise in `COSEBIs_En_nModes20.fits` ANG is the mode index 1..20 with `TUNIT5 = arcmin`. Nothing in kcap reads the unit for these statistics: `kernelTypeDict` gives unit None, `wrapper_twopoint.py:86-91` [verified]) 118.442, 166.156, 233.093, 326.994, 458.724, 643.521, 902.764, 1266.443. These equal sqrt(l_lo * l_hi) for 8 log-spaced bands in [100,1500] (edges 100, 140.285, 196.799, 276.080, 387.298, 543.322, 762.199, 1069.251, 1500) |
| `NZ_SOURCE` | 119 rows, Z_LOW=0, Z_MID=0.025.., Z_HIGH=..5.95 (dz=0.05), BIN1..BIN5; header SIG_E_1..5 = 0.2696, 0.2579, 0.2726, 0.2539, 0.2703 and NGAL_1..5 = 0.5527, 1.0601, 1.6626, 1.1292, 1.1755 (per arcmin^2). Mean z = 0.257, 0.403, 0.564, 0.792, 0.984 |
| `NZ_LENS` | 79 rows, dz=0.01, Z_HIGH max 0.79, NGAL_1,2 = 0.00663, 0.00775, mean z 0.385, 0.586 |

The file name contains "with_m_bias" and "BlindC". It holds no xiP/xiM/En extensions. There is no m-bias or
c-term object inside the FITS beyond what is baked into VALUE/COVMAT.

**Other KiDS-1000 material in the clone**

- `data/KiDS1000/Flinc/` holds Flinc mock inputs only:
  - xi+/- data and theory vectors, 195 values each (1-D text, already scale-cut).
  - Four 195 x 195 covariances (`cov_{complexMask,complexTheory,simpleMask,theory2}_GG_obs_cut.ascii`).
  - `nofz/nofz.fits` (extension `NZ_flinc`, 2001 rows, Z_lOW/Z_HIGH/Z_MID, BIN1..5).
- `data/KiDS1000/mocks/3x2pt_sample_logA_parameter_covariance.txt` is a (20,20) parameter covariance.
- The real KiDS-1000 xi+/- and COSEBI data are NOT in the clone. `data/README.md` is a one-line GitHub URL
  (`.../kcap/tree/master/runs/3x2pt/data/cosmology/multinest_blindA_EE_nE_w_Planck`); that directory does not
  exist in the clone (only `runs/3x2pt/data_iterated_cov/` does). `data/` also holds `BOSS/nofz` and
  `Planck/COM_Likelihood_Data-baseline_R3.00` [verified].
- The K1000 config generators `runs/3x2pt/3x2pt_configs.py` point at `.../data/kids/fits_iterative_covariance/bp_KIDS1000_Blind{blind}_with_m_bias_...fits`, which supports (does not prove) that the COVMAT of the committed FITS is the "iterative" covariance of the K1000 papers [verified].

**BOSS** (`runs/3x2pt/data_iterated_cov/.../data/BOSS/`)

| Item | Value |
|---|---|
| `BOSS.DR12.{lowz,highz}.3xiwedges_measurements.txt` | (32, 7): s (2.5, 7.5 ... 157.5 Mpc/h in steps of 5), then (xi_w, sigma_w) for 3 wedges. 0.2<z<0.5 (lowz) and 0.4<z<0.6 (highz) (Sanchez+16 pre-reconstruction, per file headers; s in Mpc/h, xi dimensionless) |
| `..._covmat.txt` | (96,96) = 3 wedges x 32, symmetric, min eigenvalue 5.6e-8 (lowz), 8.4e-8 (highz). Wedge-major layout confirmed: sqrt(cov[32,32]) = 0.034564 = sigma of wedge 2 at s = 2.5, sqrt(cov[64,64]) = 0.04800 = sigma of wedge 3 [verified] |
| After `points_range = 4 32` | 28 points per wedge (s = 22.5 ... 157.5 Mpc/h, consistent with `bands_range = 20 160` of the wedge module), 84 per z-bin, 168 total |

---------------------------------------------------------------------------------------------------

## 2. Mathematical description

Notation: d = data vector, mu = model vector, C = covariance, Psi = C^-1, r = d - mu.

### 2.1 Gaussian likelihood (all paths)

    chi2 = r^T Psi r,    ln L = -chi2 / 2.

- No -1/2 ln|C| term is added. The covariance is constant (fixed, "iterated" or analytic), so it is dropped.
- Path A: `mini_2pt_like.py:238-245`. Path B: `mini_like.py:19-25`. Path C: `mini_BOSS_like.py:75-88`
  (chi2 summed over the two BOSS bins, so ln L = -(chi2_lowz + chi2_highz)/2, assuming no cross-covariance
  between the two BOSS bins; the same applies between BOSS and KiDS, whose `_LIKE` values are simply added by
  the sampler).
- Psi is computed ONCE in `setup` as `np.linalg.inv` of the already-cut covariance (A: l.99; B:
  `scale_cuts.py:89`; C: l.51). The inverse of the sliced matrix is used (not a slice of the inverse), which is
  the correct operation.

Prior terms are NOT part of the likelihood in the CosmoSIS paths. They are applied by the sampler from
`priors.ini`. The MontePython path (section 2.9) instead adds them to chi2 inside the likelihood.

### 2.2 Path A model vector (KV450 xi+/-)

For each unordered tomographic pair (a,b) and each of "+"/"-", with k indexing kept theta bins:

    mu^{+}_{ab}[k] = B_k[xi^{+}_{ab}] + [delta_c or delta_c^2] + A_c^2 * c^{+}[k]
    mu^{-}_{ab}[k] = B_k[xi^{-}_{ab}]                          + A_c^2 * c^{-}[k]

(Note the delta_c term appears for "+" only; the "-" line is commented out, l.188 and l.227.)

Data side, applied once in `setup`:

    d^{+/-}_{ab}[k] = x^{+/-}_{ab}[k] / ((1 + m_a)(1 + m_b)),  with m = `m_correction` (length n_bin)

(`mini_2pt_like.py:140-141`). The covariance is NOT rescaled by the same factor.

**Binning operator B_k** (`get_theory_point`, l.11-33). `theta_n` = theory grid in rad, `y_n` = xi_n.

(i) `angular_binning_mode = integrate` (used by all KV450 inis, e.g. `KV450_fiducial.ini:189`). Bin edges
`angular_bin_edges` (arcmin, converted to rad by `*= pi/180/60`, l.63). For bin k with edges
[e_k, e_{k+1}), take the nodes S_k = { n : e_k <= theta_n < e_{k+1} } (lower inclusive, upper exclusive) and
compute

    B_k[xi] = T[ theta * xi ; S_k ] / T[ theta ; S_k ]

where T[f; S] is the composite trapezoid rule over the consecutive nodes of S (numpy `trapz`). Equivalently
B_k[xi] ~ ∫ theta xi dtheta / ∫ theta dtheta, with weight w(theta) = theta, but:

- only nodes inside the bin are used;
- there is no interpolation to the bin edges;
- there is no pair-count weighting.

The `weighting` argument passed is the theta array itself (l.177, 183), so `norm = trapz(theta[mask], theta[mask])`.

(ii) `interpolate` (the default if the option is absent, l.60). Build the interpolating cubic spline
`InterpolatedUnivariateSpline(ln theta, xi)` (k=3, s=0, no smoothing) and evaluate it at the data thetas
`theta_data` = column 0 of the data file (xi+: first 9 rows, xi-: last 9 rows), arcmin -> rad (l.45-46).

(iii) Any other string, e.g. `none` as in the Flinc inis, raises `ValueError` (l.27-28).

After either mode, `np.delete(result, cut_bins)` removes the cut theta bins (l.30-31).

**c-term pieces** (l.186-191 for `xi_pm-bin-theta` order; l.225-230 for `montepython` order)

- `constant_c_offset = True` (default True, l.101): for xi+ add `delta_c` if `order_cov = xi_pm-bin-theta`, but
  add `delta_c**2` if `order_cov = montepython`. See QUIRK Q4.
- If `xi_pm_c_file` is given: `xi_p_c, xi_m_c = loadtxt(file, usecols=[3,4])` (0-based columns 3 and 4 =
  xip and xim of the TreeCorr output), cut with the same theta indices (l.103-106), and added as
  `A_c^2 * xi_p_c` (to xi+) and `A_c^2 * xi_m_c` (to xi-). The SAME 9-vector is added to every tomographic
  pair, including cross-pairs.

**Chi2 assembly**

- `order_cov = xi_pm-bin-theta` (l.158-199): data and model are concatenated as
  [all xi+ pairs | all xi- pairs]. Pairs run (i=0..n-1, j=0..i), i.e. (0,0),(1,0),(1,1),(2,0),... and theory
  is read from `bin_{i+1}_{j+1}`. Within a pair the cut theta bins are in increasing order.
- `order_cov = montepython` (l.200-236): vector = for each pair (i=0..n-1, j=i..n-1) in that loop order,
  [xi+ (kept thetas), xi- (kept thetas)]. Data come from `data_vectors[j][i]`, theory from `bin_{j+1}_{i+1}`.

**Covariance cut indices** (l.72-97), with P = n_bin(n_bin+1)/2 and n_theta = 9:

- `xi_pm-bin-theta`: xi+ index = pair*n_theta + cut; xi- index = P*n_theta + pair*n_theta + cut.
- `montepython`: xi+ index = pair*2*n_theta + cut; xi- index = pair*2*n_theta + n_theta + cut.

Rows and columns with those indices are deleted before inversion.

[verified] `order_data` only selects which file column feeds `data_vectors[i][j]` (l.122-131); the order of
the assembled chi2 vector is set by `order_cov` alone (`execute` branches on `order_cov`). Because the
data are stored per pair `(i,j)`, `order_data != order_cov` is legal and harmless: each pair is read with the
data-file column map and then placed in the covariance order. `data_vectors[i][j]` are NumPy views into the loaded array and are
scaled in place (l.140-141), which is safe here because each (i,j) maps to a unique column.

**Data file column to pair maps** (l.122-131). For a pair (i>=j), column = 1 + bin_idx with:

- `xi_pm-bin-theta` (row-major lower triangle): bin_idx = i(i+1)/2 + j.
- `montepython` (row-major upper triangle, symmetric): bin_idx = j(2n-j+1)/2 + (i-j).

I checked the second map against the FITS (`twoPoint_KV450.fits`): xi+ and xi- of all 15 pairs agree to all digits (see 1.4 for the FITS covariance, which does NOT agree).

### 2.3 Path B model vector (scale_cuts + mini_like)

Let the FITS extensions (in file order) be E_1..E_m. Each spectrum has arrays BIN1, BIN2, ANGBIN, VALUE, ANG.

Setup (`scale_cuts.py:57-89`):

1. Load all spectra and `COVMAT` (`TwoPointFile.from_fits(..., covmat_name='COVMAT')`,
   `twopoint.py:827-870`).
2. Apply scale cuts to spectra and covariance together (section 2.4).
3. Keep only extensions named in `use_stats` (`choose_data_sets`, which slices `covmat` but NOT
   `covmat_info`).
4. d = concatenate(VALUE of the kept spectra in FILE order). C = covmat. Psi = inv(C).

Execute (every likelihood call, l.101-198):

1. Build the theory spectra for every statistic in `use_stats`, in the FIXED order xiP, xiM, PneE, PeeE, En
   (`sectionNameList`, l.117-123).
2. Build the same (BIN1, BIN2, ANGBIN, ANG) layout as the data would have, with ANG taken from the datablock:
   `theta_bin_1_1` for xi, `10^{(log10 l_min + log10 l_max)/2}` per band for band powers (l.139-146), `cosebis_n`
   for COSEBIs.
3. Apply the SAME scale-cut operations, driven by the THEORY ANG values (l.191-195).
4. `theory = concatenate(VALUE)`.

There is no re-binning or interpolation here. The upstream C++ modules (`xipm_binned.cc`, `BandPower_interface.cc`)
already deliver the vectors at the data's angular samples. The only coupling to the data is by position.

The theory for a bandpower is the band-averaged quantity from `libbandpower.so`, not C_l at the band centre.
The band-centre ell is used only for the cut.

**xi binning in the C++ module** (`cosebis/xipm_binned.cc:704-745`), used when scale_cuts consumes xi:

- ln theta table in arcmin, GSL **natural cubic spline** (`gsl_interp_cspline`,
  `modules/function_cosebis.cc:775`), `extrapolationOff()` (so zero outside the table, l.712).
- Default weighted binning (`weight_theta_bins_by_theta`): for each bin [t_lo, t_hi] take 100 log-spaced
  sub-nodes

      theta_s = exp( ln t_lo + ln(t_hi/t_lo) * (s + 0.5)/100 ),
      dtheta_s = theta_{s+1/2 edge} - theta_{s-1/2 edge},
      B_k = Σ_s xi(theta_s) * theta_s * dtheta_s / Σ_s theta_s * dtheta_s.

  This is a midpoint-rule version of ∫ theta xi dtheta / ∫ theta dtheta over the true bin edges.
- Optional pair-count weighting (`InputNpair` files, weight = N_pair read per tomo pair), or no weighting
  (evaluate at the bin centre `theta = exp(ln t_min + ln(t_max/t_min)*(i+0.5)/nTheta)`).
- c-terms inside this module (only if `add_c_term`/`add_2D_cterm`):
  - plus: `delta_c = c1^2 + c2^2` if `shear_c_bias/c1` and `c2` exist, else `shear_c_bias/delta_c`
    (l.652), then added LINEARLY to xi+ (`xi_binned += xi_plus_c1c2`, l.772). So in the C++ module
    `shear_c_bias/delta_c` is a variance-like offset (c^2), whereas in `mini_2pt_like` (montepython order) it is
    squared again [verified]; the same parameter name has two different meanings (see Q4). The minus case
    requires both `c1` and `c2` (l.664-677) and reads cos4phi/sin4phi arrays.
  - minus: `(c1^2 - c2^2) cos4phi + 2 c1 c2 sin4phi` from input arrays (l.811).
  - 2-D term: `+ Ac * xi_2D` (linear in `Ac`, l.851; the source comment asks "Should this be Ac or Ac^2").
  - The KV450 `mini_2pt_like` instead uses `A_c^2`. See Q7.

### 2.4 Scale-cut algebra (`wrapper_twopoint.py:107-160, 238-255`, `twopoint.py:663-744`)

Options (inline in the module section, or in a named section of `scale_cuts.ini`):

| Option | Parsed as | Effect |
|---|---|---|
| `use_stats = xiP xiM ...` | list | statistics kept (case-sensitive tags `xiP xiM PneE PeeE En`; after ini lower-casing the code compares lower-cased keys) |
| `cut_cross = ...` | `bool(value)` | `mask_cross`: keep only entries with BIN1==BIN2 (see Q1) |
| `cut_pair_<stat> = a+b c+d ...` | pairs (BIN1+BIN2) | drop the whole pair (`mask_scales(bin_cuts=...)`) |
| `cut_bin_<stat> = i j k` | 0-based indices into that spectrum's rows | `mask_indices` |
| `keep_ang_<stat> = lo hi` | floats | `mask_scale`: keep `lo < ANG < hi` (**strict**) |
| `keep_ang_<stat>_<a>_<b> = lo hi` | floats | `mask_scales(cuts)`: per-pair; drops `ANG < lo` or `ANG > hi`, i.e. keeps `lo <= ANG <= hi` (**inclusive**) |
| `keep_ang_all = lo hi` | floats | like `keep_ang_<stat>` but for every spectrum |

Order of application (identically on data and theory): `cutCross`, then `cut_pair`, then `cut_bin`
(`cutScales`), then per-pair `keep_ang_<stat>_a_b`, then per-stat `keep_ang_<stat>` (`keepScales`). The cuts
intersect. Every mask is also applied to the covariance rows and columns (`_mask_covmat`, `twopoint.py:628-634`).

The cut is on the ANG value of the data point (bin centre for xi, band geometric centre for band powers), not
on bin edges.

[verified] Parsing details (`wrapper_twopoint.py:120-156`): the option name is split on `_`; `keep_ang` keys with
more than 4 tokens are per-pair (tokens 3,4,5 = stat, BIN1, BIN2; indices are 1-based BIN values) and keys with
fewer than 4 tokens are per-stat; a key with exactly 4 tokens (e.g. `keep_ang_pnee_1`) matches neither branch and is
silently ignored. `cut_pair_*` pairs are `a+b` with 1-based bin numbers. `cut_bin_*` indices are 0-based rows of
the spectrum AFTER `cut_cross` and `cut_pair` have already removed rows (the same order is applied to the theory,
so it is consistent but not the row numbers of the FITS file). `mask_indices` prints `len`, the index array and the
whole boolean mask for every spectrum on every call (`twopoint.py:663-677`). With the inline-option route the
`use_/cut_/keep_` keys are selected by the first underscore-token (`scale_cuts.py:53`), values are stringified
and stripped of ` []` (l.54). The documented example in `scale_cuts.ini` (`cut_cross = False ;; Trivial`) therefore
turns cross-cut ON (Q1). Stat names are matched case-insensitively (`labDict[keySplit[2]]` with lower-cased keys).

### 2.5 Worked K1000 example: committed 3x2pt run (computed)

Options in `config/pipeline.ini` [scale_cuts] (the ini also carries `keep_ang_xip/xim = 0.5 300` and the
xi/cosebis section names, which are inert because `use_stats` omits xiP/xiM/En; I re-derived the 22/120/142
counts below by applying the cuts to the FITS ANG columns, and also the cond 3.3e4 and min eigenvalue 2.8e-11
[verified]): `use_stats = PneE PeeE`; `cut_pair_pnee = 1+1 2+1 2+2 2+3`;
`keep_ang_peee = 100 1500`; `keep_ang_pnee = 100 1070`; `keep_ang_pnee_1_b = 100 300` for source b=1..5
(lens 1); `keep_ang_pnee_2_b = 100 600` (lens 2).

| Spectrum | Kept |
|---|---|
| PeeE | all 120 (ANG 118..1266 lies inside (100,1500)) |
| PneE | pairs (1,2),(1,3),(1,4),(1,5): bands 1-3 (ANG <= 300): 12 points. Pairs (2,4),(2,5): bands 1-5 (ANG <= 600): 10 points. Total 22 (pair (1,1) and (2,1),(2,2),(2,3) fully cut) |
| Total | **142** points, covariance 142 x 142, cond 3.3e4, min eigenvalue 2.8e-11 |

The BOSS likelihood adds 168 points (2 x 84). [verified] The scale-cut presets used by `utils/run_kcap.py`
(`run_kcap.py:1156-1171`) are: mocks/no-cut mode `ell_range_EE = ell_range_nE = [100,1500]`, `theta_range_xiP/M =
[0.5,300]`, `cut_bin_nE = []`; data mode `cut_bin_nE = 1+1 2+1 2+2 2+3`, `ell_range_EE = [100,1500]`,
`ell_range_nE = [100,1070]` (drops the last nE band), plus the per-pair 100-300 / 100-600 limits set by
`runs/3x2pt/3x2pt_configs.py:35-44`. The committed K1000 run has `simulate = F`.

KV450-like cuts through scale_cuts (`scale_cuts.ini [KV450_like]`: `keep_ang_xiP = 0.5 75`,
`keep_ang_xiM = 4 300`) applied to the nominal centres of the test FITS keep xi+ bins 1..7 and xi- bins 4..9,
i.e. the same 13 bins per pair, 195 points, as the `cut_xi_plus = 7 8`, `cut_xi_minus = 0 1 2` of Path A.

### 2.6 Multiplicative bias (m) and the shear-bias module

(a) Likelihood-level treatment actually used in kcap fiducial runs:

- KV450: deterministic division of the DATA by (1+m_a)(1+m_b) with fixed m (section 2.2). No m nuisance
  parameter, no marginalisation, no covariance change.
- KiDS-1000: nothing in the code. The FITS file name says "with_m_bias", so the correction (and any
  uncertainty contribution to the covariance) is baked in upstream; not verifiable here.

(b) `cosmosis-standard-library/shear/shear_bias/shear_m_bias.py` (the generic CosmoSIS module; not in any kcap
`modules =` list. It appears only as an unused ini section in `runs/config/Flinc/xipm_Flinc_complextheory_no_noise.ini:223`,
but it IS used by the vendored library demos/tests: `demos/demo17.ini:45`, `likelihood/2pt/test_cmb/*.ini`,
`structure/projection/test/params_y1.ini` [verified]). The K1000 band-power run likewise has no m module or m
parameter: its `values.ini` carries `[shear_c_bias] c1, c2, ac` that no module in that pipeline reads [verified]:

    C^{ab}_l  ->  (1 + m_a)(1 + m_b) C^{ab}_l          (shear-shear)
    C^{ab}_l  ->  (1 + m_b) C^{ab}_l                   (position-shear; m only for the shear index)
    C^{a,kappa_CMB}_l -> (1 + m_a) C_l                 (shear-CMB kappa)

It multiplies `bin_i_j` in place in the sections `shear_cl`, `galaxy_shear_cl`, `shear_cmbkappa_cl` using
`shear_calibration_parameters/m{i}` (or `m0` if `m_per_bin=F`). It is a model-side correction (theory times
(1+m)(1+m)), the opposite sense of kcap's data-side division. The two are NOT equivalent in chi2: dividing
the data leaves the covariance unscaled; multiplying the theory is the form used by the CosmoSIS library module
and the more common choice. [verified] Precisely, with F = diag(f_p), f_p = (1+m_a)(1+m_b) per pair block,
kcap's chi2 is `(F^-1 d - mu)^T C^-1 (F^-1 d - mu) = (d - F mu)^T (F C F)^-1 (d - F mu)`, i.e. the model-side
chi2 with an effective covariance `F C F` (block pair p scaled by f_p^2 on the diagonal blocks, f_p f_q across
blocks). Relative to the model-side form the errors of pair p are inflated by f_p, which is |f_p - 1| = 0.33-2.5%
for the KV450 m values (computed), so the difference between the two forms is small compared with the 1-2% m uncertainty
that is ignored altogether.

### 2.7 Photo-z shift treatment (nuisance, applied upstream of the likelihood)

`cosmosis-standard-library/number_density/photoz_bias/photoz_bias.py:33-68` (kcap-modified copy that also
outputs the true mean shift `delta_z_out/bin_i = mean_z(shifted) - mean_z(input)`, with `np.average(z, weights=n)`
[verified]; the module also has a `multiplicative` mode `n(z(1-b))` that kcap does not use):

    n_i'(z) = n_i(z - b_i) / ∫ n_i(z - b_i) dz,   (mode = additive, interp1d on the n(z) grid, zero fill outside)

with `b_i = nofz_shifts/bias_i`, `interpolation = linear` (KV450) or `cubic` (K1000), and the normalisation
done with `np.trapz`. Positive b_i moves the distribution to higher z.

- **KV450**: bias_1..5 have independent Gaussian priors N(0, sigma_i) with sigma = (0.039, 0.023, 0.026,
  0.012, 0.011) (`KV450_priors.ini:9-13`; the file has only 13 lines). The correlation in `DIR_cov.asc` is ignored. Note
  sqrt(diag(DIR_cov)) is (0.0377, 0.0225, 0.0247, 0.0119, 0.0101), slightly different from those sigmas.
- **K1000** (`utils/correlated_priors.py:6-28`): sampled p_1..5, each with prior N(mu_i, 1)
  (`priors.ini`), mapped to bias = L p with L = `np.linalg.cholesky(SOM_cov_multiplied.asc)` (lower
  triangular) read from the K1000 run directory (`.../data/KiDS/SOM_cov_multiplied.asc`). With p ~ N(mu, I) this
  gives bias ~ N(L mu, C) with C = LL^T. For the committed mu =
  (0, -0.18125, -1.11014, -1.39501, 1.26543) the prior mean of the shifts is, with the K1000-directory matrix,
  **L mu = (0.000, -0.002, -0.013, -0.011, +0.006)** to 6 decimals, i.e. mu was constructed to reproduce these
  round shift values (computed, [verified]). The earlier numbers (0, -0.00203, -0.01280, -0.01080, +0.00718)
  come from the DIFFERENT `data/KV450/nofz/SOM_cov_multiplied.asc` and are not what the K1000 pipeline uses.
  The shifts enter as additive `b_i`. The `values.ini` ranges for p are [-5, 5] with start values at mu.
- The `SOM_cov_multiplied.asc` in the K1000 run directory differs numerically from the one in
  `data/KV450/nofz/` (np.allclose = False; max abs entry difference 2.1e-5), although both are 5x5.

### 2.8 Covariance and its corrections

| Source | Estimator | Correction in the kcap code |
|---|---|---|
| KV450 `cov_analytic_montepython_mcorr.txt` | analytic (name says so) | none |
| K1000 FITS COVMAT | composition undocumented in the clone (the run directory is called `data_iterated_cov`, i.e. an iterated covariance; the MontePython data file name mentions "theoryVultur") | none in `mini_like.py` |
| BOSS wedge covariances | mock-based | Hartlap factor applied (below) |
| Generic `2pt_like.py` | if `covariance_realizations = r > 0` and not `sellentin` | **Anderson-Hartlap**: C -> C (r-1)/(r-p-2), p = covariance dimension (`2pt_like.py:240-251`) |
| `2pt_like.py` with `sellentin = T` | Sellentin-Heavens (2016, arXiv:1511.05969) | ln L = -1/2 ln|C| (only if the covariance is non-constant) - (N/2) ln(1 + chi2/(N-1)), N = `covariance_realizations` (l.323-337); `covariance_realizations = 0` falls back to Gaussian; incompatible with MOPED |
| `mini_BOSS_like.py` | Hartlap on the INVERSE | Psi -> (nu - p - 1)/nu * Psi, nu = n_mocks - 1, p = dimension of the cut covariance (l.53-57). Algebraically (N-p-2)/(N-1) with N = n_mocks, the same as the Hartlap factor (here (2047-84-1)/2047 = 1962/2047 = **0.95848** per bin (computed)). Default `n_mocks = 2048` (`config/pipeline.ini` sets 2048). The Hartlap factor is applied after the cut (p = 84, not 96) [verified] |
| `xipm_like.py` (legacy CFHTLenS, `cosmosis-standard-library/likelihood/shear_xi`) | Hartlap on the inverse | alpha = (n - p - 2)/(n - 1) (l.38-42) |

For analytic covariances (KV450, KiDS-1000) there is no finite-simulation correction anywhere in the kcap code.
This is consistent with an analytic matrix.

Noise model in the generic `2pt_like.py` Gaussian covariance mode (`gaussian_covariance = T`, not used by the
committed pipelines): add shot noise to the theory auto-spectra in `_lookup_theory_cl` (l.526-541) as
`sigma_e^2 / n_sr` for shear E auto bins and `1/n_sr` for position bins, with

    n_sr = n_arcmin^-2 * (41253 * 60 * 60) / (4 pi)       (`convert_nz_steradian`, l.22-23)

A sky area `survey_area` in deg^2 is converted to steradians as `area * pi^2/180^2`.

### 2.9 MontePython K1K likelihoods (`montepython/K1K_BandPowers`, `K1K_COSEBIs`)

Superseded by an external repository (`montepython/README.md`), but a working likelihood that reuses
`scale_cuts.py`:

- Calls the kcap `scale_cuts` module in `setup` to get `data`, `covariance` (`__init__.py:122-127`), then
  forms `L = cholesky(C, lower=True)`.
- chi2 = |L^-1 (d - mu)|^2 via `solve_triangular` (no inverse; l.250-253). NaN/inf in the difference gives
  chi2 = 2e12.
- Adds the nuisance priors inside the likelihood: chi2 += (D_z,i - center_i)^2 / sigma_i^2 with the scale
  factor of the parameter (l.256-261).
- The file `K1K_BandPowers.data` uses `keep_ang_PeeE = '99.5 1500.5'`, 8 bands in [100,1500], tophat response,
  analytic=True, apodise (delta_x 0.5), theta in [0.25, 397], nzmax 500, `slinear` n(z) interpolation, sigma =
  (0.039, 0.023, 0.026, 0.012, 0.011) for D_z. [verified] Further settings in the same file: C_l precision `ell_min = 1`, `ell_max = 2.5e5`,
  `nells = 200` (versus `ell_min = 1`, `ell_max = 3e4`, `n_ell = 400` in the CosmoSIS K1000 pipeline), non-linear
  `hmcode` with `k_max = 500 h/Mpc`, `use_linear_pk_for_IA = False`, `filename_corrmat_D_z = DIR_cov.asc`, whose Cholesky
  factor correlates the sampled D_z (`__init__.py:152-160`, `453-465`; falls back to identity if the file cannot be
  loaded), while the Gaussian prior term in `loglkl` (l.256-261) is on the raw sampled parameters with independent
  centre 0 and the quoted sigma. This differs from the CosmoSIS K1000 run (Cholesky of `SOM_cov_multiplied`, p
  with N(mu, 1) priors, nonzero mean). The two prescriptions are not the same model for the shifts.

### 2.10 Correlated-prior transform (`utils/correlated_priors.py`)

    q = L p,  L = chol(cov)   (lower),   q_i written to the configured (section, name) i

Input names are `section/name` tokens; `uncorrelated_parameters` and `output_parameters` must be in the same
order as the matrix rows. Cholesky is computed once at `setup` (l.14).

---------------------------------------------------------------------------------------------------

## 3. Numerical choices (with source citations)

### 3.1 Theta grid delivered to Path A (not owned by this subsystem but required to reproduce numbers)

From `cosmosis-standard-library/shear/cl_to_xi_nicaea/maths.h:34-38` and `maths.c:1902-2002`,
`cl_to_xi_interface.c:361-372`:

| Constant | Value |
|---|---|
| `l_min`, `l_max` (Hankel range) | 1e-4, 5e6 |
| `N_thetaH` | 1024 |
| dlnl | ln(5e10)/1023 = 0.0240814 (computed) |
| nc | N/2 + 1 = 513 |
| ln theta_min | (nc - N + 1) dlnl - 0.5 (ln lmax + ln lmin) = ln(2.0736e-7 rad) |
| ln theta_max | nc dlnl - 0.5 (...) = ln(1.0368e4 rad) |
| theta_n | exp(ln theta_min + n * dlnl), n = 0..1023, in radians (metadata unit "radians") |
| Hankel kernels | order mu=0 for xi+, mu=4 for xi-; norm 2 pi (`maths.c:1920-1925, 1965`) |

**Nodes per KV450 integration bin** (computed with the edges at `KV450_fiducial.ini:191`):
[29, 30, 29, 30, 29, 30, 29, 30, 29]; 265 nodes in [0.5, 300) arcmin. So B_k averages roughly 30 nodes,
spacing dln theta = 0.024 against a bin width 0.711 in ln theta.

C_l input: `projection` with `ell_min=0.1, ell_max=5e5, n_ell=400` (KV450) or `ell_min=1, ell_max=3e4, n_ell=400`
(K1000 BP). The Hankel module interpolates C_l in ln l (log-log when all C_l have the same sign, linear
otherwise) before FFT (`cl_to_xi_interface.c:100-118, 270-338`). [verified] Outside the input ell range the
all-same-sign table is extrapolated linearly in (ln l, ln|C|) with slope +1 below `ell_min` and -3 above
`ell_max` (`init_interTable(..., 1.0, -3.0)`), and the mixed-sign table is extrapolated as a constant
(slopes 0). A macro typo `#define EPS 1.0-5` (`maths.c:1617`, evaluates to -4) means the "slope is zero, raise
an error" branch of `sm2_interpol` can never fire, so out-of-range queries never error. The Hankel transform is
`xi_{+/-}(theta) = (1/2pi) Int l dl C_l J_{0/4}(l theta)` (`lP = l C_l`, `norm = 2 pi`; no (2l+1) or 4 pi, C_l
is the continuous-l convolution-theorem form), implemented with `fftw` over the 1024 log-spaced nodes. The Hankel module doc says the xi output is
numerically valid only between about 1 and a few hundred arcmin (`cl_to_xi_nicaea/module.yaml`); the KV450
cut range reaches 0.5 arcmin for xi+ bins that are kept and 300 arcmin edge for xi-, so parts of those bins
fall outside the validity window stated in the module's own documentation (see Q9).

### 3.2 Path A options in the fiducial KV450 ini (`runs/config/KV450_fiducial.ini:185-201`)

| Option | Value |
|---|---|
| `n_bin` | 5 |
| `data_filename` | `data/KV450/data_vector/KV450_reweight_3x4x4_v2_good_xipm_mcor_5bin.dat` |
| `angular_binning_mode` | `integrate` |
| `angular_bin_edges` | 0.5 1.01777898 2.0717481 4.21716333 8.58428037 17.4738002 35.56893304 72.40262468 147.37973879 300.0 (these equal exp(linspace(ln 0.5, ln 300, 10)), verified) |
| `cut_xi_plus` | 7 8 (0-based theta indices; bins 8 and 9 of 9) |
| `cut_xi_minus` | 0 1 2 |
| `order_cov`, `order_data` | `montepython`, `montepython` |
| `cov` | `data/KV450/covariance/cov_analytic_montepython_mcorr.txt` |
| `m_correction` | -0.0128 -0.0104 -0.0114 0.0072 0.0061 |
| `constant_c_offset` | T |
| `xi_pm_c_file` | `data/KV450/systematics/KV450_ALL_c12_treecorr.out` |
| `like_name` | `mini_2pt_like` |
| `keep_theory_vector` | T (this triggers an RNG draw every call; section 4, Q11) |

Hard-coded: arcmin -> rad factor `pi/180/60`; default `constant_c_offset=True`; default `keep_theory_vector=False`;
default `order_*` `xi_pm-bin-theta`; spline order 3.

Nuisance parameter ranges and priors (`KV450_values.ini`, `KV450_priors.ini`):

| Parameter | Range [low start high] | Prior |
|---|---|---|
| `shear_c_bias/delta_c` | [-6e-4, 0, 6e-4] | N(0, 2e-4) |
| `shear_c_bias/A_c` | [0.62, 1.13, 1.4] | N(1.01, 0.13) |
| `nofz_shifts/bias_1..5` | [-0.1, 0, 0.1] | N(0, sigma_i) as in 2.7 |
| `intrinsic_alignment_parameters/A` | [-6, 0, 6] | flat (the Gaussian prior line `;A = gaussian 1.13 0.48` is commented out, `KV450_priors.ini:2`) |

`KV450_no_sys.ini:197-213` disables the c and m treatments: `constant_c_offset = F`, `m_correction` and
`xi_pm_c_file` commented out, `shear_c_bias` fixed at delta_c = 0, A_c = 1 (`KV450_no_sys_values.ini:35-37`).
[verified] It also removes `source_photoz_bias`, `linear_alignment` and `add_intrinsic` from `modules =`
(`KV450_no_sys.ini:11`), so "no_sys" means no IA and no photo-z shift as well as no c/m treatment. It reads the
same "mcor" data file and the same covariance. `BOSS_KV450.ini:223-239` repeats the fiducial `[mini_2pt_like]`
block; its BOSS likelihood section is named `[xi_wedges_like]` (`:157-164`), uses the default `n_mocks = 2048`
(not set in the ini) and `keep_theory_vector = F`.

### 3.3 Path B

| Item | Value / behaviour | Source |
|---|---|---|
| Extension names | defaults `xiP`, `xiM`, `PneE`, `PeeE`, `En` | `scale_cuts.py:21-25` |
| Theory section defaults | `shear_xi_plus`, `shear_xi_minus`, `bandpower_ggl`, `bandpower_e_cosmic_shear`, `cosebis` | l.28-32 |
| Max tomographic bins probed | 1000, loop breaks at first missing `bin_i_i` (or `bin_i_1` for GGL) | l.112, 150-174 |
| Band centre | 10^(0.5 (log10 l_lo + log10 l_hi)) | l.141-144 |
| Covariance extension | `COVMAT` (hard-coded) | l.59 |
| Covariance inversion | `np.linalg.inv`, once at setup | l.89 |
| Mock generation | optional: `simulate`, `simulate_with_noise` (default True), `number_of_simulations`, `mock_filename`; draws `multivariate_normal(mu, C)` per call | l.91-94, 200-219 |
| Prints | many `print`/stdout lines on every call | l.102, 195; `twopoint.py` mask functions |
| Per-call datablock writes | `data` (n), `covariance` (n x n) and `inv_covariance` (n x n) are re-written to the block on every call although constant | l.106-108 [verified] |
| Error handling | bare `except:` around `from_fits` (any error becomes "not found", l.58-61, 38-42) and around each `block[...]` read in the tomographic loops (l.157-161, 170-174): any error, not just a missing key, silently ends the loop, so a mistyped or malformed `bin_i_j` truncates the theory vector instead of failing | [verified] |
| Simulation mode | with `simulate = T`, `config['data']` and the stored `inv_covariance` are NOT updated: only `TP_data` is replaced and written to `mock_filename`, so the chi2 of that very run still uses the original data | l.200-219 vs l.87-89 [verified] |
| Entry points | `mini_like.py` and `correlated_priors.py` define `clean`, not `cleanup`, so CosmoSIS never calls them; harmless | `mini_like.py:29`, `correlated_priors.py:30` [verified] |

Per-run settings of the committed K1000 3x2pt run (`pipeline.ini`): bandpower modules `l_min = 100`,
`l_max = 1500`, `nbands = 8`, `response_function_type = tophat`, `analytic = 1`, `apodise = 1`, `delta_x = 0.5`,
`theta_min = 0.5`, `theta_max = 300`; projection `ell_min = 1`, `ell_max = 30000`, `n_ell = 400`; photo-z
shift `interpolation = cubic`; `magnification_alphas alpha_binned = 1.8 2.62`.

### 3.4 Path C

| Item | Value | Source |
|---|---|---|
| Data columns | wedge j value = column 2j+1 of the (n_s, 1+2 n_wedge) text file | `mini_BOSS_like.py:20-22` |
| Cuts | `points_range = [4, 32]` default; keep indices 4..31 in every wedge block | l.36-48 |
| Hartlap | `n_mocks = 2048` default (the K1000 `[BOSS_like]` sets 2048 explicitly; `BOSS_KV450.ini` relies on the default); set `n_mocks = 0` to turn off. p = 84 for both bins, giving 1962/2047 = 0.958476 (computed) | l.26, 53-57 |
| keep_theory_vector | the committed K1000 `[BOSS_like]` sets `keep_theory_vector = T` (and `run_kcap.py:750` hard-codes True), so the RNG draw `multivariate_normal(mu, cov[b])` (84-dim) runs for both z-bins on every call; `BOSS_KV450.ini` uses F [verified] | l.77-83 |
| Chi2 | summed over bins, `einsum('i,ij,j')` | l.75 |
| BOSS ini | `bands_range = 20 160` and `points_range = 4 32` in the wedge module section; `num_ell = 3`; `z_eff = 0.38 0.61` | `pipeline.ini` [wedges] |

The wedge cov blocks are assumed to be wedge-major, as are the data (`np.concatenate` of three columns).

### 3.5 Generic `2pt_like.py` (not used by committed runs)

| Item | Behaviour | Source |
|---|---|---|
| Angular units | real-space spectra converted to radians | l.160-168 |
| Theory evaluation at data angle | `SpectrumInterp`: interpolation in ln(angle), `interp1d` linear; log-log if all theory > 0 (`loglog`), negative of log-log if all < 0, else linear in ln(angle) | `spec_tools.py:21-50`; `2pt_like.py:390` |
| Out-of-range angle | `ValueError` ("increase the range when computing/projecting") | l.398-404; the 1e-9 nudge of first/last interpolation points in `SpectrumInterp.__call__` |
| Scale cuts | per bin `angle_range_<name>_<b1>_<b2>`, bin-pair cuts `cut_<name> = (1,2) (1,3)` (parsed with `eval`) | l.116-141 |
| `cut_zeros`, `cut_cross`, `data_sets` | options | l.89-106 |
| Theory lookup | tries `bin_b1_b2`, then `bin_b2_b1` only if type1==type2 | l.381-388 |
| Type/section table | `type_table.txt` maps e.g. (G+R, G+R) -> `shear_xi_plus`/`theta`/`bin_{0}_{1}` | |

---------------------------------------------------------------------------------------------------

## 4. Quirks, approximations, deviations from the textbook formula

Items marked QUIRK are behaviours I derived from reading the code. None was executed.

- **QUIRK Q1 (`cut_cross` always True when present).** `wrapper_twopoint.py:127` does
  `cutCross = bool(value)`, and `scale_cuts.py:54` turns every inline value into a string via `str(...)`. In
  Python `bool('False') == bool('F') == True`. Any `cut_cross = ...` line, even `False`, enables cross-bin
  removal. Omitting the key gives False. [verified] Confirmed with the installed CosmoSIS option parser:
  `cut_cross = False` and `= F` both arrive as Python `False`, which `str()` turns into `'False'`, and
  `bool('False')` is True. The shipped `scale_cuts.ini` comment example `cut_cross = False ;; Trivial` therefore
  does the opposite of what it says. Also, the cross-cut keeps rows with BIN1 == BIN2 (`auto_bins`, `twopoint.py:318`)
  and their covariance blocks; for PneE that is the numerical coincidence lens index == source index (e.g. (1,1),
  (2,2)), which has no physical meaning.
- **QUIRK Q2 (inconsistent inequalities).** Per-stat `keep_ang_<stat> lo hi` uses strict `lo < ANG < hi`
  (`twopoint.py:735-736`), per-pair `keep_ang_<stat>_a_b lo hi` is inclusive (`:710-712`). K1K MontePython uses
  `99.5 1500.5` to dodge edge cases.
- **QUIRK Q3 (cuts are applied to the theory ANG, not by index).** The theory vector is masked using the
  angles in the datablock (`theta_bin_1_1`, band centres), not by reusing the data mask. No check verifies
  that theory ANG equals data ANG; a different number of bands or thetas would produce a vector of the wrong
  length (caught only when chi2 shapes disagree) or, if the length happens to agree, silently wrong
  alignment.
- **QUIRK Q4 (delta_c is squared in one ordering only).** In `order_cov = montepython` the constant offset is
  `delta_c**2` (l.226), in `xi_pm-bin-theta` it is `delta_c` (l.187). The fiducial values/priors
  (`delta_c` in [-6e-4, 6e-4], N(0, 2e-4)) only make physical sense for the squared version (c^2 ~ 4e-8,
  comparable to the c-term file amplitudes 5e-7..1e-9). In the linear branch a 2e-4 offset would be orders of magnitude above the xi+
  amplitude at the largest kept scales (computed from the data file: |xi+| at 55.5 arcmin spans 2.5e-7 to 4.8e-6
  over the 15 pairs and at 112.6 arcmin 5e-8 to 2.4e-6; the single numbers 1.8e-6 and 1.4e-6 quoted earlier
  were individual pairs, not "the first pairs"). The offset is applied to xi+ only. Same value
  for all pairs and all thetas. [verified] Counter-point: the C++ module `xipm_binned.cc` adds
  `delta_c = c1^2 + c2^2` linearly (l.652, 772), and `values.ini` for K1000 has `c1`, `c2` rather than
  `delta_c`, so "delta_c" is the c^2-like quantity there; in `mini_2pt_like` the montepython branch squares the
  sampled `delta_c` again, so the sampled parameter is the amplitude c there. The prior N(0, 2e-4) is
  consistent with delta_c being an amplitude c (c ~ 2e-4 gives c^2 = 4e-8), which is the only reading under which
  the montepython branch is physically sensible.
- **QUIRK Q5 (m-bias applied to data with covariance untouched).** `d /= (1+m_a)(1+m_b)`
  (`mini_2pt_like.py:140-141`) and Psi is computed from the unmodified covariance. A model-side
  multiplication would be the standard form. The two differ in how errors are weighted at the (1-2)% level.
  The m values in `m_correction` are fixed; there is no m uncertainty anywhere.
- **QUIRK Q6 (possible double m-correction, unresolved).** The data file name contains "mcor" and the
  covariance "mcorr", yet the fiducial ini divides the data by (1+m)(1+m) again, and `KV450_no_sys.ini` reads
  the same file without any m correction. If the file is already m-corrected, the fiducial run corrects twice.
  The clone does not say (Q-open 1).
- **QUIRK Q7 (A_c convention differs between modules).** `mini_2pt_like.py` and `shear_c_bias.py` use `A_c**2`
  times the c-term file; `xipm_binned.cc:851` uses `Ac * xi_2D` and its own comment questions it.
- **QUIRK Q8 (integrate mode is a node trapezoid, not an integral over the bin).** Only nodes with
  e_k <= theta_n < e_{k+1} are used; the partial intervals at both ends are dropped (about 0.5 node on each end
  out of ~30). The result depends on where the 1024-node Hankel grid falls relative to the edges (magnitude of the
  effect not measured). The C++ module uses 100 log sub-nodes with a spline instead.
  The two binning implementations do not give identical vectors.
  [verified, computed] Magnitude: for a power-law xi = theta^s on the actual 1024-node grid and the fiducial
  edges, the node-trapezoid (theta-weighted) bin average differs from the exact bin average by +0.32 / -0.04 /
  +0.40 / +0.03 / +0.47 / +0.11 / +0.54 / +0.18 / +0.62 % for s = -1 (bins 1..9), by 0 for s = 0 and by
  -0.66 ... -0.95 % (odd bins) for s = +1, i.e. below 1% but systematic and bin-parity dependent (the 29/30
  node alternation), which is small compared with the KV450 errors yet not negligible for a precision clone.
- **QUIRK Q9 (validity window of the Hankel module).** The module doc says numerically valid only from about 1
  arcmin to a few hundred arcmin, but the fiducial KV450 selection keeps xi+ bins starting at 0.5 arcmin and xi-
  bins up to 300 arcmin (the last kept xi+ bin, bin 7, ends at 72.4 arcmin; the last xi- bin spans 147-300 arcmin). I did not measure the
  numerical error there.
- **QUIRK Q10 (`angular_binning_mode = none` crashes).** The Flinc inis set `none` in `[mini_2pt_like]`
  (`runs/config/Flinc/xipm_Flinc.ini:258`), which raises `ValueError` in `get_theory_point`. These sections are
  inert because those pipelines list `pcfs` (a `lib2pcfs_likelihood.so` whose source is not in the clone), not
  `mini_2pt_like`, in `modules =`.
- **QUIRK Q11 (RNG draw and 195x195 inverse saved on every call).** With `keep_theory_vector = T` (fiducial)
  `np.random.multivariate_normal(mu, cov)` runs on EVERY likelihood evaluation (l.251). Its cost (~17 ms for
  n=195 on my box, SVD based) exceeds the rest of the module. The inverse covariance is also re-published
  to the datablock each call. [verified] On re-timing during this audit (same numpy 1.26.4, other machine load)
  the draw took 4.5 ms, so the absolute figure is machine/thread dependent; the structural claim (SVD-based
  draw at every call, larger than the chi2) stands. The same pattern occurs in `mini_BOSS_like` for the
  committed K1000 run (`keep_theory_vector = T`), two 84-dim draws per call. The installed CosmoSIS
  `GaussianLikelihood` instead simulates with a precomputed Cholesky factor (`mu + chol @ randn`).
- **QUIRK Q12 (dead/inconsistent `shear_c_bias.py`).** It reads `block["shear_xi", "nbin_a"]` (l.22), a section
  that does not exist with the kcap `cl_to_xi` output (`shear_xi_plus`). It writes `delta_c` into the xi-
  entry too (l.37) while the likelihood does not. Its `xiplus_i_j`/`ximinus_i_j` keys are not read by any
  module I found. It is not in any committed `modules =` list.
- **QUIRK Q13 (fragile option handling in `mini_2pt_like.setup`).** (a) `cut_xi_*` with a single value parses to
  a scalar, so `cut_xi_minus_idx[:, None]` (l.74, 78) raises. (b) `cut_xi_plus_idx != []` is applied to a
  numpy array. [verified, RESOLVED] The installed CosmoSIS option parser returns `cut_xi_plus = 7 8` as an
  `int32` ndarray and a single `7` as a Python int. On numpy 1.26.4 (this machine) `np.array([7, 8]) != []` raises
  `ValueError: operands could not be broadcast together with shapes (2,) (0,)` (the elementwise-comparison
  FutureWarning of numpy < 1.25 was finalised into an error in 1.25). So `mini_2pt_like.setup` raises at
  l.73 with the fiducial `cut_xi_minus = 0 1 2` on numpy >= 1.25; it only works on older numpy, where the
  comparison degraded to a scalar True with a warning. A single-value cut raises `TypeError` instead
  (`int[:, None]`). Porting needs `len(cut) > 0` / `np.atleast_1d`. (c) With
  per-bin input files (`bin_i_j` options, l.132-138) `n_theta` is undefined, so the covariance cut at l.74/78 can
  raise NameError. (d) `angular_bin_edges *= pi/180/60` mutates the options array in place.
- **QUIRK Q14 (statistics order is hard-wired on the theory side).** The theory vector is built in the fixed
  order xiP, xiM, PneE, PeeE, En (`scale_cuts.py:117-123`). The data vector uses FITS extension order. A FITS
  with a different order (e.g. PeeE before PneE) silently mismatches. The committed K1000 file has PneE
  before PeeE, matching.
- **QUIRK Q15 (substring match in `mask_scale`).** `spectrum.name not in spectra_to_cut` (`twopoint.py:731`) is a
  substring test on a string when `spectra_to_cut` is a string like `'PeeE'`, not a membership test on a list.
  No current extension name is a substring of another, so no live bug.
- **QUIRK Q16 (dz prior mean is not zero in the K1000 inis).** `priors.ini` gives p_i a Gaussian with nonzero
  mean for i=2..5, so the shift prior mean is L mu = (0, -0.002, -0.013, -0.011, +0.006) with the matrix the
  K1000 run actually reads (the earlier (0, -0.00203, -0.0128, -0.0108, +0.00718) used the KV450 matrix and was
  wrong for K1000). Nothing in the code documents why; it encodes the data package's n(z) shifts, and the
  round numbers show mu was back-solved from them [verified].
- **QUIRK Q17 (hard-coded sky constant).** `convert_nz_steradian` uses 41253 deg^2 for the full sky instead of
  4 pi (180/pi)^2 = 41252.96. Only relevant to the unused Gaussian covariance mode.
- **QUIRK Q18 ([verified] FITS test covariance pair order).** `modules/scale_cuts/test_files/twoPoint_KV450.fits`
  has xiP/xiM identical to the text data for all pairs, but its COVMAT pair blocks do not follow its own BIN1/BIN2
  order (details and the shot-noise test in 1.4). Anyone using that file as a regression reference for
  `mini_2pt_like` vs `scale_cuts` will see unexplained chi2 differences.
- **QUIRK Q19 ([verified] section/array names with `nbin_a` mismatch).** `mini_2pt_like.execute` reads `nbin_a`
  from `shear_xi_plus` only (l.156); `copy_metadata` writes `nbin_a`/`nbin_b` into both xi sections
  (`cl_to_xi_interface.c:374-385`), so this works, but `n_bin` from the ini and `nbin_a` from the block are
  used for different loops (setup vs execute) with no consistency check.
- **QUIRK Q20 ([verified] column-vs-index cut asymmetry).** In Path A, `cut_xi_plus`/`cut_xi_minus` are
  0-based theta indices applied identically to every pair, to the data, the c-term file and the covariance; in
  Path B cuts are on ANG values and can differ per pair. The two paths agree only when the cut angles fall
  between the same bin centres (true for the KV450-like preset, section 2.5).
- **Approximation A1.** Single-sample ("SAMPLE") window: bin-averaged quantities are compared to their
  C++-side band-averaged values; no window function / pair-weighted theta integration inside Python.
- **Approximation A2.** Hartlap on the inverse (BOSS) and on the covariance (generic `2pt_like`) are the
  same factor; both are first-order and the Sellentin-Heavens form is only in the generic path.
- **Approximation A3.** Cross-covariance between KiDS and BOSS likelihoods is zero (separate chi2 terms).

---------------------------------------------------------------------------------------------------

## 5. Strengths

- Simple, readable Gaussian likelihoods; the inverse is computed once at setup and chi2 is one quadratic form.
  Per-call cost is dominated by data handling, not linear algebra (n = 142..195).
- Clean separation between data/cut handling and chi2: `scale_cuts.py` + `mini_like.py` give a generic path
  that works for band powers, COSEBIs and xi+/- from one twopoint FITS (the CosmoSIS/DES twopoint
  format; gamma_t is only half-supported: data side but no theory builder), with cuts applied identically and
  consistently to data, covariance and theory.
- Cutting the covariance before inversion (not slicing the inverse) is correct.
- Correct, standard Hartlap/Sellentin implementations exist where covariances are simulated (BOSS,
  generic 2pt). [verified] Checked: Hartlap factor (N-p-2)/(N-1) in all three places (`mini_BOSS_like.py:53-57`
  on Psi, `2pt_like.py:245-246` as its inverse on C, `xipm_like.py:38-42` on Psi); Sellentin-Heavens
  `-N/2 ln(1 + chi2/(N-1))` (`2pt_like.py:334`), which uses the UNcorrected C.
- Reproducible, human-readable data files: small text/FITS files, shapes and orderings are self-consistent
  (my cross-check: FITS xiP and xiM equal the text KV450 data for all digits for all 15 pairs; covariances are
  exactly symmetric and positive definite; BUT the FITS test COVMAT pair order is inconsistent, Q18).
- Photo-z uncertainty is handled as correlated Gaussian priors via a Cholesky map, so the sampler works with
  an identity prior in p (efficient for nested sampling).
- Scale cuts are specified declaratively in the ini, per statistic, per pair, per angle range, with a fixed
  application order.
- The K1000 pipeline avoids a fine Hankel transform for band powers (band-power kernels computed in C++),
  and the MontePython port avoids matrix inversion by a Cholesky solve.

## 6. Weaknesses / risks

Correctness

- Q1-Q7, Q13-Q15 above. The largest correctness risks are (a) the always-True `cut_cross` flag, (b) the
  `delta_c` vs `delta_c**2` split, (c) the unresolved possibility of double m-correction, and (d) the
  m-correction not entering the covariance or any nuisance parameter.
- No m-bias marginalisation: only a fixed value is applied. The m uncertainty (typically 1-2% per bin) is
  absent from the likelihood unless baked into the covariance externally.
- No cross-check that theory ANG and data ANG agree (Q3); no check on the number of tomographic bins between
  data and ini (`n_bin` vs `nbin_a` from the block).
- No log-determinant, so any variant with a cosmology-dependent covariance would be wrong without changes.
- Binning mismatch between Python (node trapezoid, Q8) and C++ (100-point spline) implementations; no test in
  the clone compares them.
- `lib2pcfs_likelihood.so` (Flinc inis) has no source in the clone; the `mini_2pt_like` blocks in those inis
  cannot run (Q10).

Accuracy

- Model xi is obtained from the 1024-node Hankel grid, used at 0.5 arcmin and up to 300 arcmin where the
  module itself states limited validity (Q9). The node trapezoid does not integrate the partial end
  intervals of each bin.
- Spline interpolation (mode `interpolate`) is a global cubic through noisy-looking data of oscillating sign
  changes; for xi- it can ring. Only used if the user selects it.

Maintainability / portability

- Hard dependency on CosmoSIS internals (`cosmosis.datablock`, `names`, `errors.BlockNameNotFound`) and on
  unavailable external modules (`kcap_boss_module`, `cosebis_cosmosis`).
- A vendored Python-2 compatible `enum34.py` (790 lines) and `twopoint.py` (1319 lines) are duplicated in
  `modules/scale_cuts/` and `cosmosis-standard-library/likelihood/2pt/` (1215 lines); the copies differ.
- Heavy console output on every likelihood call (`print` in `scale_cuts.execute`, the mask routines).
- `np.trapz` is used in several places (mini_2pt_like, photoz_bias, load_nz); it is deprecated in NumPy 2.x in favour of `np.trapezoid`.
- Dead or stale code: `shear_c_bias.py`, the Flinc `mini_2pt_like` sections, `module.yaml` of
  `cl_to_xi_nicaea` (documents `xiplus_i_j` rather than `bin_i_j`).
- The `ptest.ini` default test refers to `../kcap`, `../cosebis_cosmosis` paths and `.so` libraries that do not
  exist in this tree.
- The Python bool trap (Q1), the string substring test (Q15), eval-based parsing of `cut_<name>` options in
  `2pt_like.py:133-134` (arbitrary code from an ini file).

Performance (section 7 quantifies)

- Python-level object rebuilding in `scale_cuts.execute` for every likelihood call.
- `InterpolatedUnivariateSpline` is built per pair per call in `integrate` mode but never used (`intp`, l.12).
- 30 independent Python loops of `np.trapz` per call.
- The RNG draw in Q11.

## 7. Concrete efficiency opportunities

Timings below are from a stand-in micro-benchmark on this WSL machine (numpy 1.26.4, scipy 1.13.1, one thread,
my own re-implementation of the same arithmetic, same data sizes). They are not measurements of kcap itself.

1. **Precompute a binning matrix W (n_kept_bins x 1024) once.**
   - Because B_k is linear in the 1024 theory nodes with weights that depend only on the (fixed) theta grid and
     the edges, the 15 pairs x 2 stats reduce to one matmul: mu = Y @ W^T with Y of shape (30, 1024) and W of
     shape (9, 1024) (sparse support of ~30 nodes per row).
   - My stand-in: old loop (spline construction + 9 masked trapz per series, 30 series) **28.7 ms** per call;
     matrix form **~28 µs** (single thread) with relative difference 3e-16. That is a speed-up of about 10^3 for
     this step.
     [verified] Re-timed during the audit on the same stand-in (numpy 1.26.4, quieter machine): 7.2 ms vs 17 µs,
     relative difference 2e-16, speed-up ~4x10^2. Absolute times scale with machine load; the matrix form is
     exact (it reproduces trapz bit-for-bit up to rounding) and the speed-up is 10^2-10^3 either way.
   - It should also drop the unused spline.
2. **Remove the per-call `multivariate_normal`** (keep_theory_vector). Measured **16.6 ms** per call for n=195 (4.5 ms on a re-timing during the audit, see Q11),
   comparable to or larger than everything else in the module. Precompute the Cholesky of C once and draw only
   if requested. Saves ~15 ms/call. The same holds for `scale_cuts.execute` with `simulate = T`, and for
   `mini_BOSS_like` with `keep_theory_vector`.
3. **Index-based theory assembly for Path B.** Build, at setup, the integer index list mapping each retained
   data point to (statistic, pair, band) in the theory sections, and then in `execute` do
   `theory = np.concatenate([block[sec, key][idx] ...])`. This removes `SpectrumBuilder`, `TwoPointWrapper`, ~40
   Python-loop mask calls and all `print`s per call. I did not time it; the object construction and per-pair
   `np.where` loops are plausibly milliseconds to tens of milliseconds per call, versus microseconds for the
   indexed version. Even the C++ bandpower modules would then dominate.
4. **Matrix-product chi2.** `d @ Psi @ d` (37 µs measured; 7.6 µs on re-timing) beats `einsum('i,ij,j')` (68 µs measured; 43 µs on re-timing) for
   n=195; with the Cholesky factor L of C and a triangular solve (as the MontePython path does) one gets
   the same cost without ever forming an inverse and better conditioning (cond(C) = 2.4e5 for the cut KV450 matrix and 3.3e4 for the cut K1000 matrix
   in the clone, so the inverse is safe in double precision, but the Cholesky route is more robust).
5. **Replace the Hankel FFT + node binning by a direct bin-averaged transform.** For xi+/- the data are
   bin averages. With precomputed kernels K^{+/-}_{k,l} = ∫_bin theta J_{0/4}(l theta) dtheta / ∫_bin theta dtheta
   (a (9 x n_l) matrix per sign), mu^{+/-}_k = Σ_l (l/2π) C_l K_{k,l} Δ_l is one (15 x n_l) @ (n_l x 9) matmul.
   This removes (i) the 1024-point FFT per pair (30 per call), (ii) the validity-window problem (Q9) and (iii)
   the node-alignment error (Q8). It requires interpolating C_l (400 nodes) onto a finer ell grid (a few 10^4
   points) because J_0(l theta) oscillates; the K matrix is then ~9 x 3e4. Estimated cost: tens of µs per call
   after a one-off kernel precomputation; the existing Hankel module costs 30 FFT/spline passes (not measured).
   Treat the expected gain as an estimate.
6. **Analytic marginalisation of linear nuisances.** Any parameter that enters the model linearly with a
   Gaussian prior can be marginalised exactly: for a template t and prior N(0, s^2) the covariance becomes
   C + s^2 t t^T, a rank-1 Sherman-Morrison update of Psi. In the `xi_pm-bin-theta` branch `delta_c` is such a
   parameter (t = 1 on the xi+ entries). In the fiducial `montepython` branch it enters as `delta_c^2` and
   `A_c` as `A_c^2`, which are not linear, so they would need a different treatment (e.g. sample c^2
   directly). The benefit is fewer sampled dimensions, not per-call time.
7. **Whiten once.** Pre-multiplying d and mu by L^-1 turns chi2 into a dot product; useful for batched
   evaluations over many cosmologies (emulators, vectorised samplers).
8. **Cache n(z) shifts.** `photoz_bias` redoes `interp1d` + `trapz` per bin per call; for a fixed grid this is a
   precomputable linear operator in the shift for small shifts (or a vectorised `np.interp`). Small saving
   relative to the Boltzmann solver; listed for completeness.
9. **Unify and test.** One tested binning implementation (Python matrix or C++), a unit test comparing the
   two on the same C_l, and a golden test using the committed data files (195 / 142 / 168 points, chi2 of a
   reference theory vector) would catch the quirks above.

Overall expectation: the likelihood module is a small part of a full evaluation (CAMB + projection + Hankel
dominate), so items 1-3 remove ~30-50 ms of avoidable Python overhead per call, which matters mainly for
fast/emulated theory pipelines.

## 8. Open questions (could not resolve from the clone)

1. **Is the KV450 data file already m-corrected?** (Q6.) The file name says `mcor`, the fiducial ini applies
   `m_correction` again, and `KV450_no_sys` uses the same file with no correction. The provenance of the five
   `m_correction` numbers is not in the repo (two of them, -0.0128 and +0.0072, numerically match entries 3 and
   5 of L mu computed with the KV450 `SOM_cov_multiplied.asc`, -0.012799 and +0.007181 [verified]; it is not
   the K1000 matrix. This may be a coincidence).
2. Does `cov_analytic_montepython_mcorr.txt` include the m-bias and photo-z uncertainty, and how were the
   theta weights chosen in its derivation? Does the covariance assume the same (theta-weighted) binning as
   the model?
3. What is inside the KiDS-1000 FITS covariance (iterated, analytic with m-bias folded in, c-term)? No
   documentation in the clone (the config generators point at a `fits_iterative_covariance` directory, so it is
   probably the iterated covariance [verified]). The BP data file contains no xi+/-, COSEBIs, or c-term extension.
4. The source of `lib2pcfs_likelihood.so` (Flinc xi+/- pipeline), `libcosebis_likelihood.so`
   (`KV450_fiducial.ini:204` "[cosebis]"), and the `kcap_boss_module` wedge module (theory shape (n_wedge,
   n_points) and whether it is already cut to points 4..31) are not in the clone.
5. The CosmoSIS core (`GaussianLikelihood`, `names.data_vector`, `names.likelihoods`, option parsing of
   whitespace-separated ints/floats and booleans) is external; behaviours attributed to it above (section names,
   lower-casing of ini keys, bool parsing) rest on how kcap uses them, not on reading the source.
6. How many mocks produced the BOSS covariance (2048 default vs. the Patchy count in the original paper)?
7. Which statistic is used by the published KiDS-1000 3x2pt analysis for the xi+/- part and what theta
   weighting (pair-weighted vs theta-weighted) was used; `xipm_binned.cc` supports all three.
8. Exact numerical error of the Hankel module at 0.5 arcmin (xi+) and 300 arcmin (xi-): not measured.
9. [RESOLVED, verified] Behaviour of `cut_xi_plus_idx != []`: raises `ValueError` on numpy >= 1.25 (Q13b);
   `np.delete(x, [])` is fine. The NumPy version used by the authors is still unknown, but it must be < 1.25.
10. [new, verified] Which pair order does `cov_analytic_montepython_mcorr.txt` use relative to the data file?
    The shot-noise test (1.4, Q18) says natural order is self-consistent, and the FITS test COVMAT is not, but I
    could not find the script that produced `twoPoint_KV450.fits`.
11. [new, verified] `m_correction` and a possible double correction: whether `KV450_reweight_..._mcor_...dat`
    already contains an m correction remains undetermined (unchanged from item 1).

## 9. Re-implementation checklist (minimum to reproduce kcap numbers)

1. Read the montepython-ordered KV450 data (18 x 16) and cov (270 x 270) as in 1.4. Drop xi+ theta indices
   {7,8} and xi- indices {0,1,2} per pair using index = pair*18 + [0..8] (xi+) and pair*18 + 9 + [0..8] (xi-),
   with pairs ordered (0,0),(0,1)..(4,4). Check shape 195 x 195. [verified] Use the TEXT covariance, not the
   COVMAT of `twoPoint_KV450.fits` (Q18); expected checks: cond(C_cut) = 2.38e5, first retained diagonal 1.1197e-8.
2. Divide each pair's xi+ and xi- by (1+m_a)(1+m_b), m = (-0.0128, -0.0104, -0.0114, 0.0072, 0.0061).
3. For the model: take xi+/-_ab(theta_n), n = 0..1023 on the grid of 3.1. Average each retained bin as in
   2.2 (i) (trapezoid on in-bin nodes, weight theta).
4. Add `delta_c^2` to xi+ (all pairs/bins), and `A_c^2 * c_+/-[k]` (file columns 3 and 4, same cuts) to
   xi+/-.
5. chi2 = r^T inv(C_cut) r; ln L = -chi2/2; no log det, no Hartlap.
6. For K1000 band powers: use the FITS extensions, cut as in 2.5 (142 points) with inclusive per-pair and strict
   per-stat ANG limits; chi2 with the inverse of the cut covariance; no Hartlap; BOSS added separately with the
   0.95848 factor.
7. Photo-z: n'(z) = n(z - b) renormalised by trapz; b = L p with L = chol(SOM_cov_multiplied) and p ~ N(mu_p, I), using the matrix from the K1000 run directory (not
   the KV450 one); expected prior mean of b = (0, -0.002, -0.013, -0.011, 0.006) [verified]. The KV450 pipeline
   instead uses independent N(0, sigma_i) with sigma = (0.039, 0.023, 0.026, 0.012, 0.011).
8. [verified] A numpy >= 1.25 port must not copy `cut != []`; use explicit `len()` checks (Q13b).
