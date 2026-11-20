# 04 — C(l) to correlation functions, COSEBIs and band powers (kcap)

Scope: Hankel transform C(l) -> xi_+/-, gamma_t, w (nicaea module), theta-binning of xi, COSEBIs (E_n from C(l) and from xi), band powers (cosmic shear E/B, GGL), and the other `cl_to_*` modules shipped in the tree.

Reference tree: `/mnt/d/cloning_kids_Kappa_pipeline/reference/kcap`. Paths below are relative to it. `CSL` = `cosmosis-standard-library`.

Conventions in this document:
- "read" means taken directly from source, with file:line.
- "VERIFIED-NUM" means I re-derived the formula in an independent Python script (scripts in `docs/study/04_verification/`; kcap code was NOT built or run) and it reproduced either the shipped tables or a direct quadrature.
- "QUIRK" marks a deviation from the textbook formula or a surprising implementation detail.
- Every number I state as measured is flagged; run times are estimates and are labelled as such.
- "[verified]" marks text added or corrected by an independent adversarial re-read of the kcap source (and, where stated, re-running the scripts in `docs/study/04_verification/` or fresh numpy checks). "[verified correction]" marks a statement of the original draft that was wrong or imprecise and has been fixed in place. Original text that survived the re-read is left untouched.

---------------------------------------------------------------------------------------------------

## 1. Purpose, role in the pipeline, inputs and outputs

### 1.1 Where this sits

Pipeline order in the KV450/KiDS fiducial inis (`runs/config/KV450_fiducial.ini:11` [verified correction: draft cited :13, the `modules =` line is 11], `utils/run_kcap.py:60-160`):

```
camb -> extrapolate_power -> load_nz -> photoz_bias -> linear_alignment
     -> projection (project_2d: Limber, writes shear_cl)  -> add_intrinsic
     -> [ cl2xi (nicaea) -> (xipm_binned | mini_2pt_like "integrate") ]     <- xi_+/- route
     -> [ cosebis (libcosebis.so) ]                                         <- COSEBIs route (from C(l))
     -> [ bandpower_shear_e / bandpower_ggl (libbandpower.so) ]              <- KiDS-1000 3x2pt route
     -> scale_cuts / likelihood
```

Which route is used where (read):
- KV450 fiducial: `cl2xi` = `cl_to_xi_nicaea/nicaea_interface.so`, `corr_type = 0` (`KV450_fiducial.ini:165-167`); likelihood `mini_2pt_like.py` with `angular_binning_mode = integrate` (`KV450_fiducial.ini:189`).
- KiDS-1000 3x2pt "EE_nE_w" production pipeline: band powers only (`runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline.ini`, sections `[bandpower_shear_e]`, `[bandpower_ggl]`), no xi and no COSEBIs.
- `utils/run_kcap.py` has two more run types: `xipm` (`cl2xi_shear`, `bin_xi_plus`, `bin_xi_minus`) and `cosebis` (`libcosebis.so`) (`run_kcap.py:146-155`).
- `cl_to_corr` (FFTLog), `cl_to_xi_fullsky` (Legendre), `cl_to_xi_wigner` are in the tree but are NOT referenced by any kcap run config (grep over `runs/`, `utils/`).

### 1.2 Inputs (all from the projection module `project_2d.py`)

Section `shear_cl` (default; overridable with `input_section_name`):
- `ell`: 1-D float array, length `n_ell`, `np.logspace(log10(ell_min), log10(ell_max), n_ell)` (`CSL/structure/projection/project_2d.py:415`). Fiducial values: KV450 `ell_min=0.1, ell_max=5e5, n_ell=400` (`KV450_fiducial.ini:149-151`); `run_kcap.py` (KiDS-1000 pipelines) `ell_min=1, ell_max=3e4, n_ell=400` (`run_kcap.py:556-558`).
- `nbin_a`, `nbin_b` (ints; `nbin` also written for auto-correlations, `project_2d.py:691-694`).
- `bin_i_j`: 1-D array (`n_ell`,) of C(l) (no l(l+1) factor; the module.yaml says so). For auto-correlations only j <= i is stored (`project_2d.py:704-705`; the comment there says both orderings are saved but the loop only writes `j<=i`).
- metadata keys the nicaea module insists on: `is_auto`, `sample_a`, `sample_b`, `save_name` (`cl_to_xi_interface.c:123-150`). [verified correction] "Insists" is too strong: `copy_metadata` ORs the datablock status but its return value is DISCARDED at both call sites (`:375`, `:383`), so a missing key does not fail the module; instead `c_datablock_put_string` is then called with an uninitialised `char *` (undefined behaviour). All four keys are always written by `project_2d.py` (`:684-696`), so this only bites on hand-made inputs.
- [verified] Default `input_section_name` / `output_section_name` of nicaea depend on `corr_type` and `filter_type` (`cl_to_xi_interface.c:54-72`): corr_type 0 -> `shear_cl` / `shear_xi` (then `_plus`/`_minus`), 2 (ggl) -> `galaxy_shear_cl` / `galaxy_shear_xi`, 1 (matter) -> `galaxy_cl` / `galaxy_xi`; with `filter_type=1` the stem is `shear_map` / `galaxy_shear_map` / `galaxy_map`. Enum values: `shear_shear=0, matter=1, ggl=2`, `corrfct=0, map=1` (`:16-17`). Note this differs from `cl_to_xi_wigner`, whose `corr_type` is 0=xi, 1=gammat, 2=wtheta (see 2.7).
- [verified] In the KiDS-1000 `run_kcap.py` / `pipeline.ini` configs the shear C(l) written to `shear_cl` is the `fast-shear-shear-ia` spectrum (`FastShearShearIA`, `project_2d.py:313-322`, which writes `names.shear_cl`) and `add_intrinsic` has `shear-shear = F`; in the KV450 inis `shear-shear`, `shear-intrinsic`, `intrinsic-intrinsic` are projected separately and summed by `add_intrinsic`. Either way the transforms below see ONE `shear_cl` that already contains GG+GI+II.
- C(l) definition (Limber normalisation, l vs l+1/2) belongs to subsystem 03 and is not re-derived here.

### 1.3 Outputs

**nicaea `cl_to_xi_interface.c`** (default `corr_type=0`, `filter_type=0`, `cl_to_xi_interface.c:39-41`):

| output section | keys | shape / unit |
|---|---|---|
| `shear_xi_plus` | `theta`; `bin_i_j` | `theta`: (1024,) radians, ascending, log-uniform; `bin_i_j`: (1024,) xi_+ for each (i,j) present in the input |
| `shear_xi_minus` | same layout | xi_- |
| `galaxy_shear_xi` (corr_type=2) | `theta`, `bin_i_j` | gamma_t, (1024,) |
| `galaxy_xi` (corr_type=1) | `theta`, `bin_i_j` | w(theta), (1024,) |
| with `filter_type=1` | `shear_map_map2`, `galaxy_map_nap2`, `galaxy_shear_map_mapnap2` ([verified correction]: the draft dropped the `_map` stem; names are `<output_section>_<suffix>` with default `output_section` = `shear_map`/`galaxy_map`/`galaxy_shear_map`, `cl_to_xi_interface.c:203-215`) | aperture statistics (not used by kcap) |
| all | `nbin_a`, `nbin_b`, `is_auto`, `sample_a`, `sample_b`, `save_name`, `sep_name="theta"`; `theta` metadata unit = `radians` | |

Evidence: section naming `cl_to_xi_interface.c:203-246`, array saves `:342-349`, theta `:361-372`, plus/minus metadata `:375-386`. QUIRK: `module.yaml` documents `xiplus_i_j / ximinus_i_j` keys; the code actually writes `bin_i_j` in the `*_plus` / `*_minus` sections.

**`xipm_binned.cc`** (`libxipm_binned.so`): reads `shear_xi_plus`/`shear_xi_minus` (`type = plus|minus`), writes `output_section_name` (default `xi_binned_plus|minus`; kcap uses `shear_xi_plus_binned`) with `bin_i_j` (`nTheta`,) and `theta_bin_i_j` (`nTheta`,) in **arcmin**, plus `nbin_a`, `nbin_b` (`xipm_binned.cc:862-872`).

**`COSEBIs_cosmosis_interface.cc`** (`libcosebis.so`): section `cosebis`: `bin_J_I` (n_max,) for each existing C(l) pair, `cosebis_n` = int array 1..n_max, `theta_min`, `theta_max` (written as given in the ini, i.e. ARCMIN, `:402-403`), `b_modes`, `nbin_a`, `nbin_b` (written as doubles, `:627-628`). Optional likelihood key `cosebis_like` = -chi^2/2 in section `likelihoods` (`:635-664`).

**`BandPower_interface.cc`** (`libbandpower.so`): section `output_section_name` (kcap: `bandpower_shear_e`, `bandpower_galaxy_shear`): `bin_i_j` (nBands,), `index_bin_i_j` (int 1..nBands), `l_min_vec`, `l_max_vec` (nBands,), `b_modes`, `theta_min`, `theta_max` (these two in RADIANS, `:113-115`, `:574-575`), `nbin_a`, `nbin_b` (`:565-577`). `modules/scale_cuts/scale_cuts.py:140-144` ([verified correction]: draft cited `:124-128` and omitted the directory) takes the effective band ell as `10**(0.5*(log10 l_min + log10 l_max))`, i.e. the geometric centre of the band edges. [verified] The same file reads xi theory angles from the key `theta_bin_1_1` (written by `xipm_binned`), the COSEBIs mode index from `cosebis_n`, and for non-GGL spectra assembles pairs as `bin_{j+1}_{i+1}` with j >= i (`:118-122`, `:163-171`), i.e. the same smaller-index-slowest order as Q11.
- [verified] Band-power default input sections when `input_section_name` is absent: `clustering` -> `galaxy_cl`, `ggl` -> `galaxy_shear_cl`, `cosmic_shear_e/b` -> `shear_cl` (`BandPower_interface.cc:331-337, 355-361, 388-394, 424-430`). Defaults if keys are missing: `theta_min = 1` arcmin, `theta_max = 100` arcmin (`:92-111`), `l_min = 100`, `l_max = 1000` (`:135-151`), `Response_function_type = tophat`, `Delta_x = 0.5`, `Analytic = true`, `Output_FolderName = ./BandPower_weights` (`:188-197`). If `nBands` is missing the log message claims a default of 10 but `config->nBands` is left UNINITIALISED (`:129-133`); if `Apodise` is missing, `noApodise` stays true but `if(Apodise)` at `:236` then reads an uninitialised int. `type` missing -> `config->type = 0` (clustering) WITHOUT creating `BP0`, so `execute` would dereference a null/garbage pointer (`:309-314`). `is_it_bmodes` (default 0) is only echoed to the output as `b_modes`; it changes no computation (`:175-184, 573`).

### 1.4 Pair ordering (matters when reproducing data vectors)

- nicaea: loops i=1..nbin_a, j=1..nbin_b, writes `bin_i_j` for those present (`cl_to_xi_interface.c:251-261`) -> lower-triangular `j<=i` for auto spectra.
- COSEBIs: loops `i_bin` outer, `j_bin` inner but READS and WRITES the key `bin_{j_bin}_{i_bin}` (`COSEBIs_cosmosis_interface.cc:471`). With C(l) stored for j<=i this means the produced order is, for i_bin=1: bin_1_1, bin_2_1, ..., bin_N_1; then i_bin=2: bin_2_2, bin_3_2, ... So the data vector order is "(i<=j) with the smaller index slowest": pairs (1,1),(1,2)...(1,N),(2,2),... each carrying n_max modes. Within a pair the n index is fastest.
- Band powers: loops `bin_i_j` with i outer, j inner (`BandPower_interface.cc:505-513`) and `scale_cuts.py` re-assembles by key name.
- [verified] `xipm_binned.cc` (`:697-870`) loops i outer / j inner over the keys `bin_{i}_{j}` that exist, i.e. lower triangle order (1,1),(2,1),(2,2),(3,1),(3,2),(3,3),... and increments a counter `pair` per hit. `mini_2pt_like.py` has two layouts: `order_cov = xi_pm-bin-theta` loops i, then j<=i and reads `bin_{i+1}_{j+1}` (same lower-triangle order, `:164-171`), while `order_cov = montepython` (the KV450 fiducial) loops i, then j>=i and reads `bin_{j+1}_{i+1}` (`:203-210`), i.e. (1,1),(2,1),(3,1),(4,1),(5,1),(2,2),(3,2)... which equals the COSEBIs / scale_cuts order. The data-vector index for that layout is `(j*(2n-j+1))//2 + i-j` (`:126`).
- [verified] POSSIBLE BUG (not executed): `xipm_binned` builds the pair-count file list with i outer and j from i (upper-triangle names `Bin{i}_Bin{j}`, (1,1),(1,2),(1,3),...; `:508-519`) but indexes it with the lower-triangle counter `pair` (`:739-746`). From the third spectrum on (counter 2 -> file (1,3) but theory pair (2,2)) the Npair weights would belong to a different redshift pair. Only reachable when `InputNpair` is given; no kcap config does this.

---------------------------------------------------------------------------------------------------

## 2. Mathematical description

Notation: theta in radians unless stated; `arcmin = pi/(180*60)` (`cosebis/modules/globaldef.h:35`); J_mu = Bessel function of the first kind.

### 2.1 Textbook transforms (what the code is meant to compute)

```
xi_+(theta)   = 1/(2 pi) Int_0^inf dl  l  C_EE(l)  J_0(l theta)
xi_-(theta)   = 1/(2 pi) Int_0^inf dl  l  C_EE(l)  J_4(l theta)
gamma_t(theta)= 1/(2 pi) Int_0^inf dl  l  C_gE(l)  J_2(l theta)       (corr_type=2)
w(theta)      = 1/(2 pi) Int_0^inf dl  l  C_gg(l)  J_0(l theta)       (corr_type=1)
```
(`cl_to_xi_nicaea/module.yaml` explanation; the C code uses (mu, norm) = (0, 2pi), (4, 2pi), (2, 2pi), (0, 2pi), `maths.c:1920-1930`, `:1956`.) C_BB would enter xi_+ and xi_- with opposite signs; kcap passes only E-mode C(l), so xi_+/- contain no B contribution.

### 2.2 nicaea `tpstat_via_hankel`: a log-FFT (FFTLog without bias) evaluation

This is the only xi transform used by the fiducial runs. Reading `maths.c:1877-2008`.

**Grid** (all hard-coded):
```
N      = 1024   (N_thetaH, maths.h:38)
l_min  = 1e-4,  l_max = 5e6   (maths.h:34-35)
Dl     = [ln(l_max) - ln(l_min)] / (N-1)  = 0.0240814163
lnrc   = 0.5 [ln l_max + ln l_min]
nc     = N/2 + 1 = 513            (maths.c:1906)
l_i    = exp( lnrc + (i - nc) Dl ),  i = 0..1023
```
Numerically l_0 = 9.645e-5, l_1023 = 4.8226e6 (computed, not read). QUIRK: because `nc` is 513 and not (N-1)/2, the grid is shifted by 1.5 steps: it does not start exactly at l_min nor end at l_max.

**Forward transform.**
```
f_i = l_i * C(l_i)                       (maths.c:1911;  C from the interpolation routine of section 2.3)
F_m = sum_i f_i exp(-2 pi i m i_idx / N),  m = 0..N/2         (FFTW r2c, unnormalised)
k_m = 2 pi m / (Dl N)
```
**Kernel** (q = 0 for xi+-, w, gt; `maths.c:2016-2038`):
```
K_mu(k) = 2^{ i k }  Gamma( (1+mu+ i k)/2 ) / Gamma( (1+mu - i k)/2 )
```
implemented as `exp(i k ln2) * g1 / g2` with `g1 = Gamma(0.5(1+mu+q) + i k/2)`, `g2 = Gamma(0.5(1+mu-q) - i k/2)`, `pref = 2^q`; for q=0, `pref=1`. (This is the Mellin transform of J_mu: Int_0^inf x^{s-1} J_mu(x) dx = 2^{s-1} Gamma((mu+s)/2)/Gamma((mu-s)/2+1) evaluated at s = 1 + i k.) Complex Gamma is the in-house `cdgamma` (`maths.c:1185`, Lanczos-type with constant 6.00009857740312429).

QUIRK: `ln2` is `#define`d as `0.69314718` (8 digits, `maths_base.h:53`) versus 0.693147180559945. The phase error is k * 5.6e-10, at most ~7e-8 rad at Nyquist. Irrelevant physically; matters only if you want to match kcap to 1e-8.

**Convolution and inverse** (`maths.c:1958-1998`):
```
conv_m = F_m * K_mu(k_m);  Im conv_0 := 0;  Im conv_{N/2} := 0
r_i    = FFTW c2r(conv)_i                  (unnormalised: = N * numpy.fft.irfft)
t_i    = exp( (nc - i) Dl - lnrc ) = 1 / l_i
xi[N-1-i] = r_i / ( t_i * N * 2 pi ) = l_i * irfft(conv)_i / (2 pi)
```
Hence the OUTPUT grid is `theta_j = 1 / l_{N-1-j}`, i.e.
```
theta_j = exp( ln theta_min + j Dl ),   j = 0..1023
ln theta_min = (nc - N + 1) Dl - lnrc,   ln theta_max = nc Dl - lnrc          (maths.c:2001-2002)
```
= 2.0736e-7 rad (0.000713 arcmin) to 1.0368e4 rad (computed). The interface re-derives `theta_vals[i] = exp(log_theta_min + i*dlog_theta)` with `dlog_theta=(max-min)/(N-1)` (`cl_to_xi_interface.c:361-366`), identical to the above. Index range for 0.5'-300' is j=273..537 (265 points, computed).

Interpretation: xi(theta_j) = (1/(2 pi theta_j)) Sum_i Dl f_i [x J_mu(x)]_{x = l_i theta_j} with the sum done spectrally, so it is a circular (periodic in ln l, period N*Dl = 24.66) convolution, exact for band-limited data; the kernel is truncated at the Nyquist frequency pi/Dl.

xi_-: same f and F, kernel with mu=4 (`maths.c:1956`). gamma_t uses mu=2. The aperture-mass option (`tp_map2_poly`: q=-4, mu=4, `norm=2 pi/24/24`, kernel `hankel_kernel_mumu`) and the unused `tp_gsqr`, `tp_map2_gauss`, `tp_xir` exist in the same function; not used by kcap and not verified by me.

**VERIFIED-NUM** (`04_verification/nicaea_logfft_hankel_check.py`): numpy `rfft`, the kernel above built from `scipy.special.loggamma`, `irfft`, and the index reversal reproduce a direct trapezoid evaluation of the integral for a toy positive C(l) for mu=0 and mu=4. At theta = 1' and 10' the agreement is ~1e-7 relative (xi_+(1') = 0.0943430 vs 0.0943430); at 100' the absolute difference is ~4e-7 of xi(1'), consistent with the quality of my reference quadrature (which is poor for highly oscillatory integrands); I did not separate FFT error from reference error there. [verified correction] I re-ran the script. Precisely: mu=0, 1': 0.09434304181 (FFT) vs 0.09434304107 (quad), 8e-9 relative; mu=0, 10': -5.95835e-4 vs -5.95889e-4, i.e. 9e-5 relative but only 5.5e-8 absolute (xi_+ of this toy spectrum is near a zero crossing there, so "~1e-7" holds only relative to xi(1'), where it is 6e-7, not relative to the local xi(10')); mu=0, 100': -6.5246e-6 vs -6.8994e-6 (abs diff 3.7e-7 = 4e-6 of xi(1'), 6% relative); mu=4: 1': 5.0145427e-3 vs 5.0145397e-3 (6e-7 rel), 10': 1.27333828e-2 vs 1.27333284e-2 (4e-6 rel), 100': 9.8012e-5 vs 9.7637e-5 (4e-3 rel). So "~1e-7 relative" should be read as "6e-7 (1') and 4e-6 (10') relative to the local xi for mu=4, 8e-9 at 1' for mu=0, and <= 6e-7 of xi(1') in absolute terms at 10' for mu=0", and the 100' numbers are limited by the reference quadrature (not separated from FFT error).

### 2.3 Interpolation / extrapolation of C(l) inside the transform

Read `cl_to_xi_interface.c:187-189, 267-339` and `maths.c:1601-1650`.

1. The ell array is assumed log-uniform: only `ell[0]`, `ell[n-1]` are read and `dlog_ell = (ln ell[n-1] - ln ell[0])/(n_ell-1)` is used as the table spacing. QUIRK: a non-log-uniform `ell` is silently mishandled.
2. Count `neg_vals` = number of entries with `C <= 0` (zero counts as non-positive).
3. Entries with `|C| <= 1e-30` are replaced by `+-1e-30` only if all entries have the same sign (all positive, `neg_vals==0`; or all `<=0`, `neg_vals==n_ell`).
4. Cases:
   - all positive: table = `ln C` vs `ln l`, linear interpolation, `C = exp(table)`, extrapolation slope `lower = +1.0` below `ell[0]` and `upper = -3.0` above `ell[n-1]` in (ln l, ln C), i.e. `C ∝ l^{+1}` below and `C ∝ l^{-3}` above. (`interTable(... 1.0, -3.0)`, lines 288-289.)
   - all non-positive: same with `ln(-C)`, sign restored after the transform (`:294-336`).
   - mixed sign: table = C itself, linear in `ln l`, `lower=upper=0` (`:312-313`), "no extrapolation" per the printed warning (`:305-309`).
5. QUIRK (bug): the extrapolation code decides "extrapolation disabled" with `fabs(lower) < EPS` where `#define EPS 1.0-5` (`maths.c:1617`). The macro expands to `1.0-5 = -4.0`, so the test is never true. For `lower=upper=0` the routine therefore does NOT raise the documented error; it returns `f[0]` for `x<a` and `f[n-1]` for `x>b` (constant extrapolation of C, not zero) (`maths.c:1620-1640`). For the all-positive case with slopes 1 and -3 behaviour is as intended.
6. Within the table: piecewise-linear in (ln l, ln C), i.e. piecewise power law between the 400 input points (spacing in ln l: 0.0387 for the KV450 grid 0.1..5e5 [verified correction: ln(5e6)/399 = 0.03866, draft said 0.0388]; 0.0258 for the KiDS-1000 grid 1..3e4).

Why the slopes: f = l*C behaves as l^2 at the low edge and l^-2 at the high edge of the 1024-grid, i.e. it vanishes at both ends, which is what the periodic convolution needs.

### 2.4 Theta binning of xi (bin-averaged vs point)

nicaea returns POINT values at 1024 log-spaced theta; it never bin-averages. Binning is done downstream by one of two consumers.

**(a) `libxipm_binned.so`** (`cosebis/xipm_binned.cc`). The input xi (`bin_i_j`, theta in radians) is loaded into a cubic spline in `ln(theta_arcmin)` (`function_cosebis::loadWithValues(logtheta, xi, true)`, `:629,711`; spline = GSL natural cubic, `function_cosebis.cc:770-777`); outside the tabulated range it returns 0 (extrapolation off, `function_cosebis.cc:354,263-272`). Output bins, three modes:
 1. `weighted_binning=1` with edges from `theta_min, theta_max, nTheta` (no Npair file): edges `e_b = theta_min (theta_max/theta_min)^{b/nTheta}`, `b=0..nTheta` (`:243-249`); per bin 100 log sub-bins, sample points `theta_k = tmin (tmax/tmin)^{(k+0.5)/100}`, weights `w_k = theta_k * dtheta_k` with `dtheta_k = tmin (tmax/tmin)^{(k+1)/100} - tmin (tmax/tmin)^{k/100}` (`:723-736`):
    ```
    xi_bin = Sum_k w_k xi(theta_k) / Sum_k w_k      ≈  Int theta xi dtheta / Int theta dtheta
    ```
 2. `weighted_binning=1` with `InputNpair` files: weights are the pair counts from the file at the file's theta points between the indices closest to the bin edges (inclusive on both ends, `find_closest_index`), per redshift pair (`:504-558, 739-751`). Used for data-matched bins.
 3. `weighted_binning=0`: point evaluation at bin "centres" `theta_i = exp(ln tmin + ln(tmax/tmin)/n (i+0.5))` (geometric centre of the edges, `:236`).
 For KiDS-1000 settings the bins are `nTheta=9` log bins between 0.5' and 300' (`run_kcap.py:1097-1099`).

QUIRK / probable bug: in `utils/run_kcap.py` the dict entries for `bin_xi_plus`, `bin_xi_minus`, `bin_xi_ggl` set `"theta_max": xi_theta_min` (lines 592, 611, 634). With theta_min == theta_max the edges collapse (`ln(theta_max/theta_min)=0`), all `dtheta_k=0`, so the weight sum is 0 and the bin value is 0/0. I did not execute it, so I cannot say whether a downstream file overrides it; the `xipm` run type is the only user. [verified] Re-checked: `xi_theta_max` is defined (`:1098`) and passed into `create_config` (`:375, 1195`) but grep finds no other use of it; no override in the dict (`:591-593, 610-612, 633-635`). In `xipm_binned.cc` the edge construction (`:243-251`) then yields `theta_low == theta_high` for every bin, `deltaTheta = 0` for every sub-bin (`:729`), `weight_mat.sum() = 0`, so `weightedTheory = 0/0 = NaN` (`:735`) -- a NaN, not a silent zero, because the division is outside `function_cosebis::value` (which only maps NaN to 0 internally). By code reading only; not executed.

[verified] Behaviour of `xipm_binned` when pair-count (Npair) weighting is requested but unavailable: `InputNpair` missing, `nBins_in` missing, or any Npair file missing all fall back to THETA-weighted 100-sub-bin averaging (`weight_theta_bins_by_theta` stays/becomes true, `:410-413, 450-452, 483-485, 552-557`), NOT to point evaluation. Point evaluation at the geometric centres happens only for `weighted_binning = 0`, or if the user supplies `theta_file_name` / `theta_list` and no `theta_min_max_filename` (no edges known, `:389-420`). Other defaults: `weighted_binning = 1`, `theta_min = 1`, `theta_max = 100`, `nTheta = 10` (arcmin, `:203-231`), `type = plus`, `x_name = theta`. The theta axis of the incoming xi is converted `ln(theta/arcmin)` only when `x_name == "theta"` (`:626-635`).

[verified] `xipm_binned` constant-c and 2D-c additions (`:639-690, 765-852`): for `type=plus` and `add_c_term>0`, `delta_c = c1^2 + c2^2` if both `c1`,`c2` are in `shear_c_bias`, otherwise `delta_c` is taken from the block as is (NOT squared) and added to every bin of xi_+; for `type=minus` it adds `(c1^2 - c2^2) cos4phi_binned + 2 c1 c2 sin4phi_binned` (vectors from files or from sections `xim_cos4phi`/`xim_sin4phi`, one nTheta-long vector per pair); `add_2D_cterm` adds `Ac * xi_2D` with a linear `Ac` (the source carries the comment "Should this be Ac or Ac^2", `:850`). The line `config->input_2D_section_name + string("_") + config->type;` (`:283`) is a no-op statement, so the documented suffix `_plus/_minus` is never appended to the 2D section name.

**(b) `utils/mini_2pt_like.py` "integrate" mode** (KV450 fiducial). `get_theory_point` (`:11-35`): for each data bin `[edge_i, edge_{i+1})` select the native nicaea theta points with `edge_i <= theta < edge_{i+1}` and compute `trapz(theta*xi, theta)/trapz(theta, theta)` on those points only (`:17-26`, `weighting=theta`). The bin edges are `angular_bin_edges` in arcmin converted to radians (`:60-63`); KV450 uses 9 log-spaced bins between 0.5' and 300' (`KV450_fiducial.ini:191`). QUIRK: the trapezoid integrates only over native grid points inside the bin; the end segments between the first/last point and the edges are dropped (the grid step in ln theta is 0.024 so ~29 points per bin at 0.71 ln-width [verified correction: the draft said 0.5; ln(600)/9 = 0.711], error O(dlntheta) at the edges, not interpolated). A cubic `InterpolatedUnivariateSpline` is also built on every call (`:12`) but only used in `interpolate` mode.

[verified] Other `mini_2pt_like.py` behaviour that the draft omitted (all by code reading): (i) Constant additive-bias offset: `order_cov = xi_pm-bin-theta` adds `delta_c` (`:187`) to every xi_+ bin of every pair, whereas `order_cov = montepython` -- the layout the KV450 fiducial uses -- adds `delta_c**2` (`:226`). The two layouts therefore disagree by a square for the same ini parameter; the KV450 value range is `delta_c in [-6e-4, 6e-4]` (`KV450_values.ini:36`), so the fiducial offset is <= 3.6e-7. xi_- gets no offset (the lines are commented out, `:188, 227`). (ii) With `xi_pm_c_file` both xi_+ and xi_- receive `A_c**2 * xi_c` (0-based columns 3 and 4 of the file, `:104, 190-191, 229-230`); KV450 values `A_c in [0.62, 1.13, 1.4]`. (iii) The multiplicative bias is applied to the DATA, not the theory: data / `((1+m_i)(1+m_j))` (`:140-141`). (iv) Scale cuts delete bins by index from data, theory and covariance (`cut_xi_plus = 7 8`, `cut_xi_minus = 0 1 2` in the fiducial, `:52-58, 73-95, 143-144`, indices into the 9 theta bins). (v) The covariance is inverted explicitly once (`np.linalg.inv`, `:99`); chi^2 = `d^T C^-1 d` with `like = -chi2/2` (`:238-245`); with `keep_theory_vector` it draws a `np.random.multivariate_normal` every call (`:251`), which costs a Cholesky/SVD of the full covariance per likelihood evaluation.

### 2.5 COSEBIs

#### 2.5.1 Definitions

Notation: `z = ln(theta/theta_min)`, `Delta = ln(theta_max/theta_min)`.

```
E_n = 1/2 Int_{tmin}^{tmax} dtheta theta [ T_+n(theta) xi_+(theta) + T_-n(theta) xi_-(theta) ]
B_n = 1/2 Int            dtheta theta [ T_+n xi_+ - T_-n xi_- ]
E_n = 1/(2 pi) Int_0^inf dl l W_n(l) C_E(l),       W_n(l) = Int_{tmin}^{tmax} dtheta theta T_+n(theta) J_0(l theta)
```
(`COSEBIs.h:8-16`; the C(l) route `COSEBIs.cc:362`: `result/2/pi`). **VERIFIED-NUM** (`cosebis_En_cl_vs_xi_check.py`): for a toy C(l), E_n (n=1,2,3) from the W_n route and from the xi route agree to ~1e-4 relative (6.62298e-8 vs 6.62357e-8, 3.0488e-8 vs 3.0501e-8, -5.1204e-8 vs -5.1194e-8) [verified: script re-run reproduces these numbers; the actual relative differences are 9e-5, 4.0e-4 and 1.9e-4, so "~1e-4" is really 1e-4 to 4e-4], which confirms the 1/(2 pi), the factor 1/2, J_0/J_4 and the T_- relation below.

#### 2.5.2 Log-radian filter functions T_+n (log COSEBIs)

Read: `T_+n(theta) = norm_n * Prod_{i=1}^{n+1} (z - r_{n,i})`, a polynomial of degree n+1 in z with the n+1 roots `r_{n,i}` (`TpnLogRadian.cc:86-95`, `WnLog.cc:151-160`). Roots and `norm_n` are NOT computed by kcap; they are read from `cosebis/TLogsRootsAndNorms/Root_<tmin>-<tmax>.table` and `Normalization_<tmin>-<tmax>.table`, where the name uses `toString(theta/arcmin, 2)` (two decimals, e.g. `Root_0.50-300.00.table`) (`WnLog.cc:46-49`, `TpnLogRadian.cc:34-37`). File formats: root file line n = `n  r_{n,1} ... r_{n,n+1}` (8 significant digits, `Root_0.50-300.00.table`); norm file line n = `n  norm_n` (50 digits). The program calls `exit(1)` if the file for the requested (theta_min, theta_max) does not exist (`WnLog.cc:52-64`); about 30 ranges are shipped. The 0.50-300.00 file has 19 lines (n up to 19 or 20). [verified correction] Counted: 29 `Root_*.table` files; 27 have a matching `Normalization_*.table` (`Root_1.00-1000.00.table` and `Root_1.00-4000.00.table` have none and would hit the `exit(1)`). The 0.50-300.00 file has 20 mode lines (19 newline characters; the last line has no trailing `\n`, hence `wc -l` prints 19), so n_max <= 20 there. The number of modes (true line counts) differs per range: 10 for 0.20-200, 0.20-40, 1.00-200, 1.00-300 and 20.00-40; 20 for most others (including every 0.50-xxx range); 30 for the two Root-only files; 40 for 1.00-20, 1.00-400 and 20.00-400. `nMax` larger than the file's line count aborts with `exit(1)` ("nMax is too big", `WnLog.cc:68-72`). A file named `Root_0.5-300.table` (no 2-decimal form) is shipped but can never be opened by the code, which always formats `%.2f`. Shipped `Tplus_minus/T{plus,minus}<n>_0.50-100.00.table` cover only theta 0.5'-100', n = 1..20, 10^4 log-spaced points (theta in arcmin vs T; first row n=1: 0.5 -> 9.10105643).

**Defining conditions (recovered by me; VERIFIED-NUM against the shipped tables):**
1. T_+n(z) is a polynomial of degree n+1 in z.
2. `Int_0^Delta dz e^{2z} T_+n = 0` and `Int_0^Delta dz e^{4z} T_+n = 0`, i.e. `Int dtheta theta T = 0` and `Int dtheta theta^3 T = 0` (E/B separability conditions). Measured on the shipped n=1..5, 0.5-300 table: relative moments 5e-8..3e-7 (consistent with 8-digit roots).
3. Orthonormal in the plain `dtheta` measure: `Int_{tmin}^{tmax} dtheta T_+n T_+m = (theta_max - theta_min) delta_nm`; in z this is `Int_0^Delta dz e^{z} t_n t_m = (e^Delta - 1) delta_nm`. Measured: diagonal values 599.0000 (= 300/0.5 - 1), 99.0000 (1-100 table), 199.0000 (0.5-100 table); off-diagonals vanish (<1e-5 in the weight-e^z test). [verified] Independent numpy re-check on the shipped 0.50-300.00 and 1.00-100.00 tables (400-point Gauss-Legendre in z, n = 1..5): the Gram matrix of `Int dtheta T_n T_m` is diagonal with entries 299.5 and 99.0, i.e. exactly `theta_max - theta_min` for theta in ARCMIN (and the same function gives 0.0872 rad for theta in radians; `T_n(z)` itself is dimensionless). The two forms of condition 3 are therefore consistent but differ by a factor theta_min: `Int dtheta T T = theta_min * Int dz e^z T T`, which is why the z-form diagonal is 599 = 299.5/0.5 for the 0.5' tables. The lowest two moments `Int dtheta theta^1 T_n` and `Int dtheta theta^3 T_n` vanish to 5e-8..4e-7 relative (n=1..4); all norms are positive.
4. Sign: leading coefficient `norm_n > 0`.

Constructive recipe: for n = 1,2,...: unknown polynomial of degree n+1; impose conditions 2 (two linear constraints) and orthogonality to the previously found T_+1..T_+(n-1) with weight e^z (n-1 constraints); this fixes the polynomial up to scale; fix the scale by condition 3 and the sign by condition 4. `04_verification/cosebis_roots_norms_from_scratch.py` implements this with mpmath (60 digits) and reproduces the shipped roots and norms for n=1..4, [0.5', 300'] (e.g. n=1 roots 5.0883097, 6.2059688, norm 0.37596958385...). Use high precision (mpmath or exact moments): the monomial basis is ill-conditioned for n > ~8 in double.

**T_-n** (`TpnLogRadian.cc:15-21, 124-153`):
```
T_-n(z) = T_+n(z) + Int_0^z dy  4 T_+n(y) [ e^{2(y-z)} - 3 e^{4(y-z)} ]
```
(equivalently `T_-(t) = T_+(t) + 4 Int_{tmin}^{t} dt'/t' T_+(t') (t'^2/t^2 - 3 t'^4/t^4)`). Integrated with 20-point Gauss-Legendre between 0, the roots of T_+n smaller than z, and z. VERIFIED-NUM as part of the E_n check above.

#### 2.5.3 W_n(l) as computed by kcap

`WnLog.cc:143-149, 183-209`:
```
W_n(l) = (1/l^2) Int_{l tmin}^{l tmax} dx  x  T_+n(x/l) J_0(x)
```
- for `l < lthresh = (2 pi / theta_max) * n / 10`: one Gauss-Legendre rule with `10*n` points over `[l tmin, l tmax]`.
- for `l >= lthresh`: Gauss-Legendre with `accuracyG = 40` points on every sub-interval between consecutive "steps" `StepN` = sorted union of the zeros of J_0 (table `modules/besselzero04.h`) and the values `theta_min e^{r_{n,i}}` (see QUIRK Q15 below), up to `LHIGH*theta_max` (`StepFinder`, `WnLog.cc:211-248`).
- tabulated: `loadTable(LLOW=1, LHIGH, NLBINS=1e6, logarithmic=true)` -> 10^6 log-spaced points in `[1, LHIGH]`, stored in `<Wn_Output_FolderName>/WnLog<n>-<tmin>-<tmax>.table` and re-read on later runs (`WnLog.cc:97-136`, `WnLog.h:10,16,19`). Evaluated by cubic spline in `ln l`; outside `[LLOW, LHIGH]` it returns 0 (extrapolation off).
- `LHIGH = high * 20 / theta_max` with `high = 500`, i.e. `LHIGH = 10^4 / theta_max[rad]` (`WnLog.h:11`, `WnLog.cc:42`, `COSEBIs.cc:56`). Examples (computed): theta_max = 300' -> 1.146e5; 100' -> 3.438e5.

#### 2.5.4 E_n from C(l) (the kcap likelihood route)

`COSEBIs.cc:346-363` and `:313-343`:
```
E_n = 1/(2 pi) [ Int_{LLOW}^{lthresh_E}  dl l W_n(l) C(l)           (single 100-point GL; only if LLOW < lthresh_E)
               + Sum_s Int_{l_s}^{l_{s+1}} dl l W_n(l) C(l) ]       (20-point GL on each segment)
lthresh_E = (2 pi / theta_max) n / 12 ;   LLOW = 1 ;   upper limit LHIGH
```
The segment edges `l_s` start at `lthresh_E` and are the locations of all strict local extrema of the tabulated W_n found on a 10^6-point log grid between `lthresh_E` and `LHIGH`, plus `LHIGH` (`determine_integration_limits_En`, `COSEBIs.cc:313-343`, done once per run because `EnInteglimitSet`).

C(l) enters through `COSEBIs::setPower_single_withExtrapolation(ln ell, C)`: 
- GSL natural cubic spline of C (linear C, not log C) vs `ln l`, over the supplied points (`function_cosebis::loadWithValues(...,true)` + `prepareLookupTable`, `function_cosebis.cc:674-702, 770-777`);
- outside the supplied range: log-log linear extrapolation using the last two points above, but the points with indices **1 and 2** (not 0 and 1) below (`function_cosebis.cc:347-376`); NaN results (log of non-positive C) are replaced by 0 (`:272`).

`n_max` modes (kcap: 5, `run_kcap.py:1119`), one vector per C(l) pair.

[verified] Lifecycle and range facts for this route: (i) `setup()` already calls `cosebis->setWns(n_max)` (`COSEBIs_cosmosis_interface.cc:380`), so the W_n tables are built or read from disk at module set-up, not on the first likelihood call; the 10^6-point extremum scans for the E_n segments run at the first `calEn` (`COSEBIs.cc:370-371`, flag `EnInteglimitSet`). (ii) `COSEBIs::initialize` multiplies the ini `theta_min/theta_max` (arcmin) by `arcmin` (`COSEBIs.cc:30`); defaults if absent are theta_min = 1', theta_max = 100' (`COSEBIs_cosmosis_interface.cc:130-149`). (iii) In the KiDS-1000 `run_kcap.py` configuration the projected C(l) stops at `ell_max = 3e4` but `LHIGH = 1.146e5` for 300', so about 0.58 decades of the E_n integral rely on the log-log extrapolation from the last two tabulated points (`function_cosebis::extrapolate`, which has no clamp on the slope and returns 0 for negative or NaN C); the KV450 grid (5e5) covers LHIGH. nicaea, by contrast, extrapolates the same C(l) with the fixed slope -3, so the xi route and the COSEBIs route of the same pipeline do not share an extrapolation assumption (see Q12).

#### 2.5.5 Constant c-term and 2D c-term corrections

Applied to the theory E_n after the integral (`COSEBIs_cosmosis_interface.cc:507-610`):
```
E_n  +=  (c1^2 - c2^2) E_n^{cos4phi}  +  2 c1 c2 E_n^{sin4phi}  +  A_c E_n^{2D}
```
`c1, c2, Ac` from section `shear_c_bias` [verified: the exact key names are `c1`, `c2`, `Ac` (`COSEBIs_cosmosis_interface.cc:513, 520, 570`), whereas the KV450 route (`mini_2pt_like.py`, `utils/shear_c_bias.py`) uses `delta_c` and `A_c` in the same section; the two naming schemes are not interchangeable, and the c-term enters COSEBIs linearly in `Ac` but through `A_c**2` in `mini_2pt_like`]; `E_n^{cos4phi}`, `E_n^{sin4phi}`, `E_n^{2D}` are pre-computed vectors read from files (concatenated pair-by-pair, n fastest; offset advances by n_max per pair) or from datablock sections `cosebis_cos4phi`, `cosebis_sin4phi`, `cosebis_2D`. The module also computes `-chi^2/2` against an input data vector and covariance (`calChiS`: explicit matrix inverse, `calChiS.cc`).

#### 2.5.6 Gaussian covariance (separate module, for completeness)

`COSEBIs::valueCov` (`COSEBIs.cc:894-910`, `calCov` `:1176-1229`): 
`Cov[E_n^{ab}, E_m^{cd}] = 1/(2 pi A) Int dl l W_n W_m [ (C_ac + d_ac N_a)(C_bd + d_bd N_b) + (C_ad + d_ad N_a)(C_bc + d_bc N_b) ]` with `N_a = sigma_e^2/(2 nbar_a)` (`setNoise`, `:96-107`), A = survey area, lower limit `pi/(2 theta_max)` (20-pt GL from LLOW to there) then segment edges from extrema of `W_n W_m` on a 10^6-point grid recomputed on every call (`determine_integration_limitsCov`). [verified] Units and conventions in `COSEBIs_covariance_cosmosis_interface.cc`: `Area` is given in deg^2 and converted by `pi/180` squared to steradian (`:307`); `ngal_effective` in arcmin^-2 is divided by `arcmin^2` to give sr^-1 (`:281, 293`); the supplied `sigma_e` is multiplied by `sqrt(2)` on reading (`:246, 258`), so with `N = sigma_e^2/(2 nbar)` the ini value is the per-component dispersion and the effective noise is `sigma_eps^2/nbar`.

### 2.6 Band powers

Source: `BandPower.h:13-56` (comments), `BandPower*.cc`.

**Definition.** For a band i with response `S_i(l)` (kcap: top-hat, `S=1` for `l_min_i <= l < l_max_i`, `BandPower_W.cc:152-171`):
```
g_mu^i(theta) = Int_0^inf dl l S_i(l) J_mu(l theta)
W_mu^i(l)     = Int_{tmin}^{tmax} dtheta theta T(theta) J_mu(l theta) g_mu^i(theta)
N_i           = Int dl/l S_i(l) = ln(l_max_i / l_min_i)                      (BandPower.cc:365-376)
BP_mu^i       = (1/N_i) Int dl l C(l) W_mu^i(l)
              = (2 pi/N_i) Int dtheta theta T(theta) g_mu^i(theta) corr_mu(theta)    [real-space form]
```
Without a finite theta range W^i -> S_i and BP^i -> (1/N_i) Int dln l l^2 C S_i, i.e. a log-average of l^2 C(l) (not l(l+1)C/2pi) over the band (comment `BandPower.h:30`).

**Analytic top-hat g** (used always in the Fourier-space path actually executed, see Q17). With `x_u = l_max theta`, `x_l = l_min theta` (`BandPower_g.cc:268-303`, `BandPower_W.cc:469-504`):
```
g_0(theta) = (1/theta)  [ l_max J_1(x_u) - l_min J_1(x_l) ]
g_2(theta) = -(1/theta^2) [ x_u J_1(x_u) - x_l J_1(x_l) + 2 J_0(x_u) - 2 J_0(x_l) ]
g_4(theta) = (1/theta^2)  [ G4(x_u) - G4(x_l) ],     G4(x) = (x - 8/x) J_1(x) - 8 J_2(x)
```
I checked algebraically that these equal `theta^-2 [Int x J_mu dx]` from `l_min theta` to `l_max theta`.

**Combinations** (`BandPower_interface.cc:521-556`, `BandPower.h:35-47`):
```
GGL (type "ggl"):            BP = BP_2
clustering ("clustering"):   BP = BP_0
cosmic shear E ("cosmic_shear_e"):  BP_E = ( BP_0 + BP_4 ) / 2    (both computed from the single input C(l))
cosmic shear B ("cosmic_shear_b"):  BP_B = ( BP_0 - BP_4 ) / 2    (QUIRK Q18: computed from the same E-mode C(l); B-mode C(l) does not exist in the datablock, see comment :422)
```
Band edges when no `l_min_max_file` is given: `l_b = l_min (l_max/l_min)^{b/nBands}` for `b=0..nBands`, i.e. log-uniform (`:153-160`). KiDS-1000: `l_min=100, l_max=1500, nBands=8` -> ratio 1.40285, edges 100, 140.29, ... (`run_kcap.py:1109-1111`).

**Apodisation T(theta)** (`BandPower.cc:515-545`, `BandPower_W.cc:434-465`), with `x = ln theta`, `x_l = ln theta_min`, `x_u = ln theta_max`, `Dx = Delta_x` (default 0.5, kcap 0.5; a log width in ln theta although `run_kcap.py:651` says "arcmins"):
```
T = cos^2[ (pi/2) (x - (x_l + Dx/2)) / Dx ]    for  x_l - Dx/2 <= x < x_l + Dx/2     (0 -> 1)
T = 1                                          for  x_l + Dx/2 <= x < x_u - Dx/2
T = cos^2[ (pi/2) (x - (x_u - Dx/2)) / Dx ]    for  x_u - Dx/2 <= x < x_u + Dx/2     (1 -> 0)
T = 0 otherwise.   (T(theta_min) = T(theta_max) = 1/2.)
```
The theta integral for W therefore extends over `[theta_min e^{-Dx/2}, theta_max e^{+Dx/2}]` (0.390'..385' for 0.5'..300', Dx=0.5) (`BandPower_W.cc:140-145, 348-349`).

If `Apodise=0` (not used by kcap): `W(l) = Int dl' l' S(l') G_mu(l,l')` with the closed form `G_mu(l,l') = Int_{tmin}^{tmax} theta J_mu(l theta) J_mu(l' theta) dtheta = [theta (l' J_{mu+1}(l' theta) J_mu(l theta) - l J_mu(l' theta) J_{mu+1}(l theta)) / (l'^2 - l^2)]_{tmin}^{tmax}` (Lommel; `BandPower_W.cc:189-198, 414-429`), singular at `l=l'` (not guarded).

**VERIFIED-NUM** (`bandpower_W_vs_reference_check.py`): the file `cosebis/Benjamins_Bandpower/bandpower_filter_output_0_100.0_140.29.dat` is an independent reference of the weights for the band 100-140.29, theta 0.5'-300', with columns [l, unapodised E, apodised E, apodised B]. My direct Gauss-Legendre evaluation of the formulas above with Dx=0.5 gives (W_0+W_4)/2 (apodised E) and (W_0-W_4)/2 (apodised B) agreeing with columns 3 and 4 (apodised E and B; the column meaning is inferred from `python_standalone_codes/plot_bandpower.py:92-96`) to between 1e-4 and 1e-3 relative (e.g. l=126.5: 0.866818 vs 0.866928; l=63.4: -0.142333 vs -0.142333; l=31.7 B: 0.0582915 vs 0.0582906). So W is not normalised by N_i and E uses the equal-weight sum of the J_0 and J_4 versions. The 1e-4 to 1e-3 mismatch (largest at l=1 and l=504 where |W| is small) is presumably attributable to the reference code's own numerics or to table interpolation (I did not investigate which side). [verified correction] I re-ran `bandpower_W_vs_reference_check.py` and compared all ten printed rows to the reference: the RELATIVE mismatch is 1e-3 at l=1..4, 2.1e-3 (E) and 2.2e-3 (B) at l=8, 9e-4 at l=16, 1.3e-4 at the peak l=126.5, but 1.6e-3 (E) at l=504 and 3.5e-3 (B) at l=252.6 where B is only 2e-4, so "1e-4 to 1e-3" understates the small-|W| rows. In ABSOLUTE terms the picture is cleaner: for l <~ 16 my value is offset from the reference by an almost constant ~1.5e-5 (1.50e-5 at l=1, 1.49e-5 at l=2, 1.48e-5 at l=4, 1.40e-5 at l=8); the largest absolute difference is 1.1e-4 at the peak (1.3e-4 of the peak value 0.867). A constant low-ell offset points to a systematic difference (e.g. g-table or J_0/J_4 normalisation at the lowest theta in the reference code) rather than random quadrature noise; unresolved. Its effect on a band power is `Int dl l C(l) * offset` over the low-l tail, which I did not evaluate (the scale cut `keep_ang = 100 1500` acts on the band centres, not on the support of W).

### 2.7 Other `cl_to_*` modules (not used by kcap runs)

**`cl_to_corr` (FFTLog, `CSL/shear/cl_to_corr/cl_to_corr.py`).** `n_transform=8192` log-spaced ell in `[1e-4, 5e6]` (`:17-21`); C is interpolated onto it with `scipy interp1d(kind='linear')` in (ln l, ln C) (or ln(-C), or linear C for mixed sign), extrapolated with power laws `lower=+1.0`, `upper=-2.0` (`:92-93, 157-169`); `pyfftlog.fhti(n, mu, dlogr, q=0, kropt=1)` (low-ringing adjusted kr), `xi = fht(l*C, tdir=-1)/(2 pi)/theta` with `theta = exp((x-nc) dlogr) r_mid` (`:107-155`), `nc = (n+1)/2`; mu = 0 (xi+, w), 4 (xi-), 2 (gamma_t). Output theta restricted to `[0.1', 1000']` and returned in radians. [verified] Re-read confirms all numbers above (n=8192, ell in [1e-4, 5e6], slopes +1/-2, `kropt=1`, `/(2 pi)/theta_rad`). Additional behaviour: the extrapolation power law is `C[0] (l/l_min)^lower` below and `C[-1] (l/l_max)^upper` above, applied to NaN entries, and it is applied also for mixed-sign spectra (`:157-169`); default sections are `shear_cl -> shear_xi` for both xip and xim, with output keys `xiplus_i_j` / `ximinus_i_j` for xi and `bin_i_j` for gamma_t / w (`:26-39`), so, like cl_to_xi_wigner, its xi output is not the `shear_xi_plus`/`bin_i_j` layout that `mini_2pt_like` expects. The slope `upper = -2` differs from nicaea's -3.

**`cl_to_xi_fullsky`** (`cl_to_xi.py`, `legendre.py`, `cl_to_xi_interface.py`). Exact Legendre sums: w = sum (2l+1)/(4 pi) P_l(cos theta) C_l; gamma_t with P_l^2; xi_+/- from G^+/G^- of Stebbins (astro-ph/9611125 eqs 2.26-2.27) with `N_l = sqrt(2/((l-1)l(l+1)(l+2)))`; `xi_+ = C_plus + C_cross`, `xi_- = C_plus - C_cross` with `C_* = Sum (2l+1)/(4 pi) N_l^2 C_l G_*` (`cl_to_xi.py:75-101`). C(l) is interpolated onto integer l (cubic interp1d in log-log, `cl_to_xi.py:8-34`), a sine taper is applied above `high_l_filter * ell_max` (default 0.75) (`legendre.py:10-14`, `cl_to_xi_interface.py:21-23`). Theta grid: `n_theta=20` log-spaced in `[1', 300']` by default.
[verified correction] Citation: the taper option is read at `cl_to_xi_interface.py:29` and applied at `:72-77` (the draft's `:21-23` are the theta-grid lines); the taper itself is `f = y - sin(2 pi y)/(2 pi)` with `y = (ell_max - l)/(ell_max - 0.75 ell_max)` for `l > 0.75 ell_max`, 1 below (`legendre.py:8-11`). Precision: gamma_t uses `(2l+1)/(4 pi l(l+1)) * P_l^2(cos theta)` (`legendre.py:53-61`, with the normalised-recurrence `P2l_rec_norm`), i.e. the `1/(l(l+1))` factor is part of the formula; w uses `(2l+1)/(4 pi) P_l` with the l=0 term zeroed (`:45-51`); `ell_max` and `xi_type` (`22`, `22+`, `22-`, `00`, `02`, `02+`) are mandatory ini options (no default). Outside the interpolation range `SpectrumInterp` returns 0 (NaN -> 0, `cl_to_xi.py:33`); mixed-sign spectra use `fill_value=0`. Output section names follow the nicaea convention (`shear_xi_plus`/`shear_xi_minus` with `bin_i_j`; `galaxy_xi`; `galaxy_shear_xi`) (`cl_to_xi_interface.py:38-57`), `theta` in radians.

**`cl_to_xi_wigner`** (Tessore recursion). `xi = Sum_{l>=l_min} (2l+1)/(4 pi) d^l_{m1 m2}(theta) C_l`, (m1,m2) = (2,2) xi_+, (2,-2) xi_-, (2,0) gamma_t, (0,0) w (`transform.c:20-77`). [verified] Details the draft omitted: `l_min = 2` for xi_+, xi_-, gamma_t and 0 for w (`transform.c:34-53`); the prefactor is exactly `(2l+1)/(4 pi)` (`INV4PI`, `:73`) with the Wigner-d carrying the spin structure; the Python wrapper (`wigner_cl_xi_interface.py:44-64`) interpolates C(l) to every integer l in `0..int(max ell)` with `np.interp` LINEARLY in l (not in log) and sets C = 0 outside the supplied ell range (`left=0, right=0`), which is a coarse treatment of a log-spaced 400-point input; theta is a LINEAR grid `np.linspace(theta_min, theta_max, n_theta)` in arcmin converted to radians (`:134-141`, `theta_min`, `theta_max`, `n_theta` all mandatory); `corr_type` is 0/`xi`, 1/`gammat`, 2/`wtheta` (different codes from nicaea's 0/1=w/2=gt, `:112-128`); outputs use the keys `xiplus_i_j`, `ximinus_i_j` (xi) or `bin_i_j` (`:164-169`), so they are NOT drop-in for `mini_2pt_like` which reads `bin_i_j` in `shear_xi_plus`/`shear_xi_minus`; `setup_from_file` raises NotImplementedError (`:66-67`).

---------------------------------------------------------------------------------------------------

## 3. Numerical choices (with source references)

### 3.1 nicaea Hankel step

| item | value | where |
|---|---|---|
| number of points | N_thetaH = 1024 (both ell and theta) | `maths.h:38` |
| ell grid | log-uniform, l_min=1e-4, l_max=5e6 nominal; actual 9.645e-5 .. 4.8226e6 (nc=513) | `maths.h:34-35`, `maths.c:1902-1913` |
| theta grid | log-uniform, 2.0736e-7 .. 1.0368e4 rad, dlntheta = 0.0240814 | `maths.c:2001-2002` |
| theta points in 0.5'..300' | 265 (j = 273..537) | computed |
| FFT | FFTW r2c / c2r, `FFTW_ESTIMATE`, plans created and destroyed on every call and for every bin pair | `maths.c:1899-1900, 2005-2008` |
| kernel | complex Gamma via `cdgamma`; recomputed for every bin pair and for each of xi+ and xi- | `maths.c:1959-1976` |
| bias q | 0 for xi+-, w, gt (no FFTLog bias, no low-ringing kr adjustment) | `maths.c:1922-1929` |
| Nyquist/DC | Im part forced 0 | `maths.c:1990-1991` |
| C(l) interpolation | linear in (ln l, ln|C|); slopes +1 / -3 outside; mixed sign: linear C, constant extrapolation (EPS bug) | `cl_to_xi_interface.c:268-339`, `maths.c:1617-1645` |
| validity range | module.yaml: results valid "from about 1 to a few hundred arcmin"; input ell "must include the corresponding range" | `module.yaml` |
| ini defaults | `corr_type=0`, `filter_type=0`, no other numerical option is configurable | `cl_to_xi_interface.c:39-41` |
| fiducial ini | KV450: `corr_type=0` only; projection `ell 0.1..5e5, 400 pts`; KiDS-1000 pipelines via run_kcap.py: ell 1..3e4, 400 pts | ini files above |

Physical consequence of the ell range: for KiDS-1000 style pipelines (`ell_max=3e4`) everything above l=3e4 is a pure `l^-3` extrapolation of C. The nicaea module.yaml and the comment in `KV450_fiducial.ini:164` both say the transform needs a high `ell_max`. I have not quantified the bias on xi_+ at 0.5'; see Q-section and open questions.

### 3.2 xi binning

See 2.4. Constants: 100 log sub-bins per theta bin; trapezoid on native grid points in `mini_2pt_like`; `nTheta=9`, 0.5'-300' (kcap). Interpolation of xi onto sub-bin points in `xipm_binned`: GSL natural cubic spline in ln(theta), 0 outside range.

### 3.3 COSEBIs

| item | value | where |
|---|---|---|
| ell range for E_n | LLOW = 1, LHIGH = 500*20/theta_max[rad] (= 1.146e5 for 300') | `WnLog.h:10-11`, `WnLog.cc:42`, `COSEBIs.cc:56` |
| W_n table | 10^6 points, log-uniform in [1, LHIGH], cubic spline in ln l, ASCII file, 20-digit precision | `WnLog.h:19`, `function_cosebis.cc:428-450` |
| W_n quadrature (building the table) | l < lthresh=2pi n/(10 theta_max): GL 10 n points; else GL 40 points per sub-interval between J_0 zeros | `WnLog.cc:126, 143-149, 183-209`, `WnLog.h:16` |
| E_n quadrature | [LLOW, 2pi n/(12 theta_max)]: GL 100; above: GL 20 per interval between local extrema of tabulated W_n (10^6-point scan) | `COSEBIs.cc:350-361, 313-343` |
| T_-n integral | GL 20 between 0, roots below z, z | `TpnLogRadian.cc:143-149` |
| T tables for xi route | 10^4 points log-uniform in [theta_min, theta_max], spline in ln theta, files in `Tn_Output_FolderName` | `TpnLogRadian.h:7`, `TpnLogRadian.cc:97-122` |
| xi-route quadrature | GL 10 between extrema of T_+ (10^6-pt scan) and T_- (10^5-pt scan) | `COSEBIs.cc:400-443, 455-466` |
| n_max | kcap 5 (run_kcap.py:1119), demo 5; code default 10 (`:157`) though the log message says 5 | `COSEBIs_cosmosis_interface.cc:151-160` |
| C(l) interpolation | natural cubic spline of linear C in ln l; power-law extrapolation; NaN->0 | 2.5.4 |
| default folders | `WnLog/`, `TLogsRootsAndNorms/`, `TpnLog/` (relative to cwd) | `COSEBIs_cosmosis_interface.cc:341-344` |
| theta range requirement | the (theta_min, theta_max) pair must match a shipped Root/Normalization file name to 2 decimals in arcmin, else `exit(1)` | `WnLog.cc:52-64` |

### 3.4 Band powers

| item | value | where |
|---|---|---|
| ell range for W / BP integrals | LLOW = 0.1, LHIGH = 1e4 (hard-coded) | `BandPower_interface.cc:23-24` |
| W table | NLBINS = 1000 log-uniform points in [0.1, 1e4], natural cubic spline in ln l, 0 outside | `BandPower_interface.cc:22`, `BandPower_W.cc:303-304` |
| g table (needed for the apodised W when not using the analytic form directly) | `nTheta = 10000` points LINEAR in theta over `[theta_min e^{-Dx/2}, theta_max e^{+Dx/2}]`, cubic spline | `BandPower_g.h:8`, `BandPower_g.cc:171` |
| W(l) quadrature (table build) | per l: scan of 10^5 log-spaced theta for extrema of `J_mu(l theta) g(theta)`, then GL 20 points per interval | `BandPower_W.cc:326-329, 342-373` |
| BP quadrature | scan of 10^4 log-spaced l for extrema of W (ONLY done inside `calBP`, every call and every bin pair), GL 20 per interval | `BandPower.cc:351, 386-391, 476-509` |
| apodisation | Delta_x = 0.5 (ln theta), cos^2 | 2.6 |
| ini used by KiDS-1000 | `type=cosmic_shear_e|ggl`, tophat, `analytic=1`, l 100-1500, 8 bands, theta 0.5-300', `apodise=1`, `delta_x=0.5`, weights folder `./cosebis/BandPower_outputs/` | `pipeline.ini` |
| weight cache file | `<folder>/W_ap<Dx>_<mu>_<lmin>-<lmax>_<tmin>-<tmax>.table` (2 decimals), g cache `g_<mu>_<band>_<tmin>-<tmax>.table` | `BandPower_W.cc:273-275`, `BandPower_g.cc:144-146` |
| edges | `l_b = l_min (l_max/l_min)^{b/nBands}` | `BandPower_interface.cc:153-160` |
| C(l) interpolation | same as COSEBIs (`setInput_single_withExtrapolation`) | `BandPower.cc:319-331` |

---------------------------------------------------------------------------------------------------

## 4. Quirks, approximations, deviations (QUIRK list)

Numbered for reference in sections 6-8.

- **Q1 QUIRK (nicaea ell sampling):** assumes `ell` is log-uniform and uses only the endpoints (`cl_to_xi_interface.c:187-189`). Any other spacing is silently wrong. Works for project_2d (`np.logspace`).
- **Q2 QUIRK (nicaea EPS macro):** `#define EPS 1.0-5` = -4, so `fabs(x) < EPS` is never true (`maths.c:1617`). The "no extrapolation allowed" error path is dead; mixed-sign C(l) extrapolates as a constant (and the printed warning "no power law extrapolation" is misleading: it is constant, nonzero).
- **Q3 QUIRK (nicaea ln2):** `ln2` = 0.69314718 (`maths_base.h:53`).
- **Q4 QUIRK (nicaea grid centering):** `nc = N/2+1` shifts the 1024 grid; endpoints are l = 9.645e-5 .. 4.8226e6 instead of 1e-4 .. 5e6. The interface code is consistent with this (it recomputes theta from the returned limits).
- **Q5 QUIRK (non-positive test):** `C <= 0` counts zeros as negative, so a C(l) with a single exact zero is classified "mixed sign" and loses log-interpolation and power-law extrapolation (`:270-273`). The 1e-30 floor is only applied when all entries share a sign.
- **Q6 QUIRK (docs vs code):** `module.yaml` documents `xiplus_i_j`, `ximinus_i_j`, extrapolation "linearly before and cubically after" and theta "2e-7 to 1e4 rad"; the code writes `bin_i_j` in `*_plus/_minus` sections, extrapolates as power laws l^1 and l^-3 in C, and the theta range is as computed in 2.2. `auto_corr` is read and never used (`:81`). The module comment says "Only stores j<=i", true only because the input does.
- **Q7 QUIRK (circular convolution / aliasing):** a periodic log-FFT: wrap-around at the ends of the 1024-point grid. It is harmless for theta well inside the grid because l*C vanishes at both ends only if the extrapolation slopes are right. For ggl and for different spectral shapes the user must check. The same shape assumption means C(l) just above the input `ell_max` is replaced by `l^-3` regardless of physics.
- **Q8 QUIRK (xi binning in mini_2pt_like):** the trapezoid uses only native points inside `[edge_i, edge_{i+1})`, no interpolation at the edges (`mini_2pt_like.py:17-26`), plus an unused spline built each call (`:12`).
- **Q9 QUIRK (xipm_binned point/bin centre):** the point-evaluation centres are the geometric means of the edges (`:236`); the bin-weighted mode is `theta d theta` weighting, NOT the pair-count weighting unless `InputNpair` is provided. The `weighted_binning` behaviour silently degrades to point evaluation if the Npair files are missing (`:552-557`). [verified correction] WRONG as drafted: a missing `InputNpair`, `nBins_in` or Npair file falls back to THETA-weighted (theta dtheta, 100 log sub-bins) averaging, not to point evaluation (`:410-413, 450-452, 483-485, 552-557`); see the added paragraph in 2.4. A separate possible bug in the Npair path is the pair-index mismatch described in 1.4.
- **Q10 QUIRK (probable bug):** `utils/run_kcap.py:592, 611, 634` pass `theta_max = xi_theta_min` to the xi binning modules (`xi_theta_max` is defined at `:1098` and never used there).
- **Q11 QUIRK (COSEBIs bin naming):** reads/writes `bin_{j}_{i}` with `i` the outer loop (`COSEBIs_cosmosis_interface.cc:471`); for auto spectra this reproduces the "(1,1),(1,2)..." data ordering only because C(l) is stored for j<=i. The 2PCF interface uses the same convention (`COSEBIs_2PCFs_cosmosis_interface.cc:722, 733`). [verified] Precisely: the 2PCF interface loops `i_bin` outer and `j_bin = i_bin..N` inner (`:727-729`) and reads `bin_{j_bin}_{i_bin}` (`:733`); the COSEBIs C(l) interface loops both bins 1..N and reads `bin_{j}_{i}` (`:465-471`), which is only found for j >= i, so both produce the same pair order. `xipm_binned`, `nicaea` output and `mini_2pt_like` (xi_pm-bin-theta layout) instead use `bin_{i}_{j}` with j <= i; see 1.4.
- **Q12 QUIRK (inconsistent C(l) interpolation):** nicaea uses piecewise power law (linear in log-log); the COSEBIs and band-power modules use a natural cubic spline of linear C vs ln l. The two routes (C->xi->E_n vs C->E_n) therefore differ at the interpolation-error level (O(dlnl^2) ~ 1e-3 relative locally for the 400-point grid). Not measured.
- **Q13 QUIRK (extrapolation below):** `function_cosebis::extrapolate` uses table points 1 and 2 (not 0 and 1) for the low side (`function_cosebis.cc:366-369`); the table must have >= 3 points (`:354`). Negative C gives NaN -> returned as 0 (`:272`).
- **Q14 QUIRK (hard ell cutoff):** E_n sums only up to `LHIGH = 10^4/theta_max`; band powers up to 1e4 (`BandPower_interface.cc:24`; BandPower_interface.cc comment `:17-21` warns about artefacts above 1e5). Contributions from larger l are dropped, not extrapolated.
- **Q15 QUIRK (W_n step list):** `StepFinder` inserts `theta_min e^{r_{n,i}}` (a THETA, ~1e-4 rad) into a list of values of x = l*theta (the J_0 zero locations) (`WnLog.cc:229-236`); it should be `l*theta_min e^{r}`. The effect is only a misplaced (harmless) sub-interval break [verified correction: the draft put these at x ~ 1e-4..1e-3; for the 0.5'-300' tables `theta_min e^{r}` spans 1.5e-4 .. 8.7e-2 rad (r in ~0.1..6.4 times theta_min = 1.45e-4 rad), and since every value is below the first J_0 zero 2.405 all n+1 spurious entries sit at the head of `StepN`, before the first zero], so the quadrature is not split at the polynomial's roots; the GL(40) rule per J_0 half-period is still accurate for a degree-(n+1) polynomial times J_0.
- **Q16 QUIRK (undefined behaviour):** `TpnLog::TnLog` evaluates `root[n-1][counter]` before testing `counter < size` (`TpnLogRadian.cc:132`); for z above the last root (z in (6.206, 6.397] for n=1, 0.5-300) it reads one element past the vector. Only used in the xi->E_n route and when building T_- tables.
- **Q17 QUIRK (analytic flag ignored):** `BandPower::initialize` calls `set_W()` with its default `Analytic = true` (`BandPower.h:106`, `BandPower.cc:74`), builds all W tables there, and later `set_W(Analytic)` inside `calBP` is a no-op (`WSet`). So ini `analytic = 0` has no effect in the Fourier path used by the interface (`real=false`, `BandPower_interface.cc:328`). The non-analytic g integral exists (`BandPower_g.cc:185-227`) but is unreachable here. [verified] Related waste: whenever a W table file is missing, `BandPower_W::set` calls `g.set(...)` (`BandPower_W.cc:296`), which, even with `Analytic = true`, first runs `BandPower_g::determine_integration_limits()` -- a scan of 10^6 Bessel values per band and per mu (`BandPower_g.cc:166, 236-265`) whose result is only used by the unreachable non-analytic branch -- and then rebuilds and overwrites the 10^4-point g table (`makeTable` + `saveTable`, `:171-172`). The g cache file name carries the (padded) theta range and the band index but not the band edges (`:144-146`); because it is always rebuilt before use this is harmless, unlike the W/Wn caches (Q19).
- **Q18 QUIRK (B-mode band powers):** `cosmic_shear_b` is evaluated on the E-mode C(l) (`BandPower_interface.cc:422`); it is a measure of E->B leakage of the finite theta range, not a B-mode prediction. [verified] Nuance: the claim "B-mode C(l) does not exist in the datablock" is only true for the kcap configs. `project_2d.py` does define a B-mode intrinsic spectrum (`IntrinsicbIntrinsicb`, section `shear_cl_bb`, `:224-228`, fed by `intrinsic_power_bb`), which could be pointed to via `input_section_name`; no kcap ini does so. The `is_it_bmodes` ini flag of both `libbandpower` and `libcosebis` is a pure label (written as `b_modes`).
- **Q19 QUIRK (cache keys):** all cached tables (`W*.table`, `g*.table`, `WnLog*.table`, `TpRadian*/TmRadian*`) are keyed only on (n or band, theta range / ell range to 2 decimals, apodisation width); not on LLOW/LHIGH/NLBINS/analytic/`high`/code version. A stale file is silently reused (`BandPower_W.cc:277-304`, `WnLog.cc:105-135`, `TpnLogRadian.cc:187-221`). [verified correction] `g*.table` is NOT in this category: `BandPower_g::set` always rebuilds and overwrites it (`BandPower_g.cc:171-172`). The W/Wn/Tp/Tm tables are loaded with the header range stored in the file (`function_cosebis::loadTable`, `:482-526`), so changing `LLOW`, `LHIGH`, `NLBINS` or `high` in the source or ini has no effect once a file exists. Also a table file written with the 20-digit text format is parsed with `>>` on every start.
- **Q20 QUIRK (units):** `theta_min/theta_max` are stored in ARCMIN by the COSEBIs module (`COSEBIs_cosmosis_interface.cc:402-403`) but in RADIANS by the band-power module (`BandPower_interface.cc:574-575`). `nbin_a/nbin_b` are written as doubles in both (`:627-628`, `:576-577`). `nPairs` in `BandPower_interface.cc:504` is uninitialised (only incremented, not used afterwards).
- **Q21 QUIRK (thread safety/state):** `function_cosebis::value` and `interpolate` use `static` locals and a shared GSL accelerator (`function_cosebis.cc:260, 390`), and class members like `ell`, `bin_index`, `redshiftPair` are mutated while integrating, so the objects are not re-entrant.
- **Q22 QUIRK (naming mismatch):** the fiducial inis load `cosebis/libcosebis_likelihood.so` (`KV450_fiducial.ini:204`) but the Makefile builds `libcosebis.so`, `libcosebis_2pcfs.so`, `libcosebis_cov.so`, `libbandpower.so`, `libxipm_binned.so` (`cosebis/Makefile:20-41` [verified correction: draft said 19-33]). Option names in C++ such as `Apodise`, `Delta_x`, `Output_FolderName`, `InputCos4phi` rely on CosmoSIS lower-casing ini keys; I could not verify this in the CosmoSIS sources (not in this tree). [verified] RESOLVED using the locally installed CosmoSIS (`/mnt/d/CosmoSIS/cosmosis/datablock/datablock.hh:52,194-295` and `datablock.cc`): `DataBlock::get_val` calls `downcase(section); downcase(name);` on every access, and the ini parser lower-cases option names (the generated `runs/3x2pt/.../pipeline.ini` shows `apodise`, `nbands`, `output_foldername`, all lower case although `run_kcap.py` writes `Apodise`, `nBands`, `Output_FolderName`). So the capitalised C++ option names do work. The related file-name suffix `libcosebis_likelihood.so` (KV450 inis) is also present in `BOSS_KV450.ini` and `KV450_no_sys.ini`; it is not a Makefile target; in all three inis `cosebis` is not in the `modules =` list (KV450_fiducial.ini:11, KV450_no_sys.ini:11, BOSS_KV450.ini:12), so the `[cosebis]` section is dead configuration there and the mismatch only bites if someone adds `cosebis` to `modules`.
- **Q23 QUIRK (thresholds):** `lthresh` is `2 pi n/(10 theta_max)` when building W_n (`WnLog.cc:126`) but `2 pi n/(12 theta_max)` when integrating E_n (`COSEBIs.cc:350`); harmless, both are just quadrature switch points.

---------------------------------------------------------------------------------------------------

## 5. Strengths of this subsystem

1. **The Hankel step is fast and well-conditioned.** One 1024-point real FFT pair per spectrum, no oscillatory integrals, no ringing handling needed provided the extrapolation slopes make l*C vanish at both ends; the same code gives xi_+, xi_-, gamma_t, w. I verified the algorithm reproduces direct quadrature to ~1e-7 (toy spectrum). [verified correction: ~1e-8 to 4e-6 relative at 1' and 10' depending on mu and on proximity to a zero crossing; see the corrected numbers in 2.2.]
2. **Wide dynamic range for free.** The grid spans 11 decades in l and 17 decades in theta; the user only needs `ell` to cover the physically relevant range.
3. **Sign and zero handling for C(l)** (log-interpolation for strictly positive/negative spectra, linear fallback) means cross-spectra do not crash the log interpolation.
4. **COSEBIs are a clean linear estimator of C(l).** E_n = Int w_n(l) C(l) dl with fixed weights, so the covariance-aware likelihood is exact in the Gaussian approximation and E/B separation is built in (conditions 2-3 of 2.5.2). Orthonormality of T_+n and the moments vanish to 1e-7 in the shipped tables.
5. **Band powers**: analytic top-hat g_mu removes one nested integral; the real-space and Fourier forms are equivalent by construction; there is an independent reference file for the apodised weights (agreement 1e-4..1e-3 with my re-derivation [verified correction: up to 3.5e-3 relative on small-|W| rows and <=1.1e-4 absolute; see 2.6]).
6. **Precomputation / caching** of W_n and band-power weights moves the heavy integrals out of the sampling loop.
7. **Quadrature design** (split integrals at extrema of the oscillating weight, Gauss-Legendre on each piece) is robust to the strongly oscillatory integrand at high l.
8. **Self-contained C(l)->{xi, E_n, BP} stack** inside one datablock convention (same section/key layout across xi, COSEBIs, band powers), which is what makes the 3x2pt configs in `run_kcap.py` straightforward.

---------------------------------------------------------------------------------------------------

## 6. Weaknesses and risks

### 6.1 Correctness
- **W1 (Q1, Q2):** log-uniform assumption and the dead error branch (`EPS` macro) mean misuse is silent. For mixed-sign C(l) the extrapolated constant (nonzero up to l = 4.8e6) can dominate l*C wrap-around contributions.
- **W2 (3.1):** with `ell_max = 3e4` (KiDS-1000 pipelines built by `run_kcap.py`) nicaea extrapolates C ∝ l^-3 above 3e4, steeper than the nonlinear cosmic-shear spectrum; the xi_+(0.5'-1') prediction depends on that choice. The documentation and the KV450 ini warn that a high `ell_max` is needed. I did not quantify the bias.
- **W3 (Q10):** the xi binning in `run_kcap.py` appears to be misconfigured (`theta_max = theta_min`).
- **W4 (Q17/Q18):** the `analytic` ini option is a no-op; `cosmic_shear_b` is not a B-mode prediction.
- **W5 (Q12):** two different C(l) interpolators in the same stack.
- **W6 (Q14/3.4):** hard-coded ell cutoffs (LHIGH = 1e4 for BP, 10^4/theta_max for E_n) are independent of the supplied `ell` grid and of the physical signal; BP W tables use only 1000 log points in l in [0.1, 1e4] (dlnl = 0.0115), so at l ~ 5e3-1e4 the table step in l (~60-115) can exceed the oscillation period of W (2 pi/theta_max_eff ≈ 56 for 385') where |W| is small but non-zero because of the cos^2 apodisation and spline noise. I have not measured the resulting error. [verified] Measured for the bands that actually matter (l <= 1500), with an independent emulation (NOT kcap's own table): I computed the apodised E-weight (W_0+W_4)/2 for Dx=0.5, theta 0.5'-300' on a fine l grid by Gauss-Legendre (400 z-chunks x 30 nodes), built a natural cubic spline in ln l through the nodes of kcap's table grid `l_k = 0.1 (1e5)^{k/999}` (1000 points, `BandPower_interface.cc:22-24`), and compared. Max |W_spline - W_direct| over l in [l_lo/5, 5 l_hi]: band 1 (100-140.3) 2.4e-7; band 4 (276-387) 6.6e-6; band 8 (1069-1500) 1.7e-3 absolute (peak |W| = 1.08, so 1.6e-3 of peak, table step 17 vs oscillation period 56). Effect on a band power for a smooth toy spectrum C = l^-1.2: relative 1e-9, -3e-9 and -3e-8 for bands 1, 4, 8 -- the oscillating spline error integrates out. So the 1000-point W table is adequate for smooth C(l); the pointwise 1.7e-3 error in the top band only matters for a C(l) with structure on the scale of the W oscillation (e.g. a spiky emulator output). Script: `04_verification/bandpower_W_spline_error_check.py`.
- **W7 (2.4):** the `mini_2pt_like` bin integral ignores the partial grid cells at both edges (≈ ±0.5 grid step of 0.024 in ln theta).

### 6.2 Accuracy / numerics
- Natural cubic splines on 400 points of a steeply falling C(l) can overshoot (no monotonic constraint); log-log linear interpolation in nicaea has O(dlnl^2) error.
- The kernel's 8-digit `ln2`, 8-digit root files (relative error ~1e-8 in T_n).
- The log-FFT truncates the kernel at Nyquist; accuracy near the grid ends (theta < 1e-5 rad or > 1 rad) is not guaranteed (documented as "valid ~1' to few hundred arcmin").

### 6.3 Maintainability / portability
- Roots and normalisations are external, undocumented (no generator in the repo, no header explaining the definition). I reconstructed the definition (2.5.2).
- Filenames encode the theta range to 2 decimals; a new range requires external generation; mismatch aborts the process (`exit(1)`).
- Hard-coded relative default paths (`./cosebis/...`), writing cache files into the working directory/repo at run time, ASCII tables of 10^6 lines per mode, `besselzero04.h` of 200,006 lines.
- Three code paths (xi, COSEBIs, BP) re-implement the same basic building blocks (`function_cosebis` table class with Numerical-Recipes-era style, static locals, `exit()` on error).
- Compile flags come from CosmoSIS `compilers.mk`; optimisation level unknown to me (not in this tree).
- Doc/code mismatches (Q6, Q22).
- Test coverage for this subsystem: none found (the `tests/` folder tests other modules; I did not find tests of cl_to_xi, cosebis or bandpower). [verified correction] Partly wrong: `tests/test_pipeline_against_CCL.py:71-102` (and `test_pipeline_against_CCL_and_MP.py:168-191`) run `runs/config/KV450_no_sys.ini` (which includes `cl2xi` = nicaea) and compare `shear_xi_plus`/`shear_xi_minus` against `pyccl.correlation(..., corr_type="L+"/"L-")` for xi_+ over theta in (0.05, 10) deg and xi_- for theta > 0.05 deg, after dividing CCL's C(l) by the `(l-1)l(l+1)(l+2)/(l+1/2)^4` factor (`:80-84`/`:172`). They only `print` maximum fractional differences (no `assert`, no tolerance), so they are diagnostics, not regression tests, and the xi_- line prints `frac_diff_xi_plus` by copy-paste error (`:102`). There is still no test of `libcosebis`, `libbandpower`, `libxipm_binned` or `mini_2pt_like`.

### 6.4 Performance hotspots (estimates by operation count; NOT measured)
- **COSEBIs per likelihood call:** per C(l) pair and per n the integral visits ≈ (number of extrema of W_n up to LHIGH) × 20 nodes. W_n at high l oscillates with period 2 pi/theta_max = 72 in l (theta_max = 300'), so ≈ 2×LHIGH/(2π/θmax) ≈ 3000 segments → ~6e4 nodes; × 15 pairs × 5 modes ≈ 4.5e6 nodes, each calling two spline evaluations (one on a 10^6-point table) plus `exp()/log()` in `value()` → rough estimate 0.3-1 s per likelihood evaluation in C++.
- **COSEBIs first run:** building each W_n table: 10^6 l values × (≈ l theta_max/pi zeros × 40 GL points) ≈ up to 10^5 Bessel evaluations per l near the top → order 10^10-10^11 evaluations per n; plus 10^6-point extremum scans (done once per run, every run) and ASCII I/O of 10^6 lines per n. Hours the first time, tens of seconds of file parsing in every later run (estimate).
- [verified] **Quadrature-table churn (all C++ routes):** `gaussianIntegrate_gsl` (`cosebis/modules/Integrate.cc:5-20`) calls `gsl_integration_glfixed_table_alloc(N)` and `..._free` on EVERY invocation, i.e. recomputes the N Gauss-Legendre nodes/weights (Newton iteration) for each of the ~10^3-10^4 sub-intervals of every integral, every pair, every mode, every likelihood call. Caching one table per N (N in {10, 20, 40, 100, 10n}) is a trivial speed-up independent of the matrix reformulation in section 7.
- **Band powers per call:** `determine_integration_limits_Fourier` rescans W on 10^4 points for every band, for each of 2 BP objects (E uses J_0 and J_4), **inside every `calBP()` call and so for every bin pair** (15 pairs for 5 tomographic bins), i.e. ≈ 15×2×8×10^4 = 2.4×10^6 spline evaluations that do not depend on C(l) at all; then 20-point GL over every segment for every band/pair.
- **nicaea per call:** per bin pair: one FFTW plan creation/destruction pair, 513 × 2 × 2 `cdgamma` calls per xi_± (the kernel is bin-independent), one forward and two inverse FFTs. About 15×(1 fwd + 2 inv FFT of 1024 points) plus 30,000 complex-gamma evaluations → a few ms total (estimate). Not a bottleneck compared with the Limber projection.
- **xipm_binned per call:** 15 pairs × nTheta × 100 spline evaluations.

---------------------------------------------------------------------------------------------------

## 7. Concrete efficiency opportunities (for the Python/numba re-implementation)

All numbers are order-of-magnitude estimates; each should be validated by timing.

1. **Treat COSEBIs and band powers as fixed linear maps.** E_n = Q_n · C(l_k) with nodes l_k and weights `w_k l_k W_n(l_k)/(2 pi)` fixed by (theta_min, theta_max, n_max). Precompute `Q` (n_max × N_nodes) once per configuration; per likelihood call do: one vectorised interpolation of each C(l) to the nodes and one matrix product. N_nodes ≈ 6×10^4 is the number kcap implicitly uses; a uniform grid in l with spacing ≲ 7 (to resolve the 72-period) up to LHIGH, or Gauss-Legendre per half-period, gives similar accuracy. Per call cost → ~15×6×10^4 spline-evals + 15×5×6×10^4 MACs ≈ 1-5 ms. Expected speed-up vs the C++ route: ~100× (0.3-1 s → a few ms). Same trick for the 8-band × (J_0, J_4, J_2) band-power weights: `BP = Q C`. Expected ≈ 50-100×, plus removal of the C-independent rescans (6.4).
2. **Build W_n and W_mu^i with batched matrix products, not nested quadrature.** `W_n(l) = Sum_j w_j theta_j^2 T_n(z_j) J_0(l theta_j)` with a fixed Gauss-Legendre rule in z = ln(theta/theta_min) (hundreds to a few thousand nodes per decade-chunk; with chunking so that each chunk has ≲ 40 points per J_0 period) is a (N_l × N_theta) `scipy.special.j0` evaluation (I used 30,000 l × 60,000 theta nodes in the check script: ~minutes in numpy, seconds with numba/vectorisation) instead of 10^6 × 10^5 serial Bessel evaluations → hours to ~a minute or less. The same for band-power W via the analytic g_mu (my check script evaluates ten l values in a fraction of a second). Cache as `.npy` keyed by a hash of all inputs (theta range, n_max, ell grid, Dx, band edges, code version) to fix Q19.
3. **Calculate roots/norms on the fly (exact moments, mpmath)** (script supplied); removes the external-file dependence and the `exit(1)` for unlisted theta ranges. Cost: ms-seconds one-off.
4. **xi route: precompute the kernel and vectorise over bin pairs.** `K_mu(k_m)` for mu = 0, 2, 4 depends on nothing but the grid: compute once with `scipy.special.loggamma` (complex). Then `xi = irfft(rfft(l*C, axis=-1) * K, axis=-1)` for the whole (n_pairs × 1024) array in two FFT calls. Removes 30,000 `cdgamma` calls and 45 FFTW plan builds per evaluation. Expected 2-5× on this module (it is cheap already); the real gain in Python is avoiding a Python loop.
5. **Fold the theta binning into one dense matrix.** The grid is fixed, so "cubic spline in ln theta then 100-point sub-bin average" is linear in the 1024 xi values: `xi_bin = B @ xi` with `B` (nTheta × 1024) built once (apply the spline operator to the identity). All 15 pairs × 2 (±) in one matmul: removes ~15×9×100×2 spline evaluations per call (≈ 3×10^4 → 1 matmul, ~100× on that step).
6. **Skip the FFT on unused ranges.** Only θ in ~[0.3', 400'] (≈ 300 of the 1024 points) is ever used; either keep the FFT (cheap) or evaluate the bin-averaged kernel directly: `xi_bin_b = Sum_i C(l_i) l_i Dl_i Kbin_b(l_i)` with `Kbin_b(l) = (1/Norm_b) Int_b theta J_mu(l theta) dtheta / (2 pi)` precomputed, so Hankel transform + binning become one matrix (nbins × n_l) per mu. This removes spline-in-theta error and bin-edge handling, but needs a fine l grid (~10^4-10^5 nodes) for the oscillatory high-l tail unless the bin average damps it (it does: the bin-averaged J decays ~1/(l Dtheta)). Use the log-FFT version as the validated reference (VERIFIED-NUM) and test the matrix version against it. Expected: removes FFT + spline + binning (≈ ms) but not a large factor in absolute terms; main benefit is exact bin-averaging and differentiability.
7. **Unify C(l) interpolation:** do it once per spectrum onto a shared fine grid (one cubic spline in ln l on log|C| or a PCHIP, to avoid overshoot) and reuse for xi, E_n and BP; this also removes Q12.
8. **Memory/IO:** replace the ASCII 10^6-line tables with in-memory `numpy` arrays (roughly 50 MB of ASCII per mode, i.e. a few hundred MB for 5 modes by my estimate of ~45 bytes per line; ~8 MB per mode as float64 binary, or nothing if computed on the fly).
9. **Parallelism:** the 15 bin pairs and the n modes are independent; with the matrix formulation use BLAS threads rather than OpenMP loops; numba is only needed if the Q-matrix is built on the fly.
10. **Optional:** since E_n, BP and xi_binned are all linear in C(l) (log-interpolation aside), their Jacobians w.r.t. C(l) are the same matrices, which is useful for emulators/derivatives.

---------------------------------------------------------------------------------------------------

## 8. Open questions I could not resolve

1. **CosmoSIS key case:** C++ modules read `Apodise`, `Delta_x`, `Output_FolderName`, `InputCos4phi`, `Wn_Output_FolderName`, etc., while the ini files use lower case (`apodise`, `delta_x`, `output_foldername`). If CosmoSIS does not lowercase option names on the read path these options would silently fall back to defaults (e.g. `Apodise` not found -> no apodisation, `Delta_x` -> 0.5). CosmoSIS is not in this tree; I assumed lowercasing. [verified] RESOLVED: CosmoSIS is installed locally (`/mnt/d/CosmoSIS`); `DataBlock::get_val` downcases section and key on every access (`cosmosis/datablock/datablock.hh:52,194-295`) and the ini parser is a stock `configparser` subclass with the default lower-casing `optionxform` (`cosmosis/runtime/config.py:22,83`). The capitalised C++ option names therefore work and no option silently falls back to a default for this reason.
2. **Effect of `run_kcap.py` `theta_max = xi_theta_min` (Q10)** on actual `xipm` runs: I did not run it; maybe an override exists that I did not find.
3. **Origin of the Root/Normalization tables** (Mathematica? internal code?) is not documented here; my reconstruction reproduces them (n=1..4) but I did not check n>4 or other theta ranges against the shipped files beyond the three norm checks listed in 2.5.2.
4. **Quantitative impact of `ell_max=3e4` + l^-3 extrapolation** on xi_+/- at the smallest scales (0.5'-2'), and of the ln2/EPS/grid-offset quirks on the 1e-3..1e-4 level; needs a comparison against a converged high-ell reference.
5. [verified] PARTLY RESOLVED (see W6): for bands up to l = 1500 the 1000-point spline error is <= 1.7e-3 pointwise in the top band and <= 3e-8 on a smooth band power. Not resolved: l > 5e3, and the origin of the ~1.5e-5 constant low-ell offset against the reference file. **Accuracy of the W spline for band powers at l > 5e3** (1000-point table), and whether the reference file `Benjamins_Bandpower` columns differ from kcap's table by more than the ~1e-4 I see versus my own re-derivation (i.e. which of the two is more accurate).
6. **Shear-GGL conventions:** for `ggl` the code uses mu=2 and `norm=2 pi`; sign convention of gamma_t (tangential vs cross) and C_gE definition belong to subsystem 03; not checked here.
7. **Aperture-mass kernels** (`tp_map2_poly`, `hankel_kernel_mumu`) are unverified and not exercised by kcap.
8. **Performance numbers** in 6.4 and 7 are from operation counting; kcap was not built or timed (rules of this study).
9. **`libcosebis_likelihood.so`** named in the inis is not produced by the shipped Makefile (Q22); possibly an older/other build target not in this tree.
10. **Unused paths:** noisy-input and covariance-from-simulation code in `COSEBIs_2PCFs_cosmosis_interface.cc` and `KsiCovariance.cc` (1936 lines) were only skimmed; they are not in the likelihood path of any kcap config I found.

---------------------------------------------------------------------------------------------------

## Appendix A — Re-implementation checklist (minimal path to reproduce kcap numbers)

1. Hankel: build `l_i = exp(lnrc + (i-513) Dl)`, with `l_min=1e-4, l_max=5e6, N=1024`; `C` by log-log linear interpolation from the input `ell` grid (extrapolation slopes +1 / -3 in `ln C`); `f = l*C`; `F = rfft(f)`; `k = 2 pi m/(Dl N)`; `K_mu = exp(i k ln2) Gamma((1+mu+ik)/2)/Gamma((1+mu-ik)/2)` with `ln2 = 0.69314718` if bit-matching; zero the imaginary parts of bins 0 and 512; `r = irfft(F K)`; `xi[N-1-i] = r_i l_i/(2 pi)`; `theta_j = 1/l_{N-1-j}` (radians). Do mu = 0 and 4 from the same `F`.
2. Binning: 9 log bins [0.5', 300']; per bin 100 log sub-bins, centre `tmin (tmax/tmin)^{(k+.5)/100}`, weight `theta dtheta`, cubic-spline interpolation of xi in ln(theta[arcmin]); or the `mini_2pt_like` trapezoid on native grid points.
3. COSEBIs: roots/norms from the recipe in 2.5.2 (or the shipped tables); T_- from the integral formula; `W_n(l) = l^-2 Int_{l tmin}^{l tmax} x T_n(x/l) J_0(x) dx`; `E_n = (2 pi)^-1 Int_1^{10^4/theta_max} dl l W_n C` with C from a natural cubic spline in ln l (extrapolate power-law); optional c-term vectors added afterwards.
4. Band powers: `g_mu` closed forms (2.6), cos^2 apodisation with Dx = 0.5 over `[tmin e^{-Dx/2}, tmax e^{+Dx/2}]`, `W_mu^i(l)` via theta integral, `BP_E = (BP_0 + BP_4)/2` (shear) / `BP_2` (ggl), `/N_i = ln(l_max_i/l_min_i)`, `Int` over l in `[0.1, 1e4]`.
5. Test vectors / checks available in-tree: shipped Root/Normalization tables (`cosebis/TLogsRootsAndNorms`), `cosebis/Tplus_minus/T{plus,minus}<n>_0.50-100.00.table` (10^4-point T_+n/T_-n tables, theta in arcmin vs T), `cosebis/Benjamins_Bandpower/bandpower_filter_output_0_100.0_140.29.dat` (apodised W for the first KiDS-1000 band), and the ~1e-4 consistency between E_n from W_n and from xi (verification script).

## Appendix B — Verification artefacts (all mine, not kcap code)

Directory `/mnt/d/cloning_kids_Kappa_pipeline/docs/study/04_verification/`:
- `cosebis_roots_norms_from_scratch.py` — reconstructs T_+n roots and norms from the defining conditions; reproduces shipped values (n=1..4, 0.5'-300').
- `nicaea_logfft_hankel_check.py` — numpy implementation of the nicaea log-FFT Hankel transform vs direct quadrature (mu = 0, 4).
- `cosebis_En_cl_vs_xi_check.py` — E_n from W_n and from xi_+/- (with T_- formula) for a toy C(l): agreement ~1e-4.
- `bandpower_W_vs_reference_check.py` — apodised band-power weights vs `Benjamins_Bandpower` reference: agreement 1e-4..1e-3 (see the corrected 1e-4..3.5e-3 relative / <=1.1e-4 absolute statement in 2.6).
- [verified] `bandpower_W_spline_error_check.py` — added by the adversarial re-check: emulates kcap's 1000-point log-l natural-cubic-spline W table against direct quadrature for bands 1, 4, 8 (error 2.4e-7, 6.6e-6, 1.7e-3 pointwise; <= 3e-8 on a toy band power).
