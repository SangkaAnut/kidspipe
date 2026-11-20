# DR5 cosmic-shear xi+/- and COSEBIs product (bin_slop 1.5)

File: `dr5_xipm_cosebis_binslop1p5.npz` (produced by `scripts/dr5_make_product.py`, checks in `FULLVECTOR.md`).
Measured from the KiDS-Legacy gold catalogue (our extract), TreeCorr 5.1.4, bin_slop 1.5, angle_slop 0.1 (TreeCorr default),
unordered pairs (auto: one catalogue; cross: two), catalogue `weight`, e1,e2 as stored (no flip), positions float64.
The exact (bin_slop 0) measurement was NOT run for all pairs; this is the PI-chosen option (a).

## Ordering and units
- 21 tomographic pairs in official order `(1,1),(1,2),...,(1,6),(2,2),...,(6,6)` (`pair_ids`, 1-based; same as the KiDS-Legacy FITS, pair-major).
- Fine arrays are `(21, 1000)`: 1000 log bins 2-300 arcmin. `r_nom` is the geometric bin centre in **arcmin**. xi is dimensionless.
- `xip_raw, xim_raw`: m-UNcorrected xi+/xi-. `npairs`: unordered pair counts; `weight`: sum of w_i w_j.
- `m_factor = (1+m_i)(1+m_j)`, m_gold = [-0.023, -0.016, -0.011, 0.02, 0.03, 0.045] (Wright+25 Table 1). `xip, xim` = raw / m_factor: the multiplicative shear-calibration correction applied
  to both statistics (shear scales as 1+m per galaxy, xi as the product). Use `xip, xim` for science and the *_raw arrays only to compare with m-uncorrected references (e.g. the official fine ascii).
- `xip9, xim9` `(21, 9)`: m-corrected xi rebinned to the 9 official bins (`theta9_edges`, log 2-300 arcmin) with npairs weights; compare with `KiDS_Legacy_xipm.fits`.
- `En, Bn` `(21, 20)`: COSEBIs modes 1-20 from the m-corrected fine xi (scheme `edges`, theta in radians, filters for 2-300 arcmin). Compare with `KiDS_Legacy_cosebis.fits`.
- Provenance: `bin_slop`, `angle_slop_effective`, `treecorr_version`, `position_dtype`, `source_files`, `source_sizes`, `source_sha256`, `provenance_json`.

## Caveats
- The official release used TreeCorr 4.2.3 at bin_slop 1.5; this run is TreeCorr 5.1.4. Agreement with the release rests on the 9-bin comparison (max |Delta|/sigma < 0.1, see FULLVECTOR.md);
  equality with the earlier pipeline's xi is code equivalence (same library version, settings and extract), not independent validation.
- Only the 9-bin comparison is sensitive to a common amplitude error, with resolution of order 1 %.
- bin_slop 1.5 differs from the exact estimate (largest effect on xi-, E_n up to ~0.4 sigma over 20 modes in Bin6_Bin6); see `runs/dr5_measure/angle_slop/ANGLE_SLOP.md`.
- The official fine ascii has an unexplained amplitude (ours/ascii ~1.17) and is not a truth reference.
- E_n/B_n comparisons with the release have no numeric gate. Errors are not included: use the official covariances (COVMAT of the FITS files).
- Covariance of E_n for a mode set: pair-major, mode fastest; the B_n block of the 840x840 COVMAT is offset by 420.
