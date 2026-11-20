# DR5 exact xi+/xi- and COSEBIs (bin_slop 0)

`dr5_xipm_cosebis_exact.npz`: the exact TreeCorr measurement of all 21 KiDS-DR5 tomographic pairs, built by
`scripts/dr5_make_exact_product.py` from `runs/dr5_measure/xi_chunked/assembled/ns/` (chunked driver, 16 RA strips).
Settings: bin_slop 0, angle_slop 0, Log 1000 bins 2-300 arcmin, float64 positions, TreeCorr 5.1.4.
Same keys and conventions as the bin_slop 1.5 product (`../product/README.md`), plus `varxi` and `meanr`:

- `pair_ids` (21, 2) 1-based in official order; `m_factor` = (1+m_i)(1+m_j); `r_nom` (21, 1000) arcmin.
- `xip_raw, xim_raw` m-uncorrected; `xip, xim` = raw / m_factor; `xip9, xim9` npairs-weighted rebin to the 9 official bins.
- `En, Bn` (21, 20) from the m-corrected fine xi (`e_from_xi_binned`, scheme 'edges', unit 'rad').
- `npairs, weight, varxi` in the UNORDERED convention (as in the bin_slop 1.5 product). The chunked driver counts ordered pairs for
  auto pairs; those were halved (npairs, weight) and varxi doubled here. xi is unaffected.
- `meanr` (21, 1000): weight-weighted mean separation per fine bin, accumulated over the chunk files here. (The driver's
  assembled files carry the first chunk's meanr only; do not use that one.)
- `source_files/sizes/sha256`: the assembled files the product was built from.

Status: not gated (PI, 2026-11-02). Its difference from the bin_slop 1.5 product, per pair and as full vectors, is in
`EXACT_VS_B1P5.md` / `exact_vs_b1p5.json`. The release itself used bin_slop 1.5, so for comparisons with the release the
bin_slop 1.5 product is the like-for-like measurement; the exact product removes the bin_slop approximation.
Sessions: S1 2026-11-09 10:49:51 .. 2026-11-10 00:29:04; S2 2026-11-10 00:44:30 .. 13:37:56 (WSL restart ~08:50, resumed 08:56:23); S3 2026-11-10 19:55:18 .. 2026-11-11 08:48:42. Interim checks:
`runs/dr5_measure/xi_chunked/SESSION1_CHECK.md`, `SESSION2_CHECK.md`, `SESSION3_CHECK.md`.
