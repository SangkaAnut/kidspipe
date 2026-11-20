# ADOPTION_01 — their measured xi pushed through OUR code

Script: `scripts/dr5_adopt_01.py`. Run 2026-10-29 08:2x, exit 0.
Inputs: their 21-pair xi (1000 log bins) from
`/mnt/d/HI-gamma-correlation/pipeline/01_lensing_validation/outputs/xipm/fiducial/`,
our `kidspipe.measure.xi2cosebis` integrator, official KiDS-Legacy release files.
Outputs: `adoption.json`, `A1_fine_shape.json`, `A1_ascii_amplitude.json`, `ascii_vs_fits.json`,
`adopted_cosebis.npz`.

## Result table

| check | quantity | verdict |
|---|---|---|
| A3 xi vs official **xipm FITS**, 378 pts | xiP chi2=0.0303/189 = 0.0002, max\|d\|/sig=0.0528; xiM 0.0316/189, max 0.0438 | PASS, reproduces their 0.051σ |
| B1 their xi -> **our** E_n/B_n vs official COSEBIs | En 6m chi2=0.024/126 max 0.0408; En 20m 0.070/420 max 0.0455; Bn 6m 0.017/126 max 0.0299; Bn 20m 0.064/420 max 0.0520 | PASS — our integrator reproduces the release from *their* xi |
| B3 B_n null vs zero, **Bn covariance block** | 6m official chi2=151.058 pte=0.0635 / ours 150.876 pte=0.0648; 20m official 459.328 pte=0.0902 / ours 458.803 pte=0.0930 | PASS, matches published p-values |
| m-correction | En vs official: m-corrected chi2=0.024/126; raw chi2=4.793/126 | m-correction is real and needed |
| A1 their xi vs official **fine ascii**, 1000 bins | corr min 0.9772, median 0.9875, max 0.9957 | FAIL as written — see below |

> **SUPERSEDED in part (2026-10-29, PI decision in `../a1_salvage/A1_SALVAGE.md`).** A1 is now A1b-noise (PASS) and A3 gates on max|Δ|/σ < 0.1 (PASS); both are computed by the current `scripts/dr5_adopt_01.py` into `adoption.json`. The ascii is a blinded realisation of the same sample (m-uncorrected, x0.85). The 'bad reference / retire A1' discussion below is historical.

## A1 is a bad reference, not a shape problem

Correlation is scale-blind, so a correlation deficit cannot say "shape". Two things are true:

1. The fine ascii is itself **7–15% off the official FITS** in a near-separable way.
   Rebinning the ascii to the 9 official theta bins and dividing by the FITS
   (`ascii_vs_fits.json`), per-pair median ratio runs 0.80–0.95; a separable fit
   `log A = c_i + c_j + const` gives per-bin factors 0.929, 0.953, 0.967, 0.968, 0.995, 1.012
   and an overall 0.913 (xiP; xiM essentially the same). This is the caveat their own
   `REPORT.md` item 2 flags (0.893…0.971 per *tomographic* bin, Δχ²=21.6): the ascii is a
   different, probably blinded, realisation.
2. Their xi vs the ascii shows a median ratio of 1.01–1.20 with 13% relative scatter, and
   dividing out the documented per-bin ascii amplitude only moves the scatter 0.133 -> 0.122,
   i.e. it does **not** explain it. So the fine ascii disagrees with both their run and the FITS
   in a way no single per-bin scale fixes.

Conclusion: **A1 as specified cannot pass and should be retired.** The supported statement is
A3+B1: on the released 9-bin data vector and on COSEBIs our integrator reproduces the official
values from their xi at ≤0.05σ. The official fine ascii `cosmosis_xipm` files must not be used as
a truth reference for A1/A2 without the Wright confirmation their project says is still a TODO.

## Bug found and fixed in this script (do not reintroduce)

`KiDS_Legacy_cosebis.fits` `COVMAT` is 840x840 = **[E_n (420) | B_n (420)]**. The first version of
B3 built its index over the *E_n* block and used it for B_n, giving pte = 0.9453 (6m) / 0.8708
(20m) — plausible-looking and wrong. The B_n block is offset by `21*20 = 420`. With the offset the
reproduction is exact: 0.0635 and 0.0902, matching PRIOR_VALIDATION_01 §1 to 4 decimals. Any B_n
null test that does not hit the published 0.0635/0.0902 is using the wrong block.

## Still open

- sigma_e convention (0.4–7% high) — untouched here.
- Production run — not needed for A2/A3/B2/B3 anymore; these numbers come from their files.
- `docs/DR5_MEASUREMENT_REPORT.md` still to be written.
