# Independent verification of A1 salvage

Script: `scripts/dr5_a1_salvage_verify.py` (own rebin on r_nom, own FITS reading by BIN1/BIN2/ANGBIN, own covariance slicing,
own nonlinear GLS fit; does not import `kidspipe.measure.official`). Outputs: `verify/verify.json`, `verify/run.log`.
Cross-checked against `a1_salvage.json`, `A1_SALVAGE.md`, `RESUME.md`, `ADOPTION_01.md`.

## Verdict
The colleague's central numbers reproduce. No material error found, so nothing was appended to A1_SALVAGE.md.
The prompt's 3.34 / 3.51 / 8 sigma does not reproduce for me either.

## Structural checks (the traps listed in the task)
- COVMAT layout: header says STRT_0=0 NAME_0=xiP, STRT_1=189 NAME_1=xiM. xiP = `cov[:189,:189]`, xiM = `cov[189:,189:]`. Using the xiM block for xiP gives chi2/dof 0.232 (not 3.34), so a wrong block is not the origin of the prompt's number.
- FITS rows are pair-major with ANGBIN 1..9 and identical for xiP and xiM (checked). Pair order (1,1),(1,2)...(6,6) used consistently for ascii, their npz and FITS.
- r_nom vs meanr for the rebin selection and weights: identical chi2 (meanr differs from r_nom by <3.2e-4, no bin changes membership). No effect.
- Median vs weighted rebin: median-rebin gives median ratio 0.891 / 0.889 and chi2/dof 0.39 / 0.50, so it does not reproduce 3.34 either. The prompt's 0.879/0.858 median ratios are not matched exactly by any rebin I tried (weighted 0.866/0.865).
- Own npairs vs ascii npairs_weighted weights for their xi: same chi2 to 3 digits. Ordered vs unordered only changes weights by a constant factor, so it cannot matter here.

## Findings table (ranked by severity; none is material)
| # | claim | colleague | verifier | agree? |
|---|---|---|---|---|
| 1 | prompt's 3.34/3.51, 8 sigma not reproducible | 0.0765/0.0699, max 1.19/0.92 | xiP 0.0765 (chi2 14.45/189, max 1.19), xiM 0.0699 (13.21/189, max 0.92); diag-only 0.334/0.192 | yes |
| 2 | median ratio ascii/FITS | 0.866 / 0.865 | 0.866 / 0.865 | yes |
| 3 | their xi vs FITS | 0.00016, 0.052 sigma / 0.00017, 0.044 | 0.000160 (0.0302/189), 0.0521 / 0.000165 (0.0312/189), 0.0438 | yes |
| 4 | B_i-pattern xi+ (s_i=ascii/their, 1000 bins) | 0.894 0.918 0.925 0.933 0.953 0.961 | independent 9-bin GLS ascii/FITS: 0.897 0.916 0.927 0.937 0.952 0.974 (err 0.03-0.07) | yes, within errors |
| 5 | s_i xi- | 0.896 0.891 0.904 0.930 0.944 0.978 | 9-bin GLS: 0.915 0.902 0.905 0.933 0.953 0.974 | yes, within errors |
| 6 | 9-bin GLS values | xi+ 0.903..0.976, xi- 0.922..0.975 | 0.897..0.974, 0.915..0.974 (nonlinear fit of ascii = s_i s_j FITS vs their linearised FITS/ascii; differences <=0.02, well inside 0.03-0.08 errors) | yes |
| 7 | overall constant raw: C=1.173 (xi+) | 1.1728 | mean of per-pair plain-sum ratios of raw xi+/ascii: 1.1714 (per-pair rms scatter 1.6%) | yes |
| 8 | raw per-bin pattern flat | 1.07..1.09 flat | 1/B raw xi+: 0.925 0.929 0.930 0.915 0.919 0.926 (flat); corrected: 0.904..0.968 rising, matching 1/(1+m) pattern | yes |
| 9 | 1000-bin pattern matches prior (0.893..0.971) | "within ~1 sigma per bin" | xi+ yes (<=1 sigma). xi- bin 2: 0.891 vs prior 0.913 is 2.0 sigma; prompt-converted bin 4 xi+ is 1.6 sigma off | mostly; "~1 sigma" slightly generous |
| 10 | 9-bin data cannot discriminate prior vs prompt pattern | not stated | confirmed: 9-bin errors 0.03-0.08 exceed pattern differences (~0.01-0.04). Only the 1000-bin fit has power | note, not an error |
| 11 | xi+ and xi- B_i agree | pulls up to -2.17, chi2/6=10.0 pte 0.12 | not independently refit (xi- plain-sum ratios are unusable because xi- changes sign; I used the GLS route: 9-bin xi+ vs xi- agree to <=0.02) | consistent |
| 12 | swap-by-theta chi2 "concentrated at 2-3.5 arcmin, otherwise flat, global amplitude" | 3.4,1.3,0.6,0.7,0.7,0.45,0.7,1.0,1.5; total 12.4 | JSON matches. Per-range values sum to 10.3, not 12.4 (cross terms), and the last range (1.5) is also above the middle. "Global amplitude" is inferred from the C-vs-theta fit, not from this table | numbers agree; interpretation slightly loose |
| 13 | A3 power: scale 1.02/1.05/1.10/1.15 -> chi2/378 0.5,3.0,12,27 | as stated | JSON `A3_power_to_global_amplitude_error` matches (0.507, 2.99, 11.99, 27.04) | yes (read from JSON, not re-injected) |
| 14 | A1b-noise power numbers (tilt, noise, scatter, wrong-m) | as in md | md matches JSON exactly. Caveat: scatter 0.01 gives min pte 0.0117 and 0.02 gives 0.0244, i.e. both sit near the 0.01 line and pte is not monotone in the injection, so the "passes at 0.02, fails at 0.03" edge is one random realisation | md = JSON; edge is noisy |
| 15 | literal A1b chi2 7.06/357 | as stated | JSON 7.0556/357 | yes (read, not rerun) |
| 16 | JSON field `sigma_their_over_sigma_ascii_median` uses xi+ sigma for both | disclosed | not in md; not used by any verdict | known |
| 17 | RESUME.md A1 row and bug list 5, 7 | as stated | numbers (C 1.173/1.183, chi2 13.4/20, 18.3/20, 0.077/0.070, 1.19 sigma) match JSON; ADOPTION_01.md has the "SUPERSEDED in part" line | yes |

## Residual caveats
- The m-uncorrected + 0.85 reading depends on the prior project's `xip_raw` column. My check (item 7, 8) used that same column, so it is not independent of it.
- Items 13-15 were verified against the JSON only; I did not re-run the injections.
- I did not reproduce the COSEBIs numbers (En 6 mode 0.098 etc.) or the bootstrap errors; JSON and md agree with each other there.
