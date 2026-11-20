# A1 salvage — can "their xi reproduces the official fine ascii" be a real check?

Script: `scripts/dr5_a1_salvage.py` (re-runnable, ~1 min). Numbers: `a1_salvage/a1_salvage.json`.
Figures: `fig1_ascii_over_FITS_by_pair.png`, `fig2_their_over_ascii_residual_by_pair.png`,
`fig3_B_bin_xip_vs_xim.png`, `fig4_chi2_vs_theta.png`. Log: `run.log`.
Conventions: `A_p = their/ascii` per pair (ratio of npairs_weighted-weighted sums over the 1000 bins, NOT a
least-squares slope); `A_p = B_i B_j`; `s_i = 1/B_i` is the ascii/their factor. 9-bin rebinning always
`weighted_rebin` with the ascii `npairs_weighted`.

## Verdict in one paragraph

The fine ascii is **not** a different realisation and **not** a shape problem. It is the *same galaxy
sample* measured *without the m-bias correction* and multiplied by one global constant
`ascii = 0.85 x raw xi`. With that, the per-bin amplitude pattern is explained by 1 free parameter, the
residual against their raw xi is white-or-better, and A1 can be rewritten as a check with teeth (A1b-noise
below). The "release is internally inconsistent at ~8 sigma" number from the prompt does **not**
reproduce: with the official covariance the ascii-vs-FITS offset is chi2/dof 0.077 / 0.070, max 1.19 sigma.

## 1. The three numbers (all from `a1_salvage.json -> step1_*`)

| quantity | xi+ | xi- |
|---|---|---|
| ascii rebinned (`weighted_rebin`) vs official FITS, 189 pts, official cov | chi2/dof **0.0765**, max 1.19 sigma, median ratio 0.866 | **0.0699**, max 0.92 sigma, median ratio 0.865 |
| same, diagonal-only (sqrt of cov diagonal) | 0.334 | 0.192 |
| their xi rebinned (same weights) vs FITS | chi2/dof **0.00016**, max 0.052 sigma | **0.00017**, max 0.044 sigma |
| their xi, own npairs weights (adopt_01 style) | same to 4 digits (0.0002) | same |
| joint 378: ascii vs FITS / their vs FITS | 0.0572 / 0.00016 | |

* **The prompt's 3.34 / 3.51 and max 8.0 sigma are NOT reproduced.** I tried full cov, diagonal only, three
  weightings and an unweighted mean: nothing gives >0.4 (diag) or >0.12 (full). Only the *median ratios*
  (0.866/0.865) agree with the prompt's 0.879/0.858. The 8 sigma most plausibly came from a mis-indexed
  covariance slice; I could not locate it. Treat 3.34/8.0 as retracted.
* `weighted_rebin` vs `dr5_adopt_01.rebin9`: identical to the digits shown for their xi (both
  weight-by-pairs; own-npairs vs ascii-npairs weights give the same chi2).
* Their-xi-vs-FITS: reproduces A3 (0.0002). **Their xi is the right one**; the ascii is the odd one out.
* Their vs ascii at the 1000 fine bins: Pearson corr 0.981/0.989/0.996 (xi+), 0.977/0.986/0.995 (xi-);
  after one scale per pair the residual in units of the ascii fine-bin shape noise is rms 0.30 (xi+),
  0.32 (xi-), **lag-1 autocorrelation -0.43**, lag-5 0.01, and after a 20-bin boxcar the rms is 0.025/0.030
  versus 0.066/0.072 for white noise. i.e. the residual is *anti-correlated bin to bin* (pairs shuffled
  between neighbouring fine bins: TreeCorr bin_slop 1.5 in their run) and cancels on averaging. It is not
  structured in theta (fig2, fig4 right) and the pair-to-pair scatter of A_p is consistent with a single value (see 2).
  So (a) structured in theta: no; (b) structured in pair: only through the single constant + m pattern;
  (c) fine-bin jitter that averages away faster than white noise: yes.

Why the official covariance is blind to a 15 % offset: the offset is smooth and positive, so it lies in the
cosmic-variance modes of the covariance (xi+: the two largest-variance modes carry 5.3 of the 14.5 chi2;
cond. number 2.6e4). An injected global amplitude error on their xi against the FITS gives chi2/378 of
0.5, 3.0, 12, 27 for scale 1.02, 1.05, 1.10, 1.15 (`step4_A1b.A3_power_to_global_amplitude_error`).
**So A3 with the official covariance cannot detect a 10 % amplitude error** — "A3 passes at 0.0002" is
meaningful only because the agreement is 0.05 sigma, not because chi2/dof<1 is a stringent gate. Gate it on
chi2/dof < ~0.01 or max|d|/sigma < ~0.1 if it is to be a regression test.

## 2. Amplitude hypothesis (1000 bins, 600 moving-block bootstraps, block 25 bins, joint over pairs)

Fitted `their(m-corrected) = A_p asc`, `A_p = B_i B_j` (separable chi2/15: 6.2 xi+, 4.4 xi-, so separable).
`s_i = ascii/their`:

| bin | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| xi+ 1000-bin s_i | 0.894 | 0.918 | 0.925 | 0.933 | 0.953 | 0.961 |
| xi+ error | 0.017 | 0.007 | 0.012 | 0.005 | 0.006 | 0.011 |
| xi- 1000-bin s_i | 0.896 | 0.891 | 0.904 | 0.930 | 0.944 | 0.978 |
| xi- error | 0.017 | 0.011 | 0.010 | 0.004 | 0.005 | 0.006 |
| 9-bin GLS ascii/FITS xi+ (cov-dominated errors ~0.05) | 0.903 | 0.919 | 0.930 | 0.939 | 0.953 | 0.976 |
| 9-bin GLS ascii/FITS xi- | 0.922 | 0.906 | 0.908 | 0.935 | 0.954 | 0.975 |
| prior project (REPORT.md) | 0.893 | 0.913 | 0.917 | 0.932 | 0.949 | 0.971 |
| prompt's c_i converted (c_i*sqrt(0.913)) | 0.888 | 0.911 | 0.924 | 0.925 | 0.951 | 0.967 |

* **The three earlier patterns are the same numbers.** The prompt's c_i (0.929..1.012) split a 0.913 constant
  off the product; `c_i sqrt(0.913)` reproduces the prior project's s_i to <=0.01. They are not two
  measurements that disagree; they are one measurement with a different split of the overall constant.
  The 1000-bin fit agrees with both within ~1 sigma per bin, so the prior project's pattern is right.
* **xi+ vs xi- agree**: ln B_i differences have pulls (0.07, -2.17, -1.39, -0.40, -1.19, 1.33), chi2/6 = 10.0
  (pte 0.12; errors added in quadrature, xi+/xi- not jointly bootstrapped). Same factor in both statistics
  => calibration/normalisation offset, as predicted, not a shape error.
* **What the offset is.** With the m-UNcorrected xi, `their_raw = C * ascii` for ALL pairs:
  xi+ C = 1.1728 +/- 0.0062, chi2 13.4/20 (pte 0.86); xi- C = 1.1830 +/- 0.0051, chi2 18.3/20 (pte 0.57).
  Per-pair raw scales have no residual bin pattern (B_i^raw = 1.07..1.09 flat). The m-corrected xi
  fails the same one-constant test (xi+ chi2 45.5/20, xi- 106/20). C does not depend on theta (9 pooled ranges:
  xi+ 1.165..1.187, +/-0.005..0.03; xi- 1.16..1.20 apart from the noisy first bin 1.26+/-0.07).
  So **ascii = (0.853 +/- 0.004 xi+; 0.845 +/- 0.004 xi-) x m-uncorrected xi.** The (1+m_i) pattern
  (-2.3 %..+4.5 %) is the whole per-bin structure; sqrt(k)(1+m_i) with k=0.85 is plotted in fig3 (crosses)
  and lies on the data.
* The overall 0.85 (sqrt = 0.92) is unexplained. Candidates (untested, not claims): a different
  normalisation of the pair sum in the ascii pipeline, or a blinding scale. It needs the A. Wright
  confirmation the prior project already lists. It is not needed for any validated check.
* Attenuation trap: a least-squares slope `sum(t a / s^2)/sum(a^2/s^2)` gives A ~ 1.02-1.04 instead of
  1.17 because noise in `a` dilutes it (corr 0.98). The ratio of sums has no such bias. This is why
  my first look at the amplitude "disagreed" with the 9-bin ratios.

## 3. Alternative: is their xi the odd one out?

No. their xi vs FITS 0.0002 (A3), their xi through our COSEBIs vs official 0.0002 (B1 reproduced in the script
for all of En/Bn, 6/20 modes). The **ascii** through the same integrator: En 6 modes chi2/126 = 0.098
(max 1.58 sigma), 20 modes 0.054/420, Bn 20 modes 0.036/420 — it "passes" at chi2/dof<1 although its En is
a median 0.853 x the official (their xi: 0.998), for the same covariance reason as 1. Scaled per pair to their
xi it gives En 0.035/126 (max 0.65 sigma). So the premise "ascii passes COSEBIs and their xi passes, but
ascii-vs-their fails" is not quite what happens: **ascii-vs-their only 'fails' in correlation and amplitude,
and the COSEBIs chi2 cannot see either**. Where theta carries the difference (En 6 modes, ascii-their swapped
in one theta range at a time, total chi2 12.4/126; ranges 2-3.5, 3.5-6, 6-11, 11-19, 19-32, 32-56, 56-99,
99-172, 172-300 arcmin): 3.4, 1.3, 0.6, 0.7, 0.7, 0.45, 0.7, 1.0, 1.5 — largest at the smallest scales (where
the covariance is smallest), flat otherwise, i.e. a global amplitude, not a localised scale. xi chi2
partition for ascii-vs-FITS xi+: 4.7, 3.0, 1.8, 1.6, 1.4, 0.1, 0.6, 0.6, 0.7 (sum 14.5); xi-: flat 0.9-2.3 (13.2).

## 4. A1 forms

* **A1a (their xi vs the FITS 9 bins)** = A3. Redundant; **do not ship it under a second name.** Instead
  tighten A3's threshold (see 1).
* **A1b as literally specified (one scale per pair, official 378x378 cov)**: implemented
  (`step4_A1b`). their vs ascii: chi2 7.06/357 (pte 1.0); FITS vs ascii 7.01; their vs FITS 0.057. Passes, but
  has **no power**: an injected log-theta tilt of the xi (+/-eps across the range) reaches the pte=0.01 line
  (chi2=422) only at eps ~0.4 (chi2: 12.7 at 5 %, 28.7 at 10 %, 92 at 20 %, 197 at 30 %, 533 at 50 %). Same blindness as A3.
* **A1b-noise (RECOMMENDED; also in the script, `a1b_noise`)**: their RAW xi vs ascii, one scale per pair
  (ratio of sums), on the 1000 fine bins with the ascii's own per-bin shape-noise sigma. Pass if
  (i) on all 42 curves the 20-bin-boxcar residual rms is <= its white-noise expectation
  (`rms(z)/sqrt(20)`), i.e. no low-frequency excess; (ii) chi2 of ln A_p about one global constant has pte > 0.01
  for xi+ and xi- separately (bootstrap errors from the real data, dof 20).
  Why: both products are the same galaxies, so shape noise cancels and the residual can reveal differences
  far below the cosmic-variance covariance. Real data: **PASS** (max boxcar/white 0.61 xi+, 0.79 xi-;
  const pte 0.89 / 0.50, C = 1.173 / 1.182).
  Power (injected into their raw xi, `step4_A1b_noise.power_injections`): tilt eps=0.05, 0.1 pass (so (i) alone is
  weak: it flags a tilt at eps >= 0.2, max ratio 1.11); extra independent noise of 0.05 sigma_fine passes and
  0.1 sigma_fine fails; pair-to-pair scale scatter of 0.01, 0.02 (ln A) passes, 0.03 fails (pte 1e-6), 0.05 fails; feeding
  the m-corrected xi instead of raw **fails** (pte 4e-4 xi+, 8e-14 xi-). So it catches ~3 % pair-dependent
  amplitude errors, noise-level sample differences (>= 0.1 sigma_fine), and a wrong/absent m-correction; it does
  not catch a smooth shape tilt under ~20 %. Criterion (ii) is what gives it teeth; thresholds were
  chosen from the physics above (white-noise expectation; 1 % pte), not tuned on the injection results.
* **A1c (retire)**: not supported by the evidence. The ascii is usable once treated as `0.85 x raw`. The
  galaxy-level gate (`dr5_a1_gate.py`, PASS) remains the exact sample-identity statement; A1b-noise is the pair-level
  complement and is consistent with it.

## 5. What I did not do / limits

* No pair counting. No change to `dr5_a1_gate.py` or `dr5_adopt_01.py`.
* The m-uncorrected + 0.85 reading rests on their raw xi (their `xip_raw`), which I have not remeasured.
* xi+/xi- B_i comparison uses quadrature errors; a joint bootstrap would be cleaner.
* The 0.85 is empirical (xi+ 0.853, xi- 0.845; difference 1.2 sigma).

## PI decision needed

1. Adopt **A1b-noise** (their raw xi vs ascii, per-pair scale, thresholds above) as the replacement A1 and keep A3 as the
   9-bin check with its threshold tightened to chi2/dof<0.01 / max|d|/sigma<0.1; do not keep A1a as a separate name.
2. Retract the "ascii is a different/blinded realisation, shape problem, 8 sigma inconsistent" wording (it is the same sample,
   m-uncorrected, x0.85; confirming the 0.85 with A. Wright is optional, non-blocking).


## PI decision (2026-10-29)

1. **Adopted:** A1 = A1b-noise, implemented once in `src/kidspipe/measure/a1check.py` and run by
   `scripts/dr5_adopt_01.py` (result: PASS; boxcar/white 0.61/0.79, one-constant pte 0.89/0.50).
   A3 gate = **max |Delta|/sigma < 0.1** (real: 0.053 / 0.044, PASS). The name A1a is dropped.
2. **Wording:** the fine ascii is described as a **blinded realisation of the same galaxy sample**
   (m-uncorrected, amplitude x~0.85). Retracted: "different sample", "shape problem", "8 sigma internally
   inconsistent". Confirming the 0.85 with A. Wright is optional and non-blocking.

### Post-adoption finding: threshold calibration of criterion (i)
Found while writing `tests/test_a1check.py`: for a WHITE residual the boxcar/white ratio scatters around 1;
its maximum over 42 curves has median 1.20 and 99th percentile 1.33 (4000 simulations). So the adopted 1.0
rejects white residuals with probability 1. It passes the real comparison only because two estimates from the
same galaxies differ by anti-correlated jitter (lag-1 -0.43, simulated ratio ~0.4-0.5). The gate is therefore
valid for same-sample comparisons only, and the justification "<= white-noise level" in section 4 is
statistically wrong. On the real data, threshold 1.33 instead of 1.0 changes only the 20 % theta-tilt injection
(FAIL -> PASS; 30 % and 50 % still fail; pair scatter >= 3 %, extra noise >= 0.3 sigma and the m-corrected
input fail at both). Code keeps the signed-off 1.0; **PI to decide whether to switch to 1.33.**
Neither threshold detects a localised 10 % bump at 20' (below the fine-bin noise).

### Threshold follow-up (2026-10-29): refit null and per-curve standardisation
Details: `threshold_calibration/standardised/STANDARDISED.md` (build) and `.../standardised/verify/VERIFY.md` (independent check).

- **0.81 is retracted.** The earlier MA(2) null (`calib.log`) did not repeat the per-pair A_p refit. With the refit
  (simulated A_p scatter 2.4 % vs bootstrap 2.3 %) the 99th percentile of the max ratio over 42 curves is **1.03**.
  End-to-end false-fail: 1.0 -> ~9 %, 1.03 -> ~1 %, 0.81 -> ~79 %. Real max 0.787 (xi- 1-1, whose own null mean is
  0.89) has null pte 0.82; it is not borderline.
- **Per-curve standardisation does not work with per-curve lags.** ratio^2 is set by the low-frequency jitter power
  f(0) = 1 + 2 rho1 + 2 rho2 ~ 0.1, while one curve's fitted rho1, rho2 have noise ~0.03 / 0.05. Refitting the null from
  the data under test over-disperses the per-curve p-values: at the plug-in 1 % thresholds, Fisher false-fails ~50 %
  and Stouffer ~24 %, and min-p is worse. The plug-in powers (e.g. Fisher 0.98 for a 10 % tilt) fall to ~0.3 end-to-end.
  The plug-in real-data Fisher pte 0.0002 becomes 0.11 end-to-end, so **no shape difference is indicated.**
- `a1b_noise_standardised` stays in `a1check.py` as an **experimental, non-gating** option; the default `a1b_noise` is
  unchanged (0.606/0.787, pte 0.889/0.496). It must not be used as a gate unless lag-fit noise is put into the null
  (or lags are pooled across curves) and power is re-measured with the same fit applied to injected data.
- **PI to decide:** keep 1.0 (signed off, ~9 % false-fail on correct measurements) or move to 1.03 (1 % size).
  Either passes the real data. Power is low either way: tilt >= 20-30 %, bump >= 30 %, no single-pair sensitivity.

### PI decision (2026-10-29): threshold of criterion (i) moved 1.0 -> 1.03
`A1B_THRESH["boxcar_over_white_max"] = 1.03` in `src/kidspipe/measure/a1check.py` (1 % false-fail under the
same-sample MA(2) null with the A_p refit). `scripts/dr5_adopt_01.py` re-run: the only change in
`adoption/adoption.json` is the threshold field; A1 still PASS (0.606 / 0.787, pte 0.889 / 0.496), A3 unchanged.
The "PI to decide" items above (1.33; 1.0 vs 1.03) are closed by this decision.
