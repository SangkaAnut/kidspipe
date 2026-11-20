# dr5_amplitude_check.py — INCONCLUSIVE, do not quote

`runs/dr5_measure/amplitude_check.log` is not a usable amplitude test. Reasons:

1. **The official 9-bin xiP vector for Bin1_Bin1 contains literal zeros** in bins 3, 7, 8, 9
   (centres 8.04, 74.58, 130.15, 227.11 arcmin). Dividing by those produces the meaningless
   ratios 89.9, -9.8, 4.8 seen in the log.
2. **The 10% subsample is far noisier than assumed**: measured shape-noise ratio vs the official
   covariance is 6.58x, not the 3.16x expected for sqrt(1/F). The official `COVMAT` diagonal is a
   full (jackknife/analytic) covariance, not shape noise, so the two are not comparable this way.
   With per-bin pulls of 0.1-1.8 sigma on a 10% sample, no amplitude statement can be made.
3. Averaging 1000 fine bins into 9 wide bins suppresses shape noise by ~sqrt(N) and leaves the
   comparison dominated by the signal's own bin-averaging convention (Eq. F.1 of 2503.19441
   requires weighting by N_pair), which this script does not implement.

Deciding the xi_+ amplitude at the required precision IS check A2/A3 and needs the full-sample
production run. Do not substitute this file for it.
