# Verifier "exactness" record (C1, driver default, test fix)

Scripts: scripts/verify_angle_slop_exactness.py (a), scripts/verify_angle_slop_exactness_driver.py (b,c; runs the production
driver main() with OUTROOT redirected). Outputs: exactness_a.json/.log, exactness_tmp/{default,b0a0}/. Production files untouched.

## (a) C1 on real Bin6 sub-sample
Region ra 20-40, dec -36..-20 (KiDS-S; not patch A/B), 30000 random galaxies (seed 7, 786,847 in region), 1000 log bins 2-300 arcmin,
sep_units arcmin, driver metric (TreeCorr default Euclidean = chord), 6 threads. 3.15e8 ordered pairs in range.
Own independent numpy brute: haversine chord separation, great-circle bearing at each galaxy (north through east), ordered pairs i!=j.
Max |diff| over 1000 bins (xi scale max 0.041):

| setting | vs TC brute: xi+ / xi- / npairs | vs own numpy: xi+ / xi- / npairs |
|---|---|---|
| TC brute=True | 0 | 1.6e-9 / 1.4e-9 / 0 |
| (0,0) | 3.5e-17 / 4.9e-17 / 0 | 1.6e-9 / 1.4e-9 / 0 |
| (0,default 0.1) | 2.2e-8 / 2.1e-6 / 0 | 2.2e-8 / 2.1e-6 / 0 |
| (1.5,default) | 2.7e-3 / 4.4e-3 / 3528 (822 bins differ) | same |
Times (30k, 6 thr): brute 4.5 s, (0,0) 2.9 s, (0,def) 3.1 s -> (0,0) not more expensive here (C1 cost claim consistent; production-scale cost not rechecked).

C1 verdict: exactness PASS, but "~1e-14 vs brute" holds only against TreeCorr's own brute (3e-17). Against an independent spherical
numpy brute the residual is 1.6e-9 (30k) / 1.1e-8 (3k, scales ~1/sqrt(bin pairs): per-pair ~1e-8 error); I reproduced this at
long-double precision, so it is a real small difference in TreeCorr's projection (possibly float32 somewhere), not roundoff in mine.
Irrelevant at the science level (<<1e-4 sigma) but the 1e-14 wording should be "vs TreeCorr brute".
Gotchas I hit (my bugs): catalogue ra/dec are float32 on disk (make_treecorr_cat casts to float64; casting before my brute is mandatory);
driver metric is the chord (Euclidean), not arc: using arc separations mis-bins 1418 of 3.15M pairs at 3000 galaxies. Binning is in
chord units (differs from true angle by theta^2/24, 3e-4 at 300'); this is a property of the existing production setup, flagged for the lead.

## (b) default behaviour unchanged
Chunk pair Bin6_Bin6 c05_06 (16.2 s originally; 38 s now with 5 threads). Re-run through current driver, default settings, 16 chunks.
Same key set as production file. npairs, r_nom, fold bitwise identical. Others NOT bitwise (thread summation order): xip/xim max abs 2.2e-17
(rel 2e-11), meanr 3.4e-11 abs (1.3e-13 rel), weight rel 6.9e-14, varxi rel 7e-14. No angle_slop/bin_slop keys written (schema unchanged).
Verdict: default unchanged to floating-point summation order (~1e-11 rel).

## (c) (0,0) same chunk vs pilot b0_a0_chunks/c05_06.npz
Same key set apart from production-time 'seconds' (present in the pilot file only; reported as key diff), angle_slop=0 and bin_slop=0 recorded
and bitwise equal. npairs bitwise; xip/xim max abs 3e-17 (rel 1e-11..3e-11), weight rel 8e-14. Reproducible.
For reference: pilot (0,0) vs production default chunk differ by xim max abs 1.7e-6 (relative up to 0.95 in near-zero bins), xip 3e-10: the
angle_slop effect is real and confined to xi-.

## (d) test fix (tests/test_measure.py:818-841)
`python -m pytest tests/test_measure.py -q` (OMP 4 threads): 57 passed, 8.5 s.
The original demand (rtol 1e-9 vs module brute()) was unsatisfiable: brute() uses a flat mean-dec position angle (test line 60-80), a
different (approximate) projection; replacing the reference with TreeCorr brute=True is legitimate for the exactness claim. Weaknesses:
 1. TreeCorr-brute reference is not independent of TreeCorr's projection code; the independent flat brute is now only bounded at atol 5e-5 on
    xi+ and not checked at all for xi-. My independent spherical brute agrees to 1e-9, so a tight (1e-6) bound against a spherical-bearing numpy brute
    would restore independence and would also have caught a wrong-convention bug.
 2. Discrimination of (0,0) from default rests on bf comparison rtol 1e-9 / atol 1e-12; the default deviates from brute by 2e-6 in xi- on my
    sample, so it would fail that check - fine, but the default-vs-0.1 equality test (rtol 1e-12) is the only guard on the default.
 3. npairs line vs 2*ref atol 6/rtol 1e-3 is loose but npairs is separately asserted equal to bf.npairs exactly.
Verdict: not a real weakening of exactness vs TreeCorr, but independence is reduced; recommend adding a spherical-bearing numpy brute.

## Not checked
C2-C6 (full-pair numbers, interpretation, cost tables); production-scale timing; A3.
