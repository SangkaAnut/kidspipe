# RECOMPUTE: independent A3 and A1-boxcar for all 21 pairs

Command: `PYTHONDONTWRITEBYTECODE=1 /home/anut/anaconda3/bin/python3 <scratchpad>/rc.py` (script kept outside the repo at
/tmp/claude-1000/-home-anut/8ecc82c7-22dd-490c-b045-63f93bce2233/scratchpad/rc.py). No evaluator/official/a1check function was called.

## Independence
- Official 9-bin xi+/xi-: read with astropy directly from KiDS_Legacy_xipm.fits (xiP, xiM, COVMAT 378x378). Each row is located from the file's own
  BIN1/BIN2/ANGBIN columns (dict keyed (bin1,bin2,angbin)), NOT official.PAIRS. sigma = sqrt(diag(COVMAT)) (xi+ rows 0..188, xi- rows 189..377).
  File xiM has the same BIN1/ANGBIN as xiP (asserted).
- Ours: runs/dr5_measure/xi_unordered/b1.5/Bin{I}_Bin{J}.npz (r_nom, xip, xim, npairs, weight).
- Rebin: edges exp(linspace(ln2, ln300, 10)); fine bin in [E_c, E_c+1) (last bin includes 300.01); xi_bin = sum(w xi)/sum(w), w = our npairs.
- m-correction: xi / ((1+m_i)(1+m_j)), M = [-0.023,-0.016,-0.011,0.020,0.030,0.045] (matches scripts/dr5_adopt_01.py line 46).
- A1 boxcar: official ascii (cosmosis_xipm/XI_KiDS_Legacy_NScomb_..., read with genfromtxt), A_p = sum(w ours)/sum(w ascii) with w = ascii npairs_weighted,
  z = (ours - A_p ascii)/sigma_ascii, 50 consecutive 20-bin means, ratio = std(boxcar)/(std(z)/sqrt(20)).

## Results (all 21 pairs; max |d|/sigma over 9 bins, xi+ / xi-; recomputed vs eval.json)
| pair | A3 xi+ | A3 xi- | eval.json xi+ / xi- | A1 boxcar xi+ / xi- | eval.json boxcar | A_p xi+ / xi- |
|---|---|---|---|---|---|---|
| Bin1_Bin1 | 0.0095 | 0.0097 | 0.0095 / 0.0097 | 0.3535 / 0.7870 | 0.3535 / 0.7870 | 1.201 / 1.950 |
| Bin1_Bin2 | 0.0101 | 0.0100 | 0.0101 / 0.0100 | 0.3788 / 0.3284 | 0.3788 / 0.3284 | 1.161 / 1.191 |
| Bin1_Bin3 | 0.0094 | 0.0072 | 0.0094 / 0.0072 | 0.3773 / 0.3147 | 0.3773 / 0.3147 | 1.131 / 1.193 |
| Bin1_Bin4 | 0.0212 | 0.0175 | 0.0212 / 0.0175 | 0.4185 / 0.3512 | 0.4185 / 0.3512 | 1.316 / 1.178 |
| Bin1_Bin5 | 0.0275 | 0.0311 | 0.0275 / 0.0311 | 0.3655 / 0.2990 | 0.3655 / 0.2990 | 1.164 / 1.177 |
| Bin1_Bin6 | 0.0260 | 0.0262 | 0.0260 / 0.0262 | 0.4503 / 0.4328 | 0.4503 / 0.4328 | 1.198 / 1.211 |
| Bin2_Bin2 | 0.0070 | 0.0105 | 0.0070 / 0.0105 | 0.3060 / 0.2790 | 0.3060 / 0.2790 | 1.147 / 1.179 |
| Bin2_Bin3 | 0.0097 | 0.0081 | 0.0097 / 0.0081 | 0.2898 / 0.3497 | 0.2898 / 0.3497 | 1.161 / 1.205 |
| Bin2_Bin4 | 0.0140 | 0.0124 | 0.0140 / 0.0124 | 0.3378 / 0.3970 | 0.3378 / 0.3970 | 1.183 / 1.228 |
| Bin2_Bin5 | 0.0227 | 0.0119 | 0.0227 / 0.0119 | 0.3257 / 0.3193 | 0.3257 / 0.3193 | 1.150 / 1.199 |
| Bin2_Bin6 | 0.0133 | 0.0144 | 0.0133 / 0.0144 | 0.4491 / 0.3417 | 0.4491 / 0.3417 | 1.148 / 1.195 |
| Bin3_Bin3 | 0.0079 | 0.0140 | 0.0079 / 0.0140 | 0.3139 / 0.3580 | 0.3139 / 0.3580 | 1.176 / 1.240 |
| Bin3_Bin4 | 0.0143 | 0.0145 | 0.0143 / 0.0145 | 0.3449 / 0.4044 | 0.3449 / 0.4044 | 1.184 / 1.186 |
| Bin3_Bin5 | 0.0241 | 0.0337 | 0.0241 / 0.0337 | 0.3484 / 0.3393 | 0.3484 / 0.3393 | 1.141 / 1.215 |
| Bin3_Bin6 | 0.0216 | 0.0184 | 0.0216 / 0.0184 | 0.6061 / 0.3686 | 0.6061 / 0.3686 | 1.322 / 1.161 |
| Bin4_Bin4 | 0.0302 | 0.0306 | 0.0302 / 0.0306 | 0.3222 / 0.3197 | 0.3222 / 0.3197 | 1.180 / 1.207 |
| Bin4_Bin5 | 0.0420 | 0.0346 | 0.0420 / 0.0346 | 0.3559 / 0.4077 | 0.3559 / 0.4077 | 1.185 / 1.196 |
| Bin4_Bin6 | 0.0362 | 0.0178 | 0.0362 / 0.0178 | 0.3809 / 0.3473 | 0.3809 / 0.3473 | 1.188 / 1.168 |
| Bin5_Bin5 | 0.0300 | 0.0315 | 0.0300 / 0.0315 | 0.3536 / 0.3345 | 0.3536 / 0.3345 | 1.167 / 1.189 |
| Bin5_Bin6 | 0.0528 | 0.0310 | 0.0528 / 0.0310 | 0.4385 / 0.4231 | 0.4385 / 0.4231 | 1.183 / 1.165 |
| Bin6_Bin6 | 0.0238 | 0.0438 | 0.0238 / 0.0438 | 0.4517 / 0.4533 | 0.4517 / 0.4533 | 1.171 / 1.139 |

Summary: max A3 = 0.0528 (Bin5_Bin6 xi+; Bin6_Bin6 xi- 0.0438, Bin4_Bin5 xi+ 0.0420, Bin1_Bin1 0.0095/0.0097); max boxcar/white = 0.787 (Bin1_Bin1 xi-) <= 1.03.
Max |difference| vs eval.json: A3 = 0.0 and boxcar = 0.0 (identical to all printed digits, all 21 pairs x 2 stats). Required pairs
Bin5_Bin6, Bin6_Bin6, Bin4_Bin5, Bin1_Bin1 plus 17 more all agree.

Weight-choice robustness (A3 recomputed with weights = our `weight`, or the official ascii npairs_weighted): max change vs ours-npairs weighting
is ~2e-3 sigma (e.g. Bin5_Bin6 xi+ 0.0528 / 0.0510 / 0.0521; Bin2_Bin6 xi+ 0.0133 / 0.0153 / 0.0144); all pairs stay <= 0.053, far below 0.1.

## Pair -> official row mapping check
Pair-major order (1,1),(1,2),...,(1,6),(2,2),... confirmed from the FITS BIN1/BIN2 columns (my lookup uses them directly and reproduces eval.json exactly,
so official.PAIRS indexing is equivalent). Wrong mappings (A3 max, sigma units; right value in brackets):
- Bin5_Bin6 [0.053]: next row (6,6) 4.7; previous row (5,5) 2.9; reversed row (1,2) 15.7
- Bin6_Bin6 [0.044]: next (wraps to 1,1) 20.5; previous (5,6) 7.6
- Bin4_Bin5 [0.042]: column-major index (2,6) 6.2; next (4,6) 3.0; previous (4,4) 2.6; reversed (1,5) 9.7
- Bin1_Bin1 [0.010]: next (1,2) 4.3; previous (6,6) 10.2
- Bin2_Bin4 [0.014]: 2.3-3.8; Bin3_Bin6 [0.022]: 2.3-4.6; Bin1_Bin6 [0.026]: 1.6-4.4
Other negative controls: xi+ <-> xi- swapped gives 3.2-7.2 sigma; omitting the m-correction gives 0.12-0.92 sigma (Bin1_Bin1 0.123/0.124 would FAIL,
Bin6_Bin6 0.92), so the m-correction is also load-bearing and correctly applied (the agreement to 0.05 sigma is only possible with it).
Wrong mappings give 30-2000x larger deltas (all > 1.6 sigma), so the test is highly discriminating and the mapping is right.

## Notes / minor observations (not discrepancies)
- Bin1_Bin1 xi- has A_p = 1.95 (others ~1.14-1.32; Bin1_Bin3, Bin1_Bin4, Bin3_Bin6 xi+ 1.13-1.32): large scale factor vs the blinded ascii, but this gate is the
  one-constant test (reported pte 0.496), which I did not recompute (out of scope); boxcar for that curve is 0.787 (<=1.03).
- A1 per-pair here is the boxcar part only; the const-A_p pte (0.889 / 0.496) was not recomputed.
- Ours-vs-theirs equivalence, B_n and E_n were not checked here.

VERDICT: CONFIRMED
Discrepancies: none (A3 and A1-boxcar for all 21 pairs match eval.json to full printed precision; mapping verified by wrong-mapping controls).
