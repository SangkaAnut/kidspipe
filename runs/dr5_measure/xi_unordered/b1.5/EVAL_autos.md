# Unordered auto-pair evaluation, source=ours, bin_slop=1.5

| pair | A1 boxcar max boxcar/white (<= 1.03; const-A_p test is below, not per pair) | A3 max|d|/sig xi+ / xi- (< 0.10) | E_n-off max/sig 6m / 20m | chi2 E_n 6 / 20 | B_n null chi2 6 (official) / 20 (official) | ours-vs-theirs fine z rms, 9-bin max/sigma_off: xi+ / xi- | npairs ratio | theirs A3 xi+ / xi- | theirs E_n20 max |
|---|---|---|---|---|---|---|---|---|---|
| Bin1_Bin1 | 0.787 PASS (boxcar only) | 0.0095 / 0.0097 PASS | 0.007 / 0.007 | 0.000 / 0.000 | 7.9 (off 7.9) / 26.5 (off 26.4) | 0.000, 0.000 / 0.000, 0.000 | 1.0000 | 0.0095 / 0.0097 | 0.007 |
| Bin2_Bin2 | 0.306 PASS (boxcar only) | 0.0070 / 0.0105 PASS | 0.009 / 0.009 | 0.000 / 0.000 | 5.7 (off 5.7) / 18.1 (off 18.1) | 0.000, 0.000 / 0.000, 0.000 | 1.0000 | 0.0070 / 0.0105 | 0.009 |
| Bin3_Bin3 | 0.358 PASS (boxcar only) | 0.0079 / 0.0140 PASS | 0.007 / 0.015 | 0.000 / 0.001 | 7.7 (off 7.7) / 28.7 (off 28.6) | 0.000, 0.000 / 0.000, 0.000 | 1.0000 | 0.0079 / 0.0140 | 0.015 |
| Bin4_Bin4 | 0.322 PASS (boxcar only) | 0.0302 / 0.0306 PASS | 0.020 / 0.034 | 0.001 / 0.005 | 3.3 (off 3.3) / 15.9 (off 15.7) | 0.000, 0.000 / 0.000, 0.000 | 1.0000 | 0.0302 / 0.0306 | 0.034 |
| Bin5_Bin5 | 0.354 PASS (boxcar only) | 0.0300 / 0.0315 PASS | 0.023 / 0.046 | 0.001 / 0.007 | 6.4 (off 6.6) / 17.8 (off 18.1) | 0.000, 0.000 / 0.000, 0.000 | 1.0000 | 0.0300 / 0.0315 | 0.046 |
| Bin6_Bin6 | 0.453 PASS (boxcar only) | 0.0238 / 0.0438 PASS | 0.041 / 0.043 | 0.004 / 0.008 | 19.1 (off 19.2) / 32.4 (off 32.6) | 0.000, 0.000 / 0.000, 0.000 | 1.0000 | 0.0238 / 0.0438 | 0.043 |

A1 one-constant A_p test over 6 pairs: xi+ pte 0.933 (C 1.169), xi- pte 0.130 (C 1.182); need pte > 0.01; A1 overall PASS (partial set: 6 of 21 pairs; gate calibrated on 21 pairs / 42 curves)

A3 = 9-bin max|Delta|/sigma_official after m-correction (1+m_i)(1+m_j) (m from dr5_adopt_01.py M, Wright+25 Table 1).
