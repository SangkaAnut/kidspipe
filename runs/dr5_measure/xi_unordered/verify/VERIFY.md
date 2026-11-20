# VERIFY (critic): slop 1.5 unordered auto pairs

Read RECOMPUTE.md, OURSVSTHEIRS.md, ROBUSTNESS.md. Own spot-checks (eval.json + npz, B6):
- eval.json A3 B6 = 0.0458 / 0.0633 (all six pairs < 0.1, A1 boxcar/white 0.28-0.79, all < 1.03).
- B6 ours vs theirs xi+ (fine bins, z = diff/sqrt(varxi)): mean -5e-6, rms 0.143, max 0.56, lag-1 -0.52; npairs ratio 1.0000001, weight ratio 1.0000001.

## (1) Confirmed?
Yes, with caveats. Three independent recomputations agree with EVAL.md to <=0.003 (A3, A1, E_n, B_n chi2 vs official). Caveats: the A1 pass is a 6-pair partial verdict (12 curves, dof 5; xi- constant-A pte 0.109, driven by Bin6, same in theirs); B6 E_n(20-mode) max 0.120 sigma exceeds 0.1 (2.8x theirs), which matters only if an E_n gate exists (PI); angle_slop and e1/e2 sign are attested only by manifest/runner, not stored in the npz.

## (2) Ours vs theirs
Inputs identical (e1, e2, weight, order exactly equal) except ra/dec float32 (ours) vs float64 (theirs; max shift 0.055 arcsec). That changes pair-bin assignment at edges: per-bin npairs differ in ~100% of bins (rms 5e-5) while totals agree to 1e-5, so the 1.0000 ratio hides it. Effect on xi is zero-mean, anticorrelated between adjacent bins (lag-1 ~ -0.5), 0.10-0.15 sigma rms per fine bin, no bin >0.7 sigma, 9-bin <=0.041 sigma. A 200k test showed the same size at bin_slop 0 (so it is a position-precision effect, not slop), row-order noise exactly 0. It does threaten margin modestly: B6 A3 xi- moves 0.044 -> 0.063 and xi+ 0.005 -> 0.046; float32 alone could move A3 by ~0.04 (inferred from subsample, not full run). Margin remains 0.037 (B6 xi-). Not a coherent bias; no gate fails.

## (3) Fixes for evaluator/runner
- Replace "ours-vs-theirs max rel diff" (8e2..2.8e4, divides by near-zero xi) with fine-bin z rms/max/lag-1, 9-bin max|d|/sigma_official, and per-bin npairs relative diff.
- Store angle_slop, sign convention, ra/dec dtype, peak RSS, load/nproc and per-pair seconds in npz/manifest.
- State the partial-set power (12 curves, dof 5) in EVAL.md A1 text; report E_n20 against PI gate explicitly.
- Official ascii npairs is 1.41-1.43x ours (weight 2.00x, ordered vs unordered): document as convention, gate-neutral.
- Consider float64 positions for the cross run (loader casts to float32); decide with PI whether auto pairs need a float64 re-run (cost ~76 min for 6 autos).

## (4) 15 cross pairs
Memory: ~5.1 GB extrapolated (peak ~5.5-6 GB) vs 11-12 GB available (free -g: 15 total, 12 available); fine. Time: cross ~1.9x auto per pair; estimated 5.5-7.1 h, up to ~9 h if the Bin5/Bin6-type slowdown (ns/pair 0.57 -> 1.22 over the auto run, unexplained, host throttling not excludable) recurs. Theirs: crosses 1176-1393 s each, 19780 s total at their settings. Tests: 64 passed. Order Bin5/Bin6 pair early and log time vs position to separate data vs elapsed-time effects.

## (5) Missing
- Full-catalogue float64 comparison (forbidden to run here); the f32 effect on A3 is inferred.
- Cause of Bin5/Bin6 slowdown; peak RSS of the real run.
- TreeCorr 4.2.3 (official) vs 5.1.4 untested.
- 21-pair A1 verdict (needs crosses); E_n gate status from PI.
