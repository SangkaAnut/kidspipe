# Chunking the 21-pair production run

Answering: "can we make the workload into small chunks?" — **yes, and it is exact, not an
approximation.** This note records the proof, the driver, and the honest cost.

## 1. The fold rule (the one thing that must be right)

TreeCorr counts **ordered** pairs. For a partition of the sky into chunks `{c}`:

```
process(cat, cat)  ==  sum_c process(c_c, c_c)  +  2 * sum_{a<b} process(c_a, c_b)
```

The factor **2** on cross-chunk pairs is not optional. Omitting it — the obvious first
implementation — loses **3.3%** of the pairs (measured: npairs ratio 0.9672). That number is
dangerously close to the size of effect we are chasing, so it would have been easy to mistake for
a real result. Recorded here so nobody re-derives it: `chunk_diagnose.log`.

For a **cross** tomographic pair (bins i≠j) every ordered chunk-pair `(a,b)` is computed and there
is **no fold** — chunk a of bin i with chunk b of bin j and chunk b of bin i with chunk a of bin j
are different galaxy sets.

## 2. Proofs, at three levels

| test | script | result |
|---|---|---|
| fold rule vs single shot, K=4 | `dr5_chunk_diagnose.py` | npairs ratio 1.000000000, w ratio 1.000000000; native TreeCorr patches also 1.000000000 |
| fold rule, K=8, auto **and** cross pair | `dr5_chunk_verify.py` | Bin1_Bin1 36 chunk-pairs: npairs 1.000000000000, w 1.000000000008, xi+ corr 1.000000000000, max pull 1.4e-4 sigma. Bin1_Bin2 64 chunk-pairs: same, max pull 7.1e-5 sigma |
| full driver path (write → assemble), 2% subsample, K=4 | `dr5_chunk_proof.py` | npairs 1.000000000000, max abs diff **0**, sum(weight) 1.000000000001, xi+ corr 1.000000000000, max pull 8.5e-5 sigma → **PASS** |

`runs/dr5_measure/chunk_proof.json` holds the machine-readable result.

## 3. What the driver gives you

`scripts/dr5_measure_xi_chunked.py`. The unit of work is **(tomographic pair, chunk-a, chunk-b)**,
each written to its own npz the moment it finishes.

```
runs/dr5_measure/xi_chunked/
  chunks/Bin{i}_Bin{j}/c{a:02d}_{b:02d}.npz
  assembled/ns/Bin{i}_Bin{j}.npz      # same schema as dr5_measure_xi.py
  manifest.json
```

Commands:

```bash
# progress + honest cost estimate
PYTHONPATH=src $PY scripts/dr5_measure_xi_chunked.py --status --nchunks 16

# run one chunk-pair (this is what you farm out to a worker)
PYTHONPATH=src $PY scripts/dr5_measure_xi_chunked.py --nchunks 16 --pairs 1-1 --chunk 3-7

# run everything not yet done, sequentially
PYTHONPATH=src $PY scripts/dr5_measure_xi_chunked.py --all --nchunks 16 --threads 16

# merge finished chunk-pairs into the 21 pair files
PYTHONPATH=src $PY scripts/dr5_measure_xi_chunked.py --assemble --nchunks 16
```

Properties that matter for a 3-day job:
- **Resumable by construction.** A finished chunk-pair is never recomputed (existence check).
- **Parallelisable across processes or machines** without any coordination: each worker is given
  a disjoint set of `--chunk` ids and writes disjoint files. No locks, no shared state.
- **`--assemble` refuses incomplete pairs** ("INCOMPLETE 1/136 chunk-pairs, skipping"), so a
  half-measured tomographic pair can never silently enter the analysis. Verified.
- **Chunk ids are reproducible**: equal-count RA strips from quantile edges on the catalogue
  itself, so chunk 3-7 means the same galaxies in every session and on every machine.

Chunks are equal-**count** (not equal-area) so the work per chunk is balanced. The decomposition is
exact for any cut of the sky; only the load balance depends on the choice, so equal-count is the
right default.

## 4. Cost — measured, and it confirms the original estimate

Measured this session (`chunked_smoke.log`): Bin1_Bin1 chunk 0-0 (463,524 galaxies each leg)
produced **1.300e11 pairs in 538.8 s = 2.41e8 pairs/s**, a full-density chunk.

Calibrating the geometry: an independent 10% run gives full Bin1_Bin1 = 2.4183e12 **ordered** pairs
inside 2–300 arcmin. TreeCorr counts ordered pairs, so an auto pair has `n(n−1)` of them — the
fraction inside the cone is 2.4183e12 / n1(n1−1) = **0.04397**. Applying that across all 21 pairs:

```
total ORDERED pairs over 21 pairs      : 9.763e+14
calibrated fraction inside 300 arcmin  : 0.04397   (measured)
pairs actually binned                  : 4.292e+13
at the measured 2.41e8 pairs/s         : ~49 h
```

**~49 h**, i.e. the handoff's 30–50 h range was right and my first recalculation of 85 h was wrong.

The 85 h came from a factor-of-2 bookkeeping error: I divided by `n1²/2` (an *unordered* pair
count) to get the in-cone fraction, then applied that fraction to a sum of *ordered* counts. Both
`n(n−1)` and `n²/2` differ by exactly 2 for 7.4M galaxies, so the error was a clean 2×. Recorded
here because the same trap will catch anyone mixing TreeCorr's ordered convention with the usual
`n(n−1)/2` combinatorics.

With chunking this is not a 2-day single point of failure. Options:
- **sequential, resumable**: ~49 h, can be interrupted freely;
- **parallel workers**: 4656 chunk-pairs at `--nchunks 16`; near-diagonal ones cost ~540 s and
  dominate, distant ones are seconds. Note TreeCorr is already thread-parallel internally, so more
  *processes* do not reduce total CPU — they reduce wall time only if you have spare cores beyond
  the 16 threads, and each worker holds ~1.5 GB of bin catalogues against a 10 GB budget. On this
  22-core box the honest gain from 2–3 processes is modest and carries an RSS risk.
- **recommended**: sequential `--all`, resumable, started when the machine is otherwise idle.

## 5. What is NOT yet done

The driver is proven but the production run has **not** been started — it needs the budget
decision. When it is approved, the sequence is:

```bash
PYTHONPATH=src $PY scripts/dr5_measure_xi_chunked.py --all --nchunks 16 --threads 16
PYTHONPATH=src $PY scripts/dr5_measure_xi_chunked.py --assemble --nchunks 16
PYTHONPATH=src $PY scripts/dr5_a1_gate.py --from-npz runs/dr5_measure/xi_chunked/assembled/ns
```
