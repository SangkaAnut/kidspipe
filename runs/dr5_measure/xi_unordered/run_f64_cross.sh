#!/bin/bash
# Resume of run_f64_all.sh after the OOM kill (2026-11-01, 14.6 GB RSS on Bin1_Bin6, memory accumulated over
# 4 crosses in one process). One process per pair so memory is returned to the OS; peak RSS logged per pair.
cd /mnt/d/cloning_kids_Kappa_pipeline
export PYTHONPATH=src OMP_NUM_THREADS=20 PYTHONUNBUFFERED=1
PY=/home/anut/anaconda3/bin/python3; R=scripts/dr5_measure_xi_unordered.py
for p in 1-6 2-3 2-4 2-5 2-6 3-4 3-5 3-6 4-5 4-6 5-6; do
  echo "PAIR_START $p $(date +%T) free=$(free -g | awk '/Mem:/{print $7}')G"
  /usr/bin/time -f "PAIR_RSS $p maxrss_kB=%M elapsed=%e exit=%x" $PY $R run --pairs $p --bin-slop 1.5 --threads 20 \
    || { echo CHAIN_ABORT=run_$p; exit 1; }
done
$PY $R eval --pairs all --bin-slop 1.5
echo ALL_EVAL_EXIT=$?
echo CHAIN_DONE
