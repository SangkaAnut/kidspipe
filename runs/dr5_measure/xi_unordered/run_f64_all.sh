#!/bin/bash
# PI-approved 2026-11-01: float64 re-extraction -> 6 autos -> 15 crosses at bin_slop 1.5, official style.
cd /mnt/d/cloning_kids_Kappa_pipeline
export PYTHONPATH=src OMP_NUM_THREADS=20 PYTHONUNBUFFERED=1
PY=/home/anut/anaconda3/bin/python3; R=scripts/dr5_measure_xi_unordered.py
$PY scripts/dr5_extract_check_f64.py || { echo CHAIN_ABORT=extract_check; exit 1; }
[ -d runs/dr5_measure/xi_unordered/b1.5 ] && [ ! -d runs/dr5_measure/xi_unordered/b1.5_f32 ] && \
  mv runs/dr5_measure/xi_unordered/b1.5 runs/dr5_measure/xi_unordered/b1.5_f32
mkdir -p runs/dr5_measure/xi_unordered/b1.5
$PY $R run --pairs autos --bin-slop 1.5 --threads 20 || { echo CHAIN_ABORT=run_autos; exit 1; }
$PY $R eval --pairs autos --bin-slop 1.5 && cp runs/dr5_measure/xi_unordered/b1.5/EVAL.md runs/dr5_measure/xi_unordered/b1.5/EVAL_autos.md
echo AUTOS_EVAL_EXIT=$?
$PY $R run --pairs all --bin-slop 1.5 --threads 20 || { echo CHAIN_ABORT=run_all; exit 1; }
$PY $R eval --pairs all --bin-slop 1.5
echo ALL_EVAL_EXIT=$?
echo CHAIN_DONE
