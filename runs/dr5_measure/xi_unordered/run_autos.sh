#!/bin/bash
cd /mnt/d/cloning_kids_Kappa_pipeline
export PYTHONPATH=src OMP_NUM_THREADS=20 PYTHONUNBUFFERED=1
PY=/home/anut/anaconda3/bin/python3
$PY scripts/dr5_measure_xi_unordered.py run --pairs autos --bin-slop 1.5 --threads 20
echo RUN_EXIT=$?
$PY scripts/dr5_measure_xi_unordered.py eval --pairs autos --bin-slop 1.5
echo EVAL_EXIT=$?
