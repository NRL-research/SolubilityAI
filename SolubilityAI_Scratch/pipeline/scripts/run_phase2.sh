#!/bin/bash
# Phase 2: v2 tabular re-runs, then GNNs (v1 graph-only, v1 hybrid, v2 hybrid). 80 epochs, seed 42.
cd "$(dirname "$0")/.."
while [ ! -f logs/04_done.flag ]; do sleep 30; done
SOL_DATA=v2 python3 -u scripts/04_cv_sweep.py v2_fixed > logs/04_v2_fixed.log 2>&1
python3 -u scripts/05_gnn.py gnn_mpnn 0 80 42 > logs/05_run_gnn_mpnn.out 2>&1
python3 -u scripts/05_gnn.py gnn_hybrid 1 80 42 > logs/05_run_gnn_hybrid.out 2>&1
SOL_DATA=v2 python3 -u scripts/05_gnn.py gnn_hybrid_v2 1 80 42 > logs/05_run_gnn_hybrid_v2.out 2>&1
echo DONE > logs/phase2_done.flag
