#!/bin/bash
# Phase 3: Mordred models -> blends.  (Test evaluation is launched manually afterwards.)
cd "$(dirname "$0")/.."
while [ ! -f logs/phase2_done.flag ]; do sleep 30; done
while pgrep -f 03b_features_mordred > /dev/null; do sleep 30; done
python3 -u scripts/04_cv_sweep.py mordred > logs/04_mordred.log 2>&1
python3 -u scripts/06_blend.py > logs/06_blend.log 2>&1
echo DONE > logs/phase3_done.flag
