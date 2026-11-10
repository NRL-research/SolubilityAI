#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "r2_04_sweep.py features" > /dev/null; do sleep 20; done
python3 -u scripts/r2_04_sweep.py models > logs/r2_04_models.log 2>&1
echo DONE > logs/r2_tab_done.flag
