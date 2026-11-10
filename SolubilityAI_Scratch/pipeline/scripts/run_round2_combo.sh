#!/bin/bash
# Combination chosen by in-domain CV only: avalon+erg helped on top of RDKit descriptors (0.941 -> 0.929).
cd "$(dirname "$0")/.."
while [ ! -f logs/r2_tab_done.flag ]; do sleep 30; done
python3 -u scripts/r2_04_sweep.py combo "desc+mordred+avalon+erg" > logs/r2_04_combo.log 2>&1
echo DONE > logs/r2_combo_done.flag
