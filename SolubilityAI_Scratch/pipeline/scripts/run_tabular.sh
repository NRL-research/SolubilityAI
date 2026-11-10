#!/bin/bash
# Runs all tabular sweep stages sequentially (logs in logs/).
cd "$(dirname "$0")/.."
for st in baselines hp_lgbm_desc hp_svr hp_mlp hp_lgbm_dm hp_xgb_dm catboost extra; do
  python3 -u scripts/04_cv_sweep.py $st > logs/04_$st.log 2>&1
done
echo ALLDONE > logs/04_done.flag
