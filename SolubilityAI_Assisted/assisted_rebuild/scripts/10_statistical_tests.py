"""
10_statistical_tests.py
=======================
Paired bootstrap confidence intervals on every model-vs-model comparison this project
reports. 10,000 resamples of the test molecules, resampled jointly for both models in
each replicate so per-molecule difficulty is controlled for.

No comparison is stated as a finding anywhere in the report unless it appears here.

Output: results/statistical_comparisons.csv / .json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common_eval import metrics, paired_bootstrap  # noqa: E402

RES = "/home/claude/work/results"
d = np.load(f"{RES}/test_predictions.npz")
y = d["y"]
P = {k: d[k] for k in d.files if k != "y"}

COMPARISONS = [
    # (A, B) -- reported as "is B better than A?"
    ("catboost_solo[solubility_ai]", "greedy_blend"),
    ("catboost_solo[solubility_ai]", "SolubilityAI_ensemble"),
    ("catboost_solo[solubility_ai]", "lightgbm[mordred]"),
    ("SolubilityAI_ensemble", "greedy_blend"),
    ("SolubilityAI_ensemble", "diverse_blend_4"),
    ("lightgbm[mordred]", "greedy_blend"),
    ("lightgbm[mordred]", "diverse_blend_4"),
    ("diverse_blend_4", "greedy_blend"),
    ("diverse_blend_4", "diverse_blend_4_plus_gnn"),
    ("gatv2[graph+5desc]", "catboost_solo[solubility_ai]"),
    ("xgboost_branch[solubility_ai]", "catboost_solo[solubility_ai]"),
]

rows = []
for a, b in COMPARISONS:
    if a not in P or b not in P:
        print(f"skip {a} vs {b} (missing predictions)")
        continue
    r = paired_bootstrap(y, P[a], P[b], n_boot=10000, seed=42, metric="rmse")
    ma, mb = metrics(y, P[a]), metrics(y, P[b])
    verdict = ("B better" if r["ci95_low"] > 0 else
               "A better" if r["ci95_high"] < 0 else
               "no significant difference")
    rows.append({
        "A": a, "B": b,
        "A_rmse": round(ma["rmse"], 4), "B_rmse": round(mb["rmse"], 4),
        "A_r2": round(ma["r2"], 4), "B_r2": round(mb["r2"], 4),
        "rmse_gap_A_minus_B": round(r["gap_point_estimate_A_minus_B"], 4),
        "ci95_low": round(r["ci95_low"], 4), "ci95_high": round(r["ci95_high"], 4),
        "excludes_zero": r["excludes_zero"],
        "frac_resamples_B_better": round(r["frac_resamples_B_better"], 4),
        "verdict": verdict,
    })
    print(f"{a}  vs  {b}\n  RMSE {ma['rmse']:.4f} -> {mb['rmse']:.4f}  gap={r['gap_point_estimate_A_minus_B']:+.4f} "
          f"95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  "
          f"B better in {100*r['frac_resamples_B_better']:.1f}% of resamples  => {verdict}")

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/statistical_comparisons.csv", index=False)
with open(f"{RES}/statistical_comparisons.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved {RES}/statistical_comparisons.csv")
