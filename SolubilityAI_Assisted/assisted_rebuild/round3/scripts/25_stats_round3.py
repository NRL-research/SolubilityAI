"""
25_stats_round3.py
==================
Paired bootstrap confidence intervals for every round-3 claim.

Same protocol as round 2 (scripts/10_statistical_tests.py): 10,000 resamples of the
test molecules, the same indices drawn for both models in each replicate so
per-molecule difficulty is controlled for. Nothing in the round-3 summary is stated as
an improvement unless it appears here with a CI that excludes zero.

Compares the round-3 candidates against BOTH round-2 champions:
  lightgbm[mordred]  the recommended single model (test RMSE 0.9609)
  greedy_blend       the best round-2 ensemble (test RMSE 0.9520)

Output: results/statistical_comparisons_round3.csv / .json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import metrics, paired_bootstrap  # noqa: E402

RES = "/home/claude/work/results"
d3 = np.load(f"{RES}/test_predictions_round3.npz")
d2 = np.load(f"{RES}/test_predictions.npz")
y = d3["y"]
assert np.allclose(y, d2["y"])

P = {k: d3[k] for k in d3.files if k != "y"}
for k in ("greedy_blend", "diverse_blend_4", "lightgbm[mordred]"):
    if k in d2.files:
        P.setdefault(k, d2[k])

cfg = json.load(open(f"{RES}/stacking_models.json"))["chosen_for_test"]
stacks = list(cfg.keys())

# best new single model by CV, for a like-for-like single-model claim
best_single_r3 = "lightgbm_cons_huber_ca[mordred+chem_aware]"

COMPARISONS = []
for s in stacks:
    COMPARISONS += [("greedy_blend", s), ("lightgbm[mordred]", s)]
COMPARISONS += [
    ("lightgbm[mordred]", best_single_r3),
    ("greedy_blend", best_single_r3),
    ("lightgbm[mordred]", "lightgbm_conservative[mordred]"),
    ("lightgbm[mordred]", "lightgbm_huber[mordred]"),
    ("lightgbm[mordred]", "lightgbm_cons_huber[mordred]"),
    ("lightgbm_conservative[mordred]", best_single_r3),
]
if len(stacks) >= 2:
    COMPARISONS.append((stacks[1], stacks[0]))

rows = []
for a, b in COMPARISONS:
    if a not in P or b not in P:
        print(f"skip {a} vs {b} (missing)")
        continue
    r = paired_bootstrap(y, P[a], P[b], n_boot=10000, seed=42, metric="rmse")
    ma, mb = metrics(y, P[a]), metrics(y, P[b])
    verdict = ("B better" if r["ci95_low"] > 0 else
               "A better" if r["ci95_high"] < 0 else
               "no significant difference")
    rows.append({"A": a, "B": b,
                 "A_rmse": round(ma["rmse"], 4), "B_rmse": round(mb["rmse"], 4),
                 "A_r2": round(ma["r2"], 4), "B_r2": round(mb["r2"], 4),
                 "gap_A_minus_B": round(r["gap_point_estimate_A_minus_B"], 4),
                 "ci95_low": round(r["ci95_low"], 4),
                 "ci95_high": round(r["ci95_high"], 4),
                 "frac_resamples_B_better": round(r["frac_resamples_B_better"], 4),
                 "verdict": verdict})
    print(f"{a}\n  vs {b}\n  RMSE {ma['rmse']:.4f} -> {mb['rmse']:.4f}  "
          f"gap={r['gap_point_estimate_A_minus_B']:+.4f}  "
          f"95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  "
          f"B better in {100*r['frac_resamples_B_better']:.1f}%  => {verdict}\n")

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/statistical_comparisons_round3.csv", index=False)
json.dump(rows, open(f"{RES}/statistical_comparisons_round3.json", "w"), indent=2)
pd.set_option("display.width", 240)
print(df.to_string(index=False))
print(f"\nSaved {RES}/statistical_comparisons_round3.csv")
