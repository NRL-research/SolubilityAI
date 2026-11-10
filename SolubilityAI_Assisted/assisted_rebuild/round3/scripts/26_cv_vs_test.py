"""
26_cv_vs_test.py
================
How well does cross-validated RMSE predict held-out test RMSE across the 36 single
models now available?

This matters more than any individual result. Round 3's CV-best single model
(lightgbm_cons_huber_ca, CV 0.9983) came in at test 0.9667, WORSE than the incumbent
lightgbm[mordred] (CV 1.0136, test 0.9609). If CV rank and test rank are only loosely
related at this point, then further CV-driven search on this dataset has little
expected value no matter how many configurations are tried -- which is a concrete,
quantitative statement about where the project actually is.

Also estimates the smallest RMSE difference the 1,974-molecule test set can resolve,
from the width of the paired-bootstrap CIs already computed.

Output: results/cv_vs_test.csv, results/cv_vs_test.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

RES = "/home/claude/work/results"

df = pd.read_csv(f"{RES}/final_test_results_round3.csv")
singles = df[(df["stacker"] == "-") & df["cv_rmse"].notna()].copy()
singles = singles[~singles["config"].str.contains("round 2")]

rho, p_rho = stats.spearmanr(singles["cv_rmse"], singles["rmse"])
tau, p_tau = stats.kendalltau(singles["cv_rmse"], singles["rmse"])
pear, p_pear = stats.pearsonr(singles["cv_rmse"], singles["rmse"])

out = {
    "n_single_models": int(len(singles)),
    "spearman_rho": float(rho), "spearman_p": float(p_rho),
    "kendall_tau": float(tau), "kendall_p": float(p_tau),
    "pearson_r": float(pear), "pearson_p": float(p_pear),
}
print(f"Across {len(singles)} single models, CV RMSE vs test RMSE:")
print(f"  Spearman rho = {rho:.3f} (p={p_rho:.2g})")
print(f"  Kendall tau  = {tau:.3f} (p={p_tau:.2g})")
print(f"  Pearson r    = {pear:.3f} (p={p_pear:.2g})")

# restrict to the competitive band -- the regime any further search actually operates in
comp = singles[singles["cv_rmse"] < 1.06]
if len(comp) >= 5:
    r2, p2 = stats.spearmanr(comp["cv_rmse"], comp["rmse"])
    out["spearman_rho_competitive_band"] = float(r2)
    out["spearman_p_competitive_band"] = float(p2)
    out["n_competitive"] = int(len(comp))
    print(f"\nRestricted to the {len(comp)} models with CV RMSE < 1.06 "
          f"(the band any further search lives in):")
    print(f"  Spearman rho = {r2:.3f} (p={p2:.2g})")
    print("  -> within the competitive band, CV rank carries "
          f"{'essentially no' if abs(r2) < 0.3 else 'limited' if abs(r2) < 0.6 else 'useful'} "
          "information about test rank")

# resolvable difference, from the bootstrap CI widths already computed
comp_df = pd.read_csv(f"{RES}/statistical_comparisons_round3.csv")
half = ((comp_df["ci95_high"] - comp_df["ci95_low"]) / 2)
out["median_bootstrap_ci_halfwidth_rmse"] = float(half.median())
out["min_bootstrap_ci_halfwidth_rmse"] = float(half.min())
out["max_bootstrap_ci_halfwidth_rmse"] = float(half.max())
print(f"\nPaired-bootstrap 95% CI half-width on an RMSE gap, across "
      f"{len(comp_df)} comparisons: median {half.median():.4f} "
      f"(range {half.min():.4f}-{half.max():.4f})")
print(f"  -> on this 1,974-molecule test set, an RMSE improvement smaller than about "
      f"{half.median():.3f} cannot be distinguished from noise.")

singles = singles.sort_values("cv_rmse")
singles["cv_rank"] = range(1, len(singles) + 1)
singles["test_rank"] = singles["rmse"].rank().astype(int)
singles["rank_shift"] = singles["test_rank"] - singles["cv_rank"]
singles[["config", "cv_rmse", "rmse", "r2", "cv_rank", "test_rank", "rank_shift"]] \
    .to_csv(f"{RES}/cv_vs_test.csv", index=False)
json.dump(out, open(f"{RES}/cv_vs_test.json", "w"), indent=2)

pd.set_option("display.width", 200)
print("\nCV rank vs test rank (positive shift = did worse on test than CV suggested):")
print(singles[["config", "cv_rmse", "rmse", "cv_rank", "test_rank", "rank_shift"]]
      .head(15).to_string(index=False))
print(f"\nSaved {RES}/cv_vs_test.csv")
