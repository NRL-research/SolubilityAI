"""
Evaluate predictions completed so far - safe to run anytime in parallel with
the main pipeline script, since this only reads rag_results_batched.csv.

Reports overall RMSE/MAE/R2, plus a few diagnostic breakdowns:
  - metrics by confidence level (is Gemini's self-reported confidence calibrated?)
  - correlation between max_similarity and prediction error
  - the worst predictions, for manual inspection
"""

import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

INPUT_DIR = r"C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\PhD\AI\AqSolPred"
RESULTS_CSV = os.path.join(INPUT_DIR, "rag_results_batched.csv")

def main():
    df = pd.read_csv(RESULTS_CSV)
    n = len(df)
    print(f"Evaluating {n} completed predictions.\n")

    if n < 2:
        print("Not enough data yet for meaningful metrics (need at least 2 rows).")
        return

    df["error"] = (df["pred_logS"] - df["true_logS"]).abs()

    # --- Overall metrics ---
    rmse = np.sqrt(mean_squared_error(df["true_logS"], df["pred_logS"]))
    mae = mean_absolute_error(df["true_logS"], df["pred_logS"])
    r2 = r2_score(df["true_logS"], df["pred_logS"])

    print("=== OVERALL METRICS ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R2:   {r2:.3f}")
    print(f"(For reference: literature-reported solubility noise floor is ~0.6-0.8 logS units - "
          f"treat any RMSE noticeably below that with suspicion, not celebration.)\n")

    # --- Breakdown by Gemini's self-reported confidence ---
    print("=== METRICS BY SELF-REPORTED CONFIDENCE ===")
    print("(if this LLM's confidence is well-calibrated, 'high' should show lower error than 'low')")
    conf_summary = df.groupby("confidence")["error"].agg(["mean", "count"]).rename(
        columns={"mean": "mean_abs_error", "count": "n_molecules"})
    print(conf_summary.to_string())
    print()

    # --- Correlation between retrieval quality and error ---
    corr = df["max_similarity"].corr(df["error"])
    print("=== RETRIEVAL QUALITY vs ERROR ===")
    print(f"Correlation between max_similarity and absolute error: {corr:.3f}")
    print("(negative = expected: lower similarity to nearest neighbor -> higher error)\n")

    low_sim = df[df["max_similarity"] < 0.4]
    if len(low_sim) > 0:
        low_sim_mae = low_sim["error"].mean()
        print(f"For the {len(low_sim)} molecules with max_similarity < 0.4 (weak retrieval coverage):")
        print(f"  Mean absolute error: {low_sim_mae:.3f} (compare to overall MAE of {mae:.3f})\n")

    # --- Worst predictions, for manual inspection ---
    print("=== 10 WORST PREDICTIONS (by absolute error) ===")
    worst = df.sort_values("error", ascending=False).head(10)
    print(worst[["id", "smiles", "true_logS", "pred_logS", "error", "max_similarity", "confidence"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
