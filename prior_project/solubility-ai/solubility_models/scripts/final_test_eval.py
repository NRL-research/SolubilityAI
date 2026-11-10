"""
Final, single test-set evaluation. Test set is touched exactly once here,
for a small fixed set of already-decided candidates (decided via validation
in baselines.py / tree_sweep.py / ensemble_search.py -- nothing here is
chosen or tuned based on what this script finds).

Candidates:
  1. DISSOLVR reproduction: same hyperparameters as the original
     dissolvr/train.py (untuned), 5 seeds.
  2. Best tuned model from the 585-trial Optuna sweep: XGBoost on
     rdkit_descriptors with the winning hyperparameters (val RMSE=0.8137,
     R2=0.7452). The GNN blend/stacking candidates are NOT re-evaluated here
     -- validation already showed conclusively they don't help (blend
     converges to 94% XGBoost weight, stacking gains ~0.0002 R2, both within
     noise) -- there's no reason to spend the one test look confirming a
     result we already know.

Both are retrained on train+val COMBINED (standard practice for a final
model once hyperparameter selection is done -- val's job was tuning, not
held-out evaluation once we reach this step) before predicting on test.

Writes solubility_models/results/final_test_results.csv.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solubility_models.models import build_tree_model
from common.evaluate import regression_metrics, aggregate_seed_runs, format_metrics, format_aggregated

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
RESULTS_PATH = os.path.join(ROOT, "solubility_models", "results", "final_test_results.csv")

SEEDS = [42, 101, 123, 456, 789]

BEST_XGB_PARAMS = {
    "n_estimators": 887,
    "learning_rate": 0.03483547337022728,
    "max_depth": 7,
    "subsample": 0.6210145986218286,
    "colsample_bytree": 0.6154184981135327,
}


def load_cached(split, feature_set):
    return np.load(os.path.join(CACHE_DIR, f"{split}_{feature_set.replace('+', '_')}.npy"))


def load_trainval(feature_set):
    X_train = load_cached("train", feature_set)
    X_val = load_cached("val", feature_set)
    y_train = np.load(os.path.join(CACHE_DIR, "train_y.npy"))
    y_val = np.load(os.path.join(CACHE_DIR, "val_y.npy"))
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    return X, y


def append_result(rows):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(RESULTS_PATH):
        df = pd.concat([pd.read_csv(RESULTS_PATH), df], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)


def already_done(model_name):
    if not os.path.exists(RESULTS_PATH):
        return False
    df = pd.read_csv(RESULTS_PATH)
    return model_name in set(df["model"])


def run_dissolvr_final():
    if already_done("dissolvr_reproduction_TEST"):
        print("DISSOLVR final test eval already done, skipping.")
        return
    print("\n" + "=" * 70)
    print("FINAL TEST EVAL: DISSOLVR reproduction (train+val combined -> test)")
    print("=" * 70)
    X_trainval, y_trainval = load_trainval("dissolvr_full")
    X_test = load_cached("test", "dissolvr_full")
    y_test = np.load(os.path.join(CACHE_DIR, "test_y.npy"))
    print(f"  train+val: {X_trainval.shape}, test: {X_test.shape}")

    for seed in SEEDS:
        t0 = time.time()
        model = build_tree_model(
            "catboost",
            {"iterations": 10000, "learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 5},
            seed=seed,
        )
        model.fit(X_trainval, y_trainval)
        preds = model.predict(X_test)
        m = regression_metrics(y_test, preds)
        elapsed = time.time() - t0
        print(f"  seed={seed}  {format_metrics(m)}  ({elapsed:.1f}s)")
        append_result([{"model": "dissolvr_reproduction_TEST", "seed": seed, "split": "test",
                         "feature_set": "dissolvr_full", **m}])


def run_best_xgb_final():
    if already_done("best_tuned_xgboost_TEST"):
        print("Best XGBoost final test eval already done, skipping.")
        return
    print("\n" + "=" * 70)
    print("FINAL TEST EVAL: Best tuned XGBoost (train+val combined -> test)")
    print("=" * 70)
    X_trainval, y_trainval = load_trainval("rdkit_descriptors")
    X_test = load_cached("test", "rdkit_descriptors")
    y_test = np.load(os.path.join(CACHE_DIR, "test_y.npy"))
    print(f"  train+val: {X_trainval.shape}, test: {X_test.shape}")
    print(f"  params: {BEST_XGB_PARAMS}")

    t0 = time.time()
    model = build_tree_model("xgboost", BEST_XGB_PARAMS, seed=42)
    model.fit(X_trainval, y_trainval)
    preds = model.predict(X_test)
    m = regression_metrics(y_test, preds)
    elapsed = time.time() - t0
    print(f"  {format_metrics(m)}  ({elapsed:.1f}s)")
    append_result([{"model": "best_tuned_xgboost_TEST", "seed": 42, "split": "test",
                     "feature_set": "rdkit_descriptors", **m}])


if __name__ == "__main__":
    run_dissolvr_final()
    run_best_xgb_final()

    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS (touched exactly once per candidate)")
    print("=" * 70)
    df = pd.read_csv(RESULTS_PATH)
    dissolvr_rows = df[df["model"] == "dissolvr_reproduction_TEST"][["rmse", "mae", "r2"]].to_dict("records")
    if dissolvr_rows:
        agg = aggregate_seed_runs(dissolvr_rows)
        print("DISSOLVR reproduction (5 seeds): " + format_aggregated(agg))
    xgb_row = df[df["model"] == "best_tuned_xgboost_TEST"].iloc[-1]
    print(f"Best tuned XGBoost (1 seed):     RMSE={xgb_row['rmse']:.4f}  MAE={xgb_row['mae']:.4f}  R2={xgb_row['r2']:.4f}")
