"""
Task 21: robustness check. Regenerate the scaffold split with 2-3 different
seeds and see whether the best candidate's advantage holds up, or was partly
split-luck -- this dataset has shown real split-composition variance all
session (e.g. hybrid_gatv2's single-split RMSE of 0.9293 turned out to be
1.2081 under proper 5-fold CV; the original notebook's own val composition
had 3.5% acyclic molecules by shuffle luck in one case vs ~35% elsewhere).

For each seed, rebuilds a fresh scaffold_train_val_test_split from the same
raw source rebuild_scaffold_split.py uses, featurizes train/val for the
'mordred' feature set (the best single candidate: LightGBM+mordred, CV
RMSE=1.0106), retrains that exact config, and reports val RMSE. The new
seeds' test partitions are never loaded into memory beyond the split itself
-- only train/val are used here, consistent with never touching test data
for model selection.

Usage:
  python solubility_models/scripts/split_robustness_check.py --seeds 101 202 303
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.splitting import scaffold_train_val_test_split, assert_no_leakage
from common.evaluate import regression_metrics, format_metrics
from solubility_models.featurizer import build_feature_matrix
from solubility_models.models import build_tree_model

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SOURCE_DIR = os.path.join(ROOT, "misc", "original_buggy_split", "data_processed")

BEST_MODEL_TYPE = "lightgbm"
BEST_FEATURE_SET = "mordred"
BEST_PARAMS = dict(
    n_estimators=736, learning_rate=0.02210943033781764, num_leaves=180, max_depth=8,
    subsample=0.7566553417347286, colsample_bytree=0.7544350241810107,
    min_child_samples=59, reg_alpha=0.0008787301196251835, reg_lambda=0.00089760661834825,
)


def load_source_dataset():
    train = pd.read_csv(os.path.join(SOURCE_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(SOURCE_DIR, "test.csv"))
    full = pd.concat([train, test], ignore_index=True)
    return full[["ID", "Name", "InChIKey", "SMILES", "Solubility"]]


def acyclic_fraction(df):
    return float((df["scaffold"] == "").mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    args = parser.parse_args()

    df = load_source_dataset()
    print(f"Loaded {len(df)} molecules from {SOURCE_DIR}")

    results = []
    for seed in args.seeds:
        print(f"\n{'=' * 70}\nseed={seed}\n{'=' * 70}")
        train_df, val_df, test_df, stats = scaffold_train_val_test_split(df, seed=seed)
        assert_no_leakage(train_df, val_df, test_df)
        print(f"  sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        print(f"  acyclic fraction: train={acyclic_fraction(train_df):.3f} val={acyclic_fraction(val_df):.3f} test={acyclic_fraction(test_df):.3f}")

        X_train = build_feature_matrix(train_df["canonical_smiles"].tolist(), BEST_FEATURE_SET)
        X_val = build_feature_matrix(val_df["canonical_smiles"].tolist(), BEST_FEATURE_SET)
        y_train = train_df["Solubility"].values.astype(np.float32)
        y_val = val_df["Solubility"].values.astype(np.float32)

        model = build_tree_model(BEST_MODEL_TYPE, BEST_PARAMS, seed=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        m = regression_metrics(y_val, preds)
        print(f"  {BEST_MODEL_TYPE}[{BEST_FEATURE_SET}] val: {format_metrics(m)}")
        results.append({"seed": seed, "val_acyclic_frac": acyclic_fraction(val_df), **m})

    print(f"\n{'=' * 70}\nSUMMARY across seeds {args.seeds}\n{'=' * 70}")
    rmses = [r["rmse"] for r in results]
    for r in results:
        print(f"  seed={r['seed']}: RMSE={r['rmse']:.4f}  R2={r['r2']:.4f}  val_acyclic_frac={r['val_acyclic_frac']:.3f}")
    print(f"\nMean RMSE across seeds: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    print("Reference points: expanded_sweep_v3 5-fold CV RMSE=1.0106, DISSOLVR honest test RMSE=1.0802")


if __name__ == "__main__":
    main()
