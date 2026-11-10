"""
The single, final, honest test-set evaluation for this round. Test data is
touched here for the first and only time -- everything upstream (expanded
sweep, hybrid_gatv2 batch, ensembling, split-robustness check) used only
train/val, never test.

Trains final versions of the 4 diverse candidates (chosen in ensemble_cv.py)
on the FULL train+val pool, predicts once on the held-out test set, and
reports individual + ensembled test metrics against DISSOLVR's honest test
reproduction (RMSE=1.0802+/-0.0023, R2=0.7963+/-0.0009, see
solubility_models/results/final_test_results.csv).

Blend weights and the stacking meta-learner are fit ONLY on 5-fold
out-of-fold (OOF) predictions over train+val (recomputed here identically to
ensemble_cv.py, deterministic given fixed seeds) -- never on test -- then
applied as-is to the test-set predictions. This keeps the ensemble
combination step leak-free.

Usage:
  python solubility_models/scripts/final_test_evaluation.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.splitting import scaffold_group_kfold
from common.evaluate import regression_metrics, format_metrics
from solubility_models.models import build_tree_model, build_sklearn_model
from solubility_models.ensemble import optimize_blend_weights, weighted_blend, StackingMetaLearner

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
RESULTS_PATH = os.path.join(ROOT, "solubility_models", "results", "final_test_results_round2.csv")

N_FOLDS = 5

CANDIDATES = [
    dict(name="lightgbm[mordred]", family="tree", model_type="lightgbm", feature_set="mordred",
         params=dict(n_estimators=736, learning_rate=0.02210943033781764, num_leaves=180, max_depth=8,
                     subsample=0.7566553417347286, colsample_bytree=0.7544350241810107,
                     min_child_samples=59, reg_alpha=0.0008787301196251835, reg_lambda=0.00089760661834825)),
    dict(name="catboost[mordred]", family="tree", model_type="catboost", feature_set="mordred",
         params=dict(iterations=749, learning_rate=0.03231778142523447, depth=7,
                     l2_leaf_reg=8.130699985257577, bagging_temperature=2.6296309222169056,
                     random_strength=2.6996450535168885e-05)),
    dict(name="hist_gradient_boosting[rdkit_full+engineered]", family="sklearn",
         model_type="hist_gradient_boosting", feature_set="rdkit_full+engineered",
         params=dict(max_iter=794, learning_rate=0.05362853964441846, max_depth=10,
                     max_leaf_nodes=118, l2_regularization=1.6934490731313353e-08)),
    dict(name="extra_trees[rdkit_descriptors]", family="sklearn", model_type="extra_trees",
         feature_set="rdkit_descriptors",
         params=dict(n_estimators=104, max_depth=30, min_samples_leaf=7, max_features=0.9461903383828855)),
]

_CACHE = {}


def load_split_features(split, feature_set):
    key = (split, feature_set)
    if key not in _CACHE:
        _CACHE[key] = np.load(os.path.join(CACHE_DIR, f"{split}_{feature_set.replace('+', '_')}.npy"))
    return _CACHE[key]


def trainval_features(feature_set):
    return np.vstack([load_split_features("train", feature_set), load_split_features("val", feature_set)])


def build_model(c, seed=42):
    if c["family"] == "tree":
        return build_tree_model(c["model_type"], c["params"], seed=seed)
    return build_sklearn_model(c["model_type"], c["params"], seed=seed)


def get_trainval_df_and_y():
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    y = trainval["Solubility"].values.astype(np.float32)
    return trainval, y


def compute_oof_predictions(trainval, y):
    """Same procedure as ensemble_cv.py's run_ensemble_experiment -- recomputed
    here (deterministic given fixed seeds) so blend weights / stacking are
    fit only on train+val, never on test."""
    folds = scaffold_group_kfold(trainval, n_splits=N_FOLDS, seed=42)
    oof_preds = {c["name"]: np.zeros(len(y), dtype=np.float64) for c in CANDIDATES}
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"  OOF fold {fold_i}...")
        for c in CANDIDATES:
            X = trainval_features(c["feature_set"])
            model = build_model(c)
            model.fit(X[train_idx], y[train_idx])
            oof_preds[c["name"]][val_idx] = model.predict(X[val_idx])
    return oof_preds


def main():
    trainval, y_trainval = get_trainval_df_and_y()
    y_test = np.load(os.path.join(CACHE_DIR, "test_y.npy"))

    print("=" * 70)
    print("Step 1: recompute OOF predictions on train+val to fit ensemble weights (no test data touched)")
    print("=" * 70)
    oof_preds = compute_oof_predictions(trainval, y_trainval)
    for name, preds in oof_preds.items():
        print(f"  {name} OOF: {format_metrics(regression_metrics(y_trainval, preds))}")

    weights = optimize_blend_weights(oof_preds, y_trainval)
    print(f"\n  Fitted blend weights (from train+val OOF only): {weights}")

    meta = StackingMetaLearner("ridge").fit(oof_preds, y_trainval)

    print("\n" + "=" * 70)
    print("Step 2: train final candidates on FULL train+val, predict on TEST (first touch of test.csv this round)")
    print("=" * 70)
    test_preds = {}
    rows = []
    for c in CANDIDATES:
        X_trainval = trainval_features(c["feature_set"])
        X_test = load_split_features("test", c["feature_set"])
        model = build_model(c)
        model.fit(X_trainval, y_trainval)
        preds = model.predict(X_test)
        test_preds[c["name"]] = preds
        m = regression_metrics(y_test, preds)
        print(f"  {c['name']} TEST: {format_metrics(m)}")
        rows.append({"candidate": c["name"], "kind": "single_model", "split": "test", **m})

    unweighted_mean = np.mean(list(test_preds.values()), axis=0)
    m_mean = regression_metrics(y_test, unweighted_mean)
    print(f"\n  unweighted_mean TEST: {format_metrics(m_mean)}")
    rows.append({"candidate": "unweighted_mean", "kind": "ensemble", "split": "test", **m_mean})

    weighted = weighted_blend(test_preds, weights)
    m_weighted = regression_metrics(y_test, weighted)
    print(f"  weighted_blend TEST (weights from train+val OOF): {format_metrics(m_weighted)}")
    rows.append({"candidate": "weighted_blend", "kind": "ensemble", "split": "test", "weights": str(weights), **m_weighted})

    stacked = meta.predict(test_preds)
    m_stacked = regression_metrics(y_test, stacked)
    print(f"  stacking_ridge TEST (meta fit on train+val OOF): {format_metrics(m_stacked)}")
    rows.append({"candidate": "stacking_ridge", "kind": "ensemble", "split": "test", **m_stacked})

    print("\n" + "=" * 70)
    print("REFERENCE: DISSOLVR honest test reproduction (5 seeds, from prior session)")
    print("=" * 70)
    print("  RMSE=1.0802+/-0.0023, R2=0.7963+/-0.0009 (see solubility_models/results/final_test_results.csv)")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
