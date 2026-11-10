"""
Task 20: ensembling and multi-seed bagging, scored via the same 5-fold
scaffold-grouped CV protocol as expanded_sweep.py, so results stay
apples-to-apples with that study's best (LightGBM+mordred, CV RMSE=1.0106).

Two experiments:
1. Multi-seed bagging of the single best config (LightGBM+mordred): retrain
   with N different seeds within each fold, average the N predictions before
   scoring that fold, see if it reduces variance / improves mean CV RMSE
   over the single-seed number.
2. Ensemble across a handful of DIVERSE top trials from expanded_sweep_v3
   (different model type and/or feature set, not just hyperparameter
   variants of the same thing) via out-of-fold (OOF) predictions: unweighted
   mean, weighted blend (solubility_models.ensemble.optimize_blend_weights),
   and a Ridge stacking meta-learner (solubility_models.ensemble.StackingMetaLearner).
   Note: blend weights / stacking are fit on the same OOF predictions used to
   report their RMSE, so those two numbers carry a small optimism bias
   relative to the single-model and unweighted-mean numbers (which don't fit
   anything on the eval data) -- flagged here rather than adding a second
   nested CV layer, given the diminishing returns at this stage.

Usage:
  python solubility_models/scripts/ensemble_cv.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.splitting import scaffold_group_kfold
from common.evaluate import regression_metrics, format_metrics
from solubility_models.featurizer import ALL_FEATURE_SETS  # noqa: F401 -- keeps import order (tree libs before torch) consistent with expanded_sweep.py
from solubility_models.models import build_tree_model, build_sklearn_model
from solubility_models.ensemble import optimize_blend_weights, weighted_blend, StackingMetaLearner

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")

N_FOLDS = 5

# Best single config from expanded_sweep_v3 trial 45: LightGBM + mordred, CV RMSE=1.0106.
BEST_MODEL_TYPE = "lightgbm"
BEST_FEATURE_SET = "mordred"
BEST_PARAMS = dict(
    n_estimators=736, learning_rate=0.02210943033781764, num_leaves=180, max_depth=8,
    subsample=0.7566553417347286, colsample_bytree=0.7544350241810107,
    min_child_samples=59, reg_alpha=0.0008787301196251835, reg_lambda=0.00089760661834825,
)
BAGGING_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]

# 4 diverse candidates pulled from expanded_sweep_v3 (top trial per distinct
# (model_type, feature_set) pair, chosen to span different model families
# and feature representations rather than near-duplicate hyperparameter
# variants of the same combo).
CANDIDATES = [
    dict(name="lightgbm[mordred]", family="tree", model_type="lightgbm", feature_set="mordred",
         params=BEST_PARAMS),
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


def load_feature_set(feature_set):
    if feature_set not in _CACHE:
        X_train = np.load(os.path.join(CACHE_DIR, f"train_{feature_set.replace('+', '_')}.npy"))
        X_val = np.load(os.path.join(CACHE_DIR, f"val_{feature_set.replace('+', '_')}.npy"))
        _CACHE[feature_set] = np.vstack([X_train, X_val])
    return _CACHE[feature_set]


def get_folds_and_y():
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    y = trainval["Solubility"].values.astype(np.float32)
    return scaffold_group_kfold(trainval, n_splits=N_FOLDS, seed=42), y


def build_model(family, model_type, params, seed):
    if family == "tree":
        return build_tree_model(model_type, params, seed=seed)
    return build_sklearn_model(model_type, params, seed=seed)


def run_bagging_experiment(folds, y):
    print("=" * 70)
    print(f"Multi-seed bagging: {BEST_MODEL_TYPE}[{BEST_FEATURE_SET}], seeds={BAGGING_SEEDS}")
    print("=" * 70)
    X = load_feature_set(BEST_FEATURE_SET)

    single_seed_fold_rmses = []
    bagged_fold_rmses = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        seed_preds = []
        for seed in BAGGING_SEEDS:
            model = build_tree_model(BEST_MODEL_TYPE, BEST_PARAMS, seed=seed)
            model.fit(X_tr, y_tr)
            seed_preds.append(model.predict(X_va))
        seed_preds = np.array(seed_preds)  # (n_seeds, n_val)

        single_rmse = regression_metrics(y_va, seed_preds[0])["rmse"]
        bagged_rmse = regression_metrics(y_va, seed_preds.mean(axis=0))["rmse"]
        single_seed_fold_rmses.append(single_rmse)
        bagged_fold_rmses.append(bagged_rmse)
        print(f"  fold {fold_i}: single-seed RMSE={single_rmse:.4f}  bagged({len(BAGGING_SEEDS)}-seed) RMSE={bagged_rmse:.4f}")

    print(f"\nSingle-seed mean CV RMSE: {np.mean(single_seed_fold_rmses):.4f} (+/- {np.std(single_seed_fold_rmses):.4f})")
    print(f"Bagged mean CV RMSE:      {np.mean(bagged_fold_rmses):.4f} (+/- {np.std(bagged_fold_rmses):.4f})")
    return np.mean(single_seed_fold_rmses), np.mean(bagged_fold_rmses)


def run_ensemble_experiment(folds, y):
    print("\n" + "=" * 70)
    print(f"Ensemble across {len(CANDIDATES)} diverse candidates (OOF predictions)")
    print("=" * 70)

    oof_preds = {c["name"]: np.zeros(len(y), dtype=np.float64) for c in CANDIDATES}

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"  fold {fold_i}...")
        for c in CANDIDATES:
            X = load_feature_set(c["feature_set"])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]
            model = build_model(c["family"], c["model_type"], c["params"], seed=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            oof_preds[c["name"]][val_idx] = preds

    print()
    for name, preds in oof_preds.items():
        m = regression_metrics(y, preds)
        print(f"{name}: {format_metrics(m)}")

    mean_blend = np.mean(list(oof_preds.values()), axis=0)
    m_mean = regression_metrics(y, mean_blend)
    print(f"\nunweighted_mean: {format_metrics(m_mean)}")

    weights = optimize_blend_weights(oof_preds, y)
    weighted = weighted_blend(oof_preds, weights)
    m_weighted = regression_metrics(y, weighted)
    print(f"weighted_blend {weights}: {format_metrics(m_weighted)}")
    print("  (NOTE: weights fit on the same OOF data being scored here -- small optimism bias)")

    meta = StackingMetaLearner("ridge").fit(oof_preds, y)
    meta_preds = meta.predict(oof_preds)
    m_meta = regression_metrics(y, meta_preds)
    print(f"stacking_ridge: {format_metrics(m_meta)}")
    print("  (NOTE: meta-learner fit on the same OOF data being scored here -- small optimism bias)")

    return {
        "singles": {name: regression_metrics(y, preds)["rmse"] for name, preds in oof_preds.items()},
        "unweighted_mean": m_mean["rmse"],
        "weighted_blend": m_weighted["rmse"],
        "stacking_ridge": m_meta["rmse"],
    }


def main():
    folds, y = get_folds_and_y()
    bagging_single, bagging_bagged = run_bagging_experiment(folds, y)
    ensemble_results = run_ensemble_experiment(folds, y)

    print("\n" + "=" * 70)
    print("SUMMARY (all numbers: 5-fold scaffold-grouped CV mean RMSE)")
    print("=" * 70)
    print(f"Baseline (expanded_sweep_v3 best, single-seed): 1.0106")
    print(f"Bagging single-seed (sanity check):              {bagging_single:.4f}")
    print(f"Bagging {len(BAGGING_SEEDS)}-seed average:                    {bagging_bagged:.4f}")
    for name, rmse in ensemble_results["singles"].items():
        print(f"Ensemble candidate {name}: {rmse:.4f}")
    print(f"Ensemble unweighted_mean:                        {ensemble_results['unweighted_mean']:.4f}")
    print(f"Ensemble weighted_blend:                         {ensemble_results['weighted_blend']:.4f}")
    print(f"Ensemble stacking_ridge:                         {ensemble_results['stacking_ridge']:.4f}")


if __name__ == "__main__":
    main()
