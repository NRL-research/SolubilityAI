"""
Expanded Optuna sweep for the "beat DISSOLVR" round: all model families
(tree + classic sklearn) x all feature representations (original 5 +
11 new ones), scored via 5-fold scaffold-grouped cross-validation instead
of a single train/val split -- see common/splitting.scaffold_group_kfold
for why (the single-split validation set had real composition variance,
e.g. only 3.5% acyclic molecules by shuffle luck vs ~35% elsewhere).

SQLite-backed under a new study name ('expanded_sweep') so the original
585-trial 'tree_sweep' study (single-val-split protocol, RMSE=0.8137
plateau) is preserved separately as historical record, not overwritten.

GNN architectures are swept separately (gnn_sweep.py) -- MPS training has
its own one-trial-per-process reliability constraint that doesn't apply
here, and CV would multiply GNN training cost 5x, which isn't worth it
given the GNN already scored weakly standalone (R^2~0.35-0.38).

Usage:
  python solubility_models/scripts/expanded_sweep.py --n-trials 50 --timeout 480
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.splitting import scaffold_group_kfold
from common.evaluate import regression_metrics
from solubility_models.featurizer import ALL_FEATURE_SETS
from solubility_models.models import (
    build_tree_model, DEFAULT_TREE_SEARCH_SPACE,
    build_sklearn_model, DEFAULT_SKLEARN_SEARCH_SPACE,
    SKLEARN_KERNEL_METHOD_FEATURE_SETS, RANDOM_FOREST_FAMILY_FEATURE_SETS,
)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
STUDY_DIR = os.path.join(ROOT, "misc", "optuna_studies")
# v3: mlp__hidden_layer_sizes' choices changed from tuples to strings (see
# MLP_HIDDEN_LAYER_SIZES_MAP in models.py for why: tuple choices silently
# round-trip through Optuna's SQLite storage as JSON lists, and
# `[256, 128] in ((256, 128), ...)` is False in Python, which crashed the
# whole run 15 trials into v2 with "not in {choices}"). Same
# dynamic-value-space constraint as the v1->v2 rename applies here too.
# v2: model_family's choice set changed 3->4 values (RandomForest/ExtraTrees
# split from HistGB/MLP for feature-width reasons); only 7 trials lost then.
STUDY_NAME = "expanded_sweep_v3"

TREE_MODEL_TYPES = ["xgboost", "catboost", "lightgbm"]
# hist_gradient_boosting/mlp scale fine on wide feature sets (histogram
# binning / linear param growth); random_forest/extra_trees don't -- see
# RANDOM_FOREST_FAMILY_FEATURE_SETS's docstring (measured: one trial took
# ~58 minutes on a 2265-dim feature set before this split existed).
WIDE_OK_SKLEARN = ["hist_gradient_boosting", "mlp"]
RESTRICTED_SKLEARN = ["random_forest", "extra_trees"]
KERNEL_SKLEARN = ["svr", "gaussian_process"]

N_FOLDS = 5
GP_MAX_TRAIN_ROWS = 2000  # GP is O(n^3) -- subsample within each fold's train split to stay tractable

# SVR/GaussianProcess/MLP are all scale-sensitive (distance- or
# gradient-based); tree models (XGBoost/CatBoost/LightGBM/RandomForest/
# ExtraTrees/HistGB) split on raw thresholds and are scale-invariant, so
# scaling would just be wasted work for them. Without this, a single MLP
# trial produced RMSE=inf (found while sweeping) -- feature sets mix binary
# fingerprint bits (0/1) with descriptors like molecular weight (hundreds).
SCALE_SENSITIVE_MODEL_TYPES = {"svr", "gaussian_process", "mlp"}

_CACHE = {}
_FOLDS = None


def load_feature_set(feature_set):
    if feature_set not in _CACHE:
        X_train = np.load(os.path.join(CACHE_DIR, f"train_{feature_set.replace('+', '_')}.npy"))
        X_val = np.load(os.path.join(CACHE_DIR, f"val_{feature_set.replace('+', '_')}.npy"))
        _CACHE[feature_set] = np.vstack([X_train, X_val])
    return _CACHE[feature_set]


def get_folds():
    global _FOLDS
    if _FOLDS is None:
        train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
        trainval = pd.concat([train_df, val_df], ignore_index=True)
        y = trainval["Solubility"].values.astype(np.float32)
        _FOLDS = (scaffold_group_kfold(trainval, n_splits=N_FOLDS, seed=42), y)
    return _FOLDS


def suggest_params(trial, model_type, space):
    params = {}
    for name, spec in space.items():
        prefixed = f"{model_type}__{name}"
        if isinstance(spec, list):  # categorical, e.g. MLP hidden_layer_sizes
            params[name] = trial.suggest_categorical(prefixed, spec)
        elif len(spec) == 3 and spec[2] == "log":
            params[name] = trial.suggest_float(prefixed, spec[0], spec[1], log=True)
        elif isinstance(spec[0], float):
            params[name] = trial.suggest_float(prefixed, spec[0], spec[1])
        else:
            params[name] = trial.suggest_int(prefixed, spec[0], spec[1])
    return params


def objective(trial):
    model_family = trial.suggest_categorical(
        "model_family", ["tree", "sklearn_wide", "sklearn_restricted", "sklearn_kernel"]
    )

    if model_family == "tree":
        model_type = trial.suggest_categorical("tree_model_type", TREE_MODEL_TYPES)
        feature_set = trial.suggest_categorical("tree_feature_set", ALL_FEATURE_SETS)
        params = suggest_params(trial, model_type, DEFAULT_TREE_SEARCH_SPACE[model_type])
        build = lambda: build_tree_model(model_type, params, seed=42)
    elif model_family == "sklearn_wide":
        model_type = trial.suggest_categorical("wide_model_type", WIDE_OK_SKLEARN)
        feature_set = trial.suggest_categorical("wide_feature_set", ALL_FEATURE_SETS)
        params = suggest_params(trial, model_type, DEFAULT_SKLEARN_SEARCH_SPACE[model_type])
        build = lambda: build_sklearn_model(model_type, params, seed=42)
    elif model_family == "sklearn_restricted":
        model_type = trial.suggest_categorical("restricted_model_type", RESTRICTED_SKLEARN)
        feature_set = trial.suggest_categorical("restricted_feature_set", RANDOM_FOREST_FAMILY_FEATURE_SETS)
        params = suggest_params(trial, model_type, DEFAULT_SKLEARN_SEARCH_SPACE[model_type])
        build = lambda: build_sklearn_model(model_type, params, seed=42)
    else:
        model_type = trial.suggest_categorical("kernel_model_type", KERNEL_SKLEARN)
        feature_set = trial.suggest_categorical("kernel_feature_set", SKLEARN_KERNEL_METHOD_FEATURE_SETS)
        params = suggest_params(trial, model_type, DEFAULT_SKLEARN_SEARCH_SPACE[model_type])
        build = lambda: build_sklearn_model(model_type, params, seed=42)

    X = load_feature_set(feature_set)
    folds, y = get_folds()

    fold_rmses = []
    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        if model_type == "gaussian_process" and len(X_tr) > GP_MAX_TRAIN_ROWS:
            rng = np.random.RandomState(42)
            sub = rng.choice(len(X_tr), GP_MAX_TRAIN_ROWS, replace=False)
            X_tr, y_tr = X_tr[sub], y_tr[sub]
        if model_type in SCALE_SENSITIVE_MODEL_TYPES:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_tr)  # fit on this fold's train split only, no leakage
            X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
        model = build()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        m = regression_metrics(y_va, preds)
        fold_rmses.append(m["rmse"])

    mean_rmse = float(np.mean(fold_rmses))
    trial.set_user_attr("model_type", model_type)
    trial.set_user_attr("feature_set", feature_set)
    trial.set_user_attr("fold_rmses", fold_rmses)
    trial.set_user_attr("std_rmse", float(np.std(fold_rmses)))
    return mean_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    args = parser.parse_args()

    os.makedirs(STUDY_DIR, exist_ok=True)
    storage = f"sqlite:///{os.path.join(STUDY_DIR, STUDY_NAME)}.db"
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42), load_if_exists=True,
    )
    print(f"Study '{STUDY_NAME}': {len(study.trials)} trials so far.")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nTotal completed trials: {len(completed)}")
    print(f"Best mean CV RMSE: {study.best_value:.4f}")
    print(f"Best model/feature_set: {study.best_trial.user_attrs.get('model_type')} / {study.best_trial.user_attrs.get('feature_set')}")
    print(f"Best trial fold RMSEs: {study.best_trial.user_attrs.get('fold_rmses')}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
