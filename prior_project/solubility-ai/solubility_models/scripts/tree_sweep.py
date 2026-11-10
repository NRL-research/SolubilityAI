"""
Layer 1 automation: Optuna sweep over feature representation x tree-model
family x hyperparameters, scored on the validation split only. SQLite-backed
so it's resumable across sessions (the autonomous research loop calls this
repeatedly with more trials).

GNN architectures are swept separately (gnn_sweep.py) since each trial there
is far more expensive (full training run vs. a single tree fit); final
ensembling combines the best of both (ensemble_search.py).

Usage:
  python solubility_models/scripts/tree_sweep.py --n-trials 100
"""

import argparse
import os
import sys

import numpy as np
import optuna

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solubility_models.featurizer import FEATURE_SETS
from solubility_models.models import build_tree_model, DEFAULT_TREE_SEARCH_SPACE
from common.evaluate import regression_metrics

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
STUDY_DIR = os.path.join(ROOT, "misc", "optuna_studies")
STUDY_NAME = "tree_sweep"

MODEL_TYPES = ["xgboost", "catboost", "lightgbm"]


def load_cached(split, feature_set):
    X = np.load(os.path.join(CACHE_DIR, f"{split}_{feature_set.replace('+', '_')}.npy"))
    y = np.load(os.path.join(CACHE_DIR, f"{split}_y.npy"))
    return X, y


_CACHE = {}


def get_data(split, feature_set):
    key = (split, feature_set)
    if key not in _CACHE:
        _CACHE[key] = load_cached(split, feature_set)
    return _CACHE[key]


def suggest_params(trial, model_type):
    space = DEFAULT_TREE_SEARCH_SPACE[model_type]
    params = {}
    for name, spec in space.items():
        if len(spec) == 3 and spec[2] == "log":
            params[name] = trial.suggest_float(f"{model_type}__{name}", spec[0], spec[1], log=True)
        elif isinstance(spec[0], float):
            params[name] = trial.suggest_float(f"{model_type}__{name}", spec[0], spec[1])
        else:
            params[name] = trial.suggest_int(f"{model_type}__{name}", spec[0], spec[1])
    return params


def objective(trial):
    feature_set = trial.suggest_categorical("feature_set", FEATURE_SETS)
    model_type = trial.suggest_categorical("model_type", MODEL_TYPES)
    params = suggest_params(trial, model_type)

    X_train, y_train = get_data("train", feature_set)
    X_val, y_val = get_data("val", feature_set)

    model = build_tree_model(model_type, params, seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = regression_metrics(y_val, preds)

    trial.set_user_attr("rmse", metrics["rmse"])
    trial.set_user_attr("mae", metrics["mae"])
    trial.set_user_attr("r2", metrics["r2"])
    return metrics["rmse"]  # minimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
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

    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Best trial R2: {study.best_trial.user_attrs.get('r2')}")


if __name__ == "__main__":
    main()
