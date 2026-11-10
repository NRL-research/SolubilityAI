"""
Layer 1 automation, final stage: pull the best trials from the tree sweep
and GNN sweep Optuna studies, generate their validation predictions, and
search combination strategy (single best model / weighted blend / stacked
meta-learner) -- scored on validation only. Test set is never touched here.

Writes solubility_models/results/leaderboard.csv (append-only) with one row
per candidate combination, so the research loop can review it between
rounds.

Usage:
  python solubility_models/scripts/ensemble_search.py --top-k 5
"""

import argparse
import os
import sys

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solubility_models.featurizer import smiles_list_to_graphs
from solubility_models.models import build_tree_model, MolecularGCN, MolecularGATv2, HybridMolecularGATv2, train_gnn, predict_gnn, get_device
from common.evaluate import regression_metrics, format_metrics
from solubility_models.ensemble import optimize_blend_weights, weighted_blend, StackingMetaLearner

import torch  # noqa: E402 -- must come after the solubility_models imports above

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
STUDY_DIR = os.path.join(ROOT, "misc", "optuna_studies")
LEADERBOARD_PATH = os.path.join(ROOT, "solubility_models", "results", "leaderboard.csv")


def load_cached(split, feature_set):
    X = np.load(os.path.join(CACHE_DIR, f"{split}_{feature_set.replace('+', '_')}.npy"))
    y = np.load(os.path.join(CACHE_DIR, f"{split}_y.npy"))
    return X, y


def append_leaderboard(rows):
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(LEADERBOARD_PATH):
        df = pd.concat([pd.read_csv(LEADERBOARD_PATH), df], ignore_index=True)
    df.to_csv(LEADERBOARD_PATH, index=False)


def get_top_tree_trials(top_k):
    storage = f"sqlite:///{os.path.join(STUDY_DIR, 'tree_sweep.db')}"
    study = optuna.load_study(study_name="tree_sweep", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    return completed[:top_k]


def get_top_gnn_trial():
    storage = f"sqlite:///{os.path.join(STUDY_DIR, 'gnn_sweep.db')}"
    study = optuna.load_study(study_name="gnn_sweep", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None
    completed.sort(key=lambda t: t.value)
    return completed[0]


def train_tree_from_trial(trial):
    model_type = trial.params["model_type"]
    feature_set = trial.params["feature_set"]
    params = {
        k.split("__", 1)[1]: v for k, v in trial.params.items() if k.startswith(f"{model_type}__")
    }
    X_train, y_train = load_cached("train", feature_set)
    X_val, y_val = load_cached("val", feature_set)
    model = build_tree_model(model_type, params, seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    name = f"{model_type}[{feature_set}]"
    return name, preds, y_val


def train_gnn_from_trial(trial):
    architecture = trial.params["architecture"]
    heads = trial.params.get("heads", 4)
    builders = {
        "gcn": lambda: MolecularGCN(num_node_features=17),
        "gatv2": lambda: MolecularGATv2(num_node_features=17, num_edge_features=4, heads=heads),
        "hybrid_gatv2": lambda: HybridMolecularGATv2(num_node_features=17, num_edge_features=4, num_global_features=5, heads=heads),
    }
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    train_graphs = smiles_list_to_graphs(train_df["canonical_smiles"].tolist(), train_df["Solubility"].tolist())
    val_graphs = smiles_list_to_graphs(val_df["canonical_smiles"].tolist(), val_df["Solubility"].tolist())

    device = get_device()  # see gnn_sweep.py note: CPU forward pass segfaults on this torch/PyG combo
    model = builders[architecture]()
    model, _, _ = train_gnn(
        model, train_graphs, val_graphs, epochs=150,
        lr=trial.params["lr"], weight_decay=trial.params["weight_decay"],
        batch_size=trial.params["batch_size"], patience=20, device=device, seed=42,
    )
    preds, targets = predict_gnn(model, val_graphs, device=device, return_targets=True)
    return f"gnn_{architecture}", preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5, help="top-k tree trials to include as ensemble candidates")
    args = parser.parse_args()

    tree_trials = get_top_tree_trials(args.top_k)
    gnn_trial = get_top_gnn_trial()

    if not tree_trials:
        print("No completed tree_sweep trials found -- run phase3_tree_sweep.py first.")
        return

    preds_dict = {}
    y_val = None
    rows = []

    for trial in tree_trials:
        name, preds, y = train_tree_from_trial(trial)
        preds_dict[name] = preds
        y_val = y
        m = regression_metrics(y, preds)
        print(f"{name}: {format_metrics(m)}")
        rows.append({"candidate": name, "kind": "single_model", **m})

    if gnn_trial is not None:
        name, preds, y = train_gnn_from_trial(gnn_trial)
        preds_dict[name] = preds
        m = regression_metrics(y, preds)
        print(f"{name}: {format_metrics(m)}")
        rows.append({"candidate": name, "kind": "single_model", **m})
    else:
        print("No completed gnn_sweep trials found -- skipping GNN component.")

    # Weighted blend across all candidates
    weights = optimize_blend_weights(preds_dict, y_val)
    blend_preds = weighted_blend(preds_dict, weights)
    m_blend = regression_metrics(y_val, blend_preds)
    print(f"\nweighted_blend {weights}: {format_metrics(m_blend)}")
    rows.append({"candidate": f"weighted_blend({','.join(preds_dict.keys())})", "kind": "blend",
                 "weights": str(weights), **m_blend})

    # Stacked meta-learner
    meta = StackingMetaLearner("ridge").fit(preds_dict, y_val)
    meta_preds = meta.predict(preds_dict)
    m_meta = regression_metrics(y_val, meta_preds)
    print(f"stacking_meta: {format_metrics(m_meta)}")
    rows.append({"candidate": f"stacking_ridge({','.join(preds_dict.keys())})", "kind": "stacking", **m_meta})

    append_leaderboard(rows)
    print(f"\nLeaderboard updated: {LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()
