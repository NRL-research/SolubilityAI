"""
Layer 1 automation, GNN half: architecture + optimizer sweep for the graph
models, scored on validation RMSE after early stopping. Kept as a separate,
smaller-budget study from the tree sweep since each trial here is a full
training run (much more expensive than a single tree .fit()).

Usage:
  python solubility_models/scripts/gnn_sweep.py --n-trials 20
"""

import argparse
import os
import sys

import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solubility_models.featurizer import smiles_list_to_graphs
from solubility_models.models import MolecularGCN, MolecularGATv2, HybridMolecularGATv2, train_gnn, get_device

import torch  # noqa: E402 -- must come after the solubility_models imports above

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
STUDY_DIR = os.path.join(ROOT, "misc", "optuna_studies")
STUDY_NAME = "gnn_sweep"

ARCHITECTURES = ["gcn", "gatv2", "hybrid_gatv2"]

_GRAPH_CACHE = {}


def get_graphs():
    if "train" not in _GRAPH_CACHE:
        train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
        _GRAPH_CACHE["train"] = smiles_list_to_graphs(
            train_df["canonical_smiles"].tolist(), train_df["Solubility"].tolist()
        )
        _GRAPH_CACHE["val"] = smiles_list_to_graphs(
            val_df["canonical_smiles"].tolist(), val_df["Solubility"].tolist()
        )
    return _GRAPH_CACHE["train"], _GRAPH_CACHE["val"]


def build_model(architecture, heads):
    if architecture == "gcn":
        return MolecularGCN(num_node_features=17)
    if architecture == "gatv2":
        return MolecularGATv2(num_node_features=17, num_edge_features=4, heads=heads)
    if architecture == "hybrid_gatv2":
        return HybridMolecularGATv2(num_node_features=17, num_edge_features=4, num_global_features=5, heads=heads)
    raise ValueError(architecture)


def objective(trial):
    architecture = trial.suggest_categorical("architecture", ARCHITECTURES)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    heads = trial.suggest_categorical("heads", [2, 4, 8]) if architecture != "gcn" else 4

    train_graphs, val_graphs = get_graphs()
    model = build_model(architecture, heads)
    # MPS, not CPU: CPU forward pass segfaults deterministically for these
    # PyG conv layers on this torch/torch_geometric version combo (confirmed
    # via isolated repro -- crashes on a 50-molecule GCN's very first
    # forward()). MPS is the one that actually works; the earlier "killed"
    # background runs were unrelated (likely transient / one coincided with
    # the machine sleeping).
    device = get_device()

    _, _, best_val_rmse = train_gnn(
        model, train_graphs, val_graphs,
        epochs=150, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        patience=20, device=device, seed=42,
    )
    return best_val_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(STUDY_DIR, exist_ok=True)
    storage = f"sqlite:///{os.path.join(STUDY_DIR, STUDY_NAME)}.db"
    # No fixed seed: each invocation of this script is a fresh process (MPS
    # training is only reliable one trial per process -- see module docstring
    # context). A fixed-seed TPESampler restarts from the same RNG state
    # every process and suggests the IDENTICAL "random" first draw every
    # time -- confirmed: 6 consecutive single-trial runs all returned the
    # exact same hyperparameters. Unseeded lets each process diversify.
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(), load_if_exists=True,
    )
    print(f"Study '{STUDY_NAME}': {len(study.trials)} trials so far.")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best val RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
