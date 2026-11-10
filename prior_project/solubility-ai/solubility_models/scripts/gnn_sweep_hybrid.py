"""
Dedicated GNN sweep forcing the hybrid_gatv2 architecture only. The original
gnn_sweep.py study ('gnn_sweep') sampled architecture from
["gcn", "gatv2", "hybrid_gatv2"] via TPE and, after 22 trials, had picked
gatv2 9 times and gcn once -- hybrid_gatv2 (the architecture that injects 5
global descriptors alongside the graph, and the one that beat plain GNNs in
the original notebook) never got sampled once. Rather than wait on TPE to
get to it, this script fixes architecture=hybrid_gatv2 and only searches
lr/weight_decay/batch_size/heads.

Separate study name ('gnn_sweep_hybrid') so this doesn't collide with the
existing 'gnn_sweep' study's architecture categorical (changing an existing
categorical's effective choice set crashes with "dynamic value space", see
expanded_sweep.py's docstring for the same issue hit there).

Usage:
  python solubility_models/scripts/gnn_sweep_hybrid.py --n-trials 1
"""

import argparse
import os
import sys

import optuna
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solubility_models.featurizer import smiles_list_to_graphs
from solubility_models.models import HybridMolecularGATv2, train_gnn, get_device

import torch  # noqa: E402 -- must come after the solubility_models imports above

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
STUDY_DIR = os.path.join(ROOT, "misc", "optuna_studies")
STUDY_NAME = "gnn_sweep_hybrid"

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


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    heads = trial.suggest_categorical("heads", [2, 4, 8])

    train_graphs, val_graphs = get_graphs()
    model = HybridMolecularGATv2(num_node_features=17, num_edge_features=4, num_global_features=5, heads=heads)
    # MPS, not CPU -- see gnn_sweep.py's objective() comment, same constraint applies here.
    device = get_device()

    _, _, best_val_rmse = train_gnn(
        model, train_graphs, val_graphs,
        epochs=150, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        patience=20, device=device, seed=42,
    )
    return best_val_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(STUDY_DIR, exist_ok=True)
    storage = f"sqlite:///{os.path.join(STUDY_DIR, STUDY_NAME)}.db"
    # Unseeded TPESampler -- see gnn_sweep.py's comment: a fixed seed with a
    # fresh process per trial (MPS one-trial-per-process constraint) replays
    # the identical "random" first draw every single invocation.
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(), load_if_exists=True,
    )
    print(f"Study '{STUDY_NAME}': {len(study.trials)} trials so far.")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nTotal completed trials: {len(completed)}")
    if completed:
        print(f"Best val RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
