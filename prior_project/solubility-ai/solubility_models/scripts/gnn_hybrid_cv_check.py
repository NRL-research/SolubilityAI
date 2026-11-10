"""
5-fold scaffold-grouped CV check for the best hybrid_gatv2 config found by
gnn_sweep_hybrid.py (single train/val split, best val RMSE=0.9293), so its
performance is directly comparable to expanded_sweep_v3's CV-mean RMSE
(1.0106, LightGBM on 'mordred' features) instead of the single-split number,
which this dataset's split-composition variance has repeatedly shown can be
partly luck (see common.splitting.scaffold_group_kfold's docstring context).

Same 5 folds as expanded_sweep.py: scaffold_group_kfold(trainval, n_splits=5,
seed=42) over train+val combined.

Usage:
  python solubility_models/scripts/gnn_hybrid_cv_check.py
  python solubility_models/scripts/gnn_hybrid_cv_check.py --only-fold 4  # rerun a single fold (e.g. after an interrupted run)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.splitting import scaffold_group_kfold
from solubility_models.featurizer import smiles_list_to_graphs
from solubility_models.models import HybridMolecularGATv2, train_gnn, get_device

import torch  # noqa: E402 -- must come after the solubility_models imports above

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
N_FOLDS = 5

# gnn_sweep_hybrid.py trial 4 (best of 5): val RMSE=0.9293 on the single split.
BEST_PARAMS = dict(
    lr=0.0008161976277711446,
    weight_decay=0.0006560940226991143,
    batch_size=64,
    heads=2,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fold", type=int, default=None,
                         help="Run just this one fold index (0-based) instead of all N_FOLDS.")
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    trainval = pd.concat([train_df, val_df], ignore_index=True)

    folds = scaffold_group_kfold(trainval, n_splits=N_FOLDS, seed=42)
    device = get_device()

    fold_items = list(enumerate(folds))
    if args.only_fold is not None:
        fold_items = [fold_items[args.only_fold]]

    fold_rmses = []
    for i, (train_idx, val_idx) in fold_items:
        fold_train = trainval.iloc[train_idx]
        fold_val = trainval.iloc[val_idx]

        train_graphs = smiles_list_to_graphs(
            fold_train["canonical_smiles"].tolist(), fold_train["Solubility"].tolist()
        )
        val_graphs = smiles_list_to_graphs(
            fold_val["canonical_smiles"].tolist(), fold_val["Solubility"].tolist()
        )

        model = HybridMolecularGATv2(
            num_node_features=17, num_edge_features=4, num_global_features=5, heads=BEST_PARAMS["heads"]
        )
        _, _, best_val_rmse = train_gnn(
            model, train_graphs, val_graphs,
            epochs=150, lr=BEST_PARAMS["lr"], weight_decay=BEST_PARAMS["weight_decay"],
            batch_size=BEST_PARAMS["batch_size"], patience=20, device=device, seed=42,
        )
        print(f"fold {i}: val RMSE = {best_val_rmse:.4f}", flush=True)
        fold_rmses.append(best_val_rmse)

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))
    print(f"\nFold RMSEs: {fold_rmses}")
    print(f"Mean CV RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")


if __name__ == "__main__":
    main()
