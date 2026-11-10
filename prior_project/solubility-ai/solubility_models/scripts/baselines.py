"""
Re-establish honest baselines under the corrected split + validation
protocol (validation only for every tuning/selection decision; test is not
touched by this script at all).

1. DISSOLVR reproduction: same architecture/hyperparameters as the original
   dissolvr/train.py (CatBoost, iterations=10000, lr=0.02, depth=8,
   l2_leaf_reg=5), on the dissolvr_full (176-feature) representation,
   5 seeds -- this is a faithful reproduction, not yet tuned (matches what
   the original script did).

2. A first honest Holy Trinity-style ensemble: Hybrid GATv2 + XGBoost +
   CatBoost on rdkit_descriptors (the representation that beat the
   fingerprint-based one in the original notebook's own exploration, see
   cell 47 vs cell 38), with reasonable but UNTUNED hyperparameters, blended
   via validation-optimized weights (solubility_models/ensemble.optimize_blend_weights) --
   not Optuna-against-test-set like the notebook did. This gives an
   apples-to-apples "before tuning" number for both approaches; the Optuna
   sweep (Layer 1) tunes both from here.

Results appended to solubility_models/results/baselines.csv.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# solubility_models/__init__.py imports xgboost/catboost/lightgbm before torch
# gets a chance to -- must stay the first solubility_models (or torch-touching)
# import in this file, see solubility_models/__init__.py for why (macOS
# OpenMP load-order segfault).
from solubility_models.featurizer import smiles_list_to_graphs
from solubility_models.models import build_tree_model, GNN_BUILDERS, train_gnn, get_device
from common.evaluate import regression_metrics, aggregate_seed_runs, format_metrics, format_aggregated
from solubility_models.ensemble import optimize_blend_weights, weighted_blend

import torch  # noqa: E402 -- must come after the solubility_models imports above

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
RESULTS_PATH = os.path.join(ROOT, "solubility_models", "results", "baselines.csv")

SEEDS = [42, 101, 123, 456, 789]


def load_cached(split, feature_set):
    X = np.load(os.path.join(CACHE_DIR, f"{split}_{feature_set.replace('+', '_')}.npy"))
    y = np.load(os.path.join(CACHE_DIR, f"{split}_y.npy"))
    return X, y


def append_result(rows):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(RESULTS_PATH):
        df = pd.concat([pd.read_csv(RESULTS_PATH), df], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)


def completed_seeds(model_name):
    """Resume support: which seeds already have a saved result for this model."""
    if not os.path.exists(RESULTS_PATH):
        return set()
    df = pd.read_csv(RESULTS_PATH)
    return set(df.loc[df["model"] == model_name, "seed"].tolist())


def run_dissolvr_reproduction():
    print("\n" + "=" * 70)
    print("DISSOLVR reproduction (dissolvr_full features, original hyperparams)")
    print("=" * 70)
    X_train, y_train = load_cached("train", "dissolvr_full")
    X_val, y_val = load_cached("val", "dissolvr_full")

    done = completed_seeds("dissolvr_reproduction")
    metrics_per_seed = []
    for seed in SEEDS:
        if seed in done:
            print(f"  seed={seed}  already completed, skipping")
            continue
        t0 = time.time()
        model = build_tree_model(
            "catboost",
            {"iterations": 10000, "learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 5},
            seed=seed,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        m = regression_metrics(y_val, preds)
        elapsed = time.time() - t0
        print(f"  seed={seed}  {format_metrics(m)}  ({elapsed:.1f}s)")
        append_result([{"model": "dissolvr_reproduction", "seed": seed, "split": "val",
                         "feature_set": "dissolvr_full", "tuned": False, **m}])

    all_rows = pd.read_csv(RESULTS_PATH)
    all_rows = all_rows[all_rows["model"] == "dissolvr_reproduction"]
    metrics_per_seed = all_rows[["rmse", "mae", "r2"]].to_dict("records")
    agg = aggregate_seed_runs(metrics_per_seed)
    print("  " + format_aggregated(agg, prefix="AGGREGATE (val, 5 seeds): "))
    return agg


def run_holy_trinity_first_pass(max_new_seeds=None):
    print("\n" + "=" * 70)
    print("Holy Trinity (untuned first pass): Hybrid GATv2 + XGBoost + CatBoost")
    print("Feature set: rdkit_descriptors (beat morgan+global5 in original exploration)")
    print("=" * 70)

    X_train, y_train = load_cached("train", "rdkit_descriptors")
    X_val, y_val = load_cached("val", "rdkit_descriptors")

    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    train_graphs = smiles_list_to_graphs(train_df["canonical_smiles"].tolist(), train_df["Solubility"].tolist())
    val_graphs = smiles_list_to_graphs(val_df["canonical_smiles"].tolist(), val_df["Solubility"].tolist())

    done = completed_seeds("holy_trinity_blend")
    # Note: earlier interrupted runs were caused by the machine going to sleep
    # overnight (background processes stall / get reaped on wake), not by
    # using MPS -- seed 42 completed cleanly on MPS before that. Back to the
    # normal device default.
    device = get_device()
    print(f"  device: {device}")

    new_seeds_run = 0
    for seed in SEEDS[:3]:  # GNN training is more expensive; 3 seeds for now
        if seed in done:
            print(f"  seed={seed}  already completed, skipping")
            continue
        if max_new_seeds is not None and new_seeds_run >= max_new_seeds:
            print(f"  seed={seed}  reached --max-new-seeds cap for this run, stopping here")
            break
        new_seeds_run += 1
        t0 = time.time()

        xgbm = build_tree_model("xgboost", {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}, seed=seed)
        xgbm.fit(X_train, y_train)
        xgb_preds = xgbm.predict(X_val)

        cb = build_tree_model("catboost", {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3}, seed=seed)
        cb.fit(X_train, y_train)
        cb_preds = cb.predict(X_val)

        gnn = GNN_BUILDERS["hybrid_gatv2"]()
        gnn, history, best_val_rmse = train_gnn(
            gnn, train_graphs, val_graphs, epochs=100, patience=15, device=device, seed=seed,
        )
        gnn_preds, gnn_targets = None, None
        from solubility_models.models import predict_gnn
        gnn_preds, gnn_targets = predict_gnn(gnn, val_graphs, device=device, return_targets=True)

        preds_dict = {"xgboost": xgb_preds, "catboost": cb_preds, "hybrid_gnn": gnn_preds}
        weights = optimize_blend_weights(preds_dict, y_val)
        blend_preds = weighted_blend(preds_dict, weights)

        m_xgb = regression_metrics(y_val, xgb_preds)
        m_cb = regression_metrics(y_val, cb_preds)
        m_gnn = regression_metrics(gnn_targets, gnn_preds)
        m_blend = regression_metrics(y_val, blend_preds)

        elapsed = time.time() - t0
        print(f"  seed={seed}  xgb={format_metrics(m_xgb)}")
        print(f"           cb ={format_metrics(m_cb)}")
        print(f"           gnn={format_metrics(m_gnn)}  (val_rmse used for early stop: {best_val_rmse:.4f})")
        print(f"           blend weights={ {k: round(v,3) for k,v in weights.items()} }")
        print(f"           BLEND={format_metrics(m_blend)}  ({elapsed:.1f}s)")

        seed_rows = [
            {"model": "xgboost_component", "seed": seed, "split": "val",
             "feature_set": "rdkit_descriptors", "tuned": False, **m_xgb},
            {"model": "catboost_component", "seed": seed, "split": "val",
             "feature_set": "rdkit_descriptors", "tuned": False, **m_cb},
            {"model": "hybrid_gnn_component", "seed": seed, "split": "val",
             "feature_set": "graph", "tuned": False, **m_gnn},
            {"model": "holy_trinity_blend", "seed": seed, "split": "val",
             "feature_set": "rdkit_descriptors+graph", "tuned": False, **m_blend},
        ]
        append_result(seed_rows)  # persist per-seed immediately, don't lose progress if interrupted

    all_rows = pd.read_csv(RESULTS_PATH)
    all_rows = all_rows[all_rows["model"] == "holy_trinity_blend"]
    metrics_per_seed = all_rows[["rmse", "mae", "r2"]].to_dict("records")
    agg = aggregate_seed_runs(metrics_per_seed)
    print("  " + format_aggregated(agg, prefix="BLEND AGGREGATE (val, 3 seeds): "))
    return agg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-seeds", type=int, default=None,
                         help="Cap how many not-yet-completed Holy Trinity seeds to run "
                              "in this invocation (each seed's GNN training takes several "
                              "minutes; keep this small to fit within a single background call).")
    args = parser.parse_args()

    dissolvr_agg = run_dissolvr_reproduction()
    holy_trinity_agg = run_holy_trinity_first_pass(max_new_seeds=args.max_new_seeds)

    print("\n" + "=" * 70)
    print("SUMMARY (validation set, honest protocol, untuned)")
    print("=" * 70)
    print("DISSOLVR repro:  " + format_aggregated(dissolvr_agg))
    print("Holy Trinity:    " + format_aggregated(holy_trinity_agg))
