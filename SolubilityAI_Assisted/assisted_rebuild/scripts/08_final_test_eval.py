"""
08_final_test_eval.py
=====================
THE single test-set evaluation.

Everything selected upstream (which models, which feature sets, which blend weights)
was decided from out-of-fold train+val predictions only -- see 06_tree_cv.py and
07_assemble_cv_results.py. This script trains the already-decided configurations on
the full train+val pool, predicts the held-out test set exactly once per
configuration, and applies the OOF-fitted blend weights unchanged.

Nothing here refits, reselects, or re-weights using test.

Configurations to evaluate are read from results/final_configs.json, written by
07b_choose_final_configs.py, so the choice is recorded before test is touched.

Outputs: results/test_predictions.npz, results/final_test_results.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common_eval import metrics  # noqa: E402
from models import build  # noqa: E402

DATA = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
RES = "/home/claude/work/results"
GNN = f"{RES}/gnn"
SEEDS = [42, 101, 123, 456, 789]          # GATv2 test predictions already exist for all 5
TREE_SEEDS = [42, 101, 123]               # 3 seeds for tree models: compute-bound on a
                                          # 2-core container (LightGBM+Mordred is ~17 min
                                          # per fit). Recorded as a deliberate reduction,
                                          # not a silent one.

trainval = pd.read_csv(f"{DATA}/trainval.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y_tv = trainval["Solubility"].values
y_te = test["Solubility"].values

cfg = json.load(open(f"{RES}/final_configs.json"))
single = cfg["single_models"]          # {name: {model_type, feature_set, kwargs, multiseed}}
blends = cfg["blends"]                 # {name: {members, weights}}


def load_X(fs):
    tr = np.load(f"{CACHE}/train__{fs}.npy")
    va = np.load(f"{CACHE}/val__{fs}.npy")
    te = np.load(f"{CACHE}/test__{fs}.npy")
    X_tv = np.vstack([tr, va]).astype(np.float64)
    assert X_tv.shape[0] == len(trainval) and te.shape[0] == len(test)
    return X_tv, te.astype(np.float64)


test_preds = {}
rows = []

for name, spec in single.items():
    if spec["model_type"] == "gatv2":
        files = [f"{GNN}/test_pred_seed{s}.npz" for s in SEEDS]
        files = [f for f in files if os.path.exists(f)]
        assert files, "no GATv2 test predictions found -- run 05_gnn_cv.py final SEED"
        P = np.column_stack([np.load(f)["pred"] for f in files])
        per_seed = [metrics(y_te, P[:, i]) for i in range(P.shape[1])]
        pred = P.mean(axis=1)
        n_seeds = P.shape[1]
    else:
        X_tv, X_te = load_X(spec["feature_set"])
        seeds = TREE_SEEDS if spec.get("multiseed") else [42]
        P = []
        for s in seeds:
            t0 = time.time()
            m = build(spec["model_type"], seed=s, **spec.get("kwargs", {}))
            m.fit(X_tv, y_tv)
            P.append(np.asarray(m.predict(X_te)).ravel())
            print(f"{name} seed {s} fitted ({time.time()-t0:.0f}s)", flush=True)
        P = np.column_stack(P)
        per_seed = [metrics(y_te, P[:, i]) for i in range(P.shape[1])]
        pred = P.mean(axis=1)
        n_seeds = P.shape[1]

    test_preds[name] = pred
    # per-seed spread (variance estimate), plus the seed-averaged prediction's metrics
    rows.append({
        "config": name, "n_seeds": n_seeds,
        "rmse_seedmean": float(np.mean([m["rmse"] for m in per_seed])),
        "rmse_seedstd": float(np.std([m["rmse"] for m in per_seed])),
        "r2_seedmean": float(np.mean([m["r2"] for m in per_seed])),
        "r2_seedstd": float(np.std([m["r2"] for m in per_seed])),
        **{f"avgpred_{k}": v for k, v in metrics(y_te, pred).items()},
    })
    print(f"  -> {name}: {rows[-1]}", flush=True)

for name, spec in blends.items():
    members, w = spec["members"], np.array(spec["weights"], dtype=float)
    missing = [m for m in members if m not in test_preds]
    assert not missing, f"blend {name} missing members {missing}"
    P = np.column_stack([test_preds[m] for m in members])
    pred = P @ w
    test_preds[name] = pred
    rows.append({"config": name, "n_seeds": None,
                 "rmse_seedmean": None, "rmse_seedstd": None,
                 "r2_seedmean": None, "r2_seedstd": None,
                 **{f"avgpred_{k}": v for k, v in metrics(y_te, pred).items()}})
    test_preds[name + "__unweighted"] = P.mean(axis=1)
    rows.append({"config": name + " [unweighted mean]", "n_seeds": None,
                 "rmse_seedmean": None, "rmse_seedstd": None,
                 "r2_seedmean": None, "r2_seedstd": None,
                 **{f"avgpred_{k}": v for k, v in metrics(y_te, P.mean(axis=1)).items()}})

np.savez(f"{RES}/test_predictions.npz", y=y_te, **test_preds)
df = pd.DataFrame(rows).sort_values("avgpred_rmse")
df.to_csv(f"{RES}/final_test_results.csv", index=False)
pd.set_option("display.width", 220)
print("\n" + df.to_string(index=False))
print(f"\nSaved {RES}/final_test_results.csv and test_predictions.npz")
