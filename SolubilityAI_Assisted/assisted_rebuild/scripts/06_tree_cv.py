"""
06_tree_cv.py
=============
Out-of-fold predictions for tree/sklearn model x feature-set configurations, over the
5 scaffold-grouped CV folds of train+val on the corrected v2 foundation.

The test set is never loaded here. Every selection decision downstream reads only the
OOF predictions this script writes.

Each config's OOF predictions are cached to results/oof/<config>.npz, so the script is
interruptible and re-runnable.

Usage: python3 06_tree_cv.py <model_type>:<feature_set> [more...]
       python3 06_tree_cv.py --baselines
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models import BASELINE_CONFIGS, build  # noqa: E402

DATA = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
OUT = "/home/claude/work/results/oof"
os.makedirs(OUT, exist_ok=True)

trainval = pd.read_csv(f"{DATA}/trainval.csv")
n_train = len(pd.read_csv(f"{DATA}/train.csv"))
folds = json.load(open(f"{DATA}/cv_folds.json"))["folds"]
y = trainval["Solubility"].values


def load_features(feature_set):
    """trainval feature matrix = train rows then val rows, matching trainval.csv order."""
    tr = np.load(f"{CACHE}/train__{feature_set}.npy")
    va = np.load(f"{CACHE}/val__{feature_set}.npy")
    X = np.vstack([tr, va]).astype(np.float64)
    assert X.shape[0] == len(trainval), f"{feature_set}: {X.shape[0]} vs {len(trainval)}"
    assert tr.shape[0] == n_train
    return X


def run(config_name, model_type, feature_set, kwargs, seed=42):
    out = f"{OUT}/{config_name}.npz"
    if os.path.exists(out):
        print(f"skip {config_name} (exists)")
        return
    X = load_features(feature_set)
    oof = np.full(len(trainval), np.nan)
    fold_rmse = []
    t0 = time.time()
    for i, f in enumerate(folds):
        tr, va = np.array(f["train"]), np.array(f["val"])
        m = build(model_type, seed=seed, **kwargs)
        m.fit(X[tr], y[tr])
        p = np.asarray(m.predict(X[va])).ravel()
        oof[va] = p
        r = float(np.sqrt(np.mean((y[va] - p) ** 2)))
        fold_rmse.append(r)
        print(f"  {config_name} fold {i}: rmse={r:.4f} ({time.time()-t0:.0f}s)", flush=True)
    assert not np.isnan(oof).any(), "OOF predictions incomplete"
    pooled = float(np.sqrt(np.mean((y - oof) ** 2)))
    np.savez(out, oof=oof, y=y, fold_rmse=np.array(fold_rmse),
             meta=json.dumps({"model_type": model_type, "feature_set": feature_set,
                              "kwargs": kwargs, "seed": seed,
                              "mean_fold_rmse": float(np.mean(fold_rmse)),
                              "std_fold_rmse": float(np.std(fold_rmse)),
                              "pooled_oof_rmse": pooled,
                              "seconds": time.time() - t0}))
    print(f"{config_name}: mean-fold RMSE={np.mean(fold_rmse):.4f}+-{np.std(fold_rmse):.4f}  "
          f"pooled-OOF RMSE={pooled:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--baselines":
        for name, (mt, fs, kw) in BASELINE_CONFIGS.items():
            run(name, mt, fs, kw)
    else:
        for a in args:
            mt, fs = a.split(":")
            run(f"{mt}[{fs}]", mt, fs, {})
