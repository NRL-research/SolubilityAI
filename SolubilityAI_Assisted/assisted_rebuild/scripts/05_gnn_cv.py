"""
05_gnn_cv.py
============
Out-of-fold GATv2 predictions over the 5 scaffold-grouped CV folds of train+val
(corrected v2 foundation), plus a full-trainval model's test predictions.

The GNN is the expensive branch of the SolubilityAI ensemble (~1.5-2 h per training
run on this 2-core container), so every fold's predictions are written to disk as
soon as they are produced and the script skips folds already on disk -- it is safe
to interrupt and re-run.

Test set is touched only by the FINAL stage, and only to produce predictions from a
model trained on train+val; no GNN decision is made using test.

Usage: python3 05_gnn_cv.py cv          # the 5 OOF folds
       python3 05_gnn_cv.py final SEED  # full-trainval model -> test predictions
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
from gnn import build_graphs, train_gatv2, predict_gatv2  # noqa: E402

DATA = "/home/claude/work/data/v2_corrected"
OUT = "/home/claude/work/results/gnn"
os.makedirs(OUT, exist_ok=True)

trainval = pd.read_csv(f"{DATA}/trainval.csv")
folds = json.load(open(f"{DATA}/cv_folds.json"))["folds"]


def run_cv(seed=42):
    graphs, ok = build_graphs(trainval)
    assert ok.all(), "some trainval molecules failed to build a graph"
    for i, f in enumerate(folds):
        out = f"{OUT}/oof_fold{i}_seed{seed}.npz"
        if os.path.exists(out):
            print(f"skip fold {i} (exists)")
            continue
        tr, va = f["train"], f["val"]
        t0 = time.time()
        model = train_gatv2([graphs[j] for j in tr], seed=seed)
        preds = predict_gatv2(model, [graphs[j] for j in va])
        np.savez(out, idx=np.array(va), pred=preds,
                 y=trainval["Solubility"].values[va])
        rmse = float(np.sqrt(np.mean((trainval["Solubility"].values[va] - preds) ** 2)))
        print(f"fold {i}: n_val={len(va)} rmse={rmse:.4f}  ({time.time()-t0:.0f}s)", flush=True)


def run_final(seed):
    out = f"{OUT}/test_pred_seed{seed}.npz"
    if os.path.exists(out):
        print(f"skip final seed {seed} (exists)")
        return
    test = pd.read_csv(f"{DATA}/test.csv")
    gtr, ok1 = build_graphs(trainval)
    gte, ok2 = build_graphs(test)
    assert ok1.all() and ok2.all()
    t0 = time.time()
    model = train_gatv2(gtr, seed=seed)
    np.savez(out, pred=predict_gatv2(model, gte), y=test["Solubility"].values)
    print(f"final seed {seed} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if mode == "cv":
        run_cv()
    else:
        run_final(int(sys.argv[2]))
