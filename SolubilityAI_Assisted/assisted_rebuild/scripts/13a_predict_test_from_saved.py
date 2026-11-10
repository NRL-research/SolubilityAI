"""
13a_predict_test_from_saved.py
==============================
Load every saved model from disk (via its own folder's predict.py, as a future UI
would) and predict the held-out test set, writing the result to
saved_models/_test_predictions_from_saved.npz.

This exists so that the blend configs and the verification report are built from what
the SAVED WEIGHTS actually do, not from the predictions the original training run
happened to produce. If the two ever diverge, that divergence is reported rather than
papered over.

Deliberately goes through each folder's predict.py -- SMILES in, LogS out -- so the
featurization path, the feature-name ordering and the model loading are all exercised
end to end.
"""

import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/claude/work"
SAVED = f"{ROOT}/saved_models"
DATA = f"{ROOT}/data/v2_corrected"
RES = f"{ROOT}/results"

test = pd.read_csv(f"{DATA}/test.csv")
smiles = test["canonical_smiles"].tolist()
y_te = test["Solubility"].values
recorded = np.load(f"{RES}/test_predictions.npz")


def load_predict_module(folder):
    for k in list(sys.modules):
        if k.startswith("_pm_") or k in ("solubility_featurizer", "gatv2_model"):
            del sys.modules[k]
    sys.path.insert(0, folder)
    try:
        name = "_pm_" + os.path.basename(folder)
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, "predict.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(folder)


out, rows = {}, []
folders = sorted(d for d in glob.glob(f"{SAVED}/*")
                 if os.path.isdir(d) and os.path.basename(d) != "blends")

for folder in folders:
    meta = json.load(open(f"{folder}/metadata.json"))
    config = meta["config"]
    t0 = time.time()
    mod = load_predict_module(folder)
    p = np.asarray(mod.predict(smiles), dtype=float)
    assert p.shape == (len(test),), f"{config}: got shape {p.shape}"
    out[config] = p
    d = float(np.max(np.abs(p - recorded[config])))
    r = float(np.sqrt(np.mean((y_te - p) ** 2)))
    rows.append({"config": config, "folder": os.path.basename(folder),
                 "rmse_from_saved": r,
                 "rmse_originally_recorded": float(np.sqrt(np.mean(
                     (y_te - recorded[config]) ** 2))),
                 "max_abs_diff_vs_original": d,
                 "seconds": round(time.time() - t0, 1)})
    print(f"{config:32s} RMSE={r:.4f}  max|Δ| vs original run={d:.2e}  "
          f"({time.time()-t0:.0f}s)", flush=True)

np.savez(f"{SAVED}/_test_predictions_from_saved.npz", y=y_te, **out)
pd.DataFrame(rows).to_csv(f"{SAVED}/_saved_vs_original_predictions.csv", index=False)
print(f"\nSaved {SAVED}/_test_predictions_from_saved.npz")
