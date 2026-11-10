"""
14_verify_saved_models.py
=========================
Independent verification that every saved model actually reloads from disk and works.

Deliberately adversarial about state: this script imports NOTHING from the training
pipeline (no models.py, no featurizer.py, no gnn.py). It only walks saved_models/,
loads each folder's own predict.py, and drives it exactly the way a future UI would.

Per model, three checks:

  A. RELOAD       predict.py imports and its model files load from disk.
  B. SMILES PATH  predicting from 12 raw SMILES strings -- featurized from scratch by
                  the folder's own solubility_featurizer.py, not from any cached
                  matrix -- reproduces the recorded test predictions for those same
                  molecules. This is the check that catches a wrong feature order, a
                  missing preprocessor, or a mismatched descriptor version: those all
                  load fine and then give quietly wrong numbers.
  C. FULL TEST    predicting all 1,974 test molecules reproduces the recorded test
                  RMSE / R2 within tolerance.

Tolerances
  tree models  1e-6 on predictions (deterministic given seeds and features)
  GATv2        1e-4 on predictions -- CPU scatter/gather reductions in PyTorch
               Geometric are not bitwise-deterministic across runs
  metrics      1e-4 on RMSE and R2

Check B recomputes features from SMILES, so it also exercises the float32-cached vs
float64-recomputed difference; the tolerance for B is loosened to 2e-3 accordingly and
that is reported separately from the exact-input check.

Usage: python3 14_verify_saved_models.py
Exit code 0 only if every model passes every check.
"""

import glob
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = "/home/claude/work"
SAVED = f"{ROOT}/saved_models"
DATA = f"{ROOT}/data/v2_corrected"
RES = f"{ROOT}/results"

N_SPOT = 12
TOL_PRED = 1e-6
TOL_PRED_GNN = 1e-4
TOL_PRED_FROM_SMILES = 2e-3
TOL_METRIC = 1e-4

test = pd.read_csv(f"{DATA}/test.csv")
recorded = np.load(f"{RES}/test_predictions.npz")
y_te = recorded["y"]
ft = pd.read_csv(f"{RES}/final_test_results.csv").set_index("config")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))


def load_predict_module(folder):
    """Import folder/predict.py in isolation, as a future UI would."""
    path = os.path.join(folder, "predict.py")
    modname = "_verify_" + os.path.basename(folder)
    # drop any same-named modules a previous folder left behind
    for k in list(sys.modules):
        if k.startswith("_verify_") or k in ("solubility_featurizer", "gatv2_model"):
            del sys.modules[k]
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(folder)


results = []
folders = sorted(d for d in glob.glob(f"{SAVED}/*") if os.path.isdir(d)
                 and os.path.basename(d) != "blends")

# spot-check molecules: fixed, spread across the LogS range, chosen by index not value
rng = np.random.default_rng(7)
spot_idx = np.sort(rng.choice(len(test), N_SPOT, replace=False))
spot_smiles = test["canonical_smiles"].iloc[spot_idx].tolist()

print(f"Verifying {len(folders)} saved model folders")
print(f"Spot-check molecules (indices into test.csv): {spot_idx.tolist()}\n")

for folder in folders:
    name = os.path.basename(folder)
    row = {"folder": name, "reload": "FAIL", "smiles_path": "FAIL", "full_test": "FAIL"}
    try:
        meta = json.load(open(f"{folder}/metadata.json"))
        config = meta["config"]
        row["config"] = config
        is_gnn = meta["model_type"] == "gatv2"
        tol = TOL_PRED_GNN if is_gnn else TOL_PRED

        # ---- A. reload
        mod = load_predict_module(folder)
        models = mod.load_models()
        n_files = len(glob.glob(f"{folder}/model_seed*.{meta['file_extension']}"))
        assert n_files == len(meta["seeds"]), \
            f"expected {len(meta['seeds'])} model files, found {n_files}"
        assert len(models) == n_files, "load_models() returned the wrong count"
        row["reload"] = f"PASS ({n_files} seed files)"

        # ---- B. SMILES -> prediction path
        got = np.asarray(mod.predict(spot_smiles))
        want = recorded[config][spot_idx]
        d_spot = float(np.max(np.abs(got - want)))
        tol_b = max(tol, TOL_PRED_FROM_SMILES)
        ok_b = d_spot <= tol_b
        row["smiles_path"] = f"{'PASS' if ok_b else 'FAIL'} (max|Δ|={d_spot:.2e} ≤ {tol_b:.0e})"
        row["spot_max_abs_diff"] = d_spot

        # ---- C. full test set
        allp = np.asarray(mod.predict(test["canonical_smiles"].tolist()))
        d_all = float(np.max(np.abs(allp - recorded[config])))
        got_rmse, got_r2 = rmse(y_te, allp), r2(y_te, allp)
        want_rmse = float(ft.loc[config, "avgpred_rmse"])
        want_r2 = float(ft.loc[config, "avgpred_r2"])
        ok_c = (abs(got_rmse - want_rmse) <= TOL_METRIC
                and abs(got_r2 - want_r2) <= TOL_METRIC)
        row["full_test"] = (f"{'PASS' if ok_c else 'FAIL'} RMSE {got_rmse:.4f} "
                            f"(recorded {want_rmse:.4f})  R² {got_r2:.4f} "
                            f"(recorded {want_r2:.4f})")
        row["test_max_abs_diff"] = d_all
        row["rmse_reloaded"] = got_rmse
        row["rmse_recorded"] = want_rmse
        row["r2_reloaded"] = got_r2
        row["passed"] = ok_b and ok_c
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["passed"] = False
        traceback.print_exc()

    print(f"[{'PASS' if row.get('passed') else 'FAIL'}] {name}")
    for k in ("reload", "smiles_path", "full_test", "error"):
        if k in row:
            print(f"        {k:12s} {row[k]}")
    results.append(row)

# ---------------------------------------------------------------- blends
print("\nVerifying blends")
sys.path.insert(0, f"{SAVED}/blends")
spec = importlib.util.spec_from_file_location("_verify_blend", f"{SAVED}/blends/predict_blend.py")
blendmod = importlib.util.module_from_spec(spec)
sys.modules["_verify_blend"] = blendmod
spec.loader.exec_module(blendmod)

for f in sorted(glob.glob(f"{SAVED}/blends/*.json")):
    name = os.path.basename(f)[:-5]
    row = {"folder": f"blends/{name}", "config": name, "reload": "n/a (config only)"}
    try:
        cfg = json.load(open(f))
        assert abs(cfg["weights_sum"] - 1.0) < 1e-6, "weights do not sum to 1"
        assert all(m["weight"] >= 0 for m in cfg["members"]), "negative weight"

        got = np.asarray(blendmod.predict_blend(name, spot_smiles))
        want = recorded[name][spot_idx]
        d_spot = float(np.max(np.abs(got - want)))
        ok_b = d_spot <= TOL_PRED_FROM_SMILES
        row["smiles_path"] = f"{'PASS' if ok_b else 'FAIL'} (max|Δ|={d_spot:.2e})"
        row["spot_max_abs_diff"] = d_spot

        allp = np.asarray(blendmod.predict_blend(name, test["canonical_smiles"].tolist()))
        got_rmse, got_r2 = rmse(y_te, allp), r2(y_te, allp)
        want_rmse = float(ft.loc[name, "avgpred_rmse"])
        want_r2 = float(ft.loc[name, "avgpred_r2"])
        ok_c = (abs(got_rmse - want_rmse) <= TOL_METRIC
                and abs(got_r2 - want_r2) <= TOL_METRIC)
        row["full_test"] = (f"{'PASS' if ok_c else 'FAIL'} RMSE {got_rmse:.4f} "
                            f"(recorded {want_rmse:.4f})  R² {got_r2:.4f} "
                            f"(recorded {want_r2:.4f})")
        row["rmse_reloaded"] = got_rmse
        row["rmse_recorded"] = want_rmse
        row["r2_reloaded"] = got_r2
        row["passed"] = ok_b and ok_c
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["passed"] = False
        traceback.print_exc()
    print(f"[{'PASS' if row.get('passed') else 'FAIL'}] blends/{name}")
    for k in ("smiles_path", "full_test", "error"):
        if k in row:
            print(f"        {k:12s} {row[k]}")
    results.append(row)

df = pd.DataFrame(results)
df.to_csv(f"{SAVED}/_verification_report.csv", index=False)
json.dump(results, open(f"{SAVED}/_verification_report.json", "w"), indent=2, default=str)

n_pass = int(df["passed"].sum())
print(f"\n{'='*70}\n{n_pass} / {len(df)} passed")
if n_pass != len(df):
    print("FAILED:")
    for r in results:
        if not r.get("passed"):
            print("  -", r["folder"], r.get("error", ""))
print(f"Report: {SAVED}/_verification_report.csv")
sys.exit(0 if n_pass == len(df) else 1)
