"""
07_assemble_cv_results.py
=========================
Collect every OOF prediction file (tree/sklearn configs from 06_tree_cv.py, GATv2
folds from 05_gnn_cv.py) into one leaderboard, build candidate ensembles, and WRITE
DOWN the final configurations to be evaluated on test -- all strictly from out-of-fold
train+val predictions. The test set is never read by this script.

Ensembles built
---------------
SolubilityAI_ensemble       GATv2 + XGBoost branch + CatBoost branch, the project's own
                            architecture, blend weights fitted on OOF (replaces the
                            prior random-validation-carve-out Optuna tuning).
diverse_blend_4             best config from each distinct (model family, feature set)
                            pairing, 4 members -- mirrors the collaborator's round-2
                            ensemble design on this corrected foundation.
diverse_blend_4_plus_gnn    the above plus GATv2, to test whether the graph branch adds
                            anything the descriptor models miss.
greedy_blend                forward selection with replacement on pooled-OOF RMSE.
                            Flagged as the most OOF-overfit-prone candidate: its member
                            set AND its weights are both chosen on the same OOF
                            predictions, so its OOF score is optimistically biased and
                            its test number is the honest one.

Outputs
-------
results/cv_leaderboard.csv
results/blend_weights.json
results/oof_matrix.npz
results/final_configs.json   <- consumed by 08_final_test_eval.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models import optimize_blend_weights  # noqa: E402

DATA = "/home/claude/work/data/v2_corrected"
OOF = "/home/claude/work/results/oof"
GNN = "/home/claude/work/results/gnn"
RES = "/home/claude/work/results"

trainval = pd.read_csv(f"{DATA}/trainval.csv")
y = trainval["Solubility"].values
folds = json.load(open(f"{DATA}/cv_folds.json"))["folds"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))


preds, spec, rows = {}, {}, []

for f in sorted(glob.glob(f"{OOF}/*.npz")):
    d = np.load(f, allow_pickle=True)
    name = os.path.basename(f)[:-4]
    meta = json.loads(str(d["meta"]))
    preds[name] = d["oof"]
    spec[name] = {"model_type": meta["model_type"], "feature_set": meta["feature_set"],
                  "kwargs": meta["kwargs"]}
    rows.append({"config": name, "model": meta["model_type"], "feature_set": meta["feature_set"],
                 "mean_fold_rmse": round(meta["mean_fold_rmse"], 4),
                 "std_fold_rmse": round(meta["std_fold_rmse"], 4),
                 "pooled_oof_rmse": round(meta["pooled_oof_rmse"], 4),
                 "pooled_oof_r2": round(r2(y, d["oof"]), 4),
                 "fit_seconds": round(meta["seconds"])})

gnn_files = sorted(glob.glob(f"{GNN}/oof_fold*_seed42.npz"))
if len(gnn_files) == len(folds):
    oof = np.full(len(trainval), np.nan)
    fr = []
    for f in gnn_files:
        d = np.load(f)
        oof[d["idx"]] = d["pred"]
        fr.append(rmse(d["y"], d["pred"]))
    assert not np.isnan(oof).any()
    preds["gatv2[graph+5desc]"] = oof
    spec["gatv2[graph+5desc]"] = {"model_type": "gatv2", "feature_set": "graph+5desc", "kwargs": {}}
    rows.append({"config": "gatv2[graph+5desc]", "model": "gatv2", "feature_set": "graph+5desc",
                 "mean_fold_rmse": round(float(np.mean(fr)), 4),
                 "std_fold_rmse": round(float(np.std(fr)), 4),
                 "pooled_oof_rmse": round(rmse(y, oof), 4),
                 "pooled_oof_r2": round(r2(y, oof), 4), "fit_seconds": None})
else:
    print(f"WARNING: GATv2 OOF incomplete ({len(gnn_files)}/{len(folds)})")

lb = pd.DataFrame(rows).sort_values("pooled_oof_rmse").reset_index(drop=True)

# ---------------------------------------------------------------- candidate ensembles
ensembles = {}

sai = ["gatv2[graph+5desc]", "xgboost_branch[solubility_ai]", "catboost_solo[solubility_ai]"]
if all(k in preds for k in sai):
    ensembles["SolubilityAI_ensemble"] = sai

# diverse: best config per (model family, feature set), never repeating either
diverse, seen_m, seen_f = [], set(), set()
for _, r in lb.iterrows():
    if r["model"] == "gatv2":
        continue
    if r["model"] in seen_m or r["feature_set"] in seen_f:
        continue
    diverse.append(r["config"])
    seen_m.add(r["model"])
    seen_f.add(r["feature_set"])
    if len(diverse) == 4:
        break
if len(diverse) >= 3:
    ensembles["diverse_blend_4"] = diverse
    if "gatv2[graph+5desc]" in preds:
        ensembles["diverse_blend_4_plus_gnn"] = diverse + ["gatv2[graph+5desc]"]

# greedy forward selection with replacement
cands = list(preds.keys())
chosen, best_hist = [], []
cur = None
for _ in range(8):
    best_name, best_score, best_pred = None, np.inf, None
    for c in cands:
        trial = chosen + [c]
        P = np.column_stack([preds[m] for m in trial])
        w = optimize_blend_weights(P, y, seed=42)
        s = rmse(y, P @ w)
        if s < best_score:
            best_name, best_score, best_pred = c, s, trial
    if cur is not None and best_score >= cur - 1e-4:
        break
    chosen, cur = best_pred, best_score
    best_hist.append({"added": best_name, "pooled_oof_rmse": round(best_score, 4)})
if len(set(chosen)) >= 2:
    ensembles["greedy_blend"] = chosen
print("Greedy forward selection trace:")
for h in best_hist:
    print("  ", h)

# ---------------------------------------------------------------- score ensembles
ens_rows, weights_out = [], {}
for name, members in ensembles.items():
    P = np.column_stack([preds[m] for m in members])
    w = optimize_blend_weights(P, y, seed=42)
    weights_out[name] = {"members": members, "weights": [float(x) for x in w]}
    for tag, p in [("weighted", P @ w), ("unweighted mean", P.mean(axis=1))]:
        ens_rows.append({"config": f"{name} [{tag}]", "model": "blend", "feature_set": "-",
                         "mean_fold_rmse": None, "std_fold_rmse": None,
                         "pooled_oof_rmse": round(rmse(y, p), 4),
                         "pooled_oof_r2": round(r2(y, p), 4), "fit_seconds": None})
    print(f"\n{name}\n  members: {members}\n  weights: {np.round(w,4).tolist()}"
          f"\n  pooled-OOF RMSE weighted={rmse(y, P@w):.4f}  unweighted={rmse(y, P.mean(axis=1)):.4f}")

full = pd.concat([lb, pd.DataFrame(ens_rows)], ignore_index=True).sort_values("pooled_oof_rmse")
full.to_csv(f"{RES}/cv_leaderboard.csv", index=False)
with open(f"{RES}/blend_weights.json", "w") as f:
    json.dump(weights_out, f, indent=2)
np.savez(f"{RES}/oof_matrix.npz", y=y, **preds)

# ---------------------------------------------------------------- freeze final configs
needed = sorted({m for v in weights_out.values() for m in v["members"]}
                | {"catboost_solo[solubility_ai]", lb.iloc[0]["config"]})
single = {}
for n in needed:
    s = dict(spec[n])
    s["multiseed"] = True
    single[n] = s
final = {"single_models": single,
         "blends": weights_out,
         "note": "Chosen from out-of-fold train+val predictions only; test not yet read."}
with open(f"{RES}/final_configs.json", "w") as f:
    json.dump(final, f, indent=2)

pd.set_option("display.width", 220)
print("\n" + full.to_string(index=False))
print(f"\nSaved {RES}/cv_leaderboard.csv, blend_weights.json, oof_matrix.npz, final_configs.json")
