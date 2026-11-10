"""
13_save_blends.py
=================
Save the four blend/ensemble configurations: which saved models each combines, the
exact weights, and runnable code that reproduces the blended prediction from the
member models' outputs.

The weights are NOT re-derived here. They were fitted once, on out-of-fold train+val
predictions only (07_assemble_cv_results.py -> results/blend_weights.json), before the
test set was ever read, and are copied through unchanged -- which is the whole reason
the blend test numbers are honest.

Combination logic is a plain weighted sum of the member models' LogS predictions:

    logS_blend = sum_i  w_i * logS_i          with  w_i >= 0  and  sum_i w_i = 1

Each member's own prediction is itself the average over that model's saved seeds.
There is no meta-model, no stacking layer, no calibration step, and no preprocessing
at the blend level -- the weights below are the entire blend.

Output: saved_models/blends/
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import metrics  # noqa: E402

ROOT = "/home/claude/work"
RES = f"{ROOT}/results"
OUT = f"{ROOT}/saved_models/blends"
os.makedirs(OUT, exist_ok=True)


def slug(name):
    return name.replace("[", "_").replace("]", "").replace("+", "_plus_")


weights = json.load(open(f"{RES}/blend_weights.json"))
ft = pd.read_csv(f"{RES}/final_test_results.csv").set_index("config")
lb = pd.read_csv(f"{RES}/cv_leaderboard.csv").set_index("config")
recorded = np.load(f"{RES}/test_predictions.npz")
from_saved = np.load(f"{ROOT}/saved_models/_test_predictions_from_saved.npz")
y_te = recorded["y"]

PREDICT_BLEND = '''"""
predict_blend.py
Reproduce any saved blend's prediction from its member models.

    python3 predict_blend.py greedy_blend "CCO" "CC(=O)Oc1ccccc1C(=O)O"

    from predict_blend import predict_blend
    predict_blend("greedy_blend", ["CCO"])

Combination logic: a plain weighted sum of the members' LogS predictions, with
non-negative weights summing to 1. Each member's own prediction is the average over
that model's saved seeds. No meta-model, no calibration, no blend-level preprocessing.

Members with weight 0 are kept in the config for provenance (they were candidates
the weight fit rejected) and are skipped at inference time.
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.dirname(HERE)      # saved_models/


def _load_member(folder):
    """Import a member model folder's predict.py under its own name."""
    path = os.path.join(MODELS_ROOT, folder, "predict.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"member model folder not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_member_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def predict_blend(blend_name, smiles_list):
    cfg = json.load(open(os.path.join(HERE, f"{blend_name}.json")))
    total = np.zeros(len(smiles_list), dtype=float)
    used = []
    for m in cfg["members"]:
        if m["weight"] <= 1e-9:
            continue
        mod = _load_member(m["folder"])
        total += m["weight"] * np.asarray(mod.predict(smiles_list), dtype=float)
        used.append((m["folder"], m["weight"]))
    if not used:
        raise ValueError(f"{blend_name}: every member has weight 0")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    name, smis = sys.argv[1], sys.argv[2:]
    for s, p in zip(smis, predict_blend(name, smis)):
        print(f"{s}\\t{p:.4f}")
'''

rows = []
for name, spec in weights.items():
    members, w = spec["members"], spec["weights"]
    cfg = {
        "blend_name": name,
        "combination": "weighted_sum",
        "formula": "logS_blend = sum_i w_i * logS_i   (w_i >= 0, sum_i w_i = 1)",
        "weights_fitted_on": ("out-of-fold predictions over the 5 scaffold-grouped CV "
                              "folds of train+val only; never refitted on test"),
        "weights_sum": float(np.sum(w)),
        "members": [
            {"config": m, "folder": slug(m), "weight": round(float(wi), 12),
             "active": bool(wi > 1e-9)}
            for m, wi in zip(members, w)
        ],
        "member_prediction": ("each member's prediction is the mean over that model's "
                              "saved seeds -- see its folder's predict.py"),
        "test_rmse": float(ft.loc[name, "avgpred_rmse"]),
        "test_mae": float(ft.loc[name, "avgpred_mae"]),
        "test_r2": float(ft.loc[name, "avgpred_r2"]),
        "cv_pooled_oof_rmse": float(lb.loc[name + " [weighted]", "pooled_oof_rmse"]),
        "n_test": int(len(y_te)),
        "target": "LogS (log10 aqueous solubility, mol/L)",
    }

    # Rebuild the blend from what the SAVED models actually predict (loaded from disk
    # by 13a), not from the original training run's in-memory predictions, and record
    # how far that lands from the originally reported blend prediction.
    P_saved = np.column_stack([from_saved[m] for m in members])
    rebuilt = P_saved @ np.array(w)
    cfg["max_abs_diff_rebuilt_from_saved_vs_recorded"] = float(
        np.max(np.abs(rebuilt - recorded[name])))
    cfg["test_rmse_from_saved_models"] = float(
        np.sqrt(np.mean((y_te - rebuilt) ** 2)))
    cfg["test_r2_from_saved_models"] = float(
        1 - np.sum((y_te - rebuilt) ** 2) / np.sum((y_te - y_te.mean()) ** 2))
    json.dump(cfg, open(f"{OUT}/{name}.json", "w"), indent=2)

    active = [(m, round(wi, 4)) for m, wi in zip(members, w) if wi > 1e-9]
    rows.append((name, cfg["test_rmse"], cfg["test_r2"], cfg["cv_pooled_oof_rmse"], active))
    print(f"{name}: test RMSE={cfg['test_rmse']:.4f} R2={cfg['test_r2']:.4f}  "
          f"{len(active)} active members  "
          f"from-saved RMSE={cfg['test_rmse_from_saved_models']:.4f}  "
          f"rebuild Δ={cfg['max_abs_diff_rebuilt_from_saved_vs_recorded']:.2e}")

open(f"{OUT}/predict_blend.py", "w").write(PREDICT_BLEND)

readme = ["# Blends\n",
          "Each `<blend>.json` names the member models, their weights, and the",
          "combination logic. `predict_blend.py` runs any of them end to end.\n",
          "```bash",
          'python3 predict_blend.py greedy_blend "CC(=O)Oc1ccccc1C(=O)O"',
          "```\n",
          "A blend has **no weights of its own beyond these numbers** — it is a plain",
          "weighted sum of its members' LogS predictions. There is no meta-model, no",
          "stacking layer and no calibration. Weights were fitted on out-of-fold",
          "train+val predictions only, before the test set was read, and are copied",
          "here unchanged.\n",
          "| Blend | Test RMSE | Test R² | CV pooled-OOF RMSE | Active members |",
          "|---|---|---|---|---|"]
for name, rmse, r2, cv, active in sorted(rows, key=lambda r: r[1]):
    ms = "<br>".join(f"`{m}` × {wi}" for m, wi in active)
    readme.append(f"| **{name}** | {rmse:.4f} | {r2:.4f} | {cv:.4f} | {ms} |")
readme += ["",
           "Members listed with weight 0 in the JSON were candidates the weight fit",
           "rejected. They are kept for provenance and skipped at inference time.",
           "",
           "**Cost note.** Running a blend means loading and running every active",
           "member, each of which is itself a seed-average. `greedy_blend` is 6 tree",
           "models across 3 seeds plus a 5-seed GNN — 23 model loads and 4 separate",
           "featurizations per prediction — for a test-set RMSE that is not",
           "statistically distinguishable from `lightgbm_mordred` alone",
           "(gap 0.0090, 95% CI [−0.0037, +0.0217]). See `../README.md`."]
open(f"{OUT}/README.md", "w").write("\n".join(readme) + "\n")
print(f"\nSaved {OUT}/")
