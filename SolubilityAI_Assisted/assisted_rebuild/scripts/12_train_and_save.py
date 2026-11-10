"""
12_train_and_save.py
====================
Re-fit every FINAL model configuration on the full train+val pool and save it to disk
in its library's native format, together with everything needed to use it on a new
molecule without this repository.

Why a re-fit is needed: the original pipeline scripts (06_tree_cv.py, 05_gnn_cv.py,
08_final_test_eval.py) persisted predictions and metrics but never called save_model /
joblib.dump / state_dict(), so no weights survived. This script reproduces the same
models exactly -- same frozen split files, same cached feature matrices, same
hyperparameters in models.py, same seeds -- rather than training anything new. No
selection decision is revisited and the test set is used only to check that the
re-fitted models reproduce the predictions already recorded in
results/test_predictions.npz.

Each model folder gets:
  model_seed<N>.<ext>      native-format weights, one per seed
  feature_names.json       exact ordered column names the model was trained with
  solubility_featurizer.py self-contained SMILES -> feature vector code
  predict.py               runnable: SMILES -> predicted LogS
  metadata.json            model type, features, data version, hyperparameters, metrics
  README.md                what it is, what it expects, what it outputs, performance

Blends get their members, weights and combination logic in saved_models/blends/.

Output: saved_models/
"""

import glob
import json
import os
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import metrics  # noqa: E402
from models import build  # noqa: E402
from solubility_featurizer import feature_names  # noqa: E402

ROOT = "/home/claude/work"
DATA = f"{ROOT}/data/v2_corrected"
CACHE = f"{ROOT}/data/features_cache"
RES = f"{ROOT}/results"
OUT = f"{ROOT}/saved_models"
SCRIPTS = f"{ROOT}/scripts"

TREE_SEEDS = [42, 101, 123]
GNN_SEEDS = [42, 101, 123, 456, 789]

os.makedirs(OUT, exist_ok=True)

trainval = pd.read_csv(f"{DATA}/trainval.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y_tv = trainval["Solubility"].values
y_te = test["Solubility"].values

final_cfg = json.load(open(f"{RES}/final_configs.json"))
lb = pd.read_csv(f"{RES}/cv_leaderboard.csv").set_index("config")
ft = pd.read_csv(f"{RES}/final_test_results.csv").set_index("config")
recorded = np.load(f"{RES}/test_predictions.npz")

# folder name <- config name, Windows-safe
def slug(name):
    return name.replace("[", "_").replace("]", "").replace("+", "_plus_")


SAVE_EXT = {"catboost": "cbm", "xgboost": "json", "lightgbm": "txt",
            "hist_gb": "joblib", "extra_trees": "joblib", "gatv2": "pt"}


def save_model(model, model_type, path):
    if model_type == "catboost":
        model.save_model(path, format="cbm")
    elif model_type == "xgboost":
        model.get_booster().save_model(path)          # native booster JSON
    elif model_type == "lightgbm":
        model.booster_.save_model(path)                # native LightGBM text
    elif model_type in ("hist_gb", "extra_trees"):
        joblib.dump(model, path, compress=3)           # sklearn standard
    else:
        raise ValueError(model_type)


def load_X(fs):
    tr = np.load(f"{CACHE}/train__{fs}.npy")
    va = np.load(f"{CACHE}/val__{fs}.npy")
    te = np.load(f"{CACHE}/test__{fs}.npy")
    X_tv = np.vstack([tr, va]).astype(np.float64)
    assert X_tv.shape[0] == len(trainval) and te.shape[0] == len(test)
    return X_tv, te.astype(np.float64)


PREDICT_TREE = '''"""
predict.py -- {config}
Predict aqueous solubility (LogS, log10 mol/L) for one or more SMILES strings.

    python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
    from predict import predict; predict(["CCO"])

Loads all {n_seeds} saved seeds and averages their predictions -- that seed-averaged
prediction is what the recorded test metrics in metadata.json refer to. Passing a
single seed gives a slightly different (and slightly worse) model.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solubility_featurizer import featurize  # noqa: E402

FEATURE_SET = "{feature_set}"
MODEL_TYPE = "{model_type}"
FEATURE_NAMES = json.load(open(os.path.join(HERE, "feature_names.json")))


def load_models():
    paths = sorted(glob.glob(os.path.join(HERE, "model_seed*.{ext}")))
    if not paths:
        raise FileNotFoundError("no model_seed*.{ext} files next to predict.py")
    models = []
    for p in paths:
        if MODEL_TYPE == "catboost":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(p)
            models.append(("catboost", m))
        elif MODEL_TYPE == "xgboost":
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(p)
            models.append(("xgboost", b))
        elif MODEL_TYPE == "lightgbm":
            import lightgbm as lgb
            models.append(("lightgbm", lgb.Booster(model_file=p)))
        else:
            import joblib
            models.append(("sklearn", joblib.load(p)))
    return models


def predict(smiles_list, models=None):
    """SMILES -> np.ndarray of predicted LogS (log10 solubility in mol/L)."""
    X = featurize(smiles_list, FEATURE_SET, feature_names=FEATURE_NAMES)
    if models is None:
        models = load_models()
    preds = []
    for kind, m in models:
        if kind == "xgboost":
            import xgboost as xgb
            preds.append(np.asarray(m.predict(xgb.DMatrix(X))).ravel())
        else:
            preds.append(np.asarray(m.predict(X)).ravel())
    return np.mean(np.column_stack(preds), axis=1)


if __name__ == "__main__":
    smis = sys.argv[1:] or ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    for s, p in zip(smis, predict(smis)):
        print(f"{{s}}\\t{{p:.4f}}")
'''

PREDICT_GNN = '''"""
predict.py -- Hybrid GATv2
Predict aqueous solubility (LogS, log10 mol/L) for one or more SMILES strings.

    python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"

Loads all 5 saved seeds and averages. Requires torch and torch_geometric.
No fitted preprocessor: the 17 atom features, 4 bond features and 5 global
descriptors are all computed directly from the molecule.
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gatv2_model import HybridMolecularGATv2, smiles_to_hybrid_graph  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


def load_models():
    paths = sorted(glob.glob(os.path.join(HERE, "model_seed*.pt")))
    if not paths:
        raise FileNotFoundError("no model_seed*.pt files next to predict.py")
    models = []
    for p in paths:
        m = HybridMolecularGATv2()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    return models


def predict(smiles_list, models=None):
    graphs = []
    for i, s in enumerate(smiles_list):
        g = smiles_to_hybrid_graph(s)
        if g is None:
            raise ValueError(f"RDKit could not parse SMILES at index {i}: {s!r}")
        graphs.append(g)
    if models is None:
        models = load_models()
    loader = list(DataLoader(graphs, batch_size=256, shuffle=False))
    out = []
    for m in models:
        preds = []
        with torch.no_grad():
            for batch in loader:
                preds.extend(m(batch).squeeze(-1).cpu().numpy().ravel())
        out.append(np.array(preds))
    return np.mean(np.column_stack(out), axis=1)


if __name__ == "__main__":
    smis = sys.argv[1:] or ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    for s, p in zip(smis, predict(smis)):
        print(f"{{s}}\\t{{p:.4f}}")
'''


def write_readme(folder, config, meta, ext, n_seeds, is_gnn=False):
    cv = meta["cv_pooled_oof_rmse"]
    feat_line = ("a molecular graph (17 atom features, 4 bond features) plus 5 global "
                 "descriptors [MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors]"
                 if is_gnn else
                 f"a {meta['n_features']}-dimensional vector, feature set "
                 f"`{meta['feature_set']}`")
    code = ("from predict import predict\nprint(predict(['CCO']))"
            if True else "")
    txt = f"""# {config}

{meta['description']}

## What it is
- **Library / model type:** {meta['library']} — `{meta['model_type']}`
- **Saved format:** `{ext}` ({meta['save_method']}), one file per seed: {n_seeds} seeds ({', '.join(str(s) for s in meta['seeds'])})
- **Trained on:** {meta['dataset_version']} — train+val pool, {meta['n_train']} molecules
- **Split:** scaffold-grouped, seed {meta['split_seed']}; the {meta['n_test']}-molecule test set below was never seen in training

## Input
Takes {feat_line}.

`solubility_featurizer.py` in this folder does the whole SMILES → input conversion.
`feature_names.json` pins the exact column order the model was trained with — pass it
in (as `predict.py` does) so a different RDKit/Mordred version cannot silently shift
the columns.

**There is no fitted preprocessor.** No scaler, encoder, imputer or feature selection
was fit on the training data — every feature is a pure per-molecule function of the
SMILES string, and the only value-level processing is non-finite → 0 and clipping to
±1e30, both already inside `solubility_featurizer.py`. Nothing else to load.

## Output
A single number per molecule: **LogS**, the base-10 logarithm of aqueous solubility in
mol/L. LogS −4 ≈ 10⁻⁴ mol/L. Most oral drugs sit between −4 and −1.

## Performance
| Metric | Value |
|---|---|
| Test RMSE (seed-averaged) | **{meta['test_rmse']:.4f}** |
| Test MAE | {meta['test_mae']:.4f} |
| Test R² | {meta['test_r2']:.4f} |
| 5-fold scaffold-CV pooled-OOF RMSE | {cv if cv is None else f"{cv:.4f}"} |
| Per-seed test RMSE spread | {meta['test_rmse_seed_std']} |

Test set = {meta['n_test']} molecules, scaffold-isolated from training, with all SC1 /
SC2-tight / SC2-loose benchmark compounds already removed from the whole dataset.

## Use it
```bash
pip install {meta['pip']}
python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
```
```python
from predict import predict
predict(["CC(=O)Oc1ccccc1C(=O)O"])     # -> array([...]) LogS
```
`predict()` loads all {n_seeds} seeds and averages them — that average is what the
metrics above describe. A single seed is a slightly different, slightly worse model.

## Hyperparameters
```json
{json.dumps(meta['hyperparameters'], indent=2)}
```
"""
    open(f"{folder}/README.md", "w").write(txt)


# ---------------------------------------------------------------- main
manifest = []
t_start = time.time()

for config, spec in sorted(final_cfg["single_models"].items()):
    mt, fs = spec["model_type"], spec["feature_set"]
    folder = f"{OUT}/{slug(config)}"
    os.makedirs(folder, exist_ok=True)
    ext = SAVE_EXT[mt]
    ftrow = ft.loc[config]
    cvrow = lb.loc[config] if config in lb.index else None

    print(f"\n=== {config} -> {folder} ===", flush=True)

    # resumable: skip a config whose weights are already on disk
    n_seeds_expected = len(GNN_SEEDS if mt == "gatv2" else TREE_SEEDS)
    existing = glob.glob(f"{folder}/model_seed*.{ext}")
    if len(existing) == n_seeds_expected and os.path.exists(f"{folder}/metadata.json"):
        print(f"  already saved ({len(existing)} seed files) -- skipping re-fit", flush=True)
        manifest.append(json.load(open(f"{folder}/metadata.json")))
        continue

    if mt == "gatv2":
        import torch
        torch.set_num_threads(1)  # REQUIRED: 2+ threads changes float accumulation order in
        # PyG scatter reductions and yields a different (equally valid but
        # non-reproducing) model. Verified: 1 thread reproduces bit-exactly.
        from gnn import build_graphs, train_gatv2, predict_gatv2
        graphs, ok = build_graphs(trainval)
        assert ok.all()
        gte, ok2 = build_graphs(test)
        assert ok2.all()
        preds = []
        for s in GNN_SEEDS:
            p = f"{folder}/model_seed{s}.pt"
            t0 = time.time()
            m = train_gatv2(graphs, seed=s)
            torch.save(m.state_dict(), p)
            preds.append(predict_gatv2(m, gte))
            print(f"  seed {s} saved ({time.time()-t0:.0f}s)", flush=True)
        seeds, n_features, lib = GNN_SEEDS, None, "PyTorch + PyTorch Geometric"
        hyper = {"architecture": "HybridMolecularGATv2",
                 "gat1": "GATv2Conv(17 -> 32, heads=4, edge_dim=4)",
                 "gat2": "GATv2Conv(128 -> 64, heads=2, edge_dim=4)",
                 "head": "Linear(133->64) -> Linear(64->32) -> Linear(32->1)",
                 "dropout": [0.1, 0.2], "pool": "global_mean_pool",
                 "optimizer": "AdamW(lr=0.001, weight_decay=0.01)",
                 "loss": "MSELoss", "epochs": 60, "batch_size": 64,
                 "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)"}
        save_method = "torch.save(model.state_dict())"
        pip = "torch torch_geometric rdkit"
        shutil.copy(f"{SCRIPTS}/gnn.py", f"{folder}/gatv2_model.py")
        open(f"{folder}/predict.py", "w").write(PREDICT_GNN)
        json.dump({"note": "GATv2 consumes a molecular graph, not a flat feature vector",
                   "global_descriptors_in_order":
                       ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors"]},
                  open(f"{folder}/feature_names.json", "w"), indent=2)
    else:
        X_tv, X_te = load_X(fs)
        names = feature_names(fs)
        assert len(names) == X_tv.shape[1], f"{fs}: {len(names)} names vs {X_tv.shape[1]} cols"
        preds = []
        for s in TREE_SEEDS:
            p = f"{folder}/model_seed{s}.{ext}"
            t0 = time.time()
            m = build(mt, seed=s, **spec.get("kwargs", {}))
            m.fit(X_tv, y_tv)
            save_model(m, mt, p)
            preds.append(np.asarray(m.predict(X_te)).ravel())
            print(f"  seed {s} saved ({time.time()-t0:.0f}s)", flush=True)
        seeds, n_features = TREE_SEEDS, X_tv.shape[1]
        lib = {"catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM",
               "hist_gb": "scikit-learn", "extra_trees": "scikit-learn"}[mt]
        hyper = build(mt, seed=42, **spec.get("kwargs", {})).get_params()
        hyper = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in hyper.items()}
        save_method = {"catboost": "CatBoostRegressor.save_model(format='cbm')",
                       "xgboost": "Booster.save_model() (native JSON)",
                       "lightgbm": "Booster.save_model() (native text)",
                       "hist_gb": "joblib.dump", "extra_trees": "joblib.dump"}[mt]
        pip = {"catboost": "catboost rdkit", "xgboost": "xgboost rdkit",
               "lightgbm": "lightgbm rdkit", "hist_gb": "scikit-learn rdkit",
               "extra_trees": "scikit-learn rdkit"}[mt]
        if fs == "mordred":
            pip += " mordredcommunity"
        json.dump(names, open(f"{folder}/feature_names.json", "w"))
        shutil.copy(f"{SCRIPTS}/solubility_featurizer.py", folder)
        open(f"{folder}/predict.py", "w").write(
            PREDICT_TREE.format(config=config, feature_set=fs, model_type=mt,
                                ext=ext, n_seeds=len(TREE_SEEDS)))

    P = np.column_stack(preds)
    avg = P.mean(axis=1)
    per_seed = [metrics(y_te, P[:, i])["rmse"] for i in range(P.shape[1])]
    m_avg = metrics(y_te, avg)

    # reproduction check against the already-recorded predictions
    ref = recorded[config]
    maxdiff = float(np.max(np.abs(avg - ref)))

    meta = {
        "config": config, "model_type": mt, "library": lib,
        "feature_set": fs, "n_features": n_features,
        "seeds": seeds, "save_method": save_method, "file_extension": ext,
        "dataset_version": "v2_sc_filtered_scaffold_fixed",
        "split_seed": 42, "n_train": len(trainval), "n_test": len(test),
        "target": "LogS (log10 aqueous solubility, mol/L)",
        "fitted_preprocessors": [],
        "preprocessing_note": ("none fitted on training data; featurization is a pure "
                               "per-molecule function (non-finite -> 0, clip to +/-1e30)"),
        "hyperparameters": hyper,
        "test_rmse": m_avg["rmse"], "test_mae": m_avg["mae"], "test_r2": m_avg["r2"],
        "test_rmse_per_seed": per_seed,
        "test_rmse_seed_std": round(float(np.std(per_seed)), 6),
        "cv_pooled_oof_rmse": (None if cvrow is None else float(cvrow["pooled_oof_rmse"])),
        "recorded_test_rmse": float(ftrow["avgpred_rmse"]),
        "max_abs_diff_vs_recorded_predictions": maxdiff,
        "pip": pip,
        "description": {
            "lightgbm[mordred]": "Best single model in the project. LightGBM over ~1,600 Mordred 2D descriptors.",
            "xgboost[mordred]": "XGBoost over Mordred 2D descriptors. Nearly as accurate as the LightGBM version and much faster to fit.",
            "hist_gb[mordred]": "scikit-learn HistGradientBoosting over Mordred descriptors. Fully deterministic (identical across seeds).",
            "lightgbm[rdkit_descriptors]": "LightGBM over the 217 standard RDKit 2D descriptors. No Mordred dependency.",
            "catboost[rdkit_full_eng]": "CatBoost over RDKit descriptors plus 8 engineered solubility-relevant ratios.",
            "extra_trees[rdkit_descriptors]": "ExtraTrees over RDKit descriptors. Included for ensemble diversity, not accuracy.",
            "catboost_solo[solubility_ai]": "The project's original baseline: CatBoost over Morgan2048 + 5 descriptors.",
            "lightgbm[solubility_ai]": "LightGBM over the original Morgan2048 + 5 descriptor space.",
            "xgboost_branch[solubility_ai]": "The XGBoost branch of the original SolubilityAI ensemble.",
            "gatv2[graph+5desc]": "The Hybrid GATv2 graph neural network branch of the original SolubilityAI ensemble.",
        }.get(config, ""),
    }
    json.dump(meta, open(f"{folder}/metadata.json", "w"), indent=2)
    write_readme(folder, config, meta, ext, len(seeds), is_gnn=(mt == "gatv2"))
    manifest.append(meta)
    print(f"  -> test RMSE {m_avg['rmse']:.4f} (recorded {ftrow['avgpred_rmse']:.4f}), "
          f"max|Δpred| vs recorded = {maxdiff:.2e}", flush=True)

json.dump(manifest, open(f"{OUT}/_manifest.json", "w"), indent=2)
print(f"\nAll single models saved in {time.time()-t_start:.0f}s")
