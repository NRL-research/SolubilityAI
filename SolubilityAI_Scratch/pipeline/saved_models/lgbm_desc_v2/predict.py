"""Standalone predictor for this saved model folder.

Usage (CLI):   python predict.py "CCO" "c1ccccc1"
Usage (code):  from predict import load_predictor; p = load_predictor(); p(["CCO"])  -> np.ndarray LogS

Everything needed is in this folder: config.json (feature set, descriptor order, family,
hyperparameters), the native model file, preprocessor.joblib (if the model uses one), and
featurize.py (exact featurization code used in training).  Output: predicted LogS
(log10 mol/L).  Unparseable SMILES -> NaN.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_predictor(folder=HERE):
    folder = Path(folder)
    cfg = json.load(open(folder / "config.json"))
    fam = cfg["family"]
    tag = folder.name.replace("-", "_")
    F = _load_module(folder / "featurize.py", f"featurize_{tag}")

    if fam == "blend":
        comps = [load_predictor(folder.parent / c["folder"]) for c in cfg["components"]]
        w = np.array([c["weight"] for c in cfg["components"]]); b = cfg.get("intercept", 0.0)
        def predict(smiles):
            P = np.column_stack([c(smiles) for c in comps])
            return P @ w + b
        return predict

    if fam in ("gnn", "gnn_hybrid"):
        import torch
        G = _load_module(folder / "gnn_model.py", f"gnn_model_{tag}")
        torch.set_num_threads(2)
        seeds = cfg["seeds"]
        nets = []
        for s in seeds:
            net = G.MPNN(**cfg["arch"])
            net.load_state_dict(torch.load(folder / f"model_seed{s}.pt", map_location="cpu"))
            net.eval(); nets.append(net)
        pre = None
        if fam == "gnn_hybrid":
            import joblib; pre = joblib.load(folder / "preprocessor.joblib")
        def predict(smiles):
            ok = np.array([F.standardize(s) is not None for s in smiles])
            out = np.full(len(smiles), np.nan)
            sm = [s for s, o in zip(smiles, ok) if o]
            if not sm: return out
            gs = F.graphs(sm)
            gx = None
            if pre is not None:
                gx = pre.transform(F.featurize(sm, "rdkit_desc", cfg["desc_names"])).astype(np.float32)
            preds = [G.predict_graphs(n, gs, gx) for n in nets]
            out[ok] = np.mean(preds, axis=0) * cfg["y_std"] + cfg["y_mean"]
            return out
        return predict

    # ---------- tabular families ----------
    pre = None
    if (folder / "preprocessor.joblib").exists():
        import joblib; pre = joblib.load(folder / "preprocessor.joblib")
    if fam == "lgbm":
        import lightgbm as lgb; model = lgb.Booster(model_file=str(folder / "model.lgbm.txt"))
    elif fam == "xgb":
        import xgboost as xgb; model = xgb.Booster(); model.load_model(str(folder / "model.xgb.json"))
    elif fam == "catboost":
        from catboost import CatBoostRegressor; model = CatBoostRegressor(); model.load_model(str(folder / "model.cbm"))
    else:
        import joblib; model = joblib.load(folder / "model.joblib")

    def predict(smiles):
        smiles = list(smiles)
        ok = np.array([F.standardize(s) is not None for s in smiles])
        X = F.featurize(smiles, cfg["feature_set"], cfg.get("desc_names"))
        if fam == "knn": X = X.astype(bool)
        if pre is not None: X = pre.transform(X)
        if fam == "xgb":
            import xgboost as xgb; y = model.predict(xgb.DMatrix(X))
        else:
            y = model.predict(X)
        y = np.asarray(y, dtype=float).ravel()
        y[~ok] = np.nan
        return y
    return predict

if __name__ == "__main__":
    p = load_predictor()
    smi = sys.argv[1:] or ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    for s, v in zip(smi, p(smi)):
        print(f"{s}\t{v:.4f}")
