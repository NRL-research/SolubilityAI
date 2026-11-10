"""
predict.py -- catboost[rdkit_full_eng]
Predict aqueous solubility (LogS, log10 mol/L) for one or more SMILES strings.

    python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
    from predict import predict; predict(["CCO"])

Loads all 3 saved seeds and averages their predictions -- that seed-averaged
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

FEATURE_SET = "rdkit_full_eng"
MODEL_TYPE = "catboost"
FEATURE_NAMES = json.load(open(os.path.join(HERE, "feature_names.json")))


def load_models():
    paths = sorted(glob.glob(os.path.join(HERE, "model_seed*.cbm")))
    if not paths:
        raise FileNotFoundError("no model_seed*.cbm files next to predict.py")
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
        print(f"{s}\t{p:.4f}")
