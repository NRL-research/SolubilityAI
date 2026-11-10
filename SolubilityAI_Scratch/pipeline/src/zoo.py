"""Model zoo: construction, fitting, prediction and NATIVE saving for every tabular family.

Saving conventions (library-native formats):
  sklearn estimators -> joblib (sklearn's documented persistence route)
  LightGBM           -> Booster.save_model (text .txt)
  XGBoost            -> Booster.save_model (.json)
  CatBoost           -> CatBoostRegressor.save_model (.cbm)
  preprocessors      -> separate sklearn Pipeline in preprocessor.joblib
"""
import json
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

N_JOBS = 2

def make_preprocessor(family):
    if family in ("ridge", "svr", "mlp", "linear"):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    if family in ("rf", "et"):
        return Pipeline([("impute", SimpleImputer(strategy="median"))])
    return None  # GBMs handle NaN natively; kNN on binary FPs needs none

def make_model(family, p, seed):
    p = dict(p)
    if family == "linear": return LinearRegression()
    if family == "ridge": return Ridge(alpha=p.get("alpha", 1.0))
    if family == "svr": return SVR(C=p["C"], gamma=p["gamma"], epsilon=p.get("epsilon", 0.1))
    if family == "rf": return RandomForestRegressor(n_estimators=p.get("n_estimators", 500),
                               max_features=p.get("max_features", 0.3), min_samples_leaf=p.get("min_samples_leaf", 1),
                               n_jobs=N_JOBS, random_state=seed)
    if family == "et": return ExtraTreesRegressor(n_estimators=p.get("n_estimators", 500),
                               max_features=p.get("max_features", 0.3), min_samples_leaf=p.get("min_samples_leaf", 1),
                               n_jobs=N_JOBS, random_state=seed)
    if family == "knn": return KNeighborsRegressor(n_neighbors=p["k"], metric="jaccard", weights="distance", n_jobs=N_JOBS)
    if family == "mlp": return MLPRegressor(hidden_layer_sizes=tuple(p["hidden"]), alpha=p["alpha"],
                               learning_rate_init=p.get("lr", 1e-3), max_iter=p.get("max_iter", 300),
                               early_stopping=False, random_state=seed, batch_size=128)
    if family == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**p, random_state=seed, n_jobs=N_JOBS, verbose=-1, deterministic=True, force_row_wise=True)
    if family == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(**p, random_state=seed, n_jobs=N_JOBS, tree_method="hist")
    if family == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, random_seed=seed, thread_count=N_JOBS, verbose=0)
    raise ValueError(family)

class Fitted:
    def __init__(self, family, pre, model):
        self.family, self.pre, self.model = family, pre, model
    def transform(self, X):
        X = X.astype(bool) if self.family == "knn" else X
        return self.pre.transform(X) if self.pre is not None else X
    def predict(self, X):
        return np.asarray(self.model.predict(self.transform(X)), dtype=float).ravel()

def fit(family, params, seed, X, y):
    pre = make_preprocessor(family)
    Xt = X.astype(bool) if family == "knn" else X
    if pre is not None:
        Xt = pre.fit_transform(Xt)
    m = make_model(family, params, seed)
    m.fit(Xt, y)
    return Fitted(family, pre, m)

def save(f: Fitted, folder):
    """Native save.  Returns list of written file names."""
    files = []
    if f.pre is not None:
        joblib.dump(f.pre, folder / "preprocessor.joblib"); files.append("preprocessor.joblib")
    if f.family == "lgbm":
        f.model.booster_.save_model(str(folder / "model.lgbm.txt")); files.append("model.lgbm.txt")
    elif f.family == "xgb":
        f.model.get_booster().save_model(str(folder / "model.xgb.json")); files.append("model.xgb.json")
    elif f.family == "catboost":
        f.model.save_model(str(folder / "model.cbm")); files.append("model.cbm")
    else:
        joblib.dump(f.model, folder / "model.joblib", compress=3); files.append("model.joblib")
    return files
