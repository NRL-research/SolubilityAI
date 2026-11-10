"""
models.py
=========
Model builders. Two groups:

BASELINE_CONFIGS -- the exact models this project reported on, so the re-established
baseline differs from prior work ONLY in the data foundation:
  catboost_solo   CatBoost(iterations=3000, lr=0.02, depth=8, l2_leaf_reg=5)
                  on Morgan2048 + 5 descriptors      [notebook cell 83]
  xgboost_branch  XGBRegressor(n_estimators=300, max_depth=6, lr=0.05,
                  subsample=0.8, colsample_bytree=0.8)  [notebook cell 74]
  (the GATv2 branch lives in gnn.py)

Everything else is new search space for the improvement round.
"""

import numpy as np


def build(model_type, seed=42, **kw):
    if model_type == "catboost":
        from catboost import CatBoostRegressor
        p = dict(iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=5,
                 random_state=seed, verbose=0, allow_writing_files=False, thread_count=-1)
        p.update(kw)
        return CatBoostRegressor(**p)
    if model_type == "xgboost":
        import xgboost as xgb
        p = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                 objective="reg:squarederror", tree_method="hist")
        p.update(kw)
        return xgb.XGBRegressor(**p)
    if model_type == "lightgbm":
        import lightgbm as lgb
        p = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=-1,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                 reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
                 random_state=seed, n_jobs=-1, verbose=-1)
        p.update(kw)
        return lgb.LGBMRegressor(**p)
    if model_type == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        p = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
                 l2_regularization=1.0, random_state=seed, early_stopping=False)
        p.update(kw)
        return HistGradientBoostingRegressor(**p)
    if model_type == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        p = dict(n_estimators=500, max_features=0.3, min_samples_leaf=1,
                 random_state=seed, n_jobs=-1)
        p.update(kw)
        return ExtraTreesRegressor(**p)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        p = dict(n_estimators=400, max_features=0.3, min_samples_leaf=1,
                 random_state=seed, n_jobs=-1)
        p.update(kw)
        return RandomForestRegressor(**p)
    raise ValueError(model_type)


BASELINE_CONFIGS = {
    "catboost_solo[solubility_ai]": ("catboost", "solubility_ai", {}),
    "xgboost_branch[solubility_ai]": ("xgboost", "solubility_ai", {}),
}


def optimize_blend_weights(pred_matrix, y, seed=42):
    """Non-negative weights summing to 1 minimising RMSE (SLSQP, multi-start).

    pred_matrix: (n_samples, n_models). Fitted ONLY on out-of-fold train+val
    predictions; the resulting weights are then applied unchanged to test.
    """
    from scipy.optimize import minimize
    P = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    k = P.shape[1]

    def obj(w):
        return float(np.sqrt(np.mean((y - P @ w) ** 2)))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    best, best_v = None, np.inf
    rng = np.random.default_rng(seed)
    starts = [np.full(k, 1.0 / k)] + [rng.dirichlet(np.ones(k)) for _ in range(9)]
    for w0 in starts:
        r = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                     options={"maxiter": 500, "ftol": 1e-10})
        if r.success and r.fun < best_v:
            best, best_v = r.x, r.fun
    if best is None:
        best = np.full(k, 1.0 / k)
    best = np.clip(best, 0, None)
    return best / best.sum()
