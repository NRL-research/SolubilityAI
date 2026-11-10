"""
Ensemble blending and stacking.

Replaces the notebook's approach of Optuna-searching blend weights directly
against test-set R² (see notebook cells 31/42/54). Blend weights here are
fit by minimizing RMSE on a validation set via constrained optimization
(non-negative, sums to 1) -- the test set is never touched.

Also finishes the notebook's unfinished "Reloader" stacking cell: a real
meta-learner trained on base-model out-of-fold predictions.
"""

import numpy as np
from scipy.optimize import minimize


def weighted_blend(preds_dict, weights_dict):
    names = list(preds_dict.keys())
    total = sum(weights_dict[n] for n in names)
    blend = np.zeros_like(next(iter(preds_dict.values())), dtype=np.float64)
    for n in names:
        blend += (weights_dict[n] / total) * preds_dict[n]
    return blend


def optimize_blend_weights(preds_dict, y_true):
    """Fits non-negative weights (summing to 1) minimizing RMSE on
    (preds_dict, y_true) -- call this with VALIDATION predictions/targets
    only. Returns {name: weight}."""
    names = list(preds_dict.keys())
    P = np.column_stack([preds_dict[n] for n in names])  # (n_samples, n_models)
    y = np.asarray(y_true, dtype=np.float64)

    def objective(w):
        blend = P @ w
        return np.sqrt(np.mean((blend - y) ** 2))

    n = len(names)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else w0
    return {name: float(w) for name, w in zip(names, weights)}


class StackingMetaLearner:
    """Trains a small meta-model on base-model predictions as features.

    Fit on validation-set base predictions (never test), so the base models
    and the meta-learner never see the same data twice: base models are
    trained on `train`, meta-learner is trained on `val` base predictions.
    """

    def __init__(self, base_model_type="ridge", alpha=1.0):
        self.base_model_type = base_model_type
        self.alpha = alpha
        self.model = None
        self.model_names = None

    def fit(self, preds_dict, y_true):
        self.model_names = list(preds_dict.keys())
        X = np.column_stack([preds_dict[n] for n in self.model_names])
        y = np.asarray(y_true, dtype=np.float64)

        if self.base_model_type == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=self.alpha, positive=True)
        elif self.base_model_type == "catboost":
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, verbose=0)
        else:
            raise ValueError(f"Unknown base_model_type: {self.base_model_type}")

        self.model.fit(X, y)
        return self

    def predict(self, preds_dict):
        X = np.column_stack([preds_dict[n] for n in self.model_names])
        return self.model.predict(X)
