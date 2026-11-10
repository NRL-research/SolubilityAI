"""RMSE / MAE / R² utilities, plus multi-seed aggregation."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate_seed_runs(metrics_per_seed):
    """metrics_per_seed: list of dicts (one per seed) -> {metric: (mean, std)}."""
    keys = metrics_per_seed[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_per_seed])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def format_metrics(metrics, prefix=""):
    return f"{prefix}RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}"


def format_aggregated(agg, prefix=""):
    return (
        f"{prefix}RMSE={agg['rmse']['mean']:.4f}±{agg['rmse']['std']:.4f}  "
        f"MAE={agg['mae']['mean']:.4f}±{agg['mae']['std']:.4f}  "
        f"R2={agg['r2']['mean']:.4f}±{agg['r2']['std']:.4f}"
    )
