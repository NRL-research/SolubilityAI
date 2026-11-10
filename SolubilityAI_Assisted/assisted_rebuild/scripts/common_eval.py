"""
common_eval.py
==============
Metrics and statistical comparison utilities.

Every model-vs-model accuracy claim in this project is backed by paired_bootstrap_rmse
(or its R2 twin), not by a point-estimate difference.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def paired_bootstrap(y_true, pred_a, pred_b, n_boot=10000, seed=42, metric="rmse"):
    """Bootstrap the metric gap (A - B) over test molecules, resampling the SAME
    molecule indices for both models each replicate so per-molecule difficulty is
    controlled for. Returns point estimate, 95% CI, and P(B better than A).

    For rmse, a positive gap means B is better (lower error). For r2, positive
    means A is better. Read `interpretation` in the returned dict rather than
    guessing the sign.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def score(yt, yp):
        if metric == "rmse":
            return float(np.sqrt(np.mean((yt - yp) ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(yt - yp)))
        if metric == "r2":
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            return float(1 - ss_res / ss_tot)
        raise ValueError(metric)

    point = score(y_true, pred_a) - score(y_true, pred_b)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps[i] = score(y_true[idx], pred_a[idx]) - score(y_true[idx], pred_b[idx])

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    better = float(np.mean(gaps > 0)) if metric in ("rmse", "mae") else float(np.mean(gaps < 0))
    return {
        "metric": metric,
        "gap_point_estimate_A_minus_B": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "frac_resamples_B_better": better,
        "n_boot": n_boot,
        "seed": seed,
        "interpretation": ("positive gap = B has lower error = B better"
                           if metric in ("rmse", "mae")
                           else "positive gap = A has higher R2 = A better"),
    }
