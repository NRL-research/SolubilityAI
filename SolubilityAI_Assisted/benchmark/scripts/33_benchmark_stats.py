"""
33_benchmark_stats.py
=====================
Per-model, per-benchmark-set statistics with bootstrap confidence intervals.

Metrics: n, RMSE, MAE, R2, mean signed error (bias), Pearson r, Spearman rho,
fraction of compounds predicted within 0.5 and within 1.0 LogS units.

Bootstrap: 10,000 resamples of compounds with replacement, seed 42, percentile 95%
interval on RMSE and R2. R2 is unstable on sets this small -- on SC1 (n=22) and
SC2-loose (n=32) the interval is wide enough that the point estimate should not be
read as precise, and a resample that happens to narrow the spread of the experimental
values can drive R2 sharply negative. Both facts are reported rather than smoothed.

Measurement-noise context (SC2 only, where per-compound SD is supplied):
if a model's error and the measurement error are independent, the observed mean
squared error decomposes as  MSE_obs = MSE_model + mean(SD^2).  The noise floor
sqrt(mean(SD^2)) is therefore the RMSE a perfect model would still show, and
sqrt(max(0, MSE_obs - mean(SD^2))) estimates the model's own error with measurement
noise removed. Reported for SC2-tight and SC2-loose, where it matters most: the loose
set's median SD is 0.585 against the tight set's 0.170.

Output: results/benchmark/33_stats.csv / .json
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

RES = "/home/claude/work/results/benchmark"
BDATA = "/home/claude/work/data/benchmark"
N_BOOT = 10000
SEED = 42

d = np.load(f"{RES}/32_predictions.npz")
info = json.load(open(f"{RES}/32_models_evaluated.json"))
SETS = list(info["sets"])
internal = {e["model"]: e for e in info["evaluated"]}

sd_by_set = {}
for s in SETS:
    df = pd.read_csv(f"{BDATA}/{s}.csv")
    sd_by_set[s] = pd.to_numeric(df["sd"], errors="coerce").values if "sd" in df else None


def metrics(y, p):
    e = p - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(e)),
        "pearson_r": float(sps.pearsonr(y, p)[0]),
        "spearman_rho": float(sps.spearmanr(y, p)[0]),
        "within_0.5": float(np.mean(np.abs(e) <= 0.5)),
        "within_1.0": float(np.mean(np.abs(e) <= 1.0)),
    }


def boot(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    rm, r2 = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        e = pp - yy
        rm[i] = np.sqrt(np.mean(e ** 2))
        sst = np.sum((yy - yy.mean()) ** 2)
        r2[i] = 1 - np.sum(e ** 2) / sst if sst > 0 else np.nan
    return {
        "rmse_ci_low": float(np.percentile(rm, 2.5)),
        "rmse_ci_high": float(np.percentile(rm, 97.5)),
        "r2_ci_low": float(np.nanpercentile(r2, 2.5)),
        "r2_ci_high": float(np.nanpercentile(r2, 97.5)),
        "n_boot": n_boot, "boot_seed": seed,
    }


rows = []
models = sorted({k.split("__")[0] for k in d.files if not k.startswith("__y__")})
for m in models:
    for s in SETS:
        key = f"{m}__{s}"
        if key not in d.files:
            continue
        y = d[f"__y__{s}"].astype(float)
        p = d[key].astype(float)
        r = {"model": m, "set": s,
             "kind": internal.get(m, {}).get("kind", "?"),
             "headline": m in info["headline_models"]}
        r.update(metrics(y, p))
        r.update(boot(y, p))
        sd = sd_by_set[s]
        if sd is not None and np.isfinite(sd).all():
            mse_noise = float(np.mean(sd ** 2))
            r["measurement_noise_floor_rmse"] = float(np.sqrt(mse_noise))
            r["rmse_noise_corrected"] = float(
                np.sqrt(max(0.0, r["rmse"] ** 2 - mse_noise)))
            r["frac_mse_from_measurement_noise"] = float(
                min(1.0, mse_noise / r["rmse"] ** 2))
        else:
            r["measurement_noise_floor_rmse"] = None
            r["rmse_noise_corrected"] = None
            r["frac_mse_from_measurement_noise"] = None
        r["internal_test_rmse"] = internal.get(m, {}).get("internal_test_rmse")
        r["internal_test_r2"] = internal.get(m, {}).get("internal_test_r2")
        r["delta_rmse_vs_internal_test"] = (
            r["rmse"] - r["internal_test_rmse"] if r["internal_test_rmse"] else None)
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/33_stats.csv", index=False)
json.dump(rows, open(f"{RES}/33_stats.json", "w"), indent=2)

pd.set_option("display.width", 250)
for s in SETS:
    sub = df[df["set"] == s].sort_values("rmse")
    print(f"\n{'='*100}\n{s}  (n={sub['n'].iloc[0]})")
    show = sub[["model", "headline", "rmse", "rmse_ci_low", "rmse_ci_high", "mae",
                "r2", "r2_ci_low", "r2_ci_high", "bias", "pearson_r", "spearman_rho",
                "within_0.5", "within_1.0"]].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(show.to_string(index=False))
    nf = sub["measurement_noise_floor_rmse"].iloc[0]
    if nf:
        print(f"  measurement-noise floor RMSE for this set: {nf:.3f}")

print(f"\nSaved {RES}/33_stats.csv")
