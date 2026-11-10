"""
35_verify_benchmark.py
======================
Independent re-check of the external benchmark evaluation, run against the saved
per-compound CSV files rather than against anything in memory. Every number quoted in
the report should survive being recomputed from those files alone.

Checks:
  1. Every (model, set) per-compound file exists, has the expected row count, no
     missing predictions, and InChIKeys matching the benchmark source file.
  2. Statistics in 33_stats.csv are reproducible from the per-compound files
     (RMSE, MAE, R2, bias recomputed independently).
  3. No benchmark InChIKey appears in the training pool -- re-checked here from the
     split files, not from the earlier validity script's output.
  4. Prediction vectors differ across models and across sets (no accidental copies).
  5. Every model named in the report as evaluated has weight files on disk, and every
     model named as not-evaluated has none.

Exit 0 only if all pass.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/claude/work"
RES = f"{ROOT}/results/benchmark"
PRED = f"{RES}/predictions"
BDATA = f"{ROOT}/data/benchmark"
SAVED = f"{ROOT}/saved_models"

info = json.load(open(f"{RES}/32_models_evaluated.json"))
stats = pd.read_csv(f"{RES}/33_stats.csv")
SETS = list(info["sets"])
models = sorted({e["model"] for e in info["evaluated"]})

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---- 1. files present and well formed
bad = []
src = {s: pd.read_csv(f"{BDATA}/{s}.csv") for s in SETS}
for m in models:
    for s in SETS:
        f = f"{PRED}/{m}__{s}.csv"
        if not os.path.exists(f):
            bad.append(f"{m}/{s}: missing file")
            continue
        d = pd.read_csv(f)
        if len(d) != len(src[s]):
            bad.append(f"{m}/{s}: {len(d)} rows vs {len(src[s])}")
        if d["logS_pred"].isna().any():
            bad.append(f"{m}/{s}: missing predictions")
        if list(d["inchikey"]) != list(src[s]["inchikey"]):
            bad.append(f"{m}/{s}: InChIKey order/content differs from source")
check("1. per-compound files complete and aligned to the source sets", not bad,
      f"{len(models)*len(SETS)} files; " + ("; ".join(bad[:4]) if bad else "no issues"))

# ---- 2. statistics reproducible from those files
bad = []
for _, r in stats.iterrows():
    d = pd.read_csv(f"{PRED}/{r['model']}__{r['set']}.csv")
    y, p = d["logS_exp"].values, d["logS_pred"].values
    e = p - y
    rec = {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        "n": len(y),
    }
    for k, v in rec.items():
        if abs(float(r[k]) - v) > 1e-9:
            bad.append(f"{r['model']}/{r['set']}: {k} {r[k]} vs {v}")
check("2. reported statistics recompute from the per-compound files", not bad,
      f"{len(stats)} rows x 5 metrics; " + ("; ".join(bad[:4]) if bad else "all match to 1e-9"))

# ---- 3. no training overlap (re-derived here)
trainval = pd.read_csv(f"{ROOT}/data/v2_corrected/trainval.csv")
tkeys = set(trainval["inchikey"])
ov = {s: len(set(src[s]["inchikey"]) & tkeys) for s in SETS}
check("3. no benchmark compound in the training pool (by InChIKey)",
      sum(ov.values()) == 0, str(ov))

# ---- 4. distinct predictions
bad = []
for s in SETS:
    vecs = {}
    for m in models:
        vecs[m] = pd.read_csv(f"{PRED}/{m}__{s}.csv")["logS_pred"].values
    ms = list(vecs)
    for i, a in enumerate(ms):
        if np.std(vecs[a]) < 1e-6:
            bad.append(f"{a}/{s}: constant")
        for b in ms[i + 1:]:
            if np.array_equal(vecs[a], vecs[b]):
                bad.append(f"{s}: {a} == {b}")
check("4. prediction vectors distinct across models and non-constant", not bad,
      "; ".join(bad[:4]) if bad else f"{len(models)} models x {len(SETS)} sets")

# ---- 5. evaluated <-> weights on disk
bad = []
for m in models:
    folder = f"{SAVED}/{m}"
    if os.path.isdir(folder):
        if not glob.glob(f"{folder}/model_seed*"):
            bad.append(f"{m}: evaluated but no weight files")
    elif not os.path.exists(f"{SAVED}/blends/{m}.json"):
        bad.append(f"{m}: evaluated but neither a model folder nor a blend config")
for m in info["not_saved_so_not_evaluated"]:
    slug = m.replace("[", "_").replace("]", "").replace("+", "_plus_")
    if glob.glob(f"{SAVED}/{slug}/model_seed*"):
        bad.append(f"{m}: listed as not saved but weight files exist")
check("5. evaluated models have weights; not-evaluated models have none", not bad,
      f"{len(models)} evaluated, {len(info['not_saved_so_not_evaluated'])} not evaluated; "
      + ("; ".join(bad[:4]) if bad else "consistent"))

print(f"\n{len(passes)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
