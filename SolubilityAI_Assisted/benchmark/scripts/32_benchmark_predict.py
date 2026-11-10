"""
32_benchmark_predict.py
=======================
Predict SC1, SC2-tight and SC2-loose with every model that exists in a reloadable
form, loading each one FRESH FROM DISK through its own saved_models/<model>/predict.py
-- the same path a third party would use. Nothing is taken from training memory and
nothing is retrained.

The three benchmark sets are predicted SEPARATELY and never pooled: they differ in
origin and in measurement quality (SC2-tight median measurement SD 0.17, SC2-loose
0.59), so a combined score would be meaningless.

Models NOT evaluated, because they were never saved in a reloadable form:
  stack_enet@all_models, stack_enet@all_usable, stack_ridge@all_usable  (round-3 best)
  and every round-3 single model (lightgbm_huber*, lightgbm_conservative*, svr_rbf,
  mlp, elasticnet, knn_tanimoto, catboost_huber, ...).
These are listed in the output as not_evaluated. They are NOT retrained here.

Sanity checks run before anything is written:
  * every model's predictions differ from every other model's on the same set
  * every model's predictions differ across the three sets
  * no prediction vector is constant
A failure on any of these is a bug, and the script stops.

Output: results/benchmark/predictions/<model>__<set>.csv   (per-compound)
        results/benchmark/32_predictions.npz
        results/benchmark/32_models_evaluated.json
"""

import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/claude/work"
SAVED = f"{ROOT}/saved_models"
BDATA = f"{ROOT}/data/benchmark"
RES = f"{ROOT}/results/benchmark"
PRED = f"{RES}/predictions"
os.makedirs(PRED, exist_ok=True)

SETS = ["SC1_consistent", "SC1_repaired", "SC2_tight", "SC2_loose"]

NOT_SAVED = {
    "stack_enet@all_models": "round-3 best; meta-model over 36 base models, never saved",
    "stack_enet@all_usable": "round-3 CV-selected stack over 33 base models, never saved",
    "stack_ridge@all_usable": "round-3 stack over 33 base models, never saved",
    "lightgbm_cons_huber_ca[mordred+chem_aware]": "round-3 best single model by CV, never saved",
    "lightgbm_huber[mordred]": "round-3 single model, never saved",
    "lightgbm_huber_d1p0[mordred]": "round-3 single model, never saved",
    "lightgbm_huber_d4p0[mordred]": "round-3 single model, never saved",
    "lightgbm_conservative[mordred]": "round-3 single model, never saved",
    "lightgbm_conservative_ca[mordred+chem_aware]": "round-3 single model, never saved",
    "lightgbm_cons_huber[mordred]": "round-3 single model, never saved",
    "lightgbm_very_conservative[mordred]": "round-3 single model, never saved",
    "lightgbm_mae[mordred]": "round-3 single model, never saved",
    "catboost_huber[rdkit_full_eng]": "round-3 single model, never saved",
    "svr_rbf[rdkit_full_eng]": "round-3 single model, never saved",
    "mlp[rdkit_full_eng]": "round-3 single model, never saved",
    "elasticnet[mordred]": "round-3 single model, never saved",
    "knn_tanimoto[solubility_ai]": "round-3 single model, never saved",
    "lightgbm[mordred+chem_aware]": "round-3 single model, never saved",
}

HEADLINE = ["lightgbm_mordred", "greedy_blend", "diverse_blend_4"]


def load_module(path, tag):
    for k in list(sys.modules):
        if k.startswith("_bm_") or k in ("solubility_featurizer", "gatv2_model"):
            del sys.modules[k]
    folder = os.path.dirname(path)
    sys.path.insert(0, folder)
    try:
        name = "_bm_" + tag
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(folder)


# ---------------------------------------------------------------- load benchmark sets
data = {}
for s in SETS:
    d = pd.read_csv(f"{BDATA}/{s}.csv")
    assert d["logS_exp"].notna().all(), f"{s}: missing targets"
    data[s] = d
    print(f"{s}: {len(d)} compounds")

# ---------------------------------------------------------------- single models
single_folders = sorted(
    d for d in glob.glob(f"{SAVED}/*")
    if os.path.isdir(d) and os.path.basename(d) != "blends"
    and os.path.exists(os.path.join(d, "predict.py")))

preds = {}            # (model, set) -> np.array
evaluated, skipped = [], []

print(f"\nLoading {len(single_folders)} saved single models from disk")
for folder in single_folders:
    tag = os.path.basename(folder)
    meta = json.load(open(f"{folder}/metadata.json"))
    weights = glob.glob(f"{folder}/model_seed*.{meta['file_extension']}")
    if len(weights) != len(meta["seeds"]):
        skipped.append({"model": tag, "reason":
                        f"expected {len(meta['seeds'])} weight files, found {len(weights)}"})
        print(f"  SKIP {tag}: weight files missing ({len(weights)}/{len(meta['seeds'])})")
        continue
    mod = load_module(f"{folder}/predict.py", tag)
    models = mod.load_models()
    t0 = time.time()
    for s in SETS:
        p = np.asarray(mod.predict(data[s]["canonical_smiles"].tolist(), models), dtype=float)
        assert p.shape == (len(data[s]),), f"{tag}/{s}: shape {p.shape}"
        preds[(tag, s)] = p
    evaluated.append({"model": tag, "config": meta["config"], "kind": "single",
                      "feature_set": meta["feature_set"], "n_seeds": len(meta["seeds"]),
                      "internal_test_rmse": meta["test_rmse"],
                      "internal_test_r2": meta["test_r2"],
                      "seconds": round(time.time() - t0, 1)})
    print(f"  {tag:32s} ok ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- blends
blend_cfgs = sorted(glob.glob(f"{SAVED}/blends/*.json"))
bmod = load_module(f"{SAVED}/blends/predict_blend.py", "blends")
print(f"\nLoading {len(blend_cfgs)} saved blends")
for f in blend_cfgs:
    name = os.path.basename(f)[:-5]
    cfg = json.load(open(f))
    missing = [m["folder"] for m in cfg["members"]
               if m["weight"] > 1e-9 and (m["folder"], SETS[0]) not in preds]
    if missing:
        skipped.append({"model": name, "reason": f"member models unavailable: {missing}"})
        print(f"  SKIP {name}: members unavailable {missing}")
        continue
    t0 = time.time()
    for s in SETS:
        # reuse the member predictions already computed, applying the saved weights --
        # identical arithmetic to predict_blend.py, without re-featurising per member
        total = np.zeros(len(data[s]))
        for m in cfg["members"]:
            if m["weight"] <= 1e-9:
                continue
            total += m["weight"] * preds[(m["folder"], s)]
        preds[(name, s)] = total
    evaluated.append({"model": name, "config": name, "kind": "blend",
                      "feature_set": f"{sum(1 for m in cfg['members'] if m['active'])} members",
                      "n_seeds": None,
                      "internal_test_rmse": cfg["test_rmse"],
                      "internal_test_r2": cfg["test_r2"],
                      "seconds": round(time.time() - t0, 1)})
    print(f"  {name:32s} ok")

# cross-check the blend arithmetic against the folder's own predict_blend on one set
check_name = "greedy_blend"
if (check_name, "SC2_tight") in preds:
    direct = np.asarray(bmod.predict_blend(
        check_name, data["SC2_tight"]["canonical_smiles"].tolist()), dtype=float)
    d = float(np.max(np.abs(direct - preds[(check_name, "SC2_tight")])))
    print(f"\nBlend arithmetic cross-check ({check_name}, SC2_tight): max|diff| = {d:.2e}")
    assert d < 1e-9, "blend reconstruction does not match predict_blend.py"

# ---------------------------------------------------------------- sanity checks
print("\nSanity checks")
models = sorted({m for (m, _) in preds})
problems = []

for m in models:
    for s in SETS:
        p = preds[(m, s)]
        if np.std(p) < 1e-6:
            problems.append(f"{m}/{s}: constant prediction vector")
        if not np.isfinite(p).all():
            problems.append(f"{m}/{s}: non-finite predictions")

for s in SETS:
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if np.array_equal(preds[(a, s)], preds[(b, s)]):
                problems.append(f"{s}: {a} and {b} produce identical predictions")

for m in models:
    for i, s1 in enumerate(SETS):
        for s2 in SETS[i + 1:]:
            if s1.startswith("SC1") and s2.startswith("SC1"):
                continue           # the two SC1 subsets legitimately share compounds
            n = min(len(preds[(m, s1)]), len(preds[(m, s2)]))
            if n and np.array_equal(preds[(m, s1)][:n], preds[(m, s2)][:n]):
                problems.append(f"{m}: identical predictions across {s1} and {s2}")

if problems:
    print("  FAILED:")
    for p in problems:
        print("   -", p)
    sys.exit(1)
print(f"  all pass: {len(models)} models x {len(SETS)} sets, predictions distinct and finite")

# ---------------------------------------------------------------- per-compound tables
for m in models:
    for s in SETS:
        d = data[s].copy()
        d["model"] = m
        d["benchmark_set"] = s
        d["logS_pred"] = preds[(m, s)]
        d["signed_error"] = d["logS_pred"] - d["logS_exp"]
        cols = ["model", "benchmark_set", "name", "inchikey", "canonical_smiles",
                "logS_exp", "logS_pred", "signed_error"]
        if "sd" in d.columns:
            cols.append("sd")
        if "n_measurements" in d.columns:
            cols.append("n_measurements")
        d[cols].to_csv(f"{PRED}/{m}__{s}.csv", index=False)

np.savez(f"{RES}/32_predictions.npz",
         **{f"{m}__{s}": preds[(m, s)] for (m, s) in preds},
         **{f"__y__{s}": data[s]["logS_exp"].values for s in SETS})

json.dump({"evaluated": evaluated, "skipped": skipped,
           "not_saved_so_not_evaluated": NOT_SAVED,
           "headline_models": HEADLINE,
           "sets": {s: int(len(data[s])) for s in SETS}},
          open(f"{RES}/32_models_evaluated.json", "w"), indent=2)

print(f"\n{len(evaluated)} models evaluated, {len(skipped)} skipped, "
      f"{len(NOT_SAVED)} not evaluated because never saved.")
print(f"Per-compound tables: {PRED}/  ({len(models)*len(SETS)} files)")
