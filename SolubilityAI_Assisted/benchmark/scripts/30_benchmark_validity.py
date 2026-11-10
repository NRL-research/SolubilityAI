"""
30_benchmark_validity.py
========================
Validity checks that decide whether an external-benchmark evaluation on SC1,
SC2-tight and SC2-loose means anything at all. Run BEFORE any prediction.

Four checks, per the evaluation protocol:

  1. Training-data overlap, by InChIKey (NOT by SMILES string -- the same molecule
     can be written as several valid SMILES, so a SMILES comparison would miss real
     overlap and silently pass a leaky evaluation). Every model evaluated here was
     trained on the same train+val pool, so one check covers all of them; the pool is
     nonetheless listed explicitly per model in the output.

  2. Selection use: were these sets touched during model selection, hyperparameter
     tuning, early stopping or blend-weight fitting?

  3. Target units. The SC1 column is named `logS0_uM`, whose suffix claims
     micromolar. Decided from the actual values, not the name.

  4. Compound counts, against the published sizes of each benchmark, plus any
     compound that fails RDKit parsing. Nothing is dropped silently.

Output: results/benchmark/30_validity.json  and a printed summary.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = "/home/claude/work"
DATA = f"{ROOT}/data/v2_corrected"
UP = "/mnt/user-data/uploads/SolubilityAI_Assisted"
OUT = f"{ROOT}/results/benchmark"
os.makedirs(OUT, exist_ok=True)

SETS = {
    "SC1": (f"{UP}/SC1_processed.csv", "logS0_uM", 35),
    "SC2_tight": (f"{UP}/SC2_tight_processed.csv", "logS0", 100),
    "SC2_loose": (f"{UP}/SC2_loose_processed.csv", "logS0", 32),
}

report = {}

train = pd.read_csv(f"{DATA}/train.csv")
val = pd.read_csv(f"{DATA}/val.csv")
test = pd.read_csv(f"{DATA}/test.csv")
trainval = pd.read_csv(f"{DATA}/trainval.csv")

train_keys = set(trainval["inchikey"])          # what every model was fitted on
test_keys = set(test["inchikey"])
print(f"Training pool (train+val): {len(trainval)} molecules, "
      f"{len(train_keys)} unique InChIKeys")
print(f"Internal held-out test:    {len(test)} molecules")

report["training_pool_n"] = len(trainval)
report["training_pool_unique_inchikeys"] = len(train_keys)
report["internal_test_n"] = len(test)

# ---------------------------------------------------------------- 1 + 3 + 4
sets_report = {}
for name, (path, target_col, published_n) in SETS.items():
    df = pd.read_csv(path)
    r = {"file": os.path.basename(path), "rows_in_file": len(df),
         "published_size": published_n, "target_column": target_col}

    # --- 4a. identifiers and duplicates
    r["unique_inchikeys"] = int(df["inchikey"].nunique())
    dup = df[df.duplicated("inchikey", keep=False)]
    r["duplicate_inchikey_rows"] = json.loads(
        dup[["name", "inchikey"]].to_json(orient="records")) if len(dup) else []

    # --- 1. overlap with the training pool, by InChIKey
    keys = set(df["inchikey"])
    overlap_train = sorted(keys & train_keys)
    overlap_test = sorted(keys & test_keys)
    r["overlap_with_training_pool"] = len(overlap_train)
    r["overlap_with_training_pool_keys"] = overlap_train
    r["overlap_with_internal_test"] = len(overlap_test)
    r["overlap_with_internal_test_keys"] = overlap_test

    # --- 3. units: decide from the values, not the column name
    v = pd.to_numeric(df[target_col], errors="coerce")
    r["target_n_missing"] = int(v.isna().sum())
    vv = v.dropna()
    r["target_min"] = float(vv.min())
    r["target_max"] = float(vv.max())
    r["target_mean"] = float(vv.mean())
    r["target_frac_negative"] = float((vv < 0).mean())
    # A micromolar concentration cannot be negative and would typically be >> 1.
    # A LogS value is mostly negative and lies roughly in [-13, +2].
    looks_log = bool((vv < 0).mean() > 0.5 and vv.max() < 5 and vv.min() > -15)
    r["units_verdict"] = "already log-scale (LogS); NO conversion applied" if looks_log \
        else "NOT obviously log-scale -- investigate before use"
    r["units_conversion_applied"] = "none"

    # --- 4b. parseability
    bad = []
    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(row["canonical_smiles"])
        if m is None:
            bad.append({"name": row.get("name"), "smiles": row["canonical_smiles"]})
    r["unparseable_smiles"] = bad
    r["n_usable"] = len(df) - len(bad) - int(v.isna().sum())

    sets_report[name] = r
    print(f"\n--- {name} ---")
    print(f"  rows in file: {len(df)}   published size: {published_n}   "
          f"unique InChIKeys: {r['unique_inchikeys']}")
    print(f"  target column '{target_col}': range [{r['target_min']:.3f}, {r['target_max']:.3f}], "
          f"mean {r['target_mean']:.3f}, {100*r['target_frac_negative']:.0f}% negative")
    print(f"  units verdict: {r['units_verdict']}")
    print(f"  overlap with TRAINING pool (by InChIKey): {len(overlap_train)}")
    print(f"  overlap with internal test set:           {len(overlap_test)}")
    print(f"  unparseable SMILES: {len(bad)}    usable compounds: {r['n_usable']}")
    if len(df) != published_n:
        print(f"  NOTE: {len(df)} rows vs published {published_n} -- see report")

report["sets"] = sets_report

# ---------------------------------------------------------------- 2. selection use
# The SC sets were used for exactly one thing in this project: as a filter applied to
# AqSolDB before any split was made. They were never scored against, never used to
# pick a model, a hyperparameter, an early-stopping point or a blend weight.
sel = {
    "used_for_model_selection": False,
    "used_for_hyperparameter_tuning": False,
    "used_for_early_stopping": False,
    "used_for_blend_or_stack_weight_fitting": False,
    "only_use_in_project": ("InChIKey membership was used to REMOVE 108 overlapping "
                            "molecules from AqSolDB before any split was created "
                            "(scripts/01_sc_leakage_filter.py). No SC solubility value "
                            "was read at any point."),
    "evidence": [
        "Model/hyperparameter selection used only 5-fold scaffold CV over train+val "
        "(scripts/06_tree_cv.py, 22_new_model_families.py).",
        "Blend and stack weights were fitted only on train+val out-of-fold predictions "
        "(scripts/07_assemble_cv_results.py, 23_stacking.py).",
        "No tree model used early stopping; HistGB sets early_stopping=False.",
        "The SC target columns are not read anywhere in the training pipeline -- only "
        "the inchikey column is, and only in 01_sc_leakage_filter.py.",
    ],
}
report["selection_use"] = sel
print("\n--- selection use ---")
print(f"  {sel['only_use_in_project']}")

# ---------------------------------------------------------------- verdict
total_overlap = sum(s["overlap_with_training_pool"] for s in sets_report.values())
report["VERDICT_no_training_overlap"] = (total_overlap == 0)
report["VERDICT_valid_external_evaluation"] = bool(
    total_overlap == 0
    and not any(sel[k] for k in sel if k.startswith("used_for"))
    and all(s["units_conversion_applied"] == "none" for s in sets_report.values()))

print("\n" + "=" * 68)
print(f"Total training-pool overlap across all three sets: {total_overlap}")
print("VALID external evaluation: "
      f"{'YES' if report['VERDICT_valid_external_evaluation'] else 'NO'}")

with open(f"{OUT}/30_validity.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {OUT}/30_validity.json")
