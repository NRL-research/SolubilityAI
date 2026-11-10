"""
03_build_corrected_splits.py
============================
Build the corrected data foundation: SC-leakage-filtered AqSolDB (from
01_sc_leakage_filter.py) split with the corrected scaffold splitter
(scripts/common_split.py), plus the 5-fold scaffold-grouped CV folds over
train+val used for every model-selection decision downstream.

Dataset version name: v2_sc_filtered_scaffold_fixed

Outputs (data/v2_corrected/):
  train.csv val.csv test.csv           64/16/20 scaffold-grouped, seed=42
  cv_folds.json                        5-fold scaffold-grouped over train+val, seed=42
  03_counts.json                       counts + verification at every step
  random_split/{train,val,test}.csv    matched random split, for the inflated-baseline contrast

Seeds: split seed 42 (val split uses 43 internally), CV fold seed 42. Fixed, recorded.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common_split import (  # noqa: E402
    prepare, scaffold_train_val_test_split, scaffold_group_kfold,
    assert_disjoint, assert_scaffold_disjoint,
)

SRC = "/home/claude/work/data/v1_sc_filtered/aqsoldb_sc_filtered.csv"
OUT = "/home/claude/work/data/v2_corrected"
SEED = 42
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/random_split", exist_ok=True)

df = pd.read_csv(SRC)
print(f"Source: {SRC}  ({len(df)} molecules)")

df = prepare(df)
n_acyclic = int(df["split_key"].str.startswith("__acyclic__").sum())
print(f"Acyclic molecules: {n_acyclic} ({100*n_acyclic/len(df):.2f}%)")

train, val, test = scaffold_train_val_test_split(df, 0.20, 0.20, seed=SEED)
assert_disjoint(train, val, test)
assert_scaffold_disjoint(train, val, test)

counts = {
    "source_file": SRC,
    "dataset_version": "v2_sc_filtered_scaffold_fixed",
    "split_seed": SEED,
    "n_total": len(df),
    "n_acyclic": n_acyclic,
    "pct_acyclic": round(100 * n_acyclic / len(df), 2),
    "n_train": len(train), "n_val": len(val), "n_test": len(test),
    "acyclic_train": int(train["split_key"].str.startswith("__acyclic__").sum()),
    "acyclic_val": int(val["split_key"].str.startswith("__acyclic__").sum()),
    "acyclic_test": int(test["split_key"].str.startswith("__acyclic__").sum()),
    "n_distinct_split_keys": int(df["split_key"].nunique()),
}
counts["pct_of_test_acyclic"] = round(100 * counts["acyclic_test"] / len(test), 2)
print(json.dumps({k: counts[k] for k in
                  ["n_train", "n_val", "n_test", "acyclic_train", "acyclic_val",
                   "acyclic_test", "pct_of_test_acyclic"]}, indent=2))

train.to_csv(f"{OUT}/train.csv", index=False)
val.to_csv(f"{OUT}/val.csv", index=False)
test.to_csv(f"{OUT}/test.csv", index=False)

# ------------------------------------------------------------------ CV folds over train+val
trainval = pd.concat([train, val], ignore_index=True)
trainval.to_csv(f"{OUT}/trainval.csv", index=False)
folds = scaffold_group_kfold(trainval, n_splits=5, seed=SEED)

# verify the folds are a true partition, scaffold-respecting, and test-free
all_pos = set(range(len(trainval)))
seen = set()
fold_report = []
for i, (tr, va) in enumerate(folds):
    assert not (set(tr) & set(va)), f"fold {i}: train/val index overlap"
    assert set(tr) | set(va) == all_pos, f"fold {i}: not a full partition"
    assert not (seen & set(va)), f"fold {i}: validation index reused across folds"
    seen |= set(va)
    k_tr = set(trainval.iloc[tr]["split_key"])
    k_va = set(trainval.iloc[va]["split_key"])
    assert not (k_tr & k_va), f"fold {i}: scaffold group spans train and val"
    fold_report.append({"fold": i, "n_train": len(tr), "n_val": len(va),
                        "n_val_scaffold_groups": len(k_va),
                        "pct_val_acyclic": round(
                            100 * trainval.iloc[va]["split_key"].str.startswith("__acyclic__").mean(), 2)})
assert seen == all_pos, "validation folds do not cover every molecule"
counts["cv_folds"] = fold_report
counts["cv_seed"] = SEED
print("\nCV folds:")
for r in fold_report:
    print("  ", r)

# test must not appear in any fold
test_smiles = set(test["canonical_smiles"])
assert not (set(trainval["canonical_smiles"]) & test_smiles), "test molecules present in train+val"
counts["verification_test_absent_from_cv_pool"] = True

with open(f"{OUT}/cv_folds.json", "w") as f:
    json.dump({"seed": SEED, "n_splits": 5,
               "folds": [{"train": tr, "val": va} for tr, va in folds]}, f)

# ------------------------------------------------------------------ matched random split
import random as _random  # noqa: E402
rng = _random.Random(SEED)
idx = list(range(len(df)))
rng.shuffle(idx)
n_test = int(round(len(df) * 0.20))
r_test = df.iloc[sorted(idx[:n_test])].reset_index(drop=True)
rest = idx[n_test:]
n_val = int(round(len(rest) * 0.20))
r_val = df.iloc[sorted(rest[:n_val])].reset_index(drop=True)
r_train = df.iloc[sorted(rest[n_val:])].reset_index(drop=True)
assert_disjoint(r_train, r_val, r_test)
r_train.to_csv(f"{OUT}/random_split/train.csv", index=False)
r_val.to_csv(f"{OUT}/random_split/val.csv", index=False)
r_test.to_csv(f"{OUT}/random_split/test.csv", index=False)
counts["random_split"] = {"n_train": len(r_train), "n_val": len(r_val), "n_test": len(r_test)}

with open(f"{OUT}/03_counts.json", "w") as f:
    json.dump(counts, f, indent=2)
print(f"\nSaved to {OUT}/")
