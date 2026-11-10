"""
Rebuild the scaffold split using the fixed grouping logic in common/splitting.py.

Source data: the union of misc/original_buggy_split/data_processed/train.csv
and test.csv -- the original (invalidated) split, kept there for provenance.
Per common/data/README.md, those already are the canonicalized AqSolDB data
(invalid SMILES dropped) -- no additional curation applied beyond that -- so
concatenating them reconstructs the full working dataset without depending on
the dead external download link in the README.

Output: common/data/processed_scaffold_fixed/{train,val,test}.csv and
common/data/processed_random_fixed/{train,val,test}.csv -- the split every
other component (dissolvr/, solubility_models/) trains and evaluates on.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.splitting import (
    scaffold_train_val_test_split,
    random_train_val_test_split,
    assert_no_leakage,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCE_DIR = os.path.join(ROOT, "misc", "original_buggy_split", "data_processed")
OUT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
OUT_DIR_RANDOM = os.path.join(ROOT, "common", "data", "processed_random_fixed")


def load_source_dataset():
    train = pd.read_csv(os.path.join(SOURCE_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(SOURCE_DIR, "test.csv"))
    full = pd.concat([train, test], ignore_index=True)
    # Use only the original raw-ish columns; re-derive canonical_smiles/scaffold fresh.
    full = full[["ID", "Name", "InChIKey", "SMILES", "Solubility"]]
    return full


def main():
    df = load_source_dataset()
    print(f"Loaded {len(df)} molecules from combined {SOURCE_DIR}/{{train,test}}.csv")

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df, test_df, stats = scaffold_train_val_test_split(df)
    assert_no_leakage(train_df, val_df, test_df)

    print("\n=== SCAFFOLD SPLIT (fixed) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print(f"Saved to {OUT_DIR}/{{train,val,test}}.csv")

    # Verify the acyclic-molecule fix directly.
    def acyclic_count(d):
        return (d["scaffold"] == "").sum()

    print("\n=== ACYCLIC MOLECULE DISTRIBUTION (the bug being fixed) ===")
    print(f"  train: {acyclic_count(train_df)} / {len(train_df)}")
    print(f"  val:   {acyclic_count(val_df)} / {len(val_df)}")
    print(f"  test:  {acyclic_count(test_df)} / {len(test_df)}")

    os.makedirs(OUT_DIR_RANDOM, exist_ok=True)
    rtrain_df, rval_df, rtest_df, rstats = random_train_val_test_split(df)
    assert_no_leakage(rtrain_df, rval_df, rtest_df)

    print("\n=== RANDOM SPLIT (for inflated-baseline comparison) ===")
    for k, v in rstats.items():
        print(f"  {k}: {v}")

    rtrain_df.to_csv(os.path.join(OUT_DIR_RANDOM, "train.csv"), index=False)
    rval_df.to_csv(os.path.join(OUT_DIR_RANDOM, "val.csv"), index=False)
    rtest_df.to_csv(os.path.join(OUT_DIR_RANDOM, "test.csv"), index=False)
    print(f"Saved to {OUT_DIR_RANDOM}/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
