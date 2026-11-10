"""
02_audit_scaffold_split.py
==========================
Audit: is the acyclic-molecule scaffold-splitting bug present in THIS project's own
scaffold-splitting code, and what did it do to this project's own reported results?

This project's own splitter is solubility_ai_v2.ipynb cell 37 (`scaffold_split`),
reused verbatim in cell 103 (DISSOLVR calibration split) and in cell 72
(`scaffold_split_for_validation`). It is reproduced here EXACTLY as written in the
notebook, and run against the same data the notebook ran it on (the SC-filtered
AqSolDB produced by 01_sc_leakage_filter.py), with the notebook's own SEED=42 and
TEST_FRACTION=0.20.

The corrected splitter (acyclic molecules get individual keys) is run on the same
data for comparison.

Deterministic given seed. Output: data/audit/02_scaffold_split_audit.json
"""

import json
import os
import random
from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SEED = 42
TEST_FRACTION = 0.20
DATA = "/home/claude/work/data/v1_sc_filtered/aqsoldb_sc_filtered.csv"
OUTDIR = "/home/claude/work/data/audit"
os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------------ notebook code, verbatim
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split_BUGGY(df, test_fraction=TEST_FRACTION, seed=SEED):
    """solubility_ai_v2.ipynb cell 37, verbatim."""
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    scaffold_groups = defaultdict(list)
    for idx, scaffold in enumerate(df["scaffold"]):
        scaffold_groups[scaffold].append(idx)

    scaffold_keys = list(scaffold_groups.keys())
    random.seed(seed)
    random.shuffle(scaffold_keys)

    n_test_target = int(len(df) * test_fraction)
    test_indices = []
    for key in scaffold_keys:
        if len(test_indices) >= n_test_target:
            break
        test_indices.extend(scaffold_groups[key])

    test_indices = set(test_indices)
    train_indices = [i for i in range(len(df)) if i not in test_indices]
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[sorted(test_indices)].reset_index(drop=True)


# ------------------------------------------------------------------ corrected version
def split_key(scaffold, canonical_smiles):
    if pd.isna(scaffold) or not scaffold:
        return f"__acyclic__::{canonical_smiles}"
    return scaffold


def scaffold_split_FIXED(df, test_fraction=TEST_FRACTION, seed=SEED):
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    df["split_key"] = [split_key(s, c) for s, c in zip(df["scaffold"], df["canonical_smiles"])]

    groups = defaultdict(list)
    for idx, key in enumerate(df["split_key"]):
        groups[key].append(idx)

    keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(keys)

    n_test_target = int(len(df) * test_fraction)
    test_indices = []
    for key in keys:
        if len(test_indices) >= n_test_target:
            break
        test_indices.extend(groups[key])

    test_indices = set(test_indices)
    train_indices = [i for i in range(len(df)) if i not in test_indices]
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[sorted(test_indices)].reset_index(drop=True)


# ------------------------------------------------------------------ run
df = pd.read_csv(DATA)
print(f"Dataset: {len(df)} molecules ({DATA})")

df_s = df.copy()
df_s["scaffold"] = df_s["canonical_smiles"].apply(get_scaffold)
n_acyclic = int((df_s["scaffold"] == "").sum())
audit = {
    "dataset": DATA,
    "n_molecules": len(df),
    "n_acyclic_empty_scaffold": n_acyclic,
    "pct_acyclic": round(100 * n_acyclic / len(df), 2),
    "n_distinct_scaffold_strings": int(df_s["scaffold"].nunique()),
    "seed": SEED,
    "test_fraction": TEST_FRACTION,
}
print(f"Acyclic (empty Murcko scaffold): {n_acyclic} ({audit['pct_acyclic']}%)")

for label, fn in [("buggy_notebook_cell37", scaffold_split_BUGGY), ("fixed", scaffold_split_FIXED)]:
    tr, te = fn(df)
    tr_ac = int((tr["scaffold"] == "").sum())
    te_ac = int((te["scaffold"] == "").sum())
    audit[label] = {
        "n_train": len(tr),
        "n_test": len(te),
        "test_fraction_actual": round(len(te) / (len(tr) + len(te)), 4),
        "acyclic_in_train": tr_ac,
        "acyclic_in_test": te_ac,
        "pct_of_test_that_is_acyclic": round(100 * te_ac / len(te), 2) if len(te) else 0.0,
        "pct_of_all_acyclic_in_test": round(100 * te_ac / n_acyclic, 2) if n_acyclic else 0.0,
        "test_logs_mean": round(float(te["Solubility"].mean()), 4),
        "test_logs_std": round(float(te["Solubility"].std()), 4),
        "train_logs_mean": round(float(tr["Solubility"].mean()), 4),
    }
    print(f"\n--- {label} ---")
    for k, v in audit[label].items():
        print(f"  {k}: {v}")

# largest scaffold groups, for context on how coarse the grouping is
sizes = df_s["scaffold"].value_counts().head(6)
audit["largest_scaffold_groups"] = {
    ("<EMPTY / acyclic>" if k == "" else k): int(v) for k, v in sizes.items()
}

with open(f"{OUTDIR}/02_scaffold_split_audit.json", "w") as f:
    json.dump(audit, f, indent=2)
print("\nLargest scaffold groups:", audit["largest_scaffold_groups"])
print(f"\nSaved {OUTDIR}/02_scaffold_split_audit.json")
