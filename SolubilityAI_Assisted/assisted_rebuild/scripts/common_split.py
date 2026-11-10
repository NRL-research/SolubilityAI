"""
common_split.py
===============
Corrected scaffold splitting for the SolubilityAI project.

Two corrections vs. this project's own prior splitter (solubility_ai_v2.ipynb cell 37):

  1. Acyclic molecules (empty Murcko scaffold) get an individual per-molecule split
     key instead of all sharing one "" key. In the prior code all 2,940 acyclic
     molecules (29.8% of the SC-filtered dataset) formed a single group which, at
     seed=42, landed entirely in train -- zero acyclic molecules in any test set.
  2. NaN-safe: a scaffold written to CSV as "" reloads as NaN, and `if nan:` is True
     in Python, so a naive truthiness check silently reintroduces the bug on any
     reloaded dataframe. Checked with pd.isna().

Also changed vs. both prior implementations:
  3. The group-filling loop stops at whichever choice lands closer to the target test
     fraction, instead of unconditionally adding a whole group after the target is
     already exceeded. The prior loop's `if len(test) >= target: break` test happens
     BEFORE extending, so the final group added can overshoot by its entire size
     (the benzene group is 1,739 molecules -- an 88% overshoot of a 1,974-molecule
     test set is possible). This does not affect correctness of grouping, only the
     realised split proportions.

All randomness is via random.Random(seed) -- no global state is touched.
"""

from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def murcko_scaffold(canonical_smiles):
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def make_split_key(scaffold, canonical_smiles):
    """Real scaffold for ring-containing molecules; a unique singleton key for acyclic ones."""
    if scaffold is None or (isinstance(scaffold, float) and pd.isna(scaffold)) or scaffold == "":
        return f"__acyclic__::{canonical_smiles}"
    return scaffold


def prepare(df, smiles_col="canonical_smiles"):
    """Add scaffold + split_key columns. Assumes canonical_smiles already present."""
    df = df.copy()
    if "scaffold" not in df.columns:
        df["scaffold"] = df[smiles_col].apply(murcko_scaffold)
    df = df[df["scaffold"].notna() | df[smiles_col].notna()].reset_index(drop=True)
    df["split_key"] = [make_split_key(s, c) for s, c in zip(df["scaffold"], df[smiles_col])]
    return df


def _group_split(df, target_fraction, seed):
    """Assign whole split_key groups to the held-out side until the target fraction is
    best approximated (stop at whichever of before/after the overshooting group is closer)."""
    groups = defaultdict(list)
    for pos, key in enumerate(df["split_key"]):
        groups[key].append(pos)

    keys = list(groups.keys())
    import random as _random
    _random.Random(seed).shuffle(keys)

    n_target = int(round(len(df) * target_fraction))
    held = []
    for key in keys:
        if len(held) >= n_target:
            break
        g = groups[key]
        if len(held) + len(g) > n_target:
            # take the group only if doing so lands closer to the target than stopping
            if abs(len(held) + len(g) - n_target) < abs(len(held) - n_target):
                held.extend(g)
            break
        held.extend(g)

    held_set = set(held)
    rest = [i for i in range(len(df)) if i not in held_set]
    return (df.iloc[rest].reset_index(drop=True),
            df.iloc[sorted(held_set)].reset_index(drop=True))


def scaffold_train_val_test_split(df, test_fraction=0.20, val_fraction_of_trainval=0.20, seed=42):
    """3-way scaffold-grouped split. No split_key ever spans two of train/val/test."""
    df = prepare(df).reset_index(drop=True)
    train_val, test = _group_split(df, test_fraction, seed=seed)
    train, val = _group_split(train_val, val_fraction_of_trainval, seed=seed + 1)
    return train, val, test


def scaffold_group_kfold(df, n_splits=5, seed=42):
    """Scaffold-grouped K-fold over df. Returns list of (train_pos, val_pos) index lists.

    Groups are shuffled, then each whole group is assigned to whichever fold is
    currently smallest -- balances fold sizes despite very unequal group sizes while
    never splitting a scaffold across folds.
    """
    df = df.reset_index(drop=True)
    if "split_key" not in df.columns:
        df = prepare(df)
    groups = defaultdict(list)
    for pos, key in enumerate(df["split_key"]):
        groups[key].append(pos)

    keys = list(groups.keys())
    import random as _random
    _random.Random(seed).shuffle(keys)

    fold_pos = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for key in keys:
        t = min(range(n_splits), key=lambda i: sizes[i])
        fold_pos[t].extend(groups[key])
        sizes[t] += len(groups[key])

    all_pos = set(range(len(df)))
    folds = []
    for i in range(n_splits):
        v = sorted(fold_pos[i])
        folds.append((sorted(all_pos - set(v)), v))
    return folds


def assert_disjoint(*dfs, col="canonical_smiles"):
    sets = [set(d[col]) for d in dfs]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            assert not inter, f"overlap between split {i} and {j}: {len(inter)} molecules"


def assert_scaffold_disjoint(*dfs):
    sets = [set(d["split_key"]) for d in dfs]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            assert not inter, f"split_key overlap between split {i} and {j}: {len(inter)} keys"
