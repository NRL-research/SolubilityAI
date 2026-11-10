"""
Scaffold and random train/val/test splitting for AqSolDB.

Fixes a bug present in the original misc/original_buggy_split/phase0_canonicalize_and_split.py:
Murcko scaffolds are empty for acyclic (ring-free) molecules, so every acyclic
molecule shared the same scaffold key. Because the split logic assigns whole
scaffold groups to test, that single giant group either landed entirely in
train or entirely in test as one block. In the originally committed split it
landed entirely in train (2,940 / 9,980 molecules, ~29.5% of the dataset),
leaving the test set blind to all ring-free chemistry.

Fix: acyclic molecules are each given a unique singleton scaffold key (their
own canonical SMILES) instead of sharing one empty-string key, so they are
distributed across train/val/test individually, same as any other molecule
count would be under random assignment.
"""

import random
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_scaffold(canonical_smiles):
    """Murcko scaffold SMILES, or '' if the molecule has no ring system."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def get_split_key(row):
    """
    Grouping key used for scaffold-aware splitting.

    Ring-containing molecules: the real Murcko scaffold (all molecules
    sharing a scaffold stay on the same side of the split).

    Acyclic molecules (empty scaffold): a unique per-molecule key, so they
    are distributed individually rather than dumped as one giant block.

    Checks pd.isna() as well as falsiness: a fresh-from-RDKit scaffold is ''
    (falsy, caught either way), but an empty string written to CSV and
    reloaded comes back as NaN/NA -- and `if nan:` is True in Python, so a
    naive truthiness check silently treats every reloaded acyclic molecule
    as if it had a real (and identical) scaffold, reintroducing the exact
    bug this module exists to fix. Confirmed via scaffold_group_kfold()
    operating on reloaded train.csv/val.csv: 2,170 molecules' NaN scaffolds
    all hashed to one group before this check was added.
    """
    scaffold = row["scaffold"]
    if pd.isna(scaffold) or not scaffold:
        return f"__acyclic__::{row['canonical_smiles']}"
    return scaffold


def prepare_dataframe(df, smiles_col="SMILES"):
    """Canonicalize SMILES and compute scaffolds; drop unparseable rows."""
    df = df.copy()
    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)
    n_before = len(df)
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    dropped_smiles = n_before - len(df)

    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    n_before = len(df)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    dropped_scaffold = n_before - len(df)

    df["split_key"] = df.apply(get_split_key, axis=1)
    return df, dropped_smiles, dropped_scaffold


def _group_split(df, key_col, test_fraction, seed):
    """Shuffle groups by key_col, assign whole groups to 'test' until the
    target fraction is reached; everything else goes to 'train'."""
    groups = defaultdict(list)
    for idx, key in zip(df.index, df[key_col]):
        groups[key].append(idx)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_target = int(len(df) * test_fraction)
    test_idx = []
    for key in keys:
        if len(test_idx) >= n_target:
            break
        test_idx.extend(groups[key])

    test_idx = set(test_idx)
    train_idx = [i for i in df.index if i not in test_idx]
    return df.loc[train_idx].reset_index(drop=True), df.loc[sorted(test_idx)].reset_index(drop=True)


def scaffold_train_val_test_split(df, smiles_col="SMILES", test_fraction=0.20,
                                   val_fraction_of_train=0.20, seed=42):
    """
    Scaffold-grouped 3-way split. No scaffold (or acyclic singleton key)
    ever spans two splits. val_fraction_of_train is applied to the
    post-test train set (e.g. 0.20 of 80% train -> ~16% overall val,
    ~64% overall train, 20% overall test).
    """
    df, dropped_smiles, dropped_scaffold = prepare_dataframe(df, smiles_col=smiles_col)

    train_val_df, test_df = _group_split(df, "split_key", test_fraction, seed=seed)
    train_df, val_df = _group_split(train_val_df, "split_key", val_fraction_of_train, seed=seed + 1)

    stats = {
        "dropped_invalid_smiles": dropped_smiles,
        "dropped_no_scaffold": dropped_scaffold,
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    return train_df, val_df, test_df, stats


def random_train_val_test_split(df, smiles_col="SMILES", test_fraction=0.20,
                                 val_fraction_of_train=0.20, seed=42):
    """Random (non-scaffold-aware) 3-way split, for the 'inflated' comparison baseline."""
    df, dropped_smiles, dropped_scaffold = prepare_dataframe(df, smiles_col=smiles_col)

    rng = random.Random(seed)
    all_idx = list(df.index)
    rng.shuffle(all_idx)
    n_test = int(len(df) * test_fraction)
    test_idx = set(all_idx[:n_test])
    train_val_idx = all_idx[n_test:]

    rng2 = random.Random(seed + 1)
    rng2.shuffle(train_val_idx)
    n_val = int(len(train_val_idx) * val_fraction_of_train)
    val_idx = set(train_val_idx[:n_val])
    train_idx = [i for i in train_val_idx if i not in val_idx]

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[sorted(val_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)

    stats = {
        "dropped_invalid_smiles": dropped_smiles,
        "dropped_no_scaffold": dropped_scaffold,
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    return train_df, val_df, test_df, stats


def scaffold_group_kfold(df, n_splits=5, seed=42):
    """
    Scaffold-grouped K-fold split, for more robust model/hyperparameter
    selection than a single fixed train/val split.

    Why this exists: the single scaffold-grouped train/val split used
    throughout this project has real variance -- validation happened to end
    up with only 3.5% acyclic molecules vs. ~35% in train/test, purely from
    which scaffold groups landed on which side of one random shuffle.
    K-fold CV averages over many such splits instead of trusting one.

    df must already have 'scaffold' and 'canonical_smiles' columns (e.g. the
    output of scaffold_train_val_test_split / prepare_dataframe -- typically
    called on train.csv + val.csv concatenated, never on test).

    Fold assignment: shuffle scaffold groups (same acyclic-singleton-key fix
    as the 2-way/3-way splitters), then greedily assign each whole group to
    whichever fold currently has the fewest molecules -- keeps fold sizes
    balanced despite very unequal group sizes (e.g. the 1,753-molecule
    benzene group) while still never splitting a scaffold across folds.

    Returns a list of n_splits (train_idx, val_idx) index arrays into df
    (positions after df.reset_index(drop=True), not original labels).
    """
    df = df.reset_index(drop=True)
    keys = [get_split_key(row) for _, row in df.iterrows()]

    groups = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[key].append(idx)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for key in group_keys:
        target_fold = min(range(n_splits), key=lambda i: fold_sizes[i])
        fold_indices[target_fold].extend(groups[key])
        fold_sizes[target_fold] += len(groups[key])

    all_idx = set(range(len(df)))
    folds = []
    for i in range(n_splits):
        val_idx = sorted(fold_indices[i])
        train_idx = sorted(all_idx - set(val_idx))
        folds.append((train_idx, val_idx))
    return folds


def assert_no_leakage(train_df, val_df, test_df, smiles_col="canonical_smiles"):
    train_set = set(train_df[smiles_col])
    val_set = set(val_df[smiles_col])
    test_set = set(test_df[smiles_col])
    assert not (train_set & val_set), "train/val canonical_smiles overlap"
    assert not (train_set & test_set), "train/test canonical_smiles overlap"
    assert not (val_set & test_set), "val/test canonical_smiles overlap"
