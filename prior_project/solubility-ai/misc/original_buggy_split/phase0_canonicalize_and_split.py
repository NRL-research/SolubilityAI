"""
Phase 0: Canonicalize SMILES + Scaffold-based train/test split for AqSolDB.

Input:  AqSolDB CSV with columns: ID, Name, InChIKey, SMILES, Solubility
Output: train.csv and test.csv with an added 'scaffold' column for inspection
"""

import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

INPUT_CSV = r"C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\PhD\AI\AqSolPred\AqSolDB.csv"
OUTPUT_DIR = os.path.dirname(INPUT_CSV)
TRAIN_OUT = os.path.join(OUTPUT_DIR, "train.csv")
TEST_OUT = os.path.join(OUTPUT_DIR, "test.csv")
TEST_FRACTION = 0.20              # 80/20 split
RANDOM_SEED = 42

def canonicalize_smiles(smiles):
    """Convert a SMILES string to RDKit's canonical form. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_scaffold(smiles):
    """Get the Murcko scaffold (core ring system) for a molecule, as a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")

    # Step 1: canonicalize, drop invalid SMILES
    df["canonical_smiles"] = df["SMILES"].apply(canonicalize_smiles)
    n_before = len(df)
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with invalid SMILES. {len(df)} remain.")

    # Step 2: compute scaffold for each molecule
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    n_before = len(df)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows where scaffold could not be computed.")

    # Step 3: group molecules by scaffold
    scaffold_groups = defaultdict(list)
    for idx, scaffold in enumerate(df["scaffold"]):
        scaffold_groups[scaffold].append(idx)

    scaffold_keys = list(scaffold_groups.keys())
    print(f"Found {len(scaffold_keys)} unique scaffolds across {len(df)} molecules.")

    # Step 4: shuffle scaffold groups, assign whole groups to test until we hit the target fraction
    random.seed(RANDOM_SEED)
    random.shuffle(scaffold_keys)

    n_test_target = int(len(df) * TEST_FRACTION)
    test_indices = []
    for key in scaffold_keys:
        if len(test_indices) >= n_test_target:
            break
        test_indices.extend(scaffold_groups[key])

    test_indices = set(test_indices)
    train_indices = [i for i in range(len(df)) if i not in test_indices]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[sorted(test_indices)].reset_index(drop=True)

    print(f"Train set: {len(train_df)} molecules")
    print(f"Test set:  {len(test_df)} molecules")
    print(f"Actual test fraction: {len(test_df) / len(df):.3f} (target was {TEST_FRACTION})")

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    print(f"Saved {TRAIN_OUT} and {TEST_OUT}")

if __name__ == "__main__":
    main()
