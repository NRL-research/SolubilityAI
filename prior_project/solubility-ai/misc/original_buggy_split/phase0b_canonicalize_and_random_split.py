"""
Phase 0b: Canonicalize SMILES + Random train/test split for AqSolDB.

Same as phase0_canonicalize_and_split.py except splitting method is random
instead of scaffold-based.

Input:  AqSolDB CSV with columns: ID, Name, InChIKey, SMILES, Solubility
Output: train_random.csv and test_random.csv with an added 'scaffold' column for inspection
"""

import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

INPUT_CSV = r"C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\PhD\AI\AqSolPred\AqSolDB.csv"
OUTPUT_DIR = os.path.dirname(INPUT_CSV)
TRAIN_OUT = os.path.join(OUTPUT_DIR, "train_random.csv")
TEST_OUT = os.path.join(OUTPUT_DIR, "test_random.csv")
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

    # Step 2: compute scaffold for each molecule (kept for inspection column, not used for splitting)
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    n_before = len(df)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows where scaffold could not be computed.")

    print(f"Total molecules after cleaning: {len(df)}")

    # Step 3: random split - shuffle all indices, assign first TEST_FRACTION to test
    all_indices = list(range(len(df)))
    random.seed(RANDOM_SEED)
    random.shuffle(all_indices)

    n_test = int(len(df) * TEST_FRACTION)
    test_indices = set(all_indices[:n_test])
    train_indices = all_indices[n_test:]

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
