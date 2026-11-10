"""
Phase 2: Retrieval.

Loads the featurized training set, builds an in-memory fingerprint matrix,
and provides a function to retrieve the top-k most similar training molecules
for any query molecule (by Tanimoto similarity on Morgan fingerprints).

This script also runs a quick demo: picks one molecule from the test set
and shows its top-k nearest neighbors in the training set, so you can sanity-check
that the retrieval makes chemical sense before we wire it into the Claude prompt.
"""

import pandas as pd
import os
import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

INPUT_DIR = r"C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\PhD\AI\AqSolPred"
TRAIN_FEAT = os.path.join(INPUT_DIR, "train_featurized.csv")
TEST_FEAT = os.path.join(INPUT_DIR, "test_featurized.csv")

K = 20  # number of nearest neighbors to retrieve

def bitstring_to_bitvect(bitstring):
    """Convert our stored '0101...' string back into an RDKit ExplicitBitVect."""
    bv = ExplicitBitVect(len(bitstring))
    for i, bit in enumerate(bitstring):
        if bit == "1":
            bv.SetBit(i)
    return bv

def load_train_fingerprints(train_df):
    """Precompute all training fingerprints as RDKit bit vectors, once."""
    return [bitstring_to_bitvect(fp) for fp in train_df["morgan_fp"]]

def retrieve_top_k(query_bitstring, train_fps, train_df, k=K):
    """
    Given a query fingerprint (as a bitstring), return the top-k most similar
    training molecules, along with their similarity scores.
    """
    query_bv = bitstring_to_bitvect(query_bitstring)

    # BulkTanimotoSimilarity computes similarity against the whole list at once - fast.
    similarities = DataStructs.BulkTanimotoSimilarity(query_bv, train_fps)

    # Get indices of the top-k highest similarity scores
    top_k_idx = np.argsort(similarities)[::-1][:k]

    results = train_df.iloc[top_k_idx].copy()
    results["similarity"] = [similarities[i] for i in top_k_idx]
    return results

def main():
    train_df = pd.read_csv(TRAIN_FEAT)
    test_df = pd.read_csv(TEST_FEAT)

    print(f"Loaded {len(train_df)} training molecules, {len(test_df)} test molecules.")
    print("Precomputing training fingerprint bit vectors (one-time cost)...")
    train_fps = load_train_fingerprints(train_df)

    # --- Demo: retrieve neighbors for the first test molecule ---
    sample = test_df.iloc[0]
    print(f"\nQuery molecule: {sample['Name']} | SMILES: {sample['canonical_smiles']}")
    print(f"True LogS: {sample['Solubility']}")

    neighbors = retrieve_top_k(sample["morgan_fp"], train_fps, train_df, k=K)

    print(f"\nTop {K} nearest training neighbors:")
    print(neighbors[["Name", "canonical_smiles", "Solubility", "similarity"]].to_string(index=False))

    max_sim = neighbors["similarity"].max()
    print(f"\nMax similarity to any training molecule: {max_sim:.3f}")
    if max_sim < 0.4:
        print("NOTE: low max similarity - this query is in a sparsely covered region of chemical space.")

if __name__ == "__main__":
    main()
