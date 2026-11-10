"""
Phase 1: Featurization.

For every molecule in train.csv and test.csv, compute:
  (a) Morgan fingerprint (radius=2, 2048 bits) -> used only for similarity retrieval
  (b) Pillar 1 descriptors: functional group fragment counts (RDKit fr_* )
  (c) Pillar 2 descriptors: topology indices (BalabanJ, BertzCT, Kappa1/2, ring counts)

Output: train_featurized.csv and test_featurized.csv
  - fingerprint stored as a comma-separated bit string in one column (for compactness)
  - Pillar 1/2 descriptors stored as individual columns
"""

import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, GraphDescriptors
from rdkit.Chem import Fragments

INPUT_DIR = r"C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\PhD\AI\AqSolPred"
TRAIN_IN = os.path.join(INPUT_DIR, "train.csv")
TEST_IN = os.path.join(INPUT_DIR, "test.csv")
TRAIN_OUT = os.path.join(INPUT_DIR, "train_featurized.csv")
TEST_OUT = os.path.join(INPUT_DIR, "test_featurized.csv")

FP_RADIUS = 2
FP_BITS = 2048

# Pillar 1: a curated subset of RDKit's functional-group fragment counters.
# (RDKit has ~85 of these; this is a representative, solubility-relevant subset.
#  Add more from rdkit.Chem.Fragments if you want a fuller census.)
PILLAR1_FRAGMENTS = {
    "fr_ketone": Fragments.fr_ketone,
    "fr_aldehyde": Fragments.fr_aldehyde,
    "fr_ester": Fragments.fr_ester,
    "fr_ether": Fragments.fr_ether,
    "fr_amide": Fragments.fr_amide,
    "fr_amine": Fragments.fr_NH2,
    "fr_aniline": Fragments.fr_aniline,
    "fr_phenol": Fragments.fr_phenol,
    "fr_alcohol": Fragments.fr_Al_OH,
    "fr_carboxylic_acid": Fragments.fr_COO,
    "fr_halogen": Fragments.fr_halogen,
    "fr_sulfonamide": Fragments.fr_sulfonamd,
    "fr_benzene": Fragments.fr_benzene,
    "fr_pyridine": Fragments.fr_pyridine,
    "fr_nitrile": Fragments.fr_nitrile,
}

def compute_morgan_fp(mol):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    return fp.ToBitString()  # compact string of 0s/1s, easy to store in a CSV cell

def compute_pillar1(mol):
    return {name: func(mol) for name, func in PILLAR1_FRAGMENTS.items()}

def compute_pillar2(mol):
    return {
        "balaban_j": GraphDescriptors.BalabanJ(mol),
        "bertz_ct": GraphDescriptors.BertzCT(mol),
        "kappa1": GraphDescriptors.Kappa1(mol),
        "kappa2": GraphDescriptors.Kappa2(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

def featurize_dataframe(df, smiles_col="canonical_smiles"):
    fps = []
    pillar1_rows = []
    pillar2_rows = []
    valid_mask = []

    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            fps.append(None)
            pillar1_rows.append(None)
            pillar2_rows.append(None)
            continue
        valid_mask.append(True)
        fps.append(compute_morgan_fp(mol))
        pillar1_rows.append(compute_pillar1(mol))
        pillar2_rows.append(compute_pillar2(mol))

    df = df[valid_mask].reset_index(drop=True)
    fps = [f for f, v in zip(fps, valid_mask) if v]
    pillar1_rows = [p for p, v in zip(pillar1_rows, valid_mask) if v]
    pillar2_rows = [p for p, v in zip(pillar2_rows, valid_mask) if v]

    df["morgan_fp"] = fps
    pillar1_df = pd.DataFrame(pillar1_rows)
    pillar2_df = pd.DataFrame(pillar2_rows)

    return pd.concat([df.reset_index(drop=True), pillar1_df, pillar2_df], axis=1)

def main():
    train_df = pd.read_csv(TRAIN_IN)
    test_df = pd.read_csv(TEST_IN)

    print(f"Featurizing {len(train_df)} training molecules...")
    train_feat = featurize_dataframe(train_df)
    print(f"Featurizing {len(test_df)} test molecules...")
    test_feat = featurize_dataframe(test_df)

    print(f"Train: {len(train_feat)} molecules successfully featurized (dropped {len(train_df) - len(train_feat)})")
    print(f"Test:  {len(test_feat)} molecules successfully featurized (dropped {len(test_df) - len(test_feat)})")

    train_feat.to_csv(TRAIN_OUT, index=False)
    test_feat.to_csv(TEST_OUT, index=False)
    print(f"Saved {TRAIN_OUT} and {TEST_OUT}")

if __name__ == "__main__":
    main()
