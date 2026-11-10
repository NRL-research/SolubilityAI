"""Step 03: compute and cache all tabular feature sets for the curated dataset.
Output: data/07_features.npz (arrays keyed by feature block + mol_id + desc_names). Deterministic."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import featurize as F
df = pd.read_csv(R / "data/05_split_assignments.csv")
mols = [F.standardize(s) for s in df.SMILES]
assert all(m is not None for m in mols)
names = F.RDKIT_DESC + F.EXTRA_DESC
np.savez_compressed(R / "data/07_features.npz", mol_id=np.array(df.mol_id.tolist(), dtype="U10"), desc_names=np.array(names, dtype=str),
    desc=F.descriptors(mols, names), morgan_counts=F.morgan_counts(mols), maccs=F.maccs(mols), esol4=F.esol(mols))
print(len(names), "descriptors")
