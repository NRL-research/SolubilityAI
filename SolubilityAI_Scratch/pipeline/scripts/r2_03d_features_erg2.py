"""Round 2, step 03d (bug fix): cache the deterministic ErG block 'erg2' (see featurize.erg2).
Output: data/07d_features_erg2.npz"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import featurize as F
df = pd.read_csv(R / "data/05_split_assignments.csv")
X = F.erg2([F.standardize(s) for s in df.SMILES])
np.savez_compressed(R / "data/07d_features_erg2.npz", mol_id=np.array(df.mol_id.tolist(), dtype="U10"), erg2=X)
print(X.shape)
