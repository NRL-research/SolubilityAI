"""Step 03b: Mordred 2D descriptor cache (mordredcommunity, ignore_3D).  Columns kept = those
with <20% missing and non-constant on the TRAIN split only (list saved; test values not used).
Output: data/07b_features_mordred.npz"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import featurize as F
from mordred import Calculator, descriptors as mdesc
df = pd.read_csv(R / "data/05_split_assignments.csv")
names = [str(d) for d in Calculator(mdesc, ignore_3D=True).descriptors]
X = F.mordred_desc([F.standardize(s) for s in df.SMILES], names)
tr = (df.split == "train").values
keep = (np.isnan(X[tr]).mean(0) < 0.2) & (np.nanstd(X[tr], 0) > 0)
kn = [n for n, k in zip(names, keep) if k]
np.savez_compressed(R / "data/07b_features_mordred.npz", mol_id=np.array(df.mol_id.tolist(), dtype="U10"),
                    mordred_names=np.array(kn, dtype=str), mordred=X[:, keep])
print(len(names), "->", len(kn))
