"""Round 2, step 03c: cache new feature blocks (atom-pair, torsion, Avalon, ErG, Crippen atom
types, packing proxies, mol2vec) for all curated molecules.  Deterministic.
Output: data/07c_features_r2.npz"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import featurize as F
df = pd.read_csv(R / "data/05_split_assignments.csv")
mols = [F.standardize(s) for s in df.SMILES]
out = {k: fn(mols) for k, fn in F.R2_BLOCKS.items()}
for k, v in out.items(): print(k, v.shape, float(np.isnan(v).mean()))
np.savez_compressed(R / "data/07c_features_r2.npz", mol_id=np.array(df.mol_id.tolist(), dtype="U10"), **out)
