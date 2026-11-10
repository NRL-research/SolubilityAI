"""
11_bug_impact.py
================
Quantify what the acyclic-scaffold-splitting bug did to THIS project's own reported
numbers, by running the identical model on the identical data under the buggy split
and under the corrected split.

Model: solo CatBoost(3000, lr=0.02, depth=8, l2=5) on Morgan2048+5desc -- exactly the
"Solo CatBoost (diagnostic)" configuration of History Table 4, which reported
R2 = 0.7223 +- 0.0015 on the buggy split.

Buggy split: solubility_ai_v2.ipynb cell 37 verbatim (80/20, seed 42, no validation
carve-out -- the notebook's solo-CatBoost diagnostic, cell 83, trains on the full
train side), reproduced in 02_audit_scaffold_split.py.

Corrected split: data/v2_corrected (train+val used for training, test held out) -- the
same 80/20 proportion of the same 9,872 molecules.

Output: results/bug_impact.csv / .json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common_eval import metrics  # noqa: E402
from featurizer import featurize  # noqa: E402
from models import build  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("audit", os.path.join(os.path.dirname(__file__), "02_audit_scaffold_split.py"))

RES = "/home/claude/work/results"
DATA = "/home/claude/work/data/v2_corrected"
SRC = "/home/claude/work/data/v1_sc_filtered/aqsoldb_sc_filtered.csv"
CACHE = "/home/claude/work/data/features_cache"
SEEDS = [42, 101, 123]

# --- reproduce the buggy split inline (same code as 02_audit, kept here so this script
#     is standalone and does not execute the audit script's module-level work)
from collections import defaultdict  # noqa: E402
import random  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
RDLogger.DisableLog("rdApp.*")


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def scaffold_split_BUGGY(df, test_fraction=0.20, seed=42):
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    groups = defaultdict(list)
    for idx, s in enumerate(df["scaffold"]):
        groups[s].append(idx)
    keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(keys)
    n_target = int(len(df) * test_fraction)
    test_idx = []
    for k in keys:
        if len(test_idx) >= n_target:
            break
        test_idx.extend(groups[k])
    test_idx = set(test_idx)
    train_idx = [i for i in range(len(df)) if i not in test_idx]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[sorted(test_idx)].reset_index(drop=True)


rows = []

# ---------------------------------------------------------------- buggy split
full = pd.read_csv(SRC)
btr, bte = scaffold_split_BUGGY(full)
print(f"buggy split: train={len(btr)} test={len(bte)} "
      f"acyclic_in_test={(bte['scaffold']=='').sum()}")
Xtr, ok1 = featurize(btr["canonical_smiles"].tolist(), "solubility_ai")
Xte, ok2 = featurize(bte["canonical_smiles"].tolist(), "solubility_ai")
assert ok1.all() and ok2.all()
ytr, yte = btr["Solubility"].values, bte["Solubility"].values
ms = []
for s in SEEDS:
    m = build("catboost", seed=s)
    m.fit(Xtr, ytr)
    ms.append(metrics(yte, m.predict(Xte)))
    print(f"  buggy seed {s}: {ms[-1]}", flush=True)
rows.append({"split": "buggy (notebook cell 37)", "n_train": len(btr), "n_test": len(bte),
             "acyclic_in_test": int((bte["scaffold"] == "").sum()),
             "rmse_mean": float(np.mean([m["rmse"] for m in ms])),
             "rmse_std": float(np.std([m["rmse"] for m in ms])),
             "r2_mean": float(np.mean([m["r2"] for m in ms])),
             "r2_std": float(np.std([m["r2"] for m in ms])),
             "mae_mean": float(np.mean([m["mae"] for m in ms]))})

# ---------------------------------------------------------------- corrected split
ctr = pd.read_csv(f"{DATA}/trainval.csv")
cte = pd.read_csv(f"{DATA}/test.csv")
Xtr = np.vstack([np.load(f"{CACHE}/train__solubility_ai.npy"),
                 np.load(f"{CACHE}/val__solubility_ai.npy")]).astype(np.float64)
Xte = np.load(f"{CACHE}/test__solubility_ai.npy").astype(np.float64)
ytr, yte = ctr["Solubility"].values, cte["Solubility"].values
ms = []
for s in SEEDS:
    m = build("catboost", seed=s)
    m.fit(Xtr, ytr)
    ms.append(metrics(yte, m.predict(Xte)))
    print(f"  corrected seed {s}: {ms[-1]}", flush=True)
rows.append({"split": "corrected (acyclic singleton keys)", "n_train": len(ctr), "n_test": len(cte),
             "acyclic_in_test": int(cte["split_key"].str.startswith("__acyclic__").sum()),
             "rmse_mean": float(np.mean([m["rmse"] for m in ms])),
             "rmse_std": float(np.std([m["rmse"] for m in ms])),
             "r2_mean": float(np.mean([m["r2"] for m in ms])),
             "r2_std": float(np.std([m["r2"] for m in ms])),
             "mae_mean": float(np.mean([m["mae"] for m in ms]))})

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/bug_impact.csv", index=False)
with open(f"{RES}/bug_impact.json", "w") as f:
    json.dump(rows, f, indent=2)
pd.set_option("display.width", 220)
print("\n" + df.to_string(index=False))
print("\nHistory Table 4 reported R2 = 0.7223 +- 0.0015 for this exact configuration "
      "on the buggy split.")
