"""
20_error_analysis.py
====================
Where does the current best model fail?

Uses OUT-OF-FOLD train+val predictions only. The test set is deliberately not opened:
looking at test errors and then designing a fix against them is exactly the leakage
this project spent a whole audit removing. The OOF predictions are the honest proxy --
they come from models that never saw the molecule they are predicting.

Breaks the error down by:
  LogS decile, molecular weight band, salt/mixture (multi-component SMILES),
  charged species, acyclic vs ring-containing, scaffold-group size (how much
  structural support the molecule had in training), element composition,
  and the single worst-predicted molecules.

Output: results/error_analysis.json, results/error_analysis.csv (per-molecule),
        results/error_analysis_report.md
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/home/claude/work"
DATA = f"{ROOT}/data/v2_corrected"
RES = f"{ROOT}/results"
BEST = "lightgbm[mordred]"

trainval = pd.read_csv(f"{DATA}/trainval.csv")
oof = np.load(f"{RES}/oof_matrix.npz")
y = oof["y"]
assert np.allclose(y, trainval["Solubility"].values)

pred = oof[BEST]
err = pred - y
abs_err = np.abs(err)

df = trainval.copy()
df["pred"] = pred
df["error"] = err
df["abs_error"] = abs_err

# ---------------------------------------------------------------- molecule properties
props = []
for smi in df["canonical_smiles"]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        props.append({})
        continue
    frags = smi.count(".") + 1
    charges = [a.GetFormalCharge() for a in m.GetAtoms()]
    elements = {a.GetSymbol() for a in m.GetAtoms()}
    organic = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
    props.append({
        "mw": Descriptors.MolWt(m),
        "logp": Crippen.MolLogP(m),
        "tpsa": Descriptors.TPSA(m),
        "n_frags": frags,
        "is_mixture_or_salt": frags > 1,
        "net_charge": sum(charges),
        "is_charged": any(c != 0 for c in charges),
        "n_rings": Descriptors.RingCount(m),
        "is_acyclic": Descriptors.RingCount(m) == 0,
        "n_heavy": m.GetNumHeavyAtoms(),
        "has_metal_or_exotic": bool(elements - organic),
        "elements": "".join(sorted(elements)),
        "n_aromatic_rings": Descriptors.NumAromaticRings(m),
        "rotatable": Descriptors.NumRotatableBonds(m),
    })
df = pd.concat([df, pd.DataFrame(props, index=df.index)], axis=1)

# scaffold group size: how many training molecules share this molecule's split key
gsize = df["split_key"].map(df["split_key"].value_counts())
df["scaffold_group_size"] = gsize


def rmse(a):
    return float(np.sqrt(np.mean(a ** 2)))


overall = rmse(df["error"].values)
report, findings = {}, []
report["model"] = BEST
report["n"] = len(df)
report["pooled_oof_rmse"] = overall
report["pooled_oof_mae"] = float(df["abs_error"].mean())
report["oof_r2"] = float(1 - np.sum(df["error"] ** 2) / np.sum((y - y.mean()) ** 2))
report["mean_signed_error"] = float(df["error"].mean())

print(f"{BEST}: OOF RMSE={overall:.4f}  MAE={df['abs_error'].mean():.4f}  "
      f"mean signed error={df['error'].mean():+.4f}")


def group_table(col, bins=None, labels=None, name=None, min_n=30):
    name = name or col
    if bins is not None:
        key = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    else:
        key = df[col]
    g = df.groupby(key, observed=True).apply(
        lambda d: pd.Series({
            "n": len(d),
            "rmse": rmse(d["error"].values),
            "mae": float(d["abs_error"].mean()),
            "bias": float(d["error"].mean()),
            "share_of_total_sq_error": float(np.sum(d["error"] ** 2) / np.sum(df["error"] ** 2)),
        }), include_groups=False)
    g = g[g["n"] >= min_n].sort_values("rmse", ascending=False)
    report[name] = json.loads(g.reset_index().to_json(orient="records"))
    print(f"\n--- {name} ---")
    print(g.to_string())
    return g


# LogS deciles
qs = np.quantile(y, np.linspace(0, 1, 11))
qs[0] -= 1e-9
group_table("Solubility", bins=qs,
            labels=[f"d{i+1} [{qs[i]:.1f},{qs[i+1]:.1f}]" for i in range(10)],
            name="by_logS_decile")

group_table("mw", bins=[0, 150, 250, 350, 500, 900, 1e9],
            labels=["<150", "150-250", "250-350", "350-500", "500-900", ">900"],
            name="by_molecular_weight")

group_table("is_mixture_or_salt", name="by_salt_or_mixture", min_n=10)
group_table("is_charged", name="by_charged", min_n=10)
group_table("is_acyclic", name="by_acyclic", min_n=10)
group_table("has_metal_or_exotic", name="by_inorganic_element", min_n=10)
group_table("scaffold_group_size", bins=[0, 1, 2, 5, 20, 100, 1e9],
            labels=["singleton", "2", "3-5", "6-20", "21-100", ">100"],
            name="by_scaffold_group_size")
group_table("logp", bins=[-1e9, -2, 0, 2, 4, 6, 1e9],
            labels=["<-2", "-2-0", "0-2", "2-4", "4-6", ">6"], name="by_clogp")

# ---------------------------------------------------------------- worst molecules
worst = df.nlargest(30, "abs_error")[
    ["Name", "canonical_smiles", "Solubility", "pred", "error", "mw", "logp",
     "n_frags", "is_charged", "has_metal_or_exotic", "n_rings", "scaffold_group_size"]]
worst.to_csv(f"{RES}/error_analysis_worst30.csv", index=False)
report["worst_30"] = json.loads(worst.to_json(orient="records"))
print("\n--- 30 worst OOF predictions ---")
print(worst.head(30).to_string(index=False))

# how concentrated is the error?
sq = np.sort(df["error"].values ** 2)[::-1]
tot = sq.sum()
for frac in [0.01, 0.05, 0.10, 0.25]:
    k = int(len(sq) * frac)
    report[f"share_of_sq_error_in_worst_{int(frac*100)}pct"] = float(sq[:k].sum() / tot)
print("\nError concentration:")
for frac in [0.01, 0.05, 0.10, 0.25]:
    print(f"  worst {int(frac*100):>2d}% of molecules hold "
          f"{100*report[f'share_of_sq_error_in_worst_{int(frac*100)}pct']:.1f}% of squared error")

df.to_csv(f"{RES}/error_analysis.csv", index=False)
with open(f"{RES}/error_analysis.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved {RES}/error_analysis.json / .csv / _worst30.csv")
