"""Step 08: independent verification.

A. Identity-level leakage audit, recomputed from the raw SMILES with RDKit directly (does not
   reuse src/chem.py keys): full InChIKey, InChIKey connectivity block, and parent
   (largest fragment, neutralized) connectivity block, for train vs test, train vs SC*, test vs SC*.
B. CV fold audit: folds disjoint, cover train exactly, no test molecule, no shared group /
   parent identity between any fold and the rest of train.
C. Saved-model reload: each saved model is loaded in a FRESH python subprocess from disk via
   its predict.py; (i) reference-molecule predictions must match those recorded at save time
   (atol 1e-4), (ii) test metrics recomputed from scratch must match results/07_test_metrics.csv
   (atol 1e-6).  Output: results/08_verification.json
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
R = Path(__file__).resolve().parents[1]
out = {}

def keys(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return set()
    ks = set()
    k = Chem.MolToInchiKey(m); ks |= {"F:" + k, "B:" + k[:14]}
    p = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(m))
    ks.add("B:" + Chem.MolToInchiKey(p)[:14])
    return ks

tr = pd.read_csv(R / "data/06_train.csv"); te = pd.read_csv(R / "data/06_test.csv")
def keyset(smiles_iter):
    s = set()
    for x in smiles_iter: s |= keys(x)
    return s
# train/test molecules may be merged duplicates: use all raw source SMILES
raw = pd.read_csv(R / "data/00_raw/AqSolDB.csv").set_index("ID")
def raw_smiles(df): return [raw.SMILES[i] for ids in df.ID for i in ids.split("|")]
K = {"train": keyset(raw_smiles(tr)), "test": keyset(raw_smiles(te))}
sc = []
for f in ["SC1_processed", "SC2_tight_processed", "SC2_loose_processed"]:
    s = pd.read_csv(R / f"data/00_raw/{f}.csv"); sc += s.smiles.tolist() + s.canonical_smiles.tolist()
    K["sc_given_ik"] = K.get("sc_given_ik", set()) | {"F:" + k for k in s.inchikey} | {"B:" + k[:14] for k in s.inchikey}
K["sc"] = keyset(sc) | K.pop("sc_given_ik")
A = {}
for a, b in [("train", "test"), ("train", "sc"), ("test", "sc")]:
    inter = K[a] & K[b]
    A[f"{a}_vs_{b}"] = dict(n_shared_keys=len(inter), examples=sorted(inter)[:5])
A["pass"] = all(v["n_shared_keys"] == 0 for k, v in A.items() if k != "pass")
out["A_identity_leakage"] = A

# B. folds
df = pd.read_csv(R / "data/05_split_assignments.csv")
trd = df[df.split == "train"]
Bres = dict(test_in_any_fold=int((df[df.split == "test"].cv_fold != -1).sum()),
            train_without_fold=int((trd.cv_fold == -1).sum()),
            fold_sizes=trd.cv_fold.value_counts().sort_index().tolist(),
            duplicate_mol_ids=int(df.mol_id.duplicated().sum()))
pair = []
for k in range(5):
    va, rest = trd[trd.cv_fold == k], trd[trd.cv_fold != k]
    va_keys = keyset(raw_smiles(va)); rest_keys = keyset(raw_smiles(rest))
    pair.append(dict(fold=k, shared_mol_ids=len(set(va.mol_id) & set(rest.mol_id)),
                     shared_groups=len(set(va.group) & set(rest.group)),
                     shared_identity_keys=len(va_keys & rest_keys)))
Bres["per_fold"] = pair
Bres["pass"] = (Bres["test_in_any_fold"] == 0 and Bres["train_without_fold"] == 0 and Bres["duplicate_mol_ids"] == 0
                and all(p["shared_mol_ids"] == 0 and p["shared_groups"] == 0 and p["shared_identity_keys"] == 0 for p in pair))
out["B_cv_folds"] = Bres

# C. fresh-process reload
tm = pd.read_csv(R / "results/07_test_metrics_fulltest.csv").set_index("name") if (R / "results/07_test_metrics_fulltest.csv").exists() else None  # full test (n=1852) is what the child recomputes
CHILD = r'''
import sys, json, importlib.util, numpy as np, pandas as pd
folder, test_csv = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("p", folder + "/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pred = m.load_predictor(folder)
ref = pd.read_csv(folder + "/reference_predictions.csv")
rp = pred(ref.smiles.tolist())
te = pd.read_csv(test_csv); tp = pred(te.SMILES.tolist()); y = te.LogS.values
print(json.dumps(dict(ref_max_abs_diff=float(np.max(np.abs(rp - ref.pred_at_save.values))),
                      test_rmse=float(np.sqrt(np.mean((y - tp) ** 2))),
                      test_r2=float(1 - np.sum((y - tp) ** 2) / np.sum((y - y.mean()) ** 2)))))
'''
C = {}
for folder in sorted((R / "saved_models").iterdir()):
    if not (folder / "config.json").exists(): continue
    r = subprocess.run([sys.executable, "-c", CHILD, str(folder), str(R / "data/06_test.csv")],
                       capture_output=True, text=True, cwd="/")
    if r.returncode != 0:
        C[folder.name] = dict(status="FAIL", error=r.stderr[-800:]); continue
    d = json.loads(r.stdout.strip().splitlines()[-1])
    ok_ref = d["ref_max_abs_diff"] < 1e-4
    ok_test = tm is not None and folder.name in tm.index and abs(d["test_rmse"] - tm.test_rmse[folder.name]) < 1e-6 \
        and abs(d["test_r2"] - tm.test_r2[folder.name]) < 1e-6
    d.update(recorded_test_rmse=float(tm.test_rmse[folder.name]) if tm is not None and folder.name in tm.index else None,
             ref_match=ok_ref, test_metric_match=bool(ok_test), status="PASS" if ok_ref and ok_test else "FAIL")
    C[folder.name] = d
    print(folder.name, d, flush=True)
out["C_model_reload"] = C
out["all_pass"] = A["pass"] and Bres["pass"] and all(v["status"] == "PASS" for v in C.values())
json.dump(out, open(R / "results/08_verification.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "C_model_reload"}, indent=1))
print("ALL PASS:", out["all_pass"])
