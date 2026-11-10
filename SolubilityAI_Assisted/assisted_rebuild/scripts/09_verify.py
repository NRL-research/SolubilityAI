"""
09_verify.py
============
Independent verification of every structural claim the results rest on. Written to be
run standalone against the saved artefacts, not against in-memory state from the
scripts that produced them.

Checks
------
 1. SC filtering: no InChIKey from SC1/SC2-tight/SC2-loose survives in any split.
 2. Molecular identity: no molecule appears in more than one of train/val/test, by
    canonical SMILES, by InChIKey, and by AqSolDB ID -- not merely by row index.
 3. Scaffold isolation: no split_key (Murcko scaffold, or acyclic singleton) spans
    two splits.
 4. Acyclic coverage: acyclic molecules actually appear in the test set (the bug this
    foundation exists to fix).
 5. CV folds: a true partition of train+val, mutually disjoint validation folds, each
    fold scaffold-respecting, and no test molecule anywhere in the CV pool.
 6. OOF completeness: every OOF prediction file covers every train+val molecule
    exactly once, and its stored y matches the split file's y.
 7. Blend weights: non-negative, sum to 1, and were fitted before test was read
    (checked by file mtime ordering).

Exit code 0 = all checks passed.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DATA = "/home/claude/work/data/v2_corrected"
V1 = "/home/claude/work/data/v1_sc_filtered"
RES = "/home/claude/work/results"
UP = "/mnt/user-data/uploads/SolubilityAI_Assisted"

failures = []
passes = []


def check(name, ok, detail=""):
    (passes if ok else failures).append(f"{name}: {detail}")
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


train = pd.read_csv(f"{DATA}/train.csv")
val = pd.read_csv(f"{DATA}/val.csv")
test = pd.read_csv(f"{DATA}/test.csv")
trainval = pd.read_csv(f"{DATA}/trainval.csv")

# ---- 1. SC leakage
sc = set()
for f in ["SC1_processed.csv", "SC2_tight_processed.csv", "SC2_loose_processed.csv"]:
    sc |= set(pd.read_csv(f"{UP}/{f}")["inchikey"])
resid = {n: len(set(d["inchikey"]) & sc) for n, d in
         [("train", train), ("val", val), ("test", test)]}
check("1. no SC benchmark molecule in any split", sum(resid.values()) == 0, str(resid))

# ---- 2. identity-level disjointness
for col in ["canonical_smiles", "inchikey", "ID"]:
    s = {n: set(d[col]) for n, d in [("train", train), ("val", val), ("test", test)]}
    overlaps = {f"{a}&{b}": len(s[a] & s[b])
                for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    check(f"2. splits disjoint by {col}", sum(overlaps.values()) == 0, str(overlaps))

# ---- 3. scaffold isolation
s = {n: set(d["split_key"]) for n, d in [("train", train), ("val", val), ("test", test)]}
ov = {f"{a}&{b}": len(s[a] & s[b])
      for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
check("3. no scaffold group spans two splits", sum(ov.values()) == 0, str(ov))

# ---- 4. acyclic coverage
ac_test = int(test["split_key"].str.startswith("__acyclic__").sum())
check("4. acyclic molecules present in test", ac_test > 0,
      f"{ac_test} of {len(test)} test molecules are acyclic "
      f"({100*ac_test/len(test):.1f}%)")

# ---- 5. CV folds
folds = json.load(open(f"{DATA}/cv_folds.json"))["folds"]
n = len(trainval)
seen, ok5, msgs = set(), True, []
for i, f in enumerate(folds):
    tr, va = set(f["train"]), set(f["val"])
    if tr & va:
        ok5 = False; msgs.append(f"fold{i} train/val overlap")
    if tr | va != set(range(n)):
        ok5 = False; msgs.append(f"fold{i} not a partition")
    if seen & va:
        ok5 = False; msgs.append(f"fold{i} val reused")
    seen |= va
    ktr = set(trainval.iloc[sorted(tr)]["split_key"])
    kva = set(trainval.iloc[sorted(va)]["split_key"])
    if ktr & kva:
        ok5 = False; msgs.append(f"fold{i} scaffold spans folds")
if seen != set(range(n)):
    ok5 = False; msgs.append("folds do not cover all molecules")
check("5a. CV folds are a scaffold-respecting partition", ok5, "; ".join(msgs) or "5 folds OK")

tv_ids = set(trainval["canonical_smiles"])
check("5b. no test molecule in the CV pool", not (tv_ids & set(test["canonical_smiles"])),
      f"trainval n={len(trainval)}, test n={len(test)}")
check("5c. trainval == train + val exactly",
      len(trainval) == len(train) + len(val) and
      set(trainval["canonical_smiles"]) == set(train["canonical_smiles"]) | set(val["canonical_smiles"]),
      f"{len(train)}+{len(val)}={len(trainval)}")

# ---- 6. OOF completeness
y_tv = trainval["Solubility"].values
bad = []
for f in sorted(glob.glob(f"{RES}/oof/*.npz")):
    d = np.load(f, allow_pickle=True)
    if np.isnan(d["oof"]).any():
        bad.append(os.path.basename(f) + " has NaNs")
    if not np.allclose(d["y"], y_tv):
        bad.append(os.path.basename(f) + " y mismatch")
gnn = sorted(glob.glob(f"{RES}/gnn/oof_fold*_seed42.npz"))
if gnn:
    cov = np.zeros(len(trainval), dtype=int)
    for f in gnn:
        d = np.load(f)
        cov[d["idx"]] += 1
        if not np.allclose(d["y"], y_tv[d["idx"]]):
            bad.append(os.path.basename(f) + " y mismatch")
    if not (cov == 1).all():
        bad.append(f"GATv2 OOF coverage not exactly 1x (min={cov.min()}, max={cov.max()})")
check("6. OOF predictions complete and aligned to the split file", not bad, "; ".join(bad) or "all OOF files OK")

# ---- 7. blend weights
wpath = f"{RES}/blend_weights.json"
if os.path.exists(wpath):
    w = json.load(open(wpath))
    ok7 = all(abs(sum(v["weights"]) - 1) < 1e-6 and min(v["weights"]) >= -1e-9
              for v in w.values())
    check("7a. blend weights non-negative and sum to 1", ok7, str({k: [round(x, 4) for x in v["weights"]] for k, v in w.items()}))
    tp = f"{RES}/test_predictions.npz"
    if os.path.exists(tp):
        check("7b. blend weights fitted before test predictions were produced",
              os.path.getmtime(wpath) < os.path.getmtime(tp),
              f"weights mtime {os.path.getmtime(wpath):.0f} < test preds mtime {os.path.getmtime(tp):.0f}")

print(f"\n{len(passes)} passed, {len(failures)} failed")
if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
sys.exit(1 if failures else 0)
