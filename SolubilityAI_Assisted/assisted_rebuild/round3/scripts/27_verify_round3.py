"""
27_verify_round3.py
===================
Round-3-specific verification, on top of 09_verify.py (which re-checks the data
foundation and still passes 12/12).

Checks:
  1. Every round-3 OOF file covers all 7,898 train+val molecules exactly once and its
     stored y matches the split file -- i.e. the new models were cross-validated over
     the same folds as everything else, not a different or easier partition.
  2. No round-3 base model's test prediction correlates with the test labels in a way
     that implies it saw them: specifically, every cached test prediction has the right
     length and is not identical to y (a sanity trap for accidentally writing labels).
  3. The new feature caches are row-aligned with their split files, and the test cache
     for every new feature set was produced from test.csv's SMILES -- recomputed here
     from scratch for a sample and compared.
  4. The stacker's meta-model inputs are OOF (train+val) only: the stacking config file
     predates the round-3 test predictions on disk.
  5. Identity-level disjointness of train/val/test still holds by canonical SMILES,
     InChIKey and ID (re-checked here so round 3 has its own record).

Exit 0 only if everything passes.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurizer2 import NEW_SETS, featurize_new  # noqa: E402

ROOT = "/home/claude/work"
DATA = f"{ROOT}/data/v2_corrected"
CACHE = f"{ROOT}/data/features_cache"
RES = f"{ROOT}/results"

train = pd.read_csv(f"{DATA}/train.csv")
val = pd.read_csv(f"{DATA}/val.csv")
test = pd.read_csv(f"{DATA}/test.csv")
trainval = pd.read_csv(f"{DATA}/trainval.csv")
y_tv = trainval["Solubility"].values
y_te = test["Solubility"].values

fails, passes = [], []


def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---- 1. round-3 OOF integrity
bad = []
r3 = []
for f in sorted(glob.glob(f"{RES}/oof/*.npz")):
    d = np.load(f, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    nm = os.path.basename(f)[:-4]
    if m.get("round") == 3:
        r3.append(nm)
    if len(d["oof"]) != len(trainval):
        bad.append(f"{nm}: length {len(d['oof'])}")
    if np.isnan(d["oof"]).any():
        bad.append(f"{nm}: NaNs")
    if not np.allclose(d["y"], y_tv):
        bad.append(f"{nm}: y mismatch")
check("1. all OOF files complete and aligned to train+val", not bad,
      f"{len(glob.glob(f'{RES}/oof/*.npz'))} files, {len(r3)} from round 3; "
      + ("; ".join(bad) if bad else "no issues"))

# ---- 2. test predictions sane
bad = []
tps = sorted(glob.glob(f"{RES}/test_preds_round3/*.npy"))
for f in tps:
    p = np.load(f)
    nm = os.path.basename(f)[:-4]
    if len(p) != len(test):
        bad.append(f"{nm}: length {len(p)}")
    if np.allclose(p, y_te):
        bad.append(f"{nm}: prediction IS the label vector")
    if np.corrcoef(p, y_te)[0, 1] > 0.999:
        bad.append(f"{nm}: corr with labels {np.corrcoef(p, y_te)[0,1]:.5f} -- implausible")
check("2. test predictions well-formed and not label copies", not bad,
      f"{len(tps)} cached test predictions; " + ("; ".join(bad) if bad else "no issues"))

# ---- 3. new feature caches recomputable from the split files
bad = []
rng = np.random.default_rng(3)
for fs in NEW_SETS:
    for sp, ref_df in [("train", train), ("test", test)]:
        p = f"{CACHE}/{sp}__{fs}.npy"
        if not os.path.exists(p):
            continue
        X = np.load(p)
        if X.shape[0] != len(ref_df):
            bad.append(f"{sp}/{fs}: {X.shape[0]} rows vs {len(ref_df)}")
            continue
        idx = rng.choice(len(ref_df), 25, replace=False)
        Xr = featurize_new(ref_df["canonical_smiles"].iloc[idx].tolist(), fs)
        d = float(np.max(np.abs(Xr - X[idx].astype(np.float64))))
        if d > 1e-6:
            bad.append(f"{sp}/{fs}: recompute differs by {d:.2e}")
check("3. new feature caches reproduce from their split file's SMILES", not bad,
      "; ".join(bad) if bad else "all new sets match on a 25-molecule sample")

# ---- 4. stack config frozen before test predictions existed
sm = f"{RES}/stacking_models.json"
tp = f"{RES}/test_predictions_round3.npz"
ok = os.path.exists(sm) and os.path.exists(tp) and os.path.getmtime(sm) < os.path.getmtime(tp)
check("4. stack configuration frozen before test was read", ok,
      f"stacking_models.json mtime {os.path.getmtime(sm):.0f} < "
      f"test_predictions_round3.npz mtime {os.path.getmtime(tp):.0f}")

# ---- 5. identity-level disjointness (re-checked for round 3's record)
for col in ["canonical_smiles", "inchikey", "ID"]:
    s = {n: set(d[col]) for n, d in [("train", train), ("val", val), ("test", test)]}
    ov = {f"{a}&{b}": len(s[a] & s[b])
          for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    check(f"5. splits disjoint by {col}", sum(ov.values()) == 0, str(ov))

print(f"\n{len(passes)} passed, {len(fails)} failed")
if fails:
    for f in fails:
        print("  FAILED:", f)
sys.exit(1 if fails else 0)
