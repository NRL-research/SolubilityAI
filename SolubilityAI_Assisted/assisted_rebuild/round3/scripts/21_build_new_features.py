"""
21_build_new_features.py
========================
Build the round-3 feature caches: the four new per-molecule representations from
featurizer2.py, plus concatenations of existing cached sets.

Concatenations are built from the existing float32 caches so a concatenated set is
bit-identical to what the component models saw.

Usage: python3 21_build_new_features.py [set ...]   (default: all)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurizer2 import NEW_SETS, featurize_new, new_feature_names  # noqa: E402
from solubility_featurizer import feature_names as base_names  # noqa: E402

SPLIT_DIR = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
NAMES = "/home/claude/work/data/feature_name_registry.json"
SPLITS = ["train", "val", "test"]

# concatenations: new set name -> component cached sets, in order
CONCATS = {
    "mordred+chem_aware": ["mordred", "chem_aware"],
    "rdkit_full_eng+chem_aware": ["rdkit_full_eng", "chem_aware"],
    "mordred+morgan": ["mordred", "solubility_ai"],
    "mordred+rdkit_full_eng": ["mordred", "rdkit_full_eng"],
    "mordred+avalon": ["mordred", "avalon"],
    "mordred+chem_aware+avalon": ["mordred", "chem_aware", "avalon"],
}

registry = json.load(open(NAMES)) if os.path.exists(NAMES) else {}


def names_for(fs):
    if fs in NEW_SETS:
        return new_feature_names(fs)
    if fs in CONCATS:
        out = []
        for c in CONCATS[fs]:
            out += [f"{c}::{n}" for n in names_for(c)]
        return out
    return base_names(fs)


requested = sys.argv[1:] or list(NEW_SETS) + list(CONCATS)

for fs in requested:
    for sp in SPLITS:
        out = f"{CACHE}/{sp}__{fs}.npy"
        if os.path.exists(out):
            print(f"skip {sp}/{fs} (exists)")
            continue
        t0 = time.time()
        if fs in CONCATS:
            parts = []
            for c in CONCATS[fs]:
                p = f"{CACHE}/{sp}__{c}.npy"
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{p} missing -- build component '{c}' first")
                parts.append(np.load(p))
            X = np.hstack(parts)
        else:
            df = pd.read_csv(f"{SPLIT_DIR}/{sp}.csv")
            X = featurize_new(df["canonical_smiles"].tolist(), fs)
            assert X.shape[0] == len(df)
        np.save(out, X.astype(np.float32))
        print(f"{sp:5s} {fs:28s} {X.shape}  {time.time()-t0:.1f}s", flush=True)
    registry[fs] = names_for(fs)
    assert len(registry[fs]) == np.load(f"{CACHE}/train__{fs}.npy").shape[1], \
        f"{fs}: name count != column count"

json.dump(registry, open(NAMES, "w"))
print(f"\nfeature-name registry: {len(registry)} sets -> {NAMES}")
