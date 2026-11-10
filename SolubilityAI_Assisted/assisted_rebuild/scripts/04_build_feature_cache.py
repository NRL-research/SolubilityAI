"""
04_build_feature_cache.py
=========================
Precompute every feature representation for the corrected v2 split
(train / val / test), cached as .npy under data/features_cache/.

Features are computed per split file independently -- they are pure per-molecule
functions, so this cannot leak anything between splits, and it makes the cache
reusable across every downstream experiment.

Usage: python3 04_build_feature_cache.py [feature_set ...]
       (no args = all sets except mordred; "mordred" must be requested explicitly
        because it takes ~30 min on 2 cores)
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from featurizer import SINGLE_MOL_SETS, featurize  # noqa: E402

SPLIT_DIR = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
os.makedirs(CACHE, exist_ok=True)

SPLITS = ["train", "val", "test"]
requested = sys.argv[1:] or list(SINGLE_MOL_SETS.keys())

for fs in requested:
    for sp in SPLITS:
        out = f"{CACHE}/{sp}__{fs}.npy"
        if os.path.exists(out):
            print(f"skip {out} (exists)")
            continue
        df = pd.read_csv(f"{SPLIT_DIR}/{sp}.csv")
        t0 = time.time()
        X, ok = featurize(df["canonical_smiles"].tolist(), fs)
        assert ok.all(), f"{sp}/{fs}: {int((~ok).sum())} SMILES failed to parse"
        assert X.shape[0] == len(df), "row count mismatch"
        np.save(out, X.astype(np.float32))
        print(f"{sp:5s} {fs:20s} {X.shape}  {time.time()-t0:.1f}s")
print("done")
