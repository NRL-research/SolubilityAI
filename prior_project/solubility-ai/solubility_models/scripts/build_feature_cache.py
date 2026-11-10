"""
Precompute and cache feature matrices for every split x feature_set combo.
Feature computation is deterministic given SMILES -- no reason to recompute
it on every Optuna trial during the sweep. Saved as .npy under
solubility_models/data/features_cache/.

Usage:
  python solubility_models/scripts/build_feature_cache.py            # ALL_FEATURE_SETS (original 5 + expanded)
  python solubility_models/scripts/build_feature_cache.py --original-only   # just the original 5
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from solubility_models.featurizer import build_feature_matrix, FEATURE_SETS, ALL_FEATURE_SETS

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-only", action="store_true",
                         help="Only build the original 5 feature sets, skip the expanded round.")
    parser.add_argument("--feature-sets", nargs="*", default=None,
                         help="Explicit list of feature sets to build (overrides --original-only).")
    args = parser.parse_args()

    if args.feature_sets:
        feature_sets = args.feature_sets
    elif args.original_only:
        feature_sets = FEATURE_SETS
    else:
        feature_sets = ALL_FEATURE_SETS

    os.makedirs(CACHE_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(SPLIT_DIR, f"{split}.csv"))
        smiles = df["canonical_smiles"].tolist()
        y = df["Solubility"].values.astype(np.float32)
        np.save(os.path.join(CACHE_DIR, f"{split}_y.npy"), y)
        print(f"[{split}] {len(smiles)} molecules")

        for fs in feature_sets:
            out_path = os.path.join(CACHE_DIR, f"{split}_{fs.replace('+', '_')}.npy")
            if os.path.exists(out_path):
                print(f"  {fs}: cached, skipping")
                continue
            t0 = time.time()
            X = build_feature_matrix(smiles, fs)
            np.save(out_path, X)
            print(f"  {fs}: shape={X.shape}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
