"""
predict_blend.py
Reproduce any saved blend's prediction from its member models.

    python3 predict_blend.py greedy_blend "CCO" "CC(=O)Oc1ccccc1C(=O)O"

    from predict_blend import predict_blend
    predict_blend("greedy_blend", ["CCO"])

Combination logic: a plain weighted sum of the members' LogS predictions, with
non-negative weights summing to 1. Each member's own prediction is the average over
that model's saved seeds. No meta-model, no calibration, no blend-level preprocessing.

Members with weight 0 are kept in the config for provenance (they were candidates
the weight fit rejected) and are skipped at inference time.
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.dirname(HERE)      # saved_models/


def _load_member(folder):
    """Import a member model folder's predict.py under its own name."""
    path = os.path.join(MODELS_ROOT, folder, "predict.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"member model folder not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_member_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def predict_blend(blend_name, smiles_list):
    cfg = json.load(open(os.path.join(HERE, f"{blend_name}.json")))
    total = np.zeros(len(smiles_list), dtype=float)
    used = []
    for m in cfg["members"]:
        if m["weight"] <= 1e-9:
            continue
        mod = _load_member(m["folder"])
        total += m["weight"] * np.asarray(mod.predict(smiles_list), dtype=float)
        used.append((m["folder"], m["weight"]))
    if not used:
        raise ValueError(f"{blend_name}: every member has weight 0")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    name, smis = sys.argv[1], sys.argv[2:]
    for s, p in zip(smis, predict_blend(name, smis)):
        print(f"{s}\t{p:.4f}")
