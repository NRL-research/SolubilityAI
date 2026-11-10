"""Stacked-ensemble predictor (round 2).  Copied into each saved_models/stack_* folder.

prediction = stacker( [component predictions..., meta features...] )
  * components : saved sibling models, each loaded through its own predict.py
  * meta features (if any): RDKit descriptors listed in config.json:meta_desc (computed with this
    folder's featurize.py) and 'nn_sim' = max Tanimoto (Morgan r2 2048 bits, parent structure)
    to the training molecules stored in train_parent_fps.npz
  * stacker: ridge (stacker.joblib, sklearn) or LightGBM (stacker.lgbm.txt), see config.json
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

def nn_similarity(smiles, fps_file):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    bits = np.unpackbits(np.load(fps_file)["packed"], axis=1)[:, :2048]
    train = []
    for row in bits:
        bv = DataStructs.ExplicitBitVect(2048); bv.SetBitsFromList(np.where(row)[0].tolist()); train.append(bv)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.full(len(smiles), np.nan)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None: continue
        p = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(rdMolStandardize.Cleanup(m)))
        out[i] = max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(p), train))
    return out

def load_stack(folder):
    folder = Path(folder)
    cfg = json.load(open(folder / "config.json"))
    comps = [_load_module(folder.parent / c / "predict.py", f"predict_{c}").load_predictor(folder.parent / c) for c in cfg["components"]]
    F = _load_module(folder / "featurize.py", f"featurize_stack_{folder.name}")
    if cfg["stacker"] == "ridge":
        import joblib; st = joblib.load(folder / "stacker.joblib"); f = st.predict
    else:
        import lightgbm as lgb; st = lgb.Booster(model_file=str(folder / "stacker.lgbm.txt")); f = st.predict
    def predict(smiles):
        smiles = list(smiles)
        cols = [c(smiles) for c in comps]
        if cfg.get("meta_desc"):
            cols += list(F.featurize(smiles, "rdkit_desc", cfg["meta_desc"]).T)
        if cfg.get("meta_nn_sim"):
            cols.append(nn_similarity(smiles, folder / "train_parent_fps.npz"))
        X = np.column_stack(cols)
        y = np.asarray(f(X), dtype=float)
        bad = ~np.isfinite(np.column_stack(cols[:len(comps)])).all(1)
        y[bad] = np.nan
        return y
    return predict
