"""
solubility_featurizer.py
========================
Self-contained featurization for the SolubilityAI saved models.

This file is copied verbatim into every saved model folder. It is the ONLY code
needed to turn a raw SMILES string into the exact input vector a given model
expects, in the exact column order it was trained on.

Usage
-----
    from solubility_featurizer import featurize
    X = featurize(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], "mordred")   # -> (2, 1613) float64

`featurize` canonicalizes each SMILES with RDKit first (the models were trained on
canonical SMILES), computes the descriptors, replaces non-finite values with 0, and
clips to +/-1e30.

Column order
------------
Descriptor order is NOT stable across RDKit / Mordred versions. Each model folder
ships a `feature_names.json` listing the exact ordered column names the model was
trained with. `featurize(..., feature_names=<that list>)` reorders this machine's
descriptor output to match, and raises if a required descriptor is missing. ALWAYS
pass it in production -- without it, a version difference silently shifts columns
and every prediction is wrong while looking perfectly plausible.

There are no fitted preprocessors anywhere in this pipeline: no scaler, no encoder,
no imputer, no feature selection. Every feature is a pure per-molecule function of
the SMILES string. Nothing else needs to be loaded alongside the model.

Requires: rdkit; plus mordredcommunity for the `mordred` feature set only.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

_DESC_LIST = [(n, f) for n, f in Descriptors.descList]

_ENGINEERED = [
    "eng_tpsa_per_mw", "eng_logp_x_mw_over_100", "eng_hbond_per_heavy",
    "eng_aromatic_ring_fraction", "eng_rotatable_per_heavy",
    "eng_logp_minus_0.01_tpsa", "eng_log1p_mw", "eng_fracCSP3_x_heavy",
]


# ---------------------------------------------------------------- helpers
def canonicalize(smiles):
    """Canonical SMILES, or None if RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def _clean(a):
    """Non-finite -> 0, clip to a magnitude float32 can hold, then round-trip
    through float32.

    Two separate reasons, both load-bearing:

    1. RDKit's Ipc descriptor grows factorially with molecule size and routinely
       exceeds 1e38 on drug-like molecules; unclipped it becomes +inf in float32
       and poisons the model input.

    2. The float32 round-trip is NOT cosmetic. Every model in saved_models/ was
       trained on a feature cache that was stored as float32 and widened back to
       float64 at fit time, so the values the trees split on carry float32
       precision. Computing features in full float64 at inference time moves
       values across split thresholds: measured against the training-time inputs
       that shifted predictions by up to 0.166 LogS for HistGB+Mordred and
       0.027 for LightGBM+Morgan. Matching the training precision exactly removes
       that discrepancy. Do not "optimise" this line away.
    """
    a = np.asarray(a, dtype=np.float64)
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a, -1e30, 1e30)
    return a.astype(np.float32).astype(np.float64)


# ---------------------------------------------------------------- feature sets
def _rdkit_descriptor_dict(mol):
    out = {}
    for name, fn in _DESC_LIST:
        try:
            out[name] = float(fn(mol))
        except Exception:
            out[name] = 0.0
    return out


def _engineered_dict(d):
    mw = d.get("MolWt", 0.0) or 1e-9
    heavy = d.get("HeavyAtomCount", 0.0) or 1e-9
    rings = d.get("RingCount", 0.0) or 1e-9
    tpsa = d.get("TPSA", 0.0)
    logp = d.get("MolLogP", 0.0)
    return {
        "eng_tpsa_per_mw": tpsa / mw,
        "eng_logp_x_mw_over_100": logp * mw / 100.0,
        "eng_hbond_per_heavy": (d.get("NumHDonors", 0.0) + d.get("NumHAcceptors", 0.0)) / heavy,
        "eng_aromatic_ring_fraction": d.get("NumAromaticRings", 0.0) / rings,
        "eng_rotatable_per_heavy": d.get("NumRotatableBonds", 0.0) / heavy,
        "eng_logp_minus_0.01_tpsa": logp - 0.01 * tpsa,
        "eng_log1p_mw": float(np.log1p(mw)),
        "eng_fracCSP3_x_heavy": d.get("FractionCSP3", 0.0) * heavy,
    }


def _solubility_ai(mol):
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    d = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
         Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
    return np.concatenate([fp, d])


NAMED_SETS = {"rdkit_descriptors", "rdkit_full_eng", "mordred"}


def feature_names(feature_set):
    """Ordered column names this machine would produce for `feature_set`."""
    if feature_set == "solubility_ai":
        return ([f"morgan_r2_2048_bit_{i:04d}" for i in range(2048)]
                + ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors"])
    if feature_set == "rdkit_descriptors":
        return [n for n, _ in _DESC_LIST]
    if feature_set == "rdkit_full_eng":
        return [n for n, _ in _DESC_LIST] + list(_ENGINEERED)
    if feature_set == "maccs":
        return [f"maccs_bit_{i:03d}" for i in range(167)]
    if feature_set == "morgan_r3_4096":
        return [f"morgan_r3_4096_bit_{i:04d}" for i in range(4096)]
    if feature_set == "atom_pair":
        return [f"atom_pair_bit_{i:04d}" for i in range(2048)]
    if feature_set == "topological_torsion":
        return [f"topological_torsion_bit_{i:04d}" for i in range(2048)]
    if feature_set == "mordred":
        from mordred import Calculator, descriptors as _mdesc
        return [str(d) for d in Calculator(_mdesc, ignore_3D=True).descriptors]
    raise ValueError(f"unknown feature set: {feature_set}")


def featurize(smiles_list, feature_set, feature_names=None, nproc=1):
    """SMILES -> (n_molecules, n_features) float64 array.

    feature_names: the ordered column list from the model's feature_names.json.
        Strongly recommended. When given, output columns are reordered to match it
        exactly and a KeyError is raised if this machine cannot produce one of them.
        Only meaningful for the named-descriptor sets (rdkit_descriptors,
        rdkit_full_eng, mordred); fingerprint bit order is fixed by construction.

    Raises ValueError if any SMILES fails to parse -- silently dropping a molecule
    would misalign the caller's own bookkeeping.
    """
    mols = []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"RDKit could not parse SMILES at index {i}: {s!r}")
        # models were trained on canonical SMILES; re-parse to match exactly
        mols.append(Chem.MolFromSmiles(Chem.MolToSmiles(m)))

    if feature_set == "mordred":
        from mordred import Calculator, descriptors as _mdesc
        calc = Calculator(_mdesc, ignore_3D=True)
        df = calc.pandas(mols, quiet=True, nproc=nproc)
        df = df.apply(lambda c: c.map(
            lambda v: v if isinstance(v, (int, float, np.floating)) else np.nan))
        if feature_names is not None:
            df.columns = [str(c) for c in df.columns]
            missing = [n for n in feature_names if n not in df.columns]
            if missing:
                raise KeyError(f"mordred descriptors missing on this machine: {missing[:5]}"
                               f" ({len(missing)} total) -- version mismatch")
            df = df[list(feature_names)]
        return _clean(df.to_numpy(dtype=np.float64))

    if feature_set in ("rdkit_descriptors", "rdkit_full_eng"):
        rows = []
        for m in mols:
            d = _rdkit_descriptor_dict(m)
            if feature_set == "rdkit_full_eng":
                d.update(_engineered_dict(d))
            if feature_names is not None:
                missing = [n for n in feature_names if n not in d]
                if missing:
                    raise KeyError(f"RDKit descriptors missing on this machine: {missing[:5]}"
                                   f" ({len(missing)} total) -- version mismatch")
                rows.append([d[n] for n in feature_names])
            else:
                order = feature_names_default = ([n for n, _ in _DESC_LIST]
                                                 + (list(_ENGINEERED)
                                                    if feature_set == "rdkit_full_eng" else []))
                rows.append([d[n] for n in order])
        return _clean(np.array(rows, dtype=np.float64))

    simple = {
        "solubility_ai": _solubility_ai,
        "maccs": lambda m: np.array(MACCSkeys.GenMACCSKeys(m)),
        "morgan_r3_4096": lambda m: np.array(
            AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=4096)),
        "atom_pair": lambda m: np.array(
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048)),
        "topological_torsion": lambda m: np.array(
            rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=2048)),
    }
    if feature_set not in simple:
        raise ValueError(f"unknown feature set: {feature_set}")
    return _clean(np.vstack([simple[feature_set](m) for m in mols]))
