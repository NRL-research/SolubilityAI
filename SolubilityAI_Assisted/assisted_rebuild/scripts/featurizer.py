"""
featurizer.py
=============
Feature representations for the corrected SolubilityAI pipeline.

Every feature set is a pure function of a canonical SMILES string -- no fitting, no
dataset-level statistics -- so computing features for train and test separately is
guaranteed not to leak information between them.

Feature sets
------------
solubility_ai        Morgan FP r=2, 2048 bits + [MolWt, MolLogP, TPSA, HBD, HBA].
                     Exactly the feature space used by this project's own XGBoost /
                     CatBoost branches (solubility_ai_v2.ipynb cell 44
                     `extract_classical_features`). 2053 dims.
rdkit_descriptors    All RDKit 2D descriptors from Descriptors.descList.
rdkit_full_eng       rdkit_descriptors + 8 engineered ratio/interaction terms.
maccs                166-bit MACCS keys.
morgan_r3_4096       Morgan FP r=3, 4096 bits.
atom_pair            Hashed atom-pair FP, 2048 bits.
topological_torsion  Hashed topological-torsion FP, 2048 bits.
mordred              ~1600 Mordred 2D descriptors (mordredcommunity fork).
rdkit_desc_morgan    rdkit_descriptors + Morgan r=2 2048.
mordred_rdkit        mordred + rdkit_descriptors.

Non-finite values are replaced with 0 (matching the project's prior convention).
Columns that are constant across the FULL dataset are NOT dropped here -- dropping
is a dataset-level decision and is done inside each model's own pipeline if at all.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

_DESC_LIST = [(n, f) for n, f in Descriptors.descList]
_DESC_NAMES = [n for n, _ in _DESC_LIST]


def _mol(smi):
    return Chem.MolFromSmiles(smi)


def _clean(a):
    """Replace non-finite values with 0 and clip to a magnitude float32 can hold.

    RDKit's Ipc descriptor is an information-content index that grows factorially
    with molecule size and routinely exceeds 1e38 on AqSolDB molecules; without the
    clip it silently becomes +inf when the cache is stored as float32, which then
    poisons every model that reads that column.
    """
    a = np.asarray(a, dtype=np.float64)
    a[~np.isfinite(a)] = 0.0
    return np.clip(a, -1e30, 1e30)


def f_solubility_ai(mol):
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    d = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
         Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
    return np.concatenate([fp, d])


def f_rdkit_descriptors(mol):
    out = []
    for _, fn in _DESC_LIST:
        try:
            out.append(fn(mol))
        except Exception:
            out.append(0.0)
    return np.array(out, dtype=np.float64)


def f_rdkit_full_eng(mol):
    base = f_rdkit_descriptors(mol)
    d = dict(zip(_DESC_NAMES, base))
    mw = d.get("MolWt", 0.0) or 1e-9
    heavy = d.get("HeavyAtomCount", 0.0) or 1e-9
    tpsa = d.get("TPSA", 0.0)
    logp = d.get("MolLogP", 0.0)
    arom = d.get("NumAromaticRings", 0.0)
    rings = d.get("RingCount", 0.0) or 1e-9
    rot = d.get("NumRotatableBonds", 0.0)
    hbd = d.get("NumHDonors", 0.0)
    hba = d.get("NumHAcceptors", 0.0)
    eng = [
        tpsa / mw,               # polar surface density
        logp * mw / 100.0,       # lipophilicity x size (crude crystal-packing proxy)
        (hbd + hba) / heavy,     # H-bond capacity per heavy atom
        arom / rings,            # aromatic ring fraction
        rot / heavy,             # flexibility per heavy atom
        logp - 0.01 * tpsa,      # lipophilicity penalised by polarity
        np.log1p(mw),
        d.get("FractionCSP3", 0.0) * heavy,
    ]
    return np.concatenate([base, np.array(eng, dtype=np.float64)])


def f_maccs(mol):
    return np.array(MACCSkeys.GenMACCSKeys(mol))


def f_morgan_r3_4096(mol):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096))


def f_atom_pair(mol):
    return np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048))


def f_topological_torsion(mol):
    return np.array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048))


SINGLE_MOL_SETS = {
    "solubility_ai": f_solubility_ai,
    "rdkit_descriptors": f_rdkit_descriptors,
    "rdkit_full_eng": f_rdkit_full_eng,
    "maccs": f_maccs,
    "morgan_r3_4096": f_morgan_r3_4096,
    "atom_pair": f_atom_pair,
    "topological_torsion": f_topological_torsion,
}


def featurize(smiles_list, feature_set):
    """Return (X, ok_mask). ok_mask marks SMILES RDKit could parse."""
    if feature_set == "mordred":
        return _featurize_mordred(smiles_list)
    fn = SINGLE_MOL_SETS[feature_set]
    rows, ok = [], []
    for s in smiles_list:
        m = _mol(s)
        if m is None:
            ok.append(False)
            continue
        ok.append(True)
        rows.append(fn(m))
    return _clean(np.vstack(rows)), np.array(ok)


def _featurize_mordred(smiles_list, nproc=2):
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols, ok = [], []
    for s in smiles_list:
        m = _mol(s)
        ok.append(m is not None)
        if m is not None:
            mols.append(m)
    df = calc.pandas(mols, quiet=True, nproc=nproc)
    df = df.apply(lambda c: c.map(lambda v: v if isinstance(v, (int, float, np.floating)) else np.nan))
    return _clean(df.to_numpy(dtype=np.float64)), np.array(ok)
