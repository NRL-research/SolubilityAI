"""Self-contained featurization: raw SMILES -> model input.

This exact file is copied into every saved_models/<model>/ folder.  It depends only on
rdkit + numpy (+ torch for graph featurization, imported lazily).

Pipeline for every model:
  1. standardize(smiles): RDKit parse -> rdMolStandardize.Cleanup (counterions are KEPT;
     the full salt/mixture is the model input).
  2. one of the feature functions below.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- standardization
def standardize(smiles):
    """Return standardized RDKit Mol (or None if unparseable)."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        return rdMolStandardize.Cleanup(m)
    except Exception:
        return None

# ---------------------------------------------------------------- fingerprints
def morgan_counts(mols, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = gen.GetCountFingerprintAsNumPy(m)
    return X

def morgan_bits(mols, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = gen.GetFingerprintAsNumPy(m)
    return X

def maccs(mols):
    X = np.zeros((len(mols), 167), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
    return X

# ---------------------------------------------------------------- descriptors
# Full RDKit 2D descriptor list (order fixed by name list saved with each model) plus a few
# salt/mixture-aware extras.  Ipc is excluded (numerically pathological, spans 1e0..1e30+).
RDKIT_DESC = [n for n, _ in Descriptors._descList if n != "Ipc"]
_DESC_FN = dict(Descriptors._descList)
EXTRA_DESC = ["x_n_components", "x_n_pos_charge", "x_n_neg_charge", "x_has_metal",
              "x_heavy_largest_frac", "x_logp_largest", "x_mw_largest"]
_METALS = {3, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 42,
           44, 45, 46, 47, 48, 50, 55, 56, 57, 72, 74, 78, 79, 80, 81, 82, 83}

def _extras(m):
    frags = Chem.GetMolFrags(m, asMols=True)
    big = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    return [len(frags),
            sum(a.GetFormalCharge() > 0 for a in m.GetAtoms()),
            sum(a.GetFormalCharge() < 0 for a in m.GetAtoms()),
            float(any(a.GetAtomicNum() in _METALS for a in m.GetAtoms())),
            big.GetNumHeavyAtoms() / max(1, m.GetNumHeavyAtoms()),
            Crippen.MolLogP(big), Descriptors.MolWt(big)]

def descriptors(mols, names=None):
    """Returns float64 matrix; inf -> nan, |x| clipped at 1e6.  NaNs are handled by each
    model's saved preprocessor (median imputation) or natively by GBMs."""
    names = names or (RDKIT_DESC + EXTRA_DESC)
    X = np.full((len(mols), len(names)), np.nan)
    for i, m in enumerate(mols):
        if m is None:
            continue
        ex = dict(zip(EXTRA_DESC, _extras(m)))
        for j, n in enumerate(names):
            try:
                X[i, j] = ex[n] if n in ex else _DESC_FN[n](m)
            except Exception:
                X[i, j] = np.nan
    X[~np.isfinite(X)] = np.nan
    return np.clip(X, -1e6, 1e6)

ESOL_DESC = ["MolLogP", "MolWt", "NumRotatableBonds", "x_aromatic_proportion"]
def esol(mols):
    """Delaney-style 4-descriptor set."""
    X = np.zeros((len(mols), 4))
    for i, m in enumerate(mols):
        if m is None:
            X[i] = np.nan; continue
        ha = m.GetNumHeavyAtoms()
        X[i] = [Crippen.MolLogP(m), Descriptors.MolWt(m), Lipinski.NumRotatableBonds(m),
                sum(a.GetIsAromatic() for a in m.GetAtoms()) / max(1, ha)]
    return X

# ---------------------------------------------------------------- feature sets by name
def featurize(smiles_list, feature_set, desc_names=None):
    mols = [standardize(s) for s in smiles_list]
    if feature_set == "morgan_counts_r2_2048":
        return morgan_counts(mols, 2, 2048)
    if feature_set == "morgan_bits_r2_2048":
        return morgan_bits(mols, 2, 2048)
    if feature_set == "maccs":
        return maccs(mols)
    if feature_set == "rdkit_desc":
        return descriptors(mols, desc_names)
    if feature_set == "esol4":
        return esol(mols)
    if feature_set == "desc+morgan_counts_r2_2048":
        return np.hstack([descriptors(mols, desc_names), morgan_counts(mols, 2, 2048)])
    if feature_set == "desc+maccs":
        return np.hstack([descriptors(mols, desc_names), maccs(mols)])
    raise ValueError(feature_set)

# ---------------------------------------------------------------- molecular graphs (GNN)
ATOM_LIST = [1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 19, 35, 53]  # else -> 'other'
def _onehot(x, choices):
    v = [0.0] * (len(choices) + 1)
    v[choices.index(x) if x in choices else len(choices)] = 1.0
    return v

HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]
BT = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
      Chem.rdchem.BondType.AROMATIC]

def atom_features(a):
    return (_onehot(a.GetAtomicNum(), ATOM_LIST) + _onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) +
            _onehot(a.GetFormalCharge(), [-1, 0, 1]) + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3]) +
            _onehot(a.GetHybridization(), HYB) +
            [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0])

def bond_features(b):
    return _onehot(b.GetBondType(), BT) + [float(b.GetIsConjugated()), float(b.IsInRing())]

N_ATOM_FEAT = len(ATOM_LIST) + 1 + 7 + 4 + 5 + 6 + 3
N_BOND_FEAT = len(BT) + 1 + 2

def mol_to_graph(m):
    x = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, ea = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_features(b)
        src += [i, j]; dst += [j, i]; ea += [f, f]
    ei = np.array([src, dst], dtype=np.int64).reshape(2, -1)
    ea = np.array(ea, dtype=np.float32).reshape(-1, N_BOND_FEAT)
    return x, ei, ea

def graphs(smiles_list):
    return [mol_to_graph(standardize(s)) for s in smiles_list]
