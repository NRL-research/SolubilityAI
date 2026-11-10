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

# ---------------------------------------------------------------- Mordred 2D descriptors (lazy import)
_MORDRED = None
def mordred_desc(mols, names):
    """Mordred (mordredcommunity) 2D descriptors, ignore_3D=True, restricted/ordered to `names`
    (saved in config.json).  Errors/missing -> nan; |x| clipped at 1e6."""
    global _MORDRED
    from mordred import Calculator, descriptors as mdesc
    if _MORDRED is None:
        _MORDRED = Calculator(mdesc, ignore_3D=True)
    all_names = [str(d) for d in _MORDRED.descriptors]
    pos = [all_names.index(n) for n in names]
    X = np.full((len(mols), len(names)), np.nan)
    for i, m in enumerate(mols):
        if m is None:
            continue
        r = _MORDRED(m)
        v = np.array([float(x) if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) else
                      (float(x) if isinstance(x, bool) else np.nan) for x in list(r)], dtype=float)
        X[i] = v[pos]
    X[~np.isfinite(X)] = np.nan
    return np.clip(X, -1e6, 1e6)

# ---------------------------------------------------------------- feature sets by name
def featurize(smiles_list, feature_set, desc_names=None, mordred_names=None):
    mols = [standardize(s) for s in smiles_list]
    if feature_set == "mordred":
        return mordred_desc(mols, mordred_names)
    if feature_set == "desc+mordred":
        return np.hstack([descriptors(mols, desc_names), mordred_desc(mols, mordred_names)])
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
    if feature_set.startswith("r2:"):
        return featurize_r2(smiles_list, feature_set[3:], desc_names, mordred_names)
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


# ================================================================ ROUND 2 additions (2026-10)
# Nothing above this line was changed; models saved in round 1 are unaffected.
from pathlib import Path as _Path
from rdkit.Avalon import pyAvalonTools as _Avalon
from rdkit.Chem import rdReducedGraphs as _ErG

def atompair_counts(mols, n_bits=2048):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None: X[i] = gen.GetCountFingerprintAsNumPy(m)
    return X

def torsion_counts(mols, n_bits=2048):
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None: X[i] = gen.GetCountFingerprintAsNumPy(m)
    return X

def avalon_counts(mols, n_bits=1024):
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            fp = _Avalon.GetAvalonCountFP(m, nBits=n_bits)
            for k, v in fp.GetNonzeroElements().items(): X[i, k] = v
    return X

def erg(mols):
    X = np.zeros((len(mols), 315), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None: X[i] = _ErG.GetErGFingerprint(m)
    return X

# Crippen (Wildman-Crippen) atom-type counts: a group-contribution style representation
CRIPPEN_TYPES = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','CS','H1','H2','H3','H4','HS','N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13','N14','NS','O1','O2','O3','O4','O5','O6','O7','O8','O9','O10','O11','O12','OS','F','Cl','Br','I','Hal','P','S1','S2','S3','Me1','Me2']
def crippen_types(mols):
    idx = {t: j for j, t in enumerate(CRIPPEN_TYPES)}
    X = np.zeros((len(mols), len(CRIPPEN_TYPES) + 1), dtype=np.float32)   # last column = unknown type
    for i, m in enumerate(mols):
        if m is None: continue
        mh = Chem.AddHs(m)
        types, labels = [], []
        rdMolDescriptors._CalcCrippenContribs(mh, force=True, atomTypes=types, atomTypeLabels=labels)
        for t in types: X[i, idx.get(t, len(CRIPPEN_TYPES))] += 1
    return X

PACKING_DESC = ["p_sym_classes_frac", "p_n_sym_classes", "p_arom_atom_frac", "p_conj_bond_frac", "p_fused_ring_atoms",
                "p_largest_conj_system", "p_amide_like", "p_hbd_x_hba", "p_rigid_frac", "p_ring_systems"]
def packing(mols):
    """Crystal-packing / melting-point proxies (solubility ~ -MP/100 - logP in the General Solubility
    Equation; MP itself is unavailable, so these structure-only proxies stand in for it)."""
    X = np.full((len(mols), len(PACKING_DESC)), np.nan)
    for i, m in enumerate(mols):
        if m is None: continue
        heavy = m.GetNumHeavyAtoms() or 1
        ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
        nsym = len(set(ranks))
        bonds = m.GetNumBonds() or 1
        ri = m.GetRingInfo()
        atom_ring_count = [0] * m.GetNumAtoms()
        for ring in ri.AtomRings():
            for a in ring: atom_ring_count[a] += 1
        fused = sum(1 for c in atom_ring_count if c > 1)
        # largest conjugated system (connected component over conjugated bonds)
        par = list(range(m.GetNumAtoms()))
        def f(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for b in m.GetBonds():
            if b.GetIsConjugated(): par[f(b.GetBeginAtomIdx())] = f(b.GetEndAtomIdx())
        comp = {}
        for b in m.GetBonds():
            if b.GetIsConjugated():
                for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): comp.setdefault(f(a), set()).add(a)
        largest = max((len(v) for v in comp.values()), default=0)
        amide = len(m.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H1,H2][CX3]=[OX1]")))
        rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        # ring systems = connected components of ring atoms
        par2 = list(range(m.GetNumAtoms()))
        def g(x):
            while par2[x] != x: par2[x] = par2[par2[x]]; x = par2[x]
            return x
        for ring in ri.AtomRings():
            for a in ring[1:]: par2[g(a)] = g(ring[0])
        nsys = len({g(a) for ring in ri.AtomRings() for a in ring})
        X[i] = [nsym / heavy, nsym, sum(a.GetIsAromatic() for a in m.GetAtoms()) / heavy,
                sum(b.GetIsConjugated() for b in m.GetBonds()) / bonds, fused, largest, amide,
                rdMolDescriptors.CalcNumHBD(m) * rdMolDescriptors.CalcNumHBA(m), 1 - rot / bonds, nsys]
    return X

MOL2VEC_PATH = _Path(__file__).resolve().parent / "mol2vec_vectors.npz"
_M2V = None
def mol2vec(mols, radius=1):
    """Mol2vec (Jaeger et al. 2018) pretrained 300-d embedding: sum of word vectors of the unhashed
    Morgan identifiers (radius 0 and 1, per atom, in atom order); unknown identifiers -> 'UNK'.
    Vectors: mol2vec_vectors.npz (float16 copy of the published model_300dim.pkl)."""
    global _M2V
    if _M2V is None:
        z = np.load(MOL2VEC_PATH, allow_pickle=False)
        _M2V = ({k: i for i, k in enumerate(z["keys"])}, z["vectors"].astype(np.float32))
    vocab, V = _M2V
    unk = V[vocab["UNK"]]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    X = np.zeros((len(mols), V.shape[1] + 1), dtype=np.float32)   # + coverage fraction
    for i, m in enumerate(mols):
        if m is None: continue
        ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
        gen.GetSparseCountFingerprint(m, additionalOutput=ao)
        per_atom = {}
        for ident, hits in ao.GetBitInfoMap().items():
            for atom, rad in hits:
                per_atom.setdefault(atom, {})[rad] = ident
        words = []
        for atom in sorted(per_atom):
            for rad in range(radius + 1):
                if rad in per_atom[atom]: words.append(str(per_atom[atom][rad]))
        hit = [vocab.get(w) for w in words]
        known = [h for h in hit if h is not None]
        X[i, :-1] = V[known].sum(0) + unk * (len(hit) - len(known))
        X[i, -1] = len(known) / max(1, len(hit))
    return X

R2_BLOCKS = {"atompair": atompair_counts, "torsion": torsion_counts, "avalon": avalon_counts, "erg": erg,
             "crippen_types": crippen_types, "packing": packing, "mol2vec": mol2vec}

def featurize_r2(smiles_list, feature_set, desc_names=None, mordred_names=None):
    """Round-2 feature sets: '+'-joined block names, e.g. 'desc+mordred+crippen_types+packing'.
    Blocks: desc (RDKit, desc_names order), mordred (mordred_names order), morgan (r2 2048 counts),
    maccs, and every key of R2_BLOCKS."""
    mols = [standardize(s) for s in smiles_list]
    out = []
    for b in feature_set.split("+"):
        if b == "desc": out.append(descriptors(mols, desc_names))
        elif b == "mordred": out.append(mordred_desc(mols, mordred_names))
        elif b == "morgan": out.append(morgan_counts(mols, 2, 2048))
        elif b == "maccs": out.append(maccs(mols))
        else: out.append(R2_BLOCKS[b](mols))
    return np.hstack([np.asarray(x, dtype=np.float64) for x in out])

# ---------------------------------------------------------------- ROUND 2 bug fix (2026-10-02)
# rdReducedGraphs.GetErGFingerprint is NOT state-free: its output depends on properties cached on
# the Mol object by earlier calculations (e.g. RDKit descriptors / atom-pair fingerprints), and the
# result was not reproducible across processes.  erg2 computes ErG on a pristine re-parsed copy of
# the molecule (canonical SMILES -> Mol), which is deterministic and independent of call order.
# The original 'erg' block above is kept unchanged so the round-2 models that used it still load.
def erg2(mols):
    X = np.zeros((len(mols), 315), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = _ErG.GetErGFingerprint(Chem.MolFromSmiles(Chem.MolToSmiles(m)))
    return X
R2_BLOCKS["erg2"] = erg2
