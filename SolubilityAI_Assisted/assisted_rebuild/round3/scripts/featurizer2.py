"""
featurizer2.py
==============
New feature representations for round 3, motivated directly by the OOF error analysis
(scripts/20_error_analysis.py, results/error_analysis.json).

What the error analysis found, and what each new feature set does about it:

  Inorganic / metal-containing molecules are 9.5% of train+val but carry 27.2% of the
  squared error (RMSE 1.71 vs 0.91 for purely organic). Mordred and most RDKit
  descriptors are either undefined or silently zero for a bare metal or a metal salt,
  so the model has no way to even tell "this is an inorganic species" apart from a
  vector of zeros that also matches a small organic molecule.

  Salts and multi-component mixtures are 11.2% of the data and carry 25.3% of the
  error (RMSE 1.53 vs 0.93). Whole-molecule descriptors of a 14-fragment hydrate are
  close to meaningless; the chemistry lives in the largest organic fragment plus the
  identity of the counter-ions.

  Charged species carry 25.7% of the error.

-> `chem_aware` (58 features): explicit, cheap, interpretable features for exactly
   these three situations -- fragment structure, element composition, charge balance,
   and descriptors recomputed on the LARGEST ORGANIC FRAGMENT rather than the whole
   multi-component record. Intended as a supplement to a descriptor set, not a
   replacement.

Also here, as untried representations:
  `avalon`          Avalon fingerprint, 1024 bits (different substructure enumeration
                    from Morgan; often complementary on physical properties).
  `morgan_count`    Morgan r=2 2048 as COUNTS rather than presence/absence -- tree
                    models can split on "how many" instead of only "whether".
  `rdkit_fp`        RDKit's own path-based (Daylight-like) fingerprint, 2048 bits.

Every function is a pure per-molecule function of the SMILES string. No fitted state.
Values are cleaned identically to solubility_featurizer._clean (non-finite -> 0, clip
to +/-1e30, float32 round-trip) so anything trained here matches its inference path.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools

RDLogger.DisableLog("rdApp.*")

ORGANIC = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"}

# elements given their own count column: common counter-ions and metals in AqSolDB
TRACKED_ELEMENTS = ["Na", "K", "Ca", "Mg", "Al", "Fe", "Zn", "Cu", "Ba", "Li",
                    "Mn", "Ni", "Cr", "Sn", "Pb", "Ti", "Ag", "Hg", "Cd", "Co"]

HALIDES = {"F", "Cl", "Br", "I"}


def _clean(a):
    a = np.asarray(a, dtype=np.float64)
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a, -1e30, 1e30)
    return a.astype(np.float32).astype(np.float64)


def _prepare_fragment(f):
    """Make a fragment safe for descriptor calls.

    Fragments come out of GetMolFrags(sanitizeFrags=False) -- necessary, because
    sanitizing throws on plenty of the inorganic records in AqSolDB (bare metals,
    hypervalent Sb/As oxides) and those are exactly the molecules this feature set
    exists to describe. But an unsanitized fragment has no RingInfo, so RingCount and
    anything ring-based raises "RingInfo not initialized". Sanitize when possible,
    and fall back to FastFindRings, which populates ring info without the valence
    checks that fail on these species.
    """
    try:
        Chem.SanitizeMol(f)
    except Exception:
        try:
            Chem.rdmolops.FastFindRings(f)
        except Exception:
            pass
    return f


def _largest_organic_fragment(mol):
    """The fragment with the most carbon atoms (ties broken by heavy-atom count).

    For a salt like `O=S(=O)([O-])c1ccc(O)cc1.[Na+]` this returns the sulfonate,
    which is the species whose structure actually governs solubility; the sodium is
    captured separately by the element-count features.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return _prepare_fragment(frags[0])

    def key(f):
        c = sum(1 for a in f.GetAtoms() if a.GetSymbol() == "C")
        return (c, f.GetNumHeavyAtoms())
    return _prepare_fragment(max(frags, key=key))


def _safe(fn, mol, default=0.0):
    try:
        v = fn(mol)
        return float(v) if np.isfinite(v) else default
    except Exception:
        return default


CHEM_AWARE_NAMES = (
    # fragment structure
    ["n_fragments", "n_heavy_total", "n_heavy_largest", "largest_frag_heavy_fraction",
     "n_distinct_fragments", "n_water_fragments", "n_single_atom_fragments"]
    # charge
    + ["net_formal_charge", "n_positive_atoms", "n_negative_atoms",
       "sum_positive_charge", "sum_negative_charge", "is_charged", "is_charge_balanced"]
    # composition
    + ["n_carbon", "carbon_fraction", "n_hetero", "hetero_fraction",
       "n_nonorganic_atoms", "is_inorganic", "has_metal", "n_distinct_elements",
       "n_halide_atoms", "n_metal_atoms", "max_atomic_number", "mean_atomic_number"]
    + [f"count_{e}" for e in TRACKED_ELEMENTS]
    # descriptors of the largest organic fragment
    + ["lf_MolWt", "lf_MolLogP", "lf_TPSA", "lf_NumHDonors", "lf_NumHAcceptors",
       "lf_RingCount", "lf_NumAromaticRings", "lf_NumRotatableBonds",
       "lf_FractionCSP3", "lf_HeavyAtomCount", "lf_NumHeteroatoms",
       "lf_MolMR", "lf_LabuteASA", "lf_BalabanJ", "lf_logp_per_heavy",
       "lf_tpsa_per_heavy"]
)


def f_chem_aware(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    smis = [Chem.MolToSmiles(f) for f in frags]
    atoms = list(mol.GetAtoms())
    syms = [a.GetSymbol() for a in atoms]
    charges = [a.GetFormalCharge() for a in atoms]
    nums = [a.GetAtomicNum() for a in atoms]
    lf = _largest_organic_fragment(mol)

    n_heavy = mol.GetNumHeavyAtoms() or 1
    n_c = sum(1 for s in syms if s == "C")
    nonorg = [s for s in syms if s not in ORGANIC]
    metals = [s for s in syms if s in TRACKED_ELEMENTS]

    lf_heavy = lf.GetNumHeavyAtoms() or 1
    lf_logp = _safe(Crippen.MolLogP, lf)
    lf_tpsa = _safe(Descriptors.TPSA, lf)
    lf_balaban = _safe(Descriptors.BalabanJ, lf)
    lf_molmr = _safe(Crippen.MolMR, lf)
    lf_labute = _safe(Descriptors.LabuteASA, lf)

    vals = [
        len(frags), mol.GetNumHeavyAtoms(), lf.GetNumHeavyAtoms(),
        lf.GetNumHeavyAtoms() / n_heavy, len(set(smis)),
        sum(1 for s in smis if s in ("O", "[OH2]")),
        sum(1 for f in frags if f.GetNumHeavyAtoms() == 1),

        sum(charges), sum(1 for c in charges if c > 0), sum(1 for c in charges if c < 0),
        sum(c for c in charges if c > 0), sum(c for c in charges if c < 0),
        float(any(c != 0 for c in charges)),
        float(sum(charges) == 0 and any(c != 0 for c in charges)),

        n_c, n_c / n_heavy, n_heavy - n_c, (n_heavy - n_c) / n_heavy,
        len(nonorg), float(n_c == 0), float(len(metals) > 0), len(set(syms)),
        sum(1 for s in syms if s in HALIDES), len(metals),
        max(nums) if nums else 0, float(np.mean(nums)) if nums else 0.0,
    ]
    vals += [sum(1 for s in syms if s == e) for e in TRACKED_ELEMENTS]
    vals += [
        _safe(Descriptors.MolWt, lf), lf_logp, lf_tpsa,
        _safe(Descriptors.NumHDonors, lf), _safe(Descriptors.NumHAcceptors, lf),
        _safe(Descriptors.RingCount, lf), _safe(Descriptors.NumAromaticRings, lf),
        _safe(Descriptors.NumRotatableBonds, lf), _safe(Descriptors.FractionCSP3, lf),
        lf.GetNumHeavyAtoms(), _safe(Descriptors.NumHeteroatoms, lf),
        lf_molmr, lf_labute, lf_balaban,
        lf_logp / lf_heavy, lf_tpsa / lf_heavy,
    ]
    assert len(vals) == len(CHEM_AWARE_NAMES), f"{len(vals)} vs {len(CHEM_AWARE_NAMES)}"
    return np.array(vals, dtype=np.float64)


def f_avalon(mol):
    return np.array(pyAvalonTools.GetAvalonFP(mol, nBits=1024))


def f_morgan_count(mol):
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float64)
    for idx, v in fp.GetNonzeroElements().items():
        arr[idx] = v
    return arr


def f_rdkit_fp(mol):
    return np.array(Chem.RDKFingerprint(mol, fpSize=2048, maxPath=6))


NEW_SETS = {
    "chem_aware": f_chem_aware,
    "avalon": f_avalon,
    "morgan_count": f_morgan_count,
    "rdkit_fp": f_rdkit_fp,
}


def new_feature_names(fs):
    if fs == "chem_aware":
        return list(CHEM_AWARE_NAMES)
    if fs == "avalon":
        return [f"avalon_bit_{i:04d}" for i in range(1024)]
    if fs == "morgan_count":
        return [f"morgan_count_r2_{i:04d}" for i in range(2048)]
    if fs == "rdkit_fp":
        return [f"rdkit_fp_bit_{i:04d}" for i in range(2048)]
    raise ValueError(fs)


def featurize_new(smiles_list, fs):
    fn = NEW_SETS[fs]
    rows = []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"unparseable SMILES at {i}: {s!r}")
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
        rows.append(fn(m))
    return _clean(np.vstack(rows))
