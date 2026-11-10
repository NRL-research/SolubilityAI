"""Shared chemistry utilities: standardization and identity keys.

All structure identity used for decontamination / leakage checks goes through here.
"""
import re
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_unch = rdMolStandardize.Uncharger()


def mol_from_smiles(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)


def standardize_full(mol):
    """Light standardization of the full (possibly multi-component) structure:
    RDKit cleanup (normalize functional groups, reionize). Counterions are KEPT."""
    if mol is None:
        return None
    try:
        return rdMolStandardize.Cleanup(mol)
    except Exception:
        return None


def parent(mol):
    """Parent structure: largest organic fragment, neutralized. Used only for
    identity/grouping, never as the model input."""
    if mol is None:
        return None
    try:
        m = rdMolStandardize.Cleanup(mol)
        m = _lfc.choose(m)
        m = _unch.uncharge(m)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def inchikey(mol):
    if mol is None:
        return None
    try:
        k = Chem.MolToInchiKey(mol)
        return k or None
    except Exception:
        return None


def ik_block1(k):
    return k.split("-")[0] if isinstance(k, str) and len(k) >= 14 else None


def nonstereo_smiles(mol):
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def norm_name(s):
    if not isinstance(s, str):
        return None
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s or None
