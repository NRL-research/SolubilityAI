"""Applicability-domain check for UI use (saved_models/applicability_domain.py).

check(smiles_list) -> list of dicts with:
  parsed            : SMILES parsed by RDKit
  in_domain_rules   : passes the a-priori structure rules of scripts/01b_domain_filter.py
                      (no transition/heavy metals; not a multi-neutral-organic-component mixture)
  reasons           : which rule failed
  nn_tanimoto_train : max Tanimoto (Morgan r2, 2048 bits, parent structure) to the training set
                      (train_parent_fps.npz, the training molecules of data/06b_train_v2.csv)
  similarity_band   : 'low' (<0.4), 'medium' (0.4-0.6), 'high' (>=0.6); see REPORT for the
                      test-set error observed in each band.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
ORGANIC = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
COUNTER_METALS = {3, 11, 19, 37, 55, 12, 20, 38, 56}
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_train = None

def _train_fps():
    global _train
    if _train is None:
        bits = np.unpackbits(np.load(HERE / "train_parent_fps.npz")["packed"], axis=1)[:, :2048]
        _train = []
        for row in bits:
            bv = DataStructs.ExplicitBitVect(2048); bv.SetBitsFromList(np.where(row)[0].tolist()); _train.append(bv)
    return _train

def _parent(m):
    m = rdMolStandardize.Cleanup(m)
    m = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(m)
    return rdMolStandardize.Uncharger().uncharge(m)

def check(smiles_list):
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            out.append(dict(smiles=s, parsed=False, in_domain_rules=False, reasons=["unparseable"], nn_tanimoto_train=None, similarity_band=None)); continue
        reasons = []
        bad = sorted({a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() not in ORGANIC | COUNTER_METALS})
        if bad: reasons.append("contains " + ",".join(bad))
        fr = Chem.GetMolFrags(m, asMols=True)
        if sum(1 for f in fr if any(a.GetAtomicNum() == 6 for a in f.GetAtoms()) and Chem.GetFormalCharge(f) == 0) >= 2:
            reasons.append(">=2 neutral organic components (mixture/UVCB)")
        if not any(a.GetAtomicNum() == 6 for a in m.GetAtoms()): reasons.append("no carbon (inorganic)")
        nn = max(DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(_parent(m)), _train_fps()))
        out.append(dict(smiles=s, parsed=True, in_domain_rules=not reasons, reasons=reasons, nn_tanimoto_train=float(nn),
                        similarity_band="low" if nn < 0.4 else ("medium" if nn < 0.6 else "high")))
    return out

if __name__ == "__main__":
    import sys, json
    print(json.dumps(check(sys.argv[1:] or ["CCO", "[Cu++].CC(=O)[O-].CC(=O)[O-]"]), indent=1))
