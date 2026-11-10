"""Step 01b: applicability-domain filter -> dataset version v2 (subset of v1; SAME split/groups/folds).

A-priori, structure-only rules (no model errors or labels used):
  (1) exclude entries containing any element outside {H,B,C,N,O,F,Si,P,S,Cl,Se,Br,I} other than
      simple group-1/2 counterion metals {Li,Na,K,Rb,Cs,Mg,Ca,Sr,Ba}
      -> removes organometallics / transition-metal complexes (descriptor/graph features are not
         designed for coordination chemistry);
  (2) exclude entries with >= 2 NEUTRAL carbon-containing fragments
      -> removes reaction masses, UVCB substances, polymers written as monomer lists
         (e.g. 'bisphenol A . epichlorohydrin'), whose 'structure' does not define the substance.
Charged organic ion pairs (e.g. R-NH3+ . R'-COO-) and inorganic-counterion salts are KEPT.
Outputs: data/04b_domain_flags.csv, data/05b_split_assignments_v2.csv, data/06b_train_v2.csv, data/06b_test_v2.csv,
         results/01b_domain_log.json
"""
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
R = Path(__file__).resolve().parents[1]
ORGANIC = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
COUNTER_METALS = {3, 11, 19, 37, 55, 12, 20, 38, 56}
d = pd.read_csv(R / "data/05_split_assignments.csv")
def flags(s):
    m = Chem.MolFromSmiles(s)
    fr = Chem.GetMolFrags(m, asMols=True)
    n_neu = sum(1 for f in fr if any(a.GetAtomicNum() == 6 for a in f.GetAtoms()) and Chem.GetFormalCharge(f) == 0)
    bad = sorted({a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() not in ORGANIC | COUNTER_METALS})
    return n_neu, ",".join(bad)
f = d.SMILES.map(flags)
d["n_neutral_organic_fragments"] = f.map(lambda t: t[0]); d["disallowed_elements"] = f.map(lambda t: t[1])
d["ood_metal"] = d.disallowed_elements != ""
d["ood_mixture"] = d.n_neutral_organic_fragments >= 2
d["in_domain"] = ~(d.ood_metal | d.ood_mixture)
d[["mol_id", "ID", "SMILES", "split", "n_neutral_organic_fragments", "disallowed_elements", "ood_metal", "ood_mixture", "in_domain"]].to_csv(R / "data/04b_domain_flags.csv", index=False)
v2 = d[d.in_domain].drop(columns=["n_neutral_organic_fragments", "disallowed_elements", "ood_metal", "ood_mixture", "in_domain"])
v2.to_csv(R / "data/05b_split_assignments_v2.csv", index=False)
v2[v2.split == "train"].to_csv(R / "data/06b_train_v2.csv", index=False)
v2[v2.split == "test"].to_csv(R / "data/06b_test_v2.csv", index=False)
log = dict(v1=len(d), ood_metal=int(d.ood_metal.sum()), ood_mixture=int(d.ood_mixture.sum()), v2=len(v2),
           v2_train=int((v2.split == "train").sum()), v2_test=int((v2.split == "test").sum()),
           removed_train=int((~d.in_domain & (d.split == "train")).sum()), removed_test=int((~d.in_domain & (d.split == "test")).sum()),
           v2_acyclic_frac_train=float((~v2[v2.split == "train"].has_ring_parent).mean()),
           v2_acyclic_frac_test=float((~v2[v2.split == "test"].has_ring_parent).mean()),
           v2_fold_sizes=v2[v2.split == "train"].cv_fold.value_counts().sort_index().tolist(),
           v2_logS_mean_train=float(v2[v2.split == "train"].LogS.mean()), v2_logS_mean_test=float(v2[v2.split == "test"].LogS.mean()))
json.dump(log, open(R / "results/01b_domain_log.json", "w"), indent=2); print(log)
