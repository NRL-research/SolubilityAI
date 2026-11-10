"""External benchmark evaluation, step 2: interpretation tables (no new predictions, no model changes).
Reads results/external/pred_*.csv and ext_metrics.csv written by ext_01_benchmark_eval.py.
Outputs results/external/ext_analysis.json and ext_worst_compounds.csv.
  * comparison with each model's own in-domain test-set performance;
  * error decomposition RMSE^2 = bias^2 + SD(error)^2 (descriptive only; no correction is applied);
  * bias by ionization class (SMARTS-based; HYPOTHESIS check for intrinsic-vs-apparent solubility);
  * error by nearest-neighbour similarity to the training set;
  * range restriction: R^2 the test-set RMSE would give on each benchmark's spread of values;
  * worst-predicted compounds per set for the headline models.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
R = Path(__file__).resolve().parents[1]; X = R / "results/external"
M = pd.read_csv(X / "ext_metrics.csv")
HEAD = ["blend_nnls", "blend_nnls_family", "r2_blend_nnls_ergfix"]
SETS = ["SC1", "SC2_tight", "SC2_loose"]

t1 = pd.read_csv(R / "results/07_test_metrics.csv").set_index("name")
t2 = pd.read_csv(R / "results/r2_07_test_metrics.csv").set_index("name")
t3 = pd.read_json(R / "results/r2_07b_test_metrics.json"); t3 = t3[t3.test_set == "indomain"].set_index("name")
def test_row(n):
    for t in (t1, t3, t2):
        if n in t.index: return float(t.test_rmse[n]), float(t.test_r2[n]), float(t.test_mae[n])
    return (None, None, None)
test_sd = pd.read_csv(R / "data/06b_test_v2.csv").LogS.std()

out = {"own_test_logS_sd": float(test_sd), "comparison": [], "decomposition": [], "ionization": [], "similarity": [], "range_restriction": {}}
for _, r in M.iterrows():
    tr, t2_, tm = test_row(r.model)
    out["comparison"].append(dict(model=r.model, set=r.set, ext_rmse=r.rmse, ext_r2=r.r2, ext_mae=r.mae, test_rmse=tr, test_r2=t2_, test_mae=tm,
                                  rmse_ratio=None if tr is None else r.rmse / tr))
ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
SULFONAMIDE_ACIDIC = Chem.MolFromSmarts("[#6]S(=O)(=O)[NH1][#6,#7]")      # sulfonylurea / N-aryl sulfonamide NH
PHENOL = Chem.MolFromSmarts("c[OH1]")
BASIC_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N-a);!$(N-C=N);!$(N#*);!$(N=*)]")
AMIDINE = Chem.MolFromSmarts("[NX3][CX3]=[NX2]")
def ion_class(smi):
    m = Chem.MolFromSmiles(smi)
    acid = m.HasSubstructMatch(ACID) or m.HasSubstructMatch(SULFONAMIDE_ACIDIC)
    base = m.HasSubstructMatch(BASIC_AMINE) or m.HasSubstructMatch(AMIDINE)
    return "acid+base" if acid and base else ("acid" if acid else ("base" if base else "neutral/weak"))
for s in SETS:
    for h in HEAD:
        d = pd.read_csv(X / f"pred_{h}__{s}.csv")
        e = d.error_pred_minus_exp.values
        out["decomposition"].append(dict(model=h, set=s, rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
                                         error_sd=float(e.std(ddof=0)), bias_share_of_mse=float(e.mean() ** 2 / np.mean(e ** 2))))
        d["ion"] = d.canonical_smiles.map(ion_class)
        for c, g in d.groupby("ion"):
            out["ionization"].append(dict(model=h, set=s, ion_class=c, n=len(g), bias=float(g.error_pred_minus_exp.mean()),
                                          rmse=float(np.sqrt(np.mean(g.error_pred_minus_exp ** 2)))))
        for lo, hi in [(0, 0.4), (0.4, 0.6), (0.6, 1.01)]:
            g = d[(d.nn_tanimoto_train >= lo) & (d.nn_tanimoto_train < hi)]
            if len(g): out["similarity"].append(dict(model=h, set=s, band=f"{lo}-{min(hi,1.0)}", n=len(g),
                                                     rmse=float(np.sqrt(np.mean(g.error_pred_minus_exp ** 2))), bias=float(g.error_pred_minus_exp.mean())))
    d = pd.read_csv(X / f"pred_blend_nnls__{s}.csv")
    sd = d.logS_exp.std(ddof=0); tr = test_row("blend_nnls")[0]
    out["range_restriction"][s] = dict(benchmark_logS_sd=float(sd), r2_if_rmse_equal_to_test=float(1 - tr ** 2 / sd ** 2),
                                       median_nn_sim=float(d.nn_tanimoto_train.median()), frac_in_domain=float(d.in_domain_rules.mean()))
te = pd.read_csv(R / "data/06b_test_v2.csv")
worst = []
for s in SETS:
    d = pd.read_csv(X / f"pred_blend_nnls__{s}.csv")
    others = {h: pd.read_csv(X / f"pred_{h}__{s}.csv").set_index("inchikey").logS_pred for h in HEAD[1:]}
    d["abs_err"] = d.error_pred_minus_exp.abs()
    for _, r in d.sort_values("abs_err", ascending=False).head(6).iterrows():
        m = Chem.MolFromSmiles(r.canonical_smiles)
        worst.append(dict(set=s, name=r["name"], logS_exp=r.logS_exp, pred_blend_nnls=r.logS_pred,
                          pred_blend_nnls_family=others["blend_nnls_family"][r.inchikey], pred_r2_blend_nnls_ergfix=others["r2_blend_nnls_ergfix"][r.inchikey],
                          error=r.error_pred_minus_exp, exp_sd=r.get("exp_sd", np.nan), ion_class=ion_class(r.canonical_smiles),
                          MW=Descriptors.MolWt(m), nn_sim=r.nn_tanimoto_train, smiles=r.canonical_smiles))
pd.DataFrame(worst).to_csv(X / "ext_worst_compounds.csv", index=False)
json.dump(out, open(X / "ext_analysis.json", "w"), indent=2)
pd.set_option("display.width", 250)
print(pd.DataFrame(out["decomposition"]).round(3).to_string())
print(pd.DataFrame(out["ionization"]).query("model=='blend_nnls'").round(3).to_string())
print(pd.DataFrame(out["similarity"]).query("model=='blend_nnls'").round(3).to_string())
print(json.dumps(out["range_restriction"], indent=1))
print(pd.DataFrame(worst)[["set", "name", "logS_exp", "pred_blend_nnls", "error", "exp_sd", "ion_class", "MW", "nn_sim"]].round(2).to_string())
