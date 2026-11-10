"""External benchmark evaluation, step 3: collect everything the Word report's 'External Benchmark
Evaluation' section needs into results/external/ext_report_rows.json (no new computation of
predictions; all values come from ext_metrics.csv, ext_analysis.json, ext_validity.json, ext_sanity.json)."""
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]; X = R / "results/external"
M = pd.read_csv(X / "ext_metrics.csv")
for c in ["rmse_ci", "r2_ci"]: M[c] = M[c].map(lambda s: json.loads(s) if isinstance(s, str) else s)
A = json.load(open(X / "ext_analysis.json")); V = json.load(open(X / "ext_validity.json")); S = json.load(open(X / "ext_sanity.json"))
cmp = pd.DataFrame(A["comparison"]).set_index(["model", "set"])
rows = []
for _, r in M.iterrows():
    c = cmp.loc[(r.model, r.set)]
    rows.append(dict(model=r.model, headline=bool(r.headline), set=r.set, n=int(r.n), rmse=r.rmse, rmse_ci=r.rmse_ci, mae=r.mae, r2=r.r2, r2_ci=r.r2_ci,
                     bias=r.bias, pearson=r.pearson, spearman=r.spearman, w05=r.within_0_5, w10=r.within_1_0,
                     noise=None if pd.isna(r.get("exp_noise_rmse", float("nan"))) else r.exp_noise_rmse,
                     beyond=None if pd.isna(r.get("model_rmse_beyond_noise", float("nan"))) else r.model_rmse_beyond_noise,
                     w2sd=None if pd.isna(r.get("frac_within_2sd", float("nan"))) else r.frac_within_2sd,
                     test_rmse=c.test_rmse, test_r2=c.test_r2))
_w = pd.read_csv(X / "ext_worst_compounds.csv"); worst = [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()} for r in _w.to_dict("records")]
json.dump(dict(rows=rows, analysis=A, validity=dict(counts=V["counts"], excluded=V["non_numeric_excluded"], units=V["units"],
               any_overlap=V["any_overlap"], overlap_test=V["overlap_with_test_set"], between=V["overlap_between_sc_sets"],
               code_refs=[u["file"] for u in V["code_references"]]), sanity=S, worst=worst),
          open(X / "ext_report_rows.json", "w"), indent=1, default=float)
print(len(rows), V["overlap_between_sc_sets"], [u["file"] for u in V["code_references"]])
