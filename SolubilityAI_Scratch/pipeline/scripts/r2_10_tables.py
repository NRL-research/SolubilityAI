"""Round 2, step 10: tables for the report (markdown) and rows for the Word report, generated from
results/r2_log.csv, results/r2_07_cv_indomain_all.csv and results/r2_07_test_metrics.json."""
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
cv = pd.read_csv(R / "results/r2_07_cv_indomain_all.csv").set_index("name")
_lg = pd.read_csv(R / "results/r2_log.csv").set_index("name")
for n in ["r2_lgbm_d_av_erg2", "r2_lgbm_desc_mordred_avalon_erg2", "r2_blend_nnls_ergfix"]:
    if n in _lg.index and n not in cv.index:
        cv.loc[n, ["feature_set", "cv_rmse_indomain", "cv_r2_indomain", "cv_rmse_fold_sd"]] = [_lg.feature_set[n], _lg.cv_rmse_indomain[n], _lg.cv_r2_indomain[n], _lg.cv_rmse_fold_sd_own[n]]
t = pd.read_json(R / "results/r2_07_test_metrics.json")
if (R / "results/r2_07b_test_metrics.json").exists():   # bug-fix (deterministic ErG) replacements
    t = pd.concat([t, pd.read_json(R / "results/r2_07b_test_metrics.json")], ignore_index=True)
_cvb = pd.read_csv(R / "results/cv_results.csv").set_index("name")
ti = t[t.test_set == "indomain"].set_index("name"); tf = t[t.test_set == "full"].set_index("name")
log = pd.read_csv(R / "results/r2_log.csv")
why = dict(zip(log.name, log.why))
for n in ["gnn_hybrid_s43", "gnn_hybrid_s44", "gnn_mpnn_s43"]:
    why[n] = "Extra seed for the GNN (member of the seed ensembles)."
why["blend_nnls"] = "Round-1 final model (reference)."
names = [n for n in ti.index]
names = sorted(names, key=lambda n: cv.cv_rmse_indomain[n])
rows = []
for n in names:
    a, b = ti.loc[n], tf.loc[n]
    rows.append(dict(name=n, fs=cv.feature_set[n], cv=float(cv.cv_rmse_indomain[n]), cvsd=float(cv.cv_rmse_fold_sd[n]), cvr2=float(cv.cv_r2_indomain[n]),
                     rmse=float(a.test_rmse), rci=list(map(float, a.rmse_ci_cluster)), mae=float(a.test_mae), r2=float(a.test_r2),
                     r2ci=list(map(float, a.r2_ci_cluster)), d=float(a.d_rmse_vs_blend_nnls), dci=list(map(float, a.d_ci_cluster)),
                     p=float(a.p_not_better_than_blend_nnls_cluster), full_rmse=float(b.test_rmse), full_r2=float(b.test_r2), why=why.get(n, "")))
json.dump(rows, open(R / "results/r2_10_report_rows.json", "w"), indent=1)
L = ["### Table R2-A. Round-2 configurations (plus the round-1 reference `blend_nnls`), sorted by in-domain CV RMSE\n",
     "CV = 5-fold grouped CV on the 7,108 in-domain training molecules (± SD of per-fold RMSE on the model's own training set). "
     "Test = one-time evaluation on the 1,804 in-domain test molecules [95% cluster-bootstrap CI]. "
     "Δ = test RMSE minus blend_nnls's on the same resamples; p = share of paired cluster-bootstrap resamples in which the model is NOT better than blend_nnls.\n",
     "| config | features | CV RMSE ± SD | CV R² | Test RMSE [95% CI] | Test MAE | Test R² [95% CI] | Δ vs blend_nnls [95% CI] | p | Full-test RMSE / R² |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| `{r['name']}` | {r['fs']} | {r['cv']:.3f} ± {r['cvsd']:.3f} | {r['cvr2']:.3f} | {r['rmse']:.3f} [{r['rci'][0]:.3f}, {r['rci'][1]:.3f}] | {r['mae']:.3f} | "
             f"{r['r2']:.3f} [{r['r2ci'][0]:.3f}, {r['r2ci'][1]:.3f}] | {r['d']:+.3f} [{r['dci'][0]:+.3f}, {r['dci'][1]:+.3f}] | {r['p']:.3f} | {r['full_rmse']:.3f} / {r['full_r2']:.3f} |")
L.append("\n### Table R2-B. CV-only exploratory variants (not refit or saved)\n")
L.append("| variant | in-domain nested CV RMSE | CV R² | note |\n|---|---|---|---|")
for _, r in log[log.name.str.startswith("(exploratory)")].iterrows():
    L.append(f"| {r['name'].replace('(exploratory) ', '')} | {r.cv_rmse_indomain:.4f} | {r.cv_r2_indomain:.4f} | {r.params} |")
open(R / "results/r2_10_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
