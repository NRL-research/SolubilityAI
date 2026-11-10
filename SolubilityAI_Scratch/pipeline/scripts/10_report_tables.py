"""Step 10: generate the result tables used in REPORT.md from saved results (no hand-typed numbers)."""
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
cv = pd.read_csv(R / "results/07_cv_indomain.csv")
tm = pd.read_json(R / "results/07_test_metrics.json")
ind = tm[tm.test_set == "indomain"].set_index("name"); full = tm[tm.test_set == "full"].set_index("name")
L = []
L.append("### Table A. All configurations: cross-validation and one-time test results\n")
L.append("CV = pooled out-of-fold predictions, 5-fold grouped CV. 'own' = on the model's own training set; "
         "'in-domain' = on the 7,108 in-domain training molecules (same folds for all models, so this column is comparable). "
         "Fold SD = standard deviation of per-fold RMSE. Test = in-domain test (n=1804), with 95% CI from a cluster bootstrap. "
         "p = paired bootstrap (molecule / cluster level) probability that the model is at least as good as the pre-registered `blend_nnls`.\n")
L.append("| config | family | features | train set | CV RMSE own (fold SD) | CV RMSE in-domain | CV R² in-domain | Test RMSE [95% CI] | Test R² | Full-test RMSE / R² | ΔRMSE vs best [95% CI, molecule] | p (mol / cluster) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in cv.itertuples():
    t, f = ind.loc[r.name], full.loc[r.name]
    ds = "v2" if r.name.endswith("_v2") else ("v1+v2" if r.family == "blend" else "v1")
    p = "—" if pd.isna(t.p_other_at_least_as_good_mol) else f"{t.p_other_at_least_as_good_mol:.4f} / {t.p_other_at_least_as_good_cluster:.4f}"
    L.append(f"| `{r.name}` | {r.family} | {r.feature_set} | {ds} | {r.cv_rmse_pooled:.3f} ({r.cv_rmse_fold_sd:.3f}) | {r.cv_rmse_indomain:.3f} | {r.cv_r2_indomain:.3f} | "
             f"{t.test_rmse:.3f} [{t.rmse_ci_cluster[0]:.3f}, {t.rmse_ci_cluster[1]:.3f}] | {t.test_r2:.3f} | {f.test_rmse:.3f} / {f.test_r2:.3f} | "
             f"{t.d_rmse_vs_best:+.3f} [{t.d_rmse_ci_mol[0]:+.3f}, {t.d_rmse_ci_mol[1]:+.3f}] | {p} |")
hp = []
for f in sorted((R / "results/hpsearch").glob("*.csv")):
    h = pd.read_csv(f); hp.append(f"| {f.stem} | {len(h)} | {h.cv_rmse_pooled.min():.3f} | {h.cv_rmse_pooled.max():.3f} |")
L.append("\n### Table B. Random hyperparameter searches (CV RMSE on the model's own training set)\n")
L.append("| search | trials | best CV RMSE | worst CV RMSE |\n|---|---|---|---|"); L += hp
sub = json.load(open(R / "results/07_subgroups_best.json"))
L.append(f"\n### Table C. Subgroups on the test set, pre-registered model `{sub['model']}`\n")
L.append("| subgroup | n | RMSE | MAE | R² |\n|---|---|---|---|---|")
for k, v in sub["subgroups"].items():
    L.append(f"| {k} | {v['n']} | {v['rmse']:.3f} | {v['mae']:.3f} | {v['r2']:.3f} |")
ext = json.load(open(R / "results/07_external_sc.json"))
L.append("\n### Table D. External SC sets (read the caveats in the report)\n")
L.append("| model | " + " | ".join(f"{k.replace('_processed','')} RMSE / R² (n={ext[k]['n']})" for k in ext) + " |")
L.append("|---|" + "---|" * len(ext))
for n in ["blend_nnls", "blend_nnls_family", "lgbm_desc_mordred", "lgbm_desc", "gnn_hybrid", "svr_desc", "esol4_linear"]:
    L.append(f"| `{n}` | " + " | ".join(f"{ext[k][n]['rmse']:.2f} / {ext[k][n]['r2']:.2f}" for k in ext) + " |")
blend = json.load(open(R / "saved_models/blend_nnls/config.json")); bf = json.load(open(R / "saved_models/blend_nnls_family/config.json"))
L.append("\n### Table E. Blend weights (fitted by NNLS on the in-domain training OOF predictions)\n")
L.append("| component | `blend_nnls` weight | `blend_nnls_family` weight |\n|---|---|---|")
wf = {c["folder"]: c["weight"] for c in bf["components"]}; wb = {c["folder"]: c["weight"] for c in blend["components"]}
for k in sorted(set(wf) | set(wb), key=lambda k: -wb.get(k, 0)):
    L.append(f"| `{k}` | {wb.get(k, 0):.4f} | {wf.get(k, 0):.4f} |")
L.append(f"| intercept | {blend['intercept']:.4f} | {bf['intercept']:.4f} |")
open(R / "results/10_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:8]))
