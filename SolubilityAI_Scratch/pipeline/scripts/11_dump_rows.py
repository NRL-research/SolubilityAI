"""Step 11a: dump results-table rows for the Word report (input to 11_make_docx_report.js)."""
import json, sys
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
cv = pd.read_csv(R / "results/07_cv_indomain.csv").set_index("name")
t = pd.read_json(R / "results/07_test_metrics.json"); t = t[t.test_set == "indomain"].set_index("name")
rows = [dict(name=n, fs=cv.feature_set[n], cv=float(cv.cv_rmse_indomain[n]), cvsd=float(cv.cv_rmse_fold_sd[n]),
             rmse=float(t.test_rmse[n]), rci=list(map(float, t.rmse_ci_cluster[n])), mae=float(t.test_mae[n]),
             r2=float(t.test_r2[n]), r2ci=list(map(float, t.r2_ci_cluster[n]))) for n in cv.index]
json.dump(rows, open(sys.argv[1], "w"), indent=1)
