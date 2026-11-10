"""External benchmark evaluation (SC1, SC2-tight, SC2-loose) of already-saved models.

NO model is trained, tuned, re-weighted or swapped here.  Every model is loaded fresh from its
saved folder in a separate Python subprocess (saved_models/<name>/predict.py -> native model file,
preprocessor, featurize.py).

Steps
  A. Validity checks (written to results/external/ext_validity.json):
     A1 identity overlap (full InChIKey, InChIKey connectivity block, parent connectivity block)
        between each benchmark set and the training molecules of every evaluated model;
     A2 static audit of where the SC files are referenced in the code base;
     A3 target-column unit check (value ranges) - no conversion is applied;
     A4 compound counts, non-numeric labels (TSTM = "too soluble to measure"), SMILES parsing,
        InChIKey consistency, and the SC1 name/structure misalignment with its correction.
  B. Per-compound predictions for each model and set -> results/external/pred_<model>__<set>.csv
  C. Statistics per model and set (RMSE, MAE, R2, bias, Pearson, Spearman, within 0.5 / 1.0,
     bootstrap 95% CIs: 2000 compound resamples, seed 20261011) -> results/external/ext_metrics.csv
  D. Sanity checks (identical predictions across models / sets; blend = weighted sum of parts).
Sets are never pooled.
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
R = Path(__file__).resolve().parents[1]
OUT = R / "results/external"; OUT.mkdir(parents=True, exist_ok=True)
SEED, B = 20261011, 2000

HEADLINE = ["blend_nnls", "blend_nnls_family", "r2_blend_nnls_ergfix"]
COMPONENTS = []
for h in HEADLINE:
    for c in json.load(open(R / f"saved_models/{h}/config.json"))["components"]:
        if c["folder"] not in COMPONENTS: COMPONENTS.append(c["folder"])
MODELS = HEADLINE + COMPONENTS

# ------------------------------------------------------------------ A4 / A3: load sets
raw = {k: pd.read_csv(R / f"data/00_raw/{f}.csv") for k, f in
       [("SC1", "SC1_processed"), ("SC2_tight", "SC2_tight_processed"), ("SC2_loose", "SC2_loose_processed")]}
val = {}
# SC1 name/structure misalignment (see REPORT): the CAS, SMILES and InChIKey columns are shifted
# relative to the name and value columns in rows 0-5 (and rows 22/23 are swapped).  The values follow
# the NAMES: e.g. the row named Benzthiazide has -4.83 (SC2-tight measures benzthiazide at -4.84) and
# the very soluble 2-chloromandelic acid / pseudoephedrine rows carry "TSTM".  Correction = move the
# structure columns back to the row whose name they belong to (identified by CAS number).
SC1_FIX = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 0, 22: 23, 23: 22}   # row r takes the structure of row SC1_FIX[r]
sc1 = raw["SC1"].copy()
struct_cols = ["cas_number", "smiles", "canonical_smiles", "inchikey"]
fixed = sc1.copy()
for r, src in SC1_FIX.items():
    fixed.loc[r, struct_cols] = sc1.loc[src, struct_cols].values
fixed.to_csv(R / "data/ext_SC1_name_corrected.csv", index=False)
sets = {"SC1": fixed.rename(columns={"logS0_uM": "logS_exp"}),
        "SC1_as_provided": sc1.rename(columns={"logS0_uM": "logS_exp"}),
        "SC2_tight": raw["SC2_tight"].rename(columns={"logS0": "logS_exp"}),
        "SC2_loose": raw["SC2_loose"].rename(columns={"logS0": "logS_exp"})}
counts, excluded = {}, {}
for k, d in sets.items():
    num = pd.to_numeric(d.logS_exp, errors="coerce")
    excluded[k] = d.loc[num.isna(), ["name", "logS_exp"]].to_dict("records")
    d["logS_exp_num"] = num
    d["parsed"] = [Chem.MolFromSmiles(s) is not None for s in d.canonical_smiles]
    d["ik_recomputed"] = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if p else None for s, p in zip(d.canonical_smiles, d.parsed)]
    d["ik_matches_file"] = d.ik_recomputed == d.inchikey
    counts[k] = dict(rows=len(d), non_numeric=int(num.isna().sum()), unparseable=int((~d.parsed).sum()),
                     inchikey_mismatch=int((~d.ik_matches_file).sum()), evaluable=int((num.notna() & d.parsed).sum()))
units = {k: dict(column={"SC1": "logS0_uM", "SC1_as_provided": "logS0_uM"}.get(k, "logS0"),
                 min=float(d.logS_exp_num.min()), median=float(d.logS_exp_num.median()), max=float(d.logS_exp_num.max()))
         for k, d in sets.items()}

# ------------------------------------------------------------------ A1: identity overlap with training data
def keys(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: return set()
    k = Chem.MolToInchiKey(m)
    p = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(m))
    return {"F:" + k, "B:" + k[:14], "B:" + Chem.MolToInchiKey(p)[:14]}
rawdb = pd.read_csv(R / "data/00_raw/AqSolDB.csv").set_index("ID")
def train_keys(train_csv):
    t = pd.read_csv(R / train_csv); s = set()
    for ids in t.ID:
        for i in ids.split("|"): s |= keys(rawdb.SMILES[i])
    return s, len(t)
TRAIN = {"v1": train_keys("data/06_train.csv"), "v2": train_keys("data/06b_train_v2.csv")}
def model_train_sets(name):
    cfg = json.load(open(R / f"saved_models/{name}/config.json"))
    if cfg["family"] in ("blend", "stack"):
        s = {"v2"}   # blend weights fitted on in-domain (v2) training OOF predictions
        for c in cfg["components"]: s |= model_train_sets(c["folder"] if isinstance(c, dict) else c)
        return s
    return {cfg.get("dataset", "v1")}
sc_keys = {}
for k, d in sets.items():
    sc_keys[k] = [keys(s) | {"F:" + ik, "B:" + ik[:14]} for s, ik in zip(d.canonical_smiles, d.inchikey)]
overlap = {}
for mname in MODELS:
    ts = model_train_sets(mname); tk = set().union(*[TRAIN[t][0] for t in ts])
    overlap[mname] = dict(training_sets=sorted(ts), n_train_keys=len(tk),
                          **{k: [sets[k].name.iloc[i] for i, ks in enumerate(sc_keys[k]) if ks & tk] for k in sets})
overlap_any = {m: any(len(v[k]) for k in sets) for m, v in overlap.items()}
test_keys = set()
for ids in pd.read_csv(R / "data/06_test.csv").ID:
    for i in ids.split("|"): test_keys |= keys(rawdb.SMILES[i])
overlap_test = {k: [sets[k].name.iloc[i] for i, ks in enumerate(sc_keys[k]) if ks & test_keys] for k in sets}
between = {}
for a in ["SC1", "SC2_tight", "SC2_loose"]:
    for b in ["SC1", "SC2_tight", "SC2_loose"]:
        if a < b:
            ia = {ik[:14]: n for n, ik in zip(sets[a].name, sets[a].inchikey)}
            between[f"{a}&{b}"] = [ia[ik[:14]] for ik in sets[b].inchikey if ik[:14] in ia]

# ------------------------------------------------------------------ A2: static usage audit
usage = []
for f in sorted(list((R / "scripts").glob("*.py")) + list((R / "src").glob("*.py"))):
    if f.name.startswith("ext_"): continue
    txt = f.read_text()
    if re.search(r"SC1|SC2", txt):
        usage.append(dict(file=str(f.relative_to(R)), lines=[l.strip()[:140] for l in txt.splitlines() if re.search(r"SC1|SC2", l)][:6]))
json.dump(dict(counts=counts, non_numeric_excluded=excluded, units=units, sc1_fix_rows=SC1_FIX,
               overlap_per_model=overlap, any_overlap=overlap_any, overlap_with_test_set=overlap_test,
               overlap_between_sc_sets=between, code_references=usage),
          open(OUT / "ext_validity.json", "w"), indent=2, default=str)
print("counts", json.dumps(counts)); print("any overlap:", {m: v for m, v in overlap_any.items() if v})

# ------------------------------------------------------------------ B: predictions (fresh subprocess per model)
all_smiles = sorted({s for d in sets.values() for s in d.canonical_smiles})
CHILD = r'''
import sys, json, importlib.util
folder = sys.argv[1]; smiles = json.load(open(sys.argv[2]))
spec = importlib.util.spec_from_file_location("p", folder + "/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p = m.load_predictor(folder)(smiles)
print(json.dumps([None if x != x else float(x) for x in p]))
'''
sm_file = OUT / "_smiles_union.json"; json.dump(all_smiles, open(sm_file, "w"))
PRED = {}
for mname in MODELS:
    r = subprocess.run([sys.executable, "-c", CHILD, str(R / "saved_models" / mname), str(sm_file)], capture_output=True, text=True, cwd="/")
    if r.returncode != 0:
        print("FAILED", mname, r.stderr[-500:]); PRED[mname] = None; continue
    PRED[mname] = dict(zip(all_smiles, json.loads(r.stdout.strip().splitlines()[-1])))
    print("predicted", mname, flush=True)
sm_file.unlink()

sys.path.insert(0, str(R / "saved_models"))
import applicability_domain as AD
ad = {c["smiles"]: c for c in AD.check(all_smiles)}

# ------------------------------------------------------------------ C: statistics
def stats(y, p, rng):
    e = p - y
    out = dict(n=len(y), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
               r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), bias=float(np.mean(e)),
               pearson=float(pearsonr(p, y)[0]), spearman=float(spearmanr(p, y)[0]),
               within_0_5=float(np.mean(np.abs(e) <= 0.5)), within_1_0=float(np.mean(np.abs(e) <= 1.0)))
    br, b2 = [], []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y)); ee = e[i]; yy = y[i]
        br.append(np.sqrt(np.mean(ee ** 2))); b2.append(1 - np.sum(ee ** 2) / np.sum((yy - yy.mean()) ** 2))
    out.update(rmse_ci=[float(x) for x in np.percentile(br, [2.5, 97.5])], r2_ci=[float(x) for x in np.percentile(b2, [2.5, 97.5])])
    return out
rows = []
for k, d in sets.items():
    ev = d[d.logS_exp_num.notna() & d.parsed].reset_index(drop=True)
    for mname in MODELS:
        if PRED[mname] is None: continue
        p = np.array([PRED[mname][s] for s in ev.canonical_smiles], dtype=float)
        assert np.isfinite(p).all(), (mname, k)
        tab = pd.DataFrame(dict(name=ev.name, inchikey=ev.inchikey, canonical_smiles=ev.canonical_smiles,
                                logS_exp=ev.logS_exp_num, logS_pred=p, error_pred_minus_exp=p - ev.logS_exp_num.values))
        if "sd" in ev: tab["exp_sd"] = ev.sd.values; tab["n_measurements"] = ev.n_measurements.values
        tab["nn_tanimoto_train"] = [ad[s]["nn_tanimoto_train"] for s in ev.canonical_smiles]
        tab["in_domain_rules"] = [ad[s]["in_domain_rules"] for s in ev.canonical_smiles]
        tab.to_csv(OUT / f"pred_{mname}__{k}.csv", index=False)
        st = stats(ev.logS_exp_num.values, p, np.random.RandomState(SEED))
        if "sd" in ev:
            noise = float(np.sqrt(np.mean(ev.sd.values ** 2)))
            st.update(exp_noise_rmse=noise, model_rmse_beyond_noise=float(np.sqrt(max(st["rmse"] ** 2 - noise ** 2, 0))),
                      frac_within_2sd=float(np.mean(np.abs(p - ev.logS_exp_num.values) <= 2 * ev.sd.values)))
        rows.append(dict(model=mname, headline=mname in HEADLINE, set=k, **st))
M = pd.DataFrame(rows)
M.to_csv(OUT / "ext_metrics.csv", index=False)

# ------------------------------------------------------------------ D: sanity checks
san = {}
names = [m for m in MODELS if PRED[m] is not None]
V = {m: np.array([PRED[m][s] for s in all_smiles]) for m in names}
san["identical_model_pairs"] = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if np.allclose(V[a], V[b], atol=1e-12)]
san["n_distinct_predictions_per_model_min"] = int(min(len(np.unique(np.round(V[m], 10))) for m in names))
for h in HEADLINE:
    cfg = json.load(open(R / f"saved_models/{h}/config.json"))
    recon = cfg["intercept"] + sum(c["weight"] * V[c["folder"]] for c in cfg["components"])
    san[f"{h}_equals_weighted_sum_of_components_maxabs"] = float(np.max(np.abs(recon - V[h])))
shared = {}
for a in ["SC1", "SC2_tight", "SC2_loose"]:
    for b in ["SC1", "SC2_tight", "SC2_loose"]:
        if a < b:
            common = set(sets[a].canonical_smiles) & set(sets[b].canonical_smiles)
            shared[f"{a}&{b}"] = len(common)
san["shared_structures_between_sets_(identical_prediction_expected)"] = shared
san["sets_have_different_prediction_vectors"] = {k: float(np.mean([PRED["blend_nnls"][s] for s in sets[k].canonical_smiles])) for k in sets}
json.dump(san, open(OUT / "ext_sanity.json", "w"), indent=2, default=str)
print(json.dumps(san, indent=1, default=str))
print(M[M.headline][["model", "set", "n", "rmse", "mae", "r2", "bias", "pearson", "spearman", "within_0_5", "within_1_0"]].round(3).to_string())
