// Step 11: build the formal Word report (SolubilityAI_Experiment_Report.docx).
// Results-table numbers are read from results/11_report_rows.json, produced by scripts/11_dump_rows.py
// from results/07_cv_indomain.csv + results/07_test_metrics.json.  Usage: node 11_make_docx_report.js <rows.json> <out.docx>
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, LevelFormat, PageOrientation, BorderStyle, Footer, PageNumber,
  TableOfContents,
} = require("docx");

const rows = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3];
const r2rows = process.argv[4] ? JSON.parse(fs.readFileSync(process.argv[4], "utf8")) : null;   // round-2 rows (optional)
const EXT = process.argv[5] ? JSON.parse(fs.readFileSync(process.argv[5], "utf8")) : null;      // external benchmark (optional)

const FONT = "Calibri";
const t = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 22, bold: o.bold, italics: o.italics, color: o.color });
// simple **bold** markup inside strings
function runs(s, o = {}) {
  const parts = s.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map(p => p.startsWith("**") ? t(p.slice(2, -2), { ...o, bold: true }) : t(p, o));
}
const P = (s, o = {}) => new Paragraph({ children: runs(s, o), spacing: { after: 120, line: 276 }, alignment: o.align });
const H1 = s => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: s, font: FONT })], spacing: { before: 360, after: 160 } });
const H2 = s => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: s, font: FONT })], spacing: { before: 240, after: 120 } });
const B = (s, lvl = 0) => new Paragraph({ numbering: { reference: "bullets", level: lvl }, children: runs(s), spacing: { after: 80, line: 264 } });
let LIST = 0;
const newList = () => { LIST += 1; };
const N = s => new Paragraph({ numbering: { reference: "numbers" + LIST, level: 0 }, children: runs(s), spacing: { after: 80, line: 264 } });
const TERM = (term, def) => new Paragraph({ children: [t(term + ": ", { bold: true }), ...runs(def)], spacing: { after: 100, line: 264 } });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, body, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, hdr, hi) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: hdr ? { fill: "1F3864", type: ShadingType.CLEAR, color: "auto" } : (hi ? { fill: "E2EFDA", type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ children: runs(String(txt), { size: o.size || 18, color: hdr ? "FFFFFF" : undefined, bold: hdr ? true : undefined }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...body.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false, o.highlight && o.highlight(ri))) }))],
  });
}
const gap = () => new Paragraph({ children: [t("")], spacing: { after: 60 } });

// ------------------------------------------------------------------ content
const FEAT = {
  "(components)": "Blend (each part uses its own features)", "rdkit_desc": "RDKit descriptors (223)",
  "desc+mordred": "RDKit + Mordred descriptors", "mordred": "Mordred descriptors (1,303)",
  "desc+maccs": "RDKit descriptors + MACCS keys", "desc+morgan_counts_r2_2048": "RDKit descriptors + Morgan counts",
  "morgan_counts_r2_2048": "Morgan fingerprint (counts)", "morgan_bits_r2_2048": "Morgan fingerprint (on/off bits)",
  "esol4": "4 simple descriptors (ESOL-style)", "graph": "Molecular graph", "graph+rdkit_desc(quantile-normal)": "Molecular graph + RDKit descriptors",
};
const HP = {
  blend_nnls: "Non-negative least-squares weights over 12 models", blend_nnls_family: "NNLS weights over the best model of each family (8 used)",
  blend_mean_top3: "Equal-weight average of the 3 best models",
  lgbm: "1,500 trees, learning rate 0.04, 31 leaves", lgbm_dm: "1,500 trees, learning rate 0.02, 31 leaves",
  catboost: "3,000 trees, depth 7, learning rate 0.03", xgb: "1,000 trees, depth 7, learning rate 0.04",
  gnn: "4 message-passing layers, width 200, 80 epochs", et: "500 trees", rf: "500 trees",
  svr: "C = 11.6, gamma = 0.002 (RBF kernel)", mlp: "2 hidden layers (256, 128)", ridge: "alpha = 10",
  linear: "ordinary least squares", knn: "k = 5 nearest neighbours (Tanimoto)",
};
function hp(n) {
  if (HP[n]) return HP[n];
  if (n.startsWith("lgbm_desc_morgan")) return HP.lgbm_dm;
  if (n.startsWith("lgbm")) return HP.lgbm;
  if (n.startsWith("catboost")) return HP.catboost;
  if (n.startsWith("xgb")) return HP.xgb;
  if (n.startsWith("gnn")) return HP.gnn;
  if (n.startsWith("et_")) return HP.et;
  if (n.startsWith("rf_")) return HP.rf;
  if (n.startsWith("svr")) return HP.svr;
  if (n.startsWith("mlp")) return HP.mlp;
  if (n.startsWith("ridge")) return HP.ridge;
  if (n.startsWith("esol")) return HP.linear;
  if (n.startsWith("knn")) return HP.knn;
  return "";
}
const f3 = x => Number(x).toFixed(3);
const resRows = rows.map(r => [
  (r.name === "blend_nnls" ? "★ " : "") + r.name + (r.name.endsWith("_v2") ? " (trained on v2)" : ""),
  FEAT[r.fs] || r.fs, hp(r.name),
  `${f3(r.cv)} ± ${f3(r.cvsd)}`,
  `${f3(r.rmse)} [${f3(r.rci[0])}–${f3(r.rci[1])}]`, f3(r.mae),
  `${f3(r.r2)} [${f3(r.r2ci[0])}–${f3(r.r2ci[1])}]`,
]);

const children = [];
// ---------- title
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800, after: 200 }, children: [t("SolubilityAI — Scratch Pipeline", { size: 44, bold: true, color: "1F3864" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [t("Predicting aqueous solubility (LogS) from molecular structure", { size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t("Experiment report", { size: 24, italics: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [t("Prepared for Shashi (DDRL-AI) · 23 September 2026", { size: 22 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [t("Companion files: pipeline/REPORT.md (technical record) and pipeline/saved_models/README.md (model index)", { size: 18, color: "595959" })] }));
children.push(new Paragraph({ pageBreakBefore: true, children: [t("Contents", { size: 32, bold: true, color: "1F3864" })], spacing: { after: 200 } }));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(P("(If the contents list appears empty, right-click it in Word and choose “Update field”.)", { size: 18, italics: true }));

// ---------- 1 Overview
children.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "1. Overview", font: FONT })], spacing: { after: 160 } }));
children.push(P("**Objective.** The goal was to build the most accurate model we could for predicting a molecule’s aqueous solubility from its structure alone. Solubility is reported as **LogS**: the base-10 logarithm of solubility in moles per litre. A LogS of −2 means 0.01 mol/L; each unit lower means ten times less soluble. Molecules are supplied as **SMILES** strings, a standard text notation for chemical structures (for example, ethanol is “CCO”)."));
children.push(P("**Data.** All training and testing used **AqSolDB**, a public compilation of 9,982 measured solubilities. The external benchmark sets SC1, SC2-tight and SC2-loose (the Solubility Challenge sets) were used only to remove overlapping molecules, and afterwards as a secondary external check. The work was deliberately built from scratch, without reference to any earlier code or results on this problem."));
children.push(P("**Headline result.** The final model, chosen in advance of testing, is **blend_nnls**: a weighted combination of 12 individual models. On 1,804 test molecules the models never saw during development, it achieved **R² = 0.880 and RMSE = 0.799 log units** (mean absolute error 0.55). These test molecules are the ones inside the model’s defined applicability domain. Allowing for the grouping of similar molecules, the 95% uncertainty range is RMSE 0.72–0.88 and R² 0.83–0.92. Across all 1,852 test molecules, including 48 metal complexes and mixtures outside that domain, the figures were R² = 0.868 and RMSE = 0.840."));
children.push(P("In plain terms, the typical prediction is off by about half a log unit, roughly a factor of 3.5 in solubility. That is comparable to the disagreement often seen between separate laboratory measurements of the same compound. The blend was statistically better than every individual model. The best single model, a gradient-boosted tree model on chemical descriptors, reached R² 0.870 and RMSE 0.832."));
children.push(P("**The most important caveat:** every model scored about 0.09 RMSE better on the test set than in cross-validation, the internal estimate. The more conservative expectation for genuinely new chemistry is therefore an RMSE of about 0.85–0.90 (R² ≈ 0.86). This is discussed in Section 10."));
if (EXT) children.push(P("**External benchmarks (Section 12).** The three headline models were also tested, without any retraining, on the external Solubility Challenge sets. On SC2-tight (100 drugs) blend_nnls scored RMSE 0.98 and R² 0.40; on SC1 (28 drugs, after correcting a file misalignment) RMSE 1.15 and R² 0.26; on SC2-loose (32 drugs) RMSE 1.69 and R² 0.38. Every model **systematically predicts these drugs as more soluble** than measured (by +0.5 to +1.0 log units). The likely main reason is that the benchmarks report intrinsic solubility (the uncharged form) whereas AqSolDB mostly does not; this is a hypothesis. Accuracy on the benchmarks is clearly lower than on our own test set."));
if (r2rows) children.push(P("**Round 2 update (2 October 2026).** A second round of experiments (Section 11) tried new fingerprints, a pretrained chemical embedding, a robust loss, CatBoost, seed-averaged graph networks and learned stacking. The best new configuration, **r2_blend_nnls**, reached test **R² = 0.882 and RMSE = 0.793**. Its advantage over blend_nnls (0.006 RMSE) was **not statistically confirmed** under the pre-registered test (p = 0.079), so blend_nnls remains the confirmed best model."));

// ---------- 2 Dataset
children.push(H1("2. Dataset and Preprocessing"));
children.push(H2("2.1 Sources"));
children.push(B("**AqSolDB.csv**: 9,982 entries, each with an identifier, name, InChIKey, SMILES and measured LogS. An **InChIKey** is a fixed-length code computed from a structure. Two differently written SMILES for the same molecule give the same InChIKey, which makes it the reliable way to check whether two entries are the same compound."));
children.push(B("**SC1 (32 compounds), SC2-tight (100) and SC2-loose (32)**: external benchmark sets, supplied already converted to standard SMILES and InChIKeys."));
children.push(H2("2.2 Removing overlap with the SC1/SC2 benchmarks (leakage control)"));
children.push(P("If a benchmark molecule also appears in the training data, a model can simply “remember” its answer, and a later benchmark test would be meaningless. This is called **leakage**. An AqSolDB entry was removed if it matched any of the 164 SC compounds under **any** of four deliberately generous tests:"));
newList(); children.push(N("Identical full InChIKey: 108 matches."));
children.push(N("Identical InChIKey “skeleton” (its first 14 characters). This ignores 3-D arrangement (stereochemistry) and protonation state: 140 matches."));
children.push(N("Identical skeleton after salts are stripped away and the molecule is neutralised. This catches, for example, a hydrochloride salt in AqSolDB against the free base in SC: 149 matches."));
children.push(N("Identical compound name: 99 matches. Five entries were caught by name alone."));
children.push(P("Taken together, **154 AqSolDB entries were removed** (a list is saved in data/03_removed_sc_overlap.csv). Removing a few borderline molecules costs almost nothing, whereas leakage would invalidate any future SC benchmark."));
children.push(P("**How this was verified.** A separate verification script recomputed all identity codes from the original raw SMILES, independently of the curation code. It confirmed that **zero** identity codes are shared between training and the SC sets, between test and the SC sets, and between training and test."));
children.push(P(EXT ? "**Data problem found in SC1 (corrected October 2026).** In SC1_processed.csv the structure columns (CAS number, SMILES, InChIKey) are out of step with the name and value columns in 8 of the 32 rows. Rows 1–6 are shifted by one (the row named “Acebutolol” carries the CAS number and structure of 2-chloromandelic acid), and the Pseudoephedrine and Probenecid rows are swapped. **An earlier version of this report concluded that the values follow the structures; that was wrong.** The earlier check shifted the whole file, which breaks the 24 correctly aligned rows, so it could not tell the two cases apart. The values follow the **names**. As given, the structures of benzthiazide and probenecid carry “too soluble to measure”, yet SC2-tight measures both at about −4.8. After re-aligning the structures to their names, the shared compounds agree with SC2-tight (benzthiazide −4.83 vs −4.84; probenecid −4.63 vs −4.83). Section 12 uses the name-corrected file (data/ext_SC1_name_corrected.csv) and also reports the as-provided version. Leakage control is unaffected, because removal was done by structure and by name across the whole set." : "**Data problem found in SC1.** In SC1_processed.csv the names are shifted by one row relative to the structures. For example, the row labelled “Acebutolol” contains the structure of 2-chloromandelic acid. The measured values appear to belong to the structures rather than the names. Model predictions correlate with them at r = 0.56 as given, but only r ≈ 0.05 when shifted by a row. Because removal was done by structure and by name across the whole set, leakage control is unaffected. **The SC1 file should nonetheless be corrected before it is used as a benchmark.**"));
children.push(H2("2.3 Further cleaning"));
children.push(table(["Step", "Entries remaining / removed", "Reason"], [
  ["Starting point", "9,982", "Raw AqSolDB"],
  ["Remove SC1/SC2 overlap", "−154 → 9,828", "Leakage control (Section 2.2)"],
  ["Unreadable structures", "−2", "The SMILES could not be interpreted by the chemistry toolkit"],
  ["No carbon atom", "−325", "Inorganic salts; the chemical features used here are designed for organic molecules"],
  ["Fewer than 2 heavy (non-hydrogen) atoms", "−1", "Too small to describe meaningfully"],
  ["Duplicate structures whose values disagree by > 1 log unit", "−38 entries", "Identical input with irreconcilable answers, and no basis for choosing one"],
  ["Duplicate structures agreeing within 1 log unit", "217 groups merged (average taken)", "Identical inputs must not be counted twice, or split between training and test"],
  ["**Dataset v1 (main curated set)**", "**9,217 molecules**", ""],
  ["Applicability-domain filter (Section 2.4)", "−176 metal-containing, −139 mixtures", "Structure-only rules decided before testing"],
  ["**Dataset v2 (in-domain subset)**", "**8,912 molecules**", "7,108 training / 1,804 test"],
], [3000, 2400, 3960]));
children.push(gap());
children.push(P("**Other curation choices.** No molecule was removed because its measured value looked unusual. There was no structure-independent justification for doing so, and removing “difficult” values would make the model look better than it is. Salt counter-ions (for example the sodium in a sodium salt) were **kept** in the model input, because a salt dissolves differently from its parent compound. The salt-free parent structure was used only for identity checks and for grouping."));
children.push(H2("2.4 Applicability domain (dataset v2)"));
children.push(P("Early analysis showed that the worst 5% of predictions accounted for 47% of all squared error. Most of these were metal complexes (copper, zirconium, palladium and similar) or poorly defined “mixtures”, such as reaction products or polymers written as lists of their ingredients. For these substances the structure does not really describe what was measured. Two rules based purely on structure, fixed before any test result was seen, define the **applicability domain**: the region of chemistry where predictions can be trusted."));
children.push(B("Exclude entries containing any element other than common organic elements (H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I) and simple salt-forming metals (Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba)."));
children.push(B("Exclude entries made of two or more separate, uncharged carbon-containing components (mixtures). Genuine salts are kept."));
children.push(P("A model trained without these entries was **no more accurate** on in-domain molecules (Section 7). The rules are therefore used to **flag unreliable predictions**, not as a way to improve accuracy. They are shipped as saved_models/applicability_domain.py for use in a future interface."));

// ---------- 3 Splitting
children.push(H1("3. Train/Test Splitting Method"));
children.push(P("A model is only useful if it works on molecules it has not seen. We therefore hold back a **test set** that plays no part in any development decision and is used once, at the end. How that set is chosen matters a great deal."));
children.push(P("**Why not a random split?** Chemical datasets contain families of near-identical molecules, for example the same core structure with one extra carbon atom. A random split scatters these families across both sides, so the model is “tested” on near-copies of training molecules and looks better than it really is. We checked this directly. In a random split, **9.5% of test molecules had an exact fingerprint twin in training**, and 83 compounds appeared on both sides in different salt forms."));
children.push(P("**What we did instead: cluster-based splitting.**"));
newList(); children.push(N("Each molecule was converted to a structural fingerprint (Section 4), and similarity was measured with the **Tanimoto score** (0 = nothing in common, 1 = identical)."));
children.push(N("Molecules were grouped into clusters of structurally similar compounds using the **Butina clustering** method, with a similarity threshold of 0.4. This gave 2,352 clusters."));
children.push(N("Clusters were merged further so that every salt form and stereo-isomer of the same parent compound always sits in the same group. This gave 2,348 groups."));
children.push(N("**Whole groups** were assigned to test until about 20% of molecules were in test. 200 random assignments were generated, and the one where training and test were most alike in solubility range and composition was kept. This choice used no model results."));
children.push(P("**Why not the common “scaffold split”?** A scaffold split groups molecules by their ring system. Every molecule with **no rings** (long chains, simple alcohols and so on) has an empty scaffold, so all 2,562 of them fall into a single giant group, which then lands entirely in training. We confirmed that a standard scaffold split would have put **0% acyclic molecules in the test set**, so this whole class would never have been evaluated. Our clustering treats ring-free molecules like any other. The result was **27.8% acyclic molecules in both training and test**."));
children.push(table(["Check", "Our cluster split", "Random split", "Scaffold split"], [
  ["Acyclic molecules, train / test", "27.8% / 27.8%", "27.7% / 28.0%", "34.8% / 0.0%"],
  ["Test molecules with an exact fingerprint twin in training", "0.0%", "9.5%", "0.4%"],
  ["Test molecules ≥ 0.8 similar to a training molecule", "1.7%", "18.7%", "2.2%"],
  ["Parent compounds present on both sides", "0", "83", "2"],
  ["Solubility distribution differs between train and test? (KS test p)", "No (p = 0.39)", "No (p = 0.40)", "Yes (p ≈ 10⁻⁴¹)"],
], [3700, 1900, 1900, 1860]));
children.push(gap());
children.push(P("**Resulting sizes.** On the main curated set (v1), 7,365 training and 1,852 test molecules. On the in-domain subset (v2), 7,108 training and 1,804 test molecules. The v2 subset uses exactly the same assignment; the out-of-domain entries are simply dropped."));

// ---------- 4 Features
children.push(H1("4. Feature Representations Used"));
children.push(P("Machine-learning models need numbers, not drawings of molecules. A **feature representation** is a recipe for turning a structure into a fixed list of numbers. Every representation below was computed with open-source chemistry software from the SMILES string, after a light standardisation step that tidies up how charges and functional groups are written."));
children.push(TERM("Molecular descriptors (RDKit, 223 values)", "Computed properties such as molecular weight, estimated oil-vs-water preference (logP), number of hydrogen-bond donors and acceptors, polar surface area, ring counts and charge distribution. We added 7 values describing salts and mixtures, such as the number of separate components and whether a metal is present. Descriptors capture physically meaningful “summary” properties."));
children.push(TERM("Mordred descriptors (1,303 values)", "A much larger, complementary descriptor library covering finer-grained counts, shape and connectivity indices, and electronic properties. Columns that were mostly missing or constant in the training data were dropped."));
children.push(TERM("Morgan fingerprint (2,048 values)", "A **fingerprint** records which small local chemical neighbourhoods (an atom plus its surroundings up to two bonds away) occur in the molecule. Each neighbourhood type is hashed into one of 2,048 slots. The “count” version records how many times each occurs; the “bit” version records only presence or absence. Fingerprints capture substructure patterns rather than overall properties."));
children.push(TERM("MACCS keys (167 values)", "A fixed, human-defined checklist of 166 structural questions (“does it contain a carboxylic acid?”, “a ring with nitrogen?”), each answered yes or no."));
children.push(TERM("ESOL-style descriptors (4 values)", "Just logP, molecular weight, rotatable-bond count and the fraction of aromatic atoms. This echoes a classic simple solubility equation and served as a baseline."));
children.push(TERM("Molecular graph", "The molecule treated as a network: atoms are points (described by element, charge, number of attached hydrogens, aromaticity and so on) and bonds are connections (single, double, aromatic, in a ring or not). Graph neural networks (Section 5) learn their own features directly from this network rather than using a predefined list."));
children.push(TERM("Graph + descriptors (hybrid)", "The molecular graph plus the 223 RDKit descriptors. The descriptors were rescaled to a comparable range and fed into the final stage of the network."));
children.push(P("**Not used: pretrained language-model embeddings.** These are numerical summaries produced by AI models pre-trained on millions of molecules. They were planned but could not be tried, because the computing environment was not allowed to download the required model files (Section 10)."));

// ---------- 5 Models
children.push(H1("5. Models Used"));
children.push(P("A **model** is a mathematical function that learns, from training examples, how the features map to LogS. The model types tried are grouped by family below."));
children.push(H2("5.1 Simple baselines"));
children.push(TERM("Linear regression (ESOL-style)", "Predicts LogS as a weighted sum of the four simple descriptors. It sets a “floor” that any useful model must beat."));
children.push(TERM("Ridge regression", "A weighted sum of all descriptors, with a penalty that stops any single weight from growing too large. It is simple and fast, but it can give extreme answers for unusual molecules."));
children.push(TERM("k-nearest neighbours (kNN)", "Finds the 5 most structurally similar training molecules by fingerprint similarity and averages their measured values."));
children.push(H2("5.2 Tree ensembles"));
children.push(P("A **decision tree** asks a sequence of yes/no questions about the features (“is logP above 3?”) and ends in a prediction. Single trees are crude, so many are combined."));
children.push(TERM("Random forest and extra trees", "Hundreds of trees, each trained on a random portion of the data and features, with their answers averaged. Extra trees also randomises where each split is made."));
children.push(TERM("Gradient-boosted trees: LightGBM, XGBoost, CatBoost", "Trees are built one after another, and each new tree corrects the remaining errors of the ones before. These three are different popular implementations of the same idea. They were consistently the strongest single models here."));
children.push(H2("5.3 Kernel and neural-network models"));
children.push(TERM("Support vector regression (SVR)", "Measures how similar a new molecule is to every training molecule (via a smooth “kernel” function on the descriptors) and makes a prediction from a weighted combination of these similarities."));
children.push(TERM("Multilayer perceptron (MLP)", "A basic neural network. The descriptors pass through two layers of simple adjustable units that together can represent non-linear relationships."));
children.push(TERM("Message-passing graph neural network (GNN, graph-only and hybrid)", "Works directly on the molecular graph. In each of four rounds, every atom gathers information from its bonded neighbours, so after several rounds each atom “knows” about its wider chemical environment. The atom information is then pooled into one summary for the molecule, and a small network converts that summary into LogS. The hybrid version also receives the RDKit descriptors."));
children.push(H2("5.4 Blends (ensembles of models)"));
children.push(TERM("Weighted blends", "The final predictions of several models are combined. In **blend_nnls**, weights for all candidate models were fitted by non-negative least squares (NNLS): the best-fitting combination of weights, none allowed below zero. **blend_nnls_family** does the same using only the best model from each family. **blend_mean_top3** is a plain average of the three best models. Weights were always fitted on cross-validation predictions, never on the test set. Blends work when their parts make *different* mistakes. Here the graph networks received large weights, despite being weaker on their own, because their errors differ from those of the descriptor-based models."));

// ---------- 6 Evaluation
children.push(H1("6. Evaluation Methodology"));
children.push(H2("6.1 Metrics"));
children.push(TERM("RMSE (root-mean-square error)", "The typical size of a prediction error, in LogS units. It weights large errors more heavily than small ones. Lower is better; an RMSE of 0.8 means predictions are typically within about 0.8 log units."));
children.push(TERM("MAE (mean absolute error)", "The plain average size of the error, in LogS units. It is less affected by a few large misses than RMSE is. Lower is better."));
children.push(TERM("R² (coefficient of determination)", "The share of the variation in true solubility that the model explains. 1.0 is perfect, and 0 means no better than always guessing the average."));
children.push(H2("6.2 Cross-validation: choosing models without touching the test set"));
children.push(P("All choices between models, settings (**hyperparameters**) and blend weights were made with **5-fold cross-validation** on the training data only. The training set is divided into five parts. Each model is trained on four parts and tested on the fifth, rotating until every training molecule has been predicted exactly once by a model that did not see it. These are called “out-of-fold” predictions. The five parts were built from the same similarity groups as the train/test split, so cross-validation also tests on unfamiliar chemistry. No decision was ever based on a single split."));
children.push(P("To compare models trained on dataset v1 with those trained on v2 fairly, every model was scored on the **same 7,108 in-domain training molecules with the same five parts**. Blends were scored with **nested** cross-validation: blend weights for each part were chosen using only the other four parts."));
children.push(H2("6.3 Using the test set exactly once"));
children.push(P("Before any test data was opened, the selection rule (“lowest cross-validation RMSE wins”), the chosen model (blend_nnls) and the list of all 27 configurations were written to a locked file (results/07_preregistration.json). Each saved model was then loaded from disk and predicted the test set once. This prevents quietly picking whichever model happens to score best on the test set."));
children.push(H2("6.4 Statistical checks"));
children.push(P("A single test score is itself uncertain. We used the **bootstrap**: the test set was resampled with replacement 2,000 times and the metrics recomputed each time, which shows how much they could vary. Because similar molecules are not independent, we also resampled whole clusters. This **cluster bootstrap** gives the wider and more honest ranges quoted in this report. To compare two models, both were evaluated on the *same* resamples (a **paired bootstrap**). The result is a confidence range for their difference, and the proportion of resamples in which the “worse” model was actually at least as good, which acts as a p-value."));

// ---------- 7 Experiments
children.push(H1("7. Experiments Conducted"));
children.push(P("The experiments are listed in the order they were run. Numbers are cross-validation RMSEs on in-domain training molecules unless stated otherwise. Lower is better."));
const exps = [
  ["Baselines", "ESOL-style linear regression (1.54), Tanimoto kNN (1.68), ridge regression (1.18), random forest on Morgan fingerprints (1.25) and random forest on descriptors (1.01).", "Descriptors beat fingerprints by a wide margin for solubility, which fits the fact that solubility is driven by overall properties like logP. Ridge was unstable: one cross-validation part had RMSE 4.27 because of a few metal complexes, so it was kept only as a minor blend component."],
  ["Gradient-boosted trees with hyperparameter search", "Random searches of 14 LightGBM settings (descriptors), 8 LightGBM settings (descriptors + Morgan) and 6 XGBoost settings; one CatBoost setting.", "All reached 0.94 ± 0.01. All 14 LightGBM settings fell within 1.00–1.05 on the full training set, so tuning mattered little. LightGBM, XGBoost and CatBoost were interchangeable."],
  ["SVR and MLP searches", "10 SVR settings and 6 MLP settings on descriptors.", "SVR reached 0.99. MLP reached 1.06; the neural network on descriptors did not beat trees."],
  ["Adding fingerprints to descriptors", "Morgan counts or MACCS keys added to RDKit descriptors.", "A small improvement (0.941 → 0.936)."],
  ["Error analysis → applicability domain", "Examined which molecules had the largest errors.", "The worst 5% of predictions, mostly metal complexes and mixtures, carried 47% of the squared error. This led to the structure-based domain rules (Section 2.4)."],
  ["Training on the domain-filtered set (v2)", "Retrained the six strongest models without out-of-domain entries, keeping the same settings.", "**No gain** on in-domain molecules (e.g. 0.941 vs 0.941, 0.990 vs 0.990). v2 models were *worse* on the full test set (e.g. 0.931 vs 0.891), because they never saw metals. The filter is therefore used to flag predictions, not to train."],
  ["Graph neural networks", "A graph-only network and a hybrid (graph + descriptors), each trained for a fixed 80 epochs with no early stopping.", "**A single-part result was misleading.** After the first part, the graph network (1.13 on that part) looked far worse than LightGBM (1.02). The full 5-part result (1.02 in-domain) showed it was only moderately worse, and in the blend it turned out to be one of the most valuable components. Hybrid: 0.99."],
  ["Mordred descriptors", "LightGBM on 1,303 Mordred descriptors, alone and combined with RDKit descriptors.", "The best single models: 0.931 and **0.923**."],
  ["Blending", "NNLS over all 24 single models, NNLS over the best model of each family, and a simple average of the top 3.", "**0.887 (NNLS)**, 0.888 (per family), 0.915 (top-3 average). Blending gave the largest single improvement of any experiment."],
  ["Held-out test (one time)", "All 27 saved configurations evaluated once, with bootstrap statistics.", "blend_nnls: RMSE 0.799, R² 0.880. It was significantly better than every single model; the per-family blend was statistically tied with it."],
];
exps.forEach(([title, what, found], i) => {
  children.push(new Paragraph({ children: [t(`7.${i + 1} ${title}`, { bold: true, size: 23, color: "1F3864" })], spacing: { before: 160, after: 60 } }));
  children.push(new Paragraph({ children: [t("What was tried: ", { bold: true }), ...runs(what)], spacing: { after: 60, line: 264 } }));
  children.push(new Paragraph({ children: [t("Finding: ", { bold: true }), ...runs(found)], spacing: { after: 100, line: 264 } }));
});
children.push(H2("7.11 Discarded or incomplete work, and incidents"));
children.push(B("**Cancelled:** a hybrid graph network trained on v2. It was stopped before training, because the v2 finding above made it uninformative, and the computing time went to the Mordred experiment instead."));
children.push(B("**Not attempted:** pretrained chemical language-model embeddings (download blocked), and multi-seed averaging of the graph networks (about 3 extra CPU-hours; a script is ready)."));
children.push(B("**Incident:** the first test-evaluation run crashed because of a programming error in the blend-loading code, *before* any test prediction was made. The error was fixed and the evaluation re-run with the identical locked selection. Both attempts are logged (logs/DECISIONS.log)."));
children.push(B("**Not persisted:** the individual models from hyperparameter-search trials and cross-validation parts. They were used only for scoring; every trial’s settings are logged and can be re-created exactly. All 27 final configurations were saved and verified (Section 9)."));

// ---------- 8 Results (landscape section built separately)
const resultsSection = [
  H1("8. Results Table"),
  P("All 27 final configurations, sorted by cross-validation RMSE, which was the pre-registered selection criterion. **★ blend_nnls** is the final, pre-registered model; its row and the statistically tied blend_nnls_family are shaded. “CV RMSE” is on the in-domain training molecules, ± the standard deviation across the 5 cross-validation parts. Test columns are the single held-out evaluation on 1,804 in-domain test molecules, with 95% cluster-bootstrap ranges in brackets. All errors are in LogS (log₁₀ mol/L) units. “(trained on v2)” marks models trained without out-of-domain entries.", { size: 20 }),
  table(["Model / configuration", "Features", "Key hyperparameters", "CV RMSE ± SD", "Test RMSE [95% CI]", "Test MAE", "Test R² [95% CI]"],
    resRows, [2750, 2250, 2900, 1250, 1650, 900, 1650], { size: 16, highlight: ri => ri < 2 }),
  gap(),
  P("**Reading the table.** The two NNLS blends are statistically indistinguishable: the paired-bootstrap difference range is −0.006 to +0.001, and in 90% of resamples the per-family blend was at least as good. blend_nnls beat the best single model (lgbm_desc_mordred) by 0.034 RMSE, with a 95% range of 0.016–0.051 (p < 0.001). It beat every single model with p ≤ 0.004 (molecule resampling) and p ≤ 0.026 (cluster resampling). Test scores are uniformly about 0.09 better than cross-validation scores. ridge_desc’s large ± reflects one unstable cross-validation part (RMSE 4.27, driven by a few metal complexes). The **ranking** of models is very stable between the two (rank correlation 0.94), but the absolute test numbers should be read with Section 10 in mind.", { size: 20 }),
  P("**Where the final model errs** (in-domain test, blend_nnls): RMSE 0.58 for molecules closely resembling training compounds (similarity ≥ 0.6), 0.80 for moderate resemblance, and 0.93 for dissimilar molecules (< 0.4). It was 1.06 for very insoluble compounds (LogS < −6), 0.98 for salts, and 1.80 for the 48 out-of-domain test entries. Ring-free molecules were predicted slightly *better* than ring-containing ones (0.70 vs 0.83).", { size: 20 }),
  ...(EXT ? [P("**Headline models on the external benchmarks** (full details, validity checks and confidence intervals in Section 12). No model was retrained or re-tuned for these sets. SC1 uses the name-corrected file.", { size: 20 }),
    table(["Model", "Own test RMSE / R² (n = 1,804)", "SC1 RMSE / R² (n = 28)", "SC2-tight RMSE / R² (n = 100)", "SC2-loose RMSE / R² (n = 32)", "Bias on SC1 / SC2-t / SC2-l"],
      ["blend_nnls", "blend_nnls_family", "r2_blend_nnls_ergfix"].map(m => { const g = k => EXT.rows.find(r => r.model === m && r.set === k);
        return [m, `${f3(g("SC2_tight").test_rmse)} / ${f3(g("SC2_tight").test_r2)}`, `${f3(g("SC1").rmse)} / ${f3(g("SC1").r2)}`, `${f3(g("SC2_tight").rmse)} / ${f3(g("SC2_tight").r2)}`, `${f3(g("SC2_loose").rmse)} / ${f3(g("SC2_loose").r2)}`,
          `+${g("SC1").bias.toFixed(2)} / +${g("SC2_tight").bias.toFixed(2)} / +${g("SC2_loose").bias.toFixed(2)}`]; }),
      [2600, 2400, 2300, 2400, 2400, 2500], { size: 16 })] :
  [P("**External SC benchmarks (secondary check, not used for any decision).** On SC2-tight (100 drugs) blend_nnls scored RMSE 0.98 and R² 0.40. SC2 reports *intrinsic* solubility (the solubility of the uncharged form) for mostly ionisable drugs, which differs systematically from AqSolDB’s measurements, and it covers a narrow range, so R² is low for every model. SC1 results are not interpretable until the file is corrected.", { size: 20 })]),
];

// ---------- 9 Tools
const tools = [
  H1("9. Tools and Versions"),
  P("All work ran on a cloud Linux machine with 2 CPU cores and no GPU, using Python 3.11.15. The exact versions are listed in pipeline/requirements.txt."),
  table(["Library", "Version", "Used for"], [
    ["RDKit", "2026.03.6", "Reading structures, standardisation, InChIKeys, descriptors, fingerprints, clustering"],
    ["mordredcommunity", "2.0.7", "Mordred descriptors"],
    ["scikit-learn", "1.8.0", "Ridge, SVR, random forest, extra trees, kNN, MLP, cross-validation, preprocessing"],
    ["LightGBM", "4.7.0", "Gradient-boosted trees"],
    ["XGBoost", "3.2.0", "Gradient-boosted trees"],
    ["CatBoost", "1.2.10", "Gradient-boosted trees"],
    ["PyTorch", "2.14.0", "Graph neural networks (written from scratch, no graph library)"],
    ["NumPy / pandas / SciPy", "2.4.4 / 3.0.2 / 1.17.1", "Numerics, tables, statistics, NNLS blend weights"],
    ["joblib", "1.5.3", "Saving scikit-learn models"],
  ], [2400, 2200, 4760]),
  gap(),
  P("**Reproducibility.** Every random step used a fixed, recorded seed: split candidates 42–241 (181 chosen), cross-validation 42, models 42, hyperparameter sampling 123, bootstrap 7/8. Every intermediate dataset is saved as a separate file. Every saved model folder contains the model file, any preprocessing object, the exact featurisation code, and a prediction script. All 27 were reloaded in a fresh process and reproduced their recorded predictions to within 10⁻¹⁵."),
];

// ---------- 10 Limitations

// ---------- 11 Round 2 (optional)
let r2Section = [];
if (r2rows) {
  newList();
  const R2FEAT = { "r2:desc+avalon+erg": "RDKit desc + Avalon + ErG", "r2:desc+atompair+torsion": "RDKit desc + atom-pair + torsion",
    "r2:desc+crippen_types+packing": "RDKit desc + Crippen types + packing proxies", "r2:desc+mordred+crippen_types+packing": "RDKit + Mordred + Crippen + packing",
    "r2:mol2vec": "mol2vec embedding", "r2:desc+mordred+mol2vec": "RDKit + Mordred + mol2vec", "r2:desc+mordred+avalon+erg": "RDKit + Mordred + Avalon + ErG", "r2:desc+avalon+erg2": "RDKit desc + Avalon + ErG (deterministic)", "r2:desc+mordred+avalon+erg2": "RDKit + Mordred + Avalon + ErG (deterministic)" };
  const R2HP = { r2_blend_nnls: "NNLS weights over 41 models (18 used)", blend_nnls: "Round-1 final (reference)", r2_stack_ridge: "Ridge stacker, alpha by inner CV",
    r2_stack_lgbm_meta: "LightGBM stacker, 400 trees, 15 leaves, + meta features", r2_lgbm_dm_huber: "LightGBM, Huber loss (alpha 1)",
    r2_catboost_dm: "3,000 trees, depth 7", r2_svr_dm: "C = 10, gamma = 1/n_features", r2_svr_m2v: "C = 10, gamma = 1/n_features",
    r2_gnn_hybrid_ens3: "3 seeds (42, 43, 44) averaged", r2_gnn_hybrid_ens2: "2 seeds averaged", r2_gnn_mpnn_ens2: "2 seeds averaged",
    gnn_hybrid_s43: "single seed 43", r2_blend_nnls_ergfix: "As r2_blend_nnls; ErG models replaced by deterministic erg2", gnn_hybrid_s44: "single seed 44", gnn_mpnn_s43: "single seed 43" };
  const FAILS = new Set(["r2_lgbm_d_av_erg", "r2_lgbm_desc_mordred_avalon_erg"]);
  const r2t = r2rows.map(r => [(r.name === "r2_blend_nnls" ? "★ " : "") + r.name + (FAILS.has(r.name) ? " (reproducibility FAIL, superseded)" : ""), R2FEAT[r.fs] || FEAT[r.fs] || r.fs,
    R2HP[r.name] || (r.name.startsWith("r2_lgbm") ? "LightGBM, round-1 settings" : ""),
    `${f3(r.cv)} ± ${f3(r.cvsd)}`, `${f3(r.rmse)} [${f3(r.rci[0])}–${f3(r.rci[1])}]`, f3(r.mae), `${f3(r.r2)} [${f3(r.r2ci[0])}–${f3(r.r2ci[1])}]`,
    r.name === "blend_nnls" ? "—" : `${r.d >= 0 ? "+" : ""}${f3(r.d)} (p = ${r.p.toFixed(3)})`]);
  r2Section = [
    H1("11. Round 2: Further Exploration (October 2026)"),
    P("The goal of this round was to raise R² further, while keeping exactly the same data, split, cross-validation folds, leakage checks and model-saving standard. Every choice was again made with cross-validation on training data only. The test set was read **once**, after a new locked selection file was written (results/r2_07_preregistration.json)."),
    H2("11.1 What the errors looked like (training data only)"),
    B("**Extremes are pulled toward the middle:** very insoluble compounds are predicted too soluble and very soluble ones too insoluble. This is typical when measurements are noisy, and it cannot be fixed by a simple rescaling."),
    B("**Larger and more aromatic molecules are harder** (RMSE about 1.2–1.3 above molecular weight 350, against 0.6 below 150). This is consistent with the missing melting-point (crystal-packing) information, which the data does not contain."),
    B("**Many of the worst errors are data problems:** wrong structures (for example a pesticide recorded as the much simpler molecule propanamide) and implausible values. The worst 5% of molecules account for 47% of the squared error."),
    B("**Silicon compounds that react with water** (alkoxysilanes) are predicted very poorly (RMSE 2.2 on 32 molecules). The measured value probably reflects their breakdown products. This is noted as a caution for users rather than used to redefine the dataset, so the test set stays comparable."),
    H2("11.2 What was tried"),
    TERM("New fingerprints", "Avalon substructure counts, ErG “pharmacophore” fingerprints (positions of hydrogen-bond donors, acceptors and charges), atom-pair and topological-torsion fingerprints (which describe pairs of atoms and their distance through the bonds)."),
    TERM("Group-contribution and packing features", "Counts of the Crippen atom types used in classic logP estimation, plus ten simple proxies for how well a molecule might pack in a crystal (symmetry, flatness and conjugation, fused rings, amide groups)."),
    TERM("Pretrained chemical embedding (mol2vec)", "A published model that learned, from millions of unlabelled molecules, a 300-number summary of each chemical substructure; a molecule’s embedding is the sum over its substructures. The larger Hugging Face models are still blocked, but this one could be downloaded."),
    TERM("Robust loss (Huber)", "A training objective that limits how much a few very wrong labels can pull the model, motivated by the data problems above."),
    TERM("Seed-averaged graph networks", "The same neural network trained 2–3 times from different random starting points, with the predictions averaged; this reduces the noise of any single training run."),
    TERM("Learned combiners (stacking)", "Instead of fixed blend weights, a second model learns how to combine the models’ predictions. One version is a ridge regression; another is a small tree model that also sees molecule size, polarity, salt status and novelty. Two further CV-only variants used separate blend weights per molecule type and a residual-correction model."),
    H2("11.3 What was found"),
    N("**Single-model features are saturated.** Descriptor-based tree models all land near CV RMSE 0.92–0.93. Avalon + ErG helped RDKit descriptors (0.941 → 0.929, better in all five folds) but added nothing once Mordred descriptors were present. The Huber loss, Crippen/packing features and mol2vec (alone 1.11; combined 0.927) gave no gain. CatBoost matched LightGBM."),
    N("**Seed averaging clearly helps the graph networks:** the hybrid network went from 0.986 (1 seed) to 0.959 (2 seeds) to 0.950 (3 seeds), and the graph-only network from 1.020 to 0.988."),
    N("**Simple NNLS blending is still the best way to combine.** NNLS over the enlarged pool of 41 models reached nested-CV RMSE 0.883, against 0.887 before. The ridge stacker (0.892), the tree-model stacker (0.914), residual correction (0.891–0.895) and type-specific weights (0.886–0.889) were all worse; the more flexible combiners overfit."),
    N("**Test result for the pre-registered candidate (r2_blend_nnls):** RMSE 0.793 and R² 0.882, against 0.799 and 0.880 for blend_nnls. The paired cluster-bootstrap difference is −0.006 (95% range −0.014 to +0.002, one-sided p = 0.079). That **does not meet** the pre-registered 0.05 threshold. With molecule-level resampling p = 0.048, and on the full test set p = 0.032, but neither was the pre-registered test. The honest reading is a small, consistent but unconfirmed improvement. **blend_nnls remains the confirmed best configuration.**"),
    N("**A reproducibility bug was found and fixed.** All round-2 models were saved with their featurisation code and preprocessing objects, then reloaded in a fresh process. 18 of 20 reproduced their recorded results exactly. Two did not: r2_lgbm_d_av_erg and r2_lgbm_desc_mordred_avalon_erg, whose predictions differed by up to 0.1 LogS on a few molecules. The cause was RDKit’s ErG fingerprint, whose output depends on calculations previously run on the same molecule object. A deterministic version (computed on a fresh copy of the molecule) was written, and the two models and the blend were retrained (*_erg2 and r2_blend_nnls_ergfix). Each was evaluated once on the test set and reloaded twice in separate processes with identical results, so all 3 now pass. The rebuilt blend gives the same answer (test RMSE 0.793, R² 0.882, p = 0.083 against blend_nnls). The two original ErG models are kept but marked as failing reproducibility; round-1 models never used ErG and are unaffected."),
    H2("11.4 Still worth investigating"),
    B("A documented audit of label and structure errors, for example cross-checking names against structures in PubChem. This needs network access and must be done without looking at test performance."),
    B("A melting-point model or dataset to supply the missing crystal-packing term."),
    B("Larger pretrained chemical language models once they can be downloaded; 5–10 seeds for both graph networks."),
    B("Repeated train/test splits, to measure how much of the persistent CV-versus-test gap is luck."),
  ];

  r2Section.push(H2("11.5 Round-2 results table"));
  r2Section.push(P("Sorted by cross-validation RMSE on in-domain training molecules (± SD across folds). Test = the single evaluation on the 1,804 in-domain test molecules, with 95% cluster-bootstrap ranges. The last column is the paired difference in test RMSE against blend_nnls; p is the probability that the model is not better than blend_nnls. Shaded: the pre-registered round-2 candidate, its reproducible rebuild, and the round-1 reference. The bug-fix replacements (*_erg2, r2_blend_nnls_ergfix) were evaluated once each, under their own pre-registration.", { size: 20 }));
  r2Section.push(table(["Model / configuration", "Features", "Key hyperparameters", "CV RMSE ± SD", "Test RMSE [95% CI]", "Test MAE", "Test R² [95% CI]", "Δ RMSE vs blend_nnls"],
    r2t, [2550, 2350, 2500, 1200, 1600, 800, 1550, 1700], { size: 15, highlight: ri => ri < 3 }));
}


// ---------- 12 External Benchmark Evaluation (optional)
let extSection = [];
if (EXT) {
  const SETN = { SC1: "SC1 (name-corrected)", SC2_tight: "SC2-tight", SC2_loose: "SC2-loose", SC1_as_provided: "SC1 as provided" };
  const HEADS = ["blend_nnls", "blend_nnls_family", "r2_blend_nnls_ergfix"];
  const g = (m, k) => EXT.rows.find(r => r.model === m && r.set === k);
  const sgn = x => (x >= 0 ? "+" : "") + x.toFixed(2);
  const pct = x => Math.round(100 * x) + "%";
  const V = EXT.validity, A = EXT.analysis, S = EXT.sanity;
  const statRow = r => [r.model, String(r.n), `${f3(r.rmse)} [${r.rmse_ci[0].toFixed(2)}–${r.rmse_ci[1].toFixed(2)}]`, f3(r.mae),
    `${f3(r.r2)} [${r.r2_ci[0].toFixed(2)}–${r.r2_ci[1].toFixed(2)}]`, sgn(r.bias), r.pearson.toFixed(2), r.spearman.toFixed(2), pct(r.w05), pct(r.w10)];
  const statHdr = ["Model", "n", "RMSE [95% CI]", "MAE", "R² [95% CI]", "Bias", "Pearson r", "Spearman ρ", "Within 0.5", "Within 1.0"];
  const statW = [2900, 600, 2000, 900, 2000, 1000, 1100, 1200, 1100, 1100];
  const dec = (m, k) => A.decomposition.find(d => d.model === m && d.set === k);
  const ion = (k, c) => A.ionization.find(d => d.model === "blend_nnls" && d.set === k && d.ion_class === c);
  const sim = (k, b) => A.similarity.find(d => d.model === "blend_nnls" && d.set === k && d.band === b);
  newList();
  extSection = [
    H1("12. External Benchmark Evaluation"),
    P("This section tests the headline models on three external benchmark sets supplied with the project: **SC1**, the test compounds of the first Solubility Challenge; **SC2-tight** (Set 1 of the second Solubility Challenge: 100 drugs with low measurement scatter); and **SC2-loose** (Set 2: 32 drugs with higher measurement scatter). This is a single, fixed evaluation. **No model was trained, tuned, re-weighted or swapped for it, and none will be changed because of these results.** Each set is evaluated separately and never pooled, because the sets differ in origin and measurement quality."),
    P("**Models evaluated.** The three headline models are blend_nnls (the confirmed best model), blend_nnls_family (the recommended UI model) and r2_blend_nnls_ergfix (the best reproducible round-2 model). All 21 of their component models were evaluated as well; their results are in Appendix A. Every model was loaded fresh, in a separate Python process, from its saved files (model file, preprocessor and featurisation code), never from memory. All 24 loaded and ran successfully, so no headline model is missing. The superseded r2_blend_nnls was deliberately not evaluated, because two of its components are not reproducible (Section 11); its reproducible rebuild r2_blend_nnls_ergfix stands in for it."),
    H2("12.1 Validity checks (done before computing any statistic)"),
    table(["Check", "How it was done", "Result"], [
      ["1. No benchmark compound in any model’s training data", "For every evaluated model, the training molecules were collected: dataset v1 or v2, plus, for blends, the in-domain training rows used to fit the weights. Each was compared with every benchmark compound by full InChIKey, by InChIKey connectivity block (ignoring stereochemistry and protonation), and by the connectivity block of the salt-stripped parent. Both the file’s InChIKeys and InChIKeys recomputed from its SMILES were used.",
        Object.values(V.any_overlap).some(x => x) ? "OVERLAP FOUND (see results/external/ext_validity.json)" : "**No overlap for any of the 24 models in any set.** The benchmark compounds are also absent from our own test set. This is expected, since all SC compounds were removed from AqSolDB before splitting (Section 2.2)."],
      ["2. Benchmarks not used for any decision", "Every script that mentions SC1 or SC2 was reviewed: " + V.code_refs.map(f => f.replace("scripts/", "")).join(", ") + ".",
        "They were used for (a) removing overlapping molecules from AqSolDB before any modelling, which uses identities only and never the SC values; (b) leakage audits; and (c) a secondary external check, run after each round’s test evaluation and reported in Sections 8 and 11. **No model choice, hyperparameter, early-stopping point, blend weight or domain rule used SC values.** One caveat: the round-1 SC results were seen before round 2 began. Round 2 was motivated by training-set error analysis, not by the SC results, but this cannot be proven formally."],
      ["3. Target units", "The actual values of each solubility column were inspected.",
        `SC1’s column is named “logS0_uM”, but its values run from ${V.units.SC1.min.toFixed(2)} to ${V.units.SC1.max.toFixed(2)}. Read as log10(µM), salicylic acid (−1.93) would have a solubility near 2 ng/mL, about a million times below its known value. Read as log10(mol/L), it matches. SC2 (“logS0”) spans ${V.units.SC2_tight.min.toFixed(2)} to ${V.units.SC2_tight.max.toFixed(2)} (tight) and ${V.units.SC2_loose.min.toFixed(2)} to ${V.units.SC2_loose.max.toFixed(2)} (loose), also log10(mol/L). **No unit conversion was applied.**`],
      ["4. Counts, parsing and alignment", "Counts were compared with the published set sizes. Every SMILES was parsed, and its InChIKey recomputed and compared with the file. SC1 was checked for name–structure alignment.",
        `SC1: 32 rows, matching the 32 test compounds of the first Solubility Challenge. ${V.excluded.SC1.length} rows have no numeric value (“TSTM”, too soluble to measure: ${V.excluded.SC1.map(e => e.name.trim()).join(", ")}) and cannot be scored, giving **n = ${V.counts.SC1.evaluable}**. SC2-tight: **n = ${V.counts.SC2_tight.evaluable}** of 100. SC2-loose: **n = ${V.counts.SC2_loose.evaluable}** of 32. Both SC2 sets match their published sizes. No SMILES failed to parse, and every recomputed InChIKey matched the file. **8 SC1 rows had their structures out of step with their names**; these were corrected (Section 2.2 and below).`],
    ], [2300, 4900, 7700], { size: 16 }),
    gap(),
    P(`**The SC1 correction in detail.** In SC1_processed.csv, the CAS, SMILES and InChIKey columns of rows 1–6 belong to the next row up. The CAS numbers make this unambiguous: for example, the “Acebutolol” row carries CAS 10421-85-9, which is 2-chloromandelic acid. The Pseudoephedrine and Probenecid rows are swapped in the same way. Three independent facts show that the measured **values** belong to the **names**. First, the structures of benzthiazide and probenecid carry “too soluble to measure” as provided, while SC2-tight measures both near −4.8. Second, the TSTM labels sit on the names of the genuinely very soluble compounds (2-chloromandelic acid, pseudoephedrine). Third, after correction, the four compounds shared with SC2-tight agree closely for benzthiazide (−4.83 vs −4.84), probenecid (−4.63 vs −4.83) and folic acid (−5.25 vs −5.96). The exception is **indomethacin (SC1 −2.94 vs SC2-tight −5.48)**: its row was already aligned, and it was left unchanged but flagged. The structures were moved back to their names (data/ext_SC1_name_corrected.csv); no value was edited. For transparency, the as-provided file was scored too. There, blend_nnls gave RMSE ${f3(g("blend_nnls","SC1_as_provided").rmse)} and R² ${f3(g("blend_nnls","SC1_as_provided").r2)}, against ${f3(g("blend_nnls","SC1").rmse)} and ${f3(g("blend_nnls","SC1").r2)} after correction. This is consistent with the correction being right, but the correction was decided from the chemistry and the CAS evidence above, not from model accuracy.`),
    P(`**Sanity checks on the predictions.** No two of the 24 models gave identical predictions. Every model produced at least ${S.n_distinct_predictions_per_model_min} distinct values for the 156 distinct structures. Each headline blend equals the weighted sum of its saved components to within ${Math.max(S.blend_nnls_equals_weighted_sum_of_components_maxabs, S.blend_nnls_family_equals_weighted_sum_of_components_maxabs, S.r2_blend_nnls_ergfix_equals_weighted_sum_of_components_maxabs).toExponential(0)}. Seven compounds appear in two sets (SC1 and SC2-tight share ${V.between["SC1&SC2_tight"].join(", ")}; SC1 and SC2-loose share ${V.between["SC1&SC2_loose"].join(", ")}). Each received the same prediction in both sets, as it should, while the sets as a whole received different predictions.`),
    H2("12.2 Results by benchmark set"),
    P("Bias is the mean of (predicted − measured); a positive value means the model predicts the drug to be more soluble than measured. “Within 0.5 / 1.0” is the share of compounds predicted within 0.5 or 1.0 log units. The 95% intervals come from 2,000 bootstrap resamples of compounds (seed 20261011). **R² on 28–100 compounds is unstable** (note the width of its intervals). It also depends strongly on how spread out each set’s values are, so RMSE and the correlations are the more reliable summaries here.", { size: 20 }),
    ...["SC1", "SC2_tight", "SC2_loose"].flatMap(k => [
      new Paragraph({ children: [t(`Table 12.${k === "SC1" ? 1 : k === "SC2_tight" ? 2 : 3}. ${SETN[k]}, n = ${g("blend_nnls", k).n}`, { bold: true, size: 20 })], spacing: { before: 160, after: 60 } }),
      table(statHdr, HEADS.map(m => statRow(g(m, k))), statW, { size: 16 }),
    ]),
    gap(),
    P(`**Measurement noise.** SC2 reports a standard deviation for every compound. On SC2-tight the typical measurement error is small (root-mean-square SD ${g("blend_nnls","SC2_tight").noise.toFixed(2)}), so it explains almost none of the model error (RMSE ${f3(g("blend_nnls","SC2_tight").rmse)}). On SC2-loose the typical SD is ${g("blend_nnls","SC2_loose").noise.toFixed(2)} log units. If errors are independent, noise accounts for only about ${pct(g("blend_nnls","SC2_loose").noise ** 2 / g("blend_nnls","SC2_loose").rmse ** 2)} of blend_nnls’s mean squared error there, leaving about ${g("blend_nnls","SC2_loose").beyond.toFixed(2)} log units of RMSE attributable to the model. Only ${pct(g("blend_nnls","SC2_loose").w2sd)} of SC2-loose compounds (and ${pct(g("blend_nnls","SC2_tight").w2sd)} of SC2-tight compounds) were predicted within two of their own measurement SDs. Higher measurement noise is therefore not the main reason SC2-loose is harder.`),
    H2("12.3 Comparison with our own test set"),
    table(["Model", "Own in-domain test RMSE / R²", "SC1 RMSE (ratio to test)", "SC2-tight RMSE (ratio)", "SC2-loose RMSE (ratio)", "Verdict"],
      HEADS.map(m => [m, `${f3(g(m,"SC1").test_rmse)} / ${f3(g(m,"SC1").test_r2)}`, ...["SC1","SC2_tight","SC2_loose"].map(k => `${f3(g(m,k).rmse)} (×${(g(m,k).rmse / g(m,k).test_rmse).toFixed(2)})`), "Degrades on all three sets"]),
      [2700, 2600, 2400, 2400, 2400, 2400], { size: 16 }),
    gap(),
    P(`All three headline models **generalise worse to the benchmarks than to our own held-out test set**. RMSE is about 1.2× higher on SC2-tight, about 1.4× higher on SC1, and about 2.1× higher on SC2-loose. The three models rank almost identically and stay within about 0.05 RMSE of each other on every set; the benchmarks do not separate them. Part of the low R² comes from the narrow spread of values in SC1 and SC2-tight (SD ${A.range_restriction.SC1.benchmark_logS_sd.toFixed(2)} and ${A.range_restriction.SC2_tight.benchmark_logS_sd.toFixed(2)}, against ${A.own_test_logS_sd.toFixed(2)} in our test set). Even with its test-set RMSE, blend_nnls would reach R² of only about ${A.range_restriction.SC2_tight.r2_if_rmse_equal_to_test.toFixed(2)} on SC2-tight.`),
    P("**Possible reasons (hypotheses, not findings):**", { size: 22 }),
    B(`**H1. Different quantity measured.** The benchmarks report intrinsic solubility (the uncharged form). Many AqSolDB values are apparent solubilities in water, which are higher for compounds that ionise. This predicts a positive bias, larger for acids and bases. That is what was observed: the bias is positive on every set (blend_nnls ${sgn(g("blend_nnls","SC1").bias)} / ${sgn(g("blend_nnls","SC2_tight").bias)} / ${sgn(g("blend_nnls","SC2_loose").bias)}) and makes up ${pct(dec("blend_nnls","SC1").bias_share_of_mse)} / ${pct(dec("blend_nnls","SC2_tight").bias_share_of_mse)} / ${pct(dec("blend_nnls","SC2_loose").bias_share_of_mse)} of the mean squared error. On SC2-tight the bias is ${sgn(ion("SC2_tight","acid").bias)} for acids (n = ${ion("SC2_tight","acid").n}) and ${sgn(ion("SC2_tight","base").bias)} for bases (n = ${ion("SC2_tight","base").n}), against ${sgn(ion("SC2_tight","neutral/weak").bias)} for neutral or weakly ionisable compounds (n = ${ion("SC2_tight","neutral/weak").n}). The ionisation classes come from simple substructure rules, not measured pKa values.`),
    B(`**H2. Drugs are less like our training data.** The benchmark drugs are larger and more complex than typical AqSolDB entries. Median similarity to the nearest training molecule is ${A.range_restriction.SC1.median_nn_sim.toFixed(2)} (SC1), ${A.range_restriction.SC2_tight.median_nn_sim.toFixed(2)} (SC2-tight) and ${A.range_restriction.SC2_loose.median_nn_sim.toFixed(2)} (SC2-loose). Error rises steeply as similarity falls: on SC2-tight, RMSE is ${f3(sim("SC2_tight","0.6-1.0").rmse)} for similarity ≥ 0.6 (n = ${sim("SC2_tight","0.6-1.0").n}), close to our test-set level, against ${f3(sim("SC2_tight","0-0.4").rmse)} below 0.4 (n = ${sim("SC2_tight","0-0.4").n}). All benchmark compounds pass the applicability-domain rules, so those rules alone do not flag this risk; the similarity score does.`),
    B("**H3. Regression toward the mean.** The worst errors are all very insoluble, large, lipophilic drugs predicted as too soluble (amiodarone, clofazimine, itraconazole, terfenadine, tamoxifen, cisapride, aripiprazole). This is the same weakness seen on very insoluble compounds in our own test set, and SC2-loose is rich in such drugs."),
    H2("12.4 Worst-predicted compounds"),
    table(["Set", "Compound", "Measured", "blend_nnls", "blend_nnls_family", "r2_blend_nnls_ergfix", "Error (blend_nnls)", "Meas. SD", "Ionisation class", "MW", "Similarity"],
      EXT.worst.map(w => [SETN[w.set], w.name, w.logS_exp.toFixed(2), w.pred_blend_nnls.toFixed(2), w.pred_blend_nnls_family.toFixed(2), w.pred_r2_blend_nnls_ergfix.toFixed(2), sgn(w.error), (w.exp_sd === null || isNaN(w.exp_sd)) ? "—" : Number(w.exp_sd).toFixed(2), w.ion_class, Math.round(w.MW), w.nn_sim.toFixed(2)]),
      [1700, 1900, 1050, 1150, 1450, 1650, 1300, 950, 1450, 750, 1000], { size: 15 }),
    gap(),
    P("**Pattern.** Of the 18 worst predictions, 17 are under-predictions of insolubility (predicted too soluble). Most are large (11 of 18 above MW 400), most have low similarity to the training set (13 of 18 at ≤ 0.47), and 15 of 18 are ionisable (basic amines, or carboxylic and sulfonyl acids). The one large error in the opposite direction is SC1’s indomethacin. Its SC1 value (−2.94) disagrees with SC2-tight’s measurement of the same drug (−5.48), which the models are much closer to. This points to a data problem rather than a model problem, but was not verified further."),
    H2("12.5 What can and cannot be concluded"),
    B("With n = 28, 100 and 32, the confidence intervals are wide (e.g. SC1 RMSE 95% CI " + `${g("blend_nnls","SC1").rmse_ci[0].toFixed(2)}–${g("blend_nnls","SC1").rmse_ci[1].toFixed(2)}` + "). The benchmarks cannot rank the headline models against each other, and small differences between them should not be interpreted."),
    B("The benchmarks clearly show a systematic over-prediction of solubility for drug-like, ionisable and very insoluble compounds. They also show that error grows quickly as similarity to the training data falls."),
    B("Benchmark R² values are not comparable with test-set R² values, because of the different value ranges and because a different quantity (intrinsic solubility) is measured."),
    P("**Recommendations (not applied).** These are suggestions only. None were implemented, so the results above stay a genuine one-time check."),
    N("Report the nearest-neighbour similarity with every prediction in a UI, and warn when it is below 0.4."),
    N("For drug-like ionisable compounds, state that predictions approximate AqSolDB-style water solubility, not intrinsic solubility. A pKa-based conversion could be explored, then validated on new data."),
    N("Re-check indomethacin and the misaligned SC1 rows against the original SC1 publication before SC1 is used again."),
  ];
  // Appendix A: all models
  const appRow = r => [r.model + (r.headline ? " (headline)" : ""), `${f3(r.rmse)} [${r.rmse_ci[0].toFixed(2)}–${r.rmse_ci[1].toFixed(2)}]`, f3(r.mae), `${f3(r.r2)} [${r.r2_ci[0].toFixed(2)}–${r.r2_ci[1].toFixed(2)}]`, sgn(r.bias), r.pearson.toFixed(2), r.spearman.toFixed(2), pct(r.w10), r.test_rmse === null ? "—" : f3(r.test_rmse)];
  extSection.push(new Paragraph({ children: [t("")], pageBreakBefore: true }));
  extSection.push(H1("Appendix A. External benchmark results for every evaluated model"));
  extSection.push(P("These cover all 24 evaluated models: the 3 headline blends and their 21 components. Columns are as in Section 12; the last column is the model’s own in-domain test RMSE. Rows are sorted by RMSE. Per-compound predictions are in results/external/pred_<model>__<set>.csv; all statistics are in results/external/ext_metrics.csv; the scripts are scripts/ext_01_benchmark_eval.py, ext_02_analysis.py and ext_03_report_rows.py.", { size: 20 }));
  for (const k of ["SC1", "SC2_tight", "SC2_loose", "SC1_as_provided"]) {
    const rs = EXT.rows.filter(r => r.set === k).sort((a, b) => a.rmse - b.rmse);
    extSection.push(new Paragraph({ children: [t(`A.${["SC1", "SC2_tight", "SC2_loose", "SC1_as_provided"].indexOf(k) + 1} ${SETN[k]}, n = ${rs[0].n}` + (k === "SC1_as_provided" ? " (structures misaligned; for reference only)" : ""), { bold: true, size: 20 })], spacing: { before: 200, after: 60 } }));
    extSection.push(table(["Model", "RMSE [95% CI]", "MAE", "R² [95% CI]", "Bias", "Pearson r", "Spearman ρ", "Within 1.0", "Own test RMSE"], rs.map(appRow),
      [3600, 2000, 1000, 2000, 1000, 1100, 1200, 1100, 1400], { size: 15, highlight: ri => rs[ri].headline }));
  }
}

newList();
const lim = [
  H1("10. Known Limitations and Next Steps"),
  N("**Test results look better than cross-validation for every model** (by about 0.09 RMSE). Plausible causes are that final models are trained on 25% more data, and chance in which clusters landed in the test set. Neither has been tested (for example with a learning curve or repeated splits). A realistic expectation for new chemistry is RMSE 0.85–0.90."),
  N("**Uncertain labels.** AqSolDB combines many sources, and some values are clearly wrong. For example, a simple sodium sulfonate salt is listed at LogS −8.4. No reliability information per entry was available, so the lowest error any model could achieve is unknown."),
  N("**Pretrained chemical language-model embeddings were not tested** (download blocked). This is the largest untested family of representations." + (r2rows ? " (Round 2 tested the smaller pretrained mol2vec embedding, with no gain; the larger models remain untested; Section 11.)" : "")),
  N("**The graph networks are under-developed:** a single random seed, one architecture and a fixed 80 training epochs. Averaging several seeds usually helps and could change the blend." + (r2rows ? " (Round 2: 2–3 seeds improved the graph networks clearly in CV, but the final blend only marginally; Section 11.)" : "")),
  N("**Hyperparameter searches were small** (6–14 trials per model type; CatBoost used one setting). The flat LightGBM results suggest little is lost, but this is not proven."),
  N("**Two thresholds were reasoned but not stress-tested:** the 1-log-unit rule for duplicates and the 0.4 similarity threshold for clustering."),
  N("**Out-of-domain behaviour** is characterised by only 48 test molecules (RMSE 1.80). Predictions for metal complexes and mixtures should be treated as unreliable. Both blends include a small ridge-regression component, which can give extreme values for such inputs. Using the applicability-domain check alongside the blend is part of the recommendation, and a blend without ridge was not evaluated."),
  N(EXT ? "**SC1 file misalignment** (Section 2.2). It was corrected by re-aligning structures to names, supported by CAS numbers and agreement with SC2-tight, but not checked against the original publication. SC1 also lists indomethacin at −2.94, while SC2-tight gives −5.48; this value was left unchanged." : "**SC1 file misalignment** (Section 2.2). The pairing of structures and values is inferred from correlations, not confirmed against the original publication."),
  N("**The SC benchmarks measure a different quantity** (intrinsic solubility). A fair external comparison would need a correction for ionisation (pH/pKa)." + (EXT ? " Section 12 shows the expected sign: all models over-predict solubility on the benchmarks, more strongly for acids and bases than for neutral compounds." : "")),
  N("**Software sensitivity.** Descriptor values can change between RDKit or Mordred versions. The saved models are verified only with the versions listed in Section 9."),
  P("**Recommended next steps:** repeat the split with several seeds to measure how much the test score varies; add graph-network seed averaging and a pretrained embedding model once downloads are possible; fix and re-run the SC1 comparison; and evaluate a blend without the ridge component for use in a user interface."),
];

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "Claude (Cowork)", title: "SolubilityAI Scratch Pipeline — Experiment Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: "1F3864", font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, color: "2E75B6", font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    ...Array.from({ length: LIST + 1 }, (_, k) => k).map(k => ({ reference: "numbers" + k, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] })),
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1270, right: 1270 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("SolubilityAI experiment report · page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080", font: FONT })] })] }) },
      children },
    { properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 1000, bottom: 1000, left: 900, right: 900 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("SolubilityAI experiment report · page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080", font: FONT })] })] }) },
      children: resultsSection },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1270, right: 1270 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("SolubilityAI experiment report · page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080", font: FONT })] })] }) },
      children: [...tools, ...lim] },
    ...(r2rows ? [{ properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 1000, bottom: 1000, left: 900, right: 900 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("SolubilityAI experiment report · page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080", font: FONT })] })] }) },
      children: r2Section }] : []),
    ...(EXT ? [{ properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 1000, bottom: 1000, left: 900, right: 900 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("SolubilityAI experiment report · page ", { size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080", font: FONT })] })] }) },
      children: extSection }] : []),
  ],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
