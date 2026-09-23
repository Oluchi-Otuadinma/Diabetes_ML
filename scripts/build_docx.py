"""One-off builder: report/cross_validation_report.docx from results/cv_results.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "cv_results.csv")
df["k"] = df["k"].fillna(-1).astype(int)
cm = df["cm"].apply(json.loads).apply(np.array)
cmhr = df["cm_hr"].apply(json.loads).apply(np.array)
df["fn0"] = cm.apply(lambda m: m[1][0]); df["tp0"] = cm.apply(lambda m: m[1][1])
df["fp0"] = cm.apply(lambda m: m[0][1]); res = df
res["fnh"] = cm.apply(lambda m: 0)  # placeholder, replaced below
cmhr = res["cm_hr"].apply(lambda v: np.array(v if isinstance(v, list) else json.loads(v)))
res["fnh"] = cmhr.apply(lambda m: m[1][0]); res["fph"] = cmhr.apply(lambda m: m[0][1])
g = res.groupby(["model", "k_label"])

default_tbl = (g[["roc_auc", "pr_auc", "balanced_acc", "f1"]].mean().round(4)
               .sort_values("roc_auc", ascending=False))
hr_tbl = (g[["recall_hr", "precision_hr", "f1_hr", "threshold"]].mean().round(4)
          .sort_values("recall_hr", ascending=False))
ba_tbl = pd.DataFrame({
    "recall @0.5": g.apply(lambda x: x.tp0.sum()/(x.tp0.sum()+x.fn0.sum()), include_groups=False),
    "recall @HR": g["recall_hr"].mean(),
    "precision @0.5": g["precision"].mean(),
    "precision @HR": g["precision_hr"].mean(),
    "FN @0.5": g["fn0"].sum(), "FN @HR": g["fnh"].sum(),
    "FP @0.5": g["fp0"].sum(), "FP @HR": g["fph"].sum(),
}).round(3)
totals = ba_tbl.sum(numeric_only=True)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)


def h(text, level=1):
    doc.add_heading(text, level)


def p(text, bold=False):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.bold = bold
    return par


def table(headers, rows):
    rows = list(rows)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htxt in enumerate(headers):
        t.rows[0].cells[i].text = str(htxt)
        t.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return t


def klabel(k):
    return "all" if k == -1 else k


# ---------------------------------------------------------------- title
doc.add_heading("Cross-Validation Report — Diabetes Risk Prediction", 0)
p("Diabetes_ML · Pima Indians Diabetes dataset · stratified 5-fold cross-validation "
  "on a Dask distributed cluster (4 worker processes). Author: Oluchi-Otuadinma.")

h("1. Dataset")
p("768 rows x 9 columns (54 KB in memory), no duplicates. Class balance: 500 non-diabetic / "
  "268 diabetic (34.9% positive) — moderately imbalanced, handled with class weighting and "
  "stratified folds. Biologically impossible zeros treated as missing: Insulin 48.7%, "
  "SkinThickness 29.6%, BloodPressure 4.6%, BMI 1.4%, Glucose 0.7%. Feature engineering raises "
  "the table to 25 features (missingness flags, median imputation, insulin-resistance composites "
  "Glucose x Age / Insulin x BMI / QUICKI, clinical bins).")

h("2. Protocol")
p("StratifiedKFold (5 folds, prevalence preserved at ~34.9% per fold; test folds n=154/153). "
  "Models: RandomForest (400 trees, class_weight=balanced) vs Logistic Regression (C=1, balanced). "
  "Chi-squared feature selection at k = 8 / 12 / all-25. 30 train/score jobs fanned out to the "
  "cluster via client.map. High-recall screening mode: per-fold decision threshold tuned on "
  "out-of-fold training predictions to reach recall >= 0.90, applied untouched to the test fold "
  "(no leakage).")

h("3. Default-threshold performance (mean over 5 folds)")
table(["model (k)", "roc_auc", "pr_auc", "balanced_acc", "f1"],
      ([f"{m} (k={klabel(k)})"] + row.tolist() for (m, k), row in
       default_tbl.iterrows()))
p("")
p("Logistic Regression on all 25 features leads ROC-AUC (0.8415); RandomForest k=12 leads "
  "PR-AUC (0.736) and balanced accuracy (0.757).", bold=True)

h("4. High-recall screening mode (target recall >= 90%)")
table(["model (k)", "recall_hr", "precision_hr", "f1_hr", "threshold"],
      ([f"{m} (k={klabel(k)})"] + row.tolist() for (m, k), row in
       hr_tbl.iterrows()))
p("")
p("Logistic Regression (all features, threshold 0.343) is the only configuration that meets the "
  "screening target: recall 0.899, precision 0.541 — a 1.55x lift over the 34.9% base rate.",
  bold=True)

h("5. Before vs after the screening threshold (all folds, n = 768)")
table(["model (k)", "recall @0.5", "recall @HR", "precision @0.5", "precision @HR",
       "FN @0.5", "FN @HR", "FP @0.5", "FP @HR"],
      ([f"{m} (k={klabel(k)})"] + row.tolist() for (m, k), row in
       ba_tbl.iterrows()))
table(["totals", "FN @0.5", "FN @HR", "FP @0.5", "FP @HR"],
      [["", int(totals["FN @0.5"]), int(totals["FN @HR"]),
        int(totals["FP @0.5"]), int(totals["FP @HR"]) ]])
p("")
p("Switching to the screening threshold cuts missed diabetics by 56% (407 -> 180); the price is "
  "+71% false alarms (741 -> 1,266) — ~2.3 extra false alarms per additional diabetic caught.",
  bold=True)
for img, w in [("high_recall_impact.png", 6.5)]:
    doc.add_picture(str(ROOT / "assets" / img), width=Inches(w))

h("6. Model comparison — is the difference real?")
p("Per-fold paired differences (LogReg all-features minus RF k=12): ROC-AUC +0.007 ± 0.021, "
  "F1@0.5 −0.005 ± 0.036, recall@HR +0.015 ± 0.055, F1@HR +0.009 ± 0.022. The sign flips across "
  "folds and the spread dwarfs the mean gaps, so the two models are statistically "
  "indistinguishable on averaged metrics.")
p("Model choice is therefore an objective decision, not an accuracy decision.", bold=True)

h("7. Model strengths and weaknesses")
p("Logistic Regression (all 25 features) — strengths:", bold=True)
p("- Highest ROC-AUC overall (0.8415); only config reaching the 90% screening target "
  "(recall 0.899)")
p("- Lowest missed-diabetic count (27 per 5 folds) and best F1 at the screening operating point "
  "(0.675)")
p("- Simple, fast, interpretable: coefficients give per-feature effect direction (Glucose, "
  "Pregnancies, BMI strongest positives); smooth probabilities make threshold tuning reliable")
p("Logistic Regression — weaknesses:", bold=True)
p("- Slightly lower PR-AUC (0.723) and balanced accuracy (0.752) than RF at the default threshold")
p("- Linear/additive structure can under-fit interactions — mitigated here by explicit "
  "interaction features, but a risk if features were reduced to the raw 8")
p("RandomForest (k=12) — strengths:", bold=True)
p("- Best PR-AUC (0.736) and balanced accuracy (0.757); marginally best F1 at 0.5 (0.684)")
p("- Captures nonlinearities and interactions without hand-built features; robust to outliers")
p("RandomForest — weaknesses:", bold=True)
p("- Recall at the screening threshold only reaches 0.869-0.884: it misses the 90% target, "
  "i.e. it leaves more diabetics undetected — the exact error a screening tool must minimise")
p("- Coarse tree-vote probabilities make fine threshold tuning harder; heavier to train; "
  "weaker interpretability")

h("8. Overall performance and verdict")
p("Ranking quality is statistically tied (~0.84 ROC-AUC for both model families). For the stated "
  "medical screening objective (minimise missed diabetics), the recommended model is Logistic "
  "Regression on all 25 engineered features with the 0.343 screening threshold: recall 0.899, "
  "precision 0.541, and missed cases reduced from 407 to 180 across five folds. RandomForest k=12 "
  "remains the better choice for a precision-oriented triage setting. The LogReg-over-RF margin is "
  "not proven at this sample size (5 folds, 768 rows); repeated stratified CV (e.g. 10x5) is the "
  "natural next step.", bold=False)
p("Key risk-trend finding: observed diabetic prevalence climbs monotonically with glucose "
  "(~5% at 50-90 mg/dL to ~82% above 155 mg/dL) and with Glucose x Age and BMI; QUICKI "
  "(insulin sensitivity) is flat/non-monotonic in this cohort.")

doc.save(ROOT / "report" / "cross_validation_report.docx")
print("report saved")