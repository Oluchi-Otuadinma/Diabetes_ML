# Diabetes_ML

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Dask](https://img.shields.io/badge/dask-distributed-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-blue)

Diabetes risk prediction on the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (768 x 9), with feature engineering, model training, and evaluation **parallelised across a Dask distributed cluster** (1 scheduler + 4 worker processes — a multi-node-ready topology; swap `LocalCluster` for remote scheduler/worker addresses to span physical machines).

## 🎯 Project Overview

This project provides:
- **Quick Data Analysis** — dataset size, class balance (500/268 → 34.9% positive), and hidden missingness (biologically impossible zeros treated as missing: Insulin 48.7%, SkinThickness 29.6%, BloodPressure 4.6%, BMI 1.4%, Glucose 0.7%)
- **Parallel Feature Engineering** — 25 features built on a worker: missingness flags, median imputation, insulin-resistance composites (Glucose x Age, Insulin x BMI, QUICKI proxy), clinical bins (ADA glucose, WHO BMI)
- **Parallel Model Training** — 30 jobs fanned out via `client.map`: RandomForest vs Logistic Regression x chi-squared feature selection (k = 8/12/all) x 5 stratified folds
- **Evaluation** — per-fold scoring on workers (ROC-AUC, PR-AUC, balanced accuracy, F1), aggregated at the scheduler

## 📁 Project Structure

```
Diabetes_ML/
│
├── data/
│   └── pima-indians-diabetes.csv   # Dataset (768 rows x 9 cols)
│
├── src/
│   └── pipeline.py                 # Full cluster pipeline (analysis -> features -> training -> evaluation)
│
├── results/
│   ├── cv_results.csv              # Per-fold results (30 runs)
│   └── feature_importance.csv      # Ranked predictors (full-data RF refit)
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 📊 Results

Mean metrics over 5 stratified folds (class imbalance handled with `class_weight="balanced"` + stratified CV):

| model | chi² k | ROC-AUC | PR-AUC | Bal. Acc | F1 |
|---|---|---|---|---|---|
| **LogReg** | 12 | **0.835** | 0.719 | 0.746 | 0.672 |
| RF | 12 | 0.835 | **0.736** | **0.757** | **0.684** |
| RF | 8 | 0.832 | 0.739 | 0.760 | 0.688 |
| LogReg | 8 | 0.815 | 0.684 | 0.728 | 0.654 |

**Key findings:**

- RandomForest and Logistic Regression are statistically tied on ROC-AUC (~0.835); RF edges ahead on PR-AUC and balanced accuracy thanks to class weighting.
- chi²-selected `k=12` beat using all 25 features — a compact set of strong clinical predictors wins.
- Top predictors (full-data RF refit): `Glucose_x_Age` (0.19), `Glucose` (0.13), `Insulin_BMI_Interaction` (0.12), `Pedigree_x_Glucose` (0.10), `QUICKI` (0.10) — glucose-dominant, with insulin-resistance composites adding real signal over raw features.

## ⚙️ Setup

```bash
git clone https://github.com/Oluchi-Otuadinma/Diabetes_ML.git
cd Diabetes_ML
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python src/pipeline.py
```

The pipeline spins up the Dask cluster, runs the quick data analysis, distributes feature engineering, fans out 30 train/score jobs (2 models x 3 chi²-k x 5 folds), aggregates the metrics, and writes `results/cv_results.csv` + `results/feature_importance.csv`.

## 📄 License

Distributed under the [MIT License](LICENSE).
