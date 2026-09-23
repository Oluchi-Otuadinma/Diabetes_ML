# Diabetes_ML

Diabetes risk prediction on the [Pima Indians Diabetes dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) (768 x 9), with feature engineering, model training, and evaluation parallelised across a **Dask distributed cluster** (1 scheduler + 4 worker processes; multi-node-ready topology).

## Pipeline
1. **Quick data analysis** — dataset size, class balance (500/268), hidden missingness (biologically impossible zeros treated as missing: Insulin 48.7%, SkinThickness 29.6%, ...).
2. **Feature engineering** (on a worker) — 25 features: missingness flags, median imputation, insulin-resistance composites (Glucose x Age, Insulin x BMI, QUICKI proxy), clinical bins (ADA glucose, WHO BMI).
3. **Parallel training** — 30 jobs fanned out via `client.map`: RandomForest vs Logistic Regression x chi-squared feature selection (k=8/12/all) x 5 stratified folds; class imbalance handled with `class_weight="balanced"` + stratification.
4. **Evaluation** — per-fold scoring on workers (ROC-AUC, PR-AUC, balanced accuracy, F1), aggregated at the scheduler.

## Results (mean over 5 stratified folds)
| model | chi2 k | ROC-AUC | PR-AUC | Bal. Acc | F1 |
|---|---|---|---|---|---|
| LogReg | 12 | **0.835** | 0.719 | 0.746 | 0.672 |
| RF | 12 | 0.835 | **0.736** | **0.757** | **0.684** |
| RF | 8 | 0.832 | 0.739 | 0.760 | 0.688 |
| LogReg | 8 | 0.815 | 0.684 | 0.728 | 0.654 |

Top predictors (full-data RF refit): `Glucose_x_Age` (0.19), `Glucose` (0.13), `Insulin_BMI_Interaction` (0.12), `Pedigree_x_Glucose` (0.10), `QUICKI` (0.10).

## Run
```bash
pip install pandas scikit-learn numpy "dask[distributed]"
python pima_cluster_pipeline.py
```

## Files
- `pima_cluster_pipeline.py` — full cluster pipeline
- `pima-indians-diabetes.csv` — dataset
- `cv_results.csv` — per-fold results
- `feature_importance.csv` — ranked predictors
