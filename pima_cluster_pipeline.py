"""
Pima Indians Diabetes - multi-node cluster pipeline
====================================================
Parallelised feature engineering, model training, and evaluation on a
Dask distributed cluster (1 scheduler + N worker processes; each worker
is a separate OS process, so the topology is identical to a multi-node
setup - swap LocalCluster for remote Scheduler/Worker addresses to span
physical machines).

Steps:
  1. Quick analysis : dataset size, class balance, zero anomalies
  2. Feature eng.   : executed on the workers via client.submit
  3. Training       : RandomForest vs Logistic Regression, chi^2 feature
                      selection, stratified 5-fold CV fanned out to workers
  4. Evaluation     : per-fold metrics scored on workers, then aggregated
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from dask.distributed import Client, LocalCluster

DATA_PATH = "pima-indians-diabetes.csv"
N_WORKERS = 4           # worker processes (= "nodes" in this cluster)
THREADS_PER_WORKER = 2
RANDOM_STATE = 42

COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


# ---------------------------------------------------------------------------
# 1. Quick analysis
# ---------------------------------------------------------------------------
def quick_analysis(df: pd.DataFrame) -> dict:
    info = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "memory_kb": df.memory_usage(deep=True).sum() / 1024,
        "duplicates": int(df.duplicated().sum()),
        "class_counts": df["Outcome"].value_counts().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    zero_anom = {}
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        zero_anom[c] = {"zeros": int((df[c] == 0).sum()),
                        "pct": round(100 * (df[c] == 0).mean(), 1)}
    info["zero_anomalies"] = zero_anom
    return info


# ---------------------------------------------------------------------------
# 2. Feature engineering - executed ON THE WORKERS
# ---------------------------------------------------------------------------
def build_features(part: pd.DataFrame) -> pd.DataFrame:
    """Pure function of its input so it can run on any node."""
    X = part.drop(columns="Outcome").astype(float).copy()
    y = part["Outcome"]

    # Biologically impossible zeros are missing values
    impossible = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    X[impossible] = X[impossible].replace(0, np.nan)

    # Missingness flags + median imputation
    med = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0,
           "Insulin": 125.0, "BMI": 32.0}
    for c in impossible:
        X[f"{c}_was_missing"] = X[c].isna().astype(int)
    X = X.fillna(med)

    # Derived clinical features
    X["Glucose_BMI_Ratio"] = X["Glucose"] / (X["BMI"] + 1e-6)
    X["Insulin_Glucose_Ratio"] = X["Insulin"] / X["Glucose"]
    X["Insulin_BMI_Interaction"] = X["Insulin"] * X["BMI"]
    X["Glucose_x_Age"] = X["Glucose"] * X["Age"]
    X["Age_x_Pregnancies"] = X["Age"] * X["Pregnancies"]
    X["BP_x_BMI"] = X["BloodPressure"] * X["BMI"]
    X["Pedigree_x_Glucose"] = X["DiabetesPedigreeFunction"] * X["Glucose"]
    X["Pedigree_x_BMI"] = X["DiabetesPedigreeFunction"] * X["BMI"]

    # QUICKI-style insulin sensitivity proxy
    quicki = (155 - 0.7 * X["Age"] - 4.7 * X["BMI"]).clip(lower=1)
    X["QUICKI"] = 1.0 / (np.log(X["Glucose"].clip(lower=1)) + np.log(quicki))

    # Clinical bins: glucose (ADA), age, BMI (WHO)
    X["Glucose_bin"] = pd.cut(X["Glucose"], bins=[0, 99, 125, 1e9],
                              labels=[0, 1, 2]).astype(int)
    X["Age_bin"] = pd.cut(X["Age"], bins=[20, 30, 40, 50, 1e9],
                          labels=[0, 1, 2, 3]).astype(int)
    X["BMI_bin"] = pd.cut(X["BMI"], bins=[0, 18.5, 25, 30, 1e9],
                          labels=[0, 1, 2, 3]).astype(int)

    return pd.concat([X, y], axis=1)


# ---------------------------------------------------------------------------
# 3+4. Per-fold train/score - executed ON THE WORKERS
# ---------------------------------------------------------------------------
def train_and_score_fold(payload):
    X_tr, y_tr, X_te, y_te, model_name, sel_k, seed = payload

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",   # imbalance handling
            n_jobs=1,                  # parallelism is ACROSS folds/nodes
            random_state=seed,
        )
    else:  # logreg
        scaler = MinMaxScaler()        # chi^2 needs non-negative input
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)

    if sel_k and sel_k < X_tr.shape[1]:
        selector = SelectKBest(chi2, k=sel_k)
        X_tr = selector.fit_transform(X_tr, y_tr)
        X_te = selector.transform(X_te)

    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "model": model_name,
        "k": sel_k,
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "balanced_acc": balanced_accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "cm": confusion_matrix(y_te, pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Final full-data fit for feature importance - executed ON A WORKER
# ---------------------------------------------------------------------------
def final_fit(sel_k: int):
    X = FEATS.drop(columns="Outcome")
    y = FEATS["Outcome"]
    if sel_k:
        scaler = MinMaxScaler().fit(X)
        Xs = scaler.transform(X)
        sel = SelectKBest(chi2, k=sel_k).fit(Xs, y)
        Xs = sel.transform(Xs)
        cols = X.columns[sel.get_support()]
    else:
        Xs = X.values
        cols = X.columns
    clf = RandomForestClassifier(
        n_estimators=600, class_weight="balanced", n_jobs=1,
        random_state=RANDOM_STATE).fit(Xs, y)
    return sorted(zip(cols, clf.feature_importances_), key=lambda t: -t[1])


FEATS = None  # populated in main(); shipped to workers by dask


def main():
    global FEATS
    t0 = time.time()

    df = pd.read_csv(DATA_PATH, header=None, names=COLS)

    print("=" * 72)
    print("1) QUICK DATA ANALYSIS")
    print("=" * 72)
    info = quick_analysis(df)
    print(f"   rows x cols     : {info['rows']} x {info['cols']}"
          f"  ({info['memory_kb']:.1f} KB in memory)")
    print(f"   duplicates      : {info['duplicates']}")
    print(f"   outcome balance : {info['class_counts']}"
          f"  ({df['Outcome'].mean():.1%} diabetic)")
    for c, v in info["zero_anomalies"].items():
        print(f"   {c:<14}: {v['zeros']:>3} zeros ({v['pct']}%) -> treated as missing")

    # ------------------------------------------------------------------ cluster
    print("\n" + "=" * 72)
    print("2) CLUSTER UP")
    print("=" * 72)
    cluster = LocalCluster(n_workers=N_WORKERS,
                           threads_per_worker=THREADS_PER_WORKER,
                           processes=True, dashboard_address=None)
    client = Client(cluster)
    print(f"   scheduler : {cluster.scheduler_address}")
    print(f"   workers   : {len(client.scheduler_info()['workers'])}"
          f" x {THREADS_PER_WORKER} threads (multi-node-ready topology)")

    # ------------------------------------------------- distributed feature eng
    print("\n" + "=" * 72)
    print("3) DISTRIBUTED FEATURE ENGINEERING")
    print("=" * 72)
    tfe = time.time()
    feats_future = client.submit(build_features, df)
    FEATS = feats_future.result()
    X = FEATS.drop(columns="Outcome")
    y = FEATS["Outcome"]
    print(f"   built on worker: {X.shape[0]} rows x {X.shape[1]} features"
          f"  ({time.time() - tfe:.2f}s)")
    print(f"   features: {', '.join(X.columns)}")

    # --------------------------------------------------------- parallel train
    print("\n" + "=" * 72)
    print("4) PARALLEL TRAINING (RF vs LogReg, chi^2 selection, stratified 5-fold)")
    print("=" * 72)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))

    tasks, meta = [], []
    for model_name in ["rf", "logreg"]:
        for k in [8, 12, None]:
            for fold_idx, (tr, te) in enumerate(folds):
                payload = (X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te],
                           model_name, k, RANDOM_STATE + fold_idx)
                tasks.append(payload)
                meta.append((model_name, k, fold_idx))
    print(f"   fanning out {len(tasks)} train/score jobs"
          f" (2 models x 3 chi2-k x 5 folds) to {N_WORKERS} workers")

    tfit = time.time()
    futures = client.map(train_and_score_fold, tasks)
    results = client.gather(futures)
    dt_fit = time.time() - tfit
    print(f"   all folds trained + scored in {dt_fit:.2f}s")

    for (model_name, k, fold_idx), res in zip(meta, results):
        res["model"], res["k"], res["fold"] = model_name, k, fold_idx

    # ------------------------------------------------------------- evaluation
    print("\n" + "=" * 72)
    print("5) AGGREGATED EVALUATION (mean over 5 stratified folds)")
    print("=" * 72)
    res_df = pd.DataFrame(results)
    summary = (res_df.groupby(["model", "k"])[
        ["roc_auc", "pr_auc", "balanced_acc", "f1"]]
        .mean().round(4).sort_values("roc_auc", ascending=False))
    print(summary.to_string())

    # --------------------------------------------- final model + importances
    best_row = summary.reset_index().iloc[0]
    best_model, best_k = best_row["model"], best_row["k"]
    best_k = None if pd.isna(best_k) else int(best_k)
    print(f"\n   best config: {best_model} (chi2 k={best_k})"
          f" - refitting on full data for feature importance")

    fut = client.submit(final_fit, best_k if best_k else 0)
    importances = fut.result()
    print("\n   Top 12 clinical predictors:")
    for name, imp in importances[:12]:
        print(f"     {name:<28} {imp:.4f}")

    # Persist a small results file
    res_df.to_csv("cv_results.csv", index=False)
    pd.DataFrame(importances, columns=["feature", "importance"]).to_csv(
        "feature_importance.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n   saved: cv_results.csv, feature_importance.csv")
    print(f"   total pipeline wall time: {elapsed:.2f}s")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
