"""
Exploratory plots describing performance trends across CV folds.
Reads results/cv_results.csv (produced by src/pipeline.py) and writes
figures to assets/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
RESULTS = Path("results/cv_results.csv")
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

METRICS = ["roc_auc", "pr_auc", "balanced_acc", "f1"]


def best_and_other(df):
    """Best (model, k) config by mean ROC-AUC + the other model's best config."""
    cfg = (df.groupby(["model", "k"])["roc_auc"].mean().sort_values(
        ascending=False))
    (b_model, b_k), (o_model, o_k) = cfg.index[0], cfg.index[1]
    other = cfg[cfg.index.get_level_values("model") != b_model]
    o_model, o_k = other.index[0]
    best = df[(df.model == b_model) & (df.k == b_k)].sort_values("fold")
    comp = df[(df.model == o_model) & (df.k == o_k)].sort_values("fold")
    return best, comp


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df["k"] = df["k"].fillna(-1).astype(int)
    df["config"] = df.apply(
        lambda r: f"{r['model']} (k={r['k'] if r['k'] > 0 else 'all'})", axis=1)
    cm = df["cm"].apply(json.loads).apply(np.array)
    df["tn"] = cm.apply(lambda m: m[0, 0])
    df["fp"] = cm.apply(lambda m: m[0, 1])
    df["fn"] = cm.apply(lambda m: m[1, 0])
    df["tp"] = cm.apply(lambda m: m[1, 1])
    cm_hr = df["cm_hr"].apply(json.loads).apply(np.array)
    df["tn_hr"] = cm_hr.apply(lambda m: m[0, 0])
    df["fp_hr"] = cm_hr.apply(lambda m: m[0, 1])
    df["fn_hr"] = cm_hr.apply(lambda m: m[1, 0])
    df["tp_hr"] = cm_hr.apply(lambda m: m[1, 1])
    df["recall"] = df["tp"] / (df["tp"] + df["fn"])
    df["specificity"] = df["tn"] / (df["tn"] + df["fp"])
    df["test_pos_rate"] = (df["tp"] + df["fn"]) / df["n_test"]
    return df


def metric_trends(df):
    """Line chart: ROC-AUC and PR-AUC per fold, best config vs runner-up."""
    best, other = best_and_other(df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title in [
        (axes[0], "roc_auc", "ROC-AUC across folds"),
        (axes[1], "pr_auc", "PR-AUC across folds"),
    ]:
        ax.plot(best.fold, best[metric], "o-", label=f"{best.config.iloc[0]} (best)")
        ax.plot(other.fold, other[metric], "s--", label=f"{other.config.iloc[0]} (runner-up)",
                alpha=0.7)
        ax.axhline(df[metric].mean(), color="gray", ls=":", lw=1,
                   label=f"mean = {df[metric].mean():.3f}")
        ax.set(title=title, xlabel="Fold", xticks=range(5), ylim=(0.6, 1.0))
        ax.legend(fontsize=13)
    fig.suptitle("Per-fold metric trends (stratified 5-fold CV)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "metric_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fold_variance(df):
    """Bar chart of per-fold means +/- spread for the best config, all metrics."""
    sub, _ = best_and_other(df)
    fig, ax = plt.subplots(figsize=(11, 6))
    melted = sub.melt(id_vars="fold", value_vars=METRICS, var_name="metric",
                      value_name="score")
    sns.barplot(data=melted, x="metric", y="score", hue="fold",
                palette="viridis", ax=ax)
    ax.set(title=f"Metric stability across folds — {sub.config.iloc[0]}",
           xlabel="", ylim=(0, 1))
    ax.legend(title="fold", fontsize=13, ncol=5)
    ax.legend(title="fold", fontsize=13)
    ax.axhline(sub.balanced_acc.mean(), color="red", ls=":", lw=2,
               label=f"mean balanced acc = {sub.balanced_acc.mean():.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "fold_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_decomposition(df):
    """FP vs FN per fold — how the class imbalance bites."""
    sub, _ = best_and_other(df)
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.38
    ax.bar(sub.fold - width / 2, sub.fp, width, label="false positives",
           color="#d95f02")
    ax.bar(sub.fold + width / 2, sub.fn, width, label="false negatives",
           color="#7570b3")
    for _, r in sub.iterrows():
        ax.text(r.fold - width / 2, r.fp + 0.5, int(r.fp), ha="center",
                fontsize=12)
        ax.text(r.fold + width / 2, r.fn + 0.5, int(r.fn), ha="center",
                fontsize=12)
    ax.set(title=f"Error decomposition across folds — {sub.config.iloc[0]}",
           xlabel="fold", ylabel="count (test set, n≈154)")
    ax.set_xticks(range(5))
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "error_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def stratification_check(df):
    """Verify stratification: positive-class rate per fold + test size."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df.fold.unique(), df.groupby("fold")["test_pos_rate"].first(),
           color="#1b9e77", alpha=0.8, label="diabetic % in test fold")
    sizes = df.groupby("fold")["n_test"].first()
    for fold, size in sizes.items():
        ax.text(fold, df.test_pos_rate.max() + 0.012, f"n={int(sizes[fold])}",
                ha="center", fontsize=12)
    ax.axhline(0.349, color="red", ls="--", lw=2,
               label=f"overall rate (34.9%)")
    ax.set(title="Stratification check — positive-class rate per fold",
           xlabel="fold", ylabel="diabetic prevalence in test fold",
           ylim=(0, 0.5), xticks=range(5))
    ax.legend(fontsize=13, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "stratification_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def high_recall_impact(df):
    """Before (0.5 threshold) vs after (high-recall threshold) comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    # -- panel A: missed diabetics (FN) + false alarms (FP), per config -----   
    agg = df.groupby(["model", "k"])[["fn", "fn_hr", "fp", "fp_hr"]].sum()
    labels = [f"{m} (k={k if k > 0 else 'all'})" for m, k in agg.index]
    x = np.arange(len(agg))
    w = 0.2
    ax = axes[0]
    ax.bar(x - w, agg.fn, w, label="FN @ default 0.5", color="#7570b3")
    ax.bar(x, agg.fn_hr, w, label="FN @ high-recall", color="#1b9e77")
    ax.bar(x + w, agg.fp, w, label="FP @ default 0.5", color="#d95f02", alpha=0.55)
    ax.bar(x + 2 * w, agg.fp_hr, w, label="FP @ high-recall", color="#d95f02",
           hatch="//", alpha=0.55)
    ax.set(title="Total errors across 5 folds: missed diabetics (FN) "
                 "vs false alarms (FP)",
           xticks=x, xticklabels=labels, ylabel="count (all folds, n=768)")
    ax.tick_params(axis="x", rotation=30, labelsize=12)
    ax.legend(fontsize=12)
    for i, (fn_a, fn_b) in enumerate(zip(agg.fn, agg.fn_hr)):
        ax.text(i - w, fn_a + 6, int(fn_a), ha="center", fontsize=11)
        ax.text(i, fn_b + 6, int(fn_b), ha="center", fontsize=11)

    # -- panel B: best config, recall/precision per fold, before vs after ---
    best, _ = best_and_other(df)
    ax = axes[1]
    rec05 = best["tp"] / (best["tp"] + best["fn"])
    recHR = best.recall_hr
    pre05 = best.precision
    preHR = best.precision_hr
    ax.plot(best.fold, rec05, "o--", color="#7570b3",
            label="recall @ 0.5", alpha=0.85)
    ax.plot(best.fold, recHR, "o-", color="#1b9e77", label="recall @ tuned thr")
    ax.plot(best.fold, pre05, "s--", color="#d95f02",
            label="precision @ 0.5", alpha=0.85)
    ax.plot(best.fold, preHR, "s-", color="#e7298a", label="precision @ tuned thr")
    ax.axhline(0.9, color="gray", ls=":", lw=1.5, label="screening target (0.9)")
    ax.set(title=f"Per-fold trade-off — {best.config.iloc[0]}",
           xlabel="fold", xticks=range(5), ylim=(0.3, 1.02))
    ax.legend(fontsize=12, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "high_recall_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    print(f"loaded {len(df)} rows from {RESULTS}")

    metric_trends(df)
    fold_variance(df)
    error_decomposition(df)
    stratification_check(df)
    high_recall_impact(df)
    print(f"saved 5 figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
