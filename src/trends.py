"""
Data-trend analysis: how does diabetes risk move as each predictor's value
rises or falls?

Produces:
  assets/risk_curves.png      - MAIN TAKEAWAY: empirical diabetic prevalence by
                                predictor value band (bars) + RandomForest
                                partial-dependence risk curve (line)
  assets/logreg_direction.png - direction + size of each feature's effect
                                (standardised Logistic Regression coefficients)

Run from the repo root:  python src/trends.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import COLS, DATA_PATH, RANDOM_STATE, build_features  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

# Panel order = strongest predictors first (from feature_importance.csv)
PANELS = [
    "Glucose_x_Age",
    "Glucose",
    "Insulin_BMI_Interaction",
    "QUICKI",
    "BMI",
    "Age",
]
PRETTY = {
    "Glucose_x_Age": "Glucose × Age",
    "Glucose": "Glucose (mg/dL)",
    "Insulin_BMI_Interaction": "Insulin × BMI",
    "QUICKI": "QUICKI (insulin sensitivity)",
    "BMI": "BMI (kg/m²)",
    "Age": "Age (years)",
    "Pregnancies": "Pregnancies",
    "DiabetesPedigreeFunction": "Pedigree function",
    "BP_x_BMI": "Blood pressure × BMI",
}


def empirical_trend(x, y, n_bins=8):
    """Observed diabetic prevalence in quantile bands of x.

    Band membership uses ranks (method='first') so repeated values (e.g.
    imputed insulin) don't collapse into one giant band; bar positions and
    widths are reported in the feature's ACTUAL units so they align with the
    partial-dependence x-axis.

    Returns (midpoints, widths, prevalence, counts)."""
    bands = pd.qcut(x.rank(method="first"), q=n_bins, duplicates="drop")
    grp = y.groupby(bands, observed=True)
    prev = grp.mean()
    counts = grp.size()
    lo = x.groupby(bands, observed=True).min()
    hi = x.groupby(bands, observed=True).max()
    mids = (lo + hi).values / 2
    widths = np.maximum((hi - lo).values, 0.02 * (x.max() - x.min()))
    return mids, widths, prev.values, counts.values


def make_risk_curves(X, y):
    """MAIN TAKEAWAY FIGURE - one panel per predictor."""
    rf = RandomForestClassifier(n_estimators=600, class_weight="balanced",
                                n_jobs=1, random_state=RANDOM_STATE)
    rf.fit(X, y)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ax, feat in zip(axes.ravel(), PANELS):
        # Model view: partial dependence on probability scale, full x-range
        res = partial_dependence(rf, X, features=[feat], grid_resolution=60,
                                 percentiles=(0.01, 0.99))
        grid = res["grid_values"][0]
        pdp = res["average"][0]

        # Data view: observed prevalence per quantile band
        mids, widths, prev, counts = empirical_trend(X[feat], y)

        ax.bar(mids, prev, width=widths * 0.9, color="#9ecae1", alpha=0.65,
               label="observed diabetic %")
        ax.set_ylim(0, 1.0)

        ax2 = ax.twinx()
        ax2.plot(grid, pdp, color="#d95f02", lw=3,
                 label="RandomForest risk curve")
        ax2.set_ylim(0, 1.0)
        ax2.grid(False)

        rho = X[feat].corr(y, method="spearman")
        arrow = "risk ↑" if rho > 0.05 else ("risk ↓" if rho < -0.05 else "~ flat")
        ax.set_title(f"{PRETTY.get(feat, feat)}  (ρ={rho:+.2f}, {arrow})",
                     fontsize=15)
        ax.set_xlabel(PRETTY.get(feat, feat), fontsize=13)
        ax.set_ylabel("observed diabetic %", fontsize=12)
        ax2.set_ylabel("model-predicted risk", fontsize=12)
        if feat == PANELS[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=11, loc="upper left")
    fig.suptitle("Diabetes risk vs predictor value — observed prevalence (bars) "
                 "and model risk curve (line)", fontsize=18, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "risk_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_direction_chart(X, y):
    """Diverging bars: standardised LogReg coefficients = direction of effect."""
    scaler = MinMaxScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                             random_state=RANDOM_STATE)
    clf.fit(scaler.transform(X), y)
    coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#7570b3" if c < 0 else "#d95f02" for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Direction of effect on diabetes risk "
                 "(Logistic Regression, min-max scaled features)", fontsize=15)
    ax.set_xlabel("coefficient  (positive → raises diabetes risk)")
    fig.tight_layout()
    fig.savefig(OUT / "logreg_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH, header=None, names=COLS)
    feats = build_features(df)
    X, y = feats.drop(columns="Outcome"), feats["Outcome"]

    print("building risk-curve figure ...")
    make_risk_curves(X, y)
    print("building direction figure ...")
    make_direction_chart(X, y)
    print(f"saved figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()