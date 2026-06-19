import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

from LongitudinalTauBC.data_extraction import extract_data
from data.data_extraction import tau_rois


def compute_tau_slopes(
    tau: pd.DataFrame,
    rois: list[str],
    min_points: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-subject baseline value and accumulation slope for each ROI."""
    subjects   = tau["RID"].unique()
    base_rows  = []
    slope_rows = []

    for rid in subjects:
        sub   = tau[tau["RID"] == rid]
        times = sub["time"].values
        row_base  = {"RID": rid}
        row_slope = {"RID": rid}

        for roi in rois:
            values = sub[roi].values
            mask   = ~np.isnan(values)

            baseline_vals = values[(times == 0) & mask]
            row_base[roi] = baseline_vals.mean() if len(baseline_vals) > 0 else np.nan

            if mask.sum() >= min_points:
                row_slope[roi + "_slope"] = np.polyfit(times[mask], values[mask], 1)[0]
            else:
                row_slope[roi + "_slope"] = np.nan

        base_rows.append(row_base)
        slope_rows.append(row_slope)

    X_base  = pd.DataFrame(base_rows).set_index("RID").fillna(pd.DataFrame(base_rows).set_index("RID").median())
    X_slope = pd.DataFrame(slope_rows).set_index("RID").fillna(pd.DataFrame(slope_rows).set_index("RID").median())

    return X_base, X_slope


def train_eval(X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    """Train logistic regression with 5-fold CV + held-out test. Scaler fit only on train data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000)),
    ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "model":         name,
        "feature_set":   name,
        "cv_auc_mean":   round(float(cv_aucs.mean()), 4),
        "cv_auc_std":    round(float(cv_aucs.std()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_auc":      round(float(roc_auc_score(y_test, y_prob)), 4),
    }


def model() -> dict:
    tau = extract_data()

    baseline_df, slope_df = compute_tau_slopes(tau, tau_rois)

    labels      = tau[["RID", "y"]].drop_duplicates()
    baseline_df = baseline_df.merge(labels, on="RID", how="inner")
    slope_df    = slope_df.merge(labels, on="RID", how="inner")
    combined_df = baseline_df.merge(slope_df, on=["RID", "y"], how="inner")

    X_base  = baseline_df.drop(columns=["RID", "y"])
    X_slope = slope_df.drop(columns=["RID", "y"])
    X_combo = combined_df.drop(columns=["RID", "y"])
    y       = combined_df["y"]

    results = [
        train_eval(X_base,  y, "Baseline tau"),
        train_eval(X_slope, y, "Tau slope"),
        train_eval(X_combo, y, "Baseline + slope"),
    ]

    print("\nLongitudinal Tau Results:")
    for r in results:
        print(f"  {r['model']}: CV AUC={r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f} | "
              f"Test Accuracy={r['test_accuracy']:.4f} | Test AUC={r['test_auc']:.4f}")

    return {"LongitudinalTau": {r["model"]: r for r in results}}


if __name__ == "__main__":
    model()
