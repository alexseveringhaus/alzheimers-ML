import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent)) # Ensure project root is in path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from LongitudinalTauBC.data_extraction import extract_data
from data.data_extraction import tau_rois

# Compute baseline and slope for each ROI per subject to use as features in models
def compute_tau_slopes(tau, tau_rois, min_points=2):
    subjects = tau['RID'].unique()
    X_base_rows = []
    X_slope_rows = []

    for rid in subjects:
        sub_df = tau[tau['RID'] == rid]
        row_base = {'RID': rid}
        row_slope = {'RID': rid}

        for roi in tau_rois:
            values = sub_df[roi].values
            times = sub_df['time'].values
            mask = ~np.isnan(values)

            # Baseline at time=0
            baseline_vals = values[(times == 0) & mask]
            row_base[roi] = baseline_vals.mean() if len(baseline_vals) > 0 else np.nan

            # Compute slope if enough points
            if mask.sum() >= min_points:
                slope = np.polyfit(times[mask], values[mask], 1)[0]
                row_slope[roi + '_slope'] = slope
            else:
                row_slope[roi + '_slope'] = np.nan

        X_base_rows.append(row_base)
        X_slope_rows.append(row_slope)

    X_base = pd.DataFrame(X_base_rows).set_index('RID')
    X_slope = pd.DataFrame(X_slope_rows).set_index('RID')

    # Median imputation for remaining NaNs
    X_base = X_base.fillna(X_base.median())
    X_slope = X_slope.fillna(X_slope.median())

    return X_base, X_slope

def train_eval(X, y, name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_prob),
    }

def model():
    tau = extract_data()

    # Compute baseline + slopes
    baseline_df, slope_df = compute_tau_slopes(tau, tau_rois)

    # Merge datasets together
    labels = tau[["RID", "y"]].drop_duplicates()
    baseline_df = baseline_df.merge(labels, on="RID", how="inner")
    slope_df = slope_df.merge(labels, on="RID", how="inner")
    combined_df = baseline_df.merge(slope_df, on=["RID", "y"], how="inner")

    # Prepare features and labels
    X_base = baseline_df.drop(columns=["RID", "y"])
    X_slope = slope_df.drop(columns=["RID", "y"])
    X_combo = combined_df.drop(columns=["RID", "y"])
    y = combined_df["y"]

    results = []
    results.append(train_eval(X_base,  y, "Baseline tau"))
    results.append(train_eval(X_slope, y, "Tau slope"))
    results.append(train_eval(X_combo, y, "Baseline + slope"))

    for r in results:
        print(f"{r['model']}: Accuracy={r['accuracy']:.3f}, AUC={r['roc_auc']:.3f}")

model()