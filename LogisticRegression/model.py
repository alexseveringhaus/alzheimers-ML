import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

from data.data_extraction import extract_data, amy_rois, tau_rois

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
CKPT_DIR    = Path(__file__).parent.parent / "checkpoints"


def run_models(X: np.ndarray, y: pd.Series, feature_set: str) -> dict:
    """Train LogisticRegression and RandomForest on a feature set.

    Performs an 80/20 stratified train/test split, then 5-fold CV on the
    training portion. Returns accuracy, CV AUC (mean ± std), and test AUC.
    Scaler is fit only on training data to prevent leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for pipe, name in [(logreg_pipe, "LogisticRegression"), (rf_pipe, "RandomForest")]:
        cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "feature_set": feature_set,
            "cv_auc_mean": round(float(cv_aucs.mean()), 4),
            "cv_auc_std": round(float(cv_aucs.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "classification_report": classification_report(y_test, y_pred),
        }

    return results


def plot_feature_importance(coefficients: dict) -> None:
    """Save horizontal bar chart of logistic regression coefficients."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (key, df) in zip(axes, coefficients.items()):
        colors = ["#d73027" if c > 0 else "#4575b4" for c in df["coef"]]
        ax.barh(df["ROI"], df["coef"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Logistic Regression Coefficients — {key.capitalize()}")
        ax.set_xlabel("Coefficient value")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "logreg_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def model() -> dict:
    _, _, combo_data = extract_data()

    X_amy   = combo_data[[col + "_x" for col in amy_rois]].values
    X_tau   = combo_data[[col + "_y" for col in tau_rois]].values
    X_combo = np.hstack([X_amy, X_tau])
    y       = combo_data["y"]

    all_results = {}
    all_results["amyloid"]  = run_models(X_amy,   y, "amyloid")
    all_results["tau"]      = run_models(X_tau,   y, "tau")
    all_results["combined"] = run_models(X_combo, y, "combined")

    # Logistic regression coefficients for feature importance plot
    # Fit a scaler+model on the full training split for each feature set
    coefficients = {}
    for feature_set, X_raw, roi_names in [
        ("amyloid",  X_amy,   amy_rois),
        ("tau",      X_tau,   tau_rois),
        ("combined", X_combo, [r + "_amy" for r in amy_rois] + [r + "_tau" for r in tau_rois]),
    ]:
        X_train, _, y_train, _ = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
        pipe.fit(X_train, y_train)
        coefs = pipe.named_steps["model"].coef_.flatten()
        coefficients[feature_set] = (
            pd.DataFrame({"ROI": roi_names, "coef": coefs})
            .sort_values("coef", ascending=True)
        )

        # Persist the combined pipeline and SHAP background for the inference API
        if feature_set == "combined":
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, CKPT_DIR / "logreg_combined.joblib")
            X_train_scaled = pipe.named_steps["scaler"].transform(X_train)
            np.save(CKPT_DIR / "shap_background.npy", X_train_scaled[:100])

    plot_feature_importance(coefficients)

    # Print summary
    for feature_set, models in all_results.items():
        print(f"\nResults for {feature_set} dataset:")
        for model_name, metrics in models.items():
            print(f"  {model_name}: CV AUC={metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f} | "
                  f"Test Accuracy={metrics['test_accuracy']:.4f} | Test AUC={metrics['test_auc']:.4f}")
            print(f"  Classification Report:\n{metrics['classification_report']}")

    return all_results


if __name__ == "__main__":
    model()
