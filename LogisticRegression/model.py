import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent)) # Ensure project root is in path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from data.data_extraction import extract_data, amy_rois, tau_rois

def model():
    amy_data, tau_data, combo_data = extract_data() #amy_data and tai_data used only for validation if necessary

    # Input features and target variable
    X_amy = combo_data[[col + '_x' for col in amy_rois]]  
    X_tau = combo_data[[col + '_y' for col in tau_rois]]
    X_combo = pd.concat([X_amy, X_tau], axis=1)
    y = combo_data['y']

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_amy_scaled = scaler.fit_transform(X_amy)
    X_tau_scaled = scaler.fit_transform(X_tau)
    X_combo_scaled = scaler.fit_transform(X_combo)

    # Split into training and test sets
    X_train_amy, X_test_amy, y_train, y_test = train_test_split(X_amy_scaled, y, test_size=0.2, random_state=42, stratify=y)
    X_train_tau, X_test_tau, _, _ = train_test_split(X_tau_scaled, y, test_size=0.2, random_state=42, stratify=y)
    X_train_combo, X_test_combo, _, _ = train_test_split(X_combo_scaled, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    results["amyloid"] = {}
    results["tau"] = {}
    results["combined"] = {}

    coefficients = {} # logistic regression coefficients for feature importance in AD prediction

    logreg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    models = [(logreg, "LogisticRegression"), (rf, "RandomForest")]

    for mod, name in models:
        mod.fit(X_train_amy, y_train) # train
        y_pred_amy = mod.predict(X_test_amy) # test
        y_proba = mod.predict_proba(X_test_amy)[:,1]
        results["amyloid"][name] = {
            "accuracy": accuracy_score(y_test, y_pred_amy),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "classification_report": classification_report(y_test, y_pred_amy)
        }
        if name == "LogisticRegression":
            coefficients["amyloid"] = pd.DataFrame({
                "ROI": amy_rois,
                "coef": mod.coef_.flatten()
            }).sort_values("coef", ascending=False)

        mod.fit(X_train_tau, y_train)
        y_pred_tau = mod.predict(X_test_tau)
        y_proba = mod.predict_proba(X_test_tau)[:,1]
        results["tau"][name] = {
            "accuracy": accuracy_score(y_test, y_pred_tau),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "classification_report": classification_report(y_test, y_pred_tau)
        }
        if name == "LogisticRegression":
            coefficients["tau"] = pd.DataFrame({
                "ROI": amy_rois,
                "coef": mod.coef_.flatten()
            }).sort_values("coef", ascending=False)

        mod.fit(X_train_combo, y_train)
        y_pred_combo = mod.predict(X_test_combo)
        y_proba = mod.predict_proba(X_test_combo)[:,1]
        results["combined"][name] = {
            "accuracy": accuracy_score(y_test, y_pred_combo),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "classification_report": classification_report(y_test, y_pred_combo)
        }
        if name == "LogisticRegression":
            combo_feature_names = (
               [roi + "_amy" for roi in amy_rois] +
               [roi + "_tau" for roi in tau_rois]
            )
            coefficients["combined"] = pd.DataFrame({
                "ROI": combo_feature_names,
                "coef": mod.coef_.flatten()
            }).sort_values("coef", ascending=False)
    
    for key in results:
        print(f"Results for {key} dataset:")
        for model_name in results[key]:
            print(f"    Model: {model_name}")
            print(f"    Accuracy: {results[key][model_name]['accuracy']:.4f}")
            print(f"    ROC AUC: {results[key][model_name]['roc_auc']:.4f}")
            print(f"    Classification Report:\n{results[key][model_name]['classification_report']}")

    print("Logistic Regression Coefficients for Feature Importance:")

    for key in coefficients:
        print(f"  {key}:")
        print(coefficients[key])

model()
    