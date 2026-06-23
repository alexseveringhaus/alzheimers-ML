"""Streamlit dashboard — two tabs: live AD/CN prediction + research methodology."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────
API_URL   = os.environ.get("API_URL", "http://localhost:8000")

# Works whether app.py lives in dashboard/ (local) or at repo root (HF Spaces)
_here     = Path(__file__).parent
REPO_ROOT = _here if (_here / "results").exists() else _here.parent
RESULTS   = REPO_ROOT / "results"
PLOTS     = RESULTS / "plots"

st.set_page_config(
    page_title="Alzheimer's Classification",
    page_icon="🧠",
    layout="wide",
)

# ── Training-set mean SUVRs (used as form defaults) ──────────────────────────
AMY_DEFAULTS = {
    "CTX_ROSTRALANTERIORCINGULATE_SUVR": 1.1531,
    "CTX_CAUDALANTERIORCINGULATE_SUVR":  1.1804,
    "CTX_MIDDLETEMPORAL_SUVR":           1.0888,
    "CTX_INFERIORTEMPORAL_SUVR":         1.1263,
    "CTX_POSTERIORCINGULATE_SUVR":       1.2075,
    "CTX_PRECUNEUS_SUVR":               1.2054,
    "CTX_SUPERIORFRONTAL_SUVR":         1.0934,
    "CTX_SUPERIORPARIETAL_SUVR":        1.0988,
    "CTX_LATERALOCCIPITAL_SUVR":        1.1380,
    "CTX_FRONTALPOLE_SUVR":             0.9905,
    "CTX_ENTORHINAL_SUVR":              0.9235,
    "CTX_FUSIFORM_SUVR":                1.1151,
    "CTX_INFERIORPARIETAL_SUVR":        1.1566,
}
TAU_DEFAULTS = {
    "CTX_ROSTRALANTERIORCINGULATE_SUVR": 1.0739,
    "CTX_CAUDALANTERIORCINGULATE_SUVR":  1.0676,
    "CTX_MIDDLETEMPORAL_SUVR":           1.2537,
    "CTX_INFERIORTEMPORAL_SUVR":         1.2936,
    "CTX_POSTERIORCINGULATE_SUVR":       1.1474,
    "CTX_PRECUNEUS_SUVR":               1.1766,
    "CTX_SUPERIORFRONTAL_SUVR":         1.0516,
    "CTX_SUPERIORPARIETAL_SUVR":        1.1045,
    "CTX_LATERALOCCIPITAL_SUVR":        1.1807,
    "CTX_FRONTALPOLE_SUVR":             1.0488,
    "CTX_ENTORHINAL_SUVR":              1.1986,
    "CTX_FUSIFORM_SUVR":                1.2485,
    "CTX_INFERIORPARIETAL_SUVR":        1.2179,
}

def pretty_roi(name: str) -> str:
    """CTX_ENTORHINAL_SUVR → Entorhinal"""
    return name.replace("CTX_", "").replace("_SUVR", "").replace("_", " ").title()


# ── SHAP chart ───────────────────────────────────────────────────────────────
def shap_chart(contributions: list[dict], predicted_class: str) -> plt.Figure:
    rois   = [pretty_roi(c["roi"]) for c in contributions[:15]]
    values = [c["shap_value"] for c in contributions[:15]]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(rois[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on AD probability)")
    ax.set_title(f"Top 15 ROI contributions — predicted: {predicted_class}")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
tab_predict, tab_research = st.tabs(["🔬 Predict", "📊 Research"])


# ── TAB 1: PREDICT ───────────────────────────────────────────────────────────
with tab_predict:
    st.title("AD vs CN Classification")
    st.markdown(
        "Enter PET SUVR values for 13 cortical ROIs from **amyloid** and **tau** scans. "
        "The model returns a predicted class (AD / CN), class probabilities, and a "
        "SHAP breakdown showing which regions drove the prediction."
    )
    st.warning(
        "⚠️ **Research tool only.** This model was trained on ADNI research data and is "
        "not validated for clinical use. Do not use for medical decision-making.",
        icon=None,
    )

    with st.form("prediction_form"):
        col_amy, col_tau = st.columns(2)

        with col_amy:
            st.subheader("Amyloid PET SUVRs")
            amyloid_inputs = {
                roi: st.number_input(
                    pretty_roi(roi),
                    value=AMY_DEFAULTS[roi],
                    min_value=0.0,
                    max_value=5.0,
                    step=0.001,
                    format="%.4f",
                    key=f"amy_{roi}",
                )
                for roi in AMY_DEFAULTS
            }

        with col_tau:
            st.subheader("Tau PET SUVRs")
            tau_inputs = {
                roi: st.number_input(
                    pretty_roi(roi),
                    value=TAU_DEFAULTS[roi],
                    min_value=0.0,
                    max_value=5.0,
                    step=0.001,
                    format="%.4f",
                    key=f"tau_{roi}",
                )
                for roi in TAU_DEFAULTS
            }

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Querying model…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"amyloid_suvrs": amyloid_inputs, "tau_suvrs": tau_inputs},
                    timeout=15,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the API at {API_URL}. Is the server running?")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e}\n\n{response.text}")
                st.stop()

        pred  = result["predicted_class"]
        p_ad  = result["probability_ad"]
        p_cn  = result["probability_cn"]
        contribs = result["shap_contributions"]

        st.divider()
        res_col, prob_col = st.columns([1, 2])

        with res_col:
            color = "#d73027" if pred == "AD" else "#4575b4"
            st.markdown(
                f"<h1 style='color:{color}; font-size:4rem; margin:0'>{pred}</h1>",
                unsafe_allow_html=True,
            )
            st.caption("Predicted class")

        with prob_col:
            st.metric("P(AD)", f"{p_ad:.1%}")
            st.progress(p_ad, text=f"AD {p_ad:.1%} / CN {p_cn:.1%}")

        st.subheader("SHAP Feature Contributions")
        st.caption(
            "Red bars push the prediction toward AD; blue bars push toward CN. "
            "Magnitude reflects each ROI's influence on this specific prediction."
        )
        st.pyplot(shap_chart(contribs, pred))


# ── TAB 2: RESEARCH ──────────────────────────────────────────────────────────
with tab_research:
    st.title("Model Research & Methodology")

    st.markdown("""
    ## About the model

    The API serves a **Logistic Regression** trained on combined amyloid + tau PET SUVR
    features from the [ADNI](https://adni.loni.usc.edu/) dataset. The binary classification
    task is Cognitively Normal (CN = 0) vs Alzheimer's Disease (AD = 1); MCI subjects are
    excluded for a cleaner signal.

    **Training approach:**
    - 13 amyloid ROI SUVRs + 13 tau ROI SUVRs = 26 input features
    - `StandardScaler` normalization inside an `sklearn.Pipeline` (fit on training data only — no leakage)
    - Stratified 80/20 train/test split; 5-fold cross-validation on the training portion
    - Model selection: the combined-feature Logistic Regression achieved Test AUC = 1.00 and
      CV AUC = 0.892 ± 0.061, outperforming random forest and matching the neural network
      while remaining fully interpretable via SHAP
    """)

    st.divider()
    st.subheader("Model Comparison")

    metrics_path = RESULTS / "metrics.json"
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text())
        rows = []
        for section, section_data in raw.items():
            if not isinstance(section_data, dict):
                continue
            for model_name, m in section_data.items():
                if not isinstance(m, dict) or "cv_auc_mean" not in m:
                    continue
                rows.append({
                    "Model":         model_name,
                    "Feature Set":   m.get("feature_set", ""),
                    "CV AUC":        f"{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}",
                    "Test AUC":      f"{m['test_auc']:.3f}",
                    "Test Accuracy": f"{m['test_accuracy']:.1%}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python main.py` to generate results/metrics.json.")

    st.divider()
    st.subheader("Logistic Regression Feature Importance")
    st.caption(
        "Coefficients across amyloid-only, tau-only, and combined feature sets. "
        "Red = positive weight (higher SUVR → more likely AD); blue = negative weight."
    )
    fi_path = PLOTS / "logreg_feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), use_container_width=True)
    else:
        st.info("Run `python main.py --model logreg` to generate this plot.")

    st.divider()
    st.subheader("Neural Network Training Curve")
    st.caption("Train vs validation loss with early stopping (red dashed line).")
    nn_path = PLOTS / "nn_training_curve.png"
    if nn_path.exists():
        st.image(str(nn_path), use_container_width=True)
    else:
        st.info("Run `python main.py --model nn` to generate this plot.")
