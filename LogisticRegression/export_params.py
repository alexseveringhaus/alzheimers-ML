import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from data.data_extraction import amy_rois, tau_rois

CKPT_DIR    = Path(__file__).parent.parent / "checkpoints"
PARAMS_PATH = CKPT_DIR / "logreg_combined_params.json"


def export_params(pipe: Pipeline, background: np.ndarray, path: Path = PARAMS_PATH) -> None:
    """Write the combined pipeline's parameters as JSON for in-browser inference.

    The browser reproduces the API exactly with these numbers:
      z      = (x - scaler_mean) / scaler_scale
      P(AD)  = sigmoid(coef · z + intercept)
      SHAP_i = coef_i * (z_i - shap_background_mean_i)
    """
    scaler = pipe.named_steps["scaler"]
    model  = pipe.named_steps["model"]

    params = {
        "model":                "LogisticRegression_combined",
        "amyloid_rois":         amy_rois,
        "tau_rois":             tau_rois,
        "feature_names":        [r + "_amy" for r in amy_rois] + [r + "_tau" for r in tau_rois],
        "scaler_mean":          scaler.mean_.tolist(),
        "scaler_scale":         scaler.scale_.tolist(),
        "coef":                 model.coef_[0].tolist(),
        "intercept":            float(model.intercept_[0]),
        "shap_background_mean": background.mean(axis=0).tolist(),
    }
    path.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Browser model params saved to {path}")


if __name__ == "__main__":
    # Export from the committed checkpoints, no retraining needed
    export_params(
        joblib.load(CKPT_DIR / "logreg_combined.joblib"),
        np.load(CKPT_DIR / "shap_background.npy"),
    )
