from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import shap

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


@lru_cache(maxsize=1)
def get_model() -> tuple:
    """Load the trained Pipeline and SHAP explainer once; cache for the process lifetime."""
    pipe = joblib.load(CKPT_DIR / "logreg_combined.joblib")
    background = np.load(CKPT_DIR / "shap_background.npy")
    explainer = shap.LinearExplainer(pipe.named_steps["model"], background)
    return pipe, explainer
