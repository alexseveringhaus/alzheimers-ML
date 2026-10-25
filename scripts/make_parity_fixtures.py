"""Generate fixtures that check the browser model against the API.

Runs synthetic inputs through the API's own predict() and saves the responses
to web/src/model/fixtures.json, which web/src/model/logreg.test.ts compares
against. Inputs are training means plus seeded noise, so no subject data is
written. Re-run after retraining:

    python scripts/make_parity_fixtures.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from api.main import predict
from api.model_loader import get_model
from api.schemas import PredictRequest
from data.data_extraction import amy_rois, tau_rois

OUT_PATH  = ROOT / "web" / "src" / "model" / "fixtures.json"
N_INPUTS  = 50


def main() -> None:
    pipe, _ = get_model()
    scaler  = pipe.named_steps["scaler"]
    rng     = np.random.default_rng(42)

    fixtures = []
    for _ in range(N_INPUTS):
        x = scaler.mean_ + rng.normal(0, 1.5, len(scaler.mean_)) * scaler.scale_
        x = np.clip(x, 0.0, 5.0)  # stay inside the form's input range
        request = PredictRequest(
            amyloid_suvrs=dict(zip(amy_rois, map(float, x[:13]))),
            tau_suvrs=dict(zip(tau_rois, map(float, x[13:]))),
        )
        fixtures.append({
            "amyloid_suvrs": request.amyloid_suvrs,
            "tau_suvrs":     request.tau_suvrs,
            "expected":      predict(request).model_dump(),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures, indent=1) + "\n")

    n_ad = sum(f["expected"]["predicted_class"] == "AD" for f in fixtures)
    print(f"Saved {len(fixtures)} fixtures ({n_ad} AD, {len(fixtures) - n_ad} CN) to {OUT_PATH}")


if __name__ == "__main__":
    main()
