import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import get_model
from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RoisResponse,
    ShapContribution,
)
from data.data_extraction import amy_rois, tau_rois


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    get_model()  # warm the cache on startup so the first request isn't slow
    yield


app = FastAPI(
    title="Alzheimer's Classification API",
    description="Predicts AD vs CN from amyloid and tau PET SUVR values using a trained logistic regression model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_NAMES = [r + "_amy" for r in amy_rois] + [r + "_tau" for r in tau_rois]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model="LogisticRegression_combined")


@app.get("/rois", response_model=RoisResponse)
def rois() -> RoisResponse:
    return RoisResponse(amyloid_rois=amy_rois, tau_rois=tau_rois)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    pipe, explainer = get_model()

    # Assemble feature vector in training order: amyloid ROIs then tau ROIs
    try:
        X = np.array(
            [request.amyloid_suvrs[r] for r in amy_rois]
            + [request.tau_suvrs[r] for r in tau_rois],
            dtype=float,
        ).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing ROI value: {e}")

    proba = pipe.predict_proba(X)[0]  # pipeline handles scaling internally
    prob_cn, prob_ad = float(proba[0]), float(proba[1])
    predicted_class = "AD" if prob_ad >= 0.5 else "CN"

    # SHAP explainer was built against the raw model, not the pipeline, so needs scaled input
    X_scaled = pipe.named_steps["scaler"].transform(X)
    shap_vals = explainer.shap_values(X_scaled)
    # LinearExplainer returns list[array] for multi-class; index 1 = AD class
    ad_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    contributions = [
        ShapContribution(
            roi=name,
            shap_value=round(float(val), 6),
            direction="toward_AD" if val > 0 else "toward_CN",
        )
        for name, val in zip(FEATURE_NAMES, ad_shap)
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return PredictResponse(
        predicted_class=predicted_class,
        probability_ad=round(prob_ad, 4),
        probability_cn=round(prob_cn, 4),
        shap_contributions=contributions,
    )
