from pydantic import BaseModel, field_validator
from data.data_extraction import amy_rois, tau_rois


class PredictRequest(BaseModel):
    amyloid_suvrs: dict[str, float]
    tau_suvrs: dict[str, float]

    @field_validator("amyloid_suvrs")
    @classmethod
    def validate_amyloid(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [r for r in amy_rois if r not in v]
        if missing:
            raise ValueError(f"Missing amyloid ROIs: {missing}")
        return v

    @field_validator("tau_suvrs")
    @classmethod
    def validate_tau(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [r for r in tau_rois if r not in v]
        if missing:
            raise ValueError(f"Missing tau ROIs: {missing}")
        return v


class ShapContribution(BaseModel):
    roi: str
    shap_value: float
    direction: str  # "toward_AD" | "toward_CN"


class PredictResponse(BaseModel):
    predicted_class: str       # "AD" | "CN"
    probability_ad: float
    probability_cn: float
    shap_contributions: list[ShapContribution]
    disclaimer: str = "Research tool only — not intended for clinical diagnosis."


class HealthResponse(BaseModel):
    status: str
    model: str
    test_auc: float


class RoisResponse(BaseModel):
    amyloid_rois: list[str]
    tau_rois: list[str]
