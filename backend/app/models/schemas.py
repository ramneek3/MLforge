from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    status: str
    database: bool
    mlflow: bool
    model_loaded: bool


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(ge=0, le=1, default=0)
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(ge=0, le=100)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float | None = Field(default=None, gt=0)
    TotalCharges: float | None = None
    monthly_charges: float | None = None
    age: int | None = None

    @model_validator(mode="after")
    def require_monthly_charges(self) -> "PredictRequest":
        if self.MonthlyCharges is None and self.monthly_charges is not None:
            self.MonthlyCharges = self.monthly_charges
        if self.MonthlyCharges is None:
            raise ValueError("MonthlyCharges is required")
        return self


class PredictResponse(BaseModel):
    prediction: int
    churn_probability: float
    model_name: str
    model_version: str
    request_id: str


class TrainRequest(BaseModel):
    trigger: str = "manual"
    model_type: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    status: str
    best_model_type: str
    model_version: str
    metrics: dict[str, float]
    promotable: bool
    promotion_reason: str
    run_id: str
    candidates: list[dict[str, Any]]


class PromoteRequest(BaseModel):
    version: str
    action: Literal["promote", "reject", "archive", "stage"] = "promote"


class ModelVersionResponse(BaseModel):
    model_name: str
    version: str
    status: str
    model_type: str | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    created_at: str | None = None
    run_id: str | None = None


class MonitoringSummary(BaseModel):
    total_predictions: int
    error_rate: float
    avg_latency_ms: float
    production_model_version: str | None
    production_f1: float | None
    production_roc_auc: float | None
    positive_prediction_rate: float
    drift_detected: bool
    retraining_required: bool
