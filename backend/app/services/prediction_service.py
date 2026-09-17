from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy.orm import Session

from backend.app.core.metrics import (
    MODEL_VERSION,
    PREDICTION_DISTRIBUTION,
    PREDICTION_REQUESTS,
    REQUEST_LATENCY,
)
from backend.app.db.repositories import MetadataRepository
from backend.app.services.model_service import ModelService
from ml.src.predict import predict_payload


class PredictionService:
    def __init__(self) -> None:
        self.model_service = ModelService()
        self._model = None
        self._version: str | None = None

    def reload(self) -> tuple[Any, str]:
        model, version = self.model_service.load_production_model()
        self._model = model
        self._version = version
        try:
            MODEL_VERSION.set(float(version))
        except ValueError:
            MODEL_VERSION.set(0)
        return model, version

    def predict(self, payload: dict[str, Any], request_id: str, db: Session) -> dict[str, Any]:
        started = time.perf_counter()
        if self._model is None:
            self.reload()
        assert self._model is not None
        assert self._version is not None
        result = predict_payload(self._model, payload)
        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)
        PREDICTION_REQUESTS.labels(result=str(result["prediction"])).inc()
        PREDICTION_DISTRIBUTION.labels(label=str(result["prediction"])).inc()
        repo = MetadataRepository(db)
        safe_features = {
            key: payload.get(key)
            for key in (
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Contract",
                "InternetService",
                "PaymentMethod",
                "SeniorCitizen",
            )
            if key in payload or key.lower() in {k.lower() for k in payload}
        }
        if "monthly_charges" in payload and "MonthlyCharges" not in safe_features:
            safe_features["MonthlyCharges"] = payload.get("monthly_charges")
        repo.add_prediction(
            request_id=request_id,
            model_name=self.model_service.model_name,
            model_version=self._version,
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            latency_ms=latency_ms,
            status="ok",
            features_json=json.dumps(safe_features),
        )
        return {
            **result,
            "model_name": self.model_service.model_name,
            "model_version": self._version,
            "request_id": request_id,
        }
