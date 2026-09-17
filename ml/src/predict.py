from __future__ import annotations

from typing import Any

import pandas as pd

from ml.src.config import load_config
from ml.src.evaluate import predict_with_proba


FEATURE_ALIASES = {
    "age": None,
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "senior_citizen": "SeniorCitizen",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
}


def normalize_payload(payload: dict[str, Any], config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    remapped: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = FEATURE_ALIASES.get(key, key)
        if canonical is None:
            continue
        remapped[canonical] = value
    columns = [*cfg["features"]["numerical"], *cfg["features"]["categorical"]]
    row = {col: remapped.get(col) for col in columns}
    return pd.DataFrame([row])


def predict_payload(model: Any, payload: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    frame = normalize_payload(payload, cfg)
    threshold = float(cfg["serving"]["probability_threshold"])
    preds, proba = predict_with_proba(model, frame, threshold=threshold)
    return {
        "prediction": int(preds[0]),
        "churn_probability": float(proba[0]),
    }
