from __future__ import annotations

from ml.src.ingest import _generate_synthetic_telco
from ml.src.predict import normalize_payload, predict_payload
from sklearn.linear_model import LogisticRegression

from ml.src.data_preprocessing import clean_dataset
from ml.src.feature_engineering import build_model_pipeline, split_dataset


def test_predict_payload_returns_probability() -> None:
    cleaned = clean_dataset(_generate_synthetic_telco(n_rows=300, seed=3))
    splits = split_dataset(cleaned)
    model = build_model_pipeline(LogisticRegression(max_iter=200, class_weight="balanced", random_state=3))
    model.fit(splits.x_train, splits.y_train)
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 4,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "monthly_charges": 89.1,
        "TotalCharges": 350.0,
    }
    frame = normalize_payload(payload)
    assert "MonthlyCharges" in frame.columns
    result = predict_payload(model, payload)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0
