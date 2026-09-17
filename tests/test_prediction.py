from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.services.prediction_service import PredictionService


def test_predict_response_shape(client: TestClient, monkeypatch) -> None:
    def fake_predict(self, payload, request_id, db):
        return {
            "prediction": 1,
            "churn_probability": 0.83,
            "model_name": "churn-model",
            "model_version": "3",
            "request_id": request_id,
        }

    monkeypatch.setattr(PredictionService, "predict", fake_predict)
    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 12,
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
            "MonthlyCharges": 75.5,
            "TotalCharges": 900.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["churn_probability"] == 0.83
    assert body["model_name"] == "churn-model"
    assert body["model_version"] == "3"
    assert "request_id" in body


def test_admin_requires_api_key(client: TestClient) -> None:
    response = client.post("/train", json={"trigger": "manual"})
    assert response.status_code == 401
