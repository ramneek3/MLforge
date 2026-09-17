from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "MLForge"
    assert "X-Request-ID" in response.headers


def test_predict_validation_error(client: TestClient) -> None:
    response = client.post("/predict", json={"tenure": "bad"})
    assert response.status_code == 422
