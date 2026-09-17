from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter("mlforge_requests_total", "Total HTTP requests", ["endpoint", "method", "status"])
PREDICTION_REQUESTS = Counter("mlforge_prediction_requests_total", "Prediction requests", ["result"])
FAILED_REQUESTS = Counter("mlforge_failed_requests_total", "Failed requests", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "mlforge_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
TRAINING_DURATION = Histogram("mlforge_training_duration_seconds", "Training job duration")
MODEL_VERSION = Gauge("mlforge_production_model_version", "Current production model version")
PREDICTION_DISTRIBUTION = Counter("mlforge_prediction_distribution", "Prediction class counts", ["label"])
