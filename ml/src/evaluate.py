from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def predict_with_proba(model: Any, features: Any, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        proba = model.predict(features).astype(float)
    preds = (proba >= threshold).astype(int)
    return preds, proba


def selection_score(metrics: dict[str, float]) -> float:
    return 0.6 * metrics["f1"] + 0.4 * metrics["roc_auc"]
