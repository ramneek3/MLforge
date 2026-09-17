from __future__ import annotations

import pandas as pd

from ml.src.data_preprocessing import clean_dataset, validate_raw_dataset
from ml.src.evaluate import classification_metrics, selection_score
from ml.src.feature_engineering import split_dataset
from ml.src.ingest import _generate_synthetic_telco
from ml.src.promotion import can_promote


def test_preprocessing_encodes_target_and_handles_total_charges() -> None:
    raw = _generate_synthetic_telco(n_rows=200, seed=7)
    raw.loc[0, "TotalCharges"] = " "
    validate_raw_dataset(raw)
    cleaned = clean_dataset(raw)
    assert cleaned["Churn"].isin([0, 1]).all()
    assert cleaned["TotalCharges"].dtype != object
    assert cleaned.isna().sum().sum() == 0
    assert "customerID" not in cleaned.columns


def test_split_is_stratified_and_reproducible() -> None:
    cleaned = clean_dataset(_generate_synthetic_telco(n_rows=400, seed=42))
    first = split_dataset(cleaned)
    second = split_dataset(cleaned)
    assert first.x_train.equals(second.x_train)
    assert set(first.y_train.unique()) == {0, 1}
    total = len(first.x_train) + len(first.x_val) + len(first.x_test)
    assert total == len(cleaned)


def test_metrics_and_selection_score() -> None:
    y_true = pd.Series([0, 1, 1, 0, 1, 1])
    y_pred = [0, 1, 0, 0, 1, 1]
    y_proba = [0.1, 0.9, 0.4, 0.2, 0.8, 0.7]
    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["f1"] > 0
    assert selection_score(metrics) > 0


def test_promotion_rejects_worse_candidate() -> None:
    config = {
        "promotion": {
            "min_f1": 0.5,
            "min_roc_auc": 0.7,
            "require_f1_improvement": True,
            "min_f1_delta": 0.0,
        }
    }
    allowed, reason = can_promote({"f1": 0.62, "roc_auc": 0.81}, None, config)
    assert allowed
    rejected, reject_reason = can_promote(
        {"f1": 0.60, "roc_auc": 0.80},
        {"f1": 0.70, "roc_auc": 0.82},
        config,
    )
    assert not rejected
    assert "does not improve" in reject_reason
    assert "Promotable" in reason
