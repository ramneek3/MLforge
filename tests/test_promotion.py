from __future__ import annotations

from ml.src.promotion import can_promote, next_status


def test_next_status_transitions() -> None:
    assert next_status("candidate", "promote") == "staging"
    assert next_status("staging", "promote") == "production"
    assert next_status("candidate", "reject") == "rejected"


def test_quality_gate_threshold() -> None:
    config = {
        "promotion": {
            "min_f1": 0.5,
            "min_roc_auc": 0.7,
            "require_f1_improvement": True,
            "min_f1_delta": 0.0,
        }
    }
    ok, _ = can_promote({"f1": 0.71, "roc_auc": 0.82}, {"f1": 0.65, "roc_auc": 0.80}, config)
    assert ok
    bad, reason = can_promote({"f1": 0.4, "roc_auc": 0.9}, None, config)
    assert not bad
    assert "F1" in reason
