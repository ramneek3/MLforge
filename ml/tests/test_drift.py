from __future__ import annotations

import pandas as pd

from ml.src.drift import compute_drift_report
from ml.src.ingest import _generate_synthetic_telco
from ml.src.data_preprocessing import clean_dataset


def test_drift_flags_shifted_monthly_charges() -> None:
    reference = clean_dataset(_generate_synthetic_telco(n_rows=400, seed=1))
    current = reference.copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] + 80
    report = compute_drift_report(reference, current)
    monthly = next(item for item in report["features"] if item["feature"] == "MonthlyCharges")
    assert monthly["drift"] in {"medium", "high"}
    assert "features" in report


def test_no_drift_on_identical_frames() -> None:
    frame = pd.DataFrame({"tenure": [1, 2, 3, 10, 20], "Contract": ["Month-to-month"] * 5})
    report = compute_drift_report(frame, frame.copy())
    tenure = next(item for item in report["features"] if item["feature"] == "tenure")
    assert tenure["drift"] == "low"
