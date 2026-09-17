from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.src.config import load_config


class DatasetValidationError(ValueError):
    pass


@dataclass
class ValidationResult:
    n_rows: int
    n_columns: int
    missing_target: int
    duplicate_ids: int


def validate_raw_dataset(frame: pd.DataFrame, config: dict | None = None) -> ValidationResult:
    cfg = config or load_config()
    target = cfg["data"]["target_column"]
    id_column = cfg["data"]["id_column"]
    required = [target, *cfg["features"]["numerical"], *cfg["features"]["categorical"]]
    missing_cols = [col for col in required if col not in frame.columns]
    if missing_cols:
        raise DatasetValidationError(f"Missing required columns: {missing_cols}")
    if id_column in frame.columns:
        duplicate_ids = int(frame[id_column].duplicated().sum())
    else:
        duplicate_ids = 0
    missing_target = int(frame[target].isna().sum())
    if missing_target:
        raise DatasetValidationError(f"Target column '{target}' contains {missing_target} nulls.")
    if len(frame) < 50:
        raise DatasetValidationError("Dataset has fewer than 50 rows.")
    return ValidationResult(
        n_rows=len(frame),
        n_columns=len(frame.columns),
        missing_target=missing_target,
        duplicate_ids=duplicate_ids,
    )


def clean_dataset(frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    cleaned = frame.copy()
    id_column = cfg["data"]["id_column"]
    if id_column in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[id_column])
        cleaned = cleaned.drop(columns=[id_column])

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)

    numerical = cfg["features"]["numerical"]
    categorical = cfg["features"]["categorical"]
    for col in numerical:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in categorical:
        cleaned[col] = cleaned[col].astype(str).replace({"nan": "Unknown"})
        cleaned[col] = cleaned[col].fillna("Unknown")

    target = cfg["data"]["target_column"]
    positive = cfg["data"]["positive_label"]
    cleaned[target] = (cleaned[target].astype(str).str.strip() == str(positive)).astype(int)
    keep = [*numerical, *categorical, target]
    return cleaned[keep]
