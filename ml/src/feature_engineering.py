from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src.config import load_config


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_features_and_target(frame: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cfg = config or load_config()
    target = cfg["data"]["target_column"]
    return frame.drop(columns=[target]), frame[target]


def split_dataset(frame: pd.DataFrame, config: dict | None = None) -> SplitData:
    cfg = config or load_config()
    seed = int(cfg["project"]["random_seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])
    features, target = split_features_and_target(frame, cfg)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=relative_val,
        random_state=seed,
        stratify=y_train_full,
    )
    return SplitData(x_train, x_val, x_test, y_train, y_val, y_test)


def build_preprocessor(config: dict | None = None) -> ColumnTransformer:
    cfg = config or load_config()
    numerical = cfg["features"]["numerical"]
    categorical = cfg["features"]["categorical"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_model_pipeline(estimator: Any, config: dict | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("model", estimator),
        ]
    )
