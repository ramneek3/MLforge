from __future__ import annotations

import json

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.repositories import MetadataRepository
from backend.app.services.model_service import ModelService
from ml.src.config import load_config, resolve_path
from ml.src.drift import try_evidently_report
from ml.src.ingest import load_raw_dataset
from ml.src.data_preprocessing import clean_dataset


class MonitoringService:
    def __init__(self) -> None:
        self.model_service = ModelService()
        self.settings = get_settings()

    def summary(self, db: Session) -> dict:
        repo = MetadataRepository(db)
        stats = repo.prediction_stats()
        production = self.model_service.get_production()
        drift = self.drift(db)
        return {
            **stats,
            "production_model_version": production["version"] if production else None,
            "production_f1": production.get("f1") if production else None,
            "production_roc_auc": production.get("roc_auc") if production else None,
            "drift_detected": drift["drift_detected"],
            "retraining_required": drift["retraining_required"],
        }

    def drift(self, db: Session) -> dict:
        repo = MetadataRepository(db)
        config = load_config(self.settings.ml_config_path)
        min_rows = int(config["drift"]["min_inference_rows"])
        records = repo.recent_prediction_features(limit=500)
        if len(records) < min_rows:
            return {
                "n_reference": 0,
                "n_current": len(records),
                "threshold": config["drift"]["threshold"],
                "drift_detected": False,
                "retraining_required": False,
                "reason": f"Not enough inference rows for drift (have {len(records)}, need {min_rows}).",
                "features": [],
                "engine": "none",
            }
        current_rows = []
        for record in records:
            if not record.features_json:
                continue
            current_rows.append(json.loads(record.features_json))
        current = pd.DataFrame(current_rows)
        processed = resolve_path(config["data"]["processed_dir"]) / "train.parquet"
        if processed.exists():
            reference = pd.read_parquet(processed)
        else:
            reference = clean_dataset(load_raw_dataset())
        overlap_cols = [col for col in current.columns if col in reference.columns]
        if not overlap_cols:
            return {
                "n_reference": int(len(reference)),
                "n_current": int(len(current)),
                "threshold": config["drift"]["threshold"],
                "drift_detected": False,
                "retraining_required": False,
                "reason": "Logged inference features do not overlap training columns yet.",
                "features": [],
                "engine": "none",
            }
        return try_evidently_report(reference[overlap_cols], current[overlap_cols], config)
