from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from backend.app.core.metrics import TRAINING_DURATION
from backend.app.db.repositories import MetadataRepository
from backend.app.services.model_service import ModelService
from ml.src.train import train_and_register


ALLOWED_HYPERPARAMS = {
    "logistic_regression": {"C", "max_iter", "class_weight"},
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "class_weight",
        "n_jobs",
    },
    "xgboost": {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "eval_metric",
    },
}


class TrainingService:
    def __init__(self) -> None:
        self.model_service = ModelService()

    def train(self, db: Session, trigger: str = "manual", hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
        overrides: dict[str, dict[str, Any]] | None = None
        if hyperparameters:
            if any(k in hyperparameters for k in ALLOWED_HYPERPARAMS):
                overrides = {
                    key: {k: v for k, v in values.items() if k in ALLOWED_HYPERPARAMS[key]}
                    for key, values in hyperparameters.items()
                    if key in ALLOWED_HYPERPARAMS and isinstance(values, dict)
                }
            else:
                allowed = ALLOWED_HYPERPARAMS["xgboost"]
                overrides = {"xgboost": {k: v for k, v in hyperparameters.items() if k in allowed}}

        summary = train_and_register(hyperparameter_overrides=overrides, trigger=trigger)
        duration = max((item["duration_seconds"] for item in summary["candidates"]), default=0.0)
        TRAINING_DURATION.observe(duration)
        repo = MetadataRepository(db)
        repo.upsert_model(summary["registered_model_name"])
        repo.add_training_run(
            run_id=summary["best_run_id"],
            experiment_name=summary["experiment_name"],
            model_type=summary["best_model_type"],
            model_version=summary["model_version"],
            trigger=trigger,
            status=summary["status"],
            duration_seconds=duration,
            metrics_json=json.dumps(summary["best_metrics"]),
            promotion_reason=summary["promotion_reason"],
            dataset_hash=summary["dataset_hash"],
        )
        metrics = summary["best_metrics"]
        repo.add_model_version(
            model_name=summary["registered_model_name"],
            version=summary["model_version"],
            status=summary["status"],
            run_id=summary["best_run_id"],
            model_type=summary["best_model_type"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
        )
        repo.add_deployment(
            model_name=summary["registered_model_name"],
            model_version=summary["model_version"],
            environment=summary["status"],
            action="register",
            reason=summary["promotion_reason"],
        )
        return summary
