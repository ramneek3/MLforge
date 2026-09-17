from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.database_models import (
    DeploymentRecord,
    ModelRecord,
    ModelVersionRecord,
    PredictionRecord,
    TrainingRunRecord,
)


class MetadataRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_prediction(self, **kwargs) -> PredictionRecord:
        record = PredictionRecord(**kwargs)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def add_training_run(self, **kwargs) -> TrainingRunRecord:
        record = TrainingRunRecord(**kwargs)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def upsert_model(self, name: str) -> ModelRecord:
        existing = self.db.scalar(select(ModelRecord).where(ModelRecord.name == name))
        if existing:
            return existing
        record = ModelRecord(name=name)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def add_model_version(self, **kwargs) -> ModelVersionRecord:
        record = ModelVersionRecord(**kwargs)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def list_versions(self, model_name: str) -> list[ModelVersionRecord]:
        return list(
            self.db.scalars(
                select(ModelVersionRecord)
                .where(ModelVersionRecord.model_name == model_name)
                .order_by(ModelVersionRecord.created_at.desc())
            )
        )

    def get_version(self, model_name: str, version: str) -> ModelVersionRecord | None:
        return self.db.scalar(
            select(ModelVersionRecord).where(
                ModelVersionRecord.model_name == model_name,
                ModelVersionRecord.version == version,
            )
        )

    def production_version(self, model_name: str) -> ModelVersionRecord | None:
        return self.db.scalar(
            select(ModelVersionRecord).where(
                ModelVersionRecord.model_name == model_name,
                ModelVersionRecord.status == "production",
            )
        )

    def set_version_status(self, record: ModelVersionRecord, status: str) -> ModelVersionRecord:
        if status == "production":
            current = self.list_versions(record.model_name)
            for item in current:
                if item.status == "production" and item.id != record.id:
                    item.status = "archived"
        record.status = status
        self.db.commit()
        self.db.refresh(record)
        return record

    def add_deployment(self, **kwargs) -> DeploymentRecord:
        record = DeploymentRecord(**kwargs)
        self.db.add(record)
        self.db.commit()
        return record

    def prediction_stats(self) -> dict:
        total = self.db.scalar(select(func.count(PredictionRecord.id))) or 0
        errors = self.db.scalar(
            select(func.count(PredictionRecord.id)).where(PredictionRecord.status != "ok")
        ) or 0
        avg_latency = self.db.scalar(select(func.avg(PredictionRecord.latency_ms))) or 0.0
        positives = self.db.scalar(
            select(func.count(PredictionRecord.id)).where(PredictionRecord.prediction == 1)
        ) or 0
        return {
            "total_predictions": int(total),
            "error_rate": float(errors / total) if total else 0.0,
            "avg_latency_ms": float(avg_latency),
            "positive_prediction_rate": float(positives / total) if total else 0.0,
        }

    def recent_prediction_features(self, limit: int = 200) -> list[PredictionRecord]:
        return list(
            self.db.scalars(
                select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
            )
        )

    def list_training_runs(self, limit: int = 50) -> list[TrainingRunRecord]:
        return list(
            self.db.scalars(select(TrainingRunRecord).order_by(TrainingRunRecord.created_at.desc()).limit(limit))
        )
