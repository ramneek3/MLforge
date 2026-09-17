from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.security import require_api_key

from backend.app.db.database import get_db
from backend.app.models.schemas import MonitoringSummary
from backend.app.services.monitoring_service import MonitoringService
from backend.app.services.training_service import TrainingService


router = APIRouter(tags=["monitoring"])
monitoring_service = MonitoringService()
training_service = TrainingService()


@router.get("/monitoring/summary", response_model=MonitoringSummary)
def monitoring_summary(db: Session = Depends(get_db)) -> MonitoringSummary:
    return MonitoringSummary(**monitoring_service.summary(db))


@router.get("/monitoring/drift")
def monitoring_drift(db: Session = Depends(get_db)) -> dict:
    return monitoring_service.drift(db)


@router.post("/monitoring/retrain")
def retrain_from_drift(db: Session = Depends(get_db), _: str = Depends(require_api_key)) -> dict:
    drift = monitoring_service.drift(db)
    if not drift.get("retraining_required"):
        return {"status": "skipped", "reason": drift.get("reason"), "drift": drift}
    summary = training_service.train(db=db, trigger="drift")
    return {"status": "trained", "reason": drift.get("reason"), "training": summary, "drift": drift}
