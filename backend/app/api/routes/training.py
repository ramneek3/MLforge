from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.security import require_api_key
from backend.app.db.database import get_db
from backend.app.models.schemas import TrainRequest, TrainResponse
from backend.app.services.training_service import TrainingService


router = APIRouter(tags=["training"], dependencies=[Depends(require_api_key)])
training_service = TrainingService()


@router.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest, db: Session = Depends(get_db)) -> TrainResponse:
    summary = training_service.train(
        db=db,
        trigger=payload.trigger,
        hyperparameters=payload.hyperparameters or None,
    )
    return TrainResponse(
        status=summary["status"],
        best_model_type=summary["best_model_type"],
        model_version=summary["model_version"],
        metrics=summary["best_metrics"],
        promotable=summary["promotable"],
        promotion_reason=summary["promotion_reason"],
        run_id=summary["best_run_id"],
        candidates=summary["candidates"],
    )
