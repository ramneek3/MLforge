from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.models.schemas import PredictRequest, PredictResponse
from backend.app.services.prediction_service import PredictionService


router = APIRouter(tags=["prediction"])
prediction_service = PredictionService()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request, db: Session = Depends(get_db)) -> PredictResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    body = payload.model_dump(exclude_none=True)
    try:
        result = prediction_service.predict(body, request_id, db)
        return PredictResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction failed: {exc}") from exc
