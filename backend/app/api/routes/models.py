from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.security import require_api_key
from backend.app.db.database import get_db
from backend.app.db.repositories import MetadataRepository
from backend.app.models.schemas import ModelVersionResponse, PromoteRequest
from backend.app.services.model_service import ModelService


router = APIRouter(tags=["models"])
model_service = ModelService()


@router.get("/models")
def list_models() -> dict:
    settings = get_settings()
    return {
        "models": [
            {
                "name": settings.mlflow_registered_model_name,
                "versions": model_service.list_versions(),
            }
        ]
    }


@router.get("/models/{model_name}")
def get_model(model_name: str) -> dict:
    if model_name != get_settings().mlflow_registered_model_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    return {"name": model_name, "versions": model_service.list_versions()}


@router.get("/models/{model_name}/versions", response_model=list[ModelVersionResponse])
def get_versions(model_name: str) -> list[ModelVersionResponse]:
    if model_name != get_settings().mlflow_registered_model_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    versions = model_service.list_versions()
    return [
        ModelVersionResponse(
            model_name=item["model_name"],
            version=item["version"],
            status=item["status"],
            model_type=item.get("model_type"),
            accuracy=item.get("accuracy"),
            precision=item.get("precision"),
            recall=item.get("recall"),
            f1=item.get("f1"),
            roc_auc=item.get("roc_auc"),
            created_at=str(item.get("created_at")),
            run_id=item.get("run_id"),
        )
        for item in versions
    ]


@router.post("/models/{model_name}/promote")
def promote_model(
    model_name: str,
    payload: PromoteRequest,
    db: Session = Depends(get_db),
    _: str = Depends(require_api_key),
) -> dict:
    if model_name != get_settings().mlflow_registered_model_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    try:
        result = model_service.promote(payload.version, payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    repo = MetadataRepository(db)
    record = repo.get_version(model_name, payload.version)
    if record:
        repo.set_version_status(record, result["status"])
    repo.add_deployment(
        model_name=model_name,
        model_version=payload.version,
        environment=result["status"],
        action=payload.action,
        reason=result["reason"],
    )
    return result


@router.get("/experiments")
def experiments() -> dict:
    return {"experiments": model_service.list_experiments()}
