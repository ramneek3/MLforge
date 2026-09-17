from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text
from starlette.responses import Response

from backend.app.api.routes import models as models_routes
from backend.app.api.routes import monitoring as monitoring_routes
from backend.app.api.routes import prediction as prediction_routes
from backend.app.api.routes import training as training_routes
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.metrics import FAILED_REQUESTS, REQUESTS_TOTAL
from backend.app.core.security import RequestContextMiddleware
from backend.app.db.database import engine, init_db
from backend.app.models.schemas import ErrorResponse, HealthResponse, ReadyResponse
from backend.app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="MLForge",
        description="Production-oriented MLOps platform for customer churn.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )
    Instrumentator().instrument(app)

    app.include_router(prediction_routes.router)
    app.include_router(training_routes.router)
    app.include_router(models_routes.router)
    app.include_router(monitoring_routes.router)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            FAILED_REQUESTS.labels(endpoint=request.url.path).inc()
            REQUESTS_TOTAL.labels(endpoint=request.url.path, method=request.method, status="500").inc()
            raise
        REQUESTS_TOTAL.labels(
            endpoint=request.url.path, method=request.method, status=str(response.status_code)
        ).inc()
        if response.status_code >= 400:
            FAILED_REQUESTS.labels(endpoint=request.url.path).inc()
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        get_logger(request_id=request_id).error("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Internal server error", request_id=request_id).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=settings.app_name)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready() -> ReadyResponse:
        database_ok = True
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            database_ok = False
        mlflow_ok = True
        model_loaded = False
        try:
            production = ModelService().get_production()
            model_loaded = production is not None
        except Exception:
            mlflow_ok = False
        status = "ok" if database_ok else "degraded"
        return ReadyResponse(status=status, database=database_ok, mlflow=mlflow_ok, model_loaded=model_loaded)

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
