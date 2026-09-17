from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///./test_mlforge.db"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
os.environ["API_KEY"] = "test-key"

from backend.app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from backend.app.db.database import get_db  # noqa: E402
from backend.app.main import create_app  # noqa: E402
from backend.app.models.database_models import Base  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    engine = create_engine("sqlite:///./test_mlforge.db", connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)
