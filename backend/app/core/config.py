from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MLForge"
    app_env: str = "development"
    log_level: str = "INFO"
    api_key: str = "changeme-admin-api-key"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080,https://mlforge-omega.vercel.app"
    database_url: str = "postgresql+psycopg2://mlforge:mlforge@localhost:5432/mlforge"
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "customer-churn-classification"
    mlflow_registered_model_name: str = "churn-model"
    ml_config_path: str = "ml/configs/config.yaml"
    data_raw_path: str = "ml/data/raw/telco_churn.csv"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
