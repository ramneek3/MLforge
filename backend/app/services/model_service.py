from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from backend.app.core.config import get_settings
from ml.src.config import load_config
from ml.src.promotion import can_promote


class ModelService:
    def __init__(self) -> None:
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self.client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        self.model_name = self.settings.mlflow_registered_model_name

    def list_versions(self) -> list[dict[str, Any]]:
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
        except Exception:
            return []
        results = []
        for version in versions:
            metrics = self._metrics_for_run(version.run_id)
            tags = version.tags or {}
            status = tags.get("status") or self._status_from_stage(version.current_stage)
            results.append(
                {
                    "model_name": version.name,
                    "version": str(version.version),
                    "status": status,
                    "model_type": tags.get("model_type"),
                    "created_at": version.creation_timestamp,
                    "run_id": version.run_id,
                    **metrics,
                }
            )
        results.sort(key=lambda item: int(item["version"]), reverse=True)
        return results

    def get_production(self) -> dict[str, Any] | None:
        versions = self.list_versions()
        for item in versions:
            if item["status"] == "production":
                return item
        try:
            prod = self.client.get_model_version_by_alias(self.model_name, "production")
            metrics = self._metrics_for_run(prod.run_id)
            return {
                "model_name": prod.name,
                "version": str(prod.version),
                "status": "production",
                "run_id": prod.run_id,
                **metrics,
            }
        except Exception:
            return None

    def load_production_model(self) -> tuple[Any, str]:
        production = self.get_production()
        if production is None:
            raise RuntimeError("No production model is registered.")
        version = production["version"]
        try:
            model = mlflow.sklearn.load_model(f"models:/{self.model_name}@production")
        except Exception:
            model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")
        return model, version

    def promote(self, version: str, action: str) -> dict[str, Any]:
        versions = {item["version"]: item for item in self.list_versions()}
        if version not in versions:
            raise ValueError(f"Version {version} not found for {self.model_name}.")
        candidate = versions[version]
        config = load_config(self.settings.ml_config_path)

        if action == "reject":
            self._set_status(version, "rejected", stage="None")
            return {"status": "rejected", "reason": "Manually rejected.", "version": version}
        if action == "archive":
            self._set_status(version, "archived", stage="Archived")
            return {"status": "archived", "reason": "Manually archived.", "version": version}
        if action == "stage":
            self._set_status(version, "staging", stage="Staging")
            return {"status": "staging", "reason": "Moved to staging.", "version": version}

        production = self.get_production()
        production_metrics = None
        if production and all(production.get(k) is not None for k in ("f1", "roc_auc")):
            production_metrics = {
                "f1": float(production["f1"]),
                "roc_auc": float(production["roc_auc"]),
            }
        candidate_metrics = {"f1": float(candidate.get("f1") or 0), "roc_auc": float(candidate.get("roc_auc") or 0)}
        allowed, reason = can_promote(candidate_metrics, production_metrics, config)
        if not allowed:
            self._set_status(version, "rejected", stage="None")
            return {"status": "rejected", "reason": reason, "version": version}

        current_status = candidate["status"]
        if current_status == "candidate":
            self._set_status(version, "staging", stage="Staging")
            return {"status": "staging", "reason": reason, "version": version}
        self._set_status(version, "production", stage="Production", archive_existing=True)
        try:
            self.client.set_registered_model_alias(self.model_name, "production", version)
        except Exception:
            pass
        return {"status": "production", "reason": reason, "version": version}

    def list_experiments(self) -> list[dict[str, Any]]:
        experiment = self.client.get_experiment_by_name(self.settings.mlflow_experiment_name)
        if experiment is None:
            return []
        runs = self.client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50)
        payload = []
        for run in runs:
            payload.append(
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "duration_seconds": ((run.info.end_time or run.info.start_time) - run.info.start_time) / 1000,
                    "model_type": run.data.params.get("model_type"),
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }
            )
        return payload

    def _set_status(self, version: str, status: str, stage: str, archive_existing: bool = False) -> None:
        self.client.set_model_version_tag(self.model_name, version, "status", status)
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
        except Exception:
            pass

    def _metrics_for_run(self, run_id: str) -> dict[str, float | None]:
        keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
        try:
            run = self.client.get_run(run_id)
            return {key: float(run.data.metrics[key]) if key in run.data.metrics else None for key in keys}
        except Exception:
            return {key: None for key in keys}

    @staticmethod
    def _status_from_stage(stage: str | None) -> str:
        mapping = {
            "Production": "production",
            "Staging": "staging",
            "Archived": "archived",
            "None": "candidate",
        }
        return mapping.get(stage or "None", "candidate")
