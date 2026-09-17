from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.src.config import load_config, resolve_path
from ml.src.data_preprocessing import clean_dataset, validate_raw_dataset
from ml.src.evaluate import classification_metrics, predict_with_proba, selection_score
from ml.src.feature_engineering import build_model_pipeline, split_dataset
from ml.src.ingest import dataset_hash, load_raw_dataset
from ml.src.promotion import can_promote


@dataclass
class CandidateResult:
    model_type: str
    metrics: dict[str, float]
    run_id: str
    duration_seconds: float
    params: dict[str, Any]


def _estimator_factory(model_type: str, params: dict[str, Any], seed: int) -> Any:
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if model_type == "xgboost":
        xgb_params = dict(params)
        xgb_params.pop("eval_metric", None)
        return XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            eval_metric=params.get("eval_metric", "logloss"),
            **xgb_params,
        )
    raise ValueError(f"Unsupported model type '{model_type}'")


def _get_production_metrics(model_name: str) -> dict[str, float] | None:
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        aliases = [v for v in versions if "production" in (v.aliases or [])]
        production = aliases
    if not production:
        return None
    run = client.get_run(production[0].run_id)
    metrics = run.data.metrics
    if "f1" not in metrics or "roc_auc" not in metrics:
        return None
    return {k: float(metrics[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in metrics}


def train_and_register(
    config_path: str | None = None,
    hyperparameter_overrides: dict[str, dict[str, Any]] | None = None,
    trigger: str = "manual",
) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["project"]["random_seed"])
    experiment_name = config["project"]["experiment_name"]
    registered_name = config["project"]["registered_model_name"]
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    validation = validate_raw_dataset(raw, config)
    cleaned = clean_dataset(raw, config)
    data_hash = dataset_hash(cleaned)
    splits = split_dataset(cleaned, config)

    cleaned.to_parquet(processed_dir / "cleaned.parquet", index=False)
    splits.x_train.assign(**{config["data"]["target_column"]: splits.y_train}).to_parquet(
        processed_dir / "train.parquet", index=False
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    candidates: list[CandidateResult] = []
    model_params = {name: dict(values) for name, values in config["models"].items()}
    if hyperparameter_overrides:
        for key, values in hyperparameter_overrides.items():
            if key in model_params:
                model_params[key] = {**model_params[key], **values}

    for model_type, params in model_params.items():
        estimator = _estimator_factory(model_type, params, seed)
        pipeline = build_model_pipeline(estimator, config)
        started = time.perf_counter()
        with mlflow.start_run(run_name=f"{model_type}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}") as run:
            pipeline.fit(splits.x_train, splits.y_train)
            duration = time.perf_counter() - started
            val_pred, val_proba = predict_with_proba(pipeline, splits.x_val)
            test_pred, test_proba = predict_with_proba(pipeline, splits.x_test)
            val_metrics = classification_metrics(splits.y_val, val_pred, val_proba)
            test_metrics = classification_metrics(splits.y_test, test_pred, test_proba)

            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", registered_name)
            mlflow.log_param("dataset_hash", data_hash)
            mlflow.log_param("trigger", trigger)
            mlflow.log_param("n_rows", validation.n_rows)
            mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
            mlflow.log_metric("training_duration_seconds", duration)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
            for key, value in test_metrics.items():
                mlflow.log_metric(key, value)
            mlflow.log_metric("selection_score", selection_score(test_metrics))

            artifact_dir = processed_dir / "artifacts" / run.info.run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = artifact_dir / "metrics.json"
            metrics_path.write_text(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(metrics_path))
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            candidates.append(
                CandidateResult(
                    model_type=model_type,
                    metrics=test_metrics,
                    run_id=run.info.run_id,
                    duration_seconds=duration,
                    params=params,
                )
            )

    best = max(candidates, key=lambda item: selection_score(item.metrics))
    production_metrics = _get_production_metrics(registered_name)
    promotable, reason = can_promote(best.metrics, production_metrics, config)

    with mlflow.start_run(run_id=best.run_id):
        mlflow.set_tag("selected_best", "true")
        mlflow.set_tag("promotion_reason", reason)
        model_uri = f"runs:/{best.run_id}/model"
        registered = mlflow.register_model(model_uri, registered_name)
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(registered_name, registered.version, "status", "candidate")
        client.set_model_version_tag(registered_name, registered.version, "model_type", best.model_type)
        client.set_registered_model_alias(registered_name, "candidate", registered.version)
        if promotable and production_metrics is None:
            client.transition_model_version_stage(
                name=registered_name,
                version=registered.version,
                stage="Production",
                archive_existing_versions=True,
            )
            client.set_model_version_tag(registered_name, registered.version, "status", "production")
            client.set_registered_model_alias(registered_name, "production", registered.version)
            stage_after = "production"
        else:
            client.transition_model_version_stage(
                name=registered_name,
                version=registered.version,
                stage="Staging" if promotable else "None",
                archive_existing_versions=False,
            )
            client.set_model_version_tag(
                registered_name,
                registered.version,
                "status",
                "staging" if promotable else "candidate",
            )
            stage_after = "staging" if promotable else "candidate"

    summary = {
        "experiment_name": experiment_name,
        "registered_model_name": registered_name,
        "dataset_hash": data_hash,
        "best_model_type": best.model_type,
        "best_run_id": best.run_id,
        "best_metrics": best.metrics,
        "model_version": str(registered.version),
        "status": stage_after,
        "promotable": promotable,
        "promotion_reason": reason,
        "trigger": trigger,
        "candidates": [asdict(item) for item in candidates],
    }
    (processed_dir / "last_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump({"summary": summary}, processed_dir / "last_training_summary.joblib")
    return summary


if __name__ == "__main__":
    result = train_and_register()
    print(json.dumps(result, indent=2))
