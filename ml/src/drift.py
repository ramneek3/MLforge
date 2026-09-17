from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ml.src.config import load_config


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0
    quantiles = ref.quantile([i / bins for i in range(bins + 1)]).values
    quantiles[0] = min(float(quantiles[0]), float(cur.min()))
    quantiles[-1] = max(float(quantiles[-1]), float(cur.max()))
    if len(set(quantiles)) < 3:
        return 0.0
    ref_counts = pd.cut(ref, bins=quantiles, include_lowest=True).value_counts(normalize=True, sort=False)
    cur_counts = pd.cut(cur, bins=quantiles, include_lowest=True).value_counts(normalize=True, sort=False)
    ref_p = ref_counts.clip(lower=1e-6)
    cur_p = cur_counts.clip(lower=1e-6)
    return float(((cur_p - ref_p) * (cur_p / ref_p).map(math.log)).sum())


def _js_divergence(reference: pd.Series, current: pd.Series) -> float:
    ref_p = reference.astype(str).value_counts(normalize=True)
    cur_p = current.astype(str).value_counts(normalize=True)
    keys = sorted(set(ref_p.index) | set(cur_p.index))
    ref_vals = [max(float(ref_p.get(k, 0.0)), 1e-6) for k in keys]
    cur_vals = [max(float(cur_p.get(k, 0.0)), 1e-6) for k in keys]
    mid = [(r + c) / 2 for r, c in zip(ref_vals, cur_vals)]
    kl_pm = sum(p * math.log(p / m) for p, m in zip(ref_vals, mid))
    kl_qm = sum(q * math.log(q / m) for q, m in zip(cur_vals, mid))
    return float((kl_pm + kl_qm) / 2)


def _severity(score: float, threshold: float) -> str:
    if score >= threshold:
        return "high"
    if score >= threshold * 0.6:
        return "medium"
    return "low"


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    threshold = float(cfg["drift"]["threshold"])
    numerical = [c for c in cfg["features"]["numerical"] if c in reference.columns and c in current.columns]
    categorical = [c for c in cfg["features"]["categorical"] if c in reference.columns and c in current.columns]

    features: list[dict[str, Any]] = []
    for col in numerical:
        score = abs(_psi(reference[col], current[col]))
        features.append({"feature": col, "type": "numerical", "score": round(score, 4), "drift": _severity(score, threshold)})
    for col in categorical:
        score = abs(_js_divergence(reference[col], current[col]))
        features.append({"feature": col, "type": "categorical", "score": round(score, 4), "drift": _severity(score, threshold)})

    high_count = sum(1 for item in features if item["drift"] == "high")
    drifted = high_count > 0 or any(item["score"] >= threshold for item in features)
    return {
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "threshold": threshold,
        "drift_detected": drifted,
        "retraining_required": drifted,
        "reason": "Significant feature drift versus training reference." if drifted else "No significant drift detected.",
        "features": features,
    }


def try_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        numerical = cfg["features"]["numerical"]
        categorical = cfg["features"]["categorical"]
        mapping = ColumnMapping(
            numerical_features=[c for c in numerical if c in reference.columns],
            categorical_features=[c for c in categorical if c in reference.columns],
            target=None,
        )
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current, column_mapping=mapping)
        payload = report.as_dict()
        metrics = payload.get("metrics", [])
        drift_detected = False
        features: list[dict[str, Any]] = []
        for metric in metrics:
            result = metric.get("result", {})
            dataset_drift = result.get("dataset_drift")
            if dataset_drift:
                drift_detected = True
            drift_by_columns = result.get("drift_by_columns") or result.get("drift_by_feature") or {}
            if isinstance(drift_by_columns, dict):
                for name, info in drift_by_columns.items():
                    score = float(info.get("drift_score") or info.get("stattest_threshold") or 0.0)
                    drifted = bool(info.get("drift_detected", score >= float(cfg["drift"]["threshold"])))
                    features.append(
                        {
                            "feature": name,
                            "type": info.get("column_type", "unknown"),
                            "score": round(score, 4),
                            "drift": "high" if drifted else "low",
                        }
                    )
        if features:
            return {
                "n_reference": int(len(reference)),
                "n_current": int(len(current)),
                "threshold": float(cfg["drift"]["threshold"]),
                "drift_detected": drift_detected or any(f["drift"] == "high" for f in features),
                "retraining_required": drift_detected or any(f["drift"] == "high" for f in features),
                "reason": "Evidently detected dataset/feature drift." if drift_detected else "No significant drift detected.",
                "features": features,
                "engine": "evidently",
            }
    except Exception:
        pass
    report = compute_drift_report(reference, current, cfg)
    report["engine"] = "builtin-psi-js"
    return report
