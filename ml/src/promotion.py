from __future__ import annotations

from typing import Any

from ml.src.config import load_config


STATUSES = ("candidate", "staging", "production", "archived", "rejected")


def can_promote(
    candidate_metrics: dict[str, float],
    production_metrics: dict[str, float] | None,
    config: dict | None = None,
) -> tuple[bool, str]:
    cfg = (config or load_config())["promotion"]
    min_f1 = float(cfg["min_f1"])
    min_auc = float(cfg["min_roc_auc"])
    require_improvement = bool(cfg.get("require_f1_improvement", True))
    min_delta = float(cfg.get("min_f1_delta", 0.0))

    candidate_f1 = float(candidate_metrics["f1"])
    candidate_auc = float(candidate_metrics["roc_auc"])

    if candidate_f1 < min_f1:
        return False, f"Rejected: candidate F1 {candidate_f1:.4f} below threshold {min_f1:.4f}."
    if candidate_auc < min_auc:
        return False, f"Rejected: candidate ROC-AUC {candidate_auc:.4f} below threshold {min_auc:.4f}."

    if production_metrics is None:
        return True, "Promotable: no production model exists and quality gates passed."

    production_f1 = float(production_metrics["f1"])
    if require_improvement and candidate_f1 <= production_f1 + min_delta:
        return (
            False,
            (
                f"Rejected: candidate F1 {candidate_f1:.4f} does not improve over "
                f"production F1 {production_f1:.4f}."
            ),
        )
    return (
        True,
        (
            f"Promotable: candidate F1 {candidate_f1:.4f} and ROC-AUC {candidate_auc:.4f} "
            f"pass gates versus production F1 {production_f1:.4f}."
        ),
    )


def next_status(current: str, action: str) -> str:
    current = current.lower()
    action = action.lower()
    if current not in STATUSES:
        raise ValueError(f"Unknown status '{current}'")
    if action == "reject":
        return "rejected"
    if action == "archive":
        return "archived"
    if action == "stage" and current in {"candidate", "rejected"}:
        return "staging"
    if action == "promote":
        if current == "candidate":
            return "staging"
        if current == "staging":
            return "production"
        if current == "production":
            return "production"
    raise ValueError(f"Cannot apply action '{action}' to status '{current}'.")
