from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.config import load_config, resolve_path


TELCO_URLS = [
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
]


def dataset_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _generate_synthetic_telco(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=n_rows)
    monthly = np.round(rng.uniform(18.0, 120.0, size=n_rows), 2)
    total = np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.8, 1.1, size=n_rows), 2)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.35, 0.45, 0.20])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_rows,
    )
    churn_prob = (
        0.08
        + 0.28 * (contracts == "Month-to-month")
        + 0.12 * (internet == "Fiber optic")
        + 0.10 * (payment == "Electronic check")
        + 0.15 * (tenure < 6)
        + 0.08 * (monthly > 80)
        - 0.12 * (contracts == "Two year")
    )
    churn_prob = np.clip(churn_prob, 0.03, 0.85)
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    def yes_no(p_yes: float) -> np.ndarray:
        return rng.choice(["Yes", "No"], size=n_rows, p=[p_yes, 1 - p_yes])

    frame = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:06d}" for i in range(n_rows)],
            "gender": rng.choice(["Female", "Male"], size=n_rows),
            "SeniorCitizen": rng.integers(0, 2, size=n_rows),
            "Partner": yes_no(0.48),
            "Dependents": yes_no(0.30),
            "tenure": tenure,
            "PhoneService": yes_no(0.90),
            "MultipleLines": rng.choice(["No phone service", "No", "Yes"], size=n_rows),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["No", "Yes", "No internet service"], size=n_rows),
            "OnlineBackup": rng.choice(["No", "Yes", "No internet service"], size=n_rows),
            "DeviceProtection": rng.choice(["No", "Yes", "No internet service"], size=n_rows),
            "TechSupport": rng.choice(["No", "Yes", "No internet service"], size=n_rows),
            "StreamingTV": rng.choice(["No", "Yes", "No internet service"], size=n_rows),
            "StreamingMovies": rng.choice(["No", "Yes", "No internet service"], size=n_rows),
            "Contract": contracts,
            "PaperlessBilling": yes_no(0.59),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(str),
            "Churn": churn,
        }
    )
    return frame


def ingest_dataset(
    destination: Path | None = None,
    *,
    allow_download: bool = True,
    fallback_synthetic: bool = True,
) -> Path:
    config = load_config()
    dest = destination or resolve_path(config["data"]["raw_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if allow_download:
        for url in TELCO_URLS:
            try:
                frame = pd.read_csv(url)
                if "Churn" not in frame.columns:
                    continue
                frame.to_csv(dest, index=False)
                return dest
            except Exception:
                continue

    if fallback_synthetic:
        frame = _generate_synthetic_telco(seed=int(config["project"]["random_seed"]))
        frame.to_csv(dest, index=False)
        return dest

    raise FileNotFoundError(
        f"Dataset not found at {dest}. Place Telco Customer Churn CSV there or re-run with download enabled."
    )


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    config = load_config()
    source = path or resolve_path(config["data"]["raw_path"])
    if not source.exists():
        ingest_dataset(source)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("Raw dataset is empty.")
    return frame


if __name__ == "__main__":
    output = ingest_dataset()
    print(f"Dataset ready at {output}")
