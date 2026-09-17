# MLForge

End-to-end MLOps platform for **customer churn prediction**: ingest data, train candidate models, track experiments in MLflow, register and promote versions, serve inference, log predictions, monitor drift, and retrain only when quality gates pass.

This is a portfolio-grade system for an AI infrastructure / MLOps internship. The model is a means to demonstrate the platform, not the other way around.

## Problem statement

Telecom (or similar subscription) businesses lose revenue when customers churn. A single notebook that trains a classifier is not enough in production: you need reproducible training, experiment comparison, a model registry, promotion gates, a serving API, prediction telemetry, drift detection, and a path to retrain without blindly replacing a good production model.

MLForge implements that lifecycle for a churn classifier.

## Architecture

```mermaid
flowchart LR
  UI[React dashboard] --> API[FastAPI]
  API --> DB[(PostgreSQL / SQLite)]
  API --> ML[ML pipeline]
  ML --> MLflow[MLflow tracking + registry]
  API --> Serve[Production model serving]
  Serve --> Prom[Prometheus]
  Prom --> Graf[Grafana]
  Serve --> Drift[Evidently / PSI-JS drift]
  Drift --> Retrain[Retrain candidate]
  Retrain --> Gates[Promotion gates]
  Gates --> Serve
```

Training path: dataset → validate/clean → encode/scale → train LR / RF / XGBoost → log params/metrics/artifacts → select by F1 + ROC-AUC → register `churn-model` → promote only if gates pass.

Serving path: `POST /predict` loads the **current production** version (not a hardcoded version), returns probability + request id, and stores non-PII prediction metadata.

## Features

- Reproducible churn training pipeline with YAML hyperparameters and a fixed random seed
- Three candidates: Logistic Regression, Random Forest, XGBoost
- MLflow experiment `customer-churn-classification` and registry model `churn-model`
- Promotion states: candidate → staging → production, plus rejected / archived
- Quality gates: minimum F1 / ROC-AUC and required improvement over production
- FastAPI inference, training, registry, monitoring, and Prometheus `/metrics`
- Prediction logging (request id, version, probability, latency) without storing customer IDs
- Drift comparison of training reference vs recent inference features
- Drift-triggered retraining that records the reason and still applies promotion gates
- React dashboard: Dashboard, Models, Experiments, Training, Monitoring
- Docker Compose, Kubernetes/Minikube manifests, GitHub Actions CI

## Tech stack

Python 3.11, pandas, numpy, scikit-learn, XGBoost, MLflow, Evidently, FastAPI, Pydantic, SQLAlchemy, PostgreSQL (SQLite for zero-setup local), React, TypeScript, Vite, Tailwind, Recharts, Docker Compose, Kubernetes, Prometheus, Grafana, GitHub Actions.

## Repository structure

```
mlforge/
  backend/          FastAPI app, services, DB models
  ml/               training, preprocessing, drift, configs, ML tests
  frontend/         React dashboard
  kubernetes/       Minikube manifests
  monitoring/       Prometheus + Grafana provisioning
  tests/            API tests
  docker-compose.yml
  Makefile
```

## Local setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
# For the fastest local path, set:
# DATABASE_URL=sqlite:///./mlforge.db
# MLFLOW_TRACKING_URI=file:./mlruns
npm install --prefix frontend
```

## Environment variables

See `.env.example`. Important values:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy URL |
| `MLFLOW_TRACKING_URI` | MLflow server or `file:./mlruns` |
| `API_KEY` | Admin key for `/train` and `/models/.../promote` (`X-API-Key`) |
| `VITE_API_BASE_URL` | Frontend API origin |

Never commit real secrets. `.env` is gitignored.

## Dataset setup

Place IBM Telco Customer Churn CSV at `ml/data/raw/telco_churn.csv`, **or** let ingestion download it / generate a schema-compatible synthetic fallback:

```bash
PYTHONPATH=. python -m ml.src.ingest
```

Target column: `Churn`. Customer IDs are dropped during cleaning and are never stored by the prediction API.

## Running ML training

```bash
PYTHONPATH=. MLFLOW_TRACKING_URI=file:./mlruns python -m ml.src.train
```

Or `make train`. This trains the three candidates, logs to MLflow, registers the best as `churn-model`, and auto-promotes only if there is no production model and quality gates pass.

Hyperparameters live in `ml/configs/config.yaml`, not in scattered code.

## MLflow usage

File store (simplest):

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

Tracking server (Compose maps host 5001 → container 5000):

```bash
mlflow server --host 0.0.0.0 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Open http://localhost:5001 and inspect experiment `customer-churn-classification`.

## Running backend

```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000
```

Docs: http://localhost:8000/docs  
Health: http://localhost:8000/health

## Running frontend

```bash
npm run dev --prefix frontend
```

Dashboard: http://localhost:5173

From repo root, `npm run dev` also starts the frontend.

## Running Docker Compose

```bash
docker compose up --build
```

Services:

- Frontend http://localhost:8080
- Backend http://localhost:8000
- MLflow http://localhost:5001
- Prometheus http://localhost:9090
- Grafana http://localhost:3000 (admin / admin)

Compose uses PostgreSQL. Point `.env` `DATABASE_URL` at `postgresql+psycopg2://mlforge:mlforge@postgres:5432/mlforge` inside the backend container (already set in `docker-compose.yml`).

## Running Kubernetes / Minikube

```bash
minikube start
eval $(minikube docker-env)
docker build -f backend/Dockerfile -t mlforge-backend:local .
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/postgres.yaml
kubectl apply -f kubernetes/mlflow.yaml
kubectl apply -f kubernetes/backend-deployment.yaml
kubectl apply -f kubernetes/backend-service.yaml
kubectl apply -f kubernetes/model-deployment.yaml
kubectl apply -f kubernetes/model-service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl -n mlforge get pods
minikube service mlforge-backend -n mlforge
```

Manifests include ConfigMaps, Secrets, probes, and resource requests/limits. Images use `imagePullPolicy: IfNotPresent` so Minikube can use locally built images.

## Monitoring

`GET /metrics` exposes Prometheus metrics: request counts, failures, latency, training duration, prediction class distribution, production model version.

Grafana dashboard **MLForge Production Monitoring** is provisioned from `monitoring/grafana/dashboards/mlforge.json`.

## Drift detection

`GET /monitoring/drift` compares training reference data with recent logged inference features (tenure, charges, contract, etc.). Evidently is used when available; otherwise PSI (numeric) and Jensen–Shannon (categorical). Threshold is `drift.threshold` in `ml/configs/config.yaml`. High drift sets `retraining_required` but does **not** silently deploy a new model.

## Retraining

- Manual: `POST /train` with `X-API-Key`
- Drift: `POST /monitoring/retrain` — trains only if drift says retrain is required, then applies the same promotion gates

A worse candidate is rejected; production stays put.

## API documentation

| Method | Path | Auth |
| --- | --- | --- |
| GET | `/health` | no |
| GET | `/ready` | no |
| POST | `/predict` | no |
| POST | `/train` | API key |
| GET | `/models` | no |
| GET | `/models/{name}` | no |
| GET | `/models/{name}/versions` | no |
| POST | `/models/{name}/promote` | API key |
| GET | `/experiments` | no |
| GET | `/monitoring/summary` | no |
| GET | `/monitoring/drift` | no |
| POST | `/monitoring/retrain` | API key |
| GET | `/metrics` | no |

Example predict request:

```bash
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 900.0
  }'
```

Example response:

```json
{
  "prediction": 1,
  "churn_probability": 0.83,
  "model_name": "churn-model",
  "model_version": "1",
  "request_id": "c0a1e8c2-..."
}
```

Train:

```bash
curl -s http://localhost:8000/train \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme-admin-api-key' \
  -d '{"trigger":"manual","hyperparameters":{}}'
```

Promote staging → production (fails closed if F1 does not improve):

```bash
curl -s http://localhost:8000/models/churn-model/promote \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme-admin-api-key' \
  -d '{"version":"2","action":"promote"}'
```

## Testing

```bash
PYTHONPATH=. DATABASE_URL=sqlite:///./test_mlforge.db \
  MLFLOW_TRACKING_URI=file:./mlruns API_KEY=test-key pytest -q
```

Or `make test`. Coverage includes health, prediction validation/shape, promotion gates, preprocessing, splits, metrics, drift, and inference payloads.

## CI/CD

`.github/workflows/ci.yml` on push/PR: install Python deps, lint, pytest, build frontend, build backend and frontend Docker images. No paid cloud services.

## Why these pieces exist (interview notes)

- **MLOps** turns a trained artifact into a governed production service: versions, gates, monitoring, rollback.
- **MLflow tracking** records params, metrics, duration, dataset hash, and artifacts per run.
- **Registry + versioning** give `churn-model` v1/v2/… with status tags instead of a loose pickle file.
- **Promotion** is `candidate → staging → production`. A worse F1 (or failed ROC-AUC/F1 thresholds) is **rejected**; production is not replaced.
- **Serving** always loads the production alias/stage. Rollback is promoting a previous version back to production (and archiving the bad one).
- **FastAPI** gives typed contracts, OpenAPI, and cheap async I/O around CPU-bound sklearn inference.
- **Docker** makes the same API/MLflow/metrics stack runnable anywhere; **Kubernetes** adds replicas, probes, and a separate serving deployment you can scale independently.
- **Prometheus / Grafana** watch volume, latency, errors, prediction mix, and current model version.
- **Drift** is a change in live feature distributions vs training. Retrain when drift is high *and* a new candidate beats production.
- **Scale**: more serving replicas behind the Service; for thousands of RPS add a dedicated model-server pool, batch or async inference, and connection pooling. Canary would split traffic between two Deployments by version before full promotion.

## Future improvements

- Alembic migrations and a real feature store
- Canary / shadow traffic and automated rollback on latency or error SLO breach
- Async training workers (Celery / Ray) instead of in-process `/train`
- Multi-model registry beyond churn
- AuthN/Z beyond a single admin API key
- Cloud blob artifact store for MLflow

## GitHub setup

```bash
git init
git add .
git commit -m "Add MLForge MLOps platform"
gh repo create mlforge --private --source . --remote origin --push
```

## Interview questions this project supports

- What problem does the platform solve, and why is MLOps necessary?
- How does experiment tracking work in MLflow here?
- Why a model registry instead of saving a pickle?
- How are versions and statuses represented?
- Walk through promotion gates. What if the new model is worse?
- How does `/predict` choose a model version?
- Why FastAPI, Docker, and Kubernetes?
- What does Prometheus scrape? What does Grafana show?
- What is data drift, how is it detected, when do we retrain?
- How does CI fit an ML repo?
- How would you scale inference, run a canary, or roll back?

## Resume bullets (implemented)

- Built MLForge, an end-to-end MLOps platform that trains churn models (LR, Random Forest, XGBoost), logs runs to MLflow, and registers `churn-model` with versioned promotion gates (F1 + ROC-AUC vs production).
- Shipped a FastAPI serving layer that loads the current production model, returns calibrated churn probabilities with request IDs, and persists non-PII prediction telemetry for monitoring.
- Added drift detection (Evidently with PSI/JS fallback), Prometheus metrics plus a Grafana dashboard, Docker Compose / Minikube manifests, and GitHub Actions for tests and image builds.
