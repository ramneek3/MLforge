PYTHON ?= python
PIP ?= pip

.PHONY: setup ingest train test backend frontend compose lint

setup:
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm install

ingest:
	PYTHONPATH=. $(PYTHON) -m ml.src.ingest

train:
	PYTHONPATH=. MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-file:./mlruns} $(PYTHON) -m ml.src.train

test:
	PYTHONPATH=. DATABASE_URL=sqlite:///./test_mlforge.db MLFLOW_TRACKING_URI=file:./mlruns API_KEY=test-key pytest -q

lint:
	ruff check ml backend tests

backend:
	PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

compose:
	docker compose up --build

mlflow:
	mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
