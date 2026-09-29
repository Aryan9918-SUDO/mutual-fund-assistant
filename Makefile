.PHONY: install ingest test cov lint typecheck check eval run api docker-build docker-run compose clean

install:        ## Install the package + dev deps
	pip install -e ".[dev]"

ingest:         ## Build & persist the FAISS vector index
	python -m mf_assistant.ingest

test:           ## Run the test suite
	pytest -q

cov:            ## Run tests with coverage report
	pytest --cov --cov-report=term-missing

lint:           ## Lint with ruff
	ruff check src tests eval app.py scripts

typecheck:      ## Static type-check with mypy
	mypy

check: lint typecheck cov eval  ## Run all quality gates (what CI runs)

eval:           ## Run the evaluation harness (accuracy report)
	python eval/run_eval.py

run:            ## Launch the Streamlit UI
	streamlit run app.py

api:            ## Launch the FastAPI backend (docs at /docs)
	uvicorn mf_assistant.api:app --reload --port 8000

docker-build:   ## Build the Docker image
	docker build -t mf-assistant .

docker-run:     ## Run the UI container
	docker run -p 8501:8501 -e GEMINI_API_KEY=$${GEMINI_API_KEY} mf-assistant

compose:        ## Run API + UI together via docker-compose
	docker compose up --build

clean:          ## Remove caches and build artifacts
	rm -rf .pytest_cache **/__pycache__ *.egg-info src/*.egg-info logs .coverage
