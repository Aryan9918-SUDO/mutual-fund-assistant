.PHONY: install ingest test eval run docker-build docker-run clean

install:        ## Install the package + dev deps
	pip install -e ".[dev]"

ingest:         ## Build & persist the FAISS vector index
	python -m mf_assistant.ingest

test:           ## Run the unit test suite
	pytest -q

eval:           ## Run the evaluation harness (accuracy report)
	python eval/run_eval.py

run:            ## Launch the Streamlit app
	streamlit run app.py

docker-build:   ## Build the Docker image
	docker build -t mf-assistant .

docker-run:     ## Run the container (pass GEMINI_API_KEY through)
	docker run -p 8501:8501 -e GEMINI_API_KEY=$${GEMINI_API_KEY} mf-assistant

clean:          ## Remove caches and build artifacts
	rm -rf .pytest_cache **/__pycache__ *.egg-info src/*.egg-info logs
