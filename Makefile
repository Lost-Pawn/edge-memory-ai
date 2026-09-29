.PHONY: help install install-llm dev download-models qdrant cloud edge edge-b lint type test fmt frontend clean

help:
	@echo "EdgeMemoryAi — make targets"
	@echo "  install         Install core + dev deps (no models downloaded)"
	@echo "  install-llm     Install llama-cpp-python (in-process LLM)"
	@echo "  download-models EXPLICIT model download (embeddings + GGUF LLM). Nothing else downloads."
	@echo "  qdrant          Run Qdrant Server (cloud brain) via docker"
	@echo "  cloud           Run the cloud sync service (:8000)"
	@echo "  edge            Run edge device A (:8001)"
	@echo "  edge-b          Run edge device B (:8002) for multi-device demo"
	@echo "  frontend        Run the React dashboard (:5173)"
	@echo "  test / lint / type / fmt   Quality gates"

install:
	pip install -e ".[dev]"

install-llm:
	pip install -e ".[llm]"

# Downloads happen ONLY here, on purpose. Runtime never fetches weights.
download-models:
	python -m edge_memory.cli download-models
	python -m edge_memory.cli download-llm

qdrant:
	docker run -p 6333:6333 -p 6334:6334 -v $(PWD)/data/qdrant:/qdrant/storage qdrant/qdrant:latest

cloud:
	uvicorn cloud_sync.api.app:app --host 0.0.0.0 --port 8000 --reload

edge:
	EDGE_DEVICE_ID=edge-A EDGE_DATA_DIR=./data/edge-A EDGE_API_PORT=8001 \
		uvicorn edge_memory.api.app:app --host 0.0.0.0 --port 8001 --reload

edge-b:
	EDGE_DEVICE_ID=edge-B EDGE_DATA_DIR=./data/edge-B EDGE_API_PORT=8002 \
		uvicorn edge_memory.api.app:app --host 0.0.0.0 --port 8002 --reload

frontend:
	cd frontend && npm install && npm run dev

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

type:
	mypy

test:
	pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
