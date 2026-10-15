SHELL := /bin/bash
PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
N ?= 5000
ENGINE ?= $(or $(ENRICH_ENGINE),spark)
# Spark 3.5 pandas UDFs need Java 17; prefer a local JDK 17 when one is installed.
JAVA17 := $(firstword $(wildcard /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/temurin-17-*))
ifneq ($(JAVA17),)
export JAVA_HOME := $(JAVA17)
export PATH := $(JAVA17)/bin:$(PATH)
endif

.PHONY: setup setup-lite data pipeline demo mlflow-ui search test lint clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

setup-lite: $(BIN)/python ## core + dev deps, no torch (what CI uses)
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt -e .

setup: setup-lite ## + CPU torch, sentence-transformers and model download (~90 MB), best effort
	$(BIN)/pip install -r requirements-model.txt || echo "model deps unavailable; HashingEmbedder will be used"
	$(BIN)/python -m feedback_nlp.embedder

data: ## generate N synthetic records + labelled hold-out
	$(BIN)/python scripts/generate_data.py --n $(N) --out data

pipeline: data ## clean -> enrich -> evaluate (logs to ./mlruns)
	$(BIN)/python -m feedback_nlp.pipeline --engine $(ENGINE)

demo: pipeline mlflow-ui

mlflow-ui:
	$(BIN)/mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

search: ## make search Q="dashboard is slow"
	$(BIN)/python -m feedback_nlp.semantic_search "$(Q)"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

clean:
	rm -rf output mlruns data/raw data/labelled spark-warehouse .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
