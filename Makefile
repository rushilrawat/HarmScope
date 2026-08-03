.PHONY: init test lint fmt clean download

VENV := .venv
PY   := $(VENV)/bin/python

$(VENV):
	uv venv $(VENV)

## init: create the venv, install base+dev deps, build an empty schema-valid DB
init: $(VENV)
	uv pip install -q --python $(PY) -r requirements-dev.txt
	$(PY) -m src.pipeline run --phase init

## test: run the test suite
test: $(VENV)
	$(PY) -m pytest

## lint: ruff check
lint: $(VENV)
	$(VENV)/bin/ruff check src tests

## fmt: ruff format
fmt: $(VENV)
	$(VENV)/bin/ruff format src tests

## download: snapshot the CFPB bulk CSV, compressed (~1.4 GB). Not run by init.
download: $(VENV)
	$(PY) -m src.pipeline run --phase download

## extract: unzip the snapshot (~15 GB). Separate on purpose — the archive is
## the reproducibility anchor; the CSV is just what DuckDB can read.
extract: $(VENV)
	$(PY) -m src.pipeline run --phase download --extract

## clean: drop the database and caches, keep the raw snapshot and ground truth
clean:
	rm -f data/harmscope.duckdb data/harmscope.duckdb.wal
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
