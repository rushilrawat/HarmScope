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

## gzip: restream the snapshot as .csv.gz (~1.4 GB). DuckDB reads gzip natively
## but cannot read a zip member, so this is the cheapest path to queryable.
gzip: $(VENV)
	$(PY) -m src.pipeline run --phase download --gzip

## extract: unzip to plain CSV (~9 GB). Only if you need the raw file itself;
## `make gzip` is enough for the pipeline.
extract: $(VENV)
	$(PY) -m src.pipeline run --phase download --extract

## clean: drop the database and caches, keep the raw snapshot and ground truth
clean:
	rm -f data/harmscope.duckdb data/harmscope.duckdb.wal
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
