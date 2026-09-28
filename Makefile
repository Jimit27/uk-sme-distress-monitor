# End-to-end pipeline. `make all` reproduces everything from the public data.
PY ?= python

.PHONY: install extract fetch-extract warehouse train figures publish score app test lint all

install:
	pip install -e ".[warehouse,app,dev]"

extract:            ## download + parse the raw Companies House data (~5 GB, slow)
	$(PY) -m smewatch.cli ingest-snapshot
	$(PY) -m smewatch.cli ingest-accounts --delete-zip
	$(PY) -m smewatch.cli export-extract

fetch-extract:      ## or: download the prebuilt extract from the GitHub release
	gh release download data-extract --dir data/extract --clobber

warehouse:          ## dbt: staging -> features -> labelled cohort + register marts, with data tests
	$(PY) -m smewatch.cli build

train:              ## fit + evaluate models, write reports/metrics.json
	$(PY) -m smewatch.cli train

figures:
	$(PY) scripts/make_figures.py

publish:            ## small files the dashboard reads
	$(PY) -m smewatch.cli publish

score:              ## score the newest daily filings (needs daily files in data/extract)
	$(PY) -m smewatch.cli ingest-daily --days 5
	cp data/processed/daily/accounts_*.parquet data/extract/
	$(PY) -m smewatch.cli build --select +fct_live_filings
	$(PY) -m smewatch.cli score

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check src tests app

all: fetch-extract warehouse train publish figures
