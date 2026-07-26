# Developer tasks for the scale-aware joint compression study.
#
# Nothing here downloads models, trains, or benchmarks -- those are explicit
# script invocations (see README.md).

PYTHON ?= python
PKG    := src/scale_aware_compression
PATHS  := src scripts tests

.DEFAULT_GOAL := help
.PHONY: help install install-dev test test-cov lint format format-check type-check check clean

help: ## Show the available targets
	@$(PYTHON) -c "import re,sys; [print(f'  {m.group(1):<14} {m.group(2)}') for line in open('Makefile', encoding='utf-8') if (m := re.match(r'^([a-zA-Z][a-zA-Z0-9_-]*):.*?## (.*)$$', line))]"

install: ## Install the package and its runtime dependencies (editable)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev: ## Install runtime + development dependencies and the git hooks
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e . -r requirements-dev.txt
	$(PYTHON) -m pre_commit install

test: ## Run the fast test suite (no downloads, no training)
	$(PYTHON) -m pytest

test-cov: ## Run the test suite with a coverage report
	$(PYTHON) -m pytest --cov=scale_aware_compression --cov-report=term-missing

lint: ## Lint with ruff
	$(PYTHON) -m ruff check $(PATHS)

format: ## Format with ruff (black-compatible) and apply safe import fixes
	$(PYTHON) -m ruff format $(PATHS)
	$(PYTHON) -m ruff check --fix $(PATHS)

format-check: ## Verify formatting without writing changes
	$(PYTHON) -m ruff format --check $(PATHS)

type-check: ## Static type check with mypy
	$(PYTHON) -m mypy

check: lint format-check type-check test ## Everything CI runs

clean: ## Remove caches and build artefacts (never touches data/ or outputs/)
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + list(pathlib.Path('.').rglob('*.egg-info')) + [pathlib.Path(d) for d in ('.pytest_cache', '.ruff_cache', '.mypy_cache', 'build', 'dist', 'htmlcov')]]"
	$(PYTHON) -c "import pathlib; [p.unlink(missing_ok=True) for p in list(pathlib.Path('.').rglob('*.py[co]')) + [pathlib.Path('.coverage'), pathlib.Path('coverage.xml')]]"
