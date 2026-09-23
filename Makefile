.PHONY: install test lint run run-local seed-demo worker eval up down

# Prefer the project's virtual-environment binaries so commands work even
# when the shell has not activated `.venv`.
UVICORN := $(if $(wildcard .venv/bin/uvicorn),.venv/bin/uvicorn,uvicorn)
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

install:
	python -m pip install -e '.[test]'

test:
	pytest

lint:
	ruff check backend tests evals
	ruff format --check backend tests evals

run:
	PYTHONPATH=backend $(UVICORN) app.main:app --reload --port 8080

# Local browser-only mode: no Docker or PostgreSQL required.
# Uses SQLite for the API and serves the operator UI from the same process.
run-local:
	DATABASE_URL=sqlite+aiosqlite:///./agent.db PYTHONPATH=backend $(UVICORN) app.main:app --reload --host 127.0.0.1 --port 8080

# Provision a demo tenant (users, runbooks, runs, approvals, tickets, audit).
# Idempotent — safe to re-run. Run `make run-local` first, or this creates the
# same SQLite file the server will open.
seed-demo:
	DATABASE_URL=sqlite+aiosqlite:///./agent.db PYTHONPATH=backend $(PYTHON) -m cli.seed_demo

# Same, but rebuilds the schema first when the database predates the current
# models. Backs the file up to agent.db.<timestamp>.bak before dropping tables.
seed-demo-reset:
	DATABASE_URL=sqlite+aiosqlite:///./agent.db PYTHONPATH=backend $(PYTHON) -m cli.seed_demo --reset

worker:
	PYTHONPATH=backend celery -A app.worker.celery_app.celery_app worker --loglevel=INFO

eval:
	python evals/run_eval.py

up:
	docker compose up --build

down:
	docker compose down
