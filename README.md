# Payment Operations Knowledge & Ticket Agent

This is a focused, runnable reference application built on top of the
`full-stack-ai-agent-template` fork. It turns the generator's capabilities into
one bounded product for an internet payment platform: a multi-tenant knowledge
base agent that helps merchant support and operations teams answer integration,
refund, settlement, and reconciliation questions, then create incident tickets
through an approval gate.

## What is implemented

- Multi-tenant payment organisations, memberships, roles, and organisation-scoped queries.
- Async document ingestion for PDF, HTML, Markdown, and text sources.
- Content-hash based incremental indexing with chunk metadata and citations for
  merchant integration, refund, settlement, incident response, API security,
  webhook reliability, tenant RBAC, RAG evaluation, Celery ingestion,
  observability, and database migration runbooks.
- PostgreSQL storage with a `pgvector`-compatible embedding column; local mode
  uses deterministic embeddings so the project works without an LLM key.
- LangGraph workflow with retrieval and ticket tools. Ticket creation is a
  high-risk action and pauses in `waiting_approval` until a reviewer decides.
- WebSocket event streaming, trace IDs, token/cost accounting, and optional
  Redis event publication.
- Celery worker tasks for ingestion, with a local fallback for development.
- A small evaluation set and runner for recall, answer correctness, and tool
  success rate.
- Docker Compose for API, PostgreSQL/pgvector, Redis, and Celery.

## Quick start

```bash
cd enterprise_knowledge_agent
cp .env.example .env
docker compose up --build
```

The operator workspace is available at `http://localhost:8080/`; Swagger remains
available at `http://localhost:8080/docs`. New workspaces include a tenant-scoped
engineering corpus automatically. Existing empty workspaces can click
`Prepare demo runbooks` in the Agent workspace. The suggested prompts cover
refund timing, 401 integration errors, and the approval-gated ticket flow.

The Agent workspace renders grounded answers as a conclusion, key points,
ordered handling steps, and citation chips. Operational identifiers and risk
signals such as `401`, `429`, `refund ID`, `trace ID`, `organization_id`,
`content_hash`, `Recall@5`, and `p95` are highlighted in the answer body while
the source text remains HTML-escaped.

For a local browser-only run without Docker or PostgreSQL, use SQLite instead:

```bash
make run-local
```

Then open `http://localhost:8080/`. Keep the terminal open while using the
workspace; closing it stops the server.

To populate a demo tenant — four users, the runbook corpus, historical agent
runs, two pending approvals, tickets across several lifecycle states, and an
audit trail — run:

```bash
make seed-demo
```

This is idempotent, so re-running it reconciles rather than duplicating. Sign in
at `http://localhost:8080/` with any of:

| Email | Role | What it demonstrates |
|---|---|---|
| `ops@acme-payments.com` | owner | full access; the only role that can approve a `critical` request |
| `risk@acme-payments.com` | admin | second approver for the dual-approval flow |
| `support@acme-payments.com` | member | can ask questions and raise approvals, cannot decide them |
| `auditor@acme-payments.com` | viewer | read-only; the ops stat endpoints return 403 by design |

The password for all four is `demo-password-2026`.

If the database predates the current models, seeding stops with a list of the
missing columns rather than failing mid-write. `make seed-demo-reset` backs the
file up to `agent.db.<timestamp>.bak` and rebuilds the schema first.

For the offline core smoke test (after installing the test extras):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

Identity is a signed JWT: `POST /v1/auth/login` (or `/v1/auth/register`) returns
an access/refresh pair, and requests carry `Authorization: Bearer <access>` plus
`X-Org-ID` to select the tenant. The `X-User-ID` header is only honoured when
`ALLOW_HEADER_IDENTITY=true`, which is intended solely for a single-tenant local
sandbox and is off by default.

## Example flow

1. Create a payment operations knowledge base with `POST /v1/knowledge-bases`.
2. Upload a document with `POST /v1/documents` (`multipart/form-data`).
3. Ask the agent with `POST /v1/agent/runs`.
4. If the answer requires a ticket, inspect `GET /v1/approvals` and decide with
   `POST /v1/approvals/{approval_id}/decision`.
5. Connect to `/v1/ws/agent?org_id=<id>&user_id=<id>` to observe run events.

See [docs/architecture.md](docs/architecture.md) for boundaries, trade-offs,
and the production hardening backlog.

## Offline benchmark

The repository includes a reproducible benchmark for the local offline core. It
generates 240 synthetic documents and reports retrieval Recall@K/MRR, chunking
throughput, hash-based incremental indexing, approval policy precision/recall,
and in-process concurrent retrieval latency:

```bash
python evals/benchmark.py
```

The command writes `evals/results.json` and `evals/benchmark-report.md`. The
report labels the machine, dataset size, and limits explicitly: these are local
offline measurements and are not production API, database, Redis, Celery, or
LLM latency/SLA claims.

The current A/B result for the local fallback is included in the report:
batch vector scoring plus Top-K selection reduced p95 from 46.736 ms to 32.809
ms (29.80%) and increased measured throughput from 21.48 to 31.07 requests/s
(44.62%). Hash-based incremental indexing reduced rewrite work by 90% versus a
full rebuild for the update batch.

When Docker Desktop and network access are available, live PostgreSQL/pgvector,
Redis, Celery worker, network, and real OpenAI latency can be checked separately:

```bash
docker compose up -d postgres redis
celery -A app.worker.celery_app.celery_app worker --loglevel=INFO
DATABASE_URL=postgresql+asyncpg://agent:agent@localhost:5432/agent \
REDIS_URL=redis://localhost:6379/0 \
CELERY_BROKER_URL=redis://localhost:6379/1 \
CELERY_RESULT_BACKEND_URL=redis://localhost:6379/1 \
python evals/integration_benchmark.py --strict
```

The live probe writes `evals/integration-results.json` and
`evals/integration-report.md`. It never uses the offline fallback for a live
probe; missing configuration is reported as `skipped` and unreachable services
as `failed`.

For an OpenAI-compatible gateway, set `OPENAI_BASE_URL` alongside a key issued
by that gateway. Gateways exposing the Responses API can set
`OPENAI_API_MODE=responses`; the default is `chat_completions`. For reasoning
models, `OPENAI_REASONING_EFFORT=low|medium|high` can reduce or increase
inference time. The Codex CLI provider configuration is separate from this
application's OpenAI SDK credentials.

To use DeepSeek's OpenAI-compatible API, set `LLM_PROVIDER=deepseek`, provide
`DEEPSEEK_API_KEY`, and keep the default `DEEPSEEK_BASE_URL=https://api.deepseek.com`.
The benchmark defaults to `deepseek-chat`; use `DEEPSEEK_MODEL=deepseek-reasoner`
when reasoning quality is more important than latency.
