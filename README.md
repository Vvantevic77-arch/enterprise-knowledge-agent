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
- SQLite FTS5/BM25 retrieval with trigger-maintained English and Chinese bigram
  indexes and a one-time backfill; short Chinese queries do not scan JSON vectors.
- Short-transaction agent idempotency: concurrent same-key requests share one
  execution; retrieval uses a separate session and model waits hold no database connection.
- LangGraph workflow with retrieval and ticket tools. Ticket creation is a
  high-risk action and pauses in `waiting_approval` until a reviewer decides.
- SSE token streaming in the operator UI, authenticated WebSocket event replay,
  trace IDs, estimated token/cost accounting, and bounded Redis/local event retention.
- Celery worker tasks for ingestion, with a local fallback for development.
- A small evaluation set and runner for recall, answer correctness, and tool
  success rate.
- A public MS MARCO retrieval benchmark with 81,039 unique passages, labelled
  queries, reproducible sampling, and corpus-size latency measurements.
- Docker Compose for API, PostgreSQL/pgvector, Redis, and Celery.

The [priority performance improvements](docs/performance-improvements.md)
record the preceding runtime changes, regression checks and a matching before/after
HTTP experiment. Measurements below describe their recorded versions.

[Recall improvements](docs/recall-improvements.md) add candidate-recall diagnostics,
optional local semantic embeddings and Cross-Encoder reranking, with frozen
full-corpus ablations. Model identity guards prevent mixing incompatible vectors;
local models require the `retrieval` extra and remain opt-in.

The UI now consumes `POST /v1/agent/runs/stream`. With a model configured it
renders tokens as they arrive; offline mode returns a complete grounded answer.
The stream emits `answer.delta`, optional `answer.reset`, then `run.result` or
`run.error`. Errors after headers are sent carry their status inside the event.
Workspace changes cancel the old stream. `POST /v1/agent/runs` remains available
for ordinary JSON clients. Document lists now default to 50 rows and accept
`limit` (1–200) and `skip`.

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

The historical in-memory A/B result for the local fallback is included in the report:
batch vector scoring plus Top-K selection reduced p95 from 79.000 ms to 49.748
ms (37.03%) and increased measured throughput from 12.72 to 20.17 requests/s
(58.62%). Hash-based incremental indexing reduced rewrite work by 90% versus a
full rebuild for the update batch.

## Public-data benchmark

The 2026-10-12 experiment uses the Microsoft MS MARCO v1.1 validation QA split:
**10,047 real queries**, **81,039 unique web passages** (34.29 MB of text), and
**81,115 chunks**. This is 337.66 times the synthetic document count. Data stays
outside the repository; the script verifies its SHA-256 checksum.

```bash
pip install -e '.[test,benchmark]'
# Download the pinned Parquet file using the command in the report below.
python evals/public_benchmark.py --queries 1000 --latency-queries 60 --repeats 3
```

| Measurement | Observed result | Conditions |
| --- | ---: | --- |
| Local index construction | 17.39 s | Chunking, deterministic vectors and lexical postings; excludes DB writes |
| Historical pooled hybrid Recall@5 | 28.02% | 1,000 sampled labelled queries over all 81,039 passages |
| Cached retrieval p95 | 21.563 ms | 180 serial requests, 3 rounds; full scan of 81,115 chunks |

Pure lexical Recall@5 was 28.43%; the non-semantic hash vectors did
**not improve Recall@5**. Candidate-only ranking over 9,706 labelled queries
reached 74.68%, but that is a different and easier task. The QA split has
incomplete relevance labels outside its per-query candidates; this is not the
official 8.8-million-passage ranking benchmark.

This historical benchmark reuses the vector-fallback scoring weights and primitives with a cached
NumPy matrix and lexical postings. Its latency excludes HTTP, database reads,
external models and production candidate selection, so it is not an API SLA
or a concurrent-user capacity claim. See the [public-data report](evals/public-benchmark-report.md)
for the pinned download URL, experimental details, limits and reproduction;
[raw results](evals/public-benchmark-results.json) include the sampled query IDs
and rankings. [Local API results](evals/api-benchmark-report.md) are a separate
in-process SQLite experiment and must retain that qualification.

### Retrieval diagnosis

Controlled ablations on the same passages and 1,000 queries reproduced every
baseline ranking. Recall@5 was **44.10%** after removing common function words,
**51.99%** with IDF weighting, and **55.57%** with fixed-parameter BM25,
compared with **28.43%** for the previous overlap formula. BM25's paired
Recall@5 difference was **+27.14 percentage points** (95% bootstrap interval
**+24.05 to +30.17**). These are diagnostic reference results, not measurements
of the current API retrieval path; see the separate actual-service regression below.

The previous overlap scorer ignores term rarity, frequency and passage length; exact
tokens also miss inflections and paraphrases. Hash vectors add no semantic
signal, and a vector-only Top-20 candidate gate can discard all labelled
positives before lexical reranking. The [diagnosis report](evals/retrieval-diagnostics-report.md)
includes tie counts, failed queries and label limitations.

```bash
python -m evals.retrieval_diagnostics
```

### Actual SQLite retrieval after optimization

The 2026-10-14 regression calls the actual `search_knowledge` service with
**81,039 documents / 81,115 chunks** and the same **1,000 query IDs and labels**.
The default SQLite/deterministic path now uses FTS5/BM25 instead of loading and
scoring every JSON vector. English stop words are filtered; indexes are updated
by database triggers and existing chunks are backfilled at startup.

| Method | Recall@5 | MRR@5 |
| --- | ---: | ---: |
| Historical lexical overlap | 28.43% | 0.1847 |
| Historical hash-vector hybrid | 28.02% | 0.1890 |
| Actual SQLite FTS5/BM25 service | **57.53%** | **0.3500** |

Recall@5 improved by **29.10 percentage points** versus lexical overlap
(paired bootstrap 95% interval **+25.78 to +32.37**), and by **29.52 percentage
points** versus the historical hybrid. This reuses the diagnostic validation
sample, not an independent held-out test. The actual service returns Top-5
chunks mapped/deduplicated to documents; the historical baseline ranked
documents. Labels are incomplete; these figures are not answer accuracy.
The quality-only isolated database omits embeddings: English FTS does not read
them. Chinese fallback, HTTP latency, real models and PostgreSQL are not part
of this experiment.

See the [actual-service report](evals/sqlite-search-report.md) and
[query rankings](evals/sqlite-search-results.json).

```bash
.venv/bin/python -m evals.sqlite_search_benchmark
```

### Transaction and retry behavior

Agent runs with an idempotency key commit a small pending reservation before
retrieval/model calls. Only its owner executes the graph; concurrent identical
requests wait for the response without retaining a database lock/connection.
The run, approval/usage rows and completed response are committed atomically.
Graph or final persistence failures roll back business rows and release the key
for retry. Ticket execution keeps its original single-transaction semantics.

A pending duplicate waits at most 30 seconds, then returns HTTP 409 with
`retry_after_seconds=1`; different payloads with the same key return 409.
Clients should retry pending conflicts with the same key and payload. A hard
process crash can leave a pending record: there is no automatic takeover,
because stealing a still-running request could duplicate work. After verifying
that its worker has stopped and no business result was committed, an operator
must recover that reservation. This is a known production-hardening limit.

## Local HTTP performance and capacity

The **historical pre-optimization** 2026-10-12 test ran a separate Uvicorn worker over real loopback TCP HTTP,
using an isolated SQLite database, JWT authentication, real database reads and
writes, and deterministic answers with external models disabled. The temporary
acceptance limits were p95 <= 1 s for reads, <= 2 s for offline RAG, and errors
<= 1% in every trial. Each level ran twice for at least 10 s / 30 requests;
the highest passing level was checked twice more for 30 s.

| Workload | Corpus | Concurrent requests | p95 | Successful requests/s | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `POST /v1/agent/runs` | 1,000 passages / 1,001 chunks | 1 | 400–403 ms | 2.69–2.70 | Passed both confirmation windows |
| `POST /v1/agent/runs` | 1,000 passages / 1,001 chunks | 2 | 2,492–2,919 ms | 2.65 | Failed; second trial had 2 HTTP 500s / 31 requests |
| `POST /v1/agent/runs` | 10,000 passages / 10,012 chunks | 1 | 3,774–3,835 ms | 0.27 | No errors, exceeded 2 s limit |
| `GET /v1/knowledge-bases` | 10,000 passages / 10,012 chunks | 32 | 419–460 ms | 207.71–231.88 | Passed both confirmation windows; 64 failed one trial |

The old RAG path reloaded all SQLite chunks and JSON vectors for every request and
held a write transaction across retrieval. Concurrent runs produced
`sqlite3.OperationalError: database is locked`. The confirmed request capacity
depends on workload and these limits: it is **not a maximum user count** or an
online capacity claim. Health checks only sampled concurrency 1 and 64.
The client and server shared the laptop; 81,039-passage API performance,
external models, PostgreSQL and long-running mixed traffic were not measured.

These results are preserved as the baseline and do not describe the optimized
FTS5/short-transaction code. See the [historical HTTP capacity report](evals/http-capacity-report.md) and
[per-request results](evals/http-capacity-results.json) for all levels, errors,
resource samples and limits. Existing application data was left intact.

The **optimized** 2026-10-14 run used the same corpus sizes, query prompts,
machine, single-worker loopback setup and provisional limits. Its isolated
SQLite database stored the same 1,536-dimensional vectors. The optimized
response check accepts 1–5 valid citations rather than requiring exactly five,
because FTS can legitimately return fewer matches; this changes the success
criterion, so the before/after figures are diagnostic rather than a controlled
A/B comparison.

| Workload | Corpus | Confirmed concurrent requests | Confirmation p95 | Successful requests/s | First failing level |
| --- | ---: | ---: | ---: | ---: | ---: |
| `POST /v1/agent/runs` | 1,000 passages / 1,001 chunks | **32** | 1,279–1,346 ms | 69.28–69.76 | 64; one trial exceeded 2 s |
| `POST /v1/agent/runs` | 10,000 passages / 10,012 chunks | **32** | 1,583–1,663 ms | 53.32–54.94 | 64; both trials exceeded 2 s |
| `GET /v1/knowledge-bases` | 10,000 passages / 10,012 chunks | **32** | 663–750 ms | 140.77–149.78 | 64; one trial exceeded 1 s |

At one concurrent request, optimized RAG p95 was **17.9–18.4 ms** for 1,000
passages and **30.1–35.6 ms** for 10,000; the historical measurements were
400–403 ms and 3,774–3,835 ms respectively. All optimized RAG trials and
confirmation windows had **0 request errors**; no SQLite lock errors appeared
in the test server logs. These are offline, deterministic-answer measurements,
not real-model response times. The confirmed 32 is a concurrent-request level
under the stated latency limit, not a maximum number of users; the boundary
between 32 and 64 was not measured.
Knowledge-base reads were slower than the historical baseline at the same
confirmed 32-request level: **141–150 vs 208–232 successful requests/s** and
**663–750 vs 419–460 ms p95**. This run does not isolate the cause of that
regression; it should be profiled before claiming an across-the-board API gain.

See the [optimized HTTP report](evals/http-optimized-report.md) and
[per-request results](evals/http-optimized-results.json). To reproduce without
replacing the historical baseline:

```bash
.venv/bin/python -m evals.http_capacity --sizes 1000 10000 --levels 1 2 4 8 16 32 64 --seconds 10 --repeats 2 --min-requests 30 --soak-seconds 30 --output evals/http-optimized-results.json
```

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
