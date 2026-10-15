# Payment Operations Agent Architecture

The reference product is scoped to an internet payment platform. Its tenants
represent merchant organisations, and the knowledge bases contain integration,
refund, settlement, reconciliation, and incident-response runbooks.

```text
Client
  │ REST / WebSocket
  ▼
FastAPI routes ── auth adapter ── tenant-scoped services
  │                         ├── PostgreSQL + pgvector
  │                         ├── Redis event stream / Celery broker
  │                         └── LangGraph agent
  ▼
Agent graph: retrieve → answer OR request approval → create_ticket
```

## Boundaries

- Routes own transport concerns and never query another organisation's rows.
- Services own workflow transitions. The `ApprovalRequest` row is the durable
  pause point; approving it is the only path that can create a ticket.
- Ingestion is idempotent on the extracted text hash and embedding/chunk configuration.
  Unchanged content/configuration is a no-op; changes replace that document's chunks.
- `DocumentChunk.embedding` uses `pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS)`
  (default 1536) when the
  dependency is present. Default SQLite/deterministic retrieval uses FTS5/BM25
  with tenant/base predicates and trigger-maintained English/CJK bigram indexes;
  existing chunks are backfilled atomically at startup. The old trigram index
  is removed after replacement; source chunks are preserved. Offline no-hit
  queries return no evidence instead of scanning hash vectors. Real embeddings
  merge independent lexical and dense candidate lists using reciprocal rank
  fusion. PostgreSQL uses HNSW for dense candidates and a GIN text index for
  lexical candidates; SQLite real-vector scoring still scans JSON vectors.
  Optional local semantic embeddings and Cross-Encoder reranking require the
  `retrieval` extra. Each branch defaults to 50 candidates; optional reranking
  operates on the fused 50 before returning five. BM25-only reranking is also
  supported. Vector identity guards exclude old or incompatible model vectors.
- Agent idempotency commits a pending reservation before graph execution, then
  atomically commits business rows and the response. Duplicates poll without
  holding a connection, with a bounded 30-second wait/409; failures free the key.
  Hard-crash recovery of pending reservations is manual, not automatic takeover.
  Ticket actions and approval execution remain single atomic transactions.
  Authorization is committed before graph execution, retrieval owns a short
  session, and model waits hold no connection. Final run/approval/usage/response
  writes remain atomic. SQLite uses WAL without reducing synchronous durability.
  Allowed GET audits are saved after the read transaction ends.
- Events carry `run_id` and `trace_id`; Redis Streams can be consumed by a UI or
  an audit pipeline. The Redis client is process-scoped and backed by a
  connection pool. Redis failures back off for 30 seconds before probing again;
  local buffers retain at most 256 runs / 1,000 events per run, expiring after
  an hour on subsequent access. Redis streams have the same TTL. Delivery is
  best-effort; an outbox is still needed for guaranteed delivery.
- The LangGraph graph and model client are compiled once per API worker and
  invoked asynchronously, so model calls do not block the FastAPI event loop.
  Synchronous embedding/scoring and Celery publication run in threads; broker
  connect/socket timeouts are bounded. SSE uses a bounded queue, closes its
  producer on ASGI disconnect/send failure, and releases unfinished reservations.
  Document lists are paginated; usage totals are aggregated in SQL; citations
  load IDs and source metadata without deserializing chunk embeddings.

## Production hardening backlog

1. Add Alembic migrations and row-level security; validate PostgreSQL query
   plans, tenant-filtered HNSW recall, and index installation on production data.
2. Store uploads in S3/MinIO instead of passing bytes through Celery; add
   MIME/antivirus checks and chunk-level embedding reuse for large document updates.
3. Add a transactional outbox for guaranteed event delivery and operational
   recovery for crashed pending idempotency reservations.
4. Evaluate real embedding/LLM providers and Chinese retrieval on an independent
   labelled holdout set; add regression gates and long mixed-load tests.
