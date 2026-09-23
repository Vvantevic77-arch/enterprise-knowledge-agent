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
- Ingestion is idempotent on the extracted text hash. Re-uploading unchanged
  content is a no-op; changed content replaces chunks for that document.
- `DocumentChunk.embedding` uses `pgvector.sqlalchemy.Vector(1536)` when the
  dependency is present. The local fallback stores JSON vectors and ranks in
  Python, which keeps tests deterministic and offline. Local scoring batches
  cosine products with NumPy and uses Top-K selection; PostgreSQL uses the HNSW
  cosine index for the coarse scan.
- Events carry `run_id` and `trace_id`; Redis Streams can be consumed by a UI or
  an audit pipeline. The Redis client is process-scoped and backed by a
  connection pool; the in-process stream is a development fallback.
- The LangGraph graph and model client are compiled once per API worker and
  invoked asynchronously, so model calls do not block the FastAPI event loop.

## Production hardening backlog

1. Replace the header identity adapter with the template's JWT/OAuth and issue
   short-lived access tokens with refresh rotation.
2. Add Alembic migrations, row-level security policies, and a real pgvector
   cosine index (`ivfflat` or `hnsw`) once the embedding model is fixed.
3. Store uploads in S3/MinIO instead of passing bytes through Celery, and add
   MIME/type/size validation plus antivirus scanning.
4. Add an outbox table for at-least-once event delivery and idempotency keys for
   tool calls.
5. Expand the eval set with labelled retrieval queries and regression gates in CI.
