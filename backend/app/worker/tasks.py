"""Celery tasks for asynchronous document ingestion.

The API registers the document and enqueues; a worker does parsing, chunking,
and embedding. ``ingest_document`` is the same function the demo seeder and the
offline benchmark call, so the incremental-indexing behaviour measured by the
benchmark is the behaviour that runs in production.

Failure handling is explicit rather than blanket-retry: a document that failed
because its PDF is corrupt will fail identically on retry, so the error is
recorded on the row instead of burning the retry budget.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Document, DocumentStatus
from app.services.ingestion import decode_payload, ingest_bytes, ingest_document
from app.services.rag import extract_text
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

#: Extraction failures are the file's fault, not the infrastructure's.
_PERMANENT_ERRORS = (
    "PDF extraction failed",
    "not found",
)


def _is_permanent(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _PERMANENT_ERRORS)


@celery_app.task(name="app.worker.tasks.healthcheck")
def healthcheck_task() -> dict[str, str]:
    """Broker/result-backend round trip used by the integration probe."""
    return {"status": "ok"}


async def _mark_failed(document_id: str, error: str) -> None:
    async with SessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED.value
            document.error = error[:2000]
            await db.commit()


async def _ingest(document_id: str, filename: str, raw: bytes) -> dict[str, object]:
    """Run ingestion in its own session and return a JSON-safe summary."""
    async with SessionLocal() as db:
        result = await ingest_bytes(
            db, document_id=document_id, filename=filename, raw=raw
        )
        await db.commit()
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunks": result.chunks,
            "chunks_rewritten": result.chunks_rewritten,
        }


@celery_app.task(
    bind=True,
    name="app.worker.tasks.ingest_document_task",
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def ingest_document_task(self, document_id: str, filename: str, raw_base64: str) -> dict[str, object]:
    try:
        return asyncio.run(_ingest(document_id, filename, decode_payload(raw_base64)))
    except Exception as exc:
        if _is_permanent(exc):
            logger.warning("permanent ingestion failure for %s: %s", document_id, exc)
            asyncio.run(_mark_failed(document_id, str(exc)))
            return {"document_id": document_id, "status": "failed", "error": str(exc)}
        # Transient: mark failed for now, but let Celery retry the task.
        asyncio.run(_mark_failed(document_id, str(exc)))
        raise


@celery_app.task(name="app.worker.tasks.reindex_task")
def reindex_task(document_id: str) -> dict[str, object]:
    """Re-ingest a document from its stored source URL.

    Used by the ``rag reindex`` command. Bytes are not stored, so only
    URL-sourced documents can be replayed; uploads are re-uploaded by operators.
    """
    async def _run() -> dict[str, object]:
        async with SessionLocal() as db:
            document = await db.get(Document, document_id)
            if document is None:
                return {"document_id": document_id, "status": "missing"}
            if not document.source_url:
                return {"document_id": document_id, "status": "skipped", "reason": "no source url"}
            import httpx

            async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
                response = await client.get(document.source_url)
                response.raise_for_status()
                raw = response.content
            result = await ingest_bytes(
                db,
                document_id=document_id,
                filename=document.filename + ".html",
                raw=raw,
            )
            await db.commit()
            return {"document_id": document_id, "status": result.status, "chunks": result.chunks}

    return asyncio.run(_run())


async def document_chunk_count(document_id: str) -> int:
    from sqlalchemy import func

    from app.models import DocumentChunk

    async with SessionLocal() as db:
        count = await db.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        )
        return int(count or 0)


def extracted_text(filename: str, raw: bytes) -> str:
    """Re-export for tests that want to check parser behaviour directly."""
    return extract_text(filename, raw)
