"""Document ingestion with content-hash incremental indexing.

The worker, the demo seeder, and ``evals/benchmark.py`` all call
``ingest_document`` so the "-90% rewrite on update" claim is measured against
the same code path that serves production traffic rather than a simulation.

Incremental contract:

* Same extracted text as last time → **zero** chunk writes. The document is
  marked ``unchanged`` and its existing chunks are left alone.
* Changed text → old chunks for that document are deleted and the new chunks
  embedded. Chunks belonging to other documents are never touched, which is
  where the saving comes from: a 24-document update in a 240-document corpus
  rewrites 96 chunks instead of all 960.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentChunk, DocumentStatus, KnowledgeBase, utcnow
from app.services.rag import chunk_text, content_hash, embed_texts, extract_text

#: Default chunk geometry. Kept in one place so the benchmark and the worker
#: cannot drift; changing it invalidates the published Recall@5 / MRR figures.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    status: str  # indexed | unchanged | failed
    chunks: int
    chunks_rewritten: int
    reason: str = ""


def file_fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


async def ingest_document(
    db: AsyncSession,
    *,
    document_id: str,
    filename: str,
    raw: bytes | None = None,
    text: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> IngestResult:
    """Extract, chunk, embed, and store one document.

    Pass either ``raw`` bytes or already-extracted ``text``. The caller owns the
    transaction: this function flushes but never commits, so ingestion can be
    composed with other work atomically.
    """
    document = await db.get(Document, document_id)
    if document is None:
        raise ValueError(f"document {document_id} not found")

    try:
        if text is None:
            if raw is None:
                raise ValueError("either raw bytes or extracted text is required")
            text = extract_text(filename, raw)
    except Exception as exc:
        document.status = DocumentStatus.FAILED.value
        document.error = str(exc)
        document.updated_at = utcnow()
        await db.flush()
        return IngestResult(document_id, DocumentStatus.FAILED.value, 0, 0, reason=str(exc))

    digest = content_hash(text)

    # --- Incremental fast path -------------------------------------------
    if document.content_hash == digest and document.status == DocumentStatus.INDEXED.value:
        existing = await db.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        )
        return IngestResult(
            document_id,
            DocumentStatus.UNCHANGED.value,
            int(existing or 0),
            0,
            reason="content hash unchanged",
        )

    pieces = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    embeddings = embed_texts(pieces) if pieces else []

    # Replace only this document's chunks.
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    for ordinal, (piece, vector) in enumerate(zip(pieces, embeddings)):
        db.add(
            DocumentChunk(
                document_id=document_id,
                ordinal=ordinal,
                content=piece,
                embedding=vector,
                metadata_json={
                    "filename": filename,
                    "source_url": document.source_url,
                    "content_hash": digest,
                },
            )
        )

    document.content_hash = digest
    document.status = DocumentStatus.INDEXED.value
    document.error = None
    document.chunk_count = len(pieces)
    document.updated_at = utcnow()
    await db.flush()

    return IngestResult(
        document_id,
        DocumentStatus.INDEXED.value,
        len(pieces),
        len(pieces),
    )


async def ingest_bytes(
    db: AsyncSession,
    *,
    document_id: str,
    filename: str,
    raw: bytes,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> IngestResult:
    return await ingest_document(
        db,
        document_id=document_id,
        filename=filename,
        raw=raw,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def create_document(
    db: AsyncSession,
    *,
    org_id: str,
    kb_id: str,
    filename: str,
    source_url: str = "",
) -> Document:
    """Register a pending document. The caller enqueues or runs ingestion."""
    document = Document(
        org_id=org_id,
        kb_id=kb_id,
        filename=filename,
        source_url=source_url,
        content_hash="pending",
        status=DocumentStatus.PENDING.value,
    )
    db.add(document)
    await db.flush()
    return document


async def resolve_knowledge_base(
    db: AsyncSession, *, org_id: str, kb_id: str
) -> KnowledgeBase | None:
    """Tenant-scoped KB lookup. Returns None rather than leaking 403 vs 404."""
    return (
        await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == kb_id, KnowledgeBase.org_id == org_id
            )
        )
    ).scalar_one_or_none()


def decode_payload(raw_base64: str) -> bytes:
    return base64.b64decode(raw_base64)


def encode_payload(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "IngestResult",
    "create_document",
    "decode_payload",
    "encode_payload",
    "file_fingerprint",
    "ingest_bytes",
    "ingest_document",
    "resolve_knowledge_base",
]
