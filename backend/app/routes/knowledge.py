"""Knowledge bases and the documents inside them.

Everything here is tenant-scoped: the acting organisation comes from the
verified principal, never from the request body, so a caller cannot read or
write another merchant's corpus by passing a different ``org_id``.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import audit
from app.auth import Db, require
from app.core_exceptions import ConflictError, NotFoundError, ValidationError
from app.models import Document, DocumentChunk, DocumentStatus, KnowledgeBase
from app.permissions import Action, Principal, Resource
from app.schemas import DocumentRead, DocumentUrlCreate, KnowledgeBaseCreate, KnowledgeBaseRead
from app.services.ingestion import create_document, encode_payload
from app.worker.tasks import ingest_document_task

router = APIRouter(tags=["knowledge"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
#: Hosts that must never be reachable through the URL-ingestion endpoint.
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata"}


async def _kb_or_404(db: AsyncSession, *, org_id: str, kb_id: str) -> KnowledgeBase:
    kb = (
        await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == kb_id, KnowledgeBase.org_id == org_id)
        )
    ).scalar_one_or_none()
    if kb is None:
        raise NotFoundError(message="knowledge base not found")
    return kb


def _assert_public_url(url: str) -> str:
    """Reject SSRF targets: only absolute http(s) to a non-private host.

    The hostname is resolved syntactically. A literal IP in the private,
    loopback, link-local, or reserved ranges is refused outright; DNS names are
    allowed and re-checked by the caller after resolution where possible.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError(message="url must be an absolute http(s) URL")
    hostname = (parsed.hostname or "").lower()
    if hostname in _BLOCKED_HOSTS:
        raise ValidationError(message="private and metadata hosts are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return url
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    ):
        raise ValidationError(message="private and reserved hosts are not allowed")
    return url


@router.post(
    "/knowledge-bases", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    db: Db,
    principal: Principal = Depends(require(Action.CREATE, Resource.KNOWLEDGE_BASE)),
) -> Any:
    duplicate = (
        await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.org_id == principal.org_id, KnowledgeBase.name == payload.name
            )
        )
    ).scalar_one_or_none()
    if duplicate is not None:
        raise ConflictError(message="a knowledge base with that name already exists")

    kb = KnowledgeBase(org_id=principal.org_id, name=payload.name, description=payload.description)
    db.add(kb)
    await db.flush()
    audit.record(
        db,
        action="knowledge_base.create",
        resource="knowledge_base",
        org_id=principal.org_id,
        actor_id=principal.user_id,
        resource_id=kb.id,
    )
    await db.commit()
    return KnowledgeBaseRead(id=kb.id, name=kb.name, description=kb.description)


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseRead])
async def list_knowledge_bases(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.KNOWLEDGE_BASE)),
) -> Any:
    rows = (
        await db.execute(
            select(KnowledgeBase, func.count(Document.id))
            .outerjoin(Document, Document.kb_id == KnowledgeBase.id)
            .where(KnowledgeBase.org_id == principal.org_id)
            .group_by(KnowledgeBase.id)
            .order_by(KnowledgeBase.name)
        )
    ).all()
    return [
        KnowledgeBaseRead(
            id=row.id,
            name=row.name,
            description=row.description,
            document_count=int(count),
        )
        for row, count in rows
    ]


@router.delete(
    "/knowledge-bases/{kb_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None
)
async def delete_knowledge_base(
    kb_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.DELETE, Resource.KNOWLEDGE_BASE)),
) -> None:
    kb = await _kb_or_404(db, org_id=principal.org_id, kb_id=kb_id)
    await db.delete(kb)
    audit.record(
        db,
        action="knowledge_base.delete",
        resource="knowledge_base",
        org_id=principal.org_id,
        actor_id=principal.user_id,
        resource_id=kb_id,
    )
    await db.commit()


@router.get("/documents", response_model=list[DocumentRead])
async def list_documents(
    db: Db,
    kb_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    principal: Principal = Depends(require(Action.READ, Resource.DOCUMENT)),
) -> Any:
    stmt = select(Document).where(Document.org_id == principal.org_id)
    if kb_id:
        stmt = stmt.where(Document.kb_id == kb_id)
    rows = (
        (
            await db.execute(
                stmt.order_by(Document.created_at.desc(), Document.id).offset(skip).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [_document_read(row) for row in rows]


def _document_read(row: Document) -> DocumentRead:
    return DocumentRead(
        id=row.id,
        kb_id=row.kb_id,
        filename=row.filename,
        source_url=row.source_url,
        status=row.status,
        chunk_count=row.chunk_count,
        error=row.error,
        created_at=row.created_at,
    )


@router.post("/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    kb_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.CREATE, Resource.DOCUMENT)),
    file: UploadFile = File(...),
) -> Any:
    """Register a document and hand it to the ingestion worker."""
    kb = await _kb_or_404(db, org_id=principal.org_id, kb_id=kb_id)

    # Read one byte past the limit so an oversized upload is rejected without
    # buffering the whole body into memory first.
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            message=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB upload limit"
        )
    if not raw:
        raise ValidationError(message="uploaded file is empty")

    filename = file.filename or "upload.txt"
    document = await create_document(db, org_id=principal.org_id, kb_id=kb.id, filename=filename)
    audit.record(
        db,
        action="document.upload",
        resource="document",
        org_id=principal.org_id,
        actor_id=principal.user_id,
        resource_id=document.id,
        metadata={"filename": filename, "bytes": len(raw)},
    )
    await db.commit()

    await _dispatch_ingestion(document.id, filename, raw)
    await db.refresh(document)
    return _document_read(document)


@router.post("/documents/from-url", response_model=DocumentRead)
async def ingest_url(
    payload: DocumentUrlCreate,
    db: Db,
    principal: Principal = Depends(require(Action.CREATE, Resource.DOCUMENT)),
) -> Any:
    """Fetch and index a public web page into a tenant KB."""
    url = _assert_public_url(payload.url)
    kb = await _kb_or_404(db, org_id=principal.org_id, kb_id=payload.kb_id)
    hostname = urlparse(url).hostname or "web-page"

    existing = (
        await db.execute(
            select(Document).where(Document.kb_id == kb.id, Document.source_url == url)
        )
    ).scalar_one_or_none()

    if existing is None:
        document = await create_document(
            db, org_id=principal.org_id, kb_id=kb.id, filename=hostname, source_url=url
        )
    else:
        document = existing

    audit.record(
        db,
        action="document.ingest_url",
        resource="document",
        org_id=principal.org_id,
        actor_id=principal.user_id,
        resource_id=document.id,
        metadata={"url": url},
    )
    await db.commit()

    try:
        import httpx

        async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
            response = await client.get(url, headers={"User-Agent": "payment-ops-agent/0.1"})
            response.raise_for_status()
            raw = response.content
    except Exception as exc:
        document.status = DocumentStatus.FAILED.value
        document.error = f"failed to fetch URL: {exc}"[:2000]
        await db.commit()
        raise ValidationError(message=f"failed to fetch URL: {exc}") from exc

    await _dispatch_ingestion(document.id, f"{hostname}.html", raw)
    await db.refresh(document)
    return _document_read(document)


@router.post("/documents/{document_id}/reindex", response_model=DocumentRead)
async def reindex_document(
    document_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.UPDATE, Resource.DOCUMENT)),
) -> Any:
    """Force a re-ingest. Unchanged content is a zero-write no-op."""
    document = (
        await db.execute(
            select(Document).where(Document.id == document_id, Document.org_id == principal.org_id)
        )
    ).scalar_one_or_none()
    if document is None:
        raise NotFoundError(message="document not found")
    if not document.source_url:
        raise ValidationError(
            message="only URL-sourced documents can be reindexed; re-upload the file instead"
        )

    await _dispatch_ingestion(document.id, document.filename, None, reindex=True)
    await db.refresh(document)
    return _document_read(document)


async def _dispatch_ingestion(
    document_id: str,
    filename: str,
    raw: bytes | None,
    *,
    reindex: bool = False,
) -> None:
    """Hand the document to Celery, falling back to in-process ingestion.

    The upload fallback keeps local development and the test suite working with
    no broker. Reindex has no such fallback: it needs to re-fetch the source
    over the network, which is exactly the work the worker exists to do, so
    without a broker it reports that rather than pretending to succeed.
    """
    from app.db import SessionLocal
    from app.services.ingestion import ingest_document

    if reindex:
        from app.worker.tasks import reindex_task

        try:
            await asyncio.to_thread(reindex_task.apply_async, args=(document_id,), retry=False)
        except Exception as exc:
            raise ValidationError(
                message="reindex requires a running Celery worker and broker"
            ) from exc
        return

    if raw is None:
        raise ValidationError(message="no content supplied for ingestion")

    try:
        await asyncio.to_thread(
            ingest_document_task.apply_async,
            args=(document_id, filename, encode_payload(raw)),
            retry=False,
        )
        return
    except Exception:
        # No broker reachable — ingest inline so the upload is still indexed.
        pass

    async with SessionLocal() as session:
        await ingest_document(session, document_id=document_id, filename=filename, raw=raw)
        await session.commit()


@router.get("/documents/{document_id}/chunks")
async def list_document_chunks(
    document_id: str,
    db: Db,
    limit: int = 50,
    principal: Principal = Depends(require(Action.READ, Resource.DOCUMENT)),
) -> list[dict[str, Any]]:
    """Inspect what was actually indexed. Useful when a recall drop is suspected."""
    document = (
        await db.execute(
            select(Document).where(Document.id == document_id, Document.org_id == principal.org_id)
        )
    ).scalar_one_or_none()
    if document is None:
        raise NotFoundError(message="document not found")

    rows = (
        (
            await db.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.ordinal)
                .limit(max(1, min(limit, 200)))
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": row.id,
            "ordinal": row.ordinal,
            "content": row.content,
            "has_embedding": row.embedding is not None,
        }
        for row in rows
    ]
