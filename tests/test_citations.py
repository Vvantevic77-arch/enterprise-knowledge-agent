"""Stored citations retain retrieval order and tenant scope."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.models import AgentRun, Document, DocumentChunk
from app.routes.agent import _context_for
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .conftest import Actor


async def test_stored_citations_follow_recorded_retrieval_order() -> None:
    rows = [
        ("lower-ranked", "secondary.md", ""),
        ("best-match", "primary.md", "https://example.com/primary"),
    ]
    db = SimpleNamespace(execute=AsyncMock(return_value=Mock(all=lambda: rows)))
    run = AgentRun(
        org_id="our-org",
        context_chunk_ids=["best-match", "deleted-chunk", "lower-ranked"],
    )

    citations = await _context_for(db, run)

    assert [citation["chunk_id"] for citation in citations] == [
        "best-match",
        "lower-ranked",
    ]
    assert citations[0]["filename"] == "primary.md"
    assert citations[0]["source_url"] == "https://example.com/primary"


async def test_stored_citations_exclude_other_tenants(
    db: AsyncSession, owner: Actor, second_org_owner: Actor
) -> None:
    chunks = (
        await db.execute(
            select(DocumentChunk.id, Document.org_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .order_by(DocumentChunk.id)
        )
    ).all()
    own_id = next(chunk_id for chunk_id, org_id in chunks if org_id == owner.org_id)
    foreign_id = next(chunk_id for chunk_id, org_id in chunks if org_id == second_org_owner.org_id)
    run = AgentRun(org_id=owner.org_id, context_chunk_ids=[foreign_id, own_id])

    citations = await _context_for(db, run)

    assert [citation["chunk_id"] for citation in citations] == [own_id]
