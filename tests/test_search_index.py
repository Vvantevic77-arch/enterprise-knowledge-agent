"""SQLite indexed retrieval must remain current and tenant-scoped."""

from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db import Base, init_sqlite_search
from app.models import Document, DocumentChunk
from app.services.agent import search_knowledge

from .conftest import Actor, seed_corpus


async def test_indexed_search_respects_tenant_and_base(
    db: AsyncSession, owner: Actor, second_org_owner: Actor
) -> None:
    own_kb = await seed_corpus(owner.client, owner)
    other_kb = await seed_corpus(second_org_owner.client, second_org_owner)
    documents = [
        Document(org_id=owner.org_id, kb_id=own_kb, filename="ours.txt",
                 content_hash="ours", status="indexed"),
        Document(org_id=second_org_owner.org_id, kb_id=other_kb, filename="theirs.txt",
                 content_hash="theirs", status="indexed"),
    ]
    db.add_all(documents)
    await db.flush()
    db.add_all([
        DocumentChunk(document_id=doc.id, ordinal=0, content="xylophonic settlement marker")
        for doc in documents
    ])
    await db.commit()

    results = await search_knowledge(db, owner.org_id, "xylophonic settlement", [own_kb])
    assert results[0]["filename"] == "ours.txt"
    assert all(item["filename"] != "theirs.txt" for item in results)
    assert await search_knowledge(db, owner.org_id, "xylophonic", [other_kb]) == []


async def test_index_tracks_updates_deletes_and_chinese(
    db: AsyncSession, owner: Actor
) -> None:
    kb_id = await seed_corpus(owner.client, owner)
    document = Document(org_id=owner.org_id, kb_id=kb_id, filename="runbook.txt",
                        content_hash="update", status="indexed")
    db.add(document)
    await db.flush()
    chunk = DocumentChunk(document_id=document.id, ordinal=0, content="结算批次缺失 xylophonic")
    db.add(chunk)
    await db.commit()

    initial = await search_knowledge(db, owner.org_id, "结算批次缺失", [kb_id])
    assert initial[0]["chunk_id"] == chunk.id
    chunk.content = "退款到账 falconry"
    await db.commit()
    assert await search_knowledge(db, owner.org_id, "xylophonic", [kb_id]) == []
    assert [row["chunk_id"] for row in await search_knowledge(
        db, owner.org_id, "falconry", [kb_id]
    )] == [chunk.id]
    await db.delete(chunk)
    await db.commit()
    assert await search_knowledge(db, owner.org_id, "falconry", [kb_id]) == []


async def test_existing_chunks_are_backfilled_when_index_is_installed(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(insert(DocumentChunk).values(
                id="legacy-chunk", document_id="legacy-doc", ordinal=0,
                content="legacy xylophonic content", metadata_json={},
            ))
            await init_sqlite_search(connection)
            found = await connection.scalar(text(
                "SELECT rowid FROM document_chunks_fts "
                "WHERE document_chunks_fts MATCH 'xylophonic'"
            ))
            assert found is not None
            await init_sqlite_search(connection)
            count = await connection.scalar(text(
                "SELECT count(*) FROM document_chunks_fts "
                "WHERE document_chunks_fts MATCH 'xylophonic'"
            ))
            assert count == 1
    finally:
        await engine.dispose()
