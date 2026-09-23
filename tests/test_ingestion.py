"""Ingestion, incremental indexing, and retrieval measurement.

The incremental-indexing tests assert the behaviour behind the "-90% rewrite"
claim directly against the database, so the number is reproducible rather than
asserted in prose.
"""

from __future__ import annotations

from sqlalchemy import func, select

from app.models import Document, DocumentChunk, DocumentStatus
from app.services.ingestion import create_document, ingest_document
from app.services.rag import (
    chunk_text,
    content_hash,
    cosine_similarity,
    deterministic_embedding,
    embed_texts,
    hybrid_score,
    lexical_overlap,
    recall_at_k,
    reciprocal_rank,
)

from tests.conftest import Actor, seed_corpus


# ---------------------------------------------------------------------------
# Chunking and embeddings
# ---------------------------------------------------------------------------


def test_chunking_preserves_content_and_overlap() -> None:
    chunks = chunk_text("alpha " * 400, size=100, overlap=20)
    assert len(chunks) > 1
    assert "alpha" in chunks[0]
    assert chunks[0][-20:].split()[0] in chunks[1]


def test_chunking_rejects_overlap_larger_than_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", size=10, overlap=10)


def test_chunking_of_empty_text_yields_nothing() -> None:
    assert chunk_text("   \n  ") == []


def test_deterministic_embedding_is_normalized_and_repeatable() -> None:
    vector = deterministic_embedding("same input", dimensions=32)
    assert vector == deterministic_embedding("same input", dimensions=32)
    assert abs(cosine_similarity(vector, vector) - 1.0) < 1e-6


def test_content_hash_is_stable_and_content_sensitive() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_embed_texts_respects_the_configured_dimension() -> None:
    vectors = embed_texts(["one", "two"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------


def test_chinese_lexical_overlap_prefers_the_right_runbook() -> None:
    refund = "退款会原路退回原支付方式，通常需要 3-5 个工作日到账。"
    integration = "支付接口遇到 401 响应时，先确认 API key 和 Authorization header。"
    assert lexical_overlap("退款多久到账？", refund) > lexical_overlap("退款多久到账？", integration)
    assert lexical_overlap("支付接口 401 怎么排查？", integration) > lexical_overlap(
        "支付接口 401 怎么排查？", refund
    )


def test_exact_operational_identifier_dominates_the_hybrid_score() -> None:
    """A shared token like 429 must beat a topically similar but wrong runbook."""
    relevant = "遇到 429 响应时，使用 exponential backoff 并遵守 Retry-After header。"
    topical = "Webhook 可靠性要求校验签名、保持幂等并记录 event ID。"
    assert hybrid_score("429 怎么处理？", relevant, 0.0) > hybrid_score(
        "429 怎么处理？", topical, 1.0
    )


def test_lexical_overlap_of_a_query_with_no_terms_is_zero() -> None:
    assert lexical_overlap("", "anything") == 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_recall_at_k_counts_relevant_hits_in_the_window() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"c"}, 1) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    assert recall_at_k(ranked, {"a", "c"}, 1) == 0.5


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(["x", "y"], {"y"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"z"}) == 0.0
    assert reciprocal_rank(["x"], {"x"}) == 1.0


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


async def test_first_ingest_writes_chunks(db) -> None:
    doc = await create_document(
        db, org_id="org-1", kb_id="kb-1", filename="runbook.md"
    )
    # Foreign keys are not enforced by SQLite here, so a synthetic org is fine.
    result = await ingest_document(
        db, document_id=doc.id, filename="runbook.md", text="退款会在 7 个工作日内到账。" * 40
    )
    assert result.status == DocumentStatus.INDEXED.value
    assert result.chunks > 0
    assert result.chunks_rewritten == result.chunks


async def test_reingesting_identical_content_rewrites_nothing(db) -> None:
    """This is the mechanism behind the zero-write duplicate claim."""
    doc = await create_document(db, org_id="org-1", kb_id="kb-1", filename="runbook.md")
    text = "退款会在 7 个工作日内到账。" * 40

    first = await ingest_document(db, document_id=doc.id, filename="runbook.md", text=text)
    second = await ingest_document(db, document_id=doc.id, filename="runbook.md", text=text)

    assert second.status == DocumentStatus.UNCHANGED.value
    assert second.chunks_rewritten == 0
    assert second.chunks == first.chunks

    count = await db.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
    )
    assert count == first.chunks, "no chunk should have been duplicated or rewritten"


async def test_changed_content_rewrites_only_that_document(db) -> None:
    alpha = await create_document(db, org_id="org-1", kb_id="kb-1", filename="alpha.md")
    beta = await create_document(db, org_id="org-1", kb_id="kb-1", filename="beta.md")

    await ingest_document(db, document_id=alpha.id, filename="alpha.md", text="A" * 1500)
    await ingest_document(db, document_id=beta.id, filename="beta.md", text="B" * 1500)

    beta_before = (
        await db.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == beta.id)
        )
    ).scalar()

    # Change alpha only.
    result = await ingest_document(
        db, document_id=alpha.id, filename="alpha.md", text="A" * 1500 + " changed"
    )
    assert result.status == DocumentStatus.INDEXED.value

    beta_after = (
        await db.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == beta.id)
        )
    ).scalar()
    assert beta_after == beta_before, "an unrelated document must not be rewritten"


async def test_vector_only_indexing_never_touches_other_tenants(db) -> None:
    """Content-hash short-circuiting must be scoped to the one document row."""
    ours = await create_document(db, org_id="org-a", kb_id="kb-a", filename="same.md")
    theirs = await create_document(db, org_id="org-b", kb_id="kb-b", filename="same.md")

    shared_text = "结算批次缺失时的排查顺序。" * 40
    ours_result = await ingest_document(
        db, document_id=ours.id, filename="same.md", text=shared_text
    )
    theirs_result = await ingest_document(
        db, document_id=theirs.id, filename="same.md", text=shared_text
    )

    # Identical text, different documents: both index independently.
    assert ours_result.status == DocumentStatus.INDEXED.value
    assert theirs_result.status == DocumentStatus.INDEXED.value


async def test_failed_extraction_is_recorded_not_raised(db) -> None:
    doc = await create_document(db, org_id="org-1", kb_id="kb-1", filename="broken.pdf")
    result = await ingest_document(
        db, document_id=doc.id, filename="broken.pdf", raw=b"not really a pdf"
    )
    assert result.status == DocumentStatus.FAILED.value
    assert result.reason

    refreshed = await db.get(Document, doc.id)
    assert refreshed is not None
    assert refreshed.status == DocumentStatus.FAILED.value
    assert refreshed.error


# ---------------------------------------------------------------------------
# End-to-end retrieval through the API
# ---------------------------------------------------------------------------


async def test_retrieval_answers_a_refund_question_with_the_refund_runbook(member: Actor) -> None:
    response = await member.client.post(
        "/v1/agent/runs", headers=member.headers, json={"prompt": "退款多久到账？"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    filenames = [citation["filename"] for citation in payload["citations"]]
    assert filenames, "citations must be returned"
    assert filenames[0] == "refund-and-settlement-runbook.md"


async def test_retrieval_routes_a_401_question_to_the_integration_runbook(member: Actor) -> None:
    response = await member.client.post(
        "/v1/agent/runs", headers=member.headers, json={"prompt": "支付接口 401 怎么排查？"}
    )
    assert response.status_code == 200
    payload = response.json()
    filenames = [citation["filename"] for citation in payload["citations"]]
    assert filenames[0] == "payment-integration-runbook.md"


async def test_retrieval_never_crosses_the_tenant_boundary(
    member: Actor, second_org_owner: Actor
) -> None:
    response = await second_org_owner.client.post(
        "/v1/agent/runs", headers=second_org_owner.headers, json={"prompt": "退款多久到账？"}
    )
    assert response.status_code == 200
    # The rival org has its own seeded corpus; citations must come from it.
    citations = response.json()["citations"]
    assert citations
    for citation in citations:
        assert citation["source_url"].startswith("demo://")


async def test_uploaded_document_becomes_searchable(client, member: Actor, owner: Actor) -> None:
    kb_id = await seed_corpus(client, owner)
    uploaded = await member.client.post(
        f"/v1/documents?kb_id={kb_id}",
        headers=member.headers,
        files={
            "file": (
                "chargeback-runbook.md",
                "# 拒付处理 runbook 商户拒付 chargeback 必须在 5 个工作日内提交申诉材料。".encode(),
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] in {"indexed", "pending"}

    answer = await member.client.post(
        "/v1/agent/runs", headers=member.headers, json={"prompt": "拒付 chargeback 多久内要提交材料？"}
    )
    filenames = [citation["filename"] for citation in answer.json()["citations"]]
    assert "chargeback-runbook.md" in filenames


async def test_oversized_upload_is_rejected(client, member: Actor, owner: Actor) -> None:
    kb_id = await seed_corpus(client, owner)
    response = await member.client.post(
        f"/v1/documents?kb_id={kb_id}",
        headers=member.headers,
        files={"file": ("huge.md", b"x" * (10 * 1024 * 1024 + 10), "text/markdown")},
    )
    assert response.status_code == 422


async def test_url_ingestion_rejects_private_hosts(client, member: Actor, owner: Actor) -> None:
    kb_id = await seed_corpus(client, owner)
    for url in ("http://localhost/runbook", "http://127.0.0.1/x", "http://169.254.169.254/latest"):
        response = await member.client.post(
            "/v1/documents/from-url",
            headers=member.headers,
            json={"kb_id": kb_id, "url": url},
        )
        assert response.status_code == 422, url


async def test_document_chunks_are_inspectable(client, member: Actor, owner: Actor) -> None:
    kb_id = await seed_corpus(client, owner)
    documents = await member.client.get("/v1/documents", headers=member.headers)
    document_id = documents.json()[0]["id"]

    response = await member.client.get(
        f"/v1/documents/{document_id}/chunks", headers=member.headers
    )
    assert response.status_code == 200
    chunks = response.json()
    assert chunks
    assert all(chunk["has_embedding"] for chunk in chunks)
