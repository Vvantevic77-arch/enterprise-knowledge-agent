"""Semantic retrieval configuration, tenant boundaries and evaluation integrity."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from app.config import Settings, settings
from app.models import Document, DocumentChunk
from app.services import rag
from app.services.agent import search_knowledge
from app.services.ingestion import create_document, ingest_document
from evals.public_benchmark import Query
from evals.semantic_retrieval import PLAN, candidate_diagnostics, freeze_manifest, summarize
from sqlalchemy import select


def test_candidate_config_is_bounded_and_models_are_optional():
    assert Settings(_env_file=None).reranker_model is None
    assert Settings(_env_file=None, reranker_model=" ", reranker_revision="").reranker_model is None
    with pytest.raises(ValueError, match="EMBEDDING_REVISION"):
        Settings(_env_file=None, embedding_provider="sentence_transformers", embedding_revision="")
    for value in (0, 201):
        with pytest.raises(ValueError):
            Settings(_env_file=None, retrieval_candidate_k=value)


def test_local_embeddings_are_normalized_and_dimension_checked(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
    monkeypatch.setattr(settings, "embedding_dimensions", 2)
    monkeypatch.setattr(
        rag,
        "_sentence_encoder",
        lambda *_args: SimpleNamespace(encode=lambda *_args, **_kwargs: np.asarray([[3.0, 4.0]])),
    )
    np.testing.assert_allclose(rag.embed_texts(["refund"]), [[0.6, 0.8]])
    with pytest.raises(ValueError, match="number"):
        rag.embed_texts(["refund", "invoice"])
    monkeypatch.setattr(settings, "embedding_dimensions", 3)
    with pytest.raises(ValueError, match="dimensions"):
        rag.embed_texts(["refund"])
    monkeypatch.setattr(settings, "embedding_dimensions", 2)
    with pytest.raises(ValueError, match="values"):
        rag._validate_embeddings([[float("nan"), 1.0]], 1)
    with pytest.raises(ValueError, match="zero"):
        rag._validate_embeddings([[0.0, 0.0]], 1)


async def test_model_change_reembeds_identical_text_and_bad_batches_retain_index(db, monkeypatch):
    document = await create_document(db, org_id="org", kb_id="kb", filename="same.txt")
    await ingest_document(db, document_id=document.id, filename="same.txt", text="same content")
    original = (await db.scalars(select(DocumentChunk))).one()
    old_id = original.id
    monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
    monkeypatch.setattr(settings, "embedding_model", "local-test-model")

    async def embed(texts):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    monkeypatch.setattr("app.services.ingestion.aembed_texts", embed)
    result = await ingest_document(
        db, document_id=document.id, filename="same.txt", text="same content"
    )
    assert result.chunks_rewritten == 1
    replacement = (await db.scalars(select(DocumentChunk))).one()
    assert replacement.id != old_id
    assert replacement.metadata_json["embedding_identity"] == rag.embedding_identity()
    assert (
        await ingest_document(db, document_id=document.id, filename="same.txt", text="same content")
    ).chunks_rewritten == 0

    async def broken(_texts):
        return []

    monkeypatch.setattr("app.services.ingestion.aembed_texts", broken)
    with pytest.raises(ValueError, match="count"):
        await ingest_document(db, document_id=document.id, filename="same.txt", text="changed text")
    assert (await db.scalars(select(DocumentChunk))).one().id == replacement.id


async def test_reranking_is_off_loop_and_never_sees_another_tenant(
    db, owner, second_org_owner, monkeypatch
):
    from .conftest import seed_corpus

    own_kb = await seed_corpus(owner.client, owner)
    other_kb = await seed_corpus(second_org_owner.client, second_org_owner)
    for actor, kb_id, content in (
        (owner, own_kb, "refund intended relevant passage"),
        (second_org_owner, other_kb, "refund PRIVATE document"),
    ):
        doc = Document(
            org_id=actor.org_id,
            kb_id=kb_id,
            filename="candidate.txt",
            content_hash="x",
            status="indexed",
        )
        db.add(doc)
        await db.flush()
        db.add(DocumentChunk(document_id=doc.id, ordinal=0, content=content))
    await db.commit()
    main_thread = threading.get_ident()
    observed = []

    def predict(pairs, **_kwargs):
        assert threading.get_ident() != main_thread
        observed.extend(content for _query, content in pairs)
        return [float("intended" in content) for _query, content in pairs]

    monkeypatch.setattr(settings, "reranker_model", "fake-reranker")
    monkeypatch.setattr(rag, "_cross_encoder", lambda *_args: SimpleNamespace(predict=predict))
    result = await search_knowledge(db, owner.org_id, "refund", [own_kb], top_k=1)
    assert result[0]["content"] == "refund intended relevant passage"
    assert observed and all("PRIVATE" not in content for content in observed)
    assert await search_knowledge(db, owner.org_id, "refund", [other_kb]) == []
    await asyncio.sleep(0)


async def test_vectors_from_a_different_model_are_excluded(db, owner, monkeypatch):
    from .conftest import seed_corpus

    kb_id = await seed_corpus(owner.client, owner)
    doc = Document(
        org_id=owner.org_id,
        kb_id=kb_id,
        filename="old-vector.txt",
        content_hash="old",
        status="indexed",
    )
    db.add(doc)
    await db.flush()
    db.add(
        DocumentChunk(
            document_id=doc.id,
            ordinal=0,
            content="unrelated old vector",
            embedding=[1.0] + [0.0] * (settings.embedding_dimensions - 1),
            metadata_json={"embedding_identity": "wrong-model"},
        )
    )
    await db.commit()

    async def embed(_texts):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1)]

    monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
    monkeypatch.setattr(rag, "aembed_texts", embed)
    assert await search_knowledge(db, owner.org_id, "zxqvnonexistent", [kb_id]) == []


def test_rrf_keeps_lexical_only_candidates_and_does_not_count_duplicates():
    a, b, c = [{"chunk_id": key, "content": key} for key in "abc"]
    rankings = rag.fuse_rankings([[a, b, a], [c, b]], 3)
    assert [row["chunk_id"] for row in rankings] == ["b", "a", "c"]
    assert rankings[1]["score"] == 1 / 61
    assert "score" not in a


def test_manifest_freezes_disjoint_queries_and_detects_changes(tmp_path):
    queries = [Query(i, str(i), (i,), frozenset({i})) for i in range(PLAN["holdout_queries"] + 2)]
    path = tmp_path / "manifest.json"
    baseline = {"pooled_rankings": [{"query_id": 0}]}
    frozen = freeze_manifest(queries, baseline, path, "checksum")
    assert not set(frozen["holdout_query_ids"]) & set(frozen["diagnostic_query_ids"])
    assert freeze_manifest(list(reversed(queries)), baseline, path, "checksum") == frozen
    with pytest.raises(ValueError, match="changed"):
        freeze_manifest(queries, baseline, path, "different checksum")


def test_diagnostics_distinguish_candidate_miss_from_ranking_miss():
    records = [
        {"relevant": [7], "bm25": list(range(10))},
        {"relevant": [100], "bm25": list(range(60))},
        {"relevant": [0, 1], "bm25": [0, 0, 0, 0, 0, 1]},
    ]
    assert candidate_diagnostics(records, "bm25") == {
        "no_positive_at_5": 2,
        "positive_only_at_6_to_50": 1,
        "no_positive_at_50": 1,
    }
    assert summarize(records, "bm25")["recall_at_5"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("provider", ["deterministic", "sentence_transformers"])
async def test_graph_holds_no_connection_during_query_encoding_or_reranking(
    db, owner, monkeypatch, provider
):
    from app.db import engine
    from app.services import agent

    await db.commit()
    monkeypatch.setattr(settings, "embedding_provider", provider)

    async def encode(_texts):
        assert engine.pool.checkedout() == 0
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1)]

    async def rerank(_query, candidates, top_k):
        assert engine.pool.checkedout() == 0
        await asyncio.sleep(0.01)
        assert engine.pool.checkedout() == 0
        return candidates[:top_k]

    async def publish(*_args):
        pass

    monkeypatch.setattr(rag, "aembed_texts", encode)
    monkeypatch.setattr(rag, "arerank", rerank)
    monkeypatch.setattr(agent, "publish_event", publish)
    result = await agent._retrieve_node(
        {"org_id": owner.org_id, "prompt": "refund", "run_id": "test"}
    )
    assert "context" in result
