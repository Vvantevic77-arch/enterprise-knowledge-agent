import numpy as np
import pytest
from app.services.rag import cosine_similarity_batch, deterministic_embedding, hybrid_score
from evals.public_benchmark import PublicIndex, Query, prepare_rows, summarize


def test_public_labels_deduplicate_without_leaking_candidate_order():
    documents, queries = prepare_rows(
        [
            {
                "query_id": 1,
                "query": "refund",
                "passages": {
                    "passage_text": ["refund policy", "invoice policy"],
                    "is_selected": [1, 0],
                },
            },
            {
                "query_id": 2,
                "query": "invoice",
                "passages": {
                    "passage_text": ["invoice policy", "refund policy", "invoice policy"],
                    "is_selected": [1, 0, 1],
                },
            },
        ]
    )
    assert set(documents) == {"refund policy", "invoice policy"}
    assert queries[1].candidates == (0, 1)
    assert queries[1].relevant == frozenset({documents.index("invoice policy")})
    reordered, _ = prepare_rows(
        [
            {
                "query_id": 3,
                "query": "x",
                "passages": {
                    "passage_text": list(reversed(documents)),
                    "is_selected": [0, 1],
                },
            },
        ]
    )
    assert reordered == documents
    with pytest.raises(ValueError, match="Duplicate query"):
        prepare_rows(
            [
                {
                    "query_id": 1,
                    "query": "x",
                    "passages": {
                        "passage_text": ["x"],
                        "is_selected": [1],
                    },
                }
            ]
            * 2
        )


def test_cached_scores_match_production_and_aggregate_split_passages():
    documents = ["settlement " * 110 + "refund 401", "invoice policy", "refund policy"]
    index = PublicIndex(documents)
    assert index.starts[1] > 1
    query = "refund 401"
    vectors = [deterministic_embedding(chunk) for chunk in index.chunks]
    dense = cosine_similarity_batch(deterministic_embedding(query), vectors)
    expected = np.asarray(
        [hybrid_score(query, chunk, score) for chunk, score in zip(index.chunks, dense)]
    )
    scores = index.scores(query, len(documents))
    np.testing.assert_allclose(scores["hybrid"], expected, atol=1e-6)
    passage_scores = [
        max(expected[index.starts[doc] : index.starts[doc + 1]]) for doc in range(len(documents))
    ]
    expected_order = sorted(range(len(documents)), key=lambda doc: -passage_scores[doc])
    assert index.rank(scores["hybrid"], len(documents)) == expected_order
    labelled_query = Query(1, query, (0, 2), frozenset({2}))
    assert index.candidate_rankings(labelled_query)["hybrid"] == [
        doc for doc in expected_order if doc in (0, 2)
    ]
    subset_scores = index.scores(query, 1)
    assert index.rank(subset_scores["hybrid"], 1) == [0]


def test_recall_counts_all_positives_and_mrr_is_truncated():
    records = [
        {"relevant": [1, 2], "hybrid": [1, 3]},
        {"relevant": [20], "hybrid": list(range(20)) + [20]},
    ]
    metrics = summarize(records, "hybrid")
    assert metrics["recall_at_5"] == 0.25
    assert metrics["hit_at_5"] == 0.5
    assert metrics["mrr_at_10"] == 0.5
    with pytest.raises(ValueError, match="without positives"):
        summarize([{"relevant": [], "hybrid": [1]}], "hybrid")
