import httpx
import pytest
from evals.http_capacity import ROOT, summarize_samples, valid_response
from evals.http_capacity import render_report as render_capacity_report
from evals.retrieval_diagnostics import LexicalReference, paired_delta
from evals.sqlite_search_benchmark import render_report, summarize_rankings


def test_bm25_uses_rarity_frequency_and_length_instead_of_coverage_ties():
    reference = LexicalReference(
        [
            "the engineer",
            "the irrelevant filler filler filler",
            "the engineer " + "padding " * 20,
        ]
    )
    scores = reference.scores("the engineer")
    assert reference.idf["engineer"] > reference.idf["the"]
    assert scores["bm25"][0] > scores["bm25"][2] > scores["bm25"][1]
    assert scores["lexical_no_stopwords"].tolist() == [1.0, 0.0, 1.0]
    assert reference.scores("engineers")["bm25"].sum() == 0


def test_capacity_counts_timeouts_and_invalid_200_bodies_as_errors():
    samples = [
        {"latency_ms": 10, "status": 200, "valid": True, "error": ""},
        {"latency_ms": 15000, "status": "exception", "valid": False, "error": "ReadTimeout"},
    ]
    metrics = summarize_samples(samples, 15, 2000)
    assert metrics["error_rate"] == 0.5
    assert not metrics["slo_pass"]
    assert metrics["successful_requests_per_second"] == pytest.approx(1 / 15)
    assert metrics["error_counts"] == {"ReadTimeout": 1}
    assert not valid_response("agent_runs", httpx.Response(200, json={"status": "succeeded"}), "kb")
    assert not valid_response("knowledge_bases", httpx.Response(200, json=[{"id": "wrong"}]), "kb")
    assert valid_response(
        "health",
        httpx.Response(200, json={"status": "ok", "database": "sqlite", "llm_configured": False}),
        "kb",
    )


@pytest.mark.parametrize("citation_count", range(7))
def test_capacity_treats_top_five_as_a_limit_not_required_padding(citation_count):
    body = {
        "status": "succeeded",
        "answer": "grounded answer",
        "replayed": False,
        "citations": [
            {"chunk_id": f"chunk-{index}", "filename": f"msmarco-{index}.txt"}
            for index in range(citation_count)
        ],
    }
    assert valid_response("agent_runs", httpx.Response(200, json=body), "kb") == (
        1 <= citation_count <= 5
    )
    if citation_count:
        body["citations"] = [body["citations"][0]] * 2
        assert not valid_response("agent_runs", httpx.Response(200, json=body), "kb")


@pytest.mark.parametrize(
    "override",
    [
        {"answer": ""},
        {"status": "waiting_approval"},
        {"replayed": True},
        {"citations": [{"chunk_id": "chunk-1", "filename": "outside-corpus.txt"}]},
    ],
)
def test_capacity_still_rejects_invalid_answer_state_and_corpus(override):
    body = {
        "status": "succeeded",
        "answer": "grounded answer",
        "replayed": False,
        "citations": [{"chunk_id": "chunk-1", "filename": "msmarco-1.txt"}],
        **override,
    }
    assert not valid_response("agent_runs", httpx.Response(200, json=body), "kb")


def test_paired_comparison_respects_multi_positive_recall():
    records = [{"relevant": [1, 2], "before": [3], "after": [1]}] * 4
    result = paired_delta(records, "after", "before")
    assert result["recall_at_5_delta_percentage_points"] == 50
    assert result["paired_bootstrap_95_ci_percentage_points"] == [50, 50]


def test_actual_search_metrics_use_top_five_and_multi_positive_labels():
    rankings = [
        {
            "relevant": [1, 2],
            "lexical_only": [3, 4, 5, 6, 7, 1],
            "hybrid": [3, 1],
            "sqlite_fts": [1, 2],
        },
    ]
    metrics = summarize_rankings(rankings)
    assert metrics["lexical_only"] == {"recall_at_5": 0, "mrr_at_5": 0}
    assert metrics["hybrid"] == {"recall_at_5": 0.5, "mrr_at_5": 0.5}
    assert metrics["sqlite_fts"] == {"recall_at_5": 1, "mrr_at_5": 1}
    report = render_report(
        {
            "documents": 10,
            "chunks": 11,
            "queries": 1,
            "metrics": metrics,
            "vs_lexical": paired_delta(rankings, "sqlite_fts", "lexical_only"),
            "vs_hybrid": paired_delta(rankings, "sqlite_fts", "hybrid"),
        }
    )
    assert "10 文档、11 chunk" in report
    assert "| sqlite_fts | 100.00% | 1.0000 |" in report
    with pytest.raises(ValueError, match="No query"):
        summarize_rankings([])


def test_capacity_report_renders_current_measurements_not_old_conclusions():
    import json

    result = json.loads((ROOT / "evals/http-capacity-results.json").read_text())
    result["environment"]["retrieval"] = "test-index-path"
    result["environment"]["idempotency"] = "test-short-transaction"
    result["method"]["response_validation"] = "test-current-validation"
    for scenario in result["scenarios"]:
        scenario["server_error_counts"] = {}
    measurement = result["scenarios"][0]["measurements"][0]
    measurement["confirmed_concurrency"] = 123
    measurement["rounds"][0]["trials"][0]["p95_ms"] = 12.34
    report = render_capacity_report(result)
    assert "test-index-path" in report and "test-short-transaction" in report
    assert "test-current-validation" in report
    assert "| 12.3 |" in report and "| 123 |" in report
    assert "服务日志未发现" in report
    assert "112 秒" not in report
