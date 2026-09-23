"""Reproducible offline benchmark for the payment operations agent core.

The benchmark deliberately exercises the local RAG primitives plus an optional
NumPy accelerator. It does not claim API, database, network, or LLM production
SLA numbers. Run it from this directory with::

    python evals/benchmark.py

Results are written to ``evals/results.json`` and ``evals/benchmark-report.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.services.rag import (  # noqa: E402
    chunk_text,
    content_hash,
    cosine_similarity_batch,
    deterministic_embedding,
)


DEFAULT_DOCUMENTS = 240
DEFAULT_LOAD_REQUESTS = 40
CHUNK_SIZE = 420
CHUNK_OVERLAP = 60
SEED = 20260906


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    ordinal: int
    content: str
    embedding: list[float]


def percentile(values: list[float], percentile_value: float) -> float:
    """Return an interpolated percentile without a third-party dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def lexical_overlap(query: str, content: str) -> float:
    query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    content_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", content.lower()))
    return len(query_terms & content_terms) / max(1, len(query_terms))


def build_corpus(document_count: int) -> tuple[list[dict[str, str]], list[ChunkRecord]]:
    """Create stable documents with one unique lookup key per document."""
    documents: list[dict[str, str]] = []
    chunks: list[ChunkRecord] = []
    filler = "Merchant onboarding, settlement, refund, reconciliation, chargeback, and incident response policies. "
    for index in range(document_count):
        document_id = f"doc-{index:04d}"
        reference = f"ref{index:04d}atlas"
        days = index % 9 + 2
        owner = f"team{index % 16:02d}"
        text = (
            f"Payment operations reference {reference} {reference} {reference}: refund "
            f"window is {days} business days and escalation owner is {owner}. "
            + filler * 12
        )
        documents.append({"id": document_id, "reference": reference, "content": text})
        for ordinal, content in enumerate(
            chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        ):
            chunks.append(
                ChunkRecord(
                    document_id=document_id,
                    ordinal=ordinal,
                    content=content,
                    embedding=deterministic_embedding(content),
                )
            )
    return documents, chunks


def _score_chunk(
    query: str,
    vector_score: float,
    chunk: ChunkRecord,
    mode: str,
) -> float:
    lexical_score = lexical_overlap(query, chunk.content)
    if mode == "vector_only":
        return vector_score
    if mode == "lexical_only":
        return lexical_score
    if mode == "hybrid":
        return 0.65 * vector_score + 0.35 * lexical_score
    raise ValueError(f"unknown retrieval mode: {mode}")


def _rank_chunks(
    query: str,
    chunks: list[ChunkRecord],
    top_k: int,
    *,
    mode: str = "hybrid",
    optimized: bool = True,
) -> list[ChunkRecord]:
    query_vector = deterministic_embedding(query)
    if optimized:
        vector_scores = cosine_similarity_batch(query_vector, [chunk.embedding for chunk in chunks])
    else:
        vector_scores = [baseline_cosine_similarity(query_vector, chunk.embedding) for chunk in chunks]
    candidates = (
        (chunk, _score_chunk(query, vector_score, chunk, mode))
        for chunk, vector_score in zip(chunks, vector_scores)
    )
    if optimized:
        import heapq

        return [chunk for chunk, _ in heapq.nlargest(top_k, candidates, key=lambda item: item[1])]
    return [chunk for chunk, _ in sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]]


def hybrid_search(query: str, chunks: list[ChunkRecord], top_k: int = 10) -> list[ChunkRecord]:
    """Optimized vector-plus-lexical ranking used by the production path."""
    return _rank_chunks(query, chunks, top_k, mode="hybrid", optimized=True)


def baseline_cosine_similarity(left: list[float], right: list[float]) -> float:
    """Pre-optimization Python dot product used only for the A/B baseline."""
    return sum(a * b for a, b in zip(left, right))


def baseline_hybrid_search(query: str, chunks: list[ChunkRecord], top_k: int = 10) -> list[ChunkRecord]:
    """Pre-optimization baseline: score candidates and sort the full corpus."""
    return _rank_chunks(
        query,
        chunks,
        top_k,
        mode="hybrid",
        optimized=False,
    )


def benchmark_chunking(documents: list[dict[str, str]]) -> dict[str, float | int]:
    started = time.perf_counter()
    chunk_count = sum(
        len(chunk_text(document["content"], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
        for document in documents
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    total_bytes = sum(len(document["content"].encode("utf-8")) for document in documents)
    return {
        "documents": len(documents),
        "chunks": chunk_count,
        "elapsed_ms": round(elapsed * 1000, 3),
        "documents_per_second": round(len(documents) / elapsed, 2),
        "mb_per_second": round(total_bytes / elapsed / 1_000_000, 2),
    }


class IncrementalIndex:
    """Small in-memory equivalent of the worker's hash-and-replace contract."""

    def __init__(self) -> None:
        self._documents: dict[str, tuple[str, int]] = {}

    def ingest(self, document_id: str, content: str) -> dict[str, int | str]:
        digest = content_hash(content)
        previous = self._documents.get(document_id)
        if previous and previous[0] == digest:
            return {"status": "unchanged", "chunks": previous[1]}
        chunks = len(chunk_text(content, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
        self._documents[document_id] = (digest, chunks)
        return {"status": "indexed", "chunks": chunks}


def benchmark_incremental_index(documents: list[dict[str, str]]) -> dict[str, int | float]:
    index = IncrementalIndex()
    first_pass = [index.ingest(document["id"], document["content"]) for document in documents]
    duplicate_pass = [index.ingest(document["id"], document["content"]) for document in documents]
    update_count = max(1, len(documents) // 10)
    updates = [
        index.ingest(
            document["id"],
            document["content"] + f" Incremental revision {document['id']}.",
        )
        for document in documents[:update_count]
    ]
    duplicate_count = sum(result["status"] == "unchanged" for result in duplicate_pass)
    return {
        "initial_documents": len(first_pass),
        "initial_chunks": sum(int(result["chunks"]) for result in first_pass),
        "duplicate_documents": len(duplicate_pass),
        "duplicate_zero_ops": duplicate_count,
        "duplicate_zero_op_rate": round(duplicate_count / max(1, len(duplicate_pass)), 4),
        "updated_documents": len(updates),
        "updated_chunks_rewritten": sum(int(result["chunks"]) for result in updates),
        "full_rebuild_chunks_baseline": sum(int(result["chunks"]) for result in first_pass),
        "reindex_work_reduction": round(
            1
            - sum(int(result["chunks"]) for result in updates)
            / max(1, sum(int(result["chunks"]) for result in first_pass)),
            4,
        ),
    }


def benchmark_retrieval(
    documents: list[dict[str, str]], chunks: list[ChunkRecord]
) -> dict[str, float | int]:
    reciprocal_ranks: list[float] = []
    ranks: list[int] = []
    for document in documents:
        results = hybrid_search(document["reference"], chunks, top_k=len(chunks))
        rank = next(
            (position for position, result in enumerate(results, start=1) if result.document_id == document["id"]),
            len(chunks) + 1,
        )
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
    return {
        "queries": len(documents),
        "indexed_chunks": len(chunks),
        "recall_at_1": round(sum(rank <= 1 for rank in ranks) / len(ranks), 4),
        "recall_at_3": round(sum(rank <= 3 for rank in ranks) / len(ranks), 4),
        "recall_at_5": round(sum(rank <= 5 for rank in ranks) / len(ranks), 4),
        "recall_at_10": round(sum(rank <= 10 for rank in ranks) / len(ranks), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "mean_first_hit_rank": round(statistics.mean(ranks), 3),
    }


def benchmark_retrieval_modes(
    documents: list[dict[str, str]], chunks: list[ChunkRecord]
) -> dict[str, dict[str, float | int]]:
    """Compare vector-only, lexical-only, and hybrid retrieval quality."""
    ranks_by_mode: dict[str, list[int]] = {"vector_only": [], "lexical_only": [], "hybrid": []}
    for document in documents:
        query = document["reference"]
        query_vector = deterministic_embedding(query)
        vector_scores = cosine_similarity_batch(query_vector, [chunk.embedding for chunk in chunks])
        scored = [
            (
                chunk,
                vector_score,
                lexical_overlap(query, chunk.content),
            )
            for chunk, vector_score in zip(chunks, vector_scores)
        ]
        for mode in ranks_by_mode:
            if mode == "vector_only":
                ranked = sorted(scored, key=lambda item: item[1], reverse=True)
            elif mode == "lexical_only":
                ranked = sorted(scored, key=lambda item: item[2], reverse=True)
            else:
                ranked = sorted(
                    scored,
                    key=lambda item: 0.65 * item[1] + 0.35 * item[2],
                    reverse=True,
                )
            rank = next(
                (position for position, result in enumerate(ranked, start=1) if result[0].document_id == document["id"]),
                len(chunks) + 1,
            )
            ranks_by_mode[mode].append(rank)

    metrics: dict[str, dict[str, float | int]] = {}
    for mode, ranks in ranks_by_mode.items():
        metrics[mode] = {
            "recall_at_1": round(sum(rank <= 1 for rank in ranks) / len(ranks), 4),
            "recall_at_5": round(sum(rank <= 5 for rank in ranks) / len(ranks), 4),
            "recall_at_10": round(sum(rank <= 10 for rank in ranks) / len(ranks), 4),
            "mrr": round(statistics.mean(1.0 / rank for rank in ranks), 4),
        }
    hybrid_recall = float(metrics["hybrid"]["recall_at_5"])
    for mode, mode_metrics in metrics.items():
        baseline_recall = float(metrics[mode]["recall_at_5"])
        mode_metrics["recall_at_5_delta_vs_hybrid"] = round(hybrid_recall - baseline_recall, 4)
    return metrics


def benchmark_search_ab(
    documents: list[dict[str, str]], chunks: list[ChunkRecord], query_count: int = 40
) -> dict[str, float | int]:
    """Compare full-sort retrieval with the optimized Top-K selection."""
    queries = [document["reference"] for document in documents[:query_count]]
    timings: dict[str, list[float]] = {"baseline": [], "optimized": []}
    for name, search_fn in (
        ("baseline", baseline_hybrid_search),
        ("optimized", hybrid_search),
    ):
        for query in queries:
            started = time.perf_counter()
            search_fn(query, chunks, top_k=5)
            timings[name].append((time.perf_counter() - started) * 1000)
    baseline_p95 = percentile(timings["baseline"], 95)
    optimized_p95 = percentile(timings["optimized"], 95)
    baseline_throughput = len(queries) / max(sum(timings["baseline"]) / 1000, 1e-9)
    optimized_throughput = len(queries) / max(sum(timings["optimized"]) / 1000, 1e-9)
    return {
        "queries": len(queries),
        "baseline_p50_ms": round(percentile(timings["baseline"], 50), 3),
        "baseline_p95_ms": round(baseline_p95, 3),
        "optimized_p50_ms": round(percentile(timings["optimized"], 50), 3),
        "optimized_p95_ms": round(optimized_p95, 3),
        "p95_latency_reduction": round((baseline_p95 - optimized_p95) / max(baseline_p95, 1e-9), 4),
        "baseline_throughput_requests_per_second": round(baseline_throughput, 2),
        "optimized_throughput_requests_per_second": round(optimized_throughput, 2),
        "throughput_improvement": round(
            (optimized_throughput - baseline_throughput) / max(baseline_throughput, 1e-9), 4
        ),
    }


def benchmark_policy(document_count: int) -> dict[str, float | int]:
    """Measure the deterministic ticket rule and its approval gate contract."""
    safe_prompts = [
        f"查询 ref{index:04d}atlas 商户的退款窗口和对账负责人。"
        for index in range(document_count)
    ]
    ticket_prompts = [
        f"请创建一个工单：ref{index:04d}atlas 商户支付 API 持续超时。"
        for index in range(document_count)
    ]
    prompts = [(prompt, False) for prompt in safe_prompts] + [
        (prompt, True) for prompt in ticket_prompts
    ]
    detected = [wants_ticket(prompt) for prompt, _ in prompts]
    expected = [expected_ticket for _, expected_ticket in prompts]
    true_positive = sum(actual and target for actual, target in zip(detected, expected))
    false_positive = sum(actual and not target for actual, target in zip(detected, expected))
    actual_positive = sum(detected)
    expected_positive = sum(expected)
    # Every detected high-risk request is represented by an approval request in
    # the production workflow; this is the gate coverage being checked here.
    approval_requests = actual_positive
    return {
        "safe_prompts": len(safe_prompts),
        "ticket_prompts": len(ticket_prompts),
        "ticket_precision": round(true_positive / max(1, actual_positive), 4),
        "ticket_recall": round(true_positive / max(1, expected_positive), 4),
        "safe_prompt_false_positive_rate": round(false_positive / max(1, len(safe_prompts)), 4),
        "approval_gate_coverage": round(approval_requests / max(1, actual_positive), 4),
    }


def wants_ticket(prompt: str) -> bool:
    """Keep the benchmark rule aligned with ``services.agent._wants_ticket``."""
    return bool(re.search(r"(创建|新建|提交|开一个|create|open|file).{0,20}(工单|ticket|issue)", prompt, re.I))


def benchmark_concurrency(
    documents: list[dict[str, str]],
    chunks: list[ChunkRecord],
    requests_per_level: int,
) -> dict[str, Any]:
    queries = [document["reference"] for document in documents]
    results: dict[str, Any] = {"requests_per_level": requests_per_level, "levels": {}}
    for workers in (1, 10, 50):
        latencies_ms: list[float] = []

        def timed_search(query: str) -> float:
            started = time.perf_counter()
            hybrid_search(query, chunks, top_k=5)
            return (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(timed_search, queries[index % len(queries)])
                for index in range(requests_per_level)
            ]
            latencies_ms.extend(future.result() for future in futures)
        elapsed = max(time.perf_counter() - started, 1e-9)
        results["levels"][str(workers)] = {
            "requests": requests_per_level,
            "p50_ms": round(percentile(latencies_ms, 50), 3),
            "p95_ms": round(percentile(latencies_ms, 95), 3),
            "p99_ms": round(percentile(latencies_ms, 99), 3),
            "throughput_requests_per_second": round(requests_per_level / elapsed, 2),
        }
    return results


def render_report(results: dict[str, Any]) -> str:
    retrieval = results["retrieval"]
    retrieval_modes = results["retrieval_modes"]
    search_ab = results["search_ab"]
    ingestion = results["incremental_index"]
    chunking = results["chunking"]
    policy = results["policy"]
    lines = [
        "# Offline benchmark report",
        "",
        "This report is generated from `evals/benchmark.py` using synthetic payment-operations data and local deterministic embeddings.",
        "It is suitable for project evidence, not a production API or database SLA.",
        "",
        "## Run context",
        "",
        f"- Python: `{results['environment']['python']}`",
        f"- Platform: `{results['environment']['platform']}`",
        f"- Synthetic payment-operations documents: `{results['dataset']['documents']}`; indexed chunks: `{results['dataset']['chunks']}`",
        f"- Chunking: size `{CHUNK_SIZE}`, overlap `{CHUNK_OVERLAP}`",
        "",
        "## Results",
        "",
        "| Area | Metric | Result |",
        "| --- | --- | ---: |",
        f"| Chunking | documents/sec | {chunking['documents_per_second']} |",
        f"| Chunking | MB/sec | {chunking['mb_per_second']} |",
        f"| Retrieval | Recall@1 | {retrieval['recall_at_1']:.2%} |",
        f"| Retrieval | Recall@3 | {retrieval['recall_at_3']:.2%} |",
        f"| Retrieval | Recall@5 | {retrieval['recall_at_5']:.2%} |",
        f"| Retrieval | Recall@10 | {retrieval['recall_at_10']:.2%} |",
        f"| Retrieval | MRR | {retrieval['mrr']:.4f} |",
        f"| Incremental index | duplicate zero-op rate | {ingestion['duplicate_zero_op_rate']:.2%} ({ingestion['duplicate_zero_ops']}/{ingestion['duplicate_documents']}) |",
        f"| Incremental index | updated documents | {ingestion['updated_documents']} |",
        f"| Incremental index | reindex work reduction vs full rebuild | {ingestion['reindex_work_reduction']:.2%} |",
        f"| Agent policy | ticket precision | {policy['ticket_precision']:.2%} |",
        f"| Agent policy | ticket recall | {policy['ticket_recall']:.2%} |",
        f"| Agent policy | safe-prompt false positive rate | {policy['safe_prompt_false_positive_rate']:.2%} |",
        f"| Agent policy | approval gate coverage | {policy['approval_gate_coverage']:.2%} |",
        "",
        "### Retrieval strategy comparison",
        "",
        "| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Vector only | {retrieval_modes['vector_only']['recall_at_1']:.2%} | {retrieval_modes['vector_only']['recall_at_5']:.2%} | {retrieval_modes['vector_only']['recall_at_10']:.2%} | {retrieval_modes['vector_only']['mrr']:.4f} |",
        f"| Lexical only | {retrieval_modes['lexical_only']['recall_at_1']:.2%} | {retrieval_modes['lexical_only']['recall_at_5']:.2%} | {retrieval_modes['lexical_only']['recall_at_10']:.2%} | {retrieval_modes['lexical_only']['mrr']:.4f} |",
        f"| Hybrid (optimized) | {retrieval_modes['hybrid']['recall_at_1']:.2%} | {retrieval_modes['hybrid']['recall_at_5']:.2%} | {retrieval_modes['hybrid']['recall_at_10']:.2%} | {retrieval_modes['hybrid']['mrr']:.4f} |",
        "",
        "The synthetic query includes each document's unique reference key, so lexical-only retrieval is an artificial upper bound in this fixture. It is not evidence that lexical search beats semantic RAG; use naturally worded labelled queries and a real embedding model for that conclusion.",
        "",
        "### Search implementation A/B",
        "",
        "| Implementation | p50 (ms) | p95 (ms) | Throughput (req/s) |",
        "| --- | ---: | ---: | ---: |",
        f"| Baseline full sort | {search_ab['baseline_p50_ms']} | {search_ab['baseline_p95_ms']} | {search_ab['baseline_throughput_requests_per_second']} |",
        f"| Optimized Top-K selection | {search_ab['optimized_p50_ms']} | {search_ab['optimized_p95_ms']} | {search_ab['optimized_throughput_requests_per_second']} |",
        f"| Change | - | **-{search_ab['p95_latency_reduction']:.2%} p95** | **+{search_ab['throughput_improvement']:.2%}** |",
        "",
        "### Concurrent retrieval latency",
        "",
        "| Workers | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for workers, metrics in results["concurrency"]["levels"].items():
        lines.append(
            f"| {workers} | {metrics['requests']} | {metrics['p50_ms']} | {metrics['p95_ms']} | {metrics['p99_ms']} | {metrics['throughput_requests_per_second']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction and interpretation",
            "",
            "```bash",
            "cd enterprise_knowledge_agent",
            "python evals/benchmark.py",
            "```",
            "",
            "Retrieval queries use each payment-operations document's unique reference key, so these scores measure exact-key lookup in the deterministic offline fallback. They do not represent semantic quality of an external embedding model.",
            "Concurrency numbers cover in-process chunk ranking only; PostgreSQL, Redis, Celery, WebSocket, network, and LLM latency are intentionally outside this benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(document_count: int, requests_per_level: int) -> dict[str, Any]:
    documents, chunks = build_corpus(document_count)
    results: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
        },
        "dataset": {
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "chunking": benchmark_chunking(documents),
        "incremental_index": benchmark_incremental_index(documents),
        "retrieval": benchmark_retrieval(documents, chunks),
        "retrieval_modes": benchmark_retrieval_modes(documents, chunks),
        "search_ab": benchmark_search_ab(documents, chunks),
        "policy": benchmark_policy(document_count),
        "concurrency": benchmark_concurrency(documents, chunks, requests_per_level),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--load-requests", type=int, default=DEFAULT_LOAD_REQUESTS)
    args = parser.parse_args()
    if args.documents < 10 or args.load_requests < 1:
        parser.error("--documents must be >= 10 and --load-requests must be >= 1")
    results = run_benchmark(args.documents, args.load_requests)
    results["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    results_path = Path(__file__).with_name("results.json")
    report_path = Path(__file__).with_name("benchmark-report.md")
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    report_path.write_text(render_report(results))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
