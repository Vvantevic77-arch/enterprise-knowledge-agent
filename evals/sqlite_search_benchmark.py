"""Quality regression of the actual SQLite search service on the baseline query IDs.

Uses an isolated database and no paid APIs. Embeddings are omitted because the
English FTS path never reads them; this measures retrieval quality, not latency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from evals.http_capacity import ROOT, isolated_environment


def summarize_rankings(records: list[dict]) -> dict:
    from app.services.rag import recall_at_k, reciprocal_rank

    if not records:
        raise ValueError("No query rankings to evaluate")
    return {
        mode: {
            "recall_at_5": sum(recall_at_k(r[mode], set(r["relevant"]), 5) for r in records)
            / len(records),
            "mrr_at_5": sum(reciprocal_rank(r[mode][:5], set(r["relevant"])) for r in records)
            / len(records),
        }
        for mode in ("lexical_only", "hybrid", "sqlite_fts")
    }


async def seed_quality_corpus(db, documents: list[str]) -> tuple[dict, int, int]:
    from app.models import Document, DocumentChunk, KnowledgeBase, Organization, User
    from app.services.rag import chunk_text
    from sqlalchemy import func, insert, select

    chunk_documents = {}
    db.add(User(id="quality-user", email="quality@example.invalid"))
    await db.flush()
    db.add(Organization(id="quality-org", name="Isolated quality test", created_by="quality-user"))
    await db.flush()
    db.add(KnowledgeBase(id="quality-kb", org_id="quality-org", name="MS MARCO"))
    await db.commit()
    for start in range(0, len(documents), 500):
        doc_rows, chunk_rows = [], []
        for index, content in enumerate(documents[start : start + 500], start):
            pieces = chunk_text(content)
            doc_rows.append(
                {
                    "id": f"document-{index}",
                    "org_id": "quality-org",
                    "kb_id": "quality-kb",
                    "filename": f"msmarco-{index}.txt",
                    "status": "indexed",
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                    "chunk_count": len(pieces),
                }
            )
            for ordinal, piece in enumerate(pieces):
                chunk_id = f"chunk-{index}-{ordinal}"
                chunk_documents[chunk_id] = index
                chunk_rows.append(
                    {
                        "id": chunk_id,
                        "document_id": f"document-{index}",
                        "ordinal": ordinal,
                        "content": piece,
                    }
                )
        await db.execute(insert(Document), doc_rows)
        await db.execute(insert(DocumentChunk), chunk_rows)
        await db.commit()
    actual_documents = await db.scalar(select(func.count(Document.id)))
    actual_chunks = await db.scalar(select(func.count(DocumentChunk.id)))
    return chunk_documents, actual_documents, actual_chunks


async def measure(dataset: Path, baseline_path: Path, database: Path, candidate_k: int = 5) -> dict:
    os.environ.update(isolated_environment(database))
    import pyarrow.parquet as pq
    from app.config import settings
    from app.db import SessionLocal, engine, init_db
    from app.services.agent import search_knowledge

    from evals.public_benchmark import EXPECTED_SHA256, prepare_rows
    from evals.retrieval_diagnostics import paired_delta

    assert settings.is_sqlite and settings.database_url == os.environ["DATABASE_URL"]
    assert settings.embedding_provider == "deterministic" and not settings.llm_api_key
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch")
    baseline = json.loads(baseline_path.read_text())
    documents, queries = prepare_rows(pq.read_table(dataset).to_pylist())
    by_id = {query.id: query for query in queries}
    chunk_documents, records = {}, []
    try:
        await init_db()
        async with SessionLocal() as db:
            chunk_documents, actual_documents, actual_chunks = await seed_quality_corpus(
                db, documents
            )
            for original in baseline["pooled_rankings"]:
                query = by_id[original["query_id"]]
                if sorted(query.relevant) != sorted(original["relevant"]):
                    raise ValueError("Baseline relevance labels do not match the dataset")
                results = await search_knowledge(
                    db, "quality-org", query.text, ["quality-kb"], top_k=candidate_k
                )
                ranked = [chunk_documents[row["chunk_id"]] for row in results]
                if candidate_k == 5:
                    ranked = list(dict.fromkeys(ranked))
                records.append(
                    {
                        "query_id": query.id,
                        "relevant": sorted(query.relevant),
                        "lexical_only": original["lexical_only"][:5],
                        "hybrid": original["hybrid"][:5],
                        "sqlite_fts": ranked,
                    }
                )
                if len(records) % 200 == 0:
                    print(f"Evaluated {len(records)} actual-service queries", flush=True)
    finally:
        await engine.dispose()
    candidate_metrics = {}
    if candidate_k >= 100:
        from evals.semantic_retrieval import candidate_diagnostics, summarize

        candidate_metrics = {
            "candidate_k": candidate_k,
            "metrics": summarize(records, "sqlite_fts"),
            "diagnostics": candidate_diagnostics(records, "sqlite_fts"),
        }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "documents": actual_documents,
        "chunks": actual_chunks,
        "queries": len(records),
        "dataset_sha256": EXPECTED_SHA256,
        "conditions": "actual search_knowledge; English MS MARCO; embeddings omitted; quality only",
        "metrics": summarize_rankings(records),
        "vs_lexical": paired_delta(records, "sqlite_fts", "lexical_only"),
        "vs_hybrid": paired_delta(records, "sqlite_fts", "hybrid"),
        "rankings": records,
        "candidate_analysis": candidate_metrics,
    }


def render_report(result: dict) -> str:
    lines = [
        "# SQLite 实际检索路径质量回归",
        "",
        f"{result['documents']:,} 文档、{result['chunks']:,} chunk，沿用原始基线的 {result['queries']:,} 个查询 ID 与全部标签。",
        "实际调用当前 `search_knowledge`，不是离线 BM25 参考实现。隔离 SQLite 不保存向量，英文索引路径不读取向量；未测延迟、HTTP、中文检索、模型回答准确性或 PostgreSQL。",
        "",
        "| 方法 | Recall@5 | MRR@5 |",
        "| --- | ---: | ---: |",
    ]
    for mode, metrics in result["metrics"].items():
        lines.append(f"| {mode} | {metrics['recall_at_5']:.2%} | {metrics['mrr_at_5']:.4f} |")
    if result.get("candidate_analysis"):
        analysis = result["candidate_analysis"]
        lines += ["", "## 候选召回诊断", ""]
        lines += [
            f"- Recall@{k}：{analysis['metrics'][f'recall_at_{k}']:.2%}" for k in (5, 20, 50, 100)
        ]
        lines += [
            "",
            f"`{json.dumps(analysis['diagnostics'], ensure_ascii=False)}`",
            "",
            "固定 Top-5 指标没有扩大返回数量；@20/@50/@100 仅用于诊断候选覆盖率。同一文档的多个 chunk 仍占据各自排名位置。",
        ]
    for name in ("vs_lexical", "vs_hybrid"):
        comparison = result[name]
        low, high = comparison["paired_bootstrap_95_ci_percentage_points"]
        lines.extend(
            [
                "",
                f"{name}：Recall@5 差 {comparison['recall_at_5_delta_percentage_points']:+.2f} 个百分点；配对 bootstrap 95% 区间 [{low:+.2f}, {high:+.2f}]。",
            ]
        )
    lines.extend(
        [
            "",
            "该验证集沿用诊断样本，未进行新的调参；不属于独立保留测试集。全库标注不穷尽相关答案，Recall@5 不是回答准确率。实际 API 取 Top-5 chunk，再映射去重到文档；原始基线按文档排名，多 chunk 文档可能使两者有轻微差异。",
            "",
            "复现：`.venv/bin/python -m evals.sqlite_search_benchmark`。语料 SHA-256、全部查询 ID、标签和排名保存在对应 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/tmp/enterprise_rag_datasets/msmarco-v1.1-validation.parquet"),
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "evals/public-benchmark-results.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "evals/sqlite-search-results.json")
    parser.add_argument("--candidate-k", type=int, choices=(5, 100), default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="agent-sqlite-quality-") as directory:
        result = asyncio.run(
            measure(args.dataset, args.baseline, Path(directory) / "quality.db", args.candidate_k)
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_name(args.output.stem.replace("-results", "") + "-report.md").write_text(
        render_report(result)
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "rankings"}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
