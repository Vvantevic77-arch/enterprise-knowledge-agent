"""Frozen full-corpus retrieval ablations; no paid APIs or business database.

BM25 calls the actual SQLite candidate service. Dense retrieval is exact cached
cosine over the whole corpus; fusion and reranking reuse application functions.
Timings are serial warm local components, never HTTP capacity or ANN latency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from evals.http_capacity import ROOT, isolated_environment

MODES = ("bm25", "dense", "hybrid", "bm25_rerank", "hybrid_rerank")
CONDITIONS = (
    "English MS MARCO QA validation pool; incomplete relevance labels; frozen non-overlapping "
    "validation query subsets, not an independent dataset or Chinese evaluation; the 500-query "
    "subset was excluded from prior full-corpus diagnostics, but historical ten-candidate aggregate "
    "metrics covered the whole validation split; exact cached dense is not ANN/HTTP latency; "
    "union rerank is shared across ablations; query encoding excluded from per-query timings"
)
PLAN = {
    "embedding_model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "embedding_revision": "b207367332321f8e44f96e224ef15bc607f4dbf0",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "reranker_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "candidate_k": 50,
    "diagnostic_k": 100,
    "rrf_constant": 60,
    "final_k": 5,
    "chunk_size": 900,
    "chunk_overlap": 120,
    "holdout_queries": 500,
}


def freeze_manifest(queries, baseline: dict, path: Path, dataset_sha: str) -> dict:
    diagnostic = [row["query_id"] for row in baseline["pooled_rankings"]]
    eligible = [query.id for query in queries if query.relevant and query.id not in set(diagnostic)]
    eligible.sort(
        key=lambda query_id: hashlib.sha256(f"recall-holdout-v1:{query_id}".encode()).digest()
    )
    if len(eligible) < PLAN["holdout_queries"]:
        raise ValueError("Not enough disjoint labelled holdout queries")
    manifest = {
        "dataset_sha256": dataset_sha,
        "plan": PLAN,
        "diagnostic_query_ids": diagnostic,
        "holdout_query_ids": eligible[: PLAN["holdout_queries"]],
    }
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Frozen manifest changed; use a separate experiment")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(records: list[dict], mode: str) -> dict:
    from app.services.rag import recall_at_k, reciprocal_rank

    if not records or any(not row["relevant"] for row in records):
        raise ValueError("Metrics require queries with relevance labels")
    return {
        **{
            f"recall_at_{k}": float(
                np.mean([recall_at_k(row[mode], set(row["relevant"]), k) for row in records])
            )
            for k in (5, 20, 50, 100)
        },
        "mrr_at_5": float(
            np.mean([reciprocal_rank(row[mode][:5], set(row["relevant"])) for row in records])
        ),
    }


def candidate_diagnostics(records: list[dict], mode: str) -> dict:
    missed = [row for row in records if not set(row[mode][:5]) & set(row["relevant"])]
    return {
        "no_positive_at_5": len(missed),
        "positive_only_at_6_to_50": sum(
            bool(set(row[mode][:50]) & set(row["relevant"])) for row in missed
        ),
        "no_positive_at_50": sum(
            not bool(set(row[mode][:50]) & set(row["relevant"])) for row in missed
        ),
    }


def cached_embeddings(texts: list[str], directory: Path, label: str) -> np.ndarray:
    from app.services import rag

    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    path = directory / f"{label}-{digest}-{rag.embedding_identity()}.npy"
    if path.exists():
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        temporary = path.with_suffix(".tmp.npy")
        matrix = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), rag.settings.embedding_dimensions),
        )
        for start in range(0, len(texts), 512):
            matrix[start : start + 512] = rag.embed_texts(texts[start : start + 512])
            print(f"Embedded {label} {min(start + 512, len(texts))}/{len(texts)}", flush=True)
        matrix.flush()
        temporary.replace(path)
    if (
        matrix.shape != (len(texts), rag.settings.embedding_dimensions)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid embedding cache")
    return matrix


async def measure(args, database: Path, documents, queries, manifest: dict) -> dict:
    os.environ.update(isolated_environment(database))
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    from app.config import settings

    settings.database_url = os.environ["DATABASE_URL"]
    from app.db import SessionLocal, engine, init_db
    from app.services import rag
    from app.services.agent import _sqlite_lexical_candidates

    from evals.retrieval_diagnostics import paired_delta
    from evals.sqlite_search_benchmark import seed_quality_corpus

    assert settings.is_sqlite and not settings.llm_api_key

    settings.embedding_provider = "sentence_transformers"
    settings.embedding_model = PLAN["embedding_model"]
    settings.embedding_dimensions = 384
    settings.embedding_revision = PLAN["embedding_revision"]
    settings.retrieval_device = args.device
    settings.reranker_model = PLAN["reranker_model"]
    settings.reranker_revision = PLAN["reranker_revision"]
    selected_ids = manifest["diagnostic_query_ids"] + manifest["holdout_query_ids"]
    by_id = {query.id: query for query in queries}
    selected = [by_id[query_id] for query_id in selected_ids]
    candidates, chunks, chunk_to_doc = [], [], {}
    for doc_id, document in enumerate(documents):
        for ordinal, content in enumerate(rag.chunk_text(document)):
            chunk_id = f"chunk-{doc_id}-{ordinal}"
            chunk_to_doc[chunk_id] = doc_id
            chunks.append(content)
            candidates.append({"chunk_id": chunk_id, "content": content})
    args.cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    matrix = cached_embeddings(chunks, args.cache, "corpus")
    vectors = cached_embeddings([query.text for query in selected], args.cache, "queries")
    embedding_seconds = time.perf_counter() - started
    records = []
    try:
        await init_db()
        async with SessionLocal() as db:
            await seed_quality_corpus(db, documents)
            # Warm models; imports/downloads are reported separately from per-query timing.
            rag.rerank("warm up", candidates[:2], 1)
            for number, query in enumerate(selected):
                timings = {}
                started = time.perf_counter()
                lexical = await _sqlite_lexical_candidates(
                    db, "quality-org", query.text, ["quality-kb"], PLAN["diagnostic_k"]
                )
                timings["bm25_ms"] = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                scores = matrix @ vectors[number]
                order = np.argsort(-scores, kind="stable")[: PLAN["diagnostic_k"]]
                dense = [candidates[index] for index in order]
                timings["cached_exact_dense_ms"] = (time.perf_counter() - started) * 1000
                k = PLAN["candidate_k"]
                hybrid = rag.fuse_rankings([lexical[:k], dense[:k]], k)
                # Score their union once, then preserve each experiment's own candidate gate.
                pool = list({item["chunk_id"]: item for item in lexical[:k] + hybrid}.values())
                started = time.perf_counter()
                reranked = rag.rerank(query.text, pool, len(pool))
                timings["union_rerank_ms"] = (time.perf_counter() - started) * 1000
                rerank_scores = {item["chunk_id"]: item["score"] for item in reranked}
                rankings = {
                    "bm25": lexical,
                    "dense": dense,
                    "hybrid": hybrid,
                    "bm25_rerank": sorted(
                        lexical[:k], key=lambda item: rerank_scores[item["chunk_id"]], reverse=True
                    ),
                    "hybrid_rerank": sorted(
                        hybrid, key=lambda item: rerank_scores[item["chunk_id"]], reverse=True
                    ),
                }
                records.append(
                    {
                        "query_id": query.id,
                        "query": query.text,
                        "cohort": "diagnostic"
                        if number < len(manifest["diagnostic_query_ids"])
                        else "holdout",
                        "relevant": sorted(query.relevant),
                        # Keep chunk Top-k before document dedup, matching the actual API.
                        **{
                            mode: [chunk_to_doc[item["chunk_id"]] for item in ranking]
                            for mode, ranking in rankings.items()
                        },
                        "timings": timings,
                    }
                )
                if (number + 1) % 50 == 0:
                    print(f"Evaluated {number + 1}/{len(selected)}", flush=True)
    finally:
        await engine.dispose()
    cohorts = {}
    for cohort in ("diagnostic", "holdout"):
        rows = [row for row in records if row["cohort"] == cohort]
        cohorts[cohort] = {
            "queries": len(rows),
            "metrics": {mode: summarize(rows, mode) for mode in MODES},
            "candidate_diagnostics": {
                mode: candidate_diagnostics(rows, mode) for mode in ("bm25", "dense", "hybrid")
            },
            "comparisons": {
                mode: paired_delta(rows, mode, "bm25") for mode in MODES if mode != "bm25"
            },
            "component_timings": {
                key: {
                    "p50_ms": float(np.percentile([r["timings"][key] for r in rows], 50)),
                    "p95_ms": float(np.percentile([r["timings"][key] for r in rows], 95)),
                }
                for key in rows[0]["timings"]
            },
        }
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "backend/app").rglob("*.py")):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "documents": len(documents),
        "chunks": len(chunks),
        "plan": PLAN,
        "embedding_revision": settings.embedding_revision,
        "reranker_revision": settings.reranker_revision,
        "model_weights_sha256": {
            kind: hashlib.sha256(
                (
                    ROOT
                    / ".cache/huggingface/hub"
                    / ("models--" + PLAN[f"{kind}_model"].replace("/", "--"))
                    / "snapshots"
                    / PLAN[f"{kind}_revision"]
                    / "model.safetensors"
                ).read_bytes()
            ).hexdigest()
            for kind in ("embedding", "reranker")
        },
        "device": args.device,
        "platform": platform.platform(),
        "backend_source_sha256": source_hash.hexdigest(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "sentence-transformers",
                "transformers",
                "sqlalchemy",
                "pyarrow",
            )
        },
        "embedding_build_or_cache_seconds": embedding_seconds,
        "conditions": CONDITIONS,
        "cohorts": cohorts,
        "rankings": records,
    }


def render_report(result: dict) -> str:
    lines = [
        "# 语义召回与重排对照",
        "",
        result["conditions"],
        "",
        f"语料：{result['documents']:,} 文档 / {result['chunks']:,} chunk。最终固定返回 5 个 chunk。",
        "",
    ]
    for cohort, data in result["cohorts"].items():
        lines += [
            f"## {cohort}（{data['queries']} 查询）",
            "",
            "| 方法 | Recall@5 | Recall@20 | Recall@50 | Recall@100 | MRR@5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode, metrics in data["metrics"].items():
            # Fusion/reranking only has 50 candidates; do not label it Recall@100.
            r100 = f"{metrics['recall_at_100']:.2%}" if mode in {"bm25", "dense"} else "—"
            lines.append(
                f"| {mode} | {metrics['recall_at_5']:.2%} | {metrics['recall_at_20']:.2%} | {metrics['recall_at_50']:.2%} | {r100} | {metrics['mrr_at_5']:.4f} |"
            )
        lines += [
            "",
            f"候选诊断：`{json.dumps(data['candidate_diagnostics'], ensure_ascii=False)}`",
            "",
        ]
        for mode, comparison in data["comparisons"].items():
            low, high = comparison["paired_bootstrap_95_ci_percentage_points"]
            lines.append(
                f"- {mode} 相对 BM25：{comparison['recall_at_5_delta_percentage_points']:+.2f} 个百分点；配对 bootstrap 95% 区间 [{low:+.2f}, {high:+.2f}]。"
            )
    lines += [
        "",
        "模型固定版本：",
        "",
        f"- embedding：`{result['plan']['embedding_model']}@{result['embedding_revision']}`",
        f"- reranker：`{result['plan']['reranker_model']}@{result['reranker_revision']}`",
        "",
        "组件延迟保存在 JSON；不含在线 query 编码、HTTP、模型生成。实际 SQLite 语义路径仍读取 JSON 向量；生产大语料需单独测 pgvector 与租户过滤后的 ANN 召回。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/tmp/enterprise_rag_datasets/msmarco-v1.1-validation.parquet"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "evals/recall-experiment-manifest.json"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/semantic-retrieval")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evals/semantic-retrieval-results.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    import pyarrow.parquet as pq

    from evals.public_benchmark import EXPECTED_SHA256, prepare_rows

    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch")
    documents, queries = prepare_rows(pq.read_table(args.dataset).to_pylist())
    baseline = json.loads((ROOT / "evals/public-benchmark-results.json").read_text())
    manifest = freeze_manifest(queries, baseline, args.manifest, EXPECTED_SHA256)
    if args.prepare_only:
        print(
            f"Frozen {len(manifest['diagnostic_query_ids'])} diagnostic + {len(manifest['holdout_query_ids'])} holdout queries"
        )
        return
    with tempfile.TemporaryDirectory(prefix="agent-semantic-quality-") as directory:
        result = asyncio.run(
            measure(args, Path(directory) / "quality.db", documents, queries, manifest)
        )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    args.output.with_name(args.output.stem.replace("-results", "") + "-report.md").write_text(
        render_report(result)
    )
    print(json.dumps(result["cohorts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
