"""MS MARCO QA candidate ranking and pooled-corpus retrieval, without paid APIs.

This reuses the application's chunker, embedding and scoring weights. Cached
NumPy vectors and lexical postings isolate retrieval from database/API overhead;
their timing must not be presented as the production search endpoint's latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
# Prevent the application's dotenv loader from activating external model calls.
os.environ["OPENAI_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["EMBEDDING_PROVIDER"] = "deterministic"

from app.services.rag import (  # noqa: E402
    LEXICAL_WEIGHT,
    VECTOR_WEIGHT,
    chunk_text,
    deterministic_embedding,
    lexical_overlap,
    lexical_terms,
    recall_at_k,
    reciprocal_rank,
)

REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
SOURCE_URL = (
    f"https://huggingface.co/datasets/microsoft/ms_marco/resolve/{REVISION}"
    "/v1.1/validation-00000-of-00001.parquet?download=true"
)
EXPECTED_SHA256 = "67aca2e1b2fffe418a6bd317142887c90973c9c98dd8d940eaae3f8521daed7c"
SEED = 20261012
DIMENSIONS = 1536
MODES = ("vector_only", "lexical_only", "hybrid")


@dataclass(frozen=True)
class Query:
    id: int
    text: str
    candidates: tuple[int, ...]
    relevant: frozenset[int]


def prepare_rows(rows: list[dict]) -> tuple[list[str], list[Query]]:
    """Deduplicate by exact passage text, preserving the split's label mapping."""
    # Corpus IDs must be independent of the source's candidate/relevance order.
    documents = sorted(
        {text for row in rows for text in row["passages"]["passage_text"]},
        key=lambda text: hashlib.sha256(text.encode("utf-8")).digest(),
    )
    document_ids = {text: index for index, text in enumerate(documents)}
    queries: list[Query] = []
    for row in rows:
        passages = row["passages"]
        candidates: list[int] = []
        relevant: set[int] = set()
        for text, selected in zip(passages["passage_text"], passages["is_selected"], strict=True):
            if not text.strip() or selected not in (0, 1):
                raise ValueError("Invalid passage text or relevance label")
            document_id = document_ids[text]
            candidates.append(document_id)
            if selected:
                relevant.add(document_id)
        queries.append(
            Query(
                int(row["query_id"]),
                row["query"],
                tuple(sorted(set(candidates))),
                frozenset(relevant),
            )
        )
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("Duplicate query IDs in the split")
    return documents, queries


class PublicIndex:
    def __init__(self, documents: list[str], *, progress: bool = False):
        started = time.perf_counter()
        self.chunks: list[str] = []
        starts = [0]
        for document in documents:
            # Defaults match ingestion: 900 characters with 120 overlap.
            pieces = chunk_text(document)
            if not pieces:
                raise ValueError("Cannot index an empty document")
            self.chunks.extend(pieces)
            starts.append(len(self.chunks))
        self.starts = np.asarray(starts, dtype=np.int32)
        self.chunking_seconds = time.perf_counter() - started
        self.matrix = np.empty((len(self.chunks), DIMENSIONS), dtype=np.float32)
        postings: dict[str, list[int]] = defaultdict(list)
        for index, text in enumerate(self.chunks):
            self.matrix[index] = deterministic_embedding(text, DIMENSIONS)
            for term in lexical_terms(text):
                postings[term].append(index)
            if progress and (index + 1) % 10000 == 0:
                print(f"Indexed {index + 1}/{len(self.chunks)} chunks", flush=True)
        self.postings = {
            term: np.asarray(indices, dtype=np.int32) for term, indices in postings.items()
        }
        self.build_seconds = time.perf_counter() - started

    def scores(self, query: str, document_count: int) -> dict[str, np.ndarray]:
        end = int(self.starts[document_count])
        vector = np.asarray(deterministic_embedding(query, DIMENSIONS), dtype=np.float32)
        dense = self.matrix[:end] @ vector
        lexical = np.zeros(end, dtype=np.float32)
        terms = lexical_terms(query)
        for term in terms:
            indices = self.postings.get(term)
            if indices is not None:
                lexical[indices[: np.searchsorted(indices, end)]] += 1
        lexical /= max(1, len(terms))
        return {
            "vector_only": dense,
            "lexical_only": lexical,
            "hybrid": VECTOR_WEIGHT * dense + LEXICAL_WEIGHT * lexical,
        }

    def rank(self, scores: np.ndarray, document_count: int, k: int = 10) -> list[int]:
        # Passage-level labels: aggregate each passage's chunks by max score.
        document_scores = np.maximum.reduceat(scores, self.starts[:document_count])
        # Stable ties follow corpus ID, never relevance or candidate order.
        return np.argsort(-document_scores, kind="stable")[:k].tolist()

    def candidate_rankings(self, query: Query) -> dict[str, list[int]]:
        indices = np.concatenate(
            [np.arange(self.starts[doc], self.starts[doc + 1]) for doc in query.candidates]
        )
        dense = self.matrix[indices] @ np.asarray(
            deterministic_embedding(query.text, DIMENSIONS), dtype=np.float32
        )
        lexical = np.asarray(
            [lexical_overlap(query.text, self.chunks[index]) for index in indices], dtype=np.float32
        )
        offsets = np.cumsum(
            [0] + [int(self.starts[doc + 1] - self.starts[doc]) for doc in query.candidates]
        )[:-1]
        scores = {
            "vector_only": dense,
            "lexical_only": lexical,
            "hybrid": VECTOR_WEIGHT * dense + LEXICAL_WEIGHT * lexical,
        }
        return {
            mode: [
                query.candidates[position]
                for position in np.argsort(-np.maximum.reduceat(values, offsets), kind="stable")[
                    :10
                ]
            ]
            for mode, values in scores.items()
        }


def summarize(records: list[dict], mode: str) -> dict:
    recalls = {k: [] for k in (1, 5, 10)}
    reciprocal_ranks: list[float] = []
    hits: list[float] = []
    for record in records:
        ranked = [str(doc) for doc in record[mode]]
        relevant = {str(doc) for doc in record["relevant"]}
        if not relevant:
            raise ValueError("Queries without positives must be excluded from quality metrics")
        for k in recalls:
            recalls[k].append(recall_at_k(ranked, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(ranked[:10], relevant))
        hits.append(float(bool(set(ranked[:5]) & relevant)))
    recall_5 = np.asarray(recalls[5])
    rng = np.random.default_rng(SEED)
    bootstrap = np.mean(rng.choice(recall_5, size=(1000, len(records))), axis=1)
    return {
        "queries": len(records),
        **{f"recall_at_{k}": float(np.mean(values)) for k, values in recalls.items()},
        "hit_at_5": float(np.mean(hits)),
        "mrr_at_10": float(np.mean(reciprocal_ranks)),
        "recall_at_5_bootstrap_95_ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    }


def run(path: Path, query_count: int, latency_queries: int, repeats: int) -> dict:
    import pyarrow
    import pyarrow.parquet as pq

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset checksum mismatch: {digest}")
    rows = pq.read_table(path).to_pylist()
    documents, queries = prepare_rows(rows)
    eligible = [query for query in queries if query.relevant]
    if not 1 <= query_count <= len(eligible):
        raise ValueError(f"--queries must be between 1 and {len(eligible)}")
    if not 1 <= latency_queries <= query_count or repeats < 1:
        raise ValueError("Invalid latency query count or repeats")
    raw_passages = sum(len(row["passages"]["passage_text"]) for row in rows)
    del rows
    sampled = random.Random(SEED).sample(sorted(eligible, key=lambda q: q.id), query_count)
    print(f"Loaded {len(documents)} unique passages, {len(eligible)} labelled queries", flush=True)
    index = PublicIndex(documents, progress=True)
    print(f"Built {len(index.chunks)} chunks in {index.build_seconds:.2f}s", flush=True)

    candidate_records: list[dict] = []
    for query in eligible:
        candidate_records.append(
            {
                "relevant": sorted(query.relevant),
                **index.candidate_rankings(query),
            }
        )
    candidate_metrics = {mode: summarize(candidate_records, mode) for mode in MODES}
    del candidate_records
    print("Completed candidate ranking", flush=True)

    pool_records: list[dict] = []
    for number, query in enumerate(sampled, start=1):
        scores = index.scores(query.text, len(documents))
        pool_records.append(
            {
                "query_id": query.id,
                "relevant": sorted(query.relevant),
                **{mode: index.rank(values, len(documents)) for mode, values in scores.items()},
            }
        )
        if number % 100 == 0:
            print(f"Pooled retrieval {number}/{len(sampled)}", flush=True)
    pool_metrics = {mode: summarize(pool_records, mode) for mode in MODES}

    latency: list[dict] = []
    for count in sorted({min(size, len(documents)) for size in (10000, 50000, len(documents))}):
        for query in sampled[:5]:
            index.rank(index.scores(query.text, count)["hybrid"], count)
        timings: list[float] = []
        for _ in range(repeats):
            for query in sampled[:latency_queries]:
                started = time.perf_counter()
                index.rank(index.scores(query.text, count)["hybrid"], count)
                timings.append((time.perf_counter() - started) * 1000)
        latency.append(
            {
                "passages": count,
                "chunks": int(index.starts[count]),
                "requests": len(timings),
                "repeats": repeats,
                "p50_ms": float(np.percentile(timings, 50)),
                "p95_ms": float(np.percentile(timings, 95)),
                "p99_ms": float(np.percentile(timings, 99)),
                "serial_queries_per_second": 1000 * len(timings) / sum(timings),
            }
        )
        print(f"Latency at {count} passages: p95={latency[-1]['p95_ms']:.2f}ms", flush=True)

    examples = []
    for query, record in zip(sampled, pool_records, strict=True):
        if not set(record["hybrid"][:5]) & query.relevant and len(examples) < 5:
            examples.append(
                {
                    "query_id": query.id,
                    "query": query.text,
                    "selected_passage": documents[min(query.relevant)],
                    "top_hybrid_passage": documents[record["hybrid"][0]],
                }
            )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "cpu_count": os.cpu_count(),
        },
        "dataset": {
            "name": "MS MARCO v1.1 validation QA split",
            "source_url": SOURCE_URL,
            "revision": REVISION,
            "sha256": digest,
            "file_bytes": path.stat().st_size,
            "queries": len(queries),
            "labelled_queries": len(eligible),
            "excluded_unlabelled_queries": len(queries) - len(eligible),
            "raw_passages": raw_passages,
            "unique_passages": len(documents),
            "text_bytes": sum(len(doc.encode("utf-8")) for doc in documents),
            "scale_vs_240_synthetic_documents": len(documents) / 240,
        },
        "index": {
            "chunks": len(index.chunks),
            "chunk_size": 900,
            "overlap": 120,
            "dimensions": DIMENSIONS,
            "embedding": "SHA-256 deterministic; not semantic",
            "vector_weight": VECTOR_WEIGHT,
            "lexical_weight": LEXICAL_WEIGHT,
            "build_seconds": index.build_seconds,
            "chunking_seconds": index.chunking_seconds,
            "vector_matrix_bytes": index.matrix.nbytes,
            "dtype": "float32",
            "layout": "cached in-memory dense matrix and lexical postings; full scan",
        },
        "sampling": {"seed": SEED, "pooled_queries": query_count},
        "candidate_ranking": candidate_metrics,
        "pooled_retrieval": pool_metrics,
        "latency": latency,
        "pooled_rankings": pool_records,
        "failure_examples": examples,
    }


def render_report(result: dict) -> str:
    data, index = result["dataset"], result["index"]
    lines = [
        "# MS MARCO 公开数据检索实验",
        "",
        f"实验时间（UTC）：{result['generated_at_utc']}",
        "",
        f"使用 **{data['unique_passages']:,} 条真实网页段落**（去重前 {data['raw_passages']:,} 条），"
        f"来自 **{data['queries']:,} 个真实问题**；其中 {data['labelled_queries']:,} 个有正例，"
        f"{data['excluded_unlabelled_queries']} 个无正例的问题不计入质量指标。"
        f"规模为原 240 篇合成文档的 **{data['scale_vs_240_synthetic_documents']:.2f} 倍**。",
        "",
        f"- 原文：{data['text_bytes'] / 1e6:.2f} MB；按 900 字符、120 重叠切分为 **{index['chunks']:,} 个 chunk**。",
        f"- 本地建索引：**{index['build_seconds']:.2f} s**（切分、确定性向量、词法倒排；不含下载、Parquet 解析、数据库写入）。",
        f"- 向量矩阵：1,536 维 float32，{index['vector_matrix_bytes'] / 2**20:.2f} MiB；不含文本和倒排索引内存。",
        f"- 复用 `app.services.rag` 的切分、向量、分词和权重：{VECTOR_WEIGHT} × 向量 + {LEXICAL_WEIGHT} × 词法重叠。",
        f"- 环境：{result['environment']}",
        "",
        "## 检索质量",
        "",
        "候选排序对每个问题只排序数据集给定的候选段落；全库检索把所有去重段落放入同一个池，"
        f"按固定种子 {SEED} 从有正例的问题中随机抽取 {result['sampling']['pooled_queries']:,} 个。"
        "语料 ID 按文本 SHA-256 排序，独立于原候选顺序；候选按该 ID 排序后打分，避免原候选顺序泄漏。"
        "段落有多个 chunk 时取最高分，Top-K 去重到段落。",
        "",
        "| 场景 | 方法 | 问题数 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, key in (("候选排序", "candidate_ranking"), ("全库检索", "pooled_retrieval")):
        for mode in MODES:
            metrics = result[key][mode]
            lines.append(
                f"| {scenario} | {mode} | {metrics['queries']:,} | {metrics['recall_at_1']:.2%} | "
                f"{metrics['recall_at_5']:.2%} | {metrics['recall_at_10']:.2%} | {metrics['mrr_at_10']:.4f} |"
            )
    ci = result["pooled_retrieval"]["hybrid"]["recall_at_5_bootstrap_95_ci"]
    delta = (
        result["pooled_retrieval"]["hybrid"]["recall_at_5"]
        - result["pooled_retrieval"]["lexical_only"]["recall_at_5"]
    )
    lines += [
        "",
        "候选集每题最多 10 条，因此候选排序的 Recall@10=100% 是候选全集覆盖的必然结果，不能作为检索能力证明。"
        "Recall@K 是逐问题计算“前 K 条命中的已标注正例数 / 该问题已标注正例数”后取宏平均；"
        "MRR@10 只计算前 10 条内第一个正例的倒数排名。命中率另存于 JSON，避免与 Recall 混淆。",
        f"混合检索的全库 Recall@5 bootstrap 95% 区间：**{ci[0]:.2%}–{ci[1]:.2%}**（1,000 次重采样）；"
        f"相对纯词法的差值为 **{delta * 100:+.2f} 个百分点**。这是现有回退方案的结果，没有调整权重追求测试集高分。",
        "",
        "## 随语料规模变化的检索耗时",
        "",
        "同一批查询串行执行，每个规模预热 5 次；每条测量包含 query embedding、全量向量扫描、"
        "词法打分和段落 Top-10 排序。索引已驻留内存，没有数据库、HTTP 或外部模型。",
        "",
        "| 段落数 | chunk 数 | 测量请求数 | 重复轮数 | p50 (ms) | p95 (ms) | p99 (ms) | 串行 req/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["latency"]:
        lines.append(
            f"| {row['passages']:,} | {row['chunks']:,} | {row['requests']} | {row['repeats']} | "
            f"{row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {row['serial_queries_per_second']:.2f} |"
        )
    lines += [
        "",
        "## 结果边界与改进方向",
        "",
        "- 这是 MS MARCO QA validation 的候选段落池，**不是官方 880 万段落的 passage-ranking 全集**，也不是支付业务知识库。",
        "- `is_selected` 标注不穷尽全库中的相关段落，池化后的 Recall 只衡量找回已选段落；不能直接当作官方排名成绩或答案准确率。",
        "- SHA-256 向量不具备语义能力；低召回揭示离线回退的局限。语义检索提升需另接真实 embedding，并用独立测试集验证。",
        "- 缓存矩阵与倒排用于控制实验资源消耗；生产路径还有数据库读取、候选粗排和矩阵构建，因此这里的耗时不能作为 API p95、线上 SLA 或并发用户上限。",
        "- 未调用 LLM、未生成答案、未写入业务数据库；本实验不测业务收益、答案准确率、Celery/Redis 或 PostgreSQL 生产容量。",
        "",
        "## 复现",
        "",
        "```bash",
        "cd enterprise_knowledge_agent",
        "pip install -e '.[test,benchmark]'",
        "mkdir -p /tmp/enterprise_rag_datasets",
        f"curl --noproxy '*' --http1.1 -fL --retry 5 '{SOURCE_URL}' \\",
        "  -o /tmp/enterprise_rag_datasets/msmarco-v1.1-validation.parquet",
        "python evals/public_benchmark.py --queries 1000 --latency-queries 60 --repeats 3",
        "```",
        "",
        f"SHA-256：`{data['sha256']}`；脚本会校验文件完整性。原始排名、采样问题 ID、指标与失败例子见同目录 `public-benchmark-results.json`。",
        "",
        f"数据来源：[Hugging Face Microsoft MS MARCO]({SOURCE_URL})；"
        "[官方项目与使用条款](https://microsoft.github.io/msmarco/)。请按原数据条款使用；仓库只保存评测输出，原语料保存在临时目录。",
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
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--latency-queries", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run(args.dataset, args.queries, args.latency_queries, args.repeats)
    Path(__file__).with_name("public-benchmark-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = Path(__file__).with_name("public-benchmark-report.md")
    report.write_text(render_report(result), encoding="utf-8")
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
