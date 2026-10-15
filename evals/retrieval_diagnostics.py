"""Controlled lexical ablations and coarse-retrieval diagnostics on MS MARCO.

This changes no production scorer. BM25 uses fixed conventional parameters,
the same chunks/token boundaries and the same queries/labels as the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from evals.public_benchmark import (
    EXPECTED_SHA256,
    SEED,
    PublicIndex,
    lexical_terms,
    prepare_rows,
    summarize,
)

# Fixed common English function words; no dataset-specific tuning or stemming.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]  # noqa: SIM905
)
REFERENCE_MODES = ("lexical_no_stopwords", "idf_overlap", "bm25", "bm25_no_stopwords")


class LexicalReference:
    def __init__(self, chunks: list[str]):
        postings = defaultdict(list)
        self.lengths = np.zeros(len(chunks), dtype=np.float32)
        for index, text in enumerate(chunks):
            counts = Counter(re.findall(r"\w+", text.lower()))
            self.lengths[index] = sum(counts.values())
            for term, count in counts.items():
                postings[term].append((index, count))
        self.postings = {}
        self.idf = {}
        for term, entries in postings.items():
            array = np.asarray(entries, dtype=np.int32)
            self.postings[term] = (array[:, 0], array[:, 1].astype(np.float32))
            self.idf[term] = float(
                np.log1p((len(chunks) - len(entries) + 0.5) / (len(entries) + 0.5))
            )
        self.length_norm = 1.2 * (0.25 + 0.75 * self.lengths / max(1, self.lengths.mean()))

    def scores(self, query: str) -> dict[str, np.ndarray]:
        result = {mode: np.zeros(len(self.lengths), dtype=np.float32) for mode in REFERENCE_MODES}
        terms = lexical_terms(query)
        for term in terms:
            if term not in self.postings:
                continue
            indices, frequencies = self.postings[term]
            idf = self.idf[term]
            result["idf_overlap"][indices] += idf
            contribution = idf * frequencies * 2.2 / (frequencies + self.length_norm[indices])
            result["bm25"][indices] += contribution
            if term not in STOPWORDS:
                result["lexical_no_stopwords"][indices] += 1
                result["bm25_no_stopwords"][indices] += contribution
        result["lexical_no_stopwords"] /= max(1, len(terms - STOPWORDS))
        return result


def paired_delta(records: list[dict], left: str, right: str) -> dict:
    def recall(record, mode):
        return len(set(record[mode][:5]) & set(record["relevant"])) / len(record["relevant"])

    differences = np.asarray([recall(r, left) - recall(r, right) for r in records])
    rng = np.random.default_rng(SEED)
    bootstrap = rng.choice(differences, size=(1000, len(records))).mean(axis=1)
    return {
        "recall_at_5_delta_percentage_points": float(differences.mean() * 100),
        "paired_bootstrap_95_ci_percentage_points": (
            np.percentile(bootstrap, [2.5, 97.5]) * 100
        ).tolist(),
    }


def run(dataset: Path, baseline_path: Path) -> dict:
    import pyarrow.parquet as pq

    if hashlib.sha256(dataset.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch")
    baseline = json.loads(baseline_path.read_text())
    documents, queries = prepare_rows(pq.read_table(dataset).to_pylist())
    by_id = {query.id: query for query in queries}
    sampled = [by_id[row["query_id"]] for row in baseline["pooled_rankings"]]
    index = PublicIndex(documents, progress=True)
    reference = LexicalReference(index.chunks)
    records, details, examples = [], [], []
    example_ids = {row["query_id"] for row in baseline["failure_examples"]}
    for number, (query, original) in enumerate(
        zip(sampled, baseline["pooled_rankings"], strict=True), 1
    ):
        scores = index.scores(query.text, len(documents))
        rankings = {mode: index.rank(values, len(documents)) for mode, values in scores.items()}
        # Independently reproduce the prior measurement before drawing causes.
        assert all(rankings[mode] == original[mode] for mode in rankings), query.id
        scores.update(reference.scores(query.text))
        rankings.update(
            {mode: index.rank(scores[mode], len(documents)) for mode in REFERENCE_MODES}
        )
        coarse_chunks = np.argsort(-scores["vector_only"], kind="stable")[:20]
        coarse = np.searchsorted(index.starts, coarse_chunks, side="right") - 1
        order = np.argsort(-scores["hybrid"][coarse_chunks], kind="stable")
        rankings["vector_top20_then_hybrid"] = coarse[order][:10].tolist()
        record = {"query_id": query.id, "relevant": sorted(query.relevant), **rankings}
        records.append(record)
        lexical_documents = np.maximum.reduceat(scores["lexical_only"], index.starts[:-1])
        best = max(float(lexical_documents[doc]) for doc in query.relevant)
        higher = int(np.count_nonzero(lexical_documents > best))
        ties = int(np.count_nonzero(lexical_documents == best))
        positive_terms = set().union(*(lexical_terms(documents[doc]) for doc in query.relevant))
        terms = lexical_terms(query.text)
        details.append(
            {
                "query_id": query.id,
                "query": query.text,
                "lexical_hit_at_5": bool(set(rankings["lexical_only"][:5]) & query.relevant),
                "best_positive_lexical_score": best,
                "strictly_higher_scoring_passages": higher,
                "passages_tied_at_best_positive_score": ties,
                "best_possible_positive_rank": higher + 1,
                "missing_query_terms": sorted(terms - positive_terms),
                "missing_content_terms": sorted(terms - positive_terms - STOPWORDS),
                "stopword_fraction": len(terms & STOPWORDS) / max(1, len(terms)),
                "vector_top20_recall": len(set(coarse.tolist()) & query.relevant)
                / len(query.relevant),
            }
        )
        if query.id in example_ids:
            rank_all = np.argsort(-lexical_documents, kind="stable").tolist()
            examples.append(
                {
                    **details[-1],
                    "selected_passage": documents[min(query.relevant)],
                    "best_positive_lexical_rank": min(
                        rank_all.index(doc) + 1 for doc in query.relevant
                    ),
                    "top_lexical_passage": documents[rankings["lexical_only"][0]],
                    "top_bm25_passage": documents[rankings["bm25"][0]],
                    "bm25_hit_at_5": bool(set(rankings["bm25"][:5]) & query.relevant),
                }
            )
        if number % 100 == 0:
            print(f"Diagnostics {number}/{len(sampled)}", flush=True)
    misses = [row for row in details if not row["lexical_hit_at_5"]]
    summary = {
        "queries": len(details),
        "lexical_missed_queries": len(misses),
        "misses_with_at_least_5_strictly_better_passages": sum(
            row["strictly_higher_scoring_passages"] >= 5 for row in misses
        ),
        "misses_only_due_to_tie_order_at_top5_boundary": sum(
            row["strictly_higher_scoring_passages"] < 5 for row in misses
        ),
        "misses_with_missing_non_stopword_terms": sum(
            bool(row["missing_content_terms"]) for row in misses
        ),
        "queries_with_any_missing_non_stopword_terms": sum(
            bool(row["missing_content_terms"]) for row in details
        ),
        "median_ties_at_best_positive_score": float(
            np.median([r["passages_tied_at_best_positive_score"] for r in details])
        ),
        "median_ties_among_misses": float(
            np.median([r["passages_tied_at_best_positive_score"] for r in misses])
        ),
        "mean_query_stopword_fraction": float(np.mean([r["stopword_fraction"] for r in details])),
        "vector_top20_recall_ceiling": float(np.mean([r["vector_top20_recall"] for r in details])),
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": baseline["dataset"],
        "sampling": baseline["sampling"],
        "settings": {
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "stemming": False,
            "stopwords": sorted(STOPWORDS),
            "chunking": "900 characters, 120 overlap",
            "production_modified": False,
        },
        "summary": summary,
        "metrics": {
            mode: summarize(records, mode)
            for mode in records[0]
            if mode not in {"query_id", "relevant"}
        },
        "paired_comparisons": {
            "hybrid_minus_lexical": paired_delta(records, "hybrid", "lexical_only"),
            "bm25_minus_lexical": paired_delta(records, "bm25", "lexical_only"),
        },
        "failure_examples": examples,
        "query_diagnostics": details,
        "rankings": records,
    }


def render_report(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# 低召回原因诊断",
        "",
        "在同一 81,039 段落池、同一 1,000 个随机问题、同一 chunk 和标签下做对照，先逐条复现原始排名。只新增离线对照，未改生产检索。",
        "",
        "| 方法 | Recall@5 | Recall@10 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode, metrics in result["metrics"].items():
        lines.append(
            f"| {mode} | {metrics['recall_at_5']:.2%} | {metrics['recall_at_10']:.2%} | {metrics['mrr_at_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据与原因",
            "",
            "1. **向量无语义。** SHA-256 摘要只是伪随机数字；重复到 1,536 维仍只有 32 个不同坐标。语义相近、单复数相近都不会接近。混合权重中的 15% 因此是噪声。",
            f"2. **词法评分太粗。** 每个 distinct query term 权重相同，不计 IDF、词频、文档长度；平均 {summary['mean_query_stopword_fraction']:.2%} 的查询词是固定列表中的常见功能词。正例分数处同分段落数中位数为 {summary['median_ties_at_best_positive_score']:,.0f}。",
            f"3. **失败不只来自同分排序。** 纯词法完全未命中的 {summary['lexical_missed_queries']} 个问题中，{summary['misses_with_at_least_5_strictly_better_passages']} 个已有至少 5 条段落严格高于最好正例，{summary['misses_only_due_to_tie_order_at_top5_boundary']} 个才属于 Top-5 边界同分排序问题；两者互斥。",
            f"4. **词形和改写失配。** {summary['queries_with_any_missing_non_stopword_terms']} / {summary['queries']} 个问题的正例缺少至少一个非停用词的精确 token；完全未命中的问题中有 {summary['misses_with_missing_non_stopword_terms']} 个。这个统计包含词形、同义改写和噪声词，不能全部归为词形问题。",
            f"5. **PostgreSQL 还会先丢候选。** 生产 pgvector 路径只取向量 Top-20 chunk，再混合排序。使用相同伪向量的精确 Top-20 chunk 对照，正例召回上限只有 {summary['vector_top20_recall_ceiling']:.2%}。这是粗排机制的离线诊断，不是实际 ANN 数据库测量。SQLite 路径没有这步裁剪。",
            "6. **标注不穷尽相关答案。** 此数据是 QA validation 的原问题候选标注，将候选合并到全库后，另一条能答题的段落可能未标成正例。指标衡量找回已选段落，不等于回答准确率；没有额外人工判断就无法量化这部分偏差。",
            "7. **原小样本任务容易。** 240篇合成基准的查询包含唯一参考键，纯词法100%只说明精确查键成功。换成自然问题并在81,039条段落中检索，词形、改写、同分和泛化词干扰同时出现；这不是同一个难度的测试，不能把下降直接解释为代码回归。",
            "",
            "## 固定参数的对照结果",
            "",
            "BM25 k1=1.2、b=0.75，仍用精确 token、没有词干化、同义词或真实 embedding；参数没有按这个验证集调优。IDF、去停用词和 BM25 分别用于定位原因，不能视为已经上线的提升。",
            "",
        ]
    )
    for name, values in result["paired_comparisons"].items():
        ci = values["paired_bootstrap_95_ci_percentage_points"]
        lines.append(
            f"- {name}: Recall@5 差 {values['recall_at_5_delta_percentage_points']:+.2f} 个百分点，配对 bootstrap 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}]（1,000 次重采样）。"
        )
    lines.extend(["", "## 失败例子", ""])
    for row in result["failure_examples"]:
        lines.extend(
            [
                f"### {row['query']}",
                "",
                f"正例纯词法排名 {row['best_positive_lexical_rank']:,}，严格更高分段落 {row['strictly_higher_scoring_passages']:,}，同分段落 {row['passages_tied_at_best_positive_score']:,}；缺失关键词：{row['missing_content_terms']}。BM25 Top-5 命中：{row['bm25_hit_at_5']}。",
                "",
                f"> 已选正例：{row['selected_passage']}",
                "",
                f"> 词法第一名：{row['top_lexical_passage']}",
                "",
                f"> BM25 第一名：{row['top_bm25_passage']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步",
            "",
            "先接真实语义 embedding，再以 BM25 与向量独立召回合并候选，最后重排；避免只靠伪向量截断候选。中文业务问题另用真实支付语料做独立评测。若实施改动，应在未用于调优的保留测试集上复验召回、答案正确性和性能。",
            "",
            "复现：`PYTHONPATH=. .venv/bin/python -m evals.retrieval_diagnostics`。原始排名、逐查询诊断和完整条件保存在 `retrieval-diagnostics-results.json`。",
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
        "--baseline", type=Path, default=Path(__file__).with_name("public-benchmark-results.json")
    )
    args = parser.parse_args()
    result = run(args.dataset, args.baseline)
    Path(__file__).with_name("retrieval-diagnostics-results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    report = Path(__file__).with_name("retrieval-diagnostics-report.md")
    report.write_text(render_report(result))
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
