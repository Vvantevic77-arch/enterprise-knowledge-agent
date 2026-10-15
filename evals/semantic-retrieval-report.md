# 语义召回与重排对照

English MS MARCO QA validation pool; incomplete relevance labels; frozen non-overlapping validation query subsets, not an independent dataset or Chinese evaluation; the 500-query subset was excluded from prior full-corpus diagnostics, but historical ten-candidate aggregate metrics covered the whole validation split; exact cached dense is not ANN/HTTP latency; union rerank is shared across ablations; query encoding excluded from per-query timings

语料：81,039 文档 / 81,115 chunk。最终固定返回 5 个 chunk。

## diagnostic（1000 查询）

| 方法 | Recall@5 | Recall@20 | Recall@50 | Recall@100 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 57.53% | 81.63% | 88.10% | 90.62% | 0.3500 |
| dense | 78.28% | 96.10% | 98.70% | 99.20% | 0.5048 |
| hybrid | 71.28% | 92.82% | 98.70% | — | 0.4518 |
| bm25_rerank | 79.53% | 87.90% | 88.10% | — | 0.5566 |
| hybrid_rerank | 86.70% | 97.85% | 98.70% | — | 0.5845 |

候选诊断：`{"bm25": {"no_positive_at_5": 404, "positive_only_at_6_to_50": 294, "no_positive_at_50": 110}, "dense": {"no_positive_at_5": 200, "positive_only_at_6_to_50": 188, "no_positive_at_50": 12}, "hybrid": {"no_positive_at_5": 266, "positive_only_at_6_to_50": 254, "no_positive_at_50": 12}}`

- dense 相对 BM25：+20.75 个百分点；配对 bootstrap 95% 区间 [+17.47, +23.93]。
- hybrid 相对 BM25：+13.74 个百分点；配对 bootstrap 95% 区间 [+11.23, +16.18]。
- bm25_rerank 相对 BM25：+22.00 个百分点；配对 bootstrap 95% 区间 [+19.27, +24.93]。
- hybrid_rerank 相对 BM25：+29.17 个百分点；配对 bootstrap 95% 区间 [+26.10, +32.28]。
## holdout（500 查询）

| 方法 | Recall@5 | Recall@20 | Recall@50 | Recall@100 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 59.27% | 86.00% | 90.60% | 93.40% | 0.3663 |
| dense | 77.60% | 95.63% | 98.23% | 98.90% | 0.5088 |
| hybrid | 73.90% | 94.63% | 97.90% | — | 0.4681 |
| bm25_rerank | 81.40% | 90.60% | 90.60% | — | 0.5842 |
| hybrid_rerank | 84.97% | 97.50% | 97.90% | — | 0.5986 |

候选诊断：`{"bm25": {"no_positive_at_5": 191, "positive_only_at_6_to_50": 148, "no_positive_at_50": 43}, "dense": {"no_positive_at_5": 103, "positive_only_at_6_to_50": 96, "no_positive_at_50": 7}, "hybrid": {"no_positive_at_5": 120, "positive_only_at_6_to_50": 111, "no_positive_at_50": 9}}`

- dense 相对 BM25：+18.33 个百分点；配对 bootstrap 95% 区间 [+13.90, +22.97]。
- hybrid 相对 BM25：+14.63 个百分点；配对 bootstrap 95% 区间 [+11.23, +18.10]。
- bm25_rerank 相对 BM25：+22.13 个百分点；配对 bootstrap 95% 区间 [+17.97, +26.40]。
- hybrid_rerank 相对 BM25：+25.70 个百分点；配对 bootstrap 95% 区间 [+21.37, +30.23]。

模型固定版本：

- embedding：`sentence-transformers/multi-qa-MiniLM-L6-cos-v1@b207367332321f8e44f96e224ef15bc607f4dbf0`
- reranker：`cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`

组件延迟保存在 JSON；不含在线 query 编码、HTTP、模型生成。实际 SQLite 语义路径仍读取 JSON 向量；生产大语料需单独测 pgvector 与租户过滤后的 ANN 召回。
