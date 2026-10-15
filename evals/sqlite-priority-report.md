# SQLite 实际检索路径质量回归

81,039 文档、81,115 chunk，沿用原始基线的 1,000 个查询 ID 与全部标签。
实际调用当前 `search_knowledge`，不是离线 BM25 参考实现。隔离 SQLite 不保存向量，英文索引路径不读取向量；未测延迟、HTTP、中文检索、模型回答准确性或 PostgreSQL。

| 方法 | Recall@5 | MRR@5 |
| --- | ---: | ---: |
| lexical_only | 28.43% | 0.1847 |
| hybrid | 28.02% | 0.1890 |
| sqlite_fts | 57.53% | 0.3500 |

vs_lexical：Recall@5 差 +29.10 个百分点；配对 bootstrap 95% 区间 [+25.78, +32.37]。

vs_hybrid：Recall@5 差 +29.52 个百分点；配对 bootstrap 95% 区间 [+26.37, +32.72]。

该验证集沿用诊断样本，未进行新的调参；不属于独立保留测试集。全库标注不穷尽相关答案，Recall@5 不是回答准确率。实际 API 取 Top-5 chunk，再映射去重到文档；原始基线按文档排名，多 chunk 文档可能使两者有轻微差异。

复现：`.venv/bin/python -m evals.sqlite_search_benchmark`。语料 SHA-256、全部查询 ID、标签和排名保存在对应 JSON。
