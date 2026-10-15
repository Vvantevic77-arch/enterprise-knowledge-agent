# Recall 改进与启用

本轮先做候选诊断，再固定模型、候选数量、融合参数和不重叠查询集进行对照。
默认离线检索继续可用；真实语义模型、重排通过配置启用，未修改现有 `.env`、业务库或正在运行的服务。

## 实测质量结果

同一全库、固定 Top-5 chunk，不调用付费 API，使用本地开源模型：

| 方案 | 原 1,000 查询 Recall@5 | 新增 500 查询 Recall@5 |
| --- | ---: | ---: |
| BM25 | 57.53% | 59.27% |
| 纯语义 | 78.28% | 77.60% |
| RRF 混合，无重排 | 71.28% | 73.90% |
| BM25 + 重排 | 79.53% | 81.40% |
| 混合召回 + 重排 | **86.70%** | **84.97%** |

混合加重排相对同组 BM25：原查询 **+29.17 个百分点**，新增查询 **+25.70 个百分点**；
配对 bootstrap 95% 区间分别为 [+26.10, +32.28]、[+21.37, +30.23]。
原查询 MRR@5 从 0.3500 提高到 0.5845，新增查询从 0.3663 提高到 0.5986。
原 1,000 查询的 BM25 Top-100 与独立运行的实际服务候选诊断逐条完全相同。

结果也说明不能只堆组件：未重排的 RRF 混合在两组上都低于纯语义；本轮没有根据结果调整融合权重。
部署顺序建议先验证不需重建向量的 BM25 加重排，再评估真实语义索引；新增查询上后者相对 BM25 加重排多约 3.57 个百分点，同时需要全库编码与向量索引运维。

完整证据：[五组对照报告](../evals/semantic-retrieval-report.md)、[逐查询排名及配置](../evals/semantic-retrieval-results.json)。
测量使用本机 Apple Silicon MPS。新增查询中，热 BM25 组件 P95 约 76.8 ms、缓存全库精确向量检索约 11.1 ms、两策略候选并集共享重排约 187.0 ms。
这些是串行组件时间，不能相加当作 API P95；不含在线 query 编码、网络、生成和模型冷启动，也不能替代 PostgreSQL ANN 或并发 HTTP 测量。

## 已确认的瓶颈

实际 SQLite `search_knowledge`，同一 81,039 文档 / 81,115 chunk / 原 1,000 查询：

| 指标 | BM25 |
| --- | ---: |
| Recall@5 | 57.53% |
| Recall@20 | 81.63% |
| Recall@50 | 88.10% |
| Recall@100 | 90.62% |

前 5 完全没有正例的 404 个查询中，294 个在第 6–50 位存在正例，另外 110 个前 50 仍没有正例。
因此先验证重排：它有机会修复前者，无法找回后者。88.10% 是这个 BM25 候选集的覆盖率，不是承诺可达到的最终 Recall@5。
这里的计数按“至少一个正例是否出现”，Recall 指标则按全部已标注正例的比例计算，不能直接相减换算。

证据：[候选诊断](../evals/recall-candidates-report.md)、[逐查询排名](../evals/recall-candidates-results.json)。

## 实现范围

- 可选本地 `sentence_transformers` embedding；可继续使用已有兼容 API 的 embedding。
- BM25 和语义候选各自独立检索，默认各取 50，以 RRF（常数 60）融合，再取 50 进行可选 Cross-Encoder 重排，最终仍返回 5 个 chunk。
- 也支持只对 BM25 候选重排，无需更换现有 embedding 或重建词法索引。
- 模型推理在工作线程执行；Agent 图的查询编码和重排期间不占用读取连接。
- 新向量记录 provider、model、revision、维度和网关的指纹。真实向量搜索只读取与当前配置匹配的向量，避免混合模型或误用旧哈希向量。
- 相同正文只有在 embedding 指纹、切块参数也相同时才跳过写入；更换配置后重新同步会重建该文档的向量。
- provider 输出检查数量、维度、有限值与零向量；校验失败发生在替换旧 chunk 之前。

默认重排关闭，普通安装不需要 Torch；安装 `retrieval` extra 才能启用本地模型。
模型首次加载有下载和初始化开销，评测中的热组件时间不包含这部分开销。

## 固定对照实验

[实验清单](../evals/recall-experiment-manifest.json) 在读取对照结果前固定：

- 原诊断集 1,000 查询，便于与 57.53% 对比。
- 另取 500 个不重叠、有标注的查询作为本轮全库检索保留查询。它们仍来自同一公开 validation split 与同一语料池，不是新的独立数据集；历史十候选内排序的汇总评测覆盖过整个 split，因此也不是严格意义上从未接触的测试数据。
- 不使用问题原有的十条候选或正例标签建立召回候选；所有检索面向完整 81,115 个 chunk。
- 语义模型 `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`，384 维；重排模型 `cross-encoder/ms-marco-MiniLM-L6-v2`；两者固定到清单中的 commit。
- 五组实验：BM25、纯语义、RRF 混合、BM25 加重排、混合加重排；不按结果调参数。
- 同一文档的重复 chunk 保留各自排名位置，与 API 的 Top-5 chunk 口径一致。
- 离线语义搜索对缓存矩阵做全库精确余弦检索，融合及重排复用服务函数；它不是实际 pgvector ANN 或 HTTP 压测。

模型和语料包含 MS MARCO 训练/验证体系，不能据此宣称跨领域泛化；标签没有穷尽全库相关答案。
这两个小模型是英文基准对照，不适合据此宣称中文支付知识库已提升。中文需要独立标注集和多语言模型对照。

进入 `enterprise_knowledge_agent` 后复现：

```bash
uv pip install --python .venv/bin/python -e '.[test,benchmark,retrieval]'
.venv/bin/python -m evals.sqlite_search_benchmark --candidate-k 100 --output evals/recall-candidates-results.json
.venv/bin/python -m evals.semantic_retrieval --prepare-only
TOKENIZERS_PARALLELISM=false .venv/bin/python -m evals.semantic_retrieval --device cpu
```

Apple Silicon 可使用 `--device mps`。模型缓存为 `.cache/huggingface`，语料矩阵缓存为 `.cache/semantic-retrieval`；两者已忽略，不应提交。
已完整下载模型后，可加 `HF_HUB_OFFLINE=1` 保证只读本地权重。JSON 记录固定版本、权重 SHA、软件版本、查询排名与配对 bootstrap 区间。
对照实验的重排时间是两个候选集并集共享一次推理的时间，不等于单个上线策略的耗时；查询向量编码预先缓存，也未计入每查询时间。

## 启用方式

英文库先尝试影响较小的 BM25 加重排，在 `.env` 中设置：

```dotenv
EMBEDDING_PROVIDER=deterministic
RETRIEVAL_CANDIDATE_K=50
RETRIEVAL_DEVICE=cpu
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
RERANKER_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a
```

服务进程通过 `HF_HOME` 指向已下载缓存，例如从应用目录启动时设为 `$PWD/.cache/huggingface`。
重启 API 后生效；清空 `RERANKER_MODEL` 并重启即可退回原 BM25 排名。这个配置仅针对英文基准，不应直接作为中文业务默认值。

要启用语义混合，再设置：

```dotenv
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
EMBEDDING_REVISION=b207367332321f8e44f96e224ef15bc607f4dbf0
EMBEDDING_DIMENSIONS=384
```

先在隔离库中导入原始文档验证。切换 provider/model/revision 后必须重新同步或导入原始文档；旧无指纹或不同模型的向量会被排除，词法召回仍可用。
PostgreSQL 已有 `vector(1536)` 列不能只改环境变量为 384；需要经验证的维度迁移或独立新库，当前代码不会自动改列或删除原库。
SQLite 真实语义路径仍读取匹配的 JSON 向量，不适合以本轮精确矩阵实验代替大规模线上性能证明。大语料生产部署还需单独验证 pgvector、租户过滤后的 ANN 召回和完整 HTTP P95。

## 验证边界

单元/回归覆盖配置边界、向量校验、模型变更后重建、错误批次保留旧索引、租户隔离、事件循环与连接释放、融合去重和评测集冻结。
全量 Python 回归 **215 passed**，修改文件 Ruff 检查通过；Ty 全库仍有 23 条诊断，不宣称全库类型检查通过。
另用真实本地模型、临时 SQLite 和两个英文样例跑通实际 Agent 检索节点，验证语义查询、融合、重排、其他租户及旧模型向量排除、连接释放；[冒烟记录](../evals/local-retrieval-smoke-results.json) 不作为质量基准。
质量结果只衡量找回已标注段落，不是生成答案准确率或线上 SLA。原先不带重排的 HTTP 性能数字不能直接套用到新策略。

## 简历表述

> 实现 BM25、语义向量独立召回、RRF 融合与 Cross-Encoder 重排，在 8.1 万篇公开英文语料上，将固定 1,000 查询的 Recall@5 从 57.53% 提升至 86.70%；另 500 条不重叠验证查询从 59.27% 提升至 84.97%，配套模型版本隔离、候选召回诊断与可复现消融实验。

需同时说明这是离线公开英文检索评测，不能改写为中文线上回答准确率或独立测试集成绩。
