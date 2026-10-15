# MS MARCO 公开数据检索实验

实验时间（UTC）：2026-10-12T06:35:14.598357+00:00

使用 **81,039 条真实网页段落**（去重前 82,360 条），来自 **10,047 个真实问题**；其中 9,706 个有正例，341 个无正例的问题不计入质量指标。规模为原 240 篇合成文档的 **337.66 倍**。

- 原文：34.29 MB；按 900 字符、120 重叠切分为 **81,115 个 chunk**。
- 本地建索引：**17.39 s**（切分、确定性向量、词法倒排；不含下载、Parquet 解析、数据库写入）。
- 向量矩阵：1,536 维 float32，475.28 MiB；不含文本和倒排索引内存。
- 复用 `app.services.rag` 的切分、向量、分词和权重：0.15 × 向量 + 0.85 × 词法重叠。
- 环境：{'python': '3.12.14', 'platform': 'macOS-15.6.1-arm64-arm-64bit', 'machine': 'arm64', 'numpy': '2.5.3', 'pyarrow': '26.0.0', 'cpu_count': 8}

## 检索质量

候选排序对每个问题只排序数据集给定的候选段落；全库检索把所有去重段落放入同一个池，按固定种子 20261012 从有正例的问题中随机抽取 1,000 个。语料 ID 按文本 SHA-256 排序，独立于原候选顺序；候选按该 ID 排序后打分，避免原候选顺序泄漏。段落有多个 chunk 时取最高分，Top-K 去重到段落。

| 场景 | 方法 | 问题数 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 候选排序 | vector_only | 9,706 | 12.80% | 62.88% | 100.00% | 0.3592 |
| 候选排序 | lexical_only | 9,706 | 19.65% | 74.27% | 100.00% | 0.4364 |
| 候选排序 | hybrid | 9,706 | 19.75% | 74.68% | 100.00% | 0.4378 |
| 全库检索 | vector_only | 1,000 | 0.00% | 0.00% | 0.00% | 0.0000 |
| 全库检索 | lexical_only | 1,000 | 11.27% | 28.43% | 35.70% | 0.1945 |
| 全库检索 | hybrid | 1,000 | 11.70% | 28.02% | 35.60% | 0.1994 |

候选集每题最多 10 条，因此候选排序的 Recall@10=100% 是候选全集覆盖的必然结果，不能作为检索能力证明。Recall@K 是逐问题计算“前 K 条命中的已标注正例数 / 该问题已标注正例数”后取宏平均；MRR@10 只计算前 10 条内第一个正例的倒数排名。命中率另存于 JSON，避免与 Recall 混淆。
混合检索的全库 Recall@5 bootstrap 95% 区间：**25.42%–30.73%**（1,000 次重采样）；相对纯词法的差值为 **-0.42 个百分点**。这是现有回退方案的结果，没有调整权重追求测试集高分。

## 随语料规模变化的检索耗时

同一批查询串行执行，每个规模预热 5 次；每条测量包含 query embedding、全量向量扫描、词法打分和段落 Top-10 排序。索引已驻留内存，没有数据库、HTTP 或外部模型。

| 段落数 | chunk 数 | 测量请求数 | 重复轮数 | p50 (ms) | p95 (ms) | p99 (ms) | 串行 req/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 10,012 | 180 | 3 | 2.185 | 2.605 | 2.765 | 450.03 |
| 50,000 | 50,046 | 180 | 3 | 10.719 | 12.695 | 13.876 | 91.02 |
| 81,039 | 81,115 | 180 | 3 | 18.314 | 21.563 | 51.261 | 50.84 |

## 结果边界与改进方向

- 这是 MS MARCO QA validation 的候选段落池，**不是官方 880 万段落的 passage-ranking 全集**，也不是支付业务知识库。
- `is_selected` 标注不穷尽全库中的相关段落，池化后的 Recall 只衡量找回已选段落；不能直接当作官方排名成绩或答案准确率。
- SHA-256 向量不具备语义能力；低召回揭示离线回退的局限。语义检索提升需另接真实 embedding，并用独立测试集验证。
- 缓存矩阵与倒排用于控制实验资源消耗；生产路径还有数据库读取、候选粗排和矩阵构建，因此这里的耗时不能作为 API p95、线上 SLA 或并发用户上限。
- 未调用 LLM、未生成答案、未写入业务数据库；本实验不测业务收益、答案准确率、Celery/Redis 或 PostgreSQL 生产容量。

## 复现

```bash
cd enterprise_knowledge_agent
pip install -e '.[test,benchmark]'
mkdir -p /tmp/enterprise_rag_datasets
curl --noproxy '*' --http1.1 -fL --retry 5 'https://huggingface.co/datasets/microsoft/ms_marco/resolve/a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a/v1.1/validation-00000-of-00001.parquet?download=true' \
  -o /tmp/enterprise_rag_datasets/msmarco-v1.1-validation.parquet
python evals/public_benchmark.py --queries 1000 --latency-queries 60 --repeats 3
```

SHA-256：`67aca2e1b2fffe418a6bd317142887c90973c9c98dd8d940eaae3f8521daed7c`；脚本会校验文件完整性。原始排名、采样问题 ID、指标与失败例子见同目录 `public-benchmark-results.json`。

数据来源：[Hugging Face Microsoft MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco/resolve/a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a/v1.1/validation-00000-of-00001.parquet?download=true)；[官方项目与使用条款](https://microsoft.github.io/msmarco/)。请按原数据条款使用；仓库只保存评测输出，原语料保存在临时目录。
