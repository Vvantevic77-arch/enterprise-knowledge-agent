# 本轮优先级性能改进

本轮范围为 `enterprise_knowledge_agent` 实际业务应用。模板生成器未改造。

后续的可选语义模型、重排及质量对照见 [Recall 改进](recall-improvements.md)。本文 HTTP 成绩对应下述关闭外部模型、未启用重排的记录版本，不能直接套用到新模型策略。

## 实测结果

10,000 篇公开段落、10,012 个 chunk，1 个 Uvicorn worker，macOS 本机 TCP HTTP，
SQLite 保存相同的 1,536 维 JSON 向量；外部 embedding/LLM 关闭。
前后按相同的 100 个查询提示循环请求，使用唯一幂等键，检查非空回答和 1–5 条有效引用。
前后代码分别运行；测量期间不同时运行回归测试。客户端与服务共用电脑，系统后台负载仍可能影响结果。

| 同口径对比 | 改动前 | 改动后 |
| --- | ---: | ---: |
| 问答，16 并发，两轮 10 秒 p95 | 1655.1–1675.1 ms | 516.1–517.5 ms |
| 问答，16 并发，成功吞吐 | 38.79–41.03 req/s | 132.26–132.33 req/s |
| 问答，暂定门限下确认的并发档位 | 16 | 至少 32 |
| 知识库读取，32 并发，两轮 15 秒确认 p95 | 762.3–768.0 ms | 233.8–313.2 ms |
| 知识库读取，32 并发，确认窗口成功吞吐 | 98.57–100.15 req/s | 307.04–350.74 req/s |

16 并发问答的**两轮 p95 均值下降 68.96%**，两轮成功吞吐均值为原来的 **3.31 倍**。
这里比较的是每轮统计值的均值，不是把两轮所有请求合并后重算的总体 p95。
改动后 32 并发问答的两轮确认 p95 为 **491.6–525.9 ms**，
成功吞吐为 **154.23–154.90 req/s**，两轮错误率均为 0。
所有改动后问答阶梯及确认窗口均无请求错误，服务日志未出现数据库锁或连接池超时。
健康接口的一个 32 并发短窗口存在一次 ReadError（约 0.05%）；其确认窗口无错误。
不能把问答结果表述成“所有接口零错误”。

改动前问答在 32 并发有一轮超过 2 秒门限；16 并发及其确认窗口也有少量 HTTP 500，
但每轮错误率仍低于既定 1% 门限。改动后的 32 是本次测试上限，**不是最大并发用户数**。
每档重复 2×10 秒，最高通过档再确认 2×15 秒，至少 30 请求，预热 5 次，超时 15 秒。
暂定标准为读取 p95≤1 秒、离线问答 p95≤2 秒、每轮错误率≤1%；并非生产 SLA。
前后是顺序实验，未做随机交叉，也没有分别消融每个改动，不能把全部收益归因于单一技术。

- [改动前原始记录](../evals/http-priority-before-results.json) / [报告](../evals/http-priority-before-report.md)
- [改动后原始记录](../evals/http-priority-after-results.json) / [报告](../evals/http-priority-after-report.md)
- [输入一致性、计算口径与依赖版本](../evals/priority-comparison.json)

## 按优先级完成的改动

| 优先级 | 改动 | 核心文件 |
| --- | --- | --- |
| P0 | 认证/幂等预留后释放请求连接；检索使用独立短会话，模型等待不持有数据库连接；最终业务行与幂等响应仍原子提交 | [agent service](../backend/app/services/agent.py)、[idempotency](../backend/app/idempotency.py) |
| P0 | SQLite 启用 WAL，保留默认 FULL 持久化；索引创建、回填处于显式事务中；备份改用 SQLite backup API，包含 WAL 中的已提交数据 | [db](../backend/app/db.py)、[demo CLI](../backend/cli/seed_demo.py) |
| P1 | 复用 embedding 客户端，将同步 embedding、解析、向量评分和 broker 发布移到线程；设置 broker 连接/socket 超时并关闭发布重试；缓存整个无状态 Agent 图 | [rag](../backend/app/services/rag.py)、[ingestion](../backend/app/services/ingestion.py)、[worker config](../backend/app/worker/celery_app.py) |
| P1 | 英文 BM25 与中文双字倒排索引；“退款”类两字查询不再触发全库哈希向量扫描；更新/删除触发器与旧数据回填保持同步 | [search](../backend/app/services/agent.py)、[index lifecycle](../backend/app/db.py) |
| P1 | 真实 embedding 路径独立获取词法和密集候选，再用 RRF 融合，避免词法命中被向量候选门槛丢弃；PostgreSQL 添加 GIN 文本索引，兼容 ndarray 向量 | [agent service](../backend/app/services/agent.py)、[models](../backend/app/models.py) |
| P2 | 知识库统计合并到查询；用量 SUM 在数据库中完成；文档列表默认 50、上限 200，支持 skip；引用仅回查 ID/来源，避免反序列化无用向量 | [knowledge routes](../backend/app/routes/knowledge.py)、[ops](../backend/app/routes/ops.py)、[citations](../backend/app/routes/agent.py) |
| P2 | 新增 SSE token 流式回答，页面实时消费；支持鉴权续期、跨组织切换取消、UTF-8 跨包解码、模型流失败后的安全回退；旧 JSON/WS 接口保留 | [stream routes](../backend/app/routes/agent.py)、[frontend](../backend/app/static/app.js) |
| P2 | 有界事件缓存与一小时过期；Redis 读回放、TTL、故障退避及恢复探测；游标兼容环形缓冲裁剪与本地故障回退 | [events](../backend/app/services/events.py) |

中文触发器依赖每个连接注册的 `cjk_index_text` 函数，API 与 worker 统一使用 `app.db.engine`。
外部脚本若直接写 chunk，需要注册该函数。旧 trigram 派生索引被替换，源文档和 chunk 不删除。
文档哈希增量索引仍是文档级：改动文档会重写其全部 chunk，本轮没有宣称 chunk 级增量复用。

## 正确性与质量验证

- 最终全量 Python 回归 **205 passed**（2 条 Starlette 状态码弃用警告）；最后补充的 23 项关键链路回归也通过。
- 20 个慢模型同时等待，带/不带幂等键两种场景均验证数据库连接占用为 0。
- 原有同键并发只执行一次、参数冲突、失败重试、最终提交失败回滚、审批双人投票与租户隔离测试通过。
- ASGI 2.0 断开及 ASGI 2.4 socket 发送失败均验证生产者退出、pending 幂等预留释放、无半成品业务行。
- 前端 2 项 Node 测试通过：逐字节拆分中文 UTF-8/SSE、续期保持原幂等键、部分文本先于最终回答显示、HTML 转义及组织切换取消。
- PostgreSQL GIN 索引 SQL 编译通过；没有把该检查表述为实际 PostgreSQL 性能验证。
- 修改文件的 Ruff 检查通过。Ty 全库检查仍有 24 条与改动前相同的诊断（包括未挂载旧 `api.py` 的失效引用及已有类型兼容问题），未新增诊断；不宣称全库类型检查通过。

重新运行实际 `search_knowledge`：**81,039 文档 / 81,115 chunk / 原 1,000 查询**。
Recall@5 仍为 **57.53%**、MRR@5 为 **0.3500**，逐查询检索排名与原实际服务结果完全一致。
见 [本轮质量报告](../evals/sqlite-priority-report.md) 和 [逐查询记录](../evals/sqlite-priority-results.json)。
这是沿用的诊断验证集，非独立保留集；不是回答准确率，也不代表中文或真实语义 embedding 的质量。

## 复现

先进入 `enterprise_knowledge_agent`，使用本项目已安装依赖的 `.venv`。

```bash
.venv/bin/python -m pytest -q
node --test tests/frontend_stream.test.cjs
.venv/bin/python -m evals.sqlite_search_benchmark --output evals/sqlite-priority-results.json
.venv/bin/python -m evals.http_capacity --sizes 10000 --levels 1 8 16 32 --seconds 10 --repeats 2 --min-requests 30 --soak-seconds 15 --timeout 15 --warmup-requests 5 --output evals/http-priority-after-results.json
```

压测只创建自己的临时数据库与 Uvicorn 进程，结束后清理，不使用业务 `agent.db`，不调用付费模型。
数据下载和 SHA-256 校验要求见 [公开语料报告](../evals/public-benchmark-report.md)。
改动前源码已保存为 [priority-before.tar.gz](../evals/baselines/priority-before.tar.gz)，
解压后在快照目录中运行同样的 `evals.http_capacity` 命令，解释器仍使用本项目 `.venv/bin/python`。
源码档案不含 `.env`、数据库或缓存。两个版本的 Python 源码 SHA-256 与依赖版本保存在比较 JSON 中；
基线 journal mode 由冻结源码与全新 SQLite 文件的默认行为推断，已在原始记录中标明。

## 简历可用的表述

> 基于开源模板优化多租户知识库 Agent，通过检索短会话、WAL、图与客户端复用以及 SQL 聚合，
> 在本机单 worker、万篇语料、关闭外部模型的 HTTP 压测中，将 16 并发问答 p95 从约 1.67 秒降至 0.52 秒，
> 成功吞吐约提升至原来的 3.3 倍；在既定门限下确认至少 32 个并发请求，并保持公开验证集 Recall@5 为 57.53%。

只能在实际承担对应工作的前提下使用；保留“本机 / 外部模型关闭 / 既定门限”的条件。
真实模型 TTFT、真实 embedding 收益、PostgreSQL 容量、多 worker 长期混合负载仍需独立实测。
本轮没有进行生产部署；本地 API 重启、页面刷新后使用新功能。
