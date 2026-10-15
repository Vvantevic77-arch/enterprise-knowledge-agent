# 本地真实 HTTP 性能与容量

用户确认项目只在本机运行。本实验启动独立 Uvicorn 进程，通过 TCP HTTP 请求实际接口，使用隔离 SQLite；未接触现有用户数据库。环境与当前默认/Render 配置的 SQLite 检索路径相同，但不是线上服务器。

测量完成时间（UTC）：2026-10-15T01:44:49.828927+00:00。环境：macOS-15.6.1-arm64-arm-64bit、Python 3.12.14、8 个逻辑 CPU、1 个 Uvicorn worker；客户端与服务在同一电脑，SQLite 保存 1,536 维 JSON 向量，外部模型关闭。完整环境参数保存在原始 JSON。

## 方法与判定

每档 10.0 秒，至少 30 请求，重复 2 轮；最高通过档再各测 15.0 秒。预热 5 次完整问答。
低吞吐档会为满足最少请求数延长窗口。语料按文本哈希排序，取当前规模的前 N 条；每个规模循环使用最多 100 个原始自然语言问题，问题的标注正例至少一条存在于当前语料中。
本次检索路径：SQLite FTS5 BM25; Chinese no-hit queries retain vector fallback。幂等策略：committed pending reservation; no write lock during graph execution。

暂定容量标准：健康/知识库读取 p95≤1秒，离线 RAG 问答 p95≤2秒，每轮错误率≤1%。这只是本次明确设置的标准，未经业务方定义为正式 SLA。阶梯到首个失败档停止；边界介于通过档和失败档之间，没有测到边界则只给下界。

闭环并发表示同时发起且等待响应的请求数，零思考时间；客户端与服务在同一台电脑，包含 TCP、认证、数据库读取、问答检索、事务写入和引用读取。连接复用，超时15.0秒。每次问答使用唯一幂等键并检查 succeeded、非空答案、1–5条不重复的语料引用和非 replay。未验证答案事实准确率。

## 逐档实测

| 文档 / chunk | 接口 | 并发 | 轮次 | 请求 | p50 ms | p95 ms | p99 ms | 成功 req/s | 错误率 | 通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10,000 / 10,012 | GET /v1/health | 1 | 1 | 3070 | 3.0 | 4.8 | 7.8 | 306.93 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/health | 1 | 2 | 3377 | 2.7 | 4.2 | 5.4 | 337.67 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/health | 32 | 1 | 1900 | 115.1 | 489.4 | 794.1 | 188.07 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/health | 32 | 2 | 2321 | 93.8 | 415.1 | 689.9 | 229.26 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 1 | 1 | 886 | 10.0 | 20.5 | 29.7 | 88.52 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 1 | 2 | 913 | 10.1 | 18.2 | 25.4 | 91.24 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 8 | 1 | 1015 | 54.4 | 186.4 | 516.3 | 100.55 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 8 | 2 | 1013 | 52.2 | 198.2 | 517.0 | 100.22 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 16 | 1 | 1031 | 89.0 | 462.7 | 1206.7 | 101.48 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 16 | 2 | 1001 | 95.3 | 535.9 | 1254.7 | 98.72 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 32 | 1 | 1039 | 238.8 | 763.7 | 1277.2 | 101.84 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 32 | 2 | 1019 | 243.5 | 820.3 | 1306.8 | 98.01 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 1 | 1 | 369 | 23.5 | 45.9 | 81.6 | 36.89 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 1 | 2 | 390 | 23.8 | 40.8 | 59.6 | 38.99 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 8 | 1 | 409 | 98.1 | 762.0 | 1628.6 | 40.22 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 8 | 2 | 450 | 87.1 | 664.4 | 1303.5 | 44.39 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 16 | 1 | 409 | 157.8 | 1655.1 | 2631.4 | 38.79 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 16 | 2 | 423 | 165.4 | 1675.1 | 2419.4 | 41.03 | 0.24% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 32 | 1 | 429 | 531.6 | 2234.8 | 3319.5 | 39.82 | 0.00% | False |
| 10,000 / 10,012 | POST /v1/agent/runs | 32 | 2 | 412 | 583.0 | 1936.3 | 3563.6 | 38.40 | 0.00% | True |

## 容量边界与稳定窗口

| 文档数 | 接口 | 阶梯最高通过 | 稳定窗口确认 | 首个失败档 | 稳定窗口 p95 ms | 稳定窗口成功 req/s |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 10,000 | GET /v1/health | 32 | 32 | None | 444.4, 361.9 | 219.79, 260.21 |
| 10,000 | GET /v1/knowledge-bases | 32 | 32 | None | 768.0, 762.3 | 98.57, 100.15 |
| 10,000 | POST /v1/agent/runs | 16 | 16 | 32 | 1557.5, 1438.4 | 40.97, 44.81 |

0 表示没有档位通过本次延迟和错误率标准，不表示接口无法处理请求。未测到失败档只说明容量下界，不是最大并发；稳定窗口确认值可能低于阶梯最高通过值。健康检查只抽测两档，仅作参照，不能据此推断完整问答容量。

## 请求错误与服务日志

- 10,000 条段落，POST /v1/agent/runs，并发 16：{'HTTP 500': 1}，共 423 请求，错误率 0.24%。
- 10,000 条段落，POST /v1/agent/runs，并发 16：{'HTTP 500': 1}，共 632 请求，错误率 0.16%。
- 10,000 条段落，POST /v1/agent/runs，并发 16：{'HTTP 500': 1}，共 692 请求，错误率 0.14%。
- 10,000 条段落的服务日志错误摘要：{'sqlite3.OperationalError: database is locked': 6, 'sqlalchemy.exc.OperationalError: (raised as a result of Query-invoked autoflush; consider using a session.no_autoflush block if this flush is occurring prematurely)': 6, '(sqlite3.OperationalError) database is locked': 6}。同一异常可能出现多行，日志行数不能当作错误请求数；请求数取自上面的 HTTP 样本。

## 限制与瓶颈

- SQLite deterministic 路径使用 FTS5/BM25，只读取匹配 chunk，不再加载全库 JSON 向量；中文索引无命中时仍可能回退全量向量扫描，本次英文语料不能验证该回退路径性能。
- 幂等预留单独提交，完整问答期间不持有写锁；最终 run/approval/usage 与幂等结果一起提交。SQLite 的短写事务仍只有一个 writer，高并发仍可能排队。
- `require` 对通过权限校验的 GET 也添加 AuditLog；后续查询触发 SQLAlchemy autoflush 写入，而 GET 结束没有 commit、会话关闭会回滚。因此认证读取也不是纯读路径，可能竞争 SQLite 写锁；这是代码路径分析，未在本次单独量化其占比。
- 单 Uvicorn worker 内的同步 embedding、矩阵构建和分词还会占用事件循环。知识库读取容量和健康检查容量都不能替代完整问答容量。
- 关闭了外部 embedding/LLM，缺少网络模型延迟、配额、Redis/Celery、反向代理、多worker、生产硬件和长期混合负载；这些边界不能当作线上最大并发用户数。
- 语料直接批量写入隔离数据库，复用生产切分和向量算法，计数来自数据库；未测上传和入库任务吞吐。
- 客户端与服务器共享资源；原始逐请求数据、HTTP/超时分布及服务RSS/CPU采样保存在 JSON，短窗口 p99 和低样本档仅作描述。
- 81,039 条的全库检索质量与缓存计算已另测，本报告 API 容量只适用于表中的实际入库规模，不能外推全库服务性能。

## 复现

```bash
.venv/bin/python -m evals.http_capacity --sizes 10000 --levels 1 8 16 32 --seconds 10.0 --repeats 2 --min-requests 30 --soak-seconds 15.0 --timeout 15.0 --warmup-requests 5 --output /Users/strawverry/Documents/Codex/2026-09-04/ba/outputs/full-stack-ai-agent-template/enterprise_knowledge_agent/evals/http-priority-before-results.json
```

结束后停止自己的测试进程并删除自己的临时数据库；原有应用数据保留。
