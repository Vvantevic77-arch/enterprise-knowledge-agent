# 本地真实 HTTP 性能与容量

用户确认项目只在本机运行。本实验启动独立 Uvicorn 进程，通过 TCP HTTP 请求实际接口，使用隔离 SQLite；未接触现有用户数据库。环境与当前默认/Render 配置的 SQLite 检索路径相同，但不是线上服务器。

测量完成时间（UTC）：2026-10-14T03:51:07.378277+00:00。环境：macOS-15.6.1-arm64-arm-64bit、Python 3.12.14、8 个逻辑 CPU、1 个 Uvicorn worker；客户端与服务在同一电脑，SQLite 保存 1,536 维 JSON 向量，外部模型关闭。完整环境参数保存在原始 JSON。

## 方法与判定

每档 10.0 秒，至少 30 请求，重复 2 轮；最高通过档再各测 30.0 秒。预热 5 次完整问答。
低吞吐档会为满足最少请求数延长窗口。语料按文本哈希排序，取当前规模的前 N 条；每个规模循环使用最多 100 个原始自然语言问题，问题的标注正例至少一条存在于当前语料中。
本次检索路径：SQLite FTS5 BM25; Chinese no-hit queries retain vector fallback。幂等策略：committed pending reservation; no write lock during graph execution。

暂定容量标准：健康/知识库读取 p95≤1秒，离线 RAG 问答 p95≤2秒，每轮错误率≤1%。这只是本次明确设置的标准，未经业务方定义为正式 SLA。阶梯到首个失败档停止；边界介于通过档和失败档之间，没有测到边界则只给下界。

闭环并发表示同时发起且等待响应的请求数，零思考时间；客户端与服务在同一台电脑，包含 TCP、认证、数据库读取、问答检索、事务写入和引用读取。连接复用，超时15.0秒。每次问答使用唯一幂等键并检查 succeeded、非空答案、1–5条不重复的语料引用和非 replay。未验证答案事实准确率。

## 逐档实测

| 文档 / chunk | 接口 | 并发 | 轮次 | 请求 | p50 ms | p95 ms | p99 ms | 成功 req/s | 错误率 | 通过 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10,000 / 10,012 | GET /v1/health | 1 | 1 | 4077 | 2.4 | 2.7 | 3.0 | 407.61 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/health | 1 | 2 | 4099 | 2.4 | 2.7 | 2.9 | 409.82 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/health | 64 | 1 | 1360 | 287.5 | 1494.3 | 2256.9 | 130.17 | 0.00% | False |
| 10,000 / 10,012 | GET /v1/health | 64 | 2 | 1583 | 234.9 | 1385.0 | 2410.5 | 151.40 | 0.00% | False |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 1 | 1 | 1168 | 8.5 | 9.2 | 10.9 | 116.78 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 1 | 2 | 1162 | 8.5 | 9.3 | 10.7 | 116.11 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 2 | 1 | 1552 | 11.0 | 21.0 | 34.2 | 154.75 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 2 | 2 | 1539 | 11.0 | 20.9 | 34.0 | 153.78 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 4 | 1 | 1551 | 22.0 | 53.3 | 99.9 | 154.59 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 4 | 2 | 1544 | 21.4 | 57.3 | 96.0 | 154.07 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 8 | 1 | 1551 | 38.3 | 127.9 | 224.1 | 154.15 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 8 | 2 | 1500 | 38.9 | 135.6 | 251.8 | 149.28 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 16 | 1 | 1404 | 94.8 | 256.6 | 343.2 | 138.61 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 16 | 2 | 1451 | 90.3 | 244.7 | 354.7 | 143.61 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 32 | 1 | 1367 | 188.2 | 560.8 | 824.9 | 133.95 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 32 | 2 | 1336 | 191.5 | 557.9 | 828.0 | 130.60 | 0.00% | True |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 64 | 1 | 1380 | 371.9 | 1138.9 | 1602.8 | 131.45 | 0.00% | False |
| 10,000 / 10,012 | GET /v1/knowledge-bases | 64 | 2 | 1169 | 407.6 | 1508.7 | 2275.3 | 112.15 | 0.00% | False |
| 10,000 / 10,012 | POST /v1/agent/runs | 1 | 1 | 396 | 21.4 | 38.6 | 81.3 | 39.52 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 1 | 2 | 377 | 20.0 | 59.6 | 143.3 | 37.66 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 2 | 1 | 454 | 37.3 | 83.5 | 136.9 | 45.24 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 2 | 2 | 494 | 35.7 | 70.3 | 118.8 | 49.31 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 4 | 1 | 425 | 55.1 | 270.8 | 635.7 | 42.39 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 4 | 2 | 492 | 53.0 | 215.2 | 525.3 | 48.55 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 8 | 1 | 463 | 78.5 | 705.9 | 1436.8 | 45.49 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 8 | 2 | 335 | 105.8 | 1078.6 | 1771.4 | 33.11 | 0.00% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 16 | 1 | 571 | 104.0 | 1207.0 | 2180.0 | 55.25 | 0.18% | True |
| 10,000 / 10,012 | POST /v1/agent/runs | 16 | 2 | 353 | 170.0 | 2092.6 | 3404.3 | 34.08 | 0.28% | False |

## 容量边界与稳定窗口

| 文档数 | 接口 | 阶梯最高通过 | 稳定窗口确认 | 首个失败档 | 稳定窗口 p95 ms | 稳定窗口成功 req/s |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 10,000 | GET /v1/health | 1 | 1 | 64 | 3.1, 2.7 | 382.77, 406.97 |
| 10,000 | GET /v1/knowledge-bases | 32 | 32 | 64 | 666.3, 613.3 | 115.60, 129.62 |
| 10,000 | POST /v1/agent/runs | 8 | 8 | 16 | 734.2, 699.9 | 46.94, 44.75 |

0 表示没有档位通过本次延迟和错误率标准，不表示接口无法处理请求。未测到失败档只说明容量下界，不是最大并发；稳定窗口确认值可能低于阶梯最高通过值。健康检查只抽测两档，仅作参照，不能据此推断完整问答容量。

## 请求错误与服务日志

- 10,000 条段落，POST /v1/agent/runs，并发 16：{'HTTP 500': 1}，共 571 请求，错误率 0.18%。
- 10,000 条段落，POST /v1/agent/runs，并发 16：{'HTTP 500': 1}，共 353 请求，错误率 0.28%。
- 10,000 条段落的服务日志错误摘要：{'|     raise exc.TimeoutError(': 2, '| sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30.00 (Background on this error at: https://sqlalche.me/e/20/3o7r)': 2, 'raise exc.TimeoutError(': 2, 'sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30.00 (Background on this error at: https://sqlalche.me/e/20/3o7r)': 2, 'sqlite3.OperationalError: database is locked': 2, 'sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked': 2}。同一异常可能出现多行，日志行数不能当作错误请求数；请求数取自上面的 HTTP 样本。

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
.venv/bin/python -m evals.http_capacity --sizes 10000 --levels 1 2 4 8 16 32 64 --seconds 10.0 --repeats 2 --min-requests 30 --soak-seconds 30.0 --timeout 15.0 --warmup-requests 5 --output evals/http-audit-results.json
```

结束后停止自己的测试进程并删除自己的临时数据库；原有应用数据保留。
