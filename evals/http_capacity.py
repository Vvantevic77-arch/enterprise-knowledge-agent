"""Real loopback HTTP load test of a separate Uvicorn worker and isolated SQLite.

No ASGITransport, paid model calls, or existing user database is used. Capacity
is a workload/SLO-specific measured boundary, never a count of registered users.
Run from the project root: .venv/bin/python -m evals.http_capacity
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def isolated_environment(database: Path) -> dict[str, str]:
    return {
        "APP_ENV": "test",
        "DATABASE_URL": f"sqlite+aiosqlite:///{database}",
        "EMBEDDING_PROVIDER": "deterministic",
        "EMBEDDING_DIMENSIONS": "1536",
        "OPENAI_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
        "LOGFIRE_TOKEN": "",
        "JWT_SECRET": "local-http-benchmark-isolated-secret-20261012",
        "ALLOW_HEADER_IDENTITY": "false",
        "ACCESS_TOKEN_TTL_MINUTES": "180",
        "REDIS_URL": "redis://127.0.0.1:1/0",
        "PYTHONPATH": str(ROOT / "backend"),
    }


def valid_response(endpoint: str, response: httpx.Response, kb_id: str) -> bool:
    if response.status_code != 200:
        return False
    try:
        body = response.json()
        if endpoint == "health":
            return (
                body["status"] == "ok"
                and body["database"] == "sqlite"
                and not body["llm_configured"]
            )
        if endpoint == "knowledge_bases":
            return isinstance(body, list) and any(row["id"] == kb_id for row in body)
        return (
            body["status"] == "succeeded"
            and bool(body["answer"])
            and 1 <= len(body["citations"]) <= 5
            and not body["replayed"]
            and len({row["chunk_id"] for row in body["citations"]}) == len(body["citations"])
            and all(row["filename"].startswith("msmarco-") for row in body["citations"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def summarize_samples(samples: list[dict], elapsed: float, p95_slo_ms: float) -> dict:
    if not samples or elapsed <= 0:
        raise ValueError("Empty samples or non-positive elapsed time")
    latency = [row["latency_ms"] for row in samples]
    errors = sum(not row["valid"] for row in samples)
    p50, p95, p99 = np.percentile(latency, [50, 95, 99]).tolist()
    return {
        "requests": len(samples),
        "elapsed_seconds": elapsed,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "error_rate": errors / len(samples),
        "throughput_requests_per_second": len(samples) / elapsed,
        "successful_requests_per_second": (len(samples) - errors) / elapsed,
        "status_counts": dict(Counter(str(row["status"]) for row in samples)),
        "error_counts": dict(Counter(row["error"] for row in samples if not row["valid"])),
        "slo_pass": p95 <= p95_slo_ms and errors / len(samples) <= 0.01,
    }


async def seed(database: Path, documents: list[str], queries: list, count: int) -> dict:
    # Set all overrides before app.config import; never expose dotenv secrets.
    os.environ.update(isolated_environment(database))
    from app.config import settings
    from app.db import SessionLocal, engine, init_db
    from app.models import Document, DocumentChunk, KnowledgeBase, Membership, Organization, User
    from app.security import create_access_token, hash_password
    from app.services.rag import chunk_text, deterministic_embedding
    from sqlalchemy import func, insert, select

    assert settings.database_url == os.environ["DATABASE_URL"] and settings.is_sqlite
    assert settings.embedding_provider == "deterministic" and settings.embedding_dimensions == 1536
    await init_db()
    org_id, user_id, kb_id = (str(uuid4()) for _ in range(3))
    async with SessionLocal() as db:
        db.add(
            User(
                id=user_id,
                email="capacity@example.invalid",
                hashed_password=hash_password("isolated-test-password"),
            )
        )
        await db.flush()
        db.add(Organization(id=org_id, name="Isolated HTTP capacity test", created_by=user_id))
        await db.flush()
        db.add_all(
            [
                Membership(user_id=user_id, org_id=org_id, role="owner"),
                KnowledgeBase(id=kb_id, org_id=org_id, name="MS MARCO performance corpus"),
            ]
        )
        await db.flush()
        await db.commit()
        started = time.perf_counter()
        for start in range(0, count, 100):
            doc_rows, chunk_rows = [], []
            for index, text in enumerate(documents[start : min(start + 100, count)], start):
                doc_id = str(uuid4())
                pieces = chunk_text(text)
                doc_rows.append(
                    {
                        "id": doc_id,
                        "org_id": org_id,
                        "kb_id": kb_id,
                        "filename": f"msmarco-{index:06d}.txt",
                        "source_url": "",
                        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                        "status": "indexed",
                        "chunk_count": len(pieces),
                    }
                )
                for ordinal, content in enumerate(pieces):
                    chunk_rows.append(
                        {
                            "id": str(uuid4()),
                            "document_id": doc_id,
                            "ordinal": ordinal,
                            "content": content,
                            "embedding": deterministic_embedding(content),
                            "metadata_json": {},
                        }
                    )
            await db.execute(insert(Document), doc_rows)
            await db.execute(insert(DocumentChunk), chunk_rows)
            await db.commit()
            if (start + 100) % 10000 == 0:
                print(f"Seeded {min(start + 100, count):,} documents", flush=True)
        actual_documents = await db.scalar(select(func.count()).select_from(Document))
        actual_chunks = await db.scalar(select(func.count()).select_from(DocumentChunk))
    assert actual_documents == count
    # Queries drawn from the split, with a known positive in this corpus.
    from app.services.policy import wants_ticket

    corpus_ids = set(range(count))
    prompts = [q.text for q in queries if q.relevant & corpus_ids and not wants_ticket(q.text)][
        :100
    ]
    assert prompts, "No eligible informational queries"
    token = create_access_token(user_id)
    await engine.dispose()
    return {
        "headers": {"Authorization": f"Bearer {token}", "X-Org-ID": org_id},
        "kb_id": kb_id,
        "prompts": prompts,
        "documents": actual_documents,
        "chunks": actual_chunks,
        "seed_seconds": time.perf_counter() - started,
        "database_bytes": database.stat().st_size,
    }


async def worker_resources(pid: int) -> dict:
    process = await asyncio.create_subprocess_exec(
        "ps",
        "-o",
        "rss=,%cpu=",
        "-p",
        str(pid),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    output, _ = await process.communicate()
    try:
        rss, cpu = output.decode().split()
        return {"rss_mib": float(rss) / 1024, "recent_cpu_percent": float(cpu)}
    except ValueError:
        return {}


async def load_window(
    client: httpx.AsyncClient,
    endpoint: str,
    context: dict,
    concurrency: int,
    seconds: float,
    min_requests: int,
    p95_slo_ms: float,
    pid: int,
) -> dict:
    samples, resources = [], []
    started = time.perf_counter()
    client_cpu_started = time.process_time()
    deadline = started + seconds
    sequence = 0
    monitoring = True

    async def monitor():
        while monitoring:
            resources.append(await worker_resources(pid))
            await asyncio.sleep(1)

    async def worker():
        nonlocal sequence
        while time.perf_counter() < deadline or len(samples) < min_requests:
            number = sequence
            sequence += 1
            request_started = time.perf_counter()
            status, valid, error = "exception", False, ""
            try:
                if endpoint == "agent_runs":
                    response = await client.post(
                        "/v1/agent/runs",
                        headers=context["headers"],
                        json={
                            "prompt": context["prompts"][number % len(context["prompts"])],
                            "knowledge_base_ids": [context["kb_id"]],
                            "idempotency_key": f"capacity-{uuid4().hex}",
                        },
                    )
                else:
                    path = "/v1/health" if endpoint == "health" else "/v1/knowledge-bases"
                    response = await client.get(path, headers=context["headers"])
                status = response.status_code
                valid = valid_response(endpoint, response, context["kb_id"])
                if not valid:
                    error = f"HTTP {status}" if status != 200 else "unexpected body"
            except httpx.HTTPError as exc:
                error = type(exc).__name__
            samples.append(
                {
                    "start_offset_seconds": request_started - started,
                    "latency_ms": (time.perf_counter() - request_started) * 1000,
                    "status": status,
                    "valid": valid,
                    "error": error,
                }
            )

    monitor_task = asyncio.create_task(monitor())
    try:
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    finally:
        monitoring = False
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
    elapsed = time.perf_counter() - started
    summary = summarize_samples(samples, elapsed, p95_slo_ms)
    summary.update(
        {
            "concurrency": concurrency,
            "target_seconds": seconds,
            "client_cpu_seconds": time.process_time() - client_cpu_started,
            "server_resources": resources,
            "samples": samples,
        }
    )
    return summary


async def measure(
    client: httpx.AsyncClient, context: dict, endpoint: str, levels: list[int], args, pid: int
) -> dict:
    slo_ms = 2000.0 if endpoint == "agent_runs" else 1000.0
    rounds = []
    for concurrency in levels:
        trials = []
        for repeat in range(args.repeats):
            result = await load_window(
                client, endpoint, context, concurrency, args.seconds, args.min_requests, slo_ms, pid
            )
            result["repeat"] = repeat + 1
            trials.append(result)
            print(
                f"{context['documents']:,} docs {endpoint} c={concurrency} repeat={repeat + 1}: p95={result['p95_ms']:.1f}ms errors={result['error_rate']:.2%} success={result['successful_requests_per_second']:.2f}/s",
                flush=True,
            )
        rounds.append(
            {
                "concurrency": concurrency,
                "slo_pass": all(t["slo_pass"] for t in trials),
                "trials": trials,
            }
        )
        if not rounds[-1]["slo_pass"]:
            break
    passing = [row["concurrency"] for row in rounds if row["slo_pass"]]
    boundary = max(passing, default=0)
    soak = []
    if boundary and args.soak_seconds:
        for _ in range(args.repeats):
            soak.append(
                await load_window(
                    client,
                    endpoint,
                    context,
                    boundary,
                    args.soak_seconds,
                    args.min_requests,
                    slo_ms,
                    pid,
                )
            )
        print(
            f"{endpoint} c={boundary} confirmation: {[round(s['p95_ms'], 1) for s in soak]}ms",
            flush=True,
        )
    confirmed = boundary if boundary and soak and all(t["slo_pass"] for t in soak) else 0
    return {
        "endpoint": endpoint,
        "p95_slo_ms": slo_ms,
        "error_rate_slo": 0.01,
        "rounds": rounds,
        "largest_passing_staircase_concurrency": boundary,
        "confirmed_concurrency": confirmed,
        "confirmation_trials": soak,
        "first_failing_concurrency": next(
            (r["concurrency"] for r in rounds if not r["slo_pass"]), None
        ),
        "ceiling_reached": bool(rounds[-1]["slo_pass"] and rounds[-1]["concurrency"] == levels[-1]),
    }


async def run(args) -> dict:
    import pyarrow.parquet as pq

    from evals.public_benchmark import EXPECTED_SHA256, prepare_rows

    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch")
    documents, queries = prepare_rows(pq.read_table(args.dataset).to_pylist())
    if not all(5 <= size <= len(documents) for size in args.sizes):
        raise ValueError("Invalid corpus size")
    source_hash = hashlib.sha256()
    for source in sorted((ROOT / "backend").rglob("*.py")):
        source_hash.update(str(source.relative_to(ROOT)).encode())
        source_hash.update(source.read_bytes())
    results = []
    # Seed in subprocesses because app.db's engine is selected at import time.
    for size in args.sizes:
        with tempfile.TemporaryDirectory(prefix="agent-http-capacity-") as directory:
            temp = Path(directory)
            database, log = temp / "benchmark.db", temp / "server.log"
            seed_context = temp / "context.json"
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "evals.http_capacity",
                "--seed",
                str(database),
                "--size",
                str(size),
                "--context",
                str(seed_context),
                "--dataset",
                str(args.dataset),
                cwd=ROOT,
            )
            if await process.wait() != 0:
                raise RuntimeError("Isolated database seeding failed")
            context = json.loads(seed_context.read_text())
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            environment = os.environ.copy()
            environment.update(isolated_environment(database))
            with log.open("w") as output:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--workers",
                        "1",
                        "--no-access-log",
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=output,
                    stderr=output,
                )
                try:
                    async with httpx.AsyncClient(
                        base_url=f"http://127.0.0.1:{port}",
                        trust_env=False,
                        timeout=args.timeout,
                        limits=httpx.Limits(
                            max_connections=max(args.levels),
                            max_keepalive_connections=max(args.levels),
                        ),
                    ) as client:
                        for _ in range(200):
                            if server.poll() is not None:
                                raise RuntimeError(log.read_text())
                            try:
                                response = await client.get("/v1/health")
                                if response.status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            await asyncio.sleep(0.1)
                        else:
                            raise RuntimeError("Server never became healthy")
                        # Warm the DB, graph, JSON parsing and connection path.
                        warmup = await load_window(
                            client,
                            "agent_runs",
                            context,
                            1,
                            0,
                            args.warmup_requests,
                            2000,
                            server.pid,
                        )
                        if warmup["error_rate"]:
                            raise RuntimeError(f"Warm-up invalid: {warmup['error_counts']}")
                        measurements = []
                        if size == args.sizes[-1]:
                            measurements.append(
                                await measure(
                                    client,
                                    context,
                                    "health",
                                    [1, min(64, max(args.levels))],
                                    args,
                                    server.pid,
                                )
                            )
                            measurements.append(
                                await measure(
                                    client,
                                    context,
                                    "knowledge_bases",
                                    args.levels,
                                    args,
                                    server.pid,
                                )
                            )
                        measurements.append(
                            await measure(
                                client, context, "agent_runs", args.levels, args, server.pid
                            )
                        )
                        results.append(
                            {
                                "corpus": {
                                    key: context[key]
                                    for key in (
                                        "documents",
                                        "chunks",
                                        "database_bytes",
                                        "seed_seconds",
                                    )
                                },
                                "query_prompts": context["prompts"],
                                "warmup_requests": warmup["requests"],
                                "measurements": measurements,
                            }
                        )
                finally:
                    server.terminate()
                    try:
                        await asyncio.wait_for(asyncio.to_thread(server.wait), timeout=20)
                    except TimeoutError:
                        server.kill()
                        await asyncio.to_thread(server.wait)
            errors = [
                line.strip()
                for line in log.read_text().splitlines()
                if "OperationalError" in line
                or "TimeoutError" in line
                or "database is locked" in line
            ]
            results[-1]["server_error_counts"] = dict(Counter(errors))
            # Persist each completed scenario even if a later measurement fails.
            args.output.write_text(json.dumps({"scenarios": results}, ensure_ascii=False) + "\n")
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "server_workers": 1,
            "database": "isolated SQLite, stored 1536-dimensional JSON embeddings",
            "retrieval": "SQLite FTS5 BM25; indexed CJK bigrams; no deterministic vector fallback",
            "idempotency": "committed reservation; separate retrieval session; no connection during model wait",
            "backend_source_sha256": source_hash.hexdigest(),
            "sqlite_journal_mode": "wal",
            "transport": "real loopback TCP HTTP with reused connections",
            "llm": "disabled; deterministic grounded answers",
            "redis": "unavailable; in-process events",
            "client_and_server": "same laptop",
            "open_files_limit": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        },
        "method": {
            "levels": args.levels,
            "seconds_per_trial": args.seconds,
            "repeats": args.repeats,
            "minimum_requests_per_trial": args.min_requests,
            "confirmation_seconds_per_trial": args.soak_seconds,
            "timeout_seconds": args.timeout,
            "warmup_requests": args.warmup_requests,
            "response_validation": "succeeded、非空答案、1–5条不重复的语料引用和非 replay",
            "load_model": "closed loop, zero think time; latency includes server queuing, client awaits and HTTP",
            "quantiles": "NumPy linear percentile; all completed/failed requests included",
            "slo": "p95 <= 1s for health/KB, <= 2s for offline RAG; errors <= 1% in every trial",
            "data_sha256": EXPECTED_SHA256,
        },
        "scenarios": results,
    }


def render_report(result: dict) -> str:
    warmup_requests = result["method"].get(
        "warmup_requests", result["scenarios"][0]["warmup_requests"]
    )
    environment = result["environment"]
    paths = {
        "health": "GET /v1/health",
        "knowledge_bases": "GET /v1/knowledge-bases",
        "agent_runs": "POST /v1/agent/runs",
    }
    lines = [
        "# 本地真实 HTTP 性能与容量",
        "",
        "用户确认项目只在本机运行。本实验启动独立 Uvicorn 进程，通过 TCP HTTP 请求实际接口，使用隔离 SQLite；未接触现有用户数据库。环境与当前默认/Render 配置的 SQLite 检索路径相同，但不是线上服务器。",
        "",
        f"测量完成时间（UTC）：{result['generated_at_utc']}。环境：{environment['platform']}、Python {environment['python']}、{environment['logical_cpus']} 个逻辑 CPU、{environment['server_workers']} 个 Uvicorn worker；客户端与服务在同一电脑，SQLite 保存 1,536 维 JSON 向量，外部模型关闭。完整环境参数保存在原始 JSON。",
        "",
        "## 方法与判定",
        "",
        f"每档 {result['method']['seconds_per_trial']} 秒，至少 {result['method']['minimum_requests_per_trial']} 请求，重复 {result['method']['repeats']} 轮；最高通过档再各测 {result['method']['confirmation_seconds_per_trial']} 秒。预热 {warmup_requests} 次完整问答。",
        "低吞吐档会为满足最少请求数延长窗口。语料按文本哈希排序，取当前规模的前 N 条；每个规模循环使用最多 100 个原始自然语言问题，问题的标注正例至少一条存在于当前语料中。",
        f"本次检索路径：{environment.get('retrieval', '见对应版本代码')}。幂等策略：{environment.get('idempotency', '见对应版本代码')}。",
        "",
        "暂定容量标准：健康/知识库读取 p95≤1秒，离线 RAG 问答 p95≤2秒，每轮错误率≤1%。这只是本次明确设置的标准，未经业务方定义为正式 SLA。阶梯到首个失败档停止；边界介于通过档和失败档之间，没有测到边界则只给下界。",
        "",
        f"闭环并发表示同时发起且等待响应的请求数，零思考时间；客户端与服务在同一台电脑，包含 TCP、认证、数据库读取、问答检索、事务写入和引用读取。连接复用，超时{result['method']['timeout_seconds']}秒。每次问答使用唯一幂等键并检查 {result['method'].get('response_validation', 'succeeded、非空答案、5条语料引用和非 replay')}。未验证答案事实准确率。",
        "",
        "## 逐档实测",
        "",
        "| 文档 / chunk | 接口 | 并发 | 轮次 | 请求 | p50 ms | p95 ms | p99 ms | 成功 req/s | 错误率 | 通过 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scenario in result["scenarios"]:
        corpus = scenario["corpus"]
        for measurement in scenario["measurements"]:
            for round_ in measurement["rounds"]:
                for trial in round_["trials"]:
                    lines.append(
                        f"| {corpus['documents']:,} / {corpus['chunks']:,} | {paths[measurement['endpoint']]} | {round_['concurrency']} | {trial['repeat']} | {trial['requests']} | {trial['p50_ms']:.1f} | {trial['p95_ms']:.1f} | {trial['p99_ms']:.1f} | {trial['successful_requests_per_second']:.2f} | {trial['error_rate']:.2%} | {trial['slo_pass']} |"
                    )
    lines.extend(
        [
            "",
            "## 容量边界与稳定窗口",
            "",
            "| 文档数 | 接口 | 阶梯最高通过 | 稳定窗口确认 | 首个失败档 | 稳定窗口 p95 ms | 稳定窗口成功 req/s |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for scenario in result["scenarios"]:
        for measurement in scenario["measurements"]:
            soak = measurement["confirmation_trials"]
            p95 = ", ".join(f"{row['p95_ms']:.1f}" for row in soak)
            throughput = ", ".join(f"{row['successful_requests_per_second']:.2f}" for row in soak)
            lines.append(
                f"| {scenario['corpus']['documents']:,} | {paths[measurement['endpoint']]} | {measurement['largest_passing_staircase_concurrency']} | {measurement['confirmed_concurrency']} | {measurement['first_failing_concurrency']} | {p95} | {throughput} |"
            )
    lines.extend(
        [
            "",
            "0 表示没有档位通过本次延迟和错误率标准，不表示接口无法处理请求。未测到失败档只说明容量下界，不是最大并发；稳定窗口确认值可能低于阶梯最高通过值。健康检查只抽测两档，仅作参照，不能据此推断完整问答容量。",
            "",
            "## 请求错误与服务日志",
            "",
        ]
    )
    for scenario in result["scenarios"]:
        for measurement in scenario["measurements"]:
            trials = [t for r in measurement["rounds"] for t in r["trials"]]
            trials += measurement["confirmation_trials"]
            for trial in trials:
                if trial["error_counts"]:
                    lines.append(
                        f"- {scenario['corpus']['documents']:,} 条段落，{paths[measurement['endpoint']]}，并发 {trial['concurrency']}：{trial['error_counts']}，共 {trial['requests']} 请求，错误率 {trial['error_rate']:.2%}。"
                    )
        if scenario["server_error_counts"]:
            lines.append(
                f"- {scenario['corpus']['documents']:,} 条段落的服务日志错误摘要：{scenario['server_error_counts']}。同一异常可能出现多行，日志行数不能当作错误请求数；请求数取自上面的 HTTP 样本。"
            )
    if not any(scenario["server_error_counts"] for scenario in result["scenarios"]):
        lines.append("- 本次服务日志未发现 OperationalError、TimeoutError 或 database is locked。")
    lines.extend(
        [
            "",
            "## 限制与瓶颈",
            "",
            "- SQLite deterministic 路径使用 FTS5/BM25 和中文双字索引；无命中返回空结果，不再回退全量向量扫描。本次英文语料不衡量中文检索质量。",
            "- 幂等预留单独提交，完整问答期间不持有写锁；最终 run/approval/usage 与幂等结果一起提交。SQLite 的短写事务仍只有一个 writer，高并发仍可能排队。",
            "- GET 的允许审计在读事务结束后用独立短事务提交；审计仍消耗 SQLite writer。本版本启用 WAL，保留默认 FULL 持久化设置。",
            "- 检索会话在模型等待前关闭；同步 embedding、向量评分和 broker 投递移至线程。知识库读取容量和健康检查容量都不能替代完整问答容量。",
            "- 关闭了外部 embedding/LLM，缺少网络模型延迟、配额、Redis/Celery、反向代理、多worker、生产硬件和长期混合负载；这些边界不能当作线上最大并发用户数。",
            "- 语料直接批量写入隔离数据库，复用生产切分和向量算法，计数来自数据库；未测上传和入库任务吞吐。",
            "- 客户端与服务器共享资源；原始逐请求数据、HTTP/超时分布及服务RSS/CPU采样保存在 JSON，短窗口 p99 和低样本档仅作描述。",
            "- 81,039 条的全库检索质量与缓存计算已另测，本报告 API 容量只适用于表中的实际入库规模，不能外推全库服务性能。",
            "",
            "## 复现",
            "",
            "```bash",
            ".venv/bin/python -m evals.http_capacity "
            f"--sizes {' '.join(str(s['corpus']['documents']) for s in result['scenarios'])} "
            f"--levels {' '.join(str(c) for c in result['method']['levels'])} "
            f"--seconds {result['method']['seconds_per_trial']} --repeats {result['method']['repeats']} "
            f"--min-requests {result['method']['minimum_requests_per_trial']} "
            f"--soak-seconds {result['method']['confirmation_seconds_per_trial']} "
            f"--timeout {result['method']['timeout_seconds']} --warmup-requests {warmup_requests} "
            f"--output {result['method'].get('output', 'evals/http-capacity-results.json')}",
            "```",
            "",
            "结束后停止自己的测试进程并删除自己的临时数据库；原有应用数据保留。",
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--min-requests", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--soak-seconds", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("http-capacity-results.json")
    )
    parser.add_argument("--seed", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--context", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seed:
        # public_benchmark imports rag/config: set DB before that import too.
        os.environ.update(isolated_environment(args.seed))
        import pyarrow.parquet as pq

        from evals.public_benchmark import prepare_rows

        documents, queries = prepare_rows(pq.read_table(args.dataset).to_pylist())
        context = asyncio.run(seed(args.seed, documents, queries, args.size))
        args.context.write_text(json.dumps(context))
        return
    if (
        min(args.levels) < 1
        or args.seconds <= 0
        or args.min_requests < 1
        or args.repeats < 1
        or args.soak_seconds < 0
        or args.timeout <= 0
        or args.warmup_requests < 1
        or args.levels != sorted(set(args.levels))
    ):
        parser.error("Invalid measurement parameters")
    result = asyncio.run(run(args))
    result["method"]["output"] = str(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_name(args.output.stem.replace("-results", "") + "-report.md").write_text(
        render_report(result)
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
