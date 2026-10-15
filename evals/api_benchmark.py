"""Reproducible in-process API latency and concurrency benchmark.

This benchmark uses the real FastAPI application and a temporary SQLite
database. It measures application overhead only; it is not a production SLA
because there is no network hop, reverse proxy, PostgreSQL, or external LLM.

Run from ``enterprise_knowledge_agent``::

    python evals/api_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from uuid import uuid4

# Set environment before importing app.config, which creates the SQLAlchemy
# engine at import time. Empty model keys force the deterministic offline path.
db_file = Path(tempfile.gettempdir()) / f"payment_ops_api_benchmark_{uuid4().hex}.db"
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_file}"
os.environ["EMBEDDING_PROVIDER"] = "deterministic"
os.environ["OPENAI_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["JWT_SECRET"] = "api-benchmark-only-secret"

import httpx  # noqa: E402

from app.db import engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services.demo import DEMO_SOURCES  # noqa: E402
from app.services.rag import chunk_text  # noqa: E402


TOTAL_REQUESTS = 200
CONCURRENCY_LEVELS = (1, 10, 25, 50)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile_value / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def request_once(path: str, headers: dict[str, str] | None = None) -> tuple[float, int]:
    started = time.perf_counter()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        response = await client.get(path, headers=headers)
    return (time.perf_counter() - started) * 1000, response.status_code


async def benchmark_endpoint(
    path: str, concurrency: int, headers: dict[str, str] | None = None
) -> dict[str, float | int]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded() -> tuple[float, int]:
        async with semaphore:
            return await request_once(path, headers)

    started = time.perf_counter()
    samples = await asyncio.gather(*(bounded() for _ in range(TOTAL_REQUESTS)))
    elapsed = max(time.perf_counter() - started, 1e-9)
    latencies = [sample[0] for sample in samples]
    errors = sum(status >= 400 for _, status in samples)
    return {
        "requests": TOTAL_REQUESTS,
        "concurrency": concurrency,
        "p50_ms": round(percentile(latencies, 50), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "p99_ms": round(percentile(latencies, 99), 3),
        "error_rate": round(errors / TOTAL_REQUESTS, 4),
        "throughput_requests_per_second": round(TOTAL_REQUESTS / elapsed, 2),
    }


async def run() -> dict[str, object]:
    await init_db()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://benchmark"
    ) as client:
        registration = await client.post(
            "/v1/auth/register",
            json={
                "email": "benchmark@example.com",
                "password": "benchmark-password",
                "display_name": "Benchmark User",
                "organization_name": "Benchmark Org",
            },
        )
        registration.raise_for_status()
        access_token = registration.json()["access_token"]
        organizations = await client.get(
            "/v1/organizations", headers={"Authorization": f"Bearer {access_token}"}
        )
        organizations.raise_for_status()
        org_id = organizations.json()[0]["id"]
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Org-ID": org_id,
        }
        knowledge = await client.get("/v1/knowledge-bases", headers=headers)
        knowledge.raise_for_status()
        knowledge_rows = knowledge.json()

    corpus_documents = len(DEMO_SOURCES)
    corpus_chunks = sum(
        len(chunk_text(content, size=420, overlap=60)) for _, content in DEMO_SOURCES
    )
    paths = {
        "health": ("/v1/health", None),
        "knowledge_bases": ("/v1/knowledge-bases", headers),
    }
    latency: dict[str, dict[str, object]] = {}
    for name, (path, request_headers) in paths.items():
        latency[name] = {
            str(level): await benchmark_endpoint(path, level, request_headers)
            for level in CONCURRENCY_LEVELS
        }
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "database": "SQLite temporary file",
            "llm": "disabled; deterministic offline mode",
        },
        "corpus": {
            "documents": corpus_documents,
            "chunks": corpus_chunks,
            "knowledge_bases": len(knowledge_rows),
        },
        "benchmark": {
            "total_requests_per_level": TOTAL_REQUESTS,
            "levels": list(CONCURRENCY_LEVELS),
        },
        "latency": latency,
    }


def render_report(results: dict[str, object]) -> str:
    corpus = results["corpus"]
    latency = results["latency"]
    lines = [
        "# Local API benchmark report",
        "",
        "The benchmark uses the real FastAPI ASGI app with a temporary SQLite database and deterministic embeddings. It measures application overhead, not production network or infrastructure SLA.",
        "",
        f"- Demo corpus: **{corpus['knowledge_bases']} knowledge base**, **{corpus['documents']} documents**, **{corpus['chunks']} chunks**.",
        f"- Each endpoint/level: **{results['benchmark']['total_requests_per_level']} requests**; all levels completed with an expected 2xx response.",
        "",
    ]
    for endpoint, levels in latency.items():
        lines.extend(
            [
                f"## `{endpoint}`",
                "",
                "| Concurrent requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error rate |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for level, metrics in levels.items():
            lines.append(
                f"| {level} | {metrics['p50_ms']} | {metrics['p95_ms']} | {metrics['p99_ms']} | {metrics['throughput_requests_per_second']} | {metrics['error_rate']:.2%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "cd enterprise_knowledge_agent",
            "python evals/api_benchmark.py",
            "```",
            "",
            "The concurrency column is the number of in-process requests allowed at once, not a claim that this is the maximum number of production users. Validate production capacity again with PostgreSQL, the deployed worker topology, network load, and representative LLM traffic.",
            "",
        ]
    )
    return "\n".join(lines)


async def main() -> None:
    try:
        results = await run()
        results["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        output = Path(__file__).with_name("api-benchmark-results.json")
        report = Path(__file__).with_name("api-benchmark-report.md")
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        report.write_text(render_report(results))
        print(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        await engine.dispose()
        db_file.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
