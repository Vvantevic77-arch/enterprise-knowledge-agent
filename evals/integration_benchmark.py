"""Probe real network, PostgreSQL, Redis, Celery, and LLM dependencies.

The script never falls back to the offline implementation. A dependency is
reported as ``skipped`` when its configuration is absent and ``failed`` when it
is configured but unreachable. Use ``--strict`` in CI to require every probe to
pass. Run from the project directory after starting Docker Compose::

    python evals/integration_benchmark.py --strict
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def error_text(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and str(cause) and str(cause) != str(exc):
        message = f"{message} (cause: {type(cause).__name__}: {' '.join(str(cause).split())[:160]})"
    return f"{type(exc).__name__}: {message[:400]}"


def failure(started: float, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "latency_ms": elapsed_ms(started),
        "error": error_text(exc),
    }


async def probe_network() -> dict[str, Any]:
    import httpx

    url = os.getenv("NETWORK_PROBE_URL", "https://api.github.com")
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=True) as client:
            response = await client.get(url, headers={"User-Agent": "payment-ops-integration-probe/0.1"})
        return {
            "status": "passed" if response.is_success else "failed",
            "latency_ms": elapsed_ms(started),
            "http_code": response.status_code,
            "url": url,
        }
    except Exception as exc:
        result = failure(started, exc)
        result["url"] = url
        return result


async def probe_postgres() -> dict[str, Any]:
    url = os.getenv("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        return {"status": "skipped", "reason": "DATABASE_URL is not a PostgreSQL URL"}
    import asyncpg

    dsn = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    started = time.perf_counter()
    connection = None
    try:
        connection = await asyncpg.connect(dsn, timeout=5)
        await connection.fetchval("SELECT 1")
        server_version = await connection.fetchval("SHOW server_version")
        has_vector = await connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        result = {
            "status": "passed",
            "latency_ms": elapsed_ms(started),
            "server_version": str(server_version),
            "pgvector_extension": bool(has_vector),
        }
        if not has_vector:
            result["status"] = "failed"
            result["error"] = "pgvector extension is not installed"
        return result
    except Exception as exc:
        return failure(started, exc)
    finally:
        if connection is not None:
            await connection.close()


async def probe_redis() -> dict[str, Any]:
    import redis.asyncio as redis

    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    started = time.perf_counter()
    client = redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    key = f"payment-ops:integration:{uuid4().hex}"
    try:
        await client.ping()
        await client.set(key, "ok", ex=30)
        value = await client.get(key)
        await client.delete(key)
        if value != "ok":
            raise RuntimeError("Redis SET/GET round-trip returned an unexpected value")
        return {"status": "passed", "latency_ms": elapsed_ms(started), "url": url}
    except Exception as exc:
        result = failure(started, exc)
        result["url"] = url
        return result
    finally:
        close = getattr(client, "aclose", client.close)
        maybe_awaitable = close()
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


async def probe_celery() -> dict[str, Any]:
    from app.worker.tasks import healthcheck_task

    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            lambda: healthcheck_task.delay().get(timeout=10, propagate=True)
        )
        if result != {"status": "ok"}:
            raise RuntimeError(f"unexpected Celery result: {result!r}")
        return {"status": "passed", "latency_ms": elapsed_ms(started)}
    except Exception as exc:
        return failure(started, exc)


async def probe_llm() -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "deepseek"}:
        return {"status": "failed", "latency_ms": 0.0, "error": "LLM_PROVIDER must be openai or deepseek"}
    api_key = os.getenv(
        "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY", ""
    ).strip()
    if not api_key:
        key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
        return {"status": "skipped", "reason": f"{key_name} is not set", "provider": provider}
    if any(ord(character) < 32 or ord(character) == 127 for character in api_key):
        return {
            "status": "failed",
            "latency_ms": 0.0,
            "error": f"{('DEEPSEEK_API_KEY' if provider == 'deepseek' else 'OPENAI_API_KEY')} contains control characters; set it on one line",
            "provider": provider,
        }
    from openai import AsyncOpenAI

    model = os.getenv(
        "DEEPSEEK_MODEL" if provider == "deepseek" else "OPENAI_MODEL",
        "deepseek-chat" if provider == "deepseek" else "gpt-4o-mini",
    )
    base_url = os.getenv(
        "DEEPSEEK_BASE_URL" if provider == "deepseek" else "OPENAI_BASE_URL",
        "https://api.deepseek.com" if provider == "deepseek" else "",
    ).strip()
    api_mode = "chat_completions" if provider == "deepseek" else os.getenv("OPENAI_API_MODE", "chat_completions").strip().lower()
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
    if api_mode not in {"chat_completions", "responses"}:
        return {"status": "failed", "latency_ms": 0.0, "error": "OPENAI_API_MODE must be chat_completions or responses", "model": model, "provider": provider}
    if reasoning_effort and reasoning_effort not in {"low", "medium", "high"}:
        return {"status": "failed", "latency_ms": 0.0, "error": "OPENAI_REASONING_EFFORT must be low, medium, or high", "model": model, "provider": provider}
    started = time.perf_counter()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    try:
        if api_mode == "responses":
            request: dict[str, Any] = {
                "model": model,
                "input": "Reply with the single word OK.",
                "max_output_tokens": 2,
            }
            if reasoning_effort:
                request["reasoning"] = {"effort": reasoning_effort}
            response = await client.responses.create(**request)
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=2,
                temperature=0,
            )
        usage = getattr(response, "usage", None)
        return {
            "status": "passed",
            "latency_ms": elapsed_ms(started),
            "model": model,
            "provider": provider,
            "base_url": base_url or "https://api.openai.com/v1",
            "api_mode": api_mode,
            "reasoning_effort": reasoning_effort or None,
            "input_tokens": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None),
        }
    except Exception as exc:
        result = failure(started, exc)
        result["model"] = model
        result["provider"] = provider
        result["base_url"] = base_url or "https://api.openai.com/v1"
        result["api_mode"] = api_mode
        result["reasoning_effort"] = reasoning_effort or None
        return result
    finally:
        await client.close()


async def run_probes() -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for name, probe in (
        ("network", probe_network),
        ("postgres", probe_postgres),
        ("redis", probe_redis),
        ("celery", probe_celery),
        ("llm", probe_llm),
    ):
        try:
            probes[name] = await probe()
        except Exception as exc:  # Keep one optional dependency from hiding others.
            probes[name] = {"status": "failed", "error": error_text(exc)}
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probes": probes,
    }


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# External integration benchmark",
        "",
        "This report contains live dependency probes. It does not substitute offline RAG quality benchmarks.",
        "",
        "## Run context",
        "",
        f"- Python: `{results['environment']['python']}`",
        f"- Platform: `{results['environment']['platform']}`",
        f"- Generated: `{results['generated_at_utc']}`",
        "",
        "## Results",
        "",
        "| Dependency | Status | Latency (ms) | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for name, probe in results["probes"].items():
        detail = probe.get("error") or probe.get("reason") or probe.get("url") or probe.get("model") or ""
        lines.append(
            f"| {name} | {probe['status']} | {probe.get('latency_ms', '-')} | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "docker compose up -d postgres redis",
            "celery -A app.worker.celery_app.celery_app worker --loglevel=INFO",
            "DATABASE_URL=postgresql+asyncpg://agent:agent@localhost:5432/agent \\",
            "REDIS_URL=redis://localhost:6379/0 \\",
            "CELERY_BROKER_URL=redis://localhost:6379/1 \\",
            "CELERY_RESULT_BACKEND_URL=redis://localhost:6379/1 \\",
            "LLM_PROVIDER=deepseek \\",
            "DEEPSEEK_BASE_URL=https://api.deepseek.com \\",
            "DEEPSEEK_MODEL=deepseek-chat \\",
            "python evals/integration_benchmark.py --strict",
            "```",
            "",
            "The LLM probe sends one real request when the selected provider key is set (`DEEPSEEK_API_KEY` for DeepSeek or `OPENAI_API_KEY` for OpenAI). Set `OPENAI_BASE_URL` for an OpenAI-compatible gateway, `OPENAI_API_MODE=responses` when it exposes the Responses API, and optionally `OPENAI_REASONING_EFFORT=low|medium|high`. Keys and model responses are never written to the report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail unless all five probes pass")
    args = parser.parse_args()
    results = asyncio.run(run_probes())
    results_path = Path(__file__).with_name("integration-results.json")
    report_path = Path(__file__).with_name("integration-report.md")
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    report_path.write_text(render_report(results))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.strict and any(probe.get("status") != "passed" for probe in results["probes"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
