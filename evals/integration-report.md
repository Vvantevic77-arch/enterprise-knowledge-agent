# External integration benchmark

This report contains live dependency probes. It does not substitute offline RAG quality benchmarks.

## Run context

- Python: `3.12.14`
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Generated: `2026-09-08T05:20:56.741685+00:00`

## Results

| Dependency | Status | Latency (ms) | Detail |
| --- | --- | ---: | --- |
| network | passed | 1171.787 | https://api.github.com |
| postgres | passed | 39.114 |  |
| redis | passed | 5.207 | redis://localhost:6379/0 |
| celery | passed | 56.338 |  |
| llm | passed | 1958.803 | deepseek-chat |

## Reproduction

```bash
docker compose up -d postgres redis
celery -A app.worker.celery_app.celery_app worker --loglevel=INFO
DATABASE_URL=postgresql+asyncpg://agent:agent@localhost:5432/agent \
REDIS_URL=redis://localhost:6379/0 \
CELERY_BROKER_URL=redis://localhost:6379/1 \
CELERY_RESULT_BACKEND_URL=redis://localhost:6379/1 \
LLM_PROVIDER=deepseek \
DEEPSEEK_BASE_URL=https://api.deepseek.com \
DEEPSEEK_MODEL=deepseek-chat \
python evals/integration_benchmark.py --strict
```

The LLM probe sends one real request when the selected provider key is set (`DEEPSEEK_API_KEY` for DeepSeek or `OPENAI_API_KEY` for OpenAI). Set `OPENAI_BASE_URL` for an OpenAI-compatible gateway, `OPENAI_API_MODE=responses` when it exposes the Responses API, and optionally `OPENAI_REASONING_EFFORT=low|medium|high`. Keys and model responses are never written to the report.
