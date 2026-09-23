# Offline benchmark report

This report is generated from `evals/benchmark.py` using synthetic payment-operations data and local deterministic embeddings.
It is suitable for project evidence, not a production API or database SLA.

## Run context

- Python: `3.12.14`
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Synthetic payment-operations documents: `240`; indexed chunks: `960`
- Chunking: size `420`, overlap `60`

## Results

| Area | Metric | Result |
| --- | --- | ---: |
| Chunking | documents/sec | 46710.03 |
| Chunking | MB/sec | 62.88 |
| Retrieval | Recall@1 | 65.42% |
| Retrieval | Recall@3 | 81.67% |
| Retrieval | Recall@5 | 86.25% |
| Retrieval | Recall@10 | 91.25% |
| Retrieval | MRR | 0.7488 |
| Incremental index | duplicate zero-op rate | 100.00% (240/240) |
| Incremental index | updated documents | 24 |
| Incremental index | reindex work reduction vs full rebuild | 90.00% |
| Agent policy | ticket precision | 100.00% |
| Agent policy | ticket recall | 100.00% |
| Agent policy | safe-prompt false positive rate | 0.00% |
| Agent policy | approval gate coverage | 100.00% |

### Retrieval strategy comparison

| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Vector only | 0.00% | 2.92% | 4.58% | 0.0250 |
| Lexical only | 100.00% | 100.00% | 100.00% | 1.0000 |
| Hybrid (optimized) | 65.42% | 86.25% | 91.25% | 0.7488 |

The synthetic query includes each document's unique reference key, so lexical-only retrieval is an artificial upper bound in this fixture. It is not evidence that lexical search beats semantic RAG; use naturally worded labelled queries and a real embedding model for that conclusion.

### Search implementation A/B

| Implementation | p50 (ms) | p95 (ms) | Throughput (req/s) |
| --- | ---: | ---: | ---: |
| Baseline full sort | 46.014 | 46.736 | 21.48 |
| Optimized Top-K selection | 32.125 | 32.809 | 31.07 |
| Change | - | **-29.80% p95** | **+44.62%** |

### Concurrent retrieval latency

| Workers | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 32.122 | 33.511 | 35.443 | 30.92 |
| 10 | 40 | 164.058 | 428.982 | 550.299 | 30.75 |
| 50 | 40 | 113.401 | 205.485 | 242.614 | 30.93 |

## Reproduction and interpretation

```bash
cd enterprise_knowledge_agent
python evals/benchmark.py
```

Retrieval queries use each payment-operations document's unique reference key, so these scores measure exact-key lookup in the deterministic offline fallback. They do not represent semantic quality of an external embedding model.
Concurrency numbers cover in-process chunk ranking only; PostgreSQL, Redis, Celery, WebSocket, network, and LLM latency are intentionally outside this benchmark.
