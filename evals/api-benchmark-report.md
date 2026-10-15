# Local API benchmark report

The benchmark uses the real FastAPI ASGI app with a temporary SQLite database and deterministic embeddings. It measures application overhead, not production network or infrastructure SLA.

- Demo corpus: **1 knowledge base**, **11 documents**, **18 chunks**.
- Each endpoint/level: **200 requests**; all levels completed with an expected 2xx response.

## `health`

| Concurrent requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.767 | 1.993 | 2.121 | 511.43 | 0.00% |
| 10 | 15.896 | 19.166 | 24.815 | 610.86 | 0.00% |
| 25 | 42.535 | 65.76 | 78.016 | 552.16 | 0.00% |
| 50 | 83.24 | 136.81 | 181.797 | 557.88 | 0.00% |

## `knowledge_bases`

| Concurrent requests | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.463 | 3.77 | 4.238 | 283.72 | 0.00% |
| 10 | 6.276 | 208.348 | 565.575 | 259.98 | 0.00% |
| 25 | 41.406 | 388.2 | 791.994 | 201.75 | 0.00% |
| 50 | 123.446 | 573.965 | 804.937 | 217.88 | 0.00% |

## Reproduction

```bash
cd enterprise_knowledge_agent
python evals/api_benchmark.py
```

The concurrency column is the number of in-process requests allowed at once, not a claim that this is the maximum number of production users. Validate production capacity again with PostgreSQL, the deployed worker topology, network load, and representative LLM traffic.
