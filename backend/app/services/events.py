"""Run event publication and replay.

Two transports, one interface:

* **In-process ring buffer** — always available. It is what makes the offline
  test suite deterministic and what keeps the demo working without Redis.
* **Redis Streams** — used when Redis is reachable, so events survive a worker
  restart and can be consumed by an audit pipeline or another UI replica.

Consumers always read through ``read_events``, so the WebSocket route does not
care which transport is live.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

#: How many events to retain per run in the in-process buffer. A long run that
#: exceeds this keeps its newest events; Redis is the durable option.
_MAX_BUFFERED_EVENTS = 1000

_streams: dict[str, list[dict[str, Any]]] = defaultdict(list)
_redis_client: Any | None = None
_redis_unavailable = False


def _get_redis_client() -> Any:
    """One process-wide Redis client backed by a connection pool."""
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as redis

        _redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


async def publish_event(run_id: str, event: dict[str, Any]) -> None:
    """Publish to the in-process buffer and, best-effort, to Redis."""
    global _redis_unavailable

    buffer = _streams[run_id]
    buffer.append(event)
    if len(buffer) > _MAX_BUFFERED_EVENTS:
        del buffer[: len(buffer) - _MAX_BUFFERED_EVENTS]

    if _redis_unavailable:
        return
    try:
        await _get_redis_client().xadd(
            f"agent:run:{run_id}",
            {"payload": json.dumps(event, ensure_ascii=False, default=str)},
            maxlen=_MAX_BUFFERED_EVENTS,
            approximate=True,
        )
    except Exception:
        # Redis is optional. Stop retrying on every event once it is known to be
        # down, otherwise a Redis outage turns into a per-event latency spike.
        _redis_unavailable = True
        logger.info("Redis unavailable; run events are in-process only")


async def read_events(run_id: str, after: int = 0) -> AsyncIterator[dict[str, Any]]:
    """Replay buffered events for a run, starting after index ``after``."""
    for event in _streams.get(run_id, [])[after:]:
        yield event


def event_count(run_id: str) -> int:
    return len(_streams.get(run_id, []))


def reset() -> None:
    """Clear the buffer. Used by tests to keep runs independent."""
    _streams.clear()
