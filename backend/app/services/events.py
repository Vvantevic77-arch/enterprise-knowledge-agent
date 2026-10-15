"""Bounded local run events with optional Redis Streams replay."""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)
_MAX_BUFFERED_EVENTS = 1000
_MAX_BUFFERED_RUNS = 256
_EVENT_TTL_SECONDS = 3600
_REDIS_RETRY_SECONDS = 30

_streams: OrderedDict[str, tuple[float, deque]] = OrderedDict()
_redis_client: Any | None = None
_redis_retry_at = 0.0
event_sink: ContextVar[Callable[[dict[str, Any]], Awaitable[None]] | None] = ContextVar(
    "agent_event_sink", default=None
)


def _prune() -> None:
    now = time.monotonic()
    while _streams:
        run_id, (touched, _) = next(iter(_streams.items()))
        if now - touched < _EVENT_TTL_SECONDS and len(_streams) <= _MAX_BUFFERED_RUNS:
            break
        _streams.pop(run_id)


def _get_redis_client() -> Any:
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as redis
        from redis.backoff import NoBackoff
        from redis.retry import Retry

        _redis_client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            retry=Retry(NoBackoff(), 0),
        )
    return _redis_client


def _redis_failed() -> None:
    global _redis_retry_at
    _redis_retry_at = time.monotonic() + _REDIS_RETRY_SECONDS
    logger.info("Redis unavailable; using bounded local run events")


async def publish_event(run_id: str, event: dict[str, Any]) -> None:
    _prune()
    _, buffer = _streams.pop(run_id, (0, deque(maxlen=_MAX_BUFFERED_EVENTS)))
    sequence = buffer[-1][0] + 1 if buffer else 1
    entry = (sequence, dict(event), None)
    buffer.append(entry)
    _streams[run_id] = (time.monotonic(), buffer)
    _prune()
    sink = event_sink.get()
    if sink is not None:
        await sink(dict(event))
    if time.monotonic() < _redis_retry_at:
        return
    try:
        key = f"agent:run:{run_id}"
        pipeline = _get_redis_client().pipeline(transaction=False)
        pipeline.xadd(
            key,
            {"payload": json.dumps(event, ensure_ascii=False, default=str)},
            maxlen=_MAX_BUFFERED_EVENTS,
            approximate=True,
        )
        pipeline.expire(key, _EVENT_TTL_SECONDS)
        event_id, _ = await pipeline.execute()
        # Another coroutine may append while publication waits for Redis.
        for index, item in enumerate(buffer):
            if item is entry:
                buffer[index] = (sequence, item[1], event_id)
                break
    except Exception:
        _redis_failed()


async def read_events(run_id: str, after: int | str = 0) -> AsyncIterator[dict[str, Any]]:
    _prune()
    if time.monotonic() >= _redis_retry_at:
        try:
            rows = await _get_redis_client().xrange(
                f"agent:run:{run_id}",
                min=f"({after}" if isinstance(after, str) else "-",
                count=_MAX_BUFFERED_EVENTS,
            )
            if isinstance(after, int) and after:
                rows = rows[after:]
            if rows:
                for event_id, fields in rows:
                    yield {**json.loads(fields["payload"]), "_cursor": event_id}
                return
        except Exception:
            _redis_failed()
    cached = _streams.get(run_id)
    if cached is None:
        return
    buffered = list(cached[1])
    acknowledged = next((seq for seq, _, redis_id in buffered if redis_id == after), None)
    for sequence, event, event_id in buffered:
        if isinstance(after, int):
            fresh = sequence > after
        elif acknowledged is not None:
            fresh = sequence > acknowledged
        else:
            fresh = event_id is not None and tuple(map(int, event_id.split("-"))) > tuple(
                map(int, after.split("-"))
            )
        if fresh:
            yield {**event, "_cursor": event_id or sequence}


def event_count(run_id: str) -> int:
    _prune()
    cached = _streams.get(run_id)
    return cached[1][-1][0] if cached and cached[1] else 0


def reset() -> None:
    _streams.clear()


async def close() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
