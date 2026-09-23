"""Idempotency for write endpoints and gated tool execution.

The contract, matching what payment APIs expect from a client:

* First request with a key: perform the work, store the response.
* Repeat with the **same** key and the **same** payload: return the stored
  response, do not perform the work again.
* Repeat with the same key but a **different** payload: reject with a conflict,
  because the caller is reusing a key for a different intent.

The stored response is keyed by ``(org_id, scope, key)`` so two tenants can use
the same client-generated key without colliding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core_exceptions import ConflictError
from app.models import IdempotencyRecord

T = TypeVar("T")


def fingerprint(payload: Any) -> str:
    """Stable hash of a request payload used to detect key reuse."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def replay_or_execute(
    db: AsyncSession,
    *,
    org_id: str,
    scope: str,
    key: str | None,
    payload: Any,
    execute: Callable[[], Awaitable[dict[str, Any]]],
) -> tuple[dict[str, Any], bool]:
    """Run ``execute`` at most once per ``(org_id, scope, key)``.

    Returns ``(response, replayed)``. When ``key`` is None the work always runs,
    which is the correct behaviour for callers that did not ask for idempotency.
    """
    if not key:
        return await execute(), False

    request_hash = fingerprint(payload)
    existing = (
        await db.execute(
            select(IdempotencyRecord).where(
                IdempotencyRecord.org_id == org_id,
                IdempotencyRecord.scope == scope,
                IdempotencyRecord.key == key,
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        if existing.request_hash != request_hash:
            raise ConflictError(
                message="idempotency key was already used with a different payload",
                details={"scope": scope, "key": key},
            )
        return dict(existing.response_json), True

    response = await execute()

    record = IdempotencyRecord(
        org_id=org_id,
        scope=scope,
        key=key,
        request_hash=request_hash,
        response_json=response,
    )
    db.add(record)
    try:
        await db.flush()
    except IntegrityError:
        # A concurrent request inserted the same key between our read and write.
        # Roll back to that row's result rather than double-executing the work.
        await db.rollback()
        winner = (
            await db.execute(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.org_id == org_id,
                    IdempotencyRecord.scope == scope,
                    IdempotencyRecord.key == key,
                )
            )
        ).scalar_one()
        if winner.request_hash != request_hash:
            raise ConflictError(
                message="idempotency key was already used with a different payload",
                details={"scope": scope, "key": key},
            ) from None
        return dict(winner.response_json), True

    return response, False
