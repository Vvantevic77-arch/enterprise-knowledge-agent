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

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core_exceptions import ConflictError
from app.models import IdempotencyRecord

_PENDING_WAIT_SECONDS = 30.0


def fingerprint(payload: Any) -> str:
    """Stable hash of a request payload used to detect key reuse."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def _wait_for_response(db: AsyncSession, record: IdempotencyRecord) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + _PENDING_WAIT_SECONDS
    record_id = record.id
    await db.commit()  # Release the losing INSERT's lock before waiting.
    while True:
        response = await db.scalar(
            select(IdempotencyRecord.response_json).where(IdempotencyRecord.id == record_id)
        )
        await db.commit()  # Do not retain a connection or a read snapshot while sleeping.
        if response:
            return dict(response)
        if response is None:
            raise ConflictError(message="previous request failed; retry with the same key")
        if asyncio.get_running_loop().time() >= deadline:
            # Never steal a pending key: a slow worker may still be executing.
            raise ConflictError(
                message="request with this idempotency key is still in progress",
                details={"scope": record.scope, "key": record.key, "retry_after_seconds": 1},
            )
        await asyncio.sleep(0.05)


async def replay_or_execute(
    db: AsyncSession,
    *,
    org_id: str,
    scope: str,
    key: str | None,
    payload: Any,
    execute: Callable[[], Awaitable[dict[str, Any]]],
    release_reservation: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Run ``execute`` at most once per ``(org_id, scope, key)``.

    Returns ``(response, replayed)``. When ``key`` is None the work always runs,
    which is the correct behaviour for callers that did not ask for idempotency.

    The caller must commit the result and business changes together. ``execute``
    must not commit: the uncommitted reservation serializes requests for this
    key, and a failure rolls it back with the business operation.

    ``release_reservation`` is for the read-only RAG/LLM preparation in an agent
    run. It commits a pending reservation before the callback, and atomically
    commits its result with the business changes afterwards. Duplicate callers
    wait without holding locks; callback failures roll back and free the key.
    Do not use it when the caller has already staged business writes.
    """
    if not key:
        return await execute(), False

    request_hash = fingerprint(payload)
    dialect = db.get_bind().dialect.name
    insert = sqlite_insert if dialect == "sqlite" else pg_insert
    reservation_id = await db.scalar(
        insert(IdempotencyRecord)
        .values(
            org_id=org_id,
            scope=scope,
            key=key,
            request_hash=request_hash,
            response_json={},
        )
        .on_conflict_do_nothing(index_elements=["org_id", "scope", "key"])
        .returning(IdempotencyRecord.id)
    )
    existing = (
        await db.execute(
            select(IdempotencyRecord).where(
                IdempotencyRecord.org_id == org_id,
                IdempotencyRecord.scope == scope,
                IdempotencyRecord.key == key,
            )
        )
    ).scalar_one()

    if reservation_id is None:
        if existing.request_hash != request_hash:
            raise ConflictError(
                message="idempotency key was already used with a different payload",
                details={"scope": scope, "key": key},
            )
        if release_reservation and not existing.response_json:
            return await _wait_for_response(db, existing), True
        return dict(existing.response_json), True

    try:
        if release_reservation:
            await db.commit()
        response = await execute()
        existing.response_json = response
        await db.flush()
        if release_reservation:
            await db.commit()
    except BaseException:
        if release_reservation:
            await db.rollback()
            saved = await db.scalar(
                select(IdempotencyRecord.response_json).where(
                    IdempotencyRecord.id == reservation_id
                )
            )
            if saved == {}:
                await db.execute(
                    delete(IdempotencyRecord).where(IdempotencyRecord.id == reservation_id)
                )
            await db.commit()
        raise
    return response, False
