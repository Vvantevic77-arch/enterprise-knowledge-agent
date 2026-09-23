"""Append-only audit trail.

Every authorisation decision and every tenant-scoped mutation is recorded here.
Writes are best-effort: an audit failure must never roll back the business
operation that succeeded, but it is logged loudly so it cannot pass silently.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog

logger = logging.getLogger(__name__)

#: Actions that represent an authorisation decision rather than a mutation.
DECISION_ACTIONS = {"access.allowed", "access.denied", "approval.vote", "approval.executed"}


def record(
    db: AsyncSession,
    *,
    action: str,
    resource: str,
    org_id: str | None = None,
    actor_id: str | None = None,
    resource_id: str | None = None,
    decision: str = "allow",
    reason: str = "",
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditLog:
    """Stage an audit row on the session.

    Deliberately does not commit: the row lands in the same transaction as the
    change it describes, so the two cannot diverge.
    """
    entry = AuditLog(
        org_id=org_id,
        actor_id=actor_id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        decision=decision,
        reason=reason[:255],
        trace_id=trace_id,
        metadata_json=metadata or {},
    )
    db.add(entry)
    return entry


def record_denied(
    db: AsyncSession,
    *,
    action: str,
    resource: str,
    org_id: str | None,
    actor_id: str | None,
    reason: str,
    resource_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """A denied request is the signal that matters most in a payment system."""
    record(
        db,
        action=f"{action}.denied",
        resource=resource,
        org_id=org_id,
        actor_id=actor_id,
        resource_id=resource_id,
        decision="deny",
        reason=reason,
        trace_id=trace_id,
    )
