"""Audit log, usage accounting, and health."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select

from app.auth import Db, require
from app.config import settings
from app.models import (
    PGVECTOR_AVAILABLE,
    AgentRun,
    AuditLog,
    Document,
    KnowledgeBase,
    UsageEvent,
)
from app.permissions import Action, Principal, Resource
from app.schemas import (
    AuditLogRead,
    HealthRead,
    PaginatedAuditLogs,
    UsageSummary,
)

router = APIRouter(tags=["ops"])


@router.get("/audit-logs", response_model=PaginatedAuditLogs)
async def list_audit_logs(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.AUDIT_LOG)),
    decision: str | None = Query(default=None, pattern="^(allow|deny)$"),
    action: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> Any:
    """Tenant audit trail. Denials are included — they are the interesting rows."""
    stmt = select(AuditLog).where(AuditLog.org_id == principal.org_id)
    if decision:
        stmt = stmt.where(AuditLog.decision == decision)
    if action:
        stmt = stmt.where(AuditLog.action == action)

    total = await db.scalar(
        select(func.count()).select_from(stmt.subquery())
    )
    rows = (
        await db.execute(
            stmt.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit)
        )
    ).scalars().all()
    return PaginatedAuditLogs(
        items=[
            AuditLogRead(
                id=row.id,
                actor_id=row.actor_id,
                action=row.action,
                resource=row.resource,
                resource_id=row.resource_id,
                decision=row.decision,
                reason=row.reason,
                trace_id=row.trace_id,
                created_at=row.created_at,
            )
            for row in rows
        ],
        total=int(total or 0),
    )


@router.get("/usage", response_model=UsageSummary)
async def usage_summary(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.USAGE)),
) -> Any:
    rows = (
        await db.execute(
            select(UsageEvent).where(UsageEvent.org_id == principal.org_id)
        )
    ).scalars().all()
    runs = await db.scalar(
        select(func.count(AgentRun.id)).where(AgentRun.org_id == principal.org_id)
    )
    return UsageSummary(
        runs=int(runs or 0),
        input_tokens=sum(row.input_tokens for row in rows),
        output_tokens=sum(row.output_tokens for row in rows),
        estimated_cost_usd=round(sum(float(row.cost_usd or 0.0) for row in rows), 6),
    )


@router.get("/health", response_model=HealthRead)
async def health(db: Db) -> Any:
    """Liveness plus a summary of which backends are actually active.

    Reports embedding provider and vector backend because a silent fallback to
    deterministic embeddings would degrade retrieval without failing anything —
    this endpoint makes that state visible.
    """
    document_count = await db.scalar(select(func.count(Document.id)))
    kb_count = await db.scalar(select(func.count(KnowledgeBase.id)))
    vector_backend = (
        "pgvector-hnsw"
        if PGVECTOR_AVAILABLE and not settings.is_sqlite
        else "json-python-scan"
    )
    return HealthRead(
        status="ok",
        service="payment-operations-agent",
        version="0.2.0",
        database="sqlite" if settings.is_sqlite else "postgresql",
        embedding_provider=settings.embedding_provider,
        vector_backend=vector_backend,
        llm_configured=bool(settings.llm_api_key),
    )


@router.get("/stats")
async def workspace_stats(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.USAGE)),
) -> dict[str, int]:
    """Small dashboard payload for the operator UI."""
    from app.models import ApprovalRequest, Ticket

    kbs = await db.scalar(
        select(func.count(KnowledgeBase.id)).where(
            KnowledgeBase.org_id == principal.org_id
        )
    )
    docs = await db.scalar(
        select(func.count(Document.id)).where(Document.org_id == principal.org_id)
    )
    runs = await db.scalar(
        select(func.count(AgentRun.id)).where(AgentRun.org_id == principal.org_id)
    )
    tickets = await db.scalar(
        select(func.count(Ticket.id)).where(Ticket.org_id == principal.org_id)
    )
    pending = await db.scalar(
        select(func.count(ApprovalRequest.id)).where(
            ApprovalRequest.org_id == principal.org_id,
            ApprovalRequest.status == "pending",
        )
    )
    return {
        "knowledge_bases": int(kbs or 0),
        "documents": int(docs or 0),
        "runs": int(runs or 0),
        "tickets": int(tickets or 0),
        "pending_approvals": int(pending or 0),
    }
