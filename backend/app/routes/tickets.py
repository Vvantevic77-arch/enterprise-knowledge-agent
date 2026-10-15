"""Ticket execution endpoints.

Every state change goes through ``services.tickets.execute_ticket_action``,
which validates the transition against ``TICKET_TRANSITIONS``. A request that
is legal but high-impact can be routed through the approval gate instead of
executing immediately — see ``_requires_approval`` below.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from app import audit
from app.auth import Db, require
from app.core_exceptions import AuthorizationError, NotFoundError
from app.idempotency import replay_or_execute
from app.models import AgentRun, ApprovalRequest, RunStatus, Ticket
from app.permissions import Action, Principal, Resource, required_approvals
from app.schemas import (
    AgentRunRead,
    TicketActionRequest,
    TicketEventRead,
    TicketRead,
)
from app.services.policy import classify_risk
from app.services.tickets import (
    RESTRICTED_ACTIONS,
    execute_ticket_action,
    get_ticket,
    list_ticket_events,
    may_perform,
)

router = APIRouter(tags=["tickets"])


def _ticket_read(ticket: Ticket) -> TicketRead:
    return TicketRead(
        id=ticket.id,
        title=ticket.title,
        description=ticket.description,
        priority=ticket.priority,
        status=ticket.status,
        risk_tier=ticket.risk_tier,
        assignee_id=ticket.assignee_id,
        run_id=ticket.run_id,
        sla_due_at=ticket.sla_due_at,
        resolution=ticket.resolution,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
    )


@router.get("/tickets", response_model=list[TicketRead])
async def list_tickets(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.TICKET)),
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
) -> Any:
    stmt = select(Ticket).where(Ticket.org_id == principal.org_id)
    if status_filter:
        stmt = stmt.where(Ticket.status == status_filter)
    rows = (await db.execute(stmt.order_by(Ticket.created_at.desc()).limit(limit))).scalars().all()
    return [_ticket_read(row) for row in rows]


@router.get("/tickets/{ticket_id}", response_model=TicketRead)
async def get_ticket_detail(
    ticket_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.TICKET)),
) -> Any:
    return _ticket_read(await get_ticket(db, org_id=principal.org_id, ticket_id=ticket_id))


@router.get("/tickets/{ticket_id}/events", response_model=list[TicketEventRead])
async def ticket_timeline(
    ticket_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.TICKET)),
) -> Any:
    events = await list_ticket_events(db, org_id=principal.org_id, ticket_id=ticket_id)
    return [
        TicketEventRead(
            id=event.id,
            action=event.action,
            from_status=event.from_status,
            to_status=event.to_status,
            note=event.note,
            actor_id=event.actor_id,
            created_at=event.created_at,
        )
        for event in events
    ]


def _requires_approval(ticket: Ticket, action: str) -> str | None:
    """Decide whether this ticket action needs a human gate.

    Closing or cancelling work another team may depend on is gated, and so is
    any action on a ticket whose own risk tier is high or critical. Everything
    else executes inline.
    """
    if ticket.risk_tier in {"high", "critical"} or action in RESTRICTED_ACTIONS:
        # Reuse the policy classifier so the tier definition stays in one place.
        return max(
            (ticket.risk_tier, classify_risk(f"ticket {action}", "ticket")),
            key=lambda tier: ["low", "medium", "high", "critical"].index(tier),
        )
    return None


@router.post("/tickets/{ticket_id}/actions", response_model=AgentRunRead)
async def apply_ticket_action(
    ticket_id: str,
    payload: TicketActionRequest,
    db: Db,
    principal: Principal = Depends(require(Action.EXECUTE, Resource.TICKET)),
) -> Any:
    """Execute a ticket action, or open an approval gate for a risky one.

    Returns a run whose ``status`` is ``succeeded`` when the action executed, or
    ``waiting_approval`` with an ``approval_id`` when it was gated.
    """
    ticket = await get_ticket(db, org_id=principal.org_id, ticket_id=ticket_id)
    if not may_perform(principal.role, payload.action):
        audit.record_denied(
            db,
            action=f"ticket.{payload.action}",
            resource="ticket",
            org_id=principal.org_id,
            actor_id=principal.user_id,
            resource_id=ticket_id,
            reason=f"role {principal.role!r} cannot {payload.action}",
        )
        raise AuthorizationError(
            message=f"Role '{principal.role}' cannot {payload.action} a ticket"
        )

    risk_tier = _requires_approval(ticket, payload.action)

    if risk_tier is not None:
        return await _gate_ticket_action(
            db, principal=principal, ticket=ticket, payload=payload, risk_tier=risk_tier
        )

    async def _execute() -> dict[str, Any]:
        run = AgentRun(
            org_id=principal.org_id,
            user_id=principal.user_id,
            prompt=f"ticket.{payload.action} on {ticket_id}",
            status=RunStatus.RUNNING.value,
            intent="ticket",
            risk_tier=ticket.risk_tier,
        )
        db.add(run)
        await db.flush()
        updated = await execute_ticket_action(
            db,
            org_id=principal.org_id,
            actor_id=principal.user_id,
            ticket_id=ticket_id,
            action=payload.action,
            note=payload.note,
            resolution=payload.resolution,
            assignee_id=payload.assignee_id,
        )
        run.status = RunStatus.SUCCEEDED.value
        run.answer = f"工单 {updated.id} 已更新为 {updated.status}。"
        return AgentRunRead(
            id=run.id,
            trace_id=run.trace_id,
            status=run.status,
            answer=run.answer,
            intent="ticket",
            risk_tier=updated.risk_tier,
        ).model_dump(mode="json")

    response, replayed = await replay_or_execute(
        db,
        org_id=principal.org_id,
        scope=f"ticket.action:{payload.action}",
        key=payload.idempotency_key,
        payload={"ticket_id": ticket_id, **payload.model_dump(exclude={"idempotency_key"})},
        execute=_execute,
    )

    # Reservation, run, transition and response become durable together.
    await db.commit()
    if "id" not in response:
        # Responses stored by earlier versions contain a trace, not a run ID.
        previous_run = await db.scalar(
            select(AgentRun).where(
                AgentRun.org_id == principal.org_id, AgentRun.trace_id == response["trace_id"]
            )
        )
        if previous_run is None:
            raise NotFoundError(message="original ticket action run not found")
        response = {
            "id": previous_run.id,
            "trace_id": previous_run.trace_id,
            "status": RunStatus.SUCCEEDED.value,
            "intent": "ticket",
            "risk_tier": previous_run.risk_tier,
            "answer": f"工单 {response['ticket_id']} 已更新为 {response['ticket_status']}。",
        }
    return AgentRunRead.model_validate({**response, "replayed": replayed})


async def _gate_ticket_action(
    db: Db,
    *,
    principal: Principal,
    ticket: Ticket,
    payload: TicketActionRequest,
    risk_tier: str,
) -> Any:
    """Pause a risky ticket action for approval instead of executing it."""
    run = AgentRun(
        org_id=principal.org_id,
        user_id=principal.user_id,
        prompt=f"ticket.{payload.action} on {ticket.id}",
        status=RunStatus.WAITING_APPROVAL.value,
        intent="ticket",
        risk_tier=risk_tier,
    )
    db.add(run)
    await db.flush()

    approval = ApprovalRequest(
        run_id=run.id,
        org_id=principal.org_id,
        requested_by=principal.user_id,
        tool_name=f"ticket.{payload.action}",
        arguments={
            "ticket_id": ticket.id,
            "note": payload.note,
            "resolution": payload.resolution,
            "assignee_id": payload.assignee_id,
        },
        risk_tier=risk_tier,
        required_approvals=required_approvals(risk_tier),
        idempotency_key=(
            payload.idempotency_key or f"{ticket.id}:{payload.action}:{principal.user_id}"
        ),
    )
    db.add(approval)
    await db.commit()

    from app.services.events import publish_event

    await publish_event(
        run.id,
        {
            "type": "approval.required",
            "approval_id": approval.id,
            "tool": approval.tool_name,
            "risk_tier": risk_tier,
            "required_approvals": approval.required_approvals,
        },
    )
    return AgentRunRead(
        id=run.id,
        trace_id=run.trace_id,
        status=RunStatus.WAITING_APPROVAL.value,
        answer=f"该操作属于 {risk_tier} 风险，已提交审批。",
        intent="ticket",
        risk_tier=risk_tier,
        approval_id=approval.id,
    )


@router.get("/tickets/{ticket_id}/approvals", response_model=list[dict])
async def ticket_approvals(
    ticket_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.APPROVAL)),
) -> Any:
    """Approvals that have ever targeted this ticket."""
    await get_ticket(db, org_id=principal.org_id, ticket_id=ticket_id)
    rows = (
        (
            await db.execute(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.org_id == principal.org_id,
                    ApprovalRequest.tool_name.like("ticket.%"),
                )
                .order_by(ApprovalRequest.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": row.id,
            "tool_name": row.tool_name,
            "status": row.status,
            "risk_tier": row.risk_tier,
        }
        for row in rows
        if str(dict(row.arguments or {}).get("ticket_id")) == ticket_id
    ]
