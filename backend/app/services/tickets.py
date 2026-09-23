"""Ticket lifecycle.

Tickets are the part of this system that touches real money and real merchant
commitments, so the state machine is explicit and enforced rather than being a
free-form string column. ``execute_ticket_action`` is the only function that
mutates ``Ticket.status``; everything else reads.

Transitions live in ``models.TICKET_TRANSITIONS``. Anything not listed there is
rejected with a 409, which means an LLM tool call cannot invent a status such as
"done" and have it silently persisted.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import audit
from app.core_exceptions import ConflictError, NotFoundError, ValidationError
from app.models import TICKET_TRANSITIONS, Role, Ticket, TicketEvent, TicketStatus, utcnow

#: Action name → the status it moves the ticket to.
ACTION_TARGET: dict[str, str] = {
    "triage": TicketStatus.TRIAGED.value,
    "start": TicketStatus.IN_PROGRESS.value,
    "wait_on_merchant": TicketStatus.WAITING_ON_MERCHANT.value,
    "resolve": TicketStatus.RESOLVED.value,
    "close": TicketStatus.CLOSED.value,
    "reopen": TicketStatus.IN_PROGRESS.value,
    "cancel": TicketStatus.CANCELLED.value,
}

#: Actions that only an owner or admin may perform, because they close out or
#: discard work that another team may still depend on.
RESTRICTED_ACTIONS = frozenset({"close", "cancel"})


async def get_ticket(db: AsyncSession, *, org_id: str, ticket_id: str) -> Ticket:
    """Tenant-scoped fetch. A ticket in another org is a 404, not a 403."""
    ticket = (
        await db.execute(
            select(Ticket).where(Ticket.id == ticket_id, Ticket.org_id == org_id)
        )
    ).scalar_one_or_none()
    if ticket is None:
        raise NotFoundError(message="ticket not found")
    return ticket


def can_transition(current: str, target: str) -> bool:
    return target in TICKET_TRANSITIONS.get(current, set())


async def execute_ticket_action(
    db: AsyncSession,
    *,
    org_id: str,
    actor_id: str,
    ticket_id: str,
    action: str,
    note: str = "",
    resolution: str = "",
    assignee_id: str | None = None,
) -> Ticket:
    """Advance a ticket, validating the transition and writing an event row.

    ``assign`` is handled separately because it does not change status — it only
    sets the owner, which keeps the state machine orthogonal to routing.
    """
    ticket = await get_ticket(db, org_id=org_id, ticket_id=ticket_id)

    if action == "assign":
        if not assignee_id:
            raise ValidationError(message="assignee_id is required for the assign action")
        previous = ticket.assignee_id
        ticket.assignee_id = assignee_id
        ticket.updated_at = utcnow()
        db.add(
            TicketEvent(
                ticket_id=ticket.id,
                org_id=org_id,
                actor_id=actor_id,
                action="assigned",
                from_status=ticket.status,
                to_status=ticket.status,
                note=note or f"assignee {previous or 'none'} -> {assignee_id}",
            )
        )
        audit.record(
            db,
            action="ticket.assign",
            resource="ticket",
            org_id=org_id,
            actor_id=actor_id,
            resource_id=ticket.id,
            metadata={"assignee_id": assignee_id},
        )
        await db.flush()
        return ticket

    if action not in ACTION_TARGET:
        raise ValidationError(
            message=f"unknown ticket action: {action}",
            details={"allowed": sorted(ACTION_TARGET) + ["assign"]},
        )

    target = ACTION_TARGET[action]
    if not can_transition(ticket.status, target):
        raise ConflictError(
            message=f"cannot {action} a ticket in status '{ticket.status}'",
            details={
                "current_status": ticket.status,
                "target_status": target,
                "allowed_targets": sorted(TICKET_TRANSITIONS.get(ticket.status, set())),
            },
        )

    if action == "resolve" and not resolution.strip():
        raise ValidationError(message="resolution is required when resolving a ticket")

    previous = ticket.status
    ticket.status = target
    ticket.updated_at = utcnow()

    if target == TicketStatus.RESOLVED.value:
        ticket.resolution = resolution.strip()
        ticket.resolved_at = utcnow()
    if target == TicketStatus.CLOSED.value:
        ticket.closed_at = utcnow()
    if action == "reopen":
        # Reopening clears the resolution so the next resolve is a fresh answer
        # rather than a stale one being silently reused.
        ticket.resolution = None
        ticket.resolved_at = None

    db.add(
        TicketEvent(
            ticket_id=ticket.id,
            org_id=org_id,
            actor_id=actor_id,
            action=action,
            from_status=previous,
            to_status=target,
            note=note or resolution,
        )
    )
    audit.record(
        db,
        action=f"ticket.{action}",
        resource="ticket",
        org_id=org_id,
        actor_id=actor_id,
        resource_id=ticket.id,
        metadata={"from": previous, "to": target},
    )
    await db.flush()
    return ticket


async def list_ticket_events(
    db: AsyncSession, *, org_id: str, ticket_id: str
) -> list[TicketEvent]:
    # Verify tenant scope first so event history cannot be enumerated across
    # organisations by guessing ticket IDs.
    await get_ticket(db, org_id=org_id, ticket_id=ticket_id)
    rows = (
        await db.execute(
            select(TicketEvent)
            .where(TicketEvent.ticket_id == ticket_id, TicketEvent.org_id == org_id)
            .order_by(TicketEvent.created_at)
        )
    ).scalars().all()
    return list(rows)


def may_perform(role: str, action: str) -> bool:
    if action in RESTRICTED_ACTIONS:
        return role in {Role.OWNER.value, Role.ADMIN.value}
    return role in {Role.OWNER.value, Role.ADMIN.value, Role.MEMBER.value}
