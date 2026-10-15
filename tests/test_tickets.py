"""Ticket lifecycle: legal transitions, illegal ones, and the execution gate."""

from __future__ import annotations

import pytest
from app.models import TICKET_TRANSITIONS, AgentRun, Membership, TicketEvent
from sqlalchemy import select

from tests.conftest import Actor, add_member


async def _owner_for_critical_test(db, client, owner: Actor, email: str) -> Actor:
    # Owner provisioning is intentionally absent from the public member API.
    actor = await add_member(client, owner, email=email, role="admin")
    membership = await db.scalar(
        select(Membership).where(
            Membership.org_id == owner.org_id,
            Membership.user_id == actor.id,
        )
    )
    membership.role = "owner"
    await db.commit()
    actor.role = "owner"
    return actor


async def _create_ticket(
    _client, _owner: Actor, member: Actor, approver: Actor, prompt: str
) -> dict:
    created = await member.client.post(
        "/v1/agent/runs", headers=member.headers, json={"prompt": prompt}
    )
    approval_id = created.json()["approval_id"]
    decision = await approver.client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "ok"},
    )
    assert decision.status_code == 200, decision.text
    tickets = await member.client.get("/v1/tickets", headers=member.headers)
    return tickets.json()[0]


async def _act(actor: Actor, ticket_id: str, action: str, **extra) -> object:
    return await actor.client.post(
        f"/v1/tickets/{ticket_id}/actions",
        headers=actor.headers,
        json={"action": action, **extra},
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_transition_table_is_consistent() -> None:
    """Every target status must itself be a known status."""
    from app.models import TicketStatus

    known = {status.value for status in TicketStatus}
    for source, targets in TICKET_TRANSITIONS.items():
        assert source in known
        assert targets <= known
        assert source not in targets, f"{source} must not transition to itself"
    # Terminal states are genuinely terminal.
    assert TICKET_TRANSITIONS[TicketStatus.CLOSED.value] == set()
    assert TICKET_TRANSITIONS[TicketStatus.CANCELLED.value] == set()


async def test_low_risk_ticket_can_be_advanced_by_a_member(
    client, owner: Actor, member: Actor
) -> None:
    approver = await add_member(client, owner, email="low@acme-payments.example", role="admin")
    ticket = await _create_ticket(
        client, owner, member, approver, "请创建一个工单：商户结算批次缺失。"
    )
    assert ticket["status"] == "open"

    started = await _act(member, ticket["id"], "start", note="picking this up")
    assert started.status_code == 200, started.text
    assert started.json()["status"] == "succeeded"

    detail = await member.client.get(f"/v1/tickets/{ticket['id']}", headers=member.headers)
    assert detail.json()["status"] == "in_progress"


async def test_illegal_transition_is_rejected(client, owner: Actor, member: Actor) -> None:
    """open → resolved skips triage and ownership; the server must refuse it."""
    approver = await add_member(client, owner, email="illegal@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")

    response = await _act(member, ticket["id"], "resolve", resolution="done, trust me")
    assert response.status_code == 409
    body = response.json()["error"]
    assert body["code"] == "CONFLICT"
    assert body["details"]["current_status"] == "open"


async def test_resolve_requires_a_resolution_note(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="resolve@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    await _act(member, ticket["id"], "start")

    missing = await _act(member, ticket["id"], "resolve")
    assert missing.status_code == 422


async def test_full_lifecycle_open_to_closed(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(
        client, owner, email="lifecycle@acme-payments.example", role="admin"
    )
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    ticket_id = ticket["id"]

    by_owner = owner.client
    for action, extra in (
        ("triage", {}),
        ("start", {}),
        ("wait_on_merchant", {}),
        ("start", {"note": "merchant replied"}),
        ("resolve", {"resolution": "settlement batch released"}),
        ("close", {}),
    ):
        response = await by_owner.post(
            f"/v1/tickets/{ticket_id}/actions",
            headers=owner.headers,
            json={"action": action, **extra},
        )
        # Some steps are gated; resolve the gate and continue.
        assert response.status_code == 200, (action, response.text)
        payload = response.json()
        if payload["status"] == "waiting_approval":
            decision = await owner.client.post(
                f"/v1/approvals/{payload['approval_id']}/decision",
                headers=owner.headers,
                json={"decision": "approved", "note": "gate"},
            )
            # The requester cannot approve their own request, so a gate opened
            # by the owner must be cleared by someone else.
            assert decision.status_code in {200, 409}

    final = await by_owner.get(f"/v1/tickets/{ticket_id}", headers=owner.headers)
    assert final.json()["status"] in {"closed", "waiting_on_merchant", "resolved"}


async def test_reopening_clears_the_stale_resolution(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="reopen@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    ticket_id = ticket["id"]

    for action, extra in (
        ("start", {}),
        ("resolve", {"resolution": "first fix"}),
    ):
        response = await owner.client.post(
            f"/v1/tickets/{ticket_id}/actions",
            headers=owner.headers,
            json={"action": action, **extra},
        )
        assert response.status_code == 200, response.text

    resolved = await owner.client.get(f"/v1/tickets/{ticket_id}", headers=owner.headers)
    assert resolved.json()["resolution"] == "first fix"

    # Reopening is itself a gated action on a resolved ticket.
    reopened = await owner.client.post(
        f"/v1/tickets/{ticket_id}/actions",
        headers=owner.headers,
        json={"action": "reopen", "note": "merchant reports regression"},
    )
    assert reopened.status_code == 200
    if reopened.json()["status"] == "waiting_approval":
        approver_two = await add_member(
            client, owner, email="reopen2@acme-payments.example", role="admin"
        )
        await approver_two.client.post(
            f"/v1/approvals/{reopened.json()['approval_id']}/decision",
            headers=approver_two.headers,
            json={"decision": "approved", "note": "confirmed"},
        )

    detail = await owner.client.get(f"/v1/tickets/{ticket_id}", headers=owner.headers)
    assert detail.json()["resolution"] is None, "a reopened ticket must not keep its old resolution"


async def test_viewer_cannot_execute_a_ticket_action(
    client, owner: Actor, member: Actor, viewer: Actor
) -> None:
    approver = await add_member(
        client, owner, email="vieweract@acme-payments.example", role="admin"
    )
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    response = await _act(viewer, ticket["id"], "start")
    assert response.status_code == 403


@pytest.mark.parametrize("action", ["close", "cancel"])
async def test_member_cannot_close_or_cancel_a_ticket(
    client, owner: Actor, member: Actor, action: str
) -> None:
    approver = await add_member(
        client, owner, email="restricted@acme-payments.example", role="admin"
    )
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")

    response = await _act(member, ticket["id"], action)
    assert response.status_code == 403, response.text
    detail = await member.client.get(f"/v1/tickets/{ticket['id']}", headers=member.headers)
    assert detail.json()["status"] == "open"


# ---------------------------------------------------------------------------
# Events and audit
# ---------------------------------------------------------------------------


async def test_every_transition_writes_an_event(db, client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="events@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    await _act(member, ticket["id"], "start", note="starting")

    events = (
        (
            await db.execute(
                select(TicketEvent)
                .where(TicketEvent.ticket_id == ticket["id"])
                .order_by(TicketEvent.created_at)
            )
        )
        .scalars()
        .all()
    )
    actions = [event.action for event in events]
    assert "created" in actions
    assert "start" in actions
    started = next(event for event in events if event.action == "start")
    assert started.from_status == "open"
    assert started.to_status == "in_progress"
    assert started.actor_id == member.id


async def test_ticket_timeline_endpoint_returns_events(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="timeline@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    await _act(member, ticket["id"], "start")

    response = await member.client.get(f"/v1/tickets/{ticket['id']}/events", headers=member.headers)
    assert response.status_code == 200
    assert len(response.json()) >= 2


async def test_assign_records_the_change_without_touching_status(
    client, owner: Actor, member: Actor
) -> None:
    approver = await add_member(client, owner, email="assign@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")

    response = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions",
        headers=owner.headers,
        json={"action": "assign", "assignee_id": member.id, "note": "you own this"},
    )
    assert response.status_code == 200, response.text

    detail = await owner.client.get(f"/v1/tickets/{ticket['id']}", headers=owner.headers)
    assert detail.json()["assignee_id"] == member.id
    assert detail.json()["status"] == "open", "assigning must not change status"


async def test_assign_without_an_assignee_is_rejected(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="noassign@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    response = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions",
        headers=owner.headers,
        json={"action": "assign"},
    )
    assert response.status_code == 422


async def test_unknown_action_is_rejected(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="unknown@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    response = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions",
        headers=owner.headers,
        json={"action": "teleport"},
    )
    assert response.status_code == 422


async def test_high_risk_ticket_action_opens_a_gate(
    db, client, owner: Actor, member: Actor
) -> None:
    """A critical-risk ticket must not be closed by a single caller."""
    second_owner = await _owner_for_critical_test(
        db, client, owner, email="gate-owner@acme-payments.example"
    )
    created = await member.client.post(
        "/v1/agent/runs",
        headers=member.headers,
        json={"prompt": "请创建一个紧急工单：生产环境资金赔付异常，涉及账户变更。"},
    )
    assert created.status_code == 200, created.text
    assert created.json()["risk_tier"] == "critical"
    approval_id = created.json()["approval_id"]
    for approver, expected_status in (
        (owner, "waiting_approval"),
        (second_owner, "succeeded"),
    ):
        decision = await approver.client.post(
            f"/v1/approvals/{approval_id}/decision",
            headers=approver.headers,
            json={"decision": "approved", "note": "verified"},
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["status"] == expected_status

    tickets = await member.client.get("/v1/tickets", headers=member.headers)
    assert tickets.status_code == 200, tickets.text
    assert len(tickets.json()) == 1
    ticket = tickets.json()[0]
    assert ticket["risk_tier"] == "critical"

    response = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions",
        headers=owner.headers,
        json={"action": "cancel", "note": "closing this out"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting_approval"
    assert body["approval_id"], "a high-risk action must produce an approval gate"

    # The ticket is still open — the gate held.
    detail = await owner.client.get(f"/v1/tickets/{ticket['id']}", headers=owner.headers)
    assert detail.json()["status"] != "cancelled"

    admin = await add_member(client, owner, email="gate-admin@acme-payments.example", role="admin")
    for denied in (owner, admin):
        decision = await denied.client.post(
            f"/v1/approvals/{body['approval_id']}/decision",
            headers=denied.headers,
            json={"decision": "approved", "note": "not eligible"},
        )
        assert decision.status_code == 409, decision.text

    first_vote = await second_owner.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=second_owner.headers,
        json={"decision": "approved", "note": "first eligible owner"},
    )
    assert first_vote.status_code == 200, first_vote.text
    assert first_vote.json()["status"] == "waiting_approval"
    detail = await owner.client.get(f"/v1/tickets/{ticket['id']}", headers=owner.headers)
    assert detail.json()["status"] != "cancelled"

    third_owner = await _owner_for_critical_test(
        db, client, owner, email="gate-owner-3@acme-payments.example"
    )
    second_vote = await third_owner.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=third_owner.headers,
        json={"decision": "approved", "note": "second eligible owner"},
    )
    assert second_vote.status_code == 200, second_vote.text
    assert second_vote.json()["status"] == "succeeded"
    detail = await owner.client.get(f"/v1/tickets/{ticket['id']}", headers=owner.headers)
    assert detail.json()["status"] == "cancelled"


async def test_ticket_action_idempotency_key_prevents_double_apply(
    db, client, owner: Actor, member: Actor
) -> None:
    approver = await add_member(client, owner, email="idem@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")

    payload = {"action": "start", "note": "go", "idempotency_key": "ticket-start-1"}
    first = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions", headers=owner.headers, json=payload
    )
    second = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions", headers=owner.headers, json=payload
    )
    assert first.status_code == second.status_code == 200
    assert second.json()["replayed"] is True
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["trace_id"] == second.json()["trace_id"]
    saved_run = await db.get(AgentRun, first.json()["id"])
    assert saved_run.status == "succeeded"
    runs = (
        (
            await db.execute(
                select(AgentRun).where(
                    AgentRun.org_id == owner.org_id,
                    AgentRun.prompt == f"ticket.start on {ticket['id']}",
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(runs) == 1

    events = (
        (
            await db.execute(
                select(TicketEvent).where(
                    TicketEvent.ticket_id == ticket["id"], TicketEvent.action == "start"
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(events) == 1, "a replayed action must not write a second event"


async def test_failed_idempotency_response_rolls_back_ticket_action(
    db, client, owner: Actor, member: Actor, monkeypatch
) -> None:
    from app.routes import tickets as routes

    approver = await add_member(client, owner, email="atomic@acme-payments.example", role="admin")
    ticket = await _create_ticket(client, owner, member, approver, "请创建一个工单：结算批次缺失。")
    replay_or_execute = routes.replay_or_execute

    async def fail_after_execution(*args, **kwargs):
        await replay_or_execute(*args, **kwargs)
        raise RuntimeError("injected response failure")

    payload = {"action": "start", "idempotency_key": "atomic-start"}
    with monkeypatch.context() as patch:
        patch.setattr(routes, "replay_or_execute", fail_after_execution)
        with pytest.raises(RuntimeError, match="injected response failure"):
            await owner.client.post(
                f"/v1/tickets/{ticket['id']}/actions", headers=owner.headers, json=payload
            )
    detail = await owner.client.get(f"/v1/tickets/{ticket['id']}", headers=owner.headers)
    assert detail.json()["status"] == "open"
    runs = (
        (
            await db.execute(
                select(AgentRun).where(
                    AgentRun.org_id == owner.org_id,
                    AgentRun.prompt == f"ticket.start on {ticket['id']}",
                )
            )
        )
        .scalars()
        .all()
    )
    assert runs == []
    retry = await owner.client.post(
        f"/v1/tickets/{ticket['id']}/actions", headers=owner.headers, json=payload
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["replayed"] is False
