"""Approval gating: risk tiering, dual control, idempotency, and resume."""

from __future__ import annotations

import asyncio

import pytest
from app.models import (
    AgentRun,
    ApprovalRequest,
    ApprovalVote,
    IdempotencyRecord,
    Ticket,
    TicketEvent,
)
from sqlalchemy import func, select

from tests.conftest import Actor, add_member


async def _start_run(actor: Actor, prompt: str, idempotency_key: str | None = None) -> dict:
    body: dict = {"prompt": prompt}
    if idempotency_key:
        body["idempotency_key"] = idempotency_key
    response = await actor.client.post("/v1/agent/runs", headers=actor.headers, json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Gating behaviour
# ---------------------------------------------------------------------------


async def test_ordinary_question_is_answered_without_an_approval(member: Actor) -> None:
    body = await _start_run(member, "退款多久到账？")
    assert body["status"] == "succeeded"
    assert body["approval_id"] is None
    assert body["answer"]
    assert body["intent"] == "refund"


async def test_ticket_request_pauses_for_approval(member: Actor) -> None:
    body = await _start_run(member, "请创建一个工单：结算批次缺失。")
    assert body["status"] == "waiting_approval"
    assert body["approval_id"]
    assert body["risk_tier"] == "medium"
    # Nothing has been created yet — the pause point is durable, not cosmetic.
    tickets = await member.client.get("/v1/tickets", headers=member.headers)
    assert tickets.json() == []


async def test_refund_ticket_is_high_risk_and_needs_two_approvers(member: Actor) -> None:
    body = await _start_run(member, "请创建一个紧急工单：商户退款失败，涉及生产环境。")
    assert body["status"] == "waiting_approval"
    assert body["risk_tier"] == "high"

    approvals = await member.client.get("/v1/approvals", headers=member.headers)
    row = next(a for a in approvals.json() if a["id"] == body["approval_id"])
    assert row["required_approvals"] == 2
    assert row["approvals_received"] == 0


async def test_answer_contains_citations_from_the_corpus(member: Actor) -> None:
    body = await _start_run(member, "支付接口返回 401 怎么排查？")
    assert body["status"] == "succeeded"
    assert body["citations"], "a grounded answer must cite retrieved chunks"
    assert all(citation["filename"] for citation in body["citations"])


# ---------------------------------------------------------------------------
# Separation of duties
# ---------------------------------------------------------------------------


async def test_requester_cannot_approve_their_own_request(member: Actor) -> None:
    body = await _start_run(member, "请创建一个工单：结算批次缺失。")
    response = await member.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=member.headers,
        json={"decision": "approved", "note": "self-approval attempt"},
    )
    # A member lacks approve permission at all, so this is refused before the
    # self-approval rule is even reached.
    assert response.status_code in {403, 409}


async def test_admin_cannot_approve_a_request_they_raised(admin: Actor) -> None:
    """The separation-of-duties rule must bind even for a privileged role."""
    body = await _start_run(admin, "请创建一个工单：商户结算批次缺失。")
    response = await admin.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=admin.headers,
        json={"decision": "approved", "note": "self-approval"},
    )
    assert response.status_code == 409
    assert "own request" in response.json()["error"]["message"]


async def test_high_risk_needs_two_distinct_approvers(client, owner: Actor, member: Actor) -> None:
    """One approval is not enough for a high-risk action, and one person
    cannot supply both."""
    approver_one = await add_member(client, owner, email="app1@acme-payments.example", role="admin")
    approver_two = await add_member(client, owner, email="app2@acme-payments.example", role="admin")

    body = await _start_run(member, "请创建一个紧急工单：生产环境退款失败。")
    approval_id = body["approval_id"]

    first = await approver_one.client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=approver_one.headers,
        json={"decision": "approved", "note": "first"},
    )
    assert first.status_code == 200
    # Quorum not reached: the run stays paused and no ticket exists.
    assert first.json()["status"] == "waiting_approval"
    assert (await member.client.get("/v1/tickets", headers=member.headers)).json() == []

    repeat = await approver_one.client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=approver_one.headers,
        json={"decision": "approved", "note": "trying again"},
    )
    assert repeat.status_code == 409, "one person must not satisfy a two-person quorum"

    second = await approver_two.client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=approver_two.headers,
        json={"decision": "approved", "note": "second"},
    )
    assert second.status_code == 200
    assert second.json()["status"] == "succeeded"

    tickets = await member.client.get("/v1/tickets", headers=member.headers)
    assert len(tickets.json()) == 1


async def test_single_rejection_blocks_execution(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="rejecter@acme-payments.example", role="admin")
    body = await _start_run(member, "请创建一个工单：商户结算批次缺失。")

    response = await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "rejected", "note": "not a real incident"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert (await member.client.get("/v1/tickets", headers=member.headers)).json() == []


async def test_decided_approval_cannot_be_decided_again(
    client, owner: Actor, member: Actor
) -> None:
    approver = await add_member(client, owner, email="once@acme-payments.example", role="admin")
    body = await _start_run(member, "请创建一个工单：结算批次缺失。")
    first = await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "ok"},
    )
    assert first.status_code == 200
    second = await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "again"},
    )
    assert second.status_code == 409


async def test_member_cannot_decide_an_approval(member: Actor) -> None:
    body = await _start_run(member, "请创建一个工单：结算批次缺失。")
    response = await member.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=member.headers,
        json={"decision": "approved", "note": ""},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Execution and resume
# ---------------------------------------------------------------------------


async def test_approval_executes_the_tool_and_resumes_the_run(
    db, client, owner: Actor, member: Actor
) -> None:
    approver = await add_member(client, owner, email="exec@acme-payments.example", role="admin")
    body = await _start_run(member, "请创建一个工单：商户结算批次缺失。")

    response = await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "verified"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert "已创建工单" in payload["answer"]

    tickets = (await member.client.get("/v1/tickets", headers=member.headers)).json()
    assert len(tickets) == 1
    assert tickets[0]["status"] == "open"

    # The event trail must record the creation, not just the final state.
    events = (
        (await db.execute(select(TicketEvent).where(TicketEvent.ticket_id == tickets[0]["id"])))
        .scalars()
        .all()
    )
    assert any(event.action == "created" for event in events)


async def test_execution_happens_exactly_once(db, client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="once2@acme-payments.example", role="admin")
    body = await _start_run(member, "请创建一个工单：结算批次缺失。")
    await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "ok"},
    )
    await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "retry"},
    )

    count = await db.scalar(select(func.count(Ticket.id)).where(Ticket.org_id == member.org_id))
    assert count == 1

    approval = await db.get(ApprovalRequest, body["approval_id"])
    assert approval is not None
    assert approval.executed_at is not None


async def test_votes_are_recorded_for_each_approver(
    db, client, owner: Actor, member: Actor
) -> None:
    approver = await add_member(client, owner, email="votes@acme-payments.example", role="admin")
    body = await _start_run(member, "请创建一个工单：结算批次缺失。")
    await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "looks fine"},
    )
    votes = (
        (
            await db.execute(
                select(ApprovalVote).where(ApprovalVote.approval_id == body["approval_id"])
            )
        )
        .scalars()
        .all()
    )
    assert len(votes) == 1
    assert votes[0].voter_id == approver.id
    assert votes[0].note == "looks fine"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_repeated_run_with_same_idempotency_key_returns_the_same_run(
    db, member: Actor
) -> None:
    first = await _start_run(member, "请创建一个工单：结算批次缺失。", idempotency_key="run-key-1")
    second = await _start_run(member, "请创建一个工单：结算批次缺失。", idempotency_key="run-key-1")

    assert first["id"] == second["id"]
    assert second["replayed"] is True

    count = await db.scalar(
        select(func.count(ApprovalRequest.id)).where(ApprovalRequest.org_id == member.org_id)
    )
    assert count == 1, "a retried request must not queue a second approval"


@pytest.mark.parametrize("prompt", ["退款多久到账？", "请创建一个工单：结算批次缺失。"])
async def test_concurrent_run_with_same_idempotency_key_executes_graph_once(
    db, member: Actor, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    """A retry racing the first request must not duplicate retrieval/model work."""
    from app.services import agent as agent_service

    original_search = agent_service._retrieve_candidates
    search_calls = 0

    async def counted_search(*args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        await asyncio.sleep(0.05)
        return await original_search(*args, **kwargs)

    monkeypatch.setattr(agent_service, "_retrieve_candidates", counted_search)
    payload = {
        "prompt": prompt,
        "idempotency_key": "concurrent-run-key",
    }
    first, second = await asyncio.gather(
        member.client.post("/v1/agent/runs", headers=member.headers, json=payload),
        member.client.post("/v1/agent/runs", headers=member.headers, json=payload),
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]
    assert sorted([first.json()["replayed"], second.json()["replayed"]]) == [False, True]
    assert search_calls == 1
    assert (
        await db.scalar(select(func.count(AgentRun.id)).where(AgentRun.org_id == member.org_id))
        == 1
    )
    assert await db.scalar(
        select(func.count(ApprovalRequest.id)).where(ApprovalRequest.org_id == member.org_id)
    ) == (1 if first.json()["approval_id"] else 0)


async def test_slow_answer_does_not_block_a_different_idempotency_key(
    member: Actor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import agent as agent_service

    answering, release = asyncio.Event(), asyncio.Event()
    original_answer = agent_service._answer

    async def slow_answer(prompt, context):
        if prompt == "退款多久到账？":
            answering.set()
            await release.wait()
        return await original_answer(prompt, context)

    monkeypatch.setattr(agent_service, "_answer", slow_answer)
    slow = asyncio.create_task(_start_run(member, "退款多久到账？", "slow-answer"))
    try:
        await asyncio.wait_for(answering.wait(), timeout=5)
        fast = await asyncio.wait_for(
            _start_run(member, "支付接口返回 401 怎么排查？", "fast-answer"), timeout=2
        )
        assert fast["status"] == "succeeded"
        assert not slow.done(), "the other request must complete before the slow model call"
    finally:
        release.set()
        await slow


async def test_failed_graph_releases_reservation_and_can_be_retried(
    db, member: Actor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import agent as agent_service

    async def fail_search(*_args, **_kwargs):
        raise RuntimeError("injected retrieval failure")

    with monkeypatch.context() as patch:
        patch.setattr(agent_service, "_retrieve_candidates", fail_search)
        with pytest.raises(RuntimeError, match="injected retrieval failure"):
            await _start_run(member, "退款多久到账？", "failed-run-key")
    assert await db.scalar(select(func.count(AgentRun.id))) == 0
    assert await db.scalar(select(func.count(IdempotencyRecord.id))) == 0
    retry = await _start_run(member, "退款多久到账？", "failed-run-key")
    assert retry["status"] == "succeeded"
    assert retry["replayed"] is False


@pytest.mark.parametrize("prompt", ["退款多久到账？", "请创建一个工单：结算批次缺失。"])
async def test_final_persistence_failure_rolls_back_business_rows_and_frees_key(
    db, member: Actor, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    from app.models import UsageEvent
    from sqlalchemy.ext.asyncio import AsyncSession

    original_flush = AsyncSession.flush

    async def fail_final_flush(session, *args, **kwargs):
        if any(isinstance(row, IdempotencyRecord) and row.response_json for row in session.dirty):
            raise RuntimeError("injected final persistence failure")
        return await original_flush(session, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "flush", fail_final_flush)
        with pytest.raises(RuntimeError, match="injected final persistence failure"):
            await _start_run(member, prompt, "failed-persistence-key")
    for model in (AgentRun, ApprovalRequest, UsageEvent, IdempotencyRecord):
        assert await db.scalar(select(func.count(model.id))) == 0
    retry = await _start_run(member, prompt, "failed-persistence-key")
    assert retry["replayed"] is False
    assert await db.scalar(select(func.count(AgentRun.id))) == 1


async def test_pending_key_has_bounded_wait_and_rejects_different_payload(
    db, member: Actor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import idempotency

    db.add(
        IdempotencyRecord(
            org_id=member.org_id,
            scope="agent.run",
            key="pending-run-key",
            request_hash=idempotency.fingerprint(
                {
                    "org": member.org_id,
                    "user": member.id,
                    "prompt": "退款多久到账？",
                    "kb": [],
                }
            ),
            response_json={},
        )
    )
    await db.commit()
    monkeypatch.setattr(idempotency, "_PENDING_WAIT_SECONDS", 0.01)
    response = await member.client.post(
        "/v1/agent/runs",
        headers=member.headers,
        json={"prompt": "退款多久到账？", "idempotency_key": "pending-run-key"},
    )
    assert response.status_code == 409
    assert "in progress" in response.json()["error"]["message"]
    conflict = await member.client.post(
        "/v1/agent/runs",
        headers=member.headers,
        json={"prompt": "完全不同的问题", "idempotency_key": "pending-run-key"},
    )
    assert conflict.status_code == 409
    assert "different payload" in conflict.json()["error"]["message"]
    assert await db.scalar(select(func.count(AgentRun.id))) == 0


async def test_run_events_start_before_retrieval_and_use_the_persisted_trace(member: Actor) -> None:
    from app.services.events import read_events

    body = await _start_run(member, "退款多久到账？", "event-run-key")
    events = [event async for event in read_events(body["id"])]
    assert [event["type"] for event in events] == [
        "run.started",
        "retrieval.completed",
        "run.completed",
    ]
    assert events[0]["trace_id"] == body["trace_id"]
    await _start_run(member, "退款多久到账？", "event-run-key")
    assert [event async for event in read_events(body["id"])] == events


async def test_same_key_with_different_prompt_is_a_conflict(member: Actor) -> None:
    await _start_run(member, "请创建一个工单：结算批次缺失。", idempotency_key="run-key-2")
    response = await member.client.post(
        "/v1/agent/runs",
        headers=member.headers,
        json={"prompt": "请创建一个工单：完全不同的问题。", "idempotency_key": "run-key-2"},
    )
    assert response.status_code == 409


async def test_idempotency_keys_are_scoped_per_tenant(
    member: Actor, second_org_owner: Actor
) -> None:
    """Two tenants may use the same client-generated key without colliding."""
    ours = await _start_run(member, "请创建一个工单：结算批次缺失。", idempotency_key="shared-key")
    theirs = await second_org_owner.client.post(
        "/v1/agent/runs",
        headers=second_org_owner.headers,
        json={"prompt": "请创建一个工单：结算批次缺失。", "idempotency_key": "shared-key"},
    )
    assert theirs.status_code == 200
    assert theirs.json()["id"] != ours["id"]


async def test_approval_list_exposes_progress(client, owner: Actor, member: Actor) -> None:
    approver = await add_member(client, owner, email="progress@acme-payments.example", role="admin")
    body = await _start_run(member, "请创建一个紧急工单：生产环境退款失败。")
    await approver.client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver.headers,
        json={"decision": "approved", "note": "first"},
    )
    approvals = (await member.client.get("/v1/approvals", headers=member.headers)).json()
    row = next(item for item in approvals if item["id"] == body["approval_id"])
    assert row["approvals_received"] == 1
    assert row["required_approvals"] == 2
    assert len(row["votes"]) == 1
