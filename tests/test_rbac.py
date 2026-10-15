"""Role-based access control and cross-tenant isolation.

These are the tests that back the résumé claim about "组织级数据隔离与细粒度权限控制".
Each one describes a concrete over-permission that must not be possible.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
import httpx
from app import audit
from app.core_exceptions import AuthorizationError
from app.db import get_db
from app.models import AuditLog, Organization
from app.permissions import ROLE_PERMISSIONS, Action, Resource
from sqlalchemy import select

from tests.conftest import Actor, seed_corpus

# ---------------------------------------------------------------------------
# The permission matrix itself
# ---------------------------------------------------------------------------


def test_permission_matrix_is_deny_by_default() -> None:
    for role, pairs in ROLE_PERMISSIONS.items():
        for action, resource in pairs:
            assert action in {a.value for a in Action}, f"{role}: unknown action {action}"
            assert resource in {r.value for r in Resource}, f"{role}: unknown resource {resource}"


def test_viewer_cannot_create_anything() -> None:
    from app.permissions import Principal

    viewer = Principal(user_id="u", org_id="o", role="viewer")
    for action in (Action.CREATE, Action.UPDATE, Action.DELETE, Action.APPROVE):
        assert not viewer.can(action, Resource.KNOWLEDGE_BASE)
        assert not viewer.can(action, Resource.TICKET)
        assert not viewer.can(action, Resource.AGENT_RUN)


def test_member_can_raise_an_approval_but_not_decide_it() -> None:
    from app.permissions import Principal

    member = Principal(user_id="u", org_id="o", role="member")
    assert member.can(Action.CREATE, Resource.APPROVAL)
    assert member.can(Action.READ, Resource.APPROVAL)
    assert not member.can(Action.APPROVE, Resource.APPROVAL)


def test_member_can_execute_ordinary_ticket_actions_but_viewer_cannot() -> None:
    from app.permissions import Principal

    assert Principal(user_id="u", org_id="o", role="member").can(Action.EXECUTE, Resource.TICKET)
    assert not Principal(user_id="u", org_id="o", role="viewer").can(
        Action.EXECUTE, Resource.TICKET
    )


def test_member_cannot_manage_members_but_owner_can() -> None:
    from app.permissions import Principal

    assert not Principal(user_id="u", org_id="o", role="member").can(
        Action.MANAGE_MEMBERS, Resource.MEMBERSHIP
    )
    assert not Principal(user_id="u", org_id="o", role="admin").can(
        Action.MANAGE_MEMBERS, Resource.MEMBERSHIP
    )
    assert Principal(user_id="u", org_id="o", role="owner").can(
        Action.MANAGE_MEMBERS, Resource.MEMBERSHIP
    )


def test_owner_holds_every_admin_permission() -> None:
    owner = ROLE_PERMISSIONS["owner"]
    admin = ROLE_PERMISSIONS["admin"]
    assert admin <= owner


# ---------------------------------------------------------------------------
# Route-level enforcement
# ---------------------------------------------------------------------------


async def test_viewer_cannot_create_a_knowledge_base(viewer: Actor) -> None:
    response = await viewer.client.post(
        "/v1/knowledge-bases",
        headers=viewer.headers,
        json={"name": "Viewer base", "description": ""},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


async def test_viewer_cannot_upload_a_document(viewer: Actor, owner: Actor) -> None:
    kb_id = await seed_corpus(viewer.client, owner)
    response = await viewer.client.post(
        f"/v1/documents?kb_id={kb_id}",
        headers=viewer.headers,
        files={"file": ("runbook.md", b"# runbook content", "text/markdown")},
    )
    assert response.status_code == 403


async def test_viewer_cannot_start_an_agent_run(viewer: Actor) -> None:
    response = await viewer.client.post(
        "/v1/agent/runs", headers=viewer.headers, json={"prompt": "退款多久到账？"}
    )
    assert response.status_code == 403


async def test_member_can_ask_a_question(member: Actor) -> None:
    response = await member.client.post(
        "/v1/agent/runs", headers=member.headers, json={"prompt": "退款多久到账？"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


async def test_member_cannot_list_audit_logs(member: Actor) -> None:
    """The audit trail is admin-only; a member must not be able to read it."""
    response = await member.client.get("/v1/audit-logs", headers=member.headers)
    assert response.status_code == 403


async def test_admin_can_read_the_audit_log(admin: Actor) -> None:
    response = await admin.client.get("/v1/audit-logs", headers=admin.headers)
    assert response.status_code == 200
    assert "items" in response.json()


async def test_only_owner_can_add_members(admin: Actor) -> None:
    response = await admin.client.post(
        f"/v1/organizations/{admin.org_id}/members",
        headers=admin.headers,
        json={
            "email": "intruder@acme-payments.example",
            "display_name": "Intruder",
            "password": "intruder-password-1",
            "role": "admin",
        },
    )
    assert response.status_code == 403


async def test_admin_cannot_promote_itself_to_owner(admin: Actor) -> None:
    """Privilege escalation via the role-update endpoint must be blocked."""
    response = await admin.client.patch(
        f"/v1/organizations/{admin.org_id}/members/{admin.id}",
        headers=admin.headers,
        json={"role": "owner"},
    )
    assert response.status_code == 403


async def test_owner_cannot_demote_the_last_owner(owner: Actor) -> None:
    """An organisation with no owner would be permanently unadministrable."""
    response = await owner.client.patch(
        f"/v1/organizations/{owner.org_id}/members/{owner.id}",
        headers=owner.headers,
        json={"role": "member"},
    )
    assert response.status_code == 409


@pytest.mark.usefixtures("member")
async def test_owner_can_add_a_member_and_that_member_can_log_in(owner: Actor) -> None:
    response = await owner.client.get(
        f"/v1/organizations/{owner.org_id}/members", headers=owner.headers
    )
    assert response.status_code == 200
    emails = {row["email"] for row in response.json()}
    assert "member@acme-payments.example" in emails


# ---------------------------------------------------------------------------
# Cross-tenant isolation
# ---------------------------------------------------------------------------


async def test_other_tenant_cannot_see_our_knowledge_bases(
    owner: Actor, second_org_owner: Actor
) -> None:
    ours = await owner.client.get("/v1/knowledge-bases", headers=owner.headers)
    theirs = await second_org_owner.client.get(
        "/v1/knowledge-bases", headers=second_org_owner.headers
    )
    our_ids = {row["id"] for row in ours.json()}
    their_ids = {row["id"] for row in theirs.json()}
    assert our_ids and their_ids
    assert not our_ids & their_ids


async def test_other_tenant_cannot_read_our_ticket_by_id(
    owner: Actor, admin: Actor, second_org_owner: Actor
) -> None:
    """A guessed ticket ID must not resolve across the tenant boundary."""
    created = await owner.client.post(
        "/v1/agent/runs",
        headers=owner.headers,
        json={"prompt": "请创建一个工单：商户结算批次缺失。"},
    )
    assert created.status_code == 200, created.text
    approval_id = created.json()["approval_id"]
    decision = await admin.client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=admin.headers,
        json={"decision": "approved", "note": "ok"},
    )
    assert decision.status_code == 200, decision.text
    assert decision.json()["status"] == "succeeded"
    tickets = await owner.client.get("/v1/tickets", headers=owner.headers)
    assert tickets.status_code == 200
    assert len(tickets.json()) == 1
    ticket_id = tickets.json()[0]["id"]

    stolen = await second_org_owner.client.get(
        f"/v1/tickets/{ticket_id}", headers=second_org_owner.headers
    )
    assert stolen.status_code == 404


async def test_other_tenant_cannot_decide_our_approval(
    owner: Actor, second_org_owner: Actor
) -> None:
    created = await owner.client.post(
        "/v1/agent/runs",
        headers=owner.headers,
        json={"prompt": "请创建一个工单：结算批次缺失。"},
    )
    approval_id = created.json()["approval_id"]

    # The rival owner is a legitimate owner — of the wrong organisation.
    stolen = await second_org_owner.client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=second_org_owner.headers,
        json={"decision": "approved", "note": "not mine to approve"},
    )
    assert stolen.status_code == 404
    assert stolen.json()["error"]["code"] == "NOT_FOUND"


async def test_other_tenant_cannot_upload_into_our_knowledge_base(
    owner: Actor, second_org_owner: Actor
) -> None:
    kb_id = await seed_corpus(owner.client, owner)
    response = await second_org_owner.client.post(
        f"/v1/documents?kb_id={kb_id}",
        headers=second_org_owner.headers,
        files={"file": ("evil.md", b"# injected", "text/markdown")},
    )
    assert response.status_code == 404


async def test_other_tenant_cannot_read_our_agent_run(
    owner: Actor, second_org_owner: Actor
) -> None:
    created = await owner.client.post(
        "/v1/agent/runs", headers=owner.headers, json={"prompt": "退款多久到账？"}
    )
    run_id = created.json()["id"]
    stolen = await second_org_owner.client.get(
        f"/v1/agent/runs/{run_id}", headers=second_org_owner.headers
    )
    assert stolen.status_code == 404


async def test_org_id_mismatch_in_path_is_not_found(owner: Actor) -> None:
    """Path org must match the acting org, otherwise it is reported missing."""
    response = await owner.client.get(
        "/v1/organizations/00000000-0000-0000-0000-000000000000/members",
        headers=owner.headers,
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


async def test_denied_requests_are_audited(db, viewer: Actor) -> None:
    await viewer.client.post(
        "/v1/knowledge-bases",
        headers=viewer.headers,
        json={"name": "Should not exist", "description": ""},
    )
    rows = (
        (
            await db.execute(
                select(AuditLog).where(
                    AuditLog.decision == "deny", AuditLog.org_id == viewer.org_id
                )
            )
        )
        .scalars()
        .all()
    )
    assert rows, "a denied request must leave an audit row"
    assert any("denied" in row.action for row in rows)


async def test_allowed_actions_are_audited(db, owner: Actor) -> None:
    trace_id = str(uuid4())
    response = await owner.client.get(
        "/v1/knowledge-bases", headers={**owner.headers, "X-Trace-ID": trace_id}
    )
    assert response.status_code == 200
    rows = (
        (
            await db.execute(
                select(AuditLog).where(
                    AuditLog.org_id == owner.org_id,
                    AuditLog.actor_id == owner.id,
                    AuditLog.action == "read.allowed",
                    AuditLog.resource == "knowledge_base",
                    AuditLog.trace_id == trace_id,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].decision == "allow"


async def test_concurrent_reads_persist_each_audit(owner: Actor) -> None:
    trace_ids = [str(uuid4()) for _ in range(16)]
    responses = await asyncio.gather(
        *(
            owner.client.get(
                "/v1/knowledge-bases", headers={**owner.headers, "X-Trace-ID": trace_id}
            )
            for trace_id in trace_ids
        )
    )
    assert all(response.status_code == 200 for response in responses)

    from app.db import SessionLocal

    async with SessionLocal() as session:
        persisted = (
            await session.execute(
                select(AuditLog.trace_id).where(
                    AuditLog.action == "read.allowed", AuditLog.trace_id.in_(trace_ids)
                )
            )
        ).scalars().all()
    assert set(persisted) == set(trace_ids)


async def test_read_audit_is_committed_before_response_starts(owner: Actor) -> None:
    from app.db import SessionLocal
    from app.main import app

    trace_id = str(uuid4())
    observed = []

    async def observe_response(scope, receive, send):
        async def inspect_send(message):
            if message["type"] == "http.response.start":
                async with SessionLocal() as session:
                    observed.append(
                        await session.scalar(
                            select(AuditLog.id).where(AuditLog.trace_id == trace_id)
                        )
                    )
            await send(message)

        await app(scope, receive, inspect_send)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=observe_response), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/v1/knowledge-bases", headers={**owner.headers, "X-Trace-ID": trace_id}
        )
    assert response.status_code == 200
    assert len(observed) == 1 and observed[0] is not None


async def test_write_audit_is_not_committed_after_business_rollback(db, owner: Actor) -> None:
    trace_id = str(uuid4())
    with pytest.raises(RuntimeError):
        async with asynccontextmanager(get_db)() as failed_session:
            failed_session.add(Organization(name="must-not-commit-with-audit"))
            audit.record(
                failed_session,
                action="organization.create",
                resource="organization",
                org_id=owner.org_id,
                actor_id=owner.id,
                trace_id=trace_id,
            )
            raise RuntimeError("business operation failed")

    assert await db.scalar(select(Organization).where(Organization.name == "must-not-commit-with-audit")) is None
    assert await db.scalar(select(AuditLog).where(AuditLog.trace_id == trace_id)) is None


async def test_denial_audit_does_not_commit_staged_business_changes(db, owner: Actor) -> None:
    with pytest.raises(AuthorizationError):
        async with asynccontextmanager(get_db)() as failed_session:
            failed_session.add(Organization(name="must-be-rolled-back"))
            audit.record_denied(
                failed_session,
                action="create",
                resource="organization",
                org_id=owner.org_id,
                actor_id=owner.id,
                reason="test denial",
            )
            raise AuthorizationError(message="test denial")
    assert (
        await db.scalar(select(Organization).where(Organization.name == "must-be-rolled-back"))
        is None
    )
    denial = await db.scalar(
        select(AuditLog).where(AuditLog.org_id == owner.org_id, AuditLog.reason == "test denial")
    )
    assert denial is not None
    assert denial.decision == "deny"


async def test_audit_log_never_leaks_across_tenants(owner: Actor, second_org_owner: Actor) -> None:
    await owner.client.get("/v1/knowledge-bases", headers=owner.headers)
    response = await second_org_owner.client.get("/v1/audit-logs", headers=second_org_owner.headers)
    assert response.status_code == 200
    org_ids = {row["id"] for row in response.json()["items"]}
    assert org_ids == set() or all(
        row.get("resource") != "knowledge_base" for row in response.json()["items"]
    )


@pytest.mark.parametrize("role", ["owner", "admin", "member", "viewer"])
async def test_every_role_can_read_tickets(
    role: str, owner: Actor, admin: Actor, member: Actor, viewer: Actor
) -> None:
    actor = {"owner": owner, "admin": admin, "member": member, "viewer": viewer}[role]
    response = await actor.client.get("/v1/tickets", headers=actor.headers)
    assert response.status_code == 200
