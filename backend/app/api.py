from __future__ import annotations

import asyncio
import base64
import ipaddress
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_principal, get_principal, require_admin, require_write
from app.db import SessionLocal, get_db
from app.models import (
    AgentRun,
    ApprovalRequest,
    ApprovalStatus,
    Document,
    DocumentStatus,
    KnowledgeBase,
    Membership,
    Organization,
    Role,
    User,
)
from app.schemas import (
    AgentRunCreate,
    AgentRunRead,
    ApprovalDecision,
    ApprovalRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    DocumentUrlCreate,
    OrganizationCreate,
    OrganizationRead,
    MemberCreate,
    MemberRead,
    TicketRead,
)
from app.services.agent import decide_approval, start_run
from app.services.demo import seed_demo_knowledge
from app.services.events import read_events
from app.worker.tasks import ingest_document_task

router = APIRouter(prefix="/v1")
Db = Annotated[AsyncSession, Depends(get_db)]


async def _ensure_demo_user(db: AsyncSession, external_id: str, org_id: str, role: str = Role.MEMBER.value) -> User:
    user = (await db.execute(select(User).where(User.external_id == external_id))).scalar_one_or_none()
    if user is None:
        user = User(external_id=external_id, display_name=external_id)
        db.add(user)
        await db.flush()
    membership = (await db.execute(select(Membership).where(Membership.user_id == user.id, Membership.org_id == org_id))).scalar_one_or_none()
    if membership is None:
        db.add(Membership(user_id=user.id, org_id=org_id, role=role))
        await db.flush()
    return user


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "payment-operations-agent"}


@router.post("/organizations", response_model=OrganizationRead)
async def create_organization(
    payload: OrganizationCreate, db: Db, x_user_id: str = Header(default="demo")
) -> OrganizationRead:
    org = Organization(name=payload.name)
    db.add(org)
    await db.flush()
    user = await _ensure_demo_user(db, x_user_id, org.id, Role.OWNER.value)
    await seed_demo_knowledge(db, org.id)
    await db.commit()
    return OrganizationRead(id=org.id, name=org.name, role=Role.OWNER.value)


@router.post("/organizations/{org_id}/members", response_model=MemberRead)
async def add_member(
    org_id: str,
    payload: MemberCreate,
    db: Db,
    principal=Depends(current_principal),
) -> MemberRead:
    if principal.org_id != org_id:
        raise HTTPException(status_code=403, detail="organisation mismatch")
    require_admin(principal)
    org = await db.get(Organization, org_id)
    if org is None:
        raise HTTPException(status_code=404, detail="organisation not found")
    user = await _ensure_demo_user(db, payload.external_id, org_id, payload.role)
    user.display_name = payload.display_name
    membership = (
        await db.execute(
            select(Membership).where(Membership.user_id == user.id, Membership.org_id == org_id)
        )
    ).scalar_one()
    membership.role = payload.role
    await db.commit()
    return MemberRead(
        user_id=user.id,
        external_id=user.external_id,
        display_name=user.display_name,
        role=membership.role,
    )


@router.get("/organizations/{org_id}/members", response_model=list[MemberRead])
async def list_members(
    org_id: str,
    db: Db,
    principal=Depends(current_principal),
) -> list[MemberRead]:
    if principal.org_id != org_id:
        raise HTTPException(status_code=403, detail="organisation mismatch")
    rows = (
        await db.execute(
            select(User, Membership.role)
            .join(Membership, Membership.user_id == User.id)
            .where(Membership.org_id == org_id)
            .order_by(User.display_name)
        )
    ).all()
    return [
        MemberRead(user_id=user.id, external_id=user.external_id, display_name=user.display_name, role=role)
        for user, role in rows
    ]


@router.post("/knowledge-bases", response_model=KnowledgeBaseRead)
async def create_knowledge_base(payload: KnowledgeBaseCreate, db: Db, principal=Depends(current_principal)) -> KnowledgeBaseRead:
    require_write(principal)
    kb = KnowledgeBase(org_id=principal.org_id, name=payload.name, description=payload.description)
    db.add(kb)
    await db.commit()
    return KnowledgeBaseRead(id=kb.id, name=kb.name, description=kb.description)


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseRead])
async def list_knowledge_bases(db: Db, principal=Depends(current_principal)) -> list[KnowledgeBaseRead]:
    rows = (await db.execute(select(KnowledgeBase).where(KnowledgeBase.org_id == principal.org_id).order_by(KnowledgeBase.name))).scalars()
    return [KnowledgeBaseRead(id=row.id, name=row.name, description=row.description) for row in rows]


@router.post("/demo/seed", response_model=KnowledgeBaseRead)
async def seed_demo(db: Db, principal=Depends(current_principal)) -> KnowledgeBaseRead:
    """Provision the tenant-scoped corpus used by the guided product demo."""
    require_write(principal)
    knowledge_base, _ = await seed_demo_knowledge(db, principal.org_id)
    return KnowledgeBaseRead(
        id=knowledge_base.id,
        name=knowledge_base.name,
        description=knowledge_base.description,
    )


@router.post("/documents")
async def upload_document(
    kb_id: str,
    db: Db,
    principal=Depends(current_principal),
    file: UploadFile = File(...),
) -> dict[str, object]:
    require_write(principal)
    kb = (await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id, KnowledgeBase.org_id == principal.org_id))).scalar_one_or_none()
    if kb is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file exceeds the 10 MiB upload limit")
    document = Document(kb_id=kb_id, filename=file.filename or "upload.txt", source_url="", content_hash="pending", status=DocumentStatus.PENDING.value)
    db.add(document)
    await db.commit()
    try:
        ingest_document_task.delay(document.id, document.filename, base64.b64encode(raw).decode("ascii"))
        queued = True
    except Exception:
        # Useful for local development without a running broker.
        await _ingest_locally(document.id, document.filename, raw)
        queued = False
    return {"document_id": document.id, "status": "pending" if queued else "indexed", "queued": queued}


@router.post("/documents/from-url")
async def ingest_url(
    payload: DocumentUrlCreate,
    db: Db,
    principal=Depends(current_principal),
) -> dict[str, object]:
    """Fetch and asynchronously index a public web page for a tenant KB."""
    require_write(principal)
    from urllib.parse import urlparse

    parsed = urlparse(payload.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="url must be an absolute http(s) URL")
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "metadata.google.internal"}:
        raise HTTPException(status_code=422, detail="private and metadata hosts are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved):
        raise HTTPException(status_code=422, detail="private and reserved hosts are not allowed")
    kb = (
        await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == payload.kb_id, KnowledgeBase.org_id == principal.org_id
            )
        )
    ).scalar_one_or_none()
    if kb is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")
    try:
        import httpx

        async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
            response = await client.get(payload.url, headers={"User-Agent": "payment-ops-agent/0.1"})
            response.raise_for_status()
            raw = response.content
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to fetch URL: {exc}") from exc
    existing = (
        await db.execute(
            select(Document).where(Document.kb_id == payload.kb_id, Document.source_url == payload.url)
        )
    ).scalar_one_or_none()
    document = existing or Document(
        kb_id=payload.kb_id,
        filename=parsed.hostname or "web-page.html",
        source_url=payload.url,
        content_hash="pending",
        status=DocumentStatus.PENDING.value,
    )
    document.filename = parsed.hostname or document.filename
    if existing is None:
        document.content_hash = "pending"
        document.status = DocumentStatus.PENDING.value
    db.add(document)
    await db.commit()
    try:
        ingest_document_task.delay(document.id, document.filename + ".html", base64.b64encode(raw).decode("ascii"))
        queued = True
    except Exception:
        await _ingest_locally(document.id, document.filename + ".html", raw)
        queued = False
    return {"document_id": document.id, "status": "pending" if queued else "indexed", "queued": queued, "source_url": payload.url}


async def _ingest_locally(document_id: str, filename: str, raw: bytes) -> None:
    from app.worker.tasks import _ingest

    await _ingest(document_id, filename, raw)


@router.post("/agent/runs", response_model=AgentRunRead)
async def run_agent(payload: AgentRunCreate, db: Db, principal=Depends(current_principal)) -> AgentRunRead:
    result = await start_run(db, principal.org_id, principal.user_id, payload.prompt, payload.knowledge_base_ids)
    return AgentRunRead(id=result.run.id, trace_id=result.run.trace_id, status=result.run.status, answer=result.run.answer, input_tokens=result.run.input_tokens, output_tokens=result.run.output_tokens, estimated_cost_usd=result.run.estimated_cost_usd, approval_id=result.approval.id if result.approval else None)


@router.get("/approvals", response_model=list[ApprovalRead])
async def list_approvals(db: Db, principal=Depends(current_principal)) -> list[ApprovalRead]:
    rows = (await db.execute(select(ApprovalRequest).where(ApprovalRequest.org_id == principal.org_id).order_by(ApprovalRequest.decided_at))).scalars()
    return [ApprovalRead(id=row.id, run_id=row.run_id, tool_name=row.tool_name, arguments=row.arguments, status=row.status, decided_by=row.decided_by, decided_at=row.decided_at) for row in rows]


@router.post("/approvals/{approval_id}/decision", response_model=AgentRunRead)
async def decide(approval_id: str, payload: ApprovalDecision, db: Db, principal=Depends(current_principal)) -> AgentRunRead:
    require_admin(principal)
    approval = (await db.execute(select(ApprovalRequest).where(ApprovalRequest.id == approval_id, ApprovalRequest.org_id == principal.org_id))).scalar_one_or_none()
    if approval is None:
        raise HTTPException(status_code=404, detail="approval not found")
    try:
        result = await decide_approval(db, approval, ApprovalStatus(payload.decision), principal.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentRunRead(id=result.run.id, trace_id=result.run.trace_id, status=result.run.status, answer=result.run.answer, input_tokens=result.run.input_tokens, output_tokens=result.run.output_tokens, estimated_cost_usd=result.run.estimated_cost_usd)


@router.get("/tickets", response_model=list[TicketRead])
async def list_tickets(db: Db, principal=Depends(current_principal)) -> list[TicketRead]:
    from app.models import Ticket

    rows = (
        await db.execute(select(Ticket).where(Ticket.org_id == principal.org_id).order_by(Ticket.created_at.desc()))
    ).scalars()
    return [
        TicketRead(id=row.id, title=row.title, description=row.description, priority=row.priority, status=row.status)
        for row in rows
    ]


@router.websocket("/ws/agent")
async def agent_stream(websocket: WebSocket, run_id: str, org_id: str, user_id: str) -> None:
    try:
        async with SessionLocal() as db:
            await get_principal(db, user_id, org_id)
            run = (
                await db.execute(
                    select(AgentRun).where(AgentRun.id == run_id, AgentRun.org_id == org_id)
                )
            ).scalar_one_or_none()
            if run is None:
                await websocket.close(code=1008, reason="run is not visible to this organisation")
                return
        await websocket.accept()
        cursor = 0
        while True:
            emitted = False
            async for event in read_events(run_id, cursor):
                await websocket.send_json(event)
                cursor += 1
                emitted = True
            if not emitted:
                await asyncio.sleep(0.15)
            if emitted and any(event.get("type") == "run.completed" for event in await _events_snapshot(run_id)):
                break
    except WebSocketDisconnect:
        return
    except HTTPException:
        await websocket.close(code=1008, reason="authentication failed")
    except Exception:
        await websocket.close(code=1011, reason="stream failed")


async def _events_snapshot(run_id: str) -> list[dict[str, object]]:
    return [event async for event in read_events(run_id)]
