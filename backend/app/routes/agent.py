"""Agent run endpoints and the run-event WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any, cast

from anyio import CancelScope
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.auth import (
    Db,
    authenticate_websocket,
    require,
    websocket_token,
)
from app.core_exceptions import AppError, AuthenticationError, NotFoundError
from app.db import SessionLocal
from app.models import AgentRun, ApprovalRequest
from app.permissions import Action, Principal, Resource
from app.schemas import AgentRunCreate, AgentRunRead, Citation
from app.services.agent import start_run
from app.services.events import event_sink, read_events

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)

#: How long the event stream may run before it gives up on the run finishing.
_WS_MAX_SECONDS = 600
_WS_POLL_SECONDS = 0.15


class _RunStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A failed ASGI send does not automatically close the suspended generator.
            with CancelScope(shield=True):
                await cast(AsyncGenerator[str, None], self.body_iterator).aclose()


def _run_read(
    run: AgentRun, approval: ApprovalRequest | None = None, replayed: bool = False
) -> AgentRunRead:
    return AgentRunRead(
        id=run.id,
        trace_id=run.trace_id,
        status=run.status,
        answer=run.answer,
        intent=run.intent,
        risk_tier=run.risk_tier,
        citations=[],
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        estimated_cost_usd=float(run.estimated_cost_usd or 0.0),
        approval_id=approval.id if approval else None,
        replayed=replayed,
    )


@router.post("/agent/runs", response_model=AgentRunRead)
async def create_run(
    payload: AgentRunCreate,
    db: Db,
    principal: Principal = Depends(require(Action.CREATE, Resource.AGENT_RUN)),
) -> Any:
    """Answer a question, or pause for approval when a gated tool is needed.

    Passing the same ``idempotency_key`` twice returns the first result instead
    of starting a second run, which is what stops a retrying UI from queueing
    duplicate refund tickets.
    """
    result = await start_run(
        db,
        principal=principal,
        prompt=payload.prompt,
        kb_ids=payload.knowledge_base_ids,
        idempotency_key=payload.idempotency_key,
    )

    # Attach the retrieved chunks that produced the answer so the UI can show
    # citations without a second round trip.
    citations = [
        Citation(
            chunk_id=str(item["chunk_id"]),
            filename=str(item["filename"]),
            source_url=str(item.get("source_url") or ""),
            score=round(float(item["score"]), 4),
        )
        for item in (await _context_for(db, result.run))
    ]
    read = _run_read(result.run, result.approval, result.replayed)
    return read.model_copy(update={"citations": citations})


@router.post("/agent/runs/stream")
async def create_run_stream(
    payload: AgentRunCreate,
    db: Db,
    principal: Principal = Depends(require(Action.CREATE, Resource.AGENT_RUN)),
) -> StreamingResponse:
    # The streaming producer owns a fresh session after authorization finishes.
    await db.commit()

    async def stream():
        queue: asyncio.Queue = asyncio.Queue(maxsize=32)

        async def execute():
            token = event_sink.set(queue.put)
            try:
                async with SessionLocal() as session:
                    result = await create_run(payload, session, principal)
                await queue.put({"type": "run.result", "run": result.model_dump(mode="json")})
            except AppError as exc:
                await queue.put(
                    {
                        "type": "run.error",
                        "status": exc.status_code,
                        "message": exc.message,
                        "details": exc.details,
                    }
                )
            except Exception:
                logger.exception("Streaming agent run failed")
                await queue.put({"type": "run.error", "status": 500, "message": "Agent run failed"})
            finally:
                event_sink.reset(token)
            await queue.put(None)

        task = asyncio.create_task(execute())
        try:
            while (event := await queue.get()) is not None:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            task.cancel()
            with CancelScope(shield=True), suppress(asyncio.CancelledError):
                await task

    return _RunStreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _context_for(db: Any, run: AgentRun) -> list[dict[str, Any]]:
    """Re-derive citations for a stored run from its recorded chunk IDs."""
    if not run.context_chunk_ids:
        return []
    from app.models import Document, DocumentChunk

    rows = (
        await db.execute(
            select(DocumentChunk.id, Document.filename, Document.source_url)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.id.in_(run.context_chunk_ids),
                Document.org_id == run.org_id,
            )
        )
    ).all()
    # An IN query does not preserve the retrieval ranking stored on the run.
    citations_by_id = {
        chunk_id: {
            "chunk_id": chunk_id,
            "filename": filename,
            "source_url": source_url,
            "score": 0.0,
        }
        for chunk_id, filename, source_url in rows
    }
    return [
        citations_by_id[chunk_id]
        for chunk_id in run.context_chunk_ids
        if chunk_id in citations_by_id
    ]


@router.get("/agent/runs", response_model=list[AgentRunRead])
async def list_runs(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.AGENT_RUN)),
    limit: int = Query(50, ge=1, le=200),
) -> Any:
    rows = (
        (
            await db.execute(
                select(AgentRun)
                .where(AgentRun.org_id == principal.org_id)
                .order_by(AgentRun.created_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [_run_read(row) for row in rows]


@router.get("/agent/runs/{run_id}", response_model=AgentRunRead)
async def get_run(
    run_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.AGENT_RUN)),
) -> Any:
    run = (
        await db.execute(
            select(AgentRun).where(AgentRun.id == run_id, AgentRun.org_id == principal.org_id)
        )
    ).scalar_one_or_none()
    if run is None:
        raise NotFoundError(message="run not found")

    approval = (
        await db.execute(select(ApprovalRequest).where(ApprovalRequest.run_id == run.id))
    ).scalar_one_or_none()
    read = _run_read(run, approval)
    citations = [
        Citation(
            chunk_id=str(item["chunk_id"]),
            filename=str(item["filename"]),
            source_url=str(item.get("source_url") or ""),
            score=float(item["score"]),
        )
        for item in (await _context_for(db, run))
    ]
    return read.model_copy(update={"citations": citations})


@router.websocket("/ws/agent")
async def agent_stream(
    websocket: WebSocket,
    run_id: str,
    org_id: str | None = None,
    token: str | None = Query(default=None, description="Access token for non-browser clients"),
) -> None:
    """Stream run events.

    Authentication happens **before** ``accept()`` so an unauthenticated client
    never reaches an open socket. The token is read from the
    ``Sec-WebSocket-Protocol`` header when present, which keeps it out of URLs
    and proxy logs; the query parameter is a fallback for non-browser clients.
    """
    resolved_token = websocket_token(websocket, token)
    try:
        async with SessionLocal() as db:
            principal = await authenticate_websocket(
                websocket, db, token=resolved_token, org_id=org_id
            )
            run = (
                await db.execute(
                    select(AgentRun).where(
                        AgentRun.id == run_id, AgentRun.org_id == principal.org_id
                    )
                )
            ).scalar_one_or_none()
            if run is None:
                await websocket.close(code=1008, reason="run is not visible to this organisation")
                return
    except AuthenticationError:
        await websocket.close(code=1008, reason="authentication failed")
        return
    except AppError:
        await websocket.close(code=1008, reason="not authorised for this run")
        return

    await websocket.accept(subprotocol=resolved_token if resolved_token else None)
    cursor = 0
    elapsed = 0.0
    try:
        while elapsed < _WS_MAX_SECONDS:
            emitted = False
            async for event in read_events(run_id, cursor):
                cursor = event.pop("_cursor", cursor + 1 if isinstance(cursor, int) else cursor)
                await websocket.send_json(event)
                emitted = True
                if event.get("type") == "run.completed":
                    await websocket.close(code=1000)
                    return
            if not emitted:
                await asyncio.sleep(_WS_POLL_SECONDS)
                elapsed += _WS_POLL_SECONDS
            else:
                elapsed += 0.01
        await websocket.close(code=1000, reason="stream timeout")
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011, reason="stream failed")
