"""Approval inspection and voting.

The endpoint that decides an approval is the highest-risk write in the product,
so the validation lives in ``services.agent.record_vote`` rather than here: the
route only resolves identity and delegates.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select

from app.auth import Db, require
from app.core_exceptions import NotFoundError
from app.models import (
    AgentRun,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalVote,
)
from app.permissions import Action, Principal, Resource
from app.schemas import (
    AgentRunRead,
    ApprovalDecision,
    ApprovalRead,
    ApprovalVoteRead,
)
from app.services.agent import record_vote

router = APIRouter(tags=["approvals"])


def _approval_read(
    approval: ApprovalRequest, votes: list[ApprovalVote], required: int
) -> ApprovalRead:
    granted = {v.voter_id for v in votes if v.decision == ApprovalStatus.APPROVED.value}
    return ApprovalRead(
        id=approval.id,
        run_id=approval.run_id,
        tool_name=approval.tool_name,
        arguments=dict(approval.arguments or {}),
        status=approval.status,
        risk_tier=approval.risk_tier,
        required_approvals=required,
        approvals_received=len(granted),
        requested_by=approval.requested_by,
        votes=[
            ApprovalVoteRead(
                voter_id=vote.voter_id,
                decision=vote.decision,
                note=vote.note,
                created_at=vote.created_at,
            )
            for vote in votes
        ],
        created_at=approval.created_at,
        decided_at=approval.decided_at,
    )


@router.get("/approvals", response_model=list[ApprovalRead])
async def list_approvals(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.APPROVAL)),
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
) -> Any:
    stmt = select(ApprovalRequest).where(ApprovalRequest.org_id == principal.org_id)
    if status_filter:
        stmt = stmt.where(ApprovalRequest.status == status_filter)
    approvals = (
        await db.execute(stmt.order_by(ApprovalRequest.created_at.desc()).limit(limit))
    ).scalars().all()
    if not approvals:
        return []

    approval_ids = [a.id for a in approvals]
    vote_rows = (
        await db.execute(
            select(ApprovalVote).where(ApprovalVote.approval_id.in_(approval_ids))
        )
    ).scalars().all()
    votes_by_approval: dict[str, list[ApprovalVote]] = {}
    for vote in vote_rows:
        votes_by_approval.setdefault(vote.approval_id, []).append(vote)

    return [
        _approval_read(
            approval,
            votes_by_approval.get(approval.id, []),
            approval.required_approvals,
        )
        for approval in approvals
    ]


@router.get("/approvals/{approval_id}", response_model=ApprovalRead)
async def get_approval(
    approval_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.APPROVAL)),
) -> Any:
    approval = (
        await db.execute(
            select(ApprovalRequest).where(
                ApprovalRequest.id == approval_id,
                ApprovalRequest.org_id == principal.org_id,
            )
        )
    ).scalar_one_or_none()
    if approval is None:
        raise NotFoundError(message="approval not found")
    votes = (
        await db.execute(
            select(ApprovalVote).where(ApprovalVote.approval_id == approval.id)
        )
    ).scalars().all()
    return _approval_read(approval, list(votes), approval.required_approvals)


@router.post("/approvals/{approval_id}/decision", response_model=AgentRunRead)
async def decide(
    approval_id: str,
    payload: ApprovalDecision,
    db: Db,
    principal: Principal = Depends(require(Action.APPROVE, Resource.APPROVAL)),
) -> Any:
    """Cast one vote.

    Returns the current run state. When quorum is reached the tool executes
    inside the same transaction, so a 200 here means the ticket exists.
    """
    approval, _executed = await record_vote(
        db,
        principal=principal,
        approval_id=approval_id,
        decision=payload.decision,
        note=payload.note,
    )
    run = await db.get(AgentRun, approval.run_id)
    if run is None:  # pragma: no cover
        raise NotFoundError(message="run not found")
    return AgentRunRead(
        id=run.id,
        trace_id=run.trace_id,
        status=run.status,
        answer=run.answer,
        intent=run.intent,
        risk_tier=run.risk_tier,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        estimated_cost_usd=float(run.estimated_cost_usd or 0.0),
        approval_id=approval.id,
    )


@router.get("/approvals/{approval_id}/votes", response_model=list[ApprovalVoteRead])
async def list_votes(
    approval_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.APPROVAL)),
) -> Any:
    approval = (
        await db.execute(
            select(ApprovalRequest).where(
                ApprovalRequest.id == approval_id,
                ApprovalRequest.org_id == principal.org_id,
            )
        )
    ).scalar_one_or_none()
    if approval is None:
        raise NotFoundError(message="approval not found")
    votes = (
        await db.execute(
            select(ApprovalVote)
            .where(ApprovalVote.approval_id == approval_id)
            .order_by(ApprovalVote.created_at)
        )
    ).scalars().all()
    return [
        ApprovalVoteRead(
            voter_id=vote.voter_id, decision=vote.decision, note=vote.note,
            created_at=vote.created_at,
        )
        for vote in votes
    ]


@router.get("/approvals/pending/count")
async def pending_count(
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.APPROVAL)),
) -> dict[str, int]:
    count = await db.scalar(
        select(func.count(ApprovalRequest.id)).where(
            ApprovalRequest.org_id == principal.org_id,
            ApprovalRequest.status == ApprovalStatus.PENDING.value,
        )
    )
    return {"pending": int(count or 0)}
