from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


class BaseSchema(BaseModel):
    model_config = {"from_attributes": True, "str_strip_whitespace": True}


# --- Auth ------------------------------------------------------------------


class RegisterRequest(BaseSchema):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="User", min_length=1, max_length=120)
    organization_name: str = Field(min_length=2, max_length=160)


class LoginRequest(BaseSchema):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseSchema):
    refresh_token: str = Field(min_length=8)


class TokenPair(BaseSchema):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class LogoutRequest(BaseSchema):
    refresh_token: str = Field(min_length=8)


class UserRead(BaseSchema):
    id: str
    email: str
    display_name: str
    is_app_admin: bool = False


# --- Organisations ---------------------------------------------------------


class OrganizationCreate(BaseSchema):
    name: str = Field(min_length=2, max_length=160)


class OrganizationRead(BaseSchema):
    id: str
    name: str
    role: str


class MemberCreate(BaseSchema):
    email: EmailStr
    display_name: str = Field(default="User", min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)
    role: Literal["admin", "member", "viewer"] = "member"


class MemberRoleUpdate(BaseSchema):
    role: Literal["owner", "admin", "member", "viewer"]


class MemberRead(BaseSchema):
    user_id: str
    email: str
    display_name: str
    role: str
    created_at: datetime | None = None


# --- Knowledge bases -------------------------------------------------------


class KnowledgeBaseCreate(BaseSchema):
    name: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=2000)


class KnowledgeBaseRead(BaseSchema):
    id: str
    name: str
    description: str
    document_count: int = 0


class DocumentRead(BaseSchema):
    id: str
    kb_id: str
    filename: str
    source_url: str
    status: str
    chunk_count: int
    error: str | None = None
    created_at: datetime | None = None


class DocumentUrlCreate(BaseSchema):
    kb_id: str
    url: str = Field(min_length=8, max_length=1024)


# --- Agent runs ------------------------------------------------------------


class AgentRunCreate(BaseSchema):
    prompt: str = Field(min_length=1, max_length=8000)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    idempotency_key: str | None = Field(default=None, max_length=128)


class Citation(BaseSchema):
    chunk_id: str
    filename: str
    source_url: str = ""
    score: float = 0.0


class AgentRunRead(BaseSchema):
    id: str
    trace_id: str
    status: str
    answer: str | None = None
    intent: str | None = None
    risk_tier: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    approval_id: str | None = None
    replayed: bool = False


# --- Approvals -------------------------------------------------------------


class ApprovalVoteRead(BaseSchema):
    voter_id: str
    decision: str
    note: str = ""
    created_at: datetime | None = None


class ApprovalRead(BaseSchema):
    id: str
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str
    risk_tier: str
    required_approvals: int
    approvals_received: int = 0
    requested_by: str
    votes: list[ApprovalVoteRead] = Field(default_factory=list)
    created_at: datetime | None = None
    decided_at: datetime | None = None


class ApprovalDecision(BaseSchema):
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=2000)


# --- Tickets ---------------------------------------------------------------


class TicketRead(BaseSchema):
    id: str
    title: str
    description: str
    priority: str
    status: str
    risk_tier: str
    assignee_id: str | None = None
    run_id: str | None = None
    sla_due_at: datetime | None = None
    resolution: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TicketEventRead(BaseSchema):
    id: str
    action: str
    from_status: str | None = None
    to_status: str | None = None
    note: str = ""
    actor_id: str | None = None
    created_at: datetime | None = None


class TicketActionRequest(BaseSchema):
    """A ticket execution request. Goes through the same approval gate."""

    action: Literal[
        "triage",
        "start",
        "wait_on_merchant",
        "resolve",
        "close",
        "reopen",
        "cancel",
        "assign",
    ]
    note: str = Field(default="", max_length=2000)
    resolution: str = Field(default="", max_length=4000)
    assignee_id: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)


class AgentRunRequest(BaseSchema):
    """Start a run from the operator UI."""

    prompt: str = Field(min_length=1, max_length=8000)


# --- Audit -----------------------------------------------------------------


class AuditLogRead(BaseSchema):
    id: str
    actor_id: str | None = None
    action: str
    resource: str
    resource_id: str | None = None
    decision: str
    reason: str = ""
    trace_id: str | None = None
    created_at: datetime | None = None


class PaginatedAuditLogs(BaseSchema):
    items: list[AuditLogRead]
    total: int


# --- Usage -----------------------------------------------------------------


class UsageSummary(BaseSchema):
    runs: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class HealthRead(BaseSchema):
    status: str
    service: str
    version: str
    database: str
    embedding_provider: str
    vector_backend: str
    llm_configured: bool
