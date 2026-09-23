"""Action + resource access control.

Permissions are evaluated as ``(action, resource)`` pairs rather than scattered
role checks, so the same matrix governs REST routes, WebSocket subscriptions,
CLI commands, and Celery tasks. ``assert_allowed`` is the single place where a
decision becomes an ``AuthorizationError``, and every decision — allow or deny —
is written to the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.models import RiskTier, Role


class Action(StrEnum):
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    APPROVE = "approve"  # decide an approval request
    EXECUTE = "execute"  # run a gated tool / advance a ticket
    MANAGE_MEMBERS = "manage_members"
    MANAGE_ORG = "manage_org"


class Resource(StrEnum):
    ORGANIZATION = "organization"
    MEMBERSHIP = "membership"
    KNOWLEDGE_BASE = "knowledge_base"
    DOCUMENT = "document"
    AGENT_RUN = "agent_run"
    APPROVAL = "approval"
    TICKET = "ticket"
    AUDIT_LOG = "audit_log"
    USAGE = "usage"


#: Role → the exact set of permitted ``(action, resource)`` pairs.
#: Anything absent is denied by default.
ROLE_PERMISSIONS: dict[str, frozenset[tuple[str, str]]] = {
    Role.VIEWER.value: frozenset(
        {
            (Action.READ.value, Resource.KNOWLEDGE_BASE.value),
            (Action.READ.value, Resource.DOCUMENT.value),
            (Action.READ.value, Resource.AGENT_RUN.value),
            (Action.READ.value, Resource.TICKET.value),
            (Action.READ.value, Resource.MEMBERSHIP.value),
        }
    ),
    Role.MEMBER.value: frozenset(
        {
            (Action.READ.value, Resource.KNOWLEDGE_BASE.value),
            (Action.READ.value, Resource.DOCUMENT.value),
            (Action.READ.value, Resource.AGENT_RUN.value),
            (Action.READ.value, Resource.TICKET.value),
            (Action.READ.value, Resource.MEMBERSHIP.value),
            (Action.CREATE.value, Resource.DOCUMENT.value),
            (Action.UPDATE.value, Resource.DOCUMENT.value),
            (Action.DELETE.value, Resource.DOCUMENT.value),
            (Action.CREATE.value, Resource.AGENT_RUN.value),
            # A member may *raise* an approval but never decide one.
            (Action.CREATE.value, Resource.APPROVAL.value),
            (Action.READ.value, Resource.APPROVAL.value),
            (Action.CREATE.value, Resource.TICKET.value),
            (Action.READ.value, Resource.USAGE.value),
        }
    ),
    Role.ADMIN.value: frozenset(
        {
            (Action.READ.value, Resource.KNOWLEDGE_BASE.value),
            (Action.CREATE.value, Resource.KNOWLEDGE_BASE.value),
            (Action.UPDATE.value, Resource.KNOWLEDGE_BASE.value),
            (Action.DELETE.value, Resource.KNOWLEDGE_BASE.value),
            (Action.READ.value, Resource.DOCUMENT.value),
            (Action.CREATE.value, Resource.DOCUMENT.value),
            (Action.UPDATE.value, Resource.DOCUMENT.value),
            (Action.DELETE.value, Resource.DOCUMENT.value),
            (Action.READ.value, Resource.AGENT_RUN.value),
            (Action.CREATE.value, Resource.AGENT_RUN.value),
            (Action.READ.value, Resource.APPROVAL.value),
            (Action.CREATE.value, Resource.APPROVAL.value),
            (Action.APPROVE.value, Resource.APPROVAL.value),
            (Action.READ.value, Resource.TICKET.value),
            (Action.CREATE.value, Resource.TICKET.value),
            (Action.UPDATE.value, Resource.TICKET.value),
            (Action.EXECUTE.value, Resource.TICKET.value),
            (Action.READ.value, Resource.AUDIT_LOG.value),
            (Action.READ.value, Resource.USAGE.value),
            (Action.READ.value, Resource.MEMBERSHIP.value),
        }
    ),
}

# Owner = every admin permission, plus organisation ownership and destructive
# operations. Derived rather than duplicated so the two cannot drift apart.
ROLE_PERMISSIONS[Role.OWNER.value] = ROLE_PERMISSIONS[Role.ADMIN.value] | frozenset(
    {
        (Action.MANAGE_MEMBERS.value, Resource.MEMBERSHIP.value),
        (Action.MANAGE_ORG.value, Resource.ORGANIZATION.value),
        (Action.DELETE.value, Resource.KNOWLEDGE_BASE.value),
        (Action.DELETE.value, Resource.AGENT_RUN.value),
    }
)

#: Roles allowed to approve a tool call of a given risk tier.
RISK_APPROVER_ROLES: dict[str, frozenset[str]] = {
    RiskTier.LOW.value: frozenset({Role.OWNER.value, Role.ADMIN.value, Role.MEMBER.value}),
    RiskTier.MEDIUM.value: frozenset({Role.OWNER.value, Role.ADMIN.value}),
    RiskTier.HIGH.value: frozenset({Role.OWNER.value, Role.ADMIN.value}),
    RiskTier.CRITICAL.value: frozenset({Role.OWNER.value}),
}

#: Risk tier → how many *distinct* humans must approve before execution.
REQUIRED_APPROVALS: dict[str, int] = {
    RiskTier.LOW.value: 0,
    RiskTier.MEDIUM.value: 1,
    RiskTier.HIGH.value: 2,
    RiskTier.CRITICAL.value: 2,
}


@dataclass(frozen=True)
class Principal:
    """The authenticated identity, already resolved against one organisation."""

    user_id: str
    org_id: str
    role: str
    email: str = ""
    display_name: str = ""

    def can(self, action: Action | str, resource: Resource | str) -> bool:
        pair = (str(action), str(resource))
        return pair in ROLE_PERMISSIONS.get(self.role, frozenset())

    def can_approve(self, risk_tier: str) -> bool:
        return self.role in RISK_APPROVER_ROLES.get(risk_tier, frozenset())

    def require(self, action: Action | str, resource: Resource | str) -> None:
        if not self.can(action, resource):
            from app.core_exceptions import AuthorizationError

            raise AuthorizationError(
                message=f"Role '{self.role}' cannot {action} {resource}",
                details={"required": f"{action}:{resource}", "role": self.role},
            )


def required_approvals(risk_tier: str) -> int:
    return REQUIRED_APPROVALS.get(risk_tier, 1)


def approver_roles(risk_tier: str) -> frozenset[str]:
    return RISK_APPROVER_ROLES.get(risk_tier, frozenset())
