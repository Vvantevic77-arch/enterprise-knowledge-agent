"""Authentication and tenant resolution.

Identity is established by a signed JWT and nothing else:

1. The bearer token proves *who* the caller is.
2. ``X-Org-ID`` selects *which* tenant they are acting in, and is verified
   against a real membership row before it is trusted.
3. The resulting ``Principal`` carries the role, and every route checks it
   against the action/resource matrix in ``app.permissions``.

The previous implementation read ``X-User-ID`` straight from the request and
looked it up in the database, which meant any caller who knew a user's
identifier *became* that user. The header path now exists only behind the
``ALLOW_HEADER_IDENTITY`` flag, which defaults to off and is intended for a
single-tenant local sandbox.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

import jwt
from fastapi import Depends, Header, Query, Request, WebSocket, WebSocketException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import audit
from app.config import settings
from app.core_exceptions import AuthenticationError, AuthorizationError, NotFoundError
from app.db import get_db
from app.models import Membership, RefreshSession, Role, User, utcnow
from app.permissions import Action, Principal, Resource
from app.security import create_access_token, create_refresh_token, decode_token

logger = logging.getLogger(__name__)

# ``auto_error=False`` so we can raise our own typed error and return a
# consistent JSON body with a WWW-Authenticate header.
_bearer = HTTPBearer(auto_error=False, description="JWT access token")

Db = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Token → user
# ---------------------------------------------------------------------------


async def _user_from_token(db: AsyncSession, token: str) -> User:
    try:
        payload = decode_token(token, expected_type="access")
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError(message="Token has expired", details={"reason": "expired"}) from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(message="Invalid token", details={"reason": "invalid"}) from exc

    user = (
        await db.execute(select(User).where(User.id == payload["sub"]))
    ).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthenticationError(message="Account is not active", details={"reason": "inactive"})
    return user


async def get_current_user(
    db: Db,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    x_user_id: Annotated[str | None, Header()] = None,
    x_org_id: Annotated[str | None, Header()] = None,
) -> User:
    """Resolve the authenticated user, or raise 401."""
    if credentials is not None and credentials.credentials:
        return await _user_from_token(db, credentials.credentials)

    if settings.allow_header_identity and x_user_id:
        # Documented local-sandbox escape hatch. It is intentionally noisy.
        logger.warning(
            "header identity accepted for user %r because ALLOW_HEADER_IDENTITY is enabled",
            x_user_id,
        )
        user = (
            await db.execute(select(User).where(User.email == x_user_id))
        ).scalar_one_or_none()
        if user is None:
            raise AuthenticationError(message="Unknown local user", details={"user": x_user_id})
        return user

    raise AuthenticationError(
        message="Missing bearer token",
        details={"hint": "POST /v1/auth/login and send Authorization: Bearer <token>"},
    )


CurrentUser = Annotated[User, Depends(get_current_user)]


# ---------------------------------------------------------------------------
# User → organisation-scoped principal
# ---------------------------------------------------------------------------


async def _membership_role(db: AsyncSession, *, user_id: str, org_id: str) -> str | None:
    return await db.scalar(
        select(Membership.role).where(
            Membership.user_id == user_id, Membership.org_id == org_id
        )
    )


async def _default_membership(db: AsyncSession, user_id: str) -> Membership | None:
    """Pick the tenant to act in when the client did not name one.

    Highest role wins so a user who owns one org and merely views another does
    not silently land in the read-only one.
    """
    rank = {
        Role.OWNER.value: 0,
        Role.ADMIN.value: 1,
        Role.MEMBER.value: 2,
        Role.VIEWER.value: 3,
    }
    rows = (
        await db.execute(select(Membership).where(Membership.user_id == user_id))
    ).scalars().all()
    if not rows:
        return None
    return sorted(rows, key=lambda m: rank.get(m.role, 9))[0]


async def get_principal(
    db: Db,
    user: CurrentUser,
    x_org_id: Annotated[str | None, Header()] = None,
) -> Principal:
    """Bind the authenticated user to a tenant they actually belong to.

    A valid token for a user with no membership in the requested organisation is
    a 403 — authentication succeeded, authorisation did not.
    """
    if x_org_id:
        role = await _membership_role(db, user_id=user.id, org_id=x_org_id)
        if role is None:
            audit.record_denied(
                db,
                action="tenant.access",
                resource="organization",
                org_id=x_org_id,
                actor_id=user.id,
                reason="no membership for requested organisation",
            )
            raise AuthorizationError(
                message="You are not a member of this organisation",
                details={"org_id": x_org_id},
            )
        return Principal(
            user_id=user.id,
            org_id=x_org_id,
            role=role,
            email=user.email,
            display_name=user.display_name,
        )

    membership = await _default_membership(db, user.id)
    if membership is None:
        raise NotFoundError(
            message="No organisation membership found for this account",
            details={"hint": "create one with POST /v1/organizations"},
        )
    return Principal(
        user_id=user.id,
        org_id=membership.org_id,
        role=membership.role,
        email=user.email,
        display_name=user.display_name,
    )


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------


def require(action: Action, resource: Resource):
    """Route dependency factory enforcing the action/resource matrix.

    Usage::

        @router.post("/knowledge-bases")
        async def create(p: Principal = Depends(require(Action.CREATE, Resource.KNOWLEDGE_BASE))): ...
    """

    async def _guard(
        db: Db,
        principal: CurrentPrincipal,
        request: Request,
    ) -> Principal:
        trace_id = getattr(request.state, "trace_id", None)
        if not principal.can(action, resource):
            audit.record_denied(
                db,
                action=str(action),
                resource=str(resource),
                org_id=principal.org_id,
                actor_id=principal.user_id,
                reason=f"role {principal.role!r} lacks {action}:{resource}",
                trace_id=trace_id,
            )
            raise AuthorizationError(
                message=f"Role '{principal.role}' cannot {action} {resource}",
                details={"required": f"{action}:{resource}", "role": principal.role},
            )
        audit.record(
            db,
            action=f"{action}.allowed",
            resource=str(resource),
            org_id=principal.org_id,
            actor_id=principal.user_id,
            trace_id=trace_id,
        )
        return principal

    return _guard


# ---------------------------------------------------------------------------
# WebSocket authentication
# ---------------------------------------------------------------------------


async def authenticate_websocket(
    websocket: WebSocket,
    db: AsyncSession,
    *,
    token: str | None,
    org_id: str | None,
) -> Principal:
    """Authenticate a WebSocket before ``accept()``.

    The token arrives either as the first ``Sec-WebSocket-Protocol`` value
    (preferred — browsers can set it and it stays out of the URL) or as a
    ``token`` query parameter for non-browser clients. Query strings are logged
    by proxies, which is why they are the fallback and not the default.
    """
    if not token:
        raise AuthenticationError(message="Missing WebSocket token")
    try:
        payload = decode_token(token, expected_type="access")
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(message="Invalid WebSocket token") from exc

    user = (await db.execute(select(User).where(User.id == payload["sub"]))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthenticationError(message="Account is not active")

    target_org = org_id or await db.scalar(
        select(Membership.org_id).where(Membership.user_id == user.id).limit(1)
    )
    if not target_org:
        raise AuthorizationError(message="No organisation membership found")
    role = await _membership_role(db, user_id=user.id, org_id=target_org)
    if role is None:
        raise AuthorizationError(message="You are not a member of this organisation")
    return Principal(
        user_id=user.id, org_id=target_org, role=role, email=user.email, display_name=user.display_name
    )


def websocket_token(websocket: WebSocket, query_token: str | None) -> str | None:
    """Prefer the subprotocol header; fall back to the query parameter."""
    protocol = websocket.headers.get("sec-websocket-protocol")
    if protocol:
        # A client may offer several protocols; the first is the token.
        return protocol.split(",")[0].strip()
    return query_token


# ---------------------------------------------------------------------------
# Refresh sessions
# ---------------------------------------------------------------------------


async def issue_token_pair(db: AsyncSession, user: User) -> tuple[str, str]:
    refresh = create_refresh_token(user.id)
    payload = decode_token(refresh, expected_type="refresh")
    db.add(
        RefreshSession(
            user_id=user.id,
            jti=payload["jti"],
            expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
        )
    )
    await db.flush()
    return create_access_token(user.id), refresh


async def rotate_refresh_token(db: AsyncSession, token: str) -> tuple[str, str]:
    """Exchange a refresh token for a new pair, invalidating the old one.

    Rotation is what makes a stolen refresh token detectable: the legitimate
    client's next refresh fails because its ``jti`` is already revoked.
    """
    try:
        payload = decode_token(token, expected_type="refresh")
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(message="Invalid refresh token") from exc

    session = (
        await db.execute(select(RefreshSession).where(RefreshSession.jti == payload["jti"]))
    ).scalar_one_or_none()
    if session is None or session.revoked_at is not None:
        raise AuthenticationError(message="Refresh token has been revoked")

    user = (await db.execute(select(User).where(User.id == payload["sub"]))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthenticationError(message="Account is not active")

    session.revoked_at = utcnow()
    await db.flush()
    return await issue_token_pair(db, user)


async def revoke_refresh_token(db: AsyncSession, token: str) -> None:
    try:
        payload = decode_token(token, expected_type="refresh")
    except jwt.InvalidTokenError:
        # Logout of an already-invalid token is a no-op, not an error.
        return
    session = (
        await db.execute(select(RefreshSession).where(RefreshSession.jti == payload["jti"]))
    ).scalar_one_or_none()
    if session is not None and session.revoked_at is None:
        session.revoked_at = utcnow()
        await db.flush()
