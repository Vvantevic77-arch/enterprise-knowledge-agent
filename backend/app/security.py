"""Password hashing and JWT primitives.

Everything that turns a credential into an identity lives here so that the
FastAPI dependency layer and the CLI provisioning script cannot drift apart.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
import jwt

from app.config import settings

TokenType = Literal["access", "refresh"]

# bcrypt hashes at most 72 bytes; longer inputs are truncated by the library
# rather than rejected, which would silently make two passwords equivalent.
_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > _MAX_PASSWORD_BYTES:
        raise ValueError("password must be at most 72 bytes")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    raw = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    try:
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash in the database — treat as a failed login, not a 500.
        return False


def _encode(subject: str, token_type: TokenType, *, org_id: str | None, ttl: timedelta, extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        # ``jti`` gives every token a unique identity so a refresh can be
        # revoked or traced without touching the signing key.
        "jti": secrets.token_urlsafe(16),
    }
    if org_id:
        payload["org"] = org_id
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, *, org_id: str | None = None) -> str:
    return _encode(
        subject,
        "access",
        org_id=org_id,
        ttl=timedelta(minutes=settings.access_token_ttl_minutes),
    )


def create_refresh_token(subject: str) -> str:
    return _encode(subject, "refresh", org_id=None, ttl=timedelta(days=settings.refresh_token_ttl_days))


def decode_token(token: str, *, expected_type: TokenType | None = None) -> dict[str, Any]:
    """Decode and validate a token.

    Raises ``jwt.InvalidTokenError`` (or a subclass) for anything untrusted:
    bad signature, expired, wrong algorithm, malformed payload.
    """
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "sub", "type"]},
    )
    token_type = payload.get("type")
    if token_type not in {"access", "refresh"}:
        raise jwt.InvalidTokenError("unexpected token type")
    if expected_type and token_type != expected_type:
        raise jwt.InvalidTokenError(f"{expected_type} token required")
    return payload


def constant_time_equals(left: str, right: str) -> bool:
    """Use for every secret comparison instead of ``==``."""
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
