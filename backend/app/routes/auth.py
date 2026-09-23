"""Registration, login, token refresh, and logout."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import audit
from app.auth import (
    CurrentUser,
    Db,
    issue_token_pair,
    revoke_refresh_token,
    rotate_refresh_token,
)
from app.config import settings
from app.core_exceptions import AuthenticationError, ConflictError, ValidationError
from app.models import Membership, Organization, Role, User
from app.schemas import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserRead,
)
from app.security import hash_password, verify_password
from app.services.demo import seed_demo_knowledge

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: Db) -> Any:
    """Create a user, their first organisation, and the demo corpus.

    The registrant becomes the organisation owner — this is the only path that
    mints an owner, so there is no way to grant yourself owner on someone
    else's tenant.
    """
    existing = (
        await db.execute(select(User).where(User.email == payload.email.lower()))
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError(
            message="Email is already registered", details={"email": payload.email}
        )

    user = User(
        email=payload.email.lower(),
        display_name=payload.display_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.flush()

    org = Organization(name=payload.organization_name, created_by=user.id)
    db.add(org)
    await db.flush()
    db.add(Membership(user_id=user.id, org_id=org.id, role=Role.OWNER.value))
    await db.flush()

    audit.record(
        db,
        action="organization.create",
        resource="organization",
        org_id=org.id,
        actor_id=user.id,
        resource_id=org.id,
        reason="registration",
    )
    await seed_demo_knowledge(db, org.id)

    access, refresh = await issue_token_pair(db, user)
    await db.commit()
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest, db: Db) -> Any:
    user = (
        await db.execute(select(User).where(User.email == payload.email.lower()))
    ).scalar_one_or_none()

    # Same message and same work for "no such user" and "wrong password", so
    # the response cannot be used to enumerate registered emails.
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise AuthenticationError(message="Incorrect email or password")
    if not user.is_active:
        raise AuthenticationError(message="Account is disabled")

    access, refresh = await issue_token_pair(db, user)
    audit.record(
        db,
        action="auth.login",
        resource="user",
        org_id=None,
        actor_id=user.id,
        resource_id=user.id,
    )
    await db.commit()
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, db: Db) -> Any:
    """Exchange a refresh token for a new pair, revoking the old one."""
    access, new_refresh = await rotate_refresh_token(db, payload.refresh_token)
    await db.commit()
    return TokenPair(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def logout(payload: LogoutRequest, db: Db) -> None:
    await revoke_refresh_token(db, payload.refresh_token)
    await db.commit()


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUser) -> Any:
    return UserRead(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_app_admin=user.is_app_admin,
    )


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def change_password(
    current_password: str,
    new_password: str,
    user: CurrentUser,
    db: Db,
) -> None:
    if not verify_password(current_password, user.hashed_password):
        raise AuthenticationError(message="Current password is incorrect")
    if len(new_password) < 8:
        raise ValidationError(message="New password must be at least 8 characters")
    user.hashed_password = hash_password(new_password)
    await db.flush()
    audit.record(
        db, action="auth.change_password", resource="user", org_id=None,
        actor_id=user.id, resource_id=user.id,
    )
    await db.commit()
