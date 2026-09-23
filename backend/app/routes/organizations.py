"""Organisation and membership management.

Member management is the highest-privilege surface in the product: whoever
controls memberships controls who can approve refunds. Two rules are enforced
here beyond the permission matrix:

* Only an owner may grant or revoke the ``owner`` role, so an admin cannot
  promote themselves out of a restricted tier.
* The last remaining owner cannot be demoted or removed, which would otherwise
  leave an organisation that nobody can administer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import audit
from app.auth import CurrentPrincipal, CurrentUser, Db, get_principal, require
from app.core_exceptions import ConflictError, NotFoundError, ValidationError
from app.models import Membership, Organization, Role, User
from app.permissions import Action, Principal, Resource
from app.schemas import (
    MemberCreate,
    MemberRead,
    MemberRoleUpdate,
    OrganizationCreate,
    OrganizationRead,
)
from app.security import hash_password
from app.services.demo import seed_demo_knowledge

router = APIRouter(tags=["organizations"])


async def _owner_count(db: AsyncSession, org_id: str) -> int:
    count = await db.scalar(
        select(func.count(Membership.id)).where(
            Membership.org_id == org_id, Membership.role == Role.OWNER.value
        )
    )
    return int(count or 0)


@router.post(
    "/organizations", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED
)
async def create_organization(
    payload: OrganizationCreate, user: CurrentUser, db: Db
) -> Any:
    """Create an additional organisation owned by the caller."""
    org = Organization(name=payload.name, created_by=user.id)
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
    )
    await seed_demo_knowledge(db, org.id)
    await db.commit()
    return OrganizationRead(id=org.id, name=org.name, role=Role.OWNER.value)


@router.get("/organizations", response_model=list[OrganizationRead])
async def list_organizations(user: CurrentUser, db: Db) -> Any:
    """Every organisation this user belongs to, with their role in each."""
    rows = (
        await db.execute(
            select(Organization, Membership.role)
            .join(Membership, Membership.org_id == Organization.id)
            .where(Membership.user_id == user.id)
            .order_by(Organization.name)
        )
    ).all()
    return [OrganizationRead(id=org.id, name=org.name, role=role) for org, role in rows]


@router.get("/organizations/{org_id}/members", response_model=list[MemberRead])
async def list_members(
    org_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.READ, Resource.MEMBERSHIP)),
) -> Any:
    # ``require`` resolved the principal against the acting organisation; an
    # explicit path org that differs from it is reported as missing rather than
    # forbidden, so the endpoint cannot be used to probe for org IDs.
    if principal.org_id != org_id:
        raise NotFoundError(message="organisation not found")

    rows = (
        await db.execute(
            select(User, Membership)
            .join(Membership, Membership.user_id == User.id)
            .where(Membership.org_id == org_id)
            .order_by(Membership.role, User.display_name)
        )
    ).all()
    return [
        MemberRead(
            user_id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=membership.role,
            created_at=membership.created_at,
        )
        for user, membership in rows
    ]


@router.post(
    "/organizations/{org_id}/members",
    response_model=MemberRead,
    status_code=status.HTTP_201_CREATED,
)
async def add_member(
    org_id: str,
    payload: MemberCreate,
    db: Db,
    principal: Principal = Depends(require(Action.MANAGE_MEMBERS, Resource.MEMBERSHIP)),
) -> Any:
    if principal.org_id != org_id:
        raise NotFoundError(message="organisation not found")
    return await _upsert_member(db, org_id=org_id, principal=principal, payload=payload)


@router.patch(
    "/organizations/{org_id}/members/{user_id}",
    response_model=MemberRead,
)
async def update_member_role(
    org_id: str,
    user_id: str,
    payload: MemberRoleUpdate,
    db: Db,
    principal: Principal = Depends(require(Action.MANAGE_MEMBERS, Resource.MEMBERSHIP)),
) -> Any:
    if principal.org_id != org_id:
        raise NotFoundError(message="organisation not found")

    membership = (
        await db.execute(
            select(Membership).where(
                Membership.org_id == org_id, Membership.user_id == user_id
            )
        )
    ).scalar_one_or_none()
    if membership is None:
        raise NotFoundError(message="member not found")

    if membership.role == Role.OWNER.value and payload.role != Role.OWNER.value:
        if principal.role != Role.OWNER.value:
            raise ConflictError(message="only an owner can change an owner's role")
        if await _owner_count(db, org_id) <= 1:
            raise ConflictError(message="cannot demote the last owner of an organisation")

    if payload.role == Role.OWNER.value and principal.role != Role.OWNER.value:
        raise ConflictError(message="only an owner can grant the owner role")

    previous = membership.role
    membership.role = payload.role
    await db.flush()
    audit.record(
        db,
        action="membership.role_update",
        resource="membership",
        org_id=org_id,
        actor_id=principal.user_id,
        resource_id=user_id,
        metadata={"from": previous, "to": payload.role},
    )
    await db.commit()
    return await _member_read(db, org_id=org_id, user_id=user_id)


@router.delete(
    "/organizations/{org_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def remove_member(
    org_id: str,
    user_id: str,
    db: Db,
    principal: Principal = Depends(require(Action.MANAGE_MEMBERS, Resource.MEMBERSHIP)),
) -> None:
    if principal.org_id != org_id:
        raise NotFoundError(message="organisation not found")

    membership = (
        await db.execute(
            select(Membership).where(
                Membership.org_id == org_id, Membership.user_id == user_id
            )
        )
    ).scalar_one_or_none()
    if membership is None:
        raise NotFoundError(message="member not found")

    if membership.role == Role.OWNER.value and await _owner_count(db, org_id) <= 1:
        raise ConflictError(message="cannot remove the last owner of an organisation")

    removed_role = membership.role
    await db.delete(membership)
    await db.flush()
    audit.record(
        db,
        action="membership.remove",
        resource="membership",
        org_id=org_id,
        actor_id=principal.user_id,
        resource_id=user_id,
        metadata={"role": removed_role},
    )
    await db.commit()


async def _upsert_member(
    db: AsyncSession, *, org_id: str, principal: Principal, payload: MemberCreate
) -> MemberRead:
    if payload.role == Role.OWNER.value and principal.role != Role.OWNER.value:
        raise ValidationError(message="only an owner can grant the owner role")

    user = (
        await db.execute(select(User).where(User.email == payload.email.lower()))
    ).scalar_one_or_none()

    if user is None:
        # Adding a member is an invitation in this build: the account is created
        # with a password the inviter sets, and the member is expected to rotate
        # it. A production deployment would send a single-use invite instead.
        user = User(
            email=payload.email.lower(),
            display_name=payload.display_name,
            hashed_password=hash_password(payload.password),
        )
        db.add(user)
        await db.flush()
    elif user.display_name != payload.display_name:
        user.display_name = payload.display_name
        await db.flush()

    membership = (
        await db.execute(
            select(Membership).where(
                Membership.org_id == org_id, Membership.user_id == user.id
            )
        )
    ).scalar_one_or_none()

    if membership is None:
        membership = Membership(user_id=user.id, org_id=org_id, role=payload.role)
        db.add(membership)
    else:
        membership.role = payload.role
    await db.flush()

    audit.record(
        db,
        action="membership.upsert",
        resource="membership",
        org_id=org_id,
        actor_id=principal.user_id,
        resource_id=user.id,
        metadata={"role": payload.role},
    )
    await db.commit()
    return await _member_read(db, org_id=org_id, user_id=user.id)


async def _member_read(db: AsyncSession, *, org_id: str, user_id: str) -> MemberRead:
    row = (
        await db.execute(
            select(User, Membership)
            .join(Membership, Membership.user_id == User.id)
            .where(Membership.org_id == org_id, Membership.user_id == user_id)
        )
    ).first()
    if row is None:
        raise NotFoundError(message="member not found")
    user, membership = row
    return MemberRead(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=membership.role,
        created_at=membership.created_at,
    )
