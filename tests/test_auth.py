"""Authentication, session handling, and tenant-resolution behaviour."""

from __future__ import annotations

import pytest

from tests.conftest import Actor


async def test_register_creates_organisation_and_seeds_corpus(owner: Actor) -> None:
    response = await owner.client.get("/v1/knowledge-bases", headers=owner.headers)
    assert response.status_code == 200
    bases = response.json()
    assert len(bases) == 1
    assert bases[0]["document_count"] >= 10


async def test_login_with_wrong_password_is_rejected(owner: Actor) -> None:
    response = await owner.client.post(
        "/v1/auth/login", json={"email": "owner@acme-payments.example", "password": "wrong"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


async def test_login_unknown_email_is_indistinguishable_from_wrong_password(owner: Actor) -> None:
    wrong_password = await owner.client.post(
        "/v1/auth/login", json={"email": "owner@acme-payments.example", "password": "wrong"}
    )
    unknown_user = await owner.client.post(
        "/v1/auth/login", json={"email": "nobody@nowhere.example", "password": "wrong"}
    )
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json()["error"]["message"] == unknown_user.json()["error"]["message"]


async def test_duplicate_registration_is_rejected(client, owner: Actor) -> None:
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "owner@acme-payments.example",
            "password": "another-password-1",
            "display_name": "Impostor",
            "organization_name": "Copycat Payments",
        },
    )
    assert response.status_code == 409


async def test_requests_without_a_token_are_rejected(owner: Actor) -> None:
    """The old header-identity path must not work by default."""
    response = await owner.client.get(
        "/v1/knowledge-bases", headers={"X-User-ID": owner.id, "X-Org-ID": owner.org_id}
    )
    assert response.status_code == 401


async def test_forged_user_header_cannot_impersonate(owner: Actor, admin: Actor) -> None:
    """Knowing another user's ID must not be enough to act as them."""
    response = await admin.client.get(
        "/v1/tickets",
        headers={"X-User-ID": owner.id, "X-Org-ID": owner.org_id},
    )
    assert response.status_code == 401


async def test_access_token_cannot_be_used_as_refresh_token(owner: Actor) -> None:
    response = await owner.client.post(
        "/v1/auth/refresh", json={"refresh_token": owner.token}
    )
    assert response.status_code == 401


async def test_refresh_rotates_and_revokes_the_previous_token(client, owner: Actor) -> None:
    login = await client.post(
        "/v1/auth/login",
        json={"email": "owner@acme-payments.example", "password": "owner-password-1"},
    )
    original = login.json()["refresh_token"]

    first = await client.post("/v1/auth/refresh", json={"refresh_token": original})
    assert first.status_code == 200
    assert first.json()["refresh_token"] != original

    # Replaying the consumed token must fail — that is what makes theft visible.
    replay = await client.post("/v1/auth/refresh", json={"refresh_token": original})
    assert replay.status_code == 401


async def test_logout_revokes_the_refresh_token(client, owner: Actor) -> None:
    login = await client.post(
        "/v1/auth/login",
        json={"email": "owner@acme-payments.example", "password": "owner-password-1"},
    )
    refresh_token = login.json()["refresh_token"]
    logout = await client.post("/v1/auth/logout", json={"refresh_token": refresh_token})
    assert logout.status_code == 204
    assert (
        await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    ).status_code == 401


async def test_token_for_other_organisation_is_forbidden(client, owner: Actor, second_org_owner: Actor) -> None:
    """A valid token plus someone else's org header is 403, not a data leak."""
    response = await owner.client.get(
        "/v1/tickets",
        headers={
            "Authorization": f"Bearer {owner.token}",
            "X-Org-ID": second_org_owner.org_id,
        },
    )
    assert response.status_code == 403


async def test_deactivated_user_is_rejected(client, db, owner: Actor) -> None:
    from app.models import User

    user = await db.get(User, owner.id)
    assert user is not None
    user.is_active = False
    await db.commit()

    response = await owner.client.get("/v1/tickets", headers=owner.headers)
    assert response.status_code == 401


@pytest.mark.parametrize("path", ["/v1/health"])
async def test_health_is_public(client, path: str) -> None:
    response = await client.get(path)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Surfacing the embedding provider makes an accidental offline fallback visible.
    assert body["embedding_provider"] == "deterministic"


async def test_change_password_requires_the_current_one(client, owner: Actor) -> None:
    response = await client.post(
        "/v1/auth/change-password",
        params={"current_password": "not-the-password", "new_password": "a-new-password-1"},
        headers=owner.headers,
    )
    assert response.status_code == 401
