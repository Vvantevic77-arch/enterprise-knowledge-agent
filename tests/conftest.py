"""Shared fixtures.

The app reads its database URL at import time, so the test database is selected
via environment variables *before* any ``app.*`` import happens. Every test gets
a fresh schema so runs cannot contaminate each other.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Must be set before importing app.config, which builds Settings at import time.
_DB_FILE = Path(tempfile.gettempdir()) / "payment_ops_test.db"
os.environ.setdefault("APP_ENV", "test")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_FILE}"
os.environ["EMBEDDING_PROVIDER"] = "deterministic"
os.environ["JWT_SECRET"] = "test-only-secret-not-used-outside-tests"
os.environ["ALLOW_HEADER_IDENTITY"] = "false"
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("DEEPSEEK_API_KEY", None)

import httpx  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Membership, Organization, Role, User  # noqa: E402
from app.security import hash_password  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_database() -> None:
    if _DB_FILE.exists():
        _DB_FILE.unlink()


@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    async with SessionLocal() as session:
        yield session


@pytest.fixture
async def client(db: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


class Actor:
    """A registered user plus the headers needed to act as them."""

    def __init__(self, client: httpx.AsyncClient, user_id: str, org_id: str, role: str, token: str):
        self.client = client
        self.id = user_id
        self.org_id = org_id
        self.role = role
        self.token = token

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "X-Org-ID": self.org_id}


async def _register(
    client: httpx.AsyncClient,
    *,
    email: str,
    password: str,
    org_name: str,
    display_name: str,
) -> Actor:
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "display_name": display_name,
            "organization_name": org_name,
        },
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    me = await client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    user_id = me.json()["id"]

    orgs = await client.get("/v1/organizations", headers={"Authorization": f"Bearer {token}"})
    assert orgs.status_code == 200, orgs.text
    org_id = orgs.json()[0]["id"]
    return Actor(client, user_id, org_id, Role.OWNER.value, token)


@pytest.fixture
async def owner(client: httpx.AsyncClient) -> Actor:
    """An organisation owner — the highest privilege in one tenant."""
    return await _register(
        client,
        email="owner@acme-payments.example",
        password="owner-password-1",
        org_name="Acme Payments",
        display_name="Org Owner",
    )


@pytest.fixture
async def second_org_owner(client: httpx.AsyncClient) -> Actor:
    """Owner of a *different* tenant, used for isolation tests."""
    return await _register(
        client,
        email="owner@rival-payments.example",
        password="owner-password-2",
        org_name="Rival Payments",
        display_name="Rival Owner",
    )


async def add_member(client: httpx.AsyncClient, owner: Actor, *, email: str, role: str) -> Actor:
    """Add a teammate to the owner's organisation and log in as them."""
    response = await client.post(
        f"/v1/organizations/{owner.org_id}/members",
        headers=owner.headers,
        json={
            "email": email,
            "display_name": email.split("@")[0],
            "password": "teammate-password-1",
            "role": role,
        },
    )
    assert response.status_code == 201, response.text
    user_id = response.json()["user_id"]

    login = await client.post(
        "/v1/auth/login", json={"email": email, "password": "teammate-password-1"}
    )
    assert login.status_code == 200, login.text
    return Actor(client, user_id, owner.org_id, role, login.json()["access_token"])


@pytest.fixture
async def admin(client: httpx.AsyncClient, owner: Actor) -> Actor:
    return await add_member(client, owner, email="admin@acme-payments.example", role=Role.ADMIN.value)


@pytest.fixture
async def member(client: httpx.AsyncClient, owner: Actor) -> Actor:
    return await add_member(client, owner, email="member@acme-payments.example", role=Role.MEMBER.value)


@pytest.fixture
async def viewer(client: httpx.AsyncClient, owner: Actor) -> Actor:
    return await add_member(client, owner, email="viewer@acme-payments.example", role=Role.VIEWER.value)


async def seed_corpus(client: httpx.AsyncClient, actor: Actor) -> str:
    """Return the demo knowledge base ID for an actor's organisation.

    Registration already seeds it; this just looks it up so tests do not depend
    on the seeder's internal naming.
    """
    response = await client.get("/v1/knowledge-bases", headers=actor.headers)
    assert response.status_code == 200, response.text
    bases = response.json()
    assert bases, "registration should have seeded the demo corpus"
    return bases[0]["id"]
