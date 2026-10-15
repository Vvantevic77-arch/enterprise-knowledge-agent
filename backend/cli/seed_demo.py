"""Provision a demo tenant with realistic payment-operations data.

Run it against the local SQLite database::

    make seed-demo

or directly::

    DATABASE_URL=sqlite+aiosqlite:///./agent.db PYTHONPATH=backend \\
        .venv/bin/python -m cli.seed_demo

Everything is written through the same service functions the HTTP API uses
(``seed_demo_knowledge``, ``start_run``, ``execute_ticket_action``,
``audit.record``), so the resulting rows satisfy the same invariants — the
intent and risk tiers are produced by the real classifier rather than being
hardcoded, and ticket statuses only move along legal transitions.

The script is idempotent: re-running reconciles the existing tenant instead of
duplicating it.
"""

from __future__ import annotations

# The app reads its database URL at import time, so the environment must be
# finalised before any ``app.*`` import. Same discipline as tests/conftest.py.
import os

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("EMBEDDING_PROVIDER", "deterministic")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./agent.db")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from app import audit  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Base, SessionLocal, engine, init_db  # noqa: E402
from app.models import (  # noqa: E402
    AgentRun,
    ApprovalRequest,
    Membership,
    Organization,
    Role,
    Ticket,
    User,
)
from app.permissions import Principal  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services.agent import start_run  # noqa: E402
from app.services.demo import seed_demo_knowledge  # noqa: E402
from app.services.tickets import execute_ticket_action  # noqa: E402
from sqlalchemy import func, inspect, select, text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

ORG_NAME = "Acme Payments"
DEMO_PASSWORD = "demo-password-2026"

#: (email, display name, role). Registration is the only path that mints an
#: owner and MemberCreate forbids the owner role outright, so the team is built
#: through the ORM rather than over HTTP.
DEMO_USERS: tuple[tuple[str, str, str], ...] = (
    ("ops@acme-payments.com", "Operations Owner", Role.OWNER.value),
    ("risk@acme-payments.com", "Risk Reviewer", Role.ADMIN.value),
    ("support@acme-payments.com", "Support Agent", Role.MEMBER.value),
    ("auditor@acme-payments.com", "Compliance Auditor", Role.VIEWER.value),
)

#: Prompts are deliberately distinct: start_run derives its idempotency key from
#: (org, user, prompt), and a collision collapses two runs into one approval.
#: Each also carries the phrasing classify_risk needs for its intended tier.
ANSWER_PROMPTS: tuple[str, ...] = (
    "退款多久到账？",
    "支付接口返回 429 应该怎么退避重试？",
    "webhook 签名校验失败怎么排查？",
)

HIGH_RISK_PROMPT = "退款超出 SLA，请创建一个高优先级工单跟进，并核查生产环境的结算批次。"
CRITICAL_RISK_PROMPT = "商户账户变更请求需要资金转账复核，请创建工单并冻结该账户的赔付操作。"

#: Tickets advanced through the lifecycle without an agent run, to populate
#: every status the Tickets table can render.
MANUAL_TICKETS: tuple[dict[str, Any], ...] = (
    {
        "title": "结算批次缺失排查",
        "description": "Merchant #4821 报告 9 月 settlement batch 未生成，需要与 Finance 核对。",
        "priority": "high",
        "transitions": [
            ("triage", {}),
            ("start", {}),
            ("wait_on_merchant", {"note": "等待商户提供原始交易明细。"}),
        ],
    },
    {
        "title": "重复扣款客户投诉",
        "description": "同一 idempotency key 出现两笔扣款，需要确认是否为 webhook 重放导致。",
        "priority": "high",
        "transitions": [
            ("triage", {}),
            ("start", {}),
            (
                "resolve",
                {"resolution": "确认是 webhook 重放，已按 idempotency key 回滚第二笔扣款。"},
            ),
        ],
    },
    {
        "title": "商户接入文档更新",
        "description": "补充 401 排查章节，把 Authorization header 示例改为 Bearer scheme。",
        "priority": "normal",
        "transitions": [("triage", {})],
    },
)


def fail(message: str) -> None:
    print(f"\n  ✗ {message}\n", file=sys.stderr)
    raise SystemExit(1)


def _read_schema(sync_connection: Any) -> dict[str, set[str]]:
    inspector = inspect(sync_connection)
    return {
        table: {column["name"] for column in inspector.get_columns(table)}
        for table in inspector.get_table_names()
    }


def schema_drift(sync_connection: Any) -> dict[str, list[str]]:
    """Columns the models declare that the existing tables do not have.

    ``init_db()`` only issues ``CREATE TABLE IF NOT EXISTS``, so a database
    written by an older revision keeps its old columns and every query against
    the new ones fails with an opaque OperationalError. Catching it here turns
    that into an actionable message.
    """
    present = _read_schema(sync_connection)
    drift: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        existing = present.get(table.name)
        if existing is None:
            continue  # Missing entirely — init_db() will create it correctly.
        missing = sorted({c.name for c in table.columns} - existing)
        if missing:
            drift[table.name] = missing
    return drift


def sqlite_path() -> Path | None:
    prefix = "sqlite+aiosqlite:///"
    if not settings.database_url.startswith(prefix):
        return None
    raw = settings.database_url[len(prefix) :]
    return Path(raw).expanduser().resolve()


def backup_database(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.{stamp}.bak")
    # Include committed WAL pages even when another process still has the DB open.
    with sqlite3.connect(path) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


async def reset_schema(path: Path) -> None:
    """Drop and recreate every table, after backing the file up."""
    backup = backup_database(path)
    print(f"  已备份     {backup.name}")
    async with engine.begin() as connection:
        for suffix in ("fts", "trigram", "cjk"):
            await connection.execute(text(f"DROP TABLE IF EXISTS document_chunks_{suffix}"))
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    print("  重建 schema 完成")


async def ensure_user(db: AsyncSession, email: str, display_name: str) -> User:
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        user = User(
            email=email,
            display_name=display_name,
            hashed_password=hash_password(DEMO_PASSWORD),
        )
        db.add(user)
        await db.flush()
        print(f"  创建用户   {email}")
    else:
        print(f"  已存在     {email}")
    return user


async def ensure_membership(db: AsyncSession, user: User, org_id: str, role: str) -> None:
    membership = (
        await db.execute(
            select(Membership).where(Membership.user_id == user.id, Membership.org_id == org_id)
        )
    ).scalar_one_or_none()
    if membership is None:
        db.add(Membership(user_id=user.id, org_id=org_id, role=role))
    else:
        membership.role = role
    await db.flush()


async def existing_run(db: AsyncSession, org_id: str, prompt: str) -> AgentRun | None:
    """Identity for idempotency: one seeded run per (org, prompt)."""
    return (
        (
            await db.execute(
                select(AgentRun).where(AgentRun.org_id == org_id, AgentRun.prompt == prompt)
            )
        )
        .scalars()
        .first()
    )


async def seed_runs(db: AsyncSession, org_id: str, requester: User, kb_ids: list[str]) -> None:
    """Historical runs, the two pending approvals, and the tickets they produced."""
    principal = Principal(
        user_id=requester.id,
        org_id=org_id,
        role=Role.OWNER.value,
        email=requester.email,
        display_name=requester.display_name,
    )

    # Answer-only runs. With no LLM key configured these take the deterministic
    # grounded-answer path, so this is offline and free.
    for prompt in ANSWER_PROMPTS:
        if await existing_run(db, org_id, prompt):
            print(f"  已存在 run  {prompt[:20]}…")
            continue
        result = await start_run(db, principal=principal, prompt=prompt, kb_ids=kb_ids)
        print(f"  Agent run  {result.run.status:<18} {prompt[:24]}…")

    # Two approvals left deliberately pending, at two different risk tiers.
    # Approving them here would execute the tool and create tickets as a side
    # effect, so the demo keeps them for the reviewer to act on in the UI.
    for prompt in (HIGH_RISK_PROMPT, CRITICAL_RISK_PROMPT):
        run = await existing_run(db, org_id, prompt)
        if run is not None:
            print(f"  已存在 run  {prompt[:20]}…")
            continue
        result = await start_run(db, principal=principal, prompt=prompt, kb_ids=kb_ids)
        approval = result.approval
        tier = approval.risk_tier if approval else "—"
        required = approval.required_approvals if approval else 0
        print(f"  待审批     {tier:<9} 需要 {required} 人 · {prompt[:20]}…")

    await db.commit()


async def seed_manual_tickets(db: AsyncSession, org_id: str, author: User) -> None:
    """Tickets walked through the lifecycle so every status renders."""
    for spec in MANUAL_TICKETS:
        existing = (
            await db.execute(
                select(Ticket).where(Ticket.org_id == org_id, Ticket.title == spec["title"])
            )
        ).scalar_one_or_none()
        if existing is not None:
            print(f"  已存在工单 {spec['title']}")
            continue

        ticket = Ticket(
            org_id=org_id,
            created_by=author.id,
            title=spec["title"],
            description=spec["description"],
            priority=spec["priority"],
            risk_tier="medium",
            status="open",
        )
        db.add(ticket)
        await db.flush()

        for action, kwargs in spec["transitions"]:
            await execute_ticket_action(
                db,
                org_id=org_id,
                actor_id=author.id,
                ticket_id=ticket.id,
                action=action,
                **kwargs,
            )
        print(f"  工单       {spec['title']} → {ticket.status}")
    await db.commit()


async def seed_audit_trail(db: AsyncSession, org_id: str, actor: User) -> None:
    """A few allow/deny rows so the audit view is not empty."""
    existing = await db.scalar(
        select(func.count(audit.AuditLog.id)).where(audit.AuditLog.org_id == org_id)
    )
    if existing:
        print(f"  审计日志   已有 {int(existing)} 条")
        return

    audit.record(
        db,
        action="organization.create",
        resource="organization",
        org_id=org_id,
        actor_id=actor.id,
        resource_id=org_id,
        reason="seeded demo tenant",
    )
    audit.record_denied(
        db,
        action="approval.vote",
        resource="approval",
        org_id=org_id,
        actor_id=actor.id,
        reason="requester cannot approve their own request",
    )
    audit.record(
        db,
        action="knowledge_base.read",
        resource="knowledge_base",
        org_id=org_id,
        actor_id=actor.id,
    )
    await db.commit()
    print("  审计日志   3 条")


async def summarise(db: AsyncSession, org: Organization) -> None:
    runs = await db.scalar(select(func.count(AgentRun.id)).where(AgentRun.org_id == org.id))
    pending = await db.scalar(
        select(func.count(ApprovalRequest.id)).where(
            ApprovalRequest.org_id == org.id,
            ApprovalRequest.status == "pending",
        )
    )
    tickets = await db.scalar(select(func.count(Ticket.id)).where(Ticket.org_id == org.id))
    print(f"\n  工作区 {org.name}  ({org.id})")
    print(f"  Agent runs {int(runs or 0)} · 待审批 {int(pending or 0)} · 工单 {int(tickets or 0)}")


async def run(reset: bool = False) -> None:
    if settings.app_env.lower() not in {"development", "dev", "local"}:
        fail(
            f"APP_ENV is {settings.app_env!r}; seed_demo only runs in a local "
            "development environment."
        )

    print("\n  Payment operations demo seed")
    print(f"  数据库 {settings.database_url}")
    if settings.llm_api_key:
        print("  ⚠️  检测到 LLM key：seed 会真实调用模型并产生费用。")

    path = sqlite_path()
    if path is not None and path.exists():
        async with engine.connect() as connection:
            drift = await connection.run_sync(schema_drift)
        if drift and reset:
            print("\n  检测到 schema 与当前模型不一致，正在重建：")
            for table, columns in sorted(drift.items()):
                print(f"    {table}: 缺少 {', '.join(columns)}")
            await engine.dispose()
            await reset_schema(path)
        elif drift:
            print()
            for table, columns in sorted(drift.items()):
                print(f"    {table}: 缺少 {', '.join(columns)}")
            fail(
                "现有数据库是旧版本 schema，缺少上面这些列。\n"
                "    init_db() 只创建缺失的表，不会修改已存在的表，所以直接跑会失败。\n\n"
                f"    重建（会先备份到 {path.name}.<时间戳>.bak）：\n"
                "        make seed-demo-reset\n"
                f"    或手动换一个空库：\n"
                f"        rm {path.name} && make seed-demo"
            )

    print()
    await init_db()

    async with SessionLocal() as db:
        org = (
            await db.execute(select(Organization).where(Organization.name == ORG_NAME))
        ).scalar_one_or_none()
        if org is None:
            org = Organization(name=ORG_NAME)
            db.add(org)
            await db.flush()
            print(f"  创建组织   {ORG_NAME}")
        else:
            print(f"  已存在组织 {ORG_NAME}")
        await db.commit()

        # Team. The owner is created first so runs have a requester.
        users: dict[str, User] = {}
        for email, display_name, role in DEMO_USERS:
            user = await ensure_user(db, email, display_name)
            await ensure_membership(db, user, org.id, role)
            users[role] = user
        await db.commit()

        # The runbooks every answer is grounded in.
        knowledge_base, created = await seed_demo_knowledge(db, org.id)
        print(f"  知识库     {knowledge_base.name}{' (新建)' if created else ' (已存在)'}")

        await seed_runs(db, org.id, users[Role.OWNER.value], [knowledge_base.id])
        await seed_manual_tickets(db, org.id, users[Role.MEMBER.value])
        await seed_audit_trail(db, org.id, users[Role.OWNER.value])
        await summarise(db, org)

    print("\n  登录账号")
    for email, _display_name, role in DEMO_USERS:
        print(f"    {email:<32} {role}")
    print(f"  密码       {DEMO_PASSWORD}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Dump the existing database and rebuild the schema when it is stale.",
    )
    args = parser.parse_args()
    asyncio.run(run(reset=args.reset))


if __name__ == "__main__":
    main()
