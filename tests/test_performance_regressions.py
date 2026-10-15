"""Resource use, indexed retrieval, and streaming under slow dependencies."""

import asyncio
import json
import sqlite3
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest
from app.config import settings
from app.db import SessionLocal, engine
from app.models import AgentRun, Document, DocumentChunk, IdempotencyRecord, UsageEvent
from app.permissions import Principal
from app.routes.agent import create_run_stream
from app.schemas import AgentRunCreate
from app.services import agent, events, rag
from sqlalchemy import func, select

from .conftest import Actor, seed_corpus


@pytest.mark.parametrize("use_key", [False, True])
async def test_twenty_slow_models_release_connections(member: Actor, monkeypatch, use_key):
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def slow_answer(*_args):
        nonlocal count
        count += 1
        if count == 20:
            entered.set()
        await release.wait()
        return "mock answer"

    monkeypatch.setattr(agent, "_answer", slow_answer)
    tasks = [
        asyncio.create_task(
            member.client.post(
                "/v1/agent/runs",
                headers=member.headers,
                json={
                    "prompt": "退款多久到账？",
                    **({"idempotency_key": f"slow-{i}"} if use_key else {}),
                },
            )
        )
        for i in range(20)
    ]
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert engine.pool.checkedout() == 0
    finally:
        release.set()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(
        not isinstance(response, BaseException) and response.status_code == 200
        for response in responses
    ), [str(response) for response in responses if isinstance(response, BaseException)]


async def test_embedding_does_not_block_event_loop(monkeypatch):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def slow(_texts):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=2)
        return [[1.0, 0.0]]

    monkeypatch.setattr(rag, "embed_texts", slow)
    task = asyncio.create_task(rag.aembed_texts(["sample"]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
    assert await task == [[1.0, 0.0]]


async def test_broker_publication_runs_off_event_loop(monkeypatch):
    from app.routes import knowledge

    loop_thread = threading.get_ident()
    called = []

    def publish(*, args, retry):
        assert threading.get_ident() != loop_thread
        called.append((args, retry))

    monkeypatch.setattr(knowledge.ingest_document_task, "apply_async", publish)
    await knowledge._dispatch_ingestion("doc-id", "runbook.txt", b"content")
    assert called[0][0][:2] == ("doc-id", "runbook.txt")
    assert called[0][1] is False


async def test_chinese_short_query_never_loads_vectors(db, owner: Actor, monkeypatch):
    kb_id = await seed_corpus(owner.client, owner)

    async def reject(*_args):
        raise AssertionError("indexed Chinese search must not scan vectors")

    monkeypatch.setattr(rag, "aembed_texts", reject)
    rows = await agent.search_knowledge(db, owner.org_id, "退款", [kb_id])
    assert rows and rows[0]["filename"] == "refund-and-settlement-runbook.md"
    assert await agent.search_knowledge(db, owner.org_id, "霸王", [kb_id]) == []


async def test_lexical_match_survives_dense_candidate_gate(db, owner: Actor, monkeypatch):
    kb_id = await seed_corpus(owner.client, owner)
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "retrieval_candidate_k", 5)
    docs = [
        Document(
            org_id=owner.org_id,
            kb_id=kb_id,
            filename=f"fusion-{i}.txt",
            content_hash=str(i),
            status="indexed",
        )
        for i in range(26)
    ]
    db.add_all(docs)
    await db.flush()
    for i, doc in enumerate(docs):
        db.add(
            DocumentChunk(
                document_id=doc.id,
                ordinal=0,
                content="zanzibar" if i == 25 else "unrelated candidate",
                embedding=([-1.0] if i == 25 else [1.0])
                + [0.0] * (settings.embedding_dimensions - 1),
                metadata_json={"embedding_identity": rag.embedding_identity()},
            )
        )
    await db.commit()

    async def embed(_texts):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1)]

    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(rag, "aembed_texts", embed)
    rows = await agent.search_knowledge(db, owner.org_id, "zanzibar", [kb_id], top_k=1)
    assert [row["filename"] for row in rows] == ["fusion-25.txt"]


async def test_document_pagination_is_stable(owner: Actor):
    first = await owner.client.get("/v1/documents?limit=3", headers=owner.headers)
    second = await owner.client.get("/v1/documents?limit=3&skip=3", headers=owner.headers)
    assert first.status_code == second.status_code == 200
    assert len(first.json()) == len(second.json()) == 3
    assert not {row["id"] for row in first.json()} & {row["id"] for row in second.json()}
    assert (
        await owner.client.get("/v1/documents?limit=0", headers=owner.headers)
    ).status_code == 422


async def test_usage_aggregates_only_current_tenant(db, owner: Actor, second_org_owner: Actor):
    assert (await owner.client.get("/v1/usage", headers=owner.headers)).json()["input_tokens"] == 0
    for actor, incoming, outgoing, cost in (
        (owner, 7, 11, 0.1),
        (owner, 13, 17, 0.2),
        (second_org_owner, 999, 999, 9),
    ):
        run = AgentRun(org_id=actor.org_id, user_id=actor.id, prompt="fixture", status="succeeded")
        db.add(run)
        await db.flush()
        db.add(
            UsageEvent(
                org_id=actor.org_id,
                run_id=run.id,
                event_type="fixture",
                input_tokens=incoming,
                output_tokens=outgoing,
                cost_usd=cost,
            )
        )
    await db.commit()
    response = await owner.client.get("/v1/usage", headers=owner.headers)
    assert response.json() == {
        "runs": 2,
        "input_tokens": 20,
        "output_tokens": 28,
        "estimated_cost_usd": 0.3,
    }


def test_wal_backup_includes_committed_log_pages(tmp_path):
    from cli.seed_demo import backup_database

    database = tmp_path / "source.db"
    with sqlite3.connect(database) as source:
        source.execute("PRAGMA journal_mode=WAL")
        source.execute("PRAGMA wal_autocheckpoint=0")
        source.execute("CREATE TABLE sample (value TEXT)")
        source.execute("INSERT INTO sample VALUES ('committed in WAL')")
        source.commit()
        assert database.with_name(database.name + "-wal").exists()
        with sqlite3.connect(backup_database(database)) as copied:
            assert copied.execute("SELECT value FROM sample").fetchone() == ("committed in WAL",)


async def test_events_have_absolute_cursors_and_expire(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(events, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(events, "_redis_retry_at", float("inf"))
    monkeypatch.setattr(events, "_MAX_BUFFERED_EVENTS", 2)
    monkeypatch.setattr(events, "_MAX_BUFFERED_RUNS", 2)
    monkeypatch.setattr(events, "_EVENT_TTL_SECONDS", 10)
    monkeypatch.setattr(events, "_streams", OrderedDict())
    for i in range(3):
        await events.publish_event("first", {"type": str(i)})
    assert [e["type"] async for e in events.read_events("first")] == ["1", "2"]
    assert [e["type"] async for e in events.read_events("first", 2)] == ["2"]
    await events.publish_event("second", {"type": "new"})
    await events.publish_event("third", {"type": "new"})
    assert not [e async for e in events.read_events("first")]
    now[0] = 11
    assert not [e async for e in events.read_events("third")]


async def test_redis_replay_without_local_buffer(monkeypatch):
    class Redis:
        async def xrange(self, key, min, count):
            assert count > 0
            assert key == "agent:run:remote"
            assert min in {"-", "(123-0"}
            return (
                [("123-0", {"payload": json.dumps({"type": "run.started"})})] if min == "-" else []
            )

    monkeypatch.setattr(events, "_streams", OrderedDict())
    monkeypatch.setattr(events, "_redis_retry_at", 0)
    monkeypatch.setattr(events, "_get_redis_client", lambda: Redis())
    assert [e async for e in events.read_events("remote")] == [
        {"type": "run.started", "_cursor": "123-0"}
    ]
    assert [e async for e in events.read_events("remote", "123-0")] == []


async def test_sse_tokens_arrive_before_completion(member: Actor, monkeypatch):
    from app.main import app

    first_token, release = asyncio.Event(), asyncio.Event()
    parts = []

    class Model:
        async def astream(self, _messages):
            yield SimpleNamespace(content="first")
            await release.wait()
            yield SimpleNamespace(content=[{"type": "text", "text": "second"}])

    monkeypatch.setattr(settings, "openai_api_key", "mock-only")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(agent, "_answer_model", lambda: Model())
    body = json.dumps({"prompt": "退款多久到账？", "idempotency_key": "stream-test"}).encode()
    headers = [(key.lower().encode(), value.encode()) for key, value in member.headers.items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/agent/runs/stream",
        "raw_path": b"/v1/agent/runs/stream",
        "query_string": b"",
        "root_path": "",
        "server": ("test", 80),
        "client": ("127.0.0.1", 1),
        "headers": headers + [(b"content-type", b"application/json")],
    }
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        if message["type"] == "http.response.body":
            data = message.get("body", b"")
            parts.append(data)
            if b'"type": "answer.delta"' in data:
                first_token.set()

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(first_token.wait(), timeout=3)
        assert not task.done()
        assert engine.pool.checkedout() == 0
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)
    frames = [
        json.loads(line[6:])
        for line in b"".join(parts).decode().splitlines()
        if line.startswith("data: ")
    ]
    result = next(frame["run"] for frame in frames if frame["type"] == "run.result")
    assert result["answer"] == "firstsecond" and result["citations"]
    replay = await member.client.post(
        "/v1/agent/runs/stream",
        headers=member.headers,
        json={"prompt": "退款多久到账？", "idempotency_key": "stream-test"},
    )
    assert '"replayed": true' in replay.text and "answer.delta" not in replay.text


async def test_sse_disconnect_releases_pending_key(db, member: Actor, monkeypatch):
    class Model:
        async def astream(self, _messages):
            yield SimpleNamespace(content="partial")
            await asyncio.Event().wait()

    monkeypatch.setattr(settings, "openai_api_key", "mock-only")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(agent, "_answer_model", lambda: Model())
    response = await create_run_stream(
        AgentRunCreate(prompt="退款多久到账？", idempotency_key="disconnect"),
        db,
        Principal(user_id=member.id, org_id=member.org_id, role=member.role),
    )
    iterator = response.body_iterator
    while "answer.delta" not in await asyncio.wait_for(anext(iterator), timeout=3):
        pass
    await iterator.aclose()
    async with SessionLocal() as session:
        assert await session.scalar(select(func.count(IdempotencyRecord.id))) == 0
        assert await session.scalar(select(func.count(AgentRun.id))) == 0


async def test_sse_authorization(client, viewer: Actor):
    assert (await client.post("/v1/agent/runs/stream", json={"prompt": "hello"})).status_code == 401
    assert (
        await viewer.client.post(
            "/v1/agent/runs/stream", headers=viewer.headers, json={"prompt": "hello"}
        )
    ).status_code == 403


async def test_redis_outage_keeps_delivering_local_events(monkeypatch):
    from collections import deque

    monkeypatch.setattr(events, "_redis_retry_at", float("inf"))
    monkeypatch.setattr(
        events,
        "_streams",
        OrderedDict(
            [
                (
                    "run",
                    (
                        events.time.monotonic(),
                        deque(
                            [
                                (1, {"type": "run.started"}, "123-0"),
                                (2, {"type": "run.completed"}, None),
                            ]
                        ),
                    ),
                )
            ]
        ),
    )
    assert [e["type"] async for e in events.read_events("run", "123-0")] == ["run.completed"]


async def test_model_stream_failure_resets_partial_answer(member: Actor, monkeypatch):
    class Model:
        async def astream(self, _messages):
            yield SimpleNamespace(content="unverified partial")
            raise RuntimeError("mock provider failure")

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "mock-only")
    monkeypatch.setattr(agent, "_answer_model", lambda: Model())
    response = await member.client.post(
        "/v1/agent/runs/stream", headers=member.headers, json={"prompt": "退款多久到账？"}
    )
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert any(frame["type"] == "answer.reset" for frame in frames)
    result = next(frame["run"] for frame in frames if frame["type"] == "run.result")
    assert "unverified partial" not in result["answer"]
    assert "3-5 个工作日" in result["answer"]


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
async def test_actual_asgi_disconnect_cleans_up_run(db, member: Actor, monkeypatch, spec_version):
    from starlette.requests import ClientDisconnect

    token_sent = asyncio.Event()

    class Model:
        async def astream(self, _messages):
            yield SimpleNamespace(content="partial")
            await asyncio.Event().wait()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "mock-only")
    monkeypatch.setattr(agent, "_answer_model", lambda: Model())
    response = await create_run_stream(
        AgentRunCreate(prompt="退款多久到账？", idempotency_key="asgi-disconnect"),
        db,
        Principal(user_id=member.id, org_id=member.org_id, role=member.role),
    )

    async def receive():
        await token_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if b'"type": "answer.delta"' in message.get("body", b""):
            token_sent.set()
            if spec_version == "2.4":
                raise OSError("client socket closed")

    scope = {"type": "http", "asgi": {"spec_version": spec_version}}
    if spec_version == "2.4":
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(response(scope, receive, send), timeout=5)
    else:
        await asyncio.wait_for(response(scope, receive, send), timeout=5)
    async with SessionLocal() as session:
        assert await session.scalar(select(func.count(IdempotencyRecord.id))) == 0
        assert await session.scalar(select(func.count(AgentRun.id))) == 0
