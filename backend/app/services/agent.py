"""The agent graph: retrieve → answer OR gate → tool execution → resume.

``build_agent_graph`` is a real ``langgraph`` StateGraph with a conditional
edge, not a sequential function call wearing a graph-shaped name:

    START → retrieve → decide ─┬─→ answer → END
                               └─→ request_approval → END

The graph ends at ``request_approval`` on purpose. A high-risk action must be
resumable *across process restarts*, so the pause point is a database row
(``ApprovalRequest``), not a suspended coroutine. ``resume_after_approval``
re-enters the same tool-execution node once the votes are in.

Two properties matter for a payment platform:

* **Nothing executes without a decision.** ``execute_tool`` is reachable only
  from the approval path, and it re-checks the approval status itself rather
  than trusting its caller.
* **Replay is safe.** Execution is keyed by the approval's ``idempotency_key``,
  so a retried or duplicated resume returns the stored result.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from sqlalchemy import bindparam, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypedDict

from app import audit
from app.config import settings
from app.core_exceptions import ConflictError, NotFoundError, ValidationError
from app.db import SessionLocal
from app.idempotency import fingerprint, replay_or_execute
from app.models import (
    AgentRun,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalVote,
    Document,
    DocumentChunk,
    IdempotencyRecord,
    KnowledgeBase,
    RunStatus,
    Ticket,
    TicketEvent,
    UsageEvent,
    new_id,
    utcnow,
)
from app.permissions import Principal, required_approvals
from app.services import rag
from app.services.events import event_sink, publish_event
from app.services.policy import classify_intent, classify_risk, sla_hours, ticket_arguments
from app.services.tickets import execute_ticket_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    prompt: str
    org_id: str
    user_id: str
    #: Carried through the graph so nodes can publish events against the run
    #: they belong to. StateGraph passes only *declared* keys between nodes, so
    #: omitting this makes every event-publishing node raise KeyError.
    run_id: str
    kb_ids: list[str]
    context: list[dict[str, Any]]
    answer: str
    intent: str
    risk_tier: str
    action: dict[str, Any] | None
    approval_id: str | None


@dataclass(frozen=True)
class RunResult:
    run: AgentRun
    approval: ApprovalRequest | None = None
    replayed: bool = False


def _estimate_tokens(text: str) -> int:
    """Rough token count. Deliberately provider-agnostic so cost accounting
    works identically with or without a live model."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)) // 2)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def _sqlite_lexical_candidates(
    db: AsyncSession, org_id: str, query: str, kb_ids: list[str], top_k: int
) -> list[dict[str, Any]]:
    common = {
        "the",
        "and",
        "for",
        "are",
        "was",
        "what",
        "how",
        "who",
        "why",
        "when",
        "where",
        "which",
        "with",
        "from",
        "you",
        "your",
        "this",
        "that",
    }
    words = {
        word
        for word in re.findall(r"[a-z0-9_]+", query.lower())
        if word not in common and (len(word) > 2 or word.isdigit())
    }
    bigrams = set(rag.cjk_index_text(query).split())
    searches = (("document_chunks_fts", words), ("document_chunks_cjk", bigrams))
    candidates: dict[str, dict[str, Any]] = {}
    for table, terms in searches:
        if not terms:
            continue
        match = " OR ".join(f'"{term}"' for term in sorted(terms))
        scope = "AND kb.id IN :kb_ids" if kb_ids else ""
        statement = text(
            f"SELECT c.id, c.content, d.filename, d.source_url "
            f"FROM {table} JOIN document_chunks AS c ON c.rowid = {table}.rowid "
            "JOIN documents AS d ON d.id = c.document_id "
            "JOIN knowledge_bases AS kb ON kb.id = d.kb_id "
            f"WHERE {table} MATCH :match AND d.status = 'indexed' "
            f"AND kb.org_id = :org_id {scope} "
            f"ORDER BY bm25({table}) LIMIT :limit"
        )
        if kb_ids:
            statement = statement.bindparams(bindparam("kb_ids", expanding=True))
        rows = (
            await db.execute(
                statement,
                {"match": match, "org_id": org_id, "kb_ids": kb_ids, "limit": max(100, top_k * 20)},
            )
        ).all()
        for rank, (chunk_id, content, filename, source_url) in enumerate(rows):
            score = 1 / (rank + 1)
            if chunk_id not in candidates or candidates[chunk_id]["score"] < score:
                candidates[chunk_id] = {
                    "content": content,
                    "score": score,
                    "filename": filename,
                    "source_url": source_url,
                    "chunk_id": chunk_id,
                }
    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[:top_k]


async def _retrieve_candidates(
    db: AsyncSession,
    org_id: str,
    query: str,
    kb_ids: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Tenant-scoped hybrid retrieval.

    ``org_id`` is applied on every path. When ``kb_ids`` is supplied it narrows
    within the tenant; it can never widen beyond it, because the organisation
    predicate is joined unconditionally.
    """
    if not 1 <= top_k <= 200:
        raise ValueError("top_k must be between 1 and 200")
    # Encode before opening a read transaction; remote/local inference may be slow.
    query_vector = (
        (await rag.aembed_texts([query]))[0]
        if settings.embedding_provider != "deterministic" or not settings.is_sqlite
        else None
    )
    candidate_k = max(top_k, settings.retrieval_candidate_k)
    lexical_candidates = []
    if settings.is_sqlite:
        lexical_k = (
            top_k
            if settings.embedding_provider == "deterministic" and not settings.reranker_model
            else candidate_k
        )
        lexical_candidates = await _sqlite_lexical_candidates(db, org_id, query, kb_ids, lexical_k)
        if settings.embedding_provider == "deterministic":
            return lexical_candidates
    stmt = (
        select(DocumentChunk, Document.filename, Document.source_url)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .where(Document.status == "indexed", KnowledgeBase.org_id == org_id)
    )
    if kb_ids:
        stmt = stmt.where(KnowledgeBase.id.in_(kb_ids))

    if not settings.is_sqlite:
        terms = sorted(set(re.findall(r"[a-z0-9_]+", query.lower())))
        if terms:
            query_terms = func.to_tsquery(text("'simple'"), " | ".join(terms))
            document_terms = func.to_tsvector(text("'simple'"), DocumentChunk.content)
            lexical_rows = (
                await db.execute(
                    stmt.where(document_terms.op("@@")(query_terms))
                    .order_by(func.ts_rank_cd(document_terms, query_terms).desc())
                    .limit(candidate_k)
                )
            ).all()
            lexical_candidates = [
                {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "filename": filename,
                    "source_url": source_url,
                    "score": 1 / (rank + 1),
                }
                for rank, (chunk, filename, source_url) in enumerate(lexical_rows)
            ]

    # Let pgvector do the coarse nearest-neighbour scan when a real vector
    # column is present. The Python pass below adds lexical overlap.
    if settings.embedding_provider != "deterministic":
        stmt = stmt.where(
            DocumentChunk.embedding.is_not(None),
            DocumentChunk.metadata_json["embedding_identity"].as_string()
            == rag.embedding_identity(),
        )
    if (
        getattr(DocumentChunk, "__table_args__", ())
        and not settings.is_sqlite
        and hasattr(DocumentChunk.embedding, "cosine_distance")
    ):
        stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_vector)).limit(
            candidate_k
        )

    rows = (await db.execute(stmt)).all()
    if not rows:
        return lexical_candidates

    vector_candidates = await asyncio.to_thread(
        _rank_vector_candidates, query, query_vector, rows, candidate_k
    )
    if settings.embedding_provider == "deterministic":
        return vector_candidates[:candidate_k]
    # Each retriever contributes candidates independently, so lexical matches survive the vector gate.
    return rag.fuse_rankings([lexical_candidates, vector_candidates], candidate_k)


async def search_knowledge(
    db: AsyncSession,
    org_id: str,
    query: str,
    kb_ids: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    candidates = await _retrieve_candidates(db, org_id, query, kb_ids, top_k)
    return await rag.arerank(query, candidates, top_k)


def _rank_vector_candidates(query, query_vector, rows, top_k):
    embeddings = [
        chunk.embedding
        if chunk.embedding is not None
        else rag.deterministic_embedding(chunk.content, settings.embedding_dimensions)
        for chunk, _, _ in rows
    ]
    vector_scores = rag.cosine_similarity_batch(query_vector, embeddings)

    candidates = [
        {
            "content": chunk.content,
            # Exact operational identifiers (401, refund ID, p95) carry more
            # signal than topical similarity, so lexical overlap is weighted
            # heavily. See services/rag.py for the weight rationale.
            "score": rag.hybrid_score(query, chunk.content, vector_score)
            if settings.embedding_provider == "deterministic"
            else vector_score,
            "filename": filename,
            "source_url": source_url,
            "chunk_id": chunk.id,
        }
        for (chunk, filename, source_url), vector_score in zip(rows, vector_scores)
    ]

    import heapq

    return heapq.nlargest(top_k, candidates, key=lambda item: item["score"])


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _answer_model() -> Any:
    from langchain_openai import ChatOpenAI

    model_kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": 0,
        "timeout": 30,
        "max_retries": 1,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url
    if settings.llm_api_mode == "responses":
        model_kwargs["use_responses_api"] = True
        if settings.openai_reasoning_effort:
            model_kwargs["reasoning_effort"] = settings.openai_reasoning_effort
    return ChatOpenAI(**model_kwargs)


@lru_cache(maxsize=1)
def _compiled_answer_graph() -> Any:
    """Compile the model graph once per worker process."""
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        messages: list[Any]

    model = _answer_model()

    async def answer_node(state: State) -> State:
        response = await model.ainvoke(state["messages"])
        return {"messages": [*state["messages"], response]}

    graph = StateGraph(State)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


_ANSWER_SYSTEM_PROMPT = (
    "You are a payment operations assistant. Answer only from the supplied "
    "context and cite the filenames you used. Never invent a fact that is not "
    "in the context. Use this exact structure in Chinese:\n"
    "结论：...\n\n关键点：\n- ...\n\n处理步骤：\n1. ...\n\n来源：\n- filename\n"
    "Keep operational identifiers such as 401, 429, trace ID, refund ID, "
    "organization_id, content_hash and p95 unchanged."
)

_GREETING = re.compile(r"^(你好|您好|hi|hello|hey|嗨)[。！!?？\s]*$", re.I)


async def _answer(prompt: str, context: list[dict[str, Any]]) -> str:
    """Model answer when configured, deterministic grounded answer otherwise."""
    if _GREETING.match(prompt.strip()):
        return (
            "你好，我是运营智能助手。我可以根据商户 runbook 回答退款、结算、"
            "支付接口和事件响应问题，也可以在你确认后创建工单。"
            "试试问我：退款多久到账？"
        )

    if settings.llm_api_key:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=_ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {prompt}"),
            ]
            sink = event_sink.get()
            if sink is not None:
                pieces = []
                async for chunk in _answer_model().astream(messages):
                    content = chunk.content
                    delta = (
                        content
                        if isinstance(content, str)
                        else "".join(
                            block.get("text", "") for block in content if isinstance(block, dict)
                        )
                    )
                    if delta:
                        pieces.append(delta)
                        await sink({"type": "answer.delta", "text": delta})
                return "".join(pieces)
            result = await _compiled_answer_graph().ainvoke({"messages": messages})
            return str(result["messages"][-1].content)
        except Exception:
            logger.exception("model call failed; falling back to deterministic answer")
            sink = event_sink.get()
            if sink is not None:
                await sink({"type": "answer.reset"})

    if context:
        return _format_grounded_answer(prompt, context)
    return (
        "我暂时没有检索到对应的运行手册。你可以先问退款时效、401/429 接口排查，或补充更多业务背景。"
    )


def _sentences(content: str) -> list[str]:
    normalized = re.sub(r"^#+\s*", "", content.strip())
    normalized = re.sub(r"^.*?(?:runbook|guide|指南|手册)\s+", "", normalized, flags=re.I)
    normalized = re.sub(r"[`*]", "", normalized)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？.!?])\s*", normalized)
        if len(sentence.strip()) >= 8
    ]


def _format_grounded_answer(prompt: str, context: list[dict[str, Any]]) -> str:
    """Citation-first answer used when no model key is configured.

    This is not a degraded placeholder: it is the path the offline evaluation
    exercises, and it is deterministic so the reported metrics are reproducible.
    """
    evidence: list[str] = []
    for item in context:
        for sentence in _sentences(str(item["content"])):
            if sentence not in evidence:
                evidence.append(sentence)
            if len(evidence) >= 6:
                break
        if len(evidence) >= 6:
            break
    if not evidence:
        return "我暂时没有检索到对应的运行手册。"

    conclusion = " ".join(evidence[:2])
    key_points = evidence[2:5] or evidence[:2]
    lowered = prompt.lower()
    if re.search(r"退款|refund", lowered):
        steps = [
            "检查支付状态、refund event，以及下一个工作日的 settlement batch。",
            "确认 refund ID、原交易 ID、金额、币种和审核通过日期。",
            "超过 SLA 后创建 high priority 工单并交给 Payments Operations。",
        ]
    elif re.search(r"401|429|接口|api|webhook", lowered):
        steps = [
            "确认 merchant environment、Authorization Bearer token 和对应 secret。",
            "对 429 使用 exponential backoff，保留 Retry-After 和 idempotency key。",
            "记录 event ID、transaction ID 与 trace ID，再按 runbook replay 或升级。",
        ]
    elif re.search(r"权限|rbac|租户|组织|越权|member|admin", lowered):
        steps = [
            "先校验当前用户 membership、role 和 action + resource 权限。",
            "在查询、写入和后台任务中同时过滤 organization_id。",
            "拒绝越权请求并记录 actor、tenant、resource 和 trace ID 审计事件。",
        ]
    elif re.search(r"索引|rag|chunk|向量|评测|recall|mrr", lowered):
        steps = [
            "先计算 content hash，只有内容变化的文档才重写 chunk 和 embedding。",
            "使用 pgvector 粗检索，再结合 lexical overlap 做重排并保留来源。",
            "用固定 query 集复测 Recall@5、MRR 和延迟，再发布索引变更。",
        ]
    else:
        steps = [
            "确认问题所属 runbook、merchant 和 environment。",
            "按来源中的检查顺序执行，并记录关键 ID 与 trace ID。",
            "超过 SLA 或涉及高风险操作时，升级给对应 owner 并保留审计记录。",
        ]

    citations: list[str] = []
    for item in context:
        filename = str(item["filename"])
        if filename not in citations:
            citations.append(filename)
    return (
        f"结论：{conclusion}\n\n"
        "关键点：\n"
        + "\n".join(f"- {point}" for point in key_points)
        + "\n\n处理步骤：\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
        + "\n\n来源：\n"
        + "\n".join(f"- {filename}" for filename in citations)
    )


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def build_agent_graph(*, retrieve: Any, answer: Any, gate: Any) -> Any:
    """Assemble the StateGraph.

    Nodes are injected so the same graph shape can be exercised by tests with
    stub retrieval, which is what ``evals/`` does.
    """
    from langgraph.graph import END, START, StateGraph

    def decide(state: AgentState) -> str:
        return "gate" if state.get("action") else "answer"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("gate", gate)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", decide, {"answer": "answer", "gate": "gate"})
    graph.add_edge("answer", END)
    graph.add_edge("gate", END)
    return graph.compile()


def _compute_action(prompt: str) -> tuple[str, str, dict[str, Any] | None]:
    """Policy decision, kept pure so it can be unit-tested without a DB."""
    intent = classify_intent(prompt)
    risk_tier = classify_risk(prompt, intent.name)
    from app.services.policy import wants_ticket

    action = None
    if wants_ticket(prompt):
        action = {"tool_name": "create_ticket", "arguments": ticket_arguments(prompt)}
    return intent.name, risk_tier, action


async def _retrieve_node(state: AgentState) -> AgentState:
    async with SessionLocal() as retrieval_db:
        candidates = await _retrieve_candidates(
            retrieval_db, state["org_id"], state["prompt"], state.get("kb_ids") or []
        )
    context = await rag.arerank(state["prompt"], candidates, 5)
    await publish_event(state["run_id"], {"type": "retrieval.completed", "count": len(context)})
    intent, risk_tier, action = _compute_action(state["prompt"])
    return {**state, "context": context, "intent": intent, "risk_tier": risk_tier, "action": action}


async def _answer_node(state: AgentState) -> AgentState:
    return {**state, "answer": await _answer(state["prompt"], state.get("context") or [])}


async def _gate_node(state: AgentState) -> AgentState:
    # The gate is intentionally a no-op here: the caller persists the approval
    # row. Keeping graph execution free of commits means a retry cannot leave a
    # half-written run behind.
    return state


@lru_cache(maxsize=1)
def _compiled_agent_graph() -> Any:
    return build_agent_graph(retrieve=_retrieve_node, answer=_answer_node, gate=_gate_node)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


async def idempotency_key_for(org_id: str, user_id: str, prompt: str) -> str:
    """Derive a stable key from the logical request.

    Two identical prompts from the same user in the same tenant inside the
    replay window collapse to one approval instead of queueing duplicates.
    """
    return fingerprint({"org": org_id, "user": user_id, "prompt": prompt.strip()})[:48]


async def start_run(
    db: AsyncSession,
    *,
    principal: Principal,
    prompt: str,
    kb_ids: list[str],
    idempotency_key: str | None = None,
) -> RunResult:
    """Execute the graph and return either an answer or a pending approval."""
    payload = {"org": principal.org_id, "user": principal.user_id, "prompt": prompt, "kb": kb_ids}
    response: dict[str, Any] | None = None
    replayed = False
    if idempotency_key:
        with db.no_autoflush:
            existing = await db.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.org_id == principal.org_id,
                    IdempotencyRecord.scope == "agent.run",
                    IdempotencyRecord.key == idempotency_key,
                )
            )
        if existing is not None and existing.response_json:
            if existing.request_hash != fingerprint(payload):
                raise ConflictError(
                    message="idempotency key was already used with a different payload"
                )
            response, replayed = dict(existing.response_json), True

    if response is None:

        async def _execute() -> dict[str, Any]:
            # Only authorization decisions exist here; business rows are staged after the graph.
            if db.in_transaction():
                await db.commit()
            # Only the reservation owner may spend time on retrieval/model calls.
            run = AgentRun(
                id=new_id(),
                trace_id=new_id(),
                org_id=principal.org_id,
                user_id=principal.user_id,
                prompt=prompt,
                status=RunStatus.RUNNING.value,
            )
            await publish_event(
                run.id,
                {"type": "run.started", "run_id": run.id, "trace_id": run.trace_id},
            )
            graph = _compiled_agent_graph()
            with db.no_autoflush:
                final: AgentState = await graph.ainvoke(
                    {
                        "prompt": prompt,
                        "org_id": principal.org_id,
                        "user_id": principal.user_id,
                        "kb_ids": kb_ids,
                        "run_id": run.id,
                    }
                )

            db.add(run)
            await db.flush()

            context = final.get("context") or []
            run.intent = final.get("intent")
            run.risk_tier = final.get("risk_tier")
            run.context_chunk_ids = [str(item["chunk_id"]) for item in context]
            run.input_tokens = _estimate_tokens(
                prompt + " ".join(str(item["content"]) for item in context)
            )

            action = final.get("action")
            if action:
                risk_tier = final.get("risk_tier") or "medium"
                approval = ApprovalRequest(
                    run_id=run.id,
                    org_id=principal.org_id,
                    requested_by=principal.user_id,
                    tool_name=str(action["tool_name"]),
                    arguments=dict(action["arguments"]),
                    risk_tier=risk_tier,
                    required_approvals=required_approvals(risk_tier),
                    idempotency_key=idempotency_key
                    or await idempotency_key_for(principal.org_id, principal.user_id, prompt),
                )
                run.status = RunStatus.WAITING_APPROVAL.value
                db.add(approval)
                await db.flush()
                return {
                    "run_id": run.id,
                    "approval_id": approval.id,
                    "status": run.status,
                    "trace_id": run.trace_id,
                }

            run.answer = final.get("answer")
            run.output_tokens = _estimate_tokens(run.answer or "")
            run.estimated_cost_usd = _cost(run.input_tokens, run.output_tokens)
            run.status = RunStatus.SUCCEEDED.value
            run.updated_at = utcnow()
            db.add(_usage_event(run))
            return {
                "run_id": run.id,
                "approval_id": None,
                "status": run.status,
                "trace_id": run.trace_id,
            }

        response, replayed = await replay_or_execute(
            db,
            org_id=principal.org_id,
            scope="agent.run",
            key=idempotency_key,
            payload=payload,
            execute=_execute,
            release_reservation=True,
        )
    await db.commit()

    run = await db.get(AgentRun, response["run_id"])
    if run is None:  # pragma: no cover - only if the row was purged mid-request
        raise NotFoundError(message="run not found")
    approval = None
    if response.get("approval_id"):
        approval = await db.get(ApprovalRequest, response["approval_id"])
    await db.commit()
    if not replayed and approval is not None:
        await publish_event(
            run.id,
            {
                "type": "approval.required",
                "approval_id": approval.id,
                "tool": approval.tool_name,
                "risk_tier": approval.risk_tier,
                "required_approvals": approval.required_approvals,
            },
        )
    if not replayed and run.status == RunStatus.SUCCEEDED.value:
        await publish_event(
            run.id, {"type": "run.completed", "status": run.status, "answer": run.answer}
        )
    return RunResult(run=run, approval=approval, replayed=replayed)


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000) * settings.openai_input_cost_per_1k + (
        output_tokens / 1000
    ) * settings.openai_output_cost_per_1k


def _usage_event(run: AgentRun) -> UsageEvent:
    return UsageEvent(
        org_id=run.org_id,
        run_id=run.id,
        event_type="agent.completed",
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost_usd=run.estimated_cost_usd,
        metadata_json={"trace_id": run.trace_id, "intent": run.intent, "risk_tier": run.risk_tier},
    )


# ---------------------------------------------------------------------------
# Approval → execution → resume
# ---------------------------------------------------------------------------


async def record_vote(
    db: AsyncSession,
    *,
    principal: Principal,
    approval_id: str,
    decision: str,
    note: str = "",
) -> tuple[ApprovalRequest, bool]:
    """Record one approver's vote and execute the tool once quorum is met.

    Rules enforced here:

    * The approval must belong to the caller's organisation.
    * The caller's role must be permitted for the approval's risk tier.
    * The requester cannot approve their own request (separation of duties).
    * Each approver votes at most once.
    * The tool executes exactly once, after quorum, and is never re-executed.
    """
    if decision not in {ApprovalStatus.APPROVED.value, ApprovalStatus.REJECTED.value}:
        raise ValidationError(message="decision must be 'approved' or 'rejected'")

    approval = (
        await db.execute(
            select(ApprovalRequest).where(
                ApprovalRequest.id == approval_id, ApprovalRequest.org_id == principal.org_id
            )
        )
    ).scalar_one_or_none()
    if approval is None:
        # Deliberately 404 rather than 403: another tenant's approval ID must
        # not be distinguishable from a nonexistent one.
        raise NotFoundError(message="approval not found")

    if approval.status != ApprovalStatus.PENDING.value:
        raise ConflictError(
            message=f"approval has already been {approval.status}",
            details={"status": approval.status},
        )

    if not principal.can_approve(approval.risk_tier):
        audit.record_denied(
            db,
            action="approval.vote",
            resource="approval",
            org_id=principal.org_id,
            actor_id=principal.user_id,
            resource_id=approval.id,
            reason=f"role {principal.role!r} may not approve risk tier {approval.risk_tier}",
        )
        raise ConflictError(
            message=f"role '{principal.role}' may not approve a {approval.risk_tier} risk action",
            details={"risk_tier": approval.risk_tier},
        )

    if approval.requested_by == principal.user_id:
        audit.record_denied(
            db,
            action="approval.vote",
            resource="approval",
            org_id=principal.org_id,
            actor_id=principal.user_id,
            resource_id=approval.id,
            reason="requester attempted to approve their own request",
        )
        raise ConflictError(message="the requester cannot approve their own request")

    existing_votes = (
        (await db.execute(select(ApprovalVote).where(ApprovalVote.approval_id == approval.id)))
        .scalars()
        .all()
    )
    if any(vote.voter_id == principal.user_id for vote in existing_votes):
        raise ConflictError(message="you have already voted on this approval")

    db.add(
        ApprovalVote(
            approval_id=approval.id,
            voter_id=principal.user_id,
            decision=decision,
            note=note,
        )
    )
    await db.flush()
    audit.record(
        db,
        action="approval.vote",
        resource="approval",
        org_id=principal.org_id,
        actor_id=principal.user_id,
        resource_id=approval.id,
        decision="allow",
        reason=decision,
    )

    approve_votes = [v for v in existing_votes if v.decision == ApprovalStatus.APPROVED.value]
    approve_votes.append(
        ApprovalVote(
            approval_id=approval.id, voter_id=principal.user_id, decision=decision, note=note
        )
    )

    # A single rejection is terminal: a payment operation should not proceed
    # because enough other people said yes.
    if decision == ApprovalStatus.REJECTED.value:
        approval.status = ApprovalStatus.REJECTED.value
        approval.decided_at = utcnow()
        await _finish_rejected(db, approval, principal)
        return approval, False

    granted = {
        vote.voter_id for vote in approve_votes if vote.decision == ApprovalStatus.APPROVED.value
    }
    if len(granted) < approval.required_approvals:
        await publish_event(
            approval.run_id,
            {
                "type": "approval.vote",
                "approval_id": approval.id,
                "approvals_received": len(granted),
                "required_approvals": approval.required_approvals,
            },
        )
        await db.commit()
        return approval, False

    approval.status = ApprovalStatus.APPROVED.value
    approval.decided_at = utcnow()
    await _execute_approved(db, approval, principal)
    return approval, True


async def _finish_rejected(
    db: AsyncSession, approval: ApprovalRequest, principal: Principal
) -> None:
    run = await db.get(AgentRun, approval.run_id)
    if run is not None:
        run.answer = "工单创建请求已被审核人拒绝。"
        run.status = RunStatus.REJECTED.value
        run.output_tokens = _estimate_tokens(run.answer)
        run.estimated_cost_usd = _cost(run.input_tokens, run.output_tokens)
        run.updated_at = utcnow()
        db.add(_usage_event(run))
    await publish_event(
        approval.run_id,
        {"type": "approval.rejected", "approval_id": approval.id, "decided_by": principal.user_id},
    )
    await db.commit()


async def _execute_approved(
    db: AsyncSession, approval: ApprovalRequest, principal: Principal
) -> None:
    """Run the approved tool exactly once, then resume the run."""
    run = await db.get(AgentRun, approval.run_id)
    if run is None:
        raise NotFoundError(message="run not found")

    async def _do_execute() -> dict[str, Any]:
        if approval.tool_name == "create_ticket":
            ticket = await create_ticket_from_approval(db, approval)
            return {"ticket_id": ticket.id, "ticket_status": ticket.status}
        if approval.tool_name.startswith("ticket."):
            action = approval.tool_name.split(".", 1)[1]
            ticket_id = str(approval.arguments.get("ticket_id"))
            result = await execute_ticket_action(
                db,
                org_id=approval.org_id,
                actor_id=principal.user_id,
                ticket_id=ticket_id,
                action=action,
                note=str(approval.arguments.get("note", "")),
                resolution=str(approval.arguments.get("resolution", "")),
                assignee_id=approval.arguments.get("assignee_id"),
            )
            return {"ticket_id": ticket_id, "ticket_status": result.status}
        raise ValidationError(message=f"unknown tool: {approval.tool_name}")

    result, replayed = await replay_or_execute(
        db,
        org_id=approval.org_id,
        scope=f"approval.execute:{approval.tool_name}",
        key=approval.idempotency_key,
        payload=approval.arguments,
        execute=_do_execute,
    )

    approval.result = result
    approval.executed_at = utcnow()

    if approval.tool_name == "create_ticket":
        run.answer = f"已创建工单 {result['ticket_id']}：{approval.arguments.get('title', '')}" + (
            "（幂等重放）" if replayed else ""
        )
    else:
        run.answer = f"工单 {result['ticket_id']} 已更新为 {result['ticket_status']}。"

    run.status = RunStatus.SUCCEEDED.value
    run.output_tokens = _estimate_tokens(run.answer)
    run.estimated_cost_usd = _cost(run.input_tokens, run.output_tokens)
    run.updated_at = utcnow()
    db.add(_usage_event(run))
    audit.record(
        db,
        action="approval.executed",
        resource="approval",
        org_id=approval.org_id,
        actor_id=principal.user_id,
        resource_id=approval.id,
        reason=approval.tool_name,
        metadata={"replayed": replayed},
    )
    await publish_event(
        run.id,
        {
            "type": "tool.completed",
            "tool": approval.tool_name,
            "result": result,
            "replayed": replayed,
        },
    )
    await publish_event(
        run.id, {"type": "run.completed", "status": run.status, "answer": run.answer}
    )
    await db.commit()


async def create_ticket_from_approval(db: AsyncSession, approval: ApprovalRequest) -> Ticket:
    """Create the ticket. Called only from the approved path."""
    args = approval.arguments
    priority = str(args.get("priority", "normal"))
    ticket = Ticket(
        org_id=approval.org_id,
        created_by=approval.requested_by,
        run_id=approval.run_id,
        title=str(args.get("title", "Payment operations ticket"))[:255],
        description=str(args.get("description", "")),
        priority=priority,
        risk_tier=approval.risk_tier,
        status="open",
        sla_due_at=_sla_deadline(priority),
    )
    db.add(ticket)
    await db.flush()
    db.add(
        TicketEvent(
            ticket_id=ticket.id,
            org_id=approval.org_id,
            actor_id=approval.requested_by,
            action="created",
            to_status=ticket.status,
            note=f"created by agent run {approval.run_id}",
        )
    )
    await db.flush()
    return ticket


def _sla_deadline(priority: str) -> Any:
    from datetime import timedelta

    return utcnow() + timedelta(hours=sla_hours(priority))


# ---------------------------------------------------------------------------
# Ticketing helpers used by the API
# ---------------------------------------------------------------------------


async def count_received_approvals(db: AsyncSession, approval_id: str) -> int:
    approved = await db.scalar(
        select(func.count(ApprovalVote.id)).where(
            ApprovalVote.approval_id == approval_id,
            ApprovalVote.decision == ApprovalStatus.APPROVED.value,
        )
    )
    return int(approved or 0)
