"""Curated runbook corpus used for the guided demo and offline evaluation.

The seeder goes through ``ingest_document`` rather than writing chunks itself,
so re-seeding with unchanged content exercises the same zero-write incremental
path as a production re-upload.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentStatus, KnowledgeBase
from app.services.ingestion import create_document, ingest_document

DEMO_KB_NAME = "Merchant operations demo"
DEMO_KB_DESCRIPTION = (
    "Curated runbooks for grounded answers, citations, and approval workflows."
)

DEMO_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "refund-and-settlement-runbook.md",
        """# 退款与结算 runbook

退款会原路退回原支付方式。银行卡退款在审核通过后通常需要 3-5 个工作日到账，银行转账退款最长可能需要 7 个工作日。客服升级前应确认 refund ID、原交易 ID、金额和币种。

如果退款处于 pending 状态，请先检查支付状态、确认 refund event 已被接受，并查看下一个工作日的 settlement batch。退款 pending 时不要重复发起第二笔退款。创建支持工单时必须带上原交易 ID，方便 Finance 追踪结算记录。

如果退款超过正常 SLA，收集 merchant ID、refund ID 和审核通过日期，再将问题以 high priority 转给 Payments Operations。""",
    ),
    (
        "payment-integration-runbook.md",
        """# 支付接口 integration runbook

遇到 401 响应时，先确认 API key 属于正确的 merchant environment，并检查 Authorization header 是否使用 Bearer scheme。密钥按环境隔离，不能把 production key 复制到 sandbox。

遇到 429 响应时，使用 exponential backoff，遵守 Retry-After header；没有原始 idempotency key 时，不要直接重试非幂等请求。Webhook 只有在完成签名校验并持久化事件后，才应返回 2xx。

如果 webhook signature 校验失败，先对比 raw request body 和 signing secret，再解析 JSON。记录 event ID 和时间戳，方便 on-call 工程师安全 replay 事件。""",
    ),
    (
        "incident-response-guide.md",
        """# 商户 incident response guide

P1 事件包括大范围授权失败、重复扣款，或影响多个商户的 settlement batch 缺失。第一响应人应创建 incident ticket，记录开始时间和受影响 merchant ID，并通知 Payments Operations owner。

调整 routing 或 retry policy 前，先记录 trace ID 和一个代表性 transaction ID。对外沟通只陈述已确认的影响、最新进展和下一次更新时间；在 on-call 工程师确认 mitigation 前，不要承诺恢复时间。

恢复后，将时间线、root cause、mitigation 和 follow-up owner 补充到 incident ticket。P1 事件必须在两个工作日内完成 post-incident review。""",
    ),
    (
        "api-authentication-and-security.md",
        """# API authentication and security runbook

所有 API key 必须按 merchant 和 environment 隔离，sandbox 与 production 不得共用密钥。请求使用 Authorization: Bearer token，服务端需要校验 token 的签发方、受众、过期时间和最小权限 scope。

密钥轮换采用双密钥窗口：先创建新 key，部署并验证新 key 成功后，再撤销旧 key。secret 只能存放在 secrets manager 或受控环境变量中，不得提交到 Git、日志或工单附件。轮换完成后记录 key fingerprint、操作者、时间和影响的 merchant。

权限遵循 least privilege。读写、退款、导出和管理员操作使用不同 scope；生产环境的高风险 scope 默认关闭。发现疑似泄漏时立即撤销 key，保留 audit log，并创建 P1 security ticket。""",
    ),
    (
        "webhook-reliability.md",
        """# Webhook reliability runbook

Webhook consumer 必须先读取 raw request body，使用 merchant 对应的 signing secret 验证 signature 和 timestamp，再解析 JSON。时间戳偏差超过 5 分钟的请求默认拒绝，避免 replay attack。

以 event ID 做幂等键，并在数据库建立唯一约束。只有事件已经完成签名校验、幂等检查和持久化后，才返回 2xx；业务处理可以由 Celery 异步执行。重复事件直接返回已处理结果，不能重复扣款或重复退款。

非 2xx 响应进入指数退避重试，达到上限后写入 dead-letter queue。replay 前先确认事件版本、目标 merchant 和当前处理状态，并记录 trace ID、event ID 与 replay 操作者。""",
    ),
    (
        "rbac-and-tenant-isolation.md",
        """# Multi-tenant RBAC and isolation runbook

每个请求都必须解析 organization_id，并在 repository 查询、写入和后台任务中携带 tenant scope。禁止只依赖前端传入的 organization_id；服务端要校验当前用户的 membership，避免跨组织读取知识库、工单、审批和 Agent run。

角色分为 owner、admin、member、viewer。owner 可以管理组织和成员；admin 可以管理知识库、审批高风险工具和工单；member 可以提问、上传资料并发起审批；viewer 只能读取被授权的知识库和运行结果。权限检查使用 action + resource，而不是散落的 if 判断。

所有资源查询都应先过滤 organization_id，再过滤资源 ID；更新和删除使用同一条件并检查受影响行数。审计日志至少记录 actor、organization_id、resource、action、decision、trace ID 和时间。发现越权访问时拒绝请求并触发安全告警。""",
    ),
    (
        "rag-indexing-and-evaluation.md",
        """# RAG indexing and evaluation runbook

文档进入索引前先规范化文本、按语义边界切 chunk，并保存 document_id、ordinal、filename、source_url 和 content hash。重复提交使用 SHA-256 内容哈希去重；只有内容变化的文档才删除旧 chunk 并重写向量。

生产环境使用 pgvector 做 coarse nearest-neighbour 检索，再把向量相似度与中文词法 overlap 合并重排。检索结果必须带 chunk ID 和来源文件，生成模型只能使用返回 context，不能把未检索到的事实写进答案。

离线评测至少记录 Recall@5、MRR、命中来源和延迟。修改 chunk size、embedding 或 reranker 后，使用同一组 query、ground truth 和 tenant fixture 对比，避免只看平均分。""",
    ),
    (
        "celery-redis-ingestion.md",
        """# Celery and Redis ingestion runbook

API 上传只负责保存原文件和创建 ingestion job，文档解析、文本清洗、chunking、embedding 和写入向量库由 Celery worker 异步执行。Redis 作为 broker 和短期 result backend，生产环境需要设置密码、TLS、连接池上限和合理的 visibility timeout。

任务必须携带 organization_id、knowledge_base_id、document_id 和 trace ID。网络请求、PDF 解析和 embedding 调用使用有限次数 retry，并采用 exponential backoff + jitter；不可恢复的解析错误写入 document.error，不要无限重试。

任务成功后按事务顺序更新 document status 为 indexed，再发布完成事件。超过重试上限的任务进入 dead-letter queue，由运维人员确认后 replay。worker 需要设置并发、prefetch 和 soft time limit，避免单个大文件耗尽资源。""",
    ),
    (
        "observability-and-tracing.md",
        """# Observability and tracing runbook

每次 Agent run 从入口生成 trace ID，并把它传给检索、模型调用、工具执行、Celery task 和数据库日志。结构化日志至少包含 timestamp、level、service、organization_id、run_id、trace_id、event 和 duration_ms；不要记录 API key、完整 Authorization header 或支付敏感数据。

核心指标包括请求成功率、检索命中率、模型错误率、token usage、estimated cost、队列深度和 p50/p95/p99 latency。p95 表示 95% 的请求在该耗时以内，剩余 5% 是慢请求，应按 endpoint、tenant 和 provider 拆分分析。

告警需要绑定 runbook 和 owner。例如 p95 持续超过 SLA、429 比例升高、dead-letter queue 增长或 Recall@5 下降时，先用 trace ID 定位具体阶段，再决定回滚、限流或重建索引。""",
    ),
    (
        "database-and-migration-runbook.md",
        """# Database and migration runbook

FastAPI 请求使用 SQLAlchemy 2.0 AsyncSession，通过事务边界控制知识库、文档、chunk、审批和工单的一致性。读写操作不要跨请求复用 session；提交前完成 tenant scope 和权限校验，异常时 rollback 并记录 trace ID。

Schema 变化使用 Alembic migration，先添加兼容字段和索引，再发布应用，最后清理旧字段。生产迁移前备份并在 staging 回放；大表索引使用 CONCURRENTLY，避免阻塞在线请求。pgvector 索引参数变更也要记录 embedding model 和 dimensions。

发布后检查 migration version、连接池利用率、慢查询和错误率。回滚必须有明确的 down migration 或前向修复方案；不要直接手工改生产表，也不要在应用启动时自动执行破坏性 DDL。""",
    ),
    (
        "ticket-lifecycle-and-approval.md",
        """# 工单生命周期与审批 runbook

工单状态包含 open、triaged、in_progress、waiting_on_merchant、resolved、closed 和 cancelled。状态迁移必须由服务端校验：open 只能进入 triaged、in_progress 或 cancelled；resolved 只能进入 closed，或按商户反馈重新 open 为 in_progress。

高风险工具调用必须先暂停在 waiting_approval。审批优先级由风险层级决定：refund、密钥轮换、生产环境变更属于 high risk，需要两名不同的审核人；资金划转或账户变更属于 critical。发起人不能审批自己的请求，这一点必须由服务端强制。

审批通过后工具只能执行一次。以 idempotency key 作为唯一约束，重放同一个 key 必须返回已存储的结果而不是重复创建工单。工单执行动作包括 triage、assign、start、wait_on_merchant、resolve、close、reopen 和 cancel，每一步都要写入 ticket event 和 audit log。""",
    ),
)


async def seed_demo_knowledge(
    db: AsyncSession, org_id: str
) -> tuple[KnowledgeBase, bool]:
    """Create or refresh the tenant-scoped demo corpus.

    Idempotent by knowledge-base name and filename. Re-running with unchanged
    content rewrites nothing, because ``ingest_document`` short-circuits on the
    content hash.
    """
    knowledge_base = (
        await db.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.org_id == org_id, KnowledgeBase.name == DEMO_KB_NAME
            )
        )
    ).scalar_one_or_none()
    created = knowledge_base is None
    if knowledge_base is None:
        knowledge_base = KnowledgeBase(
            org_id=org_id, name=DEMO_KB_NAME, description=DEMO_KB_DESCRIPTION
        )
        db.add(knowledge_base)
        await db.flush()

    for filename, content in DEMO_SOURCES:
        document = (
            await db.execute(
                select(Document).where(
                    Document.kb_id == knowledge_base.id, Document.filename == filename
                )
            )
        ).scalar_one_or_none()
        if document is None:
            document = await create_document(
                db,
                org_id=org_id,
                kb_id=knowledge_base.id,
                filename=filename,
                source_url=f"demo://{filename}",
            )
        await ingest_document(
            db, document_id=document.id, filename=filename, text=content
        )

    await db.commit()
    return knowledge_base, created


DEMO_QUERIES: tuple[str, ...] = (
    "退款多久到账？",
    "支付接口返回 401 怎么排查？",
    "商户结算批次缺失，我需要创建工单。",
)
