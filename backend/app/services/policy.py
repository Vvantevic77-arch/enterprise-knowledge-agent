"""Intent classification and risk tiering for agent tool calls.

This replaces a single ``re.search`` that decided "create a ticket?" from one
sentence pattern. Two things changed:

* **Intent** is a multi-signal score across several runbook families, so a
  question phrased without the exact verb still routes correctly.
* **Risk** is tiered separately from intent. Whether a ticket needs one
  approver or two is a policy question about the *action*, not the wording of
  the prompt — a refund reversal is high risk even when asked politely.

The tier drives ``required_approvals`` and which roles may approve, so this
module is the single source of truth for the approval gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import RiskTier

# --------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------

TICKET_VERBS = r"创建|新建|提交|发起|开一个|开具|记录|登记|上报|回报|create|open|file|raise|log|report"
TICKET_NOUNS = r"工单|工单号|ticket|issue|incident|事件单|故障单"
REFUND_TERMS = r"退款|退还|refund|退单"
SETTLEMENT_TERMS = r"结算|清算|对账|打款|回款|settlement|reconcil|payout"
INTEGRATION_TERMS = r"401|403|429|接口|接入|鉴权|签名|signature|webhook|api|token|api key"
SECURITY_TERMS = r"密钥|泄露|泄漏|越权|未授权|安全|security|rotate|轮换|scope|权限"
LATENCY_TERMS = r"超时|超時|延迟|延迟|timeout|latency|p95|p99|慢"

#: Phrases that make an intent explicitly escalate a ticket.
ESCALATION_TERMS = r"紧急|urgent|p0|p1|加急|立即|马上|生产|production|线上|影响全部|大范围"

_INTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ticket", rf"({TICKET_VERBS}).{{0,24}}({TICKET_NOUNS})"),
    ("refund", REFUND_TERMS),
    ("settlement", SETTLEMENT_TERMS),
    ("integration", INTEGRATION_TERMS),
    ("security", SECURITY_TERMS),
    ("latency", LATENCY_TERMS),
)


@dataclass(frozen=True)
class Intent:
    name: str
    confidence: float


def classify_intent(prompt: str) -> Intent:
    """Pick the strongest matching runbook family.

    Returns ``general`` when nothing matches, which routes to a plain grounded
    answer rather than a tool call.
    """
    best_name = "general"
    best_hits = 0
    for name, pattern in _INTENT_PATTERNS:
        hits = len(re.findall(pattern, prompt, re.I))
        if hits > best_hits:
            best_name, best_hits = name, hits
    if best_hits == 0:
        return Intent(name="general", confidence=0.0)
    # More distinct signal classes present → higher confidence, capped at 1.0.
    distinct = sum(
        1 for _, pattern in _INTENT_PATTERNS if re.search(pattern, prompt, re.I)
    )
    return Intent(name=best_name, confidence=min(1.0, 0.5 + 0.25 * distinct))


def wants_ticket(prompt: str) -> bool:
    """True when the prompt asks to create a ticket.

    Accepts both the explicit ``verb ... noun`` shape and a bare incident
    declaration that names a ticket noun alongside an escalation word.
    """
    if re.search(rf"({TICKET_VERBS}).{{0,24}}({TICKET_NOUNS})", prompt, re.I):
        return True
    if re.search(rf"({TICKET_NOUNS}).{{0,16}}({TICKET_VERBS})", prompt, re.I):
        return True
    return bool(
        re.search(TICKET_NOUNS, prompt, re.I) and re.search(ESCALATION_TERMS, prompt, re.I)
    )


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------

#: Actions that move money, change security posture, or touch production.
#: These are high risk regardless of how the request is phrased.
_HIGH_RISK_PATTERNS = (
    r"重复扣款|double.?charge|多扣",
    r"密钥|key.{0,8}(轮换|撤销|revoke|rotate)|secret",
    r"泄露|泄漏|leak|泄露",
    r"生产环境|production|线上环境",
    r"越权|未授权|unauthori[sz]ed",
    r"关闭|禁用|停用|disable|revoke.{0,12}access",
    r"大范围|影响全部|多个商户|multiple merchants",
)

# A refund topic is not itself an action; an information question must remain
# low risk. Ticket requests and explicit refund commands still require review.
_REFUND_ACTION_PATTERNS = (
    r"(?:请|帮我|立即|马上|执行|发起|办理).{0,8}(?:退款|退还)",
    r"(?:撤销|取消|处理).{0,8}(?:退款|退还)",
    r"^(?:退款|退还)(?:这笔|该笔|订单|给|至|到)",
    r"\b(?:issue|process|execute|initiate|cancel|reverse|perform)\b.{0,16}\brefund\b",
    r"\brefund\b.{0,12}\b(?:order|transaction|payment)\b",
)

_CRITICAL_RISK_PATTERNS = (
    r"资金|赔付|转账|打款至|bank account|账户变更",
    r"紧急撤销|立即撤销|immediately revoke",
    r"数据泄露|data breach|合规|compliance violation",
)


def classify_risk(prompt: str, intent_name: str) -> str:
    """Assign a risk tier to the requested action.

    Order matters: critical beats high beats the intent default. A prompt
    containing no dangerous verb but asking for a ticket is still medium,
    because creating any ticket writes to a shared operations queue.
    """
    if any(re.search(p, prompt, re.I) for p in _CRITICAL_RISK_PATTERNS):
        return RiskTier.CRITICAL.value
    high_hits = sum(bool(re.search(p, prompt, re.I)) for p in _HIGH_RISK_PATTERNS)
    if re.search(REFUND_TERMS, prompt, re.I) and (
        wants_ticket(prompt)
        or any(re.search(p, prompt, re.I) for p in _REFUND_ACTION_PATTERNS)
    ):
        high_hits += 1
    if high_hits >= 1 and intent_name in {"refund", "security", "ticket", "settlement"}:
        return RiskTier.HIGH.value
    if high_hits >= 2:
        return RiskTier.HIGH.value
    if intent_name == "ticket":
        return RiskTier.MEDIUM.value
    return RiskTier.LOW.value


def ticket_arguments(prompt: str) -> dict[str, str]:
    """Derive ticket fields, defaulting priority from the escalation signal."""
    title = re.sub(r"\s+", " ", prompt).strip()[:120]
    if re.search(r"紧急|urgent|p0|立即|马上", prompt, re.I):
        priority = "critical"
    elif re.search(r"高优先级|p1|加急|生产|production|线上", prompt, re.I):
        priority = "high"
    else:
        priority = "normal"
    return {"title": title, "description": prompt.strip(), "priority": priority}


#: SLA budget in hours per priority, used to set ``Ticket.sla_due_at``.
SLA_HOURS: dict[str, int] = {
    "critical": 2,
    "high": 8,
    "normal": 24,
    "low": 72,
}


def sla_hours(priority: str) -> int:
    return SLA_HOURS.get(priority, 24)
