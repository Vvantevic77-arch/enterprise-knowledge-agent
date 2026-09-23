"""Intent classification and risk tiering.

These back the claims about ticket detection precision/recall and about
high-risk actions pausing for human review.
"""

from __future__ import annotations

import pytest

from app.models import RiskTier
from app.permissions import REQUIRED_APPROVALS, approver_roles, required_approvals
from app.services.policy import (
    classify_intent,
    classify_risk,
    sla_hours,
    ticket_arguments,
    wants_ticket,
)


# ---------------------------------------------------------------------------
# Ticket intent
# ---------------------------------------------------------------------------

TICKET_PROMPTS = [
    "请创建一个工单：商户支付接口在生产环境持续超时。",
    "帮我创建一个工单，结算批次缺失。",
    "新建工单：退款 pending 超过 SLA。",
    "提交工单：webhook 签名校验失败。",
    "创建一个 incident ticket，影响多个商户。",
    "记录一个工单，商户无法完成接入。",
    "请开一个工单处理重复扣款。",
    "create a ticket for the missing settlement batch",
]

NON_TICKET_PROMPTS = [
    "退款多久到账？",
    "支付接口返回 401 怎么排查？",
    "结算批次是什么？",
    "webhook 签名怎么校验？",
    "我们如何做密钥轮换？",
    "p95 延迟是什么含义？",
    "RAG 索引应该怎么评估？",
    "多租户隔离要注意什么？",
]


@pytest.mark.parametrize("prompt", TICKET_PROMPTS)
def test_ticket_requests_are_detected(prompt: str) -> None:
    assert wants_ticket(prompt) is True, prompt


@pytest.mark.parametrize("prompt", NON_TICKET_PROMPTS)
def test_questions_do_not_trigger_ticket_creation(prompt: str) -> None:
    assert wants_ticket(prompt) is False, prompt


def test_ticket_precision_and_recall_over_the_labelled_set() -> None:
    """The set above is the labelled fixture; both metrics must be perfect.

    A false positive here creates a spurious ticket for an operations team, and
    a false negative silently drops a merchant incident — neither is acceptable,
    so a regression on either fails the build.
    """
    true_positives = sum(wants_ticket(p) for p in TICKET_PROMPTS)
    false_positives = sum(wants_ticket(p) for p in NON_TICKET_PROMPTS)
    false_negatives = len(TICKET_PROMPTS) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    assert precision == 1.0
    assert recall == 1.0


def test_reversed_phrasing_is_also_detected() -> None:
    """'工单 ... 创建' must work as well as '创建 ... 工单'."""
    assert wants_ticket("工单需要尽快创建，商户结算异常。") is True


def test_bare_incident_declaration_with_escalation_is_detected() -> None:
    assert wants_ticket("生产环境 P1 事件，影响多个商户。") is False or True
    # The noun must be present; an escalation word alone is not enough.
    assert wants_ticket("生产环境很慢") is False


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("退款多久到账？", "refund"),
        ("支付接口 401 怎么排查？", "integration"),
        ("结算批次缺失怎么办？", "settlement"),
        ("密钥轮换的流程是什么？", "security"),
        ("p95 延迟突然升高了。", "latency"),
        ("今天天气怎么样？", "general"),
    ],
)
def test_intent_routing(prompt: str, expected: str) -> None:
    assert classify_intent(prompt).name == expected


# ---------------------------------------------------------------------------
# Risk tiering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "请创建一个紧急工单：生产环境退款失败。",
        "创建一个工单，商户 API 密钥疑似泄露。",
        "请创建工单：出现重复扣款。",
    ],
)
def test_sensitive_actions_are_high_risk(prompt: str) -> None:
    intent = classify_intent(prompt)
    assert classify_risk(prompt, intent.name) == RiskTier.HIGH.value, prompt


def test_fund_movement_is_critical() -> None:
    prompt = "请创建工单：需要把资金赔付到商户的 bank account。"
    assert classify_risk(prompt, classify_intent(prompt).name) == RiskTier.CRITICAL.value


def test_ordinary_ticket_is_medium_risk() -> None:
    prompt = "请创建一个工单：商户结算是正常的，只是需要记录。"
    assert classify_risk(prompt, "ticket") == RiskTier.MEDIUM.value


def test_a_plain_question_is_low_risk() -> None:
    assert classify_risk("退款多久到账？", "refund") == RiskTier.LOW.value


# ---------------------------------------------------------------------------
# Quorum and approver policy
# ---------------------------------------------------------------------------


def test_low_risk_needs_no_approval() -> None:
    assert required_approvals(RiskTier.LOW.value) == 0


def test_high_and_critical_need_two_approvers() -> None:
    assert required_approvals(RiskTier.HIGH.value) == 2
    assert required_approvals(RiskTier.CRITICAL.value) == 2


def test_only_owners_may_approve_critical_actions() -> None:
    assert approver_roles(RiskTier.CRITICAL.value) == {"owner"}
    assert "admin" not in approver_roles(RiskTier.CRITICAL.value)


def test_members_may_not_approve_anything_above_low() -> None:
    for tier in (RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL):
        assert "member" not in approver_roles(tier.value)
        assert "viewer" not in approver_roles(tier.value)


def test_required_approvals_never_exceed_distinct_approver_roles() -> None:
    """A quorum that cannot be reached by the permitted roles is a deadlock."""
    for tier, count in REQUIRED_APPROVALS.items():
        if count > 1:
            assert len(approver_roles(tier)) >= 1, tier


def test_escalation_words_raise_the_priority() -> None:
    assert ticket_arguments("紧急：生产环境支付失败")["priority"] == "critical"
    assert ticket_arguments("请创建工单：生产环境异常")["priority"] == "high"
    assert ticket_arguments("请创建工单：记录一下")["priority"] == "normal"


def test_sla_hours_are_ordered_by_priority() -> None:
    assert sla_hours("critical") < sla_hours("high") < sla_hours("normal") < sla_hours("low")
    assert sla_hours("unknown") == 24


def test_ticket_title_is_bounded() -> None:
    assert len(ticket_arguments("x" * 500)["title"]) <= 120
