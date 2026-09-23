import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.services.rag import (
    chunk_text,
    cosine_similarity,
    cosine_similarity_batch,
    deterministic_embedding,
    lexical_overlap,
)
from app.services.agent import _format_grounded_answer


def test_chunking_preserves_content_and_overlap() -> None:
    chunks = chunk_text("alpha " * 400, size=100, overlap=20)
    assert len(chunks) > 1
    assert "alpha" in chunks[0]
    assert chunks[0][-20:].split()[0] in chunks[1]


def test_deterministic_embedding_is_normalized() -> None:
    vector = deterministic_embedding("same input", dimensions=32)
    assert vector == deterministic_embedding("same input", dimensions=32)
    assert abs(cosine_similarity(vector, vector) - 1.0) < 1e-6
    scores = cosine_similarity_batch(vector, [vector])
    assert len(scores) == 1
    assert abs(scores[0] - 1.0) < 1e-6


def test_chunking_rejects_invalid_overlap() -> None:
    try:
        chunk_text("text", size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_chinese_lexical_overlap_matches_demo_runbook_terms() -> None:
    refund = "退款会原路退回原支付方式，通常需要 3-5 个工作日到账。"
    integration = "支付接口遇到 401 响应时，先确认 API key 和 Authorization header。"
    assert lexical_overlap("退款多久到账？", refund) > lexical_overlap("退款多久到账？", integration)
    assert lexical_overlap("支付接口 401 怎么排查？", integration) > lexical_overlap("支付接口 401 怎么排查？", refund)


def test_grounded_answer_has_operational_sections_and_citations() -> None:
    answer = _format_grounded_answer(
        "退款多久到账？",
        [
            {
                "filename": "refund-and-settlement-runbook.md",
                "content": "# 退款与结算 runbook 退款会原路退回原支付方式。银行卡退款在审核通过后通常需要 3-5 个工作日到账。客服升级前应确认 refund ID。",
            }
        ],
    )
    assert "结论：" in answer
    assert "关键点：" in answer
    assert "处理步骤：" in answer
    assert "3-5 个工作日" in answer
    assert "refund ID" in answer
    assert "来源：\n- refund-and-settlement-runbook.md" in answer
