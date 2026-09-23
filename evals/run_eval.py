"""Run lightweight offline checks for payment knowledge retrieval and tool policy.

The runner intentionally has no model dependency. In CI, replace `answer` with
captured production responses and keep the same metrics contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def score_answer(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(bool(re.search(re.escape(keyword), answer, re.I)) for keyword in keywords)
    return hits / len(keywords)


def run(dataset_path: Path) -> dict[str, float]:
    cases = json.loads(dataset_path.read_text())
    recall = sum(
        score_answer(case["retrieved_text"], case["relevant_keywords"]) for case in cases
    ) / max(1, len(cases))
    answer_cases = [case for case in cases if case["expected_tool"] is None]
    answer_correctness = sum(
        score_answer(case["answer"], case["relevant_keywords"]) for case in answer_cases
    ) / max(1, len(answer_cases))
    tool_cases = [case for case in cases if case["expected_tool"]]
    tool_success = sum(
        1 for case in tool_cases if case["expected_tool"] == case["observed_tool"]
    ) / max(1, len(tool_cases))
    return {"retrieval_recall": recall, "answer_correctness": answer_correctness, "tool_success_rate": tool_success}


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).with_name("dataset.json")), indent=2, ensure_ascii=False))
