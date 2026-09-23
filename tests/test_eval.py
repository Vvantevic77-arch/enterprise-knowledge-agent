import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from evals.run_eval import run


def test_eval_contract() -> None:
    scores = run(Path(__file__).parents[1] / "evals" / "dataset.json")
    assert set(scores) == {"retrieval_recall", "answer_correctness", "tool_success_rate"}
    assert all(0 <= score <= 1 for score in scores.values())
