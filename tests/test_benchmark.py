import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.benchmark import (  # noqa: E402
    benchmark_incremental_index,
    benchmark_policy,
    build_corpus,
)


def test_benchmark_core_contracts() -> None:
    documents, chunks = build_corpus(10)
    assert len(documents) == 10
    assert chunks

    incremental = benchmark_incremental_index(documents)
    assert incremental["duplicate_zero_ops"] == 10
    assert incremental["duplicate_zero_op_rate"] == 1.0

    policy = benchmark_policy(10)
    assert policy["ticket_precision"] == 1.0
    assert policy["ticket_recall"] == 1.0
    assert policy["approval_gate_coverage"] == 1.0
