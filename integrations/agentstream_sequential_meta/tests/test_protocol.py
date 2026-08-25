from __future__ import annotations

import pytest

from integrations.agentstream_sequential_meta.protocol import (
    CandidateResult,
    SplitCounts,
    pareto_frontier,
    select_winner,
    split_task_order,
)


def _candidate(
    candidate_id: str,
    *,
    score: float,
    successes: int,
    tokens: float,
    cost: float,
    order: int,
) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        validation_score=score,
        validation_successes=successes,
        validation_tasks=10,
        mean_tokens=tokens,
        mean_cost=cost,
        order=order,
        candidate_path=f"{candidate_id}/candidate.py",
        state_path=f"{candidate_id}/harness_store.json",
        candidate_sha256=candidate_id * 4,
        state_sha256=candidate_id * 5,
    )


def test_split_preserves_sequential_blocks_and_order() -> None:
    order = [
        *(('bfcl', f'b{i}') for i in range(5)),
        *(('tau2', f't{i}') for i in range(5)),
    ]

    splits = split_task_order(order, SplitCounts(train=2, validation=2, test=1))

    assert [split.benchmark for split in splits] == ['bfcl', 'tau2']
    assert splits[0].train == ('b0', 'b1')
    assert splits[0].validation == ('b2', 'b3')
    assert splits[0].test == ('b4',)
    assert splits[1].all_tasks == tuple(f't{i}' for i in range(5))


def test_split_rejects_non_contiguous_benchmark() -> None:
    order = [('bfcl', 'b0'), ('tau2', 't0'), ('bfcl', 'b1')]

    with pytest.raises(ValueError, match='not contiguous'):
        split_task_order(order, SplitCounts(train=1, validation=1, test=1))


def test_split_counts_must_match_selected_tasks() -> None:
    with pytest.raises(ValueError, match='expected 4'):
        split_task_order(
            [('bfcl', f'b{i}') for i in range(4)],
            SplitCounts(train=2, validation=1, test=2),
        )


def test_winner_uses_score_success_cost_then_stable_order() -> None:
    candidates = [
        _candidate('baseline', score=0.8, successes=8, tokens=100, cost=2, order=0),
        _candidate('more_success', score=0.8, successes=9, tokens=200, cost=3, order=1),
        _candidate('cheaper', score=0.8, successes=9, tokens=150, cost=1, order=2),
        _candidate('later_tie', score=0.8, successes=9, tokens=150, cost=1, order=3),
    ]

    assert select_winner(candidates).candidate_id == 'cheaper'


def test_baseline_remains_winner_when_candidates_regress() -> None:
    baseline = _candidate(
        'baseline', score=0.7, successes=7, tokens=100, cost=1, order=0
    )
    regressed = _candidate(
        'candidate', score=0.6, successes=9, tokens=50, cost=0.5, order=1
    )

    assert select_winner([baseline, regressed]) == baseline


def test_pareto_frontier_keeps_score_cost_tradeoffs() -> None:
    cheap = _candidate('cheap', score=0.6, successes=6, tokens=80, cost=1, order=0)
    strong = _candidate('strong', score=0.8, successes=8, tokens=150, cost=2, order=1)
    dominated = _candidate(
        'dominated', score=0.5, successes=5, tokens=200, cost=3, order=2
    )

    assert [item.candidate_id for item in pareto_frontier([cheap, strong, dominated])] == [
        'cheap',
        'strong',
    ]
