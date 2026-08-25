"""Pure protocol helpers for the Sequential Meta-Harness experiment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SplitCounts:
    """Number of ordered tasks assigned to each benchmark split."""

    train: int
    validation: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test

    def validate(self, expected_total: int | None = None) -> None:
        if self.train < 1 or self.validation < 1 or self.test < 1:
            raise ValueError("train, validation, and test must all contain tasks")
        if expected_total is not None and self.total != expected_total:
            raise ValueError(
                f"split counts total {self.total}, expected {expected_total}"
            )


@dataclass(frozen=True)
class BenchmarkSplit:
    """Ordered, non-overlapping task IDs for one benchmark."""

    benchmark: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def all_tasks(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test

    def to_manifest(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "ordered_tasks": list(self.all_tasks),
        }


def split_task_order(
    task_order: Iterable[tuple[str, str]], counts: SplitCounts
) -> list[BenchmarkSplit]:
    """Split each contiguous Sequential benchmark block without reordering it."""

    grouped: list[tuple[str, list[str]]] = []
    for benchmark, task_id in task_order:
        if not grouped or grouped[-1][0] != benchmark:
            if any(previous == benchmark for previous, _ in grouped):
                raise ValueError(
                    f"benchmark {benchmark!r} is not contiguous in Sequential order"
                )
            grouped.append((benchmark, []))
        grouped[-1][1].append(str(task_id))

    splits: list[BenchmarkSplit] = []
    for benchmark, task_ids in grouped:
        counts.validate(len(task_ids))
        train_end = counts.train
        validation_end = train_end + counts.validation
        split = BenchmarkSplit(
            benchmark=benchmark,
            train=tuple(task_ids[:train_end]),
            validation=tuple(task_ids[train_end:validation_end]),
            test=tuple(task_ids[validation_end:]),
        )
        if len(set(split.all_tasks)) != len(split.all_tasks):
            raise ValueError(f"duplicate task IDs in benchmark {benchmark!r}")
        splits.append(split)
    return splits


@dataclass(frozen=True)
class CandidateResult:
    """Validation result used for deployment selection."""

    candidate_id: str
    validation_score: float
    validation_successes: int
    validation_tasks: int
    mean_tokens: float
    mean_cost: float
    order: int
    candidate_path: str
    state_path: str
    candidate_sha256: str
    state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateResult:
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


def candidate_rank(candidate: CandidateResult) -> tuple[float, int, float, float, int]:
    """Higher tuples win; earlier candidates win a completely equal tie."""

    return (
        candidate.validation_score,
        candidate.validation_successes,
        -candidate.mean_tokens,
        -candidate.mean_cost,
        -candidate.order,
    )


def select_winner(candidates: Iterable[CandidateResult]) -> CandidateResult:
    values = list(candidates)
    if not values:
        raise ValueError("cannot select a winner from an empty frontier")
    return max(values, key=candidate_rank)


def pareto_frontier(candidates: Iterable[CandidateResult]) -> list[CandidateResult]:
    """Return score/cost non-dominated candidates in stable evaluation order."""

    values = list(candidates)
    frontier: list[CandidateResult] = []
    for candidate in values:
        dominated = any(
            other.candidate_id != candidate.candidate_id
            and other.validation_score >= candidate.validation_score
            and other.mean_cost <= candidate.mean_cost
            and (
                other.validation_score > candidate.validation_score
                or other.mean_cost < candidate.mean_cost
            )
            for other in values
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: item.order)
