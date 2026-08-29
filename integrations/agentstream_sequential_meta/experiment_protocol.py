"""Deterministic dataset commitments for the transfer/HDA experiment."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from .protocol import BenchmarkSplit

SELECTION_SEED = 42


@dataclass(frozen=True)
class FormalLayout:
    expected_pool_size: int
    search: int
    validation: int
    hidden: int
    hidden_stream_size: int

    @property
    def selected(self) -> int:
        return self.search + self.validation + self.hidden


FORMAL_LAYOUTS: dict[str, FormalLayout] = {
    "bfcl": FormalLayout(
        expected_pool_size=200,
        search=20,
        validation=10,
        hidden=50,
        hidden_stream_size=5,
    ),
    "browsecompplus": FormalLayout(
        expected_pool_size=830,
        search=20,
        validation=10,
        hidden=100,
        hidden_stream_size=10,
    ),
}


@dataclass(frozen=True)
class FormalPartition:
    benchmark: str
    split: BenchmarkSplit
    audit: tuple[str, ...]
    hidden_streams: tuple[tuple[str, ...], ...]
    pool_fingerprint: str

    def private_manifest(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "selection_seed": SELECTION_SEED,
            "pool_fingerprint": self.pool_fingerprint,
            "search": list(self.split.train),
            "validation": list(self.split.validation),
            "hidden": list(self.split.test),
            "hidden_streams": [list(stream) for stream in self.hidden_streams],
            "audit": list(self.audit),
        }

    def public_commitment(self) -> dict[str, Any]:
        private = self.private_manifest()
        return {
            "benchmark": self.benchmark,
            "selection_seed": SELECTION_SEED,
            "pool_size": len(self.split.all_tasks) + len(self.audit),
            "pool_fingerprint": self.pool_fingerprint,
            "counts": {
                "search": len(self.split.train),
                "validation": len(self.split.validation),
                "hidden": len(self.split.test),
                "audit": len(self.audit),
                "hidden_streams": len(self.hidden_streams),
                "hidden_stream_size": (
                    len(self.hidden_streams[0]) if self.hidden_streams else 0
                ),
            },
            "partition_hashes": {
                key: _fingerprint(private[key])
                for key in ("search", "validation", "hidden", "audit")
            },
        }


def _derive_seed(master_seed: int, slug: str) -> int:
    digest = hashlib.md5(f"{master_seed}_{slug}".encode()).hexdigest()
    return int(digest, 16) % (2**31)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_pool_fingerprint(task_ids: list[str]) -> str:
    """Fingerprint the complete official enumeration, including its order."""

    return _fingerprint([str(item) for item in task_ids])


def build_formal_partitions(
    inventories: dict[str, list[str]], ordering_seed: int
) -> list[FormalPartition]:
    unknown = sorted(set(inventories) - set(FORMAL_LAYOUTS))
    missing = sorted(set(FORMAL_LAYOUTS) - set(inventories))
    if unknown or missing:
        raise ValueError(
            f"Formal benchmark set mismatch; unknown={unknown}, missing={missing}"
        )

    partitions: list[FormalPartition] = []
    for benchmark in FORMAL_LAYOUTS:
        layout = FORMAL_LAYOUTS[benchmark]
        task_ids = [str(item) for item in inventories[benchmark]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"Duplicate task IDs in {benchmark}")
        if len(task_ids) != layout.expected_pool_size:
            raise ValueError(
                f"{benchmark} pool has {len(task_ids)} tasks; expected "
                f"{layout.expected_pool_size}"
            )

        selected_order = list(task_ids)
        selection_rng = random.Random(_derive_seed(SELECTION_SEED, benchmark))
        selection_rng.shuffle(selected_order)
        selected = selected_order[: layout.selected]
        audit = tuple(selected_order[layout.selected :])
        if ordering_seed != SELECTION_SEED:
            order_rng = random.Random(_derive_seed(ordering_seed, benchmark))
            order_rng.shuffle(selected)

        search_end = layout.search
        validation_end = search_end + layout.validation
        split = BenchmarkSplit(
            benchmark=benchmark,
            train=tuple(selected[:search_end]),
            validation=tuple(selected[search_end:validation_end]),
            test=tuple(selected[validation_end:]),
        )
        hidden_streams = tuple(
            split.test[index : index + layout.hidden_stream_size]
            for index in range(0, len(split.test), layout.hidden_stream_size)
        )
        partitions.append(
            FormalPartition(
                benchmark=benchmark,
                split=split,
                audit=audit,
                hidden_streams=hidden_streams,
                pool_fingerprint=task_pool_fingerprint(task_ids),
            )
        )
    return partitions
