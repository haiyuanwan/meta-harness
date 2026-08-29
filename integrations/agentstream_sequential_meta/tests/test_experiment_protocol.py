from __future__ import annotations

import pytest

from integrations.agentstream_sequential_meta.experiment_protocol import (
    build_formal_partitions,
    task_pool_fingerprint,
)


def _inventories():
    return {
        "bfcl": [f"b{i}" for i in range(200)],
        "browsecompplus": [f"q{i}" for i in range(830)],
    }


def test_formal_partition_is_deterministic_disjoint_and_complete() -> None:
    first = build_formal_partitions(_inventories(), ordering_seed=44)
    second = build_formal_partitions(_inventories(), ordering_seed=44)

    assert first == second
    for partition in first:
        groups = [
            set(partition.split.train),
            set(partition.split.validation),
            set(partition.split.test),
            set(partition.audit),
        ]
        assert sum(len(group) for group in groups) == len(set.union(*groups))
        assert len(set.union(*groups)) in {200, 830}

    bfcl, browse = first
    assert [len(bfcl.split.train), len(bfcl.split.validation)] == [20, 10]
    assert len(bfcl.split.test) == 50
    assert len(bfcl.audit) == 120
    assert len(bfcl.hidden_streams) == 10
    assert all(len(stream) == 5 for stream in bfcl.hidden_streams)
    assert len(browse.split.test) == 100
    assert len(browse.audit) == 700
    assert len(browse.hidden_streams) == 10
    assert all(len(stream) == 10 for stream in browse.hidden_streams)


def test_public_commitment_does_not_reveal_task_ids() -> None:
    partition = build_formal_partitions(_inventories(), ordering_seed=44)[0]
    public = partition.public_commitment()

    assert "search" not in public
    assert "hidden" not in public
    assert "audit" not in public
    assert public["counts"]["hidden"] == 50
    assert set(public["partition_hashes"]) == {
        "search",
        "validation",
        "hidden",
        "audit",
    }


def test_formal_partition_rejects_changed_pool_size() -> None:
    inventories = _inventories()
    inventories["bfcl"].pop()

    with pytest.raises(ValueError, match="expected 200"):
        build_formal_partitions(inventories, ordering_seed=44)


def test_pool_fingerprint_tracks_enumeration_order() -> None:
    assert task_pool_fingerprint(["a", "b"]) != task_pool_fingerprint(["b", "a"])
