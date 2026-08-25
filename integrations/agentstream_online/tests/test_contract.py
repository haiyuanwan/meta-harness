from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from exgentic.agents.harness.harness_store import HarnessStore

from integrations.agentstream_online.contract import (
    CandidateValidationError,
    validate_candidate,
)

INTEGRATION_DIR = Path(__file__).resolve().parents[1]


def _valid_pair(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate.py"
    state = tmp_path / "harness_store.json"
    shutil.copy2(INTEGRATION_DIR / "candidate.py", candidate)
    HarnessStore("harness_sequential_global").save_checkpoint(str(state))
    return candidate, state


def test_generation_zero_candidate_satisfies_contract(tmp_path: Path) -> None:
    candidate, state = _valid_pair(tmp_path)

    report = validate_candidate(candidate, state)

    assert report["valid"] is True
    assert report["session_count"] == 0
    assert report["skill_count"] == 0


def test_builtin_evolver_reference_is_rejected(tmp_path: Path) -> None:
    candidate, state = _valid_pair(tmp_path)
    candidate.write_text(
        candidate.read_text(encoding="utf-8") + "\n# run_evolver\n",
        encoding="utf-8",
    )

    with pytest.raises(CandidateValidationError, match="run_evolver"):
        validate_candidate(candidate, state)


def test_missing_checkpoint_field_is_rejected(tmp_path: Path) -> None:
    candidate, state = _valid_pair(tmp_path)
    payload = json.loads(state.read_text(encoding="utf-8"))
    del payload["memory"]
    state.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateValidationError, match="missing 'memory'"):
        validate_candidate(candidate, state)


def test_forbidden_network_import_is_rejected(tmp_path: Path) -> None:
    candidate, state = _valid_pair(tmp_path)
    candidate.write_text(
        "import requests\n" + candidate.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(CandidateValidationError, match="forbidden module"):
        validate_candidate(candidate, state)
