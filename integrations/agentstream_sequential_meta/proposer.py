"""Claude Code proposer for benchmark-level Meta-Harness iterations."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write"]
DISALLOWED_TOOLS = ["Bash", "WebSearch", "WebFetch", "Agent"]


def build_proposer_prompt(
    *, benchmark: str, iteration: int, candidate_number: int, base_model: str
) -> str:
    return f"""Run one Meta-Harness proposal for AgentStream Sequential.

Current search:
- benchmark: {benchmark}
- evolution iteration: {iteration}
- candidate number: {candidate_number}
- fixed solver model: {base_model}

The complete running harness consists of candidate.py plus persistent state.
You evolve the harness program in candidate.py. The controller will clone the exact
same incoming state for every candidate, run train/search and validation, and select
the winner using validation results. Do not edit incoming_harness_store.json.

Files:
- candidate.py: current validation-frontier parent; edit this file directly.
- incoming_harness_store.json: read-only state entering this benchmark.
- history/: sanitized Meta-Harness history with candidate code, train/validation
  scores, and agent-visible rollouts from current and prior benchmarks.
- CONTRACT.md: mandatory interface, safety, and anti-overfitting constraints.

Requirements:
1. Inspect the history and state selectively and form one falsifiable hypothesis.
2. Make a focused, generalizable mechanism change to candidate.py.
3. Preserve CandidatePolicy and AgentHarness contracts and the fixed solver/tools.
4. Do not hardcode task IDs, benchmark answers, hidden evaluator behavior, or data.
5. Do not access grader, verifier, solution, private_test, secrets, or network.
6. Do not call AgentStream's built-in run_evolver.
7. Do not create or edit any file except candidate.py.

End with a short hypothesis and change summary. The controller, not you, evaluates
and scores the candidate.
"""


def run_claude_proposer(
    *,
    workspace: Path,
    model: str,
    base_model: str,
    claude_bin: str,
    timeout_seconds: int,
    benchmark: str,
    iteration: int,
    candidate_number: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    prompt = build_proposer_prompt(
        benchmark=benchmark,
        iteration=iteration,
        candidate_number=candidate_number,
        base_model=base_model,
    )
    command = [
        claude_bin,
        "--bare",
        "--safe-mode",
        "--no-session-persistence",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--allowedTools",
        *ALLOWED_TOOLS,
        "--disallowedTools",
        *DISALLOWED_TOOLS,
    ]

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            env=env or os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

    (workspace / "proposer_stdout.jsonl").write_text(stdout, encoding="utf-8")
    (workspace / "proposer_stderr.txt").write_text(stderr, encoding="utf-8")
    result_events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result_events.append(event)
    final_event = result_events[-1] if result_events else {}
    metadata = {
        "started_at": started_at,
        "duration_seconds": round(time.monotonic() - started, 3),
        "model": model,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "is_error": final_event.get("is_error"),
        "subtype": final_event.get("subtype"),
        "session_id": final_event.get("session_id"),
        "total_cost_usd": final_event.get("total_cost_usd"),
        "usage": final_event.get("usage"),
        "allowed_tools": ALLOWED_TOOLS,
        "disallowed_tools": DISALLOWED_TOOLS,
    }
    (workspace / "proposer_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
