"""Native Claude Code proposer for per-task online evolution."""

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


def build_proposer_prompt(task_index: int, benchmark: str, task_id: str) -> str:
    return f"""You are evolving an online AgentStream harness after task {task_index}.

Setting:
- The next task will be a new test-time task. There is no train/validation split.
- You are not given the completed task's reward, grader output, ground truth, or aggregate score.
- The fixed solver model is Claude-Opus-4.8-C.
- You may improve only the harness code and persistent harness state.

Completed task metadata:
- benchmark: {benchmark}
- task_id: {task_id}

Files in this workspace:
- candidate.py: exact harness code used for the completed task.
- harness_store.json: persistent memory/skills after the completed task.
- evidence/trajectory.jsonl: full agent-visible action/observation trajectory.
- evidence/online_candidate_trace.txt: compact trajectory when available.
- evidence/agent.log and evidence/litellm_trace.jsonl: solver logs when available.
- CONTRACT.md: mandatory candidate interface and safety rules.

Prior score-free episodes are available under ../../episodes/. Inspect them selectively when useful.

Your job:
1. Diagnose concrete harness weaknesses using only the score-free experience.
2. Edit candidate.py to create the harness for the next unseen task.
3. Optionally edit harness_store.json to improve persistent memory or skills.
4. Preserve the required AgentHarness and CandidatePolicy contracts.
5. Do not add dependencies, network calls, subprocesses, grader access, or calls to run_evolver.
6. Keep changes focused and generalizable; do not encode benchmark answers or task-specific ground truth.

Make the edits directly. End with a short summary of the hypothesis and files changed.
"""


def run_claude_proposer(
    *,
    workspace: Path,
    model: str,
    claude_bin: str,
    timeout_seconds: int,
    task_index: int,
    benchmark: str,
    task_id: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run Claude Code in a score-free workspace and preserve raw events."""

    workspace = workspace.resolve()
    prompt = build_proposer_prompt(task_index, benchmark, task_id)
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

    duration = time.monotonic() - started
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
        "duration_seconds": round(duration, 3),
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
