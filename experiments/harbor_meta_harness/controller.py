"""Short-lived, trusted evaluator for generated Harbor harnesses.

Candidate code is never imported by the outer agent/controller.  This process
only stages it and launches Harbor child processes, where Harbor imports it.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).parent
TARGET_TASK = ROOT / "tasks" / "reconcile-ledger"
SMOKE_TASK = ROOT / "tasks" / "harness-smoke"
TARGET_MODEL = "openai/gpt-5.4-nano"
FORBIDDEN_REFERENCES = (
    "reconcile-ledger",
    "ledger.jsonl",
    "report.json",
    "/tests",
    "test_outputs",
    "verifier",
    "/solution",
    "task.toml",
)


@dataclass(frozen=True)
class ChildResult:
    reward: float
    summary: str
    job_dir: str


@dataclass(frozen=True)
class EvaluationResult:
    source_sha256: str
    accepted: bool
    reason: str
    smoke: ChildResult | None
    target: ChildResult | None


def validate_source(source: str) -> str | None:
    """Return a rejection reason, or None for the minimal safe interface."""
    lowered = source.lower()
    for forbidden in FORBIDDEN_REFERENCES:
        if forbidden in lowered:
            return f"forbidden benchmark reference: {forbidden}"
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return f"syntax error: {error.msg}"
    harness = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentHarness"
        ),
        None,
    )
    if harness is None:
        return "missing AgentHarness class"
    if not any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
        for node in harness.body
    ):
        return "AgentHarness must define async run"
    if not any(
        (isinstance(base, ast.Name) and base.id == "BaseAgent")
        or (isinstance(base, ast.Attribute) and base.attr == "BaseAgent")
        for base in harness.bases
    ):
        return "AgentHarness must inherit BaseAgent"
    return None


def modal_environment() -> dict[str, str]:
    """Copy Modal credentials into the child environment without logging them."""
    environment = os.environ.copy()
    if environment.get("MODAL_TOKEN_ID") and environment.get("MODAL_TOKEN_SECRET"):
        return environment
    config_path = Path.home() / ".modal.toml"
    if not config_path.exists():
        return environment
    config = tomllib.loads(config_path.read_text())
    profile = next(
        (
            value
            for value in config.values()
            if isinstance(value, dict) and value.get("active")
        ),
        next((value for value in config.values() if isinstance(value, dict)), {}),
    )
    if profile.get("token_id") and profile.get("token_secret"):
        environment["MODAL_TOKEN_ID"] = profile["token_id"]
        environment["MODAL_TOKEN_SECRET"] = profile["token_secret"]
    return environment


def command_for(
    task: Path, candidate_dir: Path, jobs_dir: Path, name: str
) -> list[str]:
    return [
        str(Path(sys.executable).with_name("harbor")),
        "run",
        "-p",
        str(task),
        "--agent",
        "candidate:AgentHarness",
        "-e",
        "modal",
        "-m",
        TARGET_MODEL,
        "-n",
        "1",
        "--n-attempts",
        "1",
        "--job-name",
        name,
        "--jobs-dir",
        str(jobs_dir),
    ]


def child_result(job_dir: Path) -> ChildResult:
    result_files = list(job_dir.glob("*/result.json"))
    if len(result_files) != 1:
        return ChildResult(0.0, "missing Harbor result", str(job_dir))
    try:
        result = json.loads(result_files[0].read_text())
        verifier = result.get("verifier_result") or {}
        reward = float(verifier["rewards"]["reward"])
        return ChildResult(reward, json.dumps(verifier, sort_keys=True), str(job_dir))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ChildResult(0.0, "malformed Harbor result", str(job_dir))


def run_child(
    task: Path, candidate_dir: Path, jobs_dir: Path, name: str
) -> ChildResult:
    job_dir = jobs_dir / name
    environment = modal_environment()
    environment["PYTHONPATH"] = (
        str(candidate_dir) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    completed = subprocess.run(
        command_for(task, candidate_dir, jobs_dir, name),
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        return ChildResult(0.0, f"Harbor exited {completed.returncode}", str(job_dir))
    return child_result(job_dir)


def evaluate_source(source_path: Path, jobs_dir: Path) -> EvaluationResult:
    source_path = source_path.resolve()
    jobs_dir = jobs_dir.resolve()
    source = source_path.read_text()
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    reason = validate_source(source)
    if reason:
        return EvaluationResult(source_sha256, False, reason, None, None)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="harbor-harness-") as temp_dir:
        candidate_dir = Path(temp_dir)
        (candidate_dir / "candidate.py").write_text(source)
        smoke = run_child(SMOKE_TASK, candidate_dir, jobs_dir, "smoke")
        if smoke.reward != 1:
            return EvaluationResult(
                source_sha256, False, "smoke test failed", smoke, None
            )
        target = run_child(TARGET_TASK, candidate_dir, jobs_dir, "target")
    return EvaluationResult(source_sha256, True, "scored", smoke, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_source(args.source, args.jobs_dir)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(asdict(result), indent=2) + "\n")


if __name__ == "__main__":
    main()
