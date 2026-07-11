#!/usr/bin/env python3
"""Prepare Harbor collection suites: inject per-task leakage and write suite.toml files."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SUITES = ROOT / "suites"
UNIVERSAL = ("/tests", "test_outputs", "verifier", "/solution", "task.toml")

# Suites probed with openai/gpt-5.4-nano.
# tb2-sample, legacy-bench, and compilebench did not improve in probes.
COLLECTIONS: dict[str, dict] = {
    "tb2-easy": {
        "tasks": [
            "datasets/terminal-bench-2/cobol-modernization",
            "datasets/terminal-bench-2/fix-git",
            "datasets/terminal-bench-2/overfull-hbox",
            "datasets/terminal-bench-2/prove-plus-comm",
        ],
        "eval_budget": 8,
        "child_timeout_sec": 600,
        "comment": "Active optimization suite: TB2 Easy (probe gap 0.0 -> 0.5).",
    },
    "humanevalfix-lite": {
        "tasks": [
            "datasets/humanevalfix/python-0",
            "datasets/humanevalfix/python-1",
            "datasets/humanevalfix/python-10",
            "datasets/humanevalfix/python-100",
            "datasets/humanevalfix/python-101",
            "datasets/humanevalfix/python-102",
            "datasets/humanevalfix/python-103",
            "datasets/humanevalfix/python-104",
            "datasets/humanevalfix/python-105",
            "datasets/humanevalfix/python-106",
        ],
        "eval_budget": 8,
        "comment": "Active optimization suite: HumanEvalFix lite (probe gap 0.0 -> 0.7).",
    },
    "codepde": {
        # Drop cns1d: hung under Modal during probe.
        "tasks": [
            "datasets/codepde/codepde-advection",
            "datasets/codepde/codepde-burgers",
            "datasets/codepde/codepde-darcy",
            "datasets/codepde/codepde-reacdiff1d",
        ],
        "eval_budget": 8,
        "child_timeout_sec": 900,
        "comment": "Active optimization suite: CodePDE without cns1d (hung under Modal).",
    },
}


def environment_names(task: Path) -> list[str]:
    env = task / "environment"
    if not env.is_dir():
        return []
    names: list[str] = []
    for path in env.rglob("*"):
        if path.is_file() and path.name not in {
            "Dockerfile",
            "docker-compose.yaml",
            "docker-compose.yml",
        }:
            names.append(path.name)
    return sorted(set(names))


def ensure_forbidden(task: Path) -> None:
    task_toml = task / "task.toml"
    text = task_toml.read_text()
    raw = tomllib.loads(text)
    existing = list((raw.get("meta_harness") or {}).get("forbidden_references") or [])
    forbidden = [task.name, *environment_names(task), *existing]
    # Keep stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in forbidden:
        if item and item not in seen and item not in UNIVERSAL:
            seen.add(item)
            ordered.append(item)
    block = ["", "[meta_harness]", "forbidden_references = ["]
    block.extend(f'  "{item}",' for item in ordered)
    block.append("]")
    if "[meta_harness]" in text:
        # Replace existing section body by rewriting file without that table, then append.
        lines = text.splitlines()
        kept: list[str] = []
        skipping = False
        for line in lines:
            if line.strip().startswith("[") and line.strip().endswith("]"):
                skipping = line.strip() == "[meta_harness]"
            if skipping:
                continue
            kept.append(line)
        text = "\n".join(kept).rstrip() + "\n"
    task_toml.write_text(text.rstrip() + "\n" + "\n".join(block) + "\n")


def write_suite(name: str, cfg: dict) -> None:
    SUITES.mkdir(parents=True, exist_ok=True)
    tasks = cfg["tasks"]
    body = [
        f"# {cfg['comment']}",
        "tasks = [",
        *[f'  "{task}",' for task in tasks],
        "]",
        "",
        'child_model = "openai/gpt-5.4-nano"',
        'environment = "modal"',
        "n_attempts = 1",
        f"eval_budget = {cfg['eval_budget']}",
        'seed_harness = "tasks/select-harness/environment/harness.py"',
        'aggregate = "mean"',
    ]
    if "child_timeout_sec" in cfg:
        body.append(f"child_timeout_sec = {cfg['child_timeout_sec']}")
    body.append("")
    (SUITES / f"{name}.toml").write_text("\n".join(body))


def main() -> None:
    for name, cfg in COLLECTIONS.items():
        paths = [(ROOT / task).resolve() for task in cfg["tasks"]]
        missing = [str(path) for path in paths if not (path / "task.toml").exists()]
        if missing:
            raise FileNotFoundError(f"{name} missing tasks: {missing}")
        for path in paths:
            ensure_forbidden(path)
        write_suite(name, cfg)
        print(f"{name}: {len(cfg['tasks'])} tasks -> suites/{name}.toml")


if __name__ == "__main__":
    main()
