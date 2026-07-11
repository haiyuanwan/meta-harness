# Harbor Meta-Harness pilot

This directory contains a pilot for running a meta-harness loop on a collection of Harbor tasks.
An outer agent edits a seed harness.
A trusted controller scores each version on an explicit task suite.
Each task declares strings that the harness may not reference.

`tasks/select-harness` is an outer task that asks an outer model to improve `/app/harness.py`.
The outer agent has terminal access and a bounded `evaluate_harness` tool.
Each call downloads the current harness and checks its source against the leakage rules.
It then scores the harness on `suite.toml`, or on another suite passed through `--suite`.

## Add a Harbor collection

1. Put Harbor tasks under `tasks/` or download them under `datasets/`.
2. Set `[meta_harness].forbidden_references` in each `task.toml`. Include the task name, input and output filenames, and other strings that would leak task details.
3. Write `suite.toml` or `suites/<name>.toml`. List the task paths and set `child_model`, `eval_budget`, `seed_harness`, and `aggregate`. The supported aggregates are `mean`, `min`, and `fraction_solved`.
4. Pass the suite to the controller with `--suite`, then run the outer `select-harness` loop.

Probe each collection before using it.
The seed harness should leave room for improvement under the child model.

## Suites

Default `suite.toml` is the handmade three-task suite (`reconcile-ledger`, `dedupe-events`, `budget-rollups`).

The following probe scores came from local runs with `openai/gpt-5.4-nano`.
They are exploratory results from the same tasks used during optimization.

| Suite | Source | Tasks | Seed / best observed |
|-------|--------|------:|-------------|
| `suites/tb2-easy.toml` | TB2 Easy | 4 | 0.0 → 0.5 |
| `suites/humanevalfix-lite.toml` | HumanEvalFix lite | 10 | 0.0 → 0.7 |
| `suites/codepde.toml` | CodePDE (no `cns1d`) | 4 | 0.0 → ≥0.75; child timeout 900s |

Set `META_HARNESS_SUITE=suites/tb2-easy.toml` when launching the outer loop.
Each collection suite uses `eval_budget = 8`; CodePDE and TB2 Easy also set `child_timeout_sec`.

Fetch and prepare:

```bash
bash scripts/fetch_collections.sh
```

`datasets/` is gitignored.
Suite paths are relative to this directory.

A small universal denylist (`/tests`, `verifier`, `/solution`, `task.toml`, `test_outputs`) always applies.

The starting harness uses one terminal turn.
It scores 0 on the default handmade suite.
An inspect-and-validate harness can score 1 on all three tasks.

## Trust boundary

Generated harness code is arbitrary Python.
It must implement the Harbor interface and pass the leakage checks.
It is never imported by the long-lived outer Harbor agent or controller.
`controller.py` runs as a short-lived subprocess, stages the source in a temporary directory, and lets the child Harbor job import it.

The frozen `EvaluationState` tracks the suite budget and an append-only history.
After the outer model stops, trusted agent code uploads `/app/evaluation_history.json`.
The outer verifier returns the best suite reward in that history.

## Run

```bash
uv sync
uv run harbor run \
  -p tasks/select-harness \
  --agent agents.meta_harness:AgentHarness \
  -e modal \
  -m openai/gpt-5.4-mini \
  --agent-timeout-multiplier 3
```

Point the controller and outer agent at another suite with:

```bash
export META_HARNESS_SUITE=suites/tb2-easy.toml
```

You can also pass `--suite suites/tb2-easy.toml` to `controller.py`.

The controller copies `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from the environment when they are set.
If they are absent, it reads the active profile from `~/.modal.toml`.
The credentials are passed only to the child subprocess.
It never logs credentials.

## Local verification

```bash
uv run pytest -q
```

Local tests cover evaluator behavior without running a Harbor sandbox.
