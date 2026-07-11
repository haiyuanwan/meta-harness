# Harbor Meta-Harness pilot

This directory is a reusable pattern for packaging any collection of Harbor tasks into a meta-harness loop: an outer agent edits a seed harness, a trusted controller scores it on an explicit task suite, and leakage is declared per task.

`tasks/select-harness` is an outer task that asks an outer model to improve `/app/harness.py`.
The outer agent has terminal access and a bounded `evaluate_harness` tool.
Each call downloads the current harness, validates it against per-task leakage rules, then scores it on the task suite in `suite.toml` (or another suite file via `--suite`).

## Pattern: any Harbor collection → meta-harness suite

1. Put Harbor tasks on disk (handwritten under `tasks/`, or downloaded under `datasets/`).
2. For each task, set `[meta_harness].forbidden_references` in `task.toml` (task name, input/output filenames, other leak strings).
3. Write a `suite.toml` (or `suites/<name>.toml`) with an explicit `tasks = [...]` path list, plus `child_model`, `eval_budget`, `seed_harness`, and `aggregate` (`mean` / `min` / `fraction_solved`).
4. Point the controller at that suite (`--suite ...`) and run the outer `select-harness` loop.

The same outer agent, controller, and trust boundary work for any suite that follows this shape.
Prefer suites where a weak seed harness fails and a stronger harness can pass under the child model; probe that gap before relying on a collection.

## Suites

Default `suite.toml` is the handmade three-task suite (`reconcile-ledger`, `dedupe-events`, `budget-rollups`).

Shipped optimization collections (measured weak→strong under nano; opt runs confirmed):

| Suite | Source | Tasks | Probe / opt |
|-------|--------|------:|-------------|
| `suites/tb2-easy.toml` | TB2 Easy | 4 | 0.0 → 0.5 |
| `suites/humanevalfix-lite.toml` | HumanEvalFix lite | 10 | 0.0 → 0.7 |
| `suites/codepde.toml` | CodePDE (no `cns1d`) | 4 | 0.0 → ≥0.75; child timeout 900s |

Set `META_HARNESS_SUITE=suites/tb2-easy.toml` (etc.) when launching the outer loop.
Each collection suite uses `eval_budget = 8`; CodePDE and TB2 Easy also set `child_timeout_sec`.

Fetch and prepare:

```bash
bash scripts/fetch_collections.sh
```

`datasets/` is gitignored. Suite paths are relative to the experiment root.

Configure each suite with an explicit task path list, child model, eval budget, seed harness, and aggregate (`mean` default; also `min`, `fraction_solved`).
Each Harbor task declares `[meta_harness].forbidden_references` in `task.toml`.
A small universal denylist (`/tests`, `verifier`, `/solution`, `task.toml`, `test_outputs`) always applies.

The starting harness is deliberately weak (one terminal turn).
On the default handmade suite it scores 0; a stronger inspect/validate-style harness can score 1 on all three.

## Trust boundary

Generated harness code is arbitrary Python, subject only to a minimum Harbor interface and leakage checks.
It is never imported by the long-lived outer Harbor agent or controller.
`controller.py` runs as a short-lived subprocess, stages the source in a temporary directory, and lets the child Harbor job import it.

The frozen `EvaluationState` keeps the suite eval budget and append-only history.
After the outer model stops, trusted agent code uploads `/app/evaluation_history.json`; the outer verifier rewards the best recorded suite reward.

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

Point the controller / outer agent at another suite with:

```bash
export META_HARNESS_SUITE=suites/tb2-easy.toml
```

Or pass `--suite suites/tb2-easy.toml` to `controller.py` directly.

The controller copies `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from the environment when set.
Otherwise it reads the active profile in `~/.modal.toml` into only the child subprocess environment.
It never logs credentials.

## Local verification

```bash
uv run pytest -q
```

Docker is unavailable on this host, so local tests cover evaluator semantics without running a Harbor sandbox.
