# Harbor Meta-Harness pilot

`tasks/select-harness` is an outer task that asks `openai/gpt-5.4-mini` to improve `/app/harness.py`.
The outer agent has terminal access and a bounded `evaluate_harness` tool with four calls.
Each call downloads the current harness, validates it, smoke-tests it, then evaluates it once on `tasks/reconcile-ledger` with `openai/gpt-5.4-nano` on Modal.

The starting harness is deliberately weak (one terminal turn).
It should pass smoke and fail the target; a stronger inspect/validate-style harness can pass both.

## Trust boundary

Generated harness code is arbitrary Python, subject only to a minimum Harbor interface and static leakage checks.
It is never imported by the long-lived outer Harbor agent or controller.
`controller.py` runs as a short-lived subprocess, stages the source in a temporary directory, and lets the child Harbor job import it.
The evaluator rejects direct references to the target name, data, output, tests, verifier, solution, and task manifest.
It requires a successful `tasks/harness-smoke` run before a scored child run.

The frozen `EvaluationState` keeps the four-call budget and append-only history.
After the outer model stops, trusted agent code uploads `/app/evaluation_history.json`; the outer verifier rewards the best recorded target reward.

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

The controller copies `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from the environment when set.
Otherwise it reads the active profile in `~/.modal.toml` into only the child subprocess environment.
It never logs credentials.

## Local verification

```bash
uv run pytest -q
```

Docker is unavailable on this host, so local tests cover evaluator semantics without running a Harbor sandbox.
