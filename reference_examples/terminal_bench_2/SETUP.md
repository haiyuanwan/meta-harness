# Terminal-Bench 2 Setup

This directory contains the Terminal-Bench 2 reference experiment from the paper. The scripts default to Harbor's `runloop` environment and also support Modal:

```bash
uv run harbor run ... -d "terminal-bench@2.0" -e runloop
HARBOR_ENVIRONMENT=modal uv run bash scripts/run_eval.sh \
  agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

## Requirements

- `ANTHROPIC_API_KEY` for the default solver model.
- One sandbox backend: `RUNLOOP_API_KEY` for Runloop, or Modal authentication.
- Enough sandbox quota for the concurrency you plan to use.
- The `terminal-bench` package installed from this directory's `pyproject.toml`.
- Python 3.12.

Install:

```bash
cd reference_examples/terminal_bench_2
uv sync
```

## Environment Variables

The repo root ships `.env.example`, but the shell wrappers in `scripts/` source `.env` from this directory. In practice:

- Put the same `.env` file in this directory, or
- Export the needed variables in your shell before running the scripts.

The shipped `runloop` path requires both `ANTHROPIC_API_KEY` and `RUNLOOP_API_KEY`. For larger runs, `runloop` is the intended path.

For Modal, run `modal token new` once or set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Then set `HARBOR_ENVIRONMENT=modal`. Runloop remains the default.

The default model in this example is `anthropic/claude-opus-4-6`. Override `HARBOR_MODEL` only if you intentionally want a non-default config.

Default concurrency in this release is `50`. This setup is sensitive to Anthropic API throughput: API tier matters, sharing the same API key with other active projects makes runs slower, and many apparent failures at higher concurrency are actually timeout failures rather than reasoning failures.

## Task Set

- Dataset ID in the paper code: `terminal-bench@2.0`
- `meta_harness.py` defaults to the full TB2 dataset: 89 tasks, 2 search trials each
- `--full-eval` adds the optional 5-trial winner pass on the full dataset
- The paper submission metadata targeted `laude-institute/terminal-bench-2` commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`

## Smoke Check

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

Modal:

```bash
HARBOR_ENVIRONMENT=modal uv run bash scripts/run_eval.sh \
  agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

## Recommended Bring-Up Order

For new harness ideas, do not start with the default 89x2 search loop.

1. Smoke-test a candidate on `extract-elf`.
2. Run the cheaper 30-task `hard` subset while the idea is still unstable.
3. Move to `uv run python meta_harness.py --iterations 1` once the idea looks promising.

Hard-subset example:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness hard 1 50
```

The shell wrappers use `timeout` when available and fall back to `gtimeout` if GNU coreutils is installed on macOS. If neither command is present, the Harbor run still works but has no shell-level wall-clock timeout. `HARBOR_TIMEOUT_SECONDS` controls both the Python and shell layers and defaults to 28800 seconds (8 hours).

Provider-free checks, runnable independently:

```bash
uv run python -m unittest -v tests.test_meta_harness.AgentClassValidationTests
uv run python -m unittest -v tests.test_meta_harness.TimeoutWiringTests
```

## Sandbox Backend

`HARBOR_ENVIRONMENT` accepts `runloop` or `modal`; the default is `runloop`. Both are remote sandboxes. Modal does not require local Docker.

## Version Note

The direct TB2 example dependencies are pinned in `pyproject.toml`:

- `harbor[modal,runloop]==0.18.0`
- `litellm==1.84.10`
- `python-dotenv==1.2.2`
- `tenacity==9.1.4`
- `terminal-bench==0.2.18` from `harbor-framework/terminal-bench` commit `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b`

If you want a local lockfile for your environment, run `uv lock` from this directory.
