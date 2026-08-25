# Online Meta-Harness for AgentStream

This integration runs AgentStream one task at a time and invokes native Claude
Code after every completed task. Candidate code and persistent memory/skills
carry across the ordered stream. Benchmark rewards and grader artifacts are
stored privately and are not copied into proposer workspaces.

## What evolves

`candidate.py` exports two objects:

- `AgentHarness`, the AgentStream agent configuration;
- `CandidatePolicy`, hooks for system-context construction, selecting a subset
  of existing benchmark tools, and deterministic memory/skill updates.

The stable layer reuses AgentStream's `HarnessAgentInstance` but overrides its
`close()` method. AgentStream's original `run_evolver()` is never invoked.
Claude Code is the only outer code proposer.

## Run layout

```text
run/
├── current/                 # exact candidate + state for the next task
├── public_history/
│   ├── episodes/            # all accepted versions and score-free traces
│   └── workspaces/          # raw Claude Code sessions and proposals
├── private_evaluations/     # complete Exgentic sessions and grader files
├── private_metrics/         # rewards and aggregate metrics
├── experiment.json
└── progress.json
```

Claude Code receives only `public_history/`. Each proposal is compiled,
imported, contract-checked, and checkpoint-checked. Valid proposals are
atomically promoted; invalid or timed-out proposals leave `current/` unchanged.

## Environment

Run from a Python environment containing AgentStream/Exgentic and the selected
benchmark dependencies. Because candidates change between tasks, the initial
implementation uses Exgentic's `direct` runner. BFCL and Tau2 therefore need to
be installed into that same environment (the Exgentic `--local` install mode).

The controller accepts a dotenv file and maps:

```text
JOYROUTER_API_KEY  -> ANTHROPIC_API_KEY
JOYROUTER_BASE_URL -> ANTHROPIC_BASE_URL
```

Keys are never written to run artifacts.

## One-task smoke

```bash
cd /mnt/public/users/wanhaiyuan/meta-harness

PYTHONPATH="$PWD:/mnt/public/users/wanhaiyuan/AgentStream/exgentic/src" \
python -m integrations.agentstream_online.controller \
  --output-dir /tmp/agentstream-online-smoke \
  --benchmarks bfcl \
  --num-tasks 1 \
  --seed 42 \
  --env-file /mnt/public/users/wanhaiyuan/.secrets/jd-joyrouter.env
```

## Four-task pilot

```bash
PYTHONPATH="$PWD:/mnt/public/users/wanhaiyuan/AgentStream/exgentic/src" \
python -m integrations.agentstream_online.controller \
  --output-dir /tmp/agentstream-online-pilot \
  --benchmarks bfcl,tau2 \
  --num-tasks 2 \
  --seed 42 \
  --env-file /mnt/public/users/wanhaiyuan/.secrets/jd-joyrouter.env
```

Use `--resume` with the same arguments to continue a stopped run. The task
order is persisted and checked before resuming.

## Tests

```bash
PYTHONPATH="$PWD:/mnt/public/users/wanhaiyuan/AgentStream/exgentic/src" \
pytest -q integrations/agentstream_online/tests
```
