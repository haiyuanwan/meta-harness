# AgentStream Sequential Meta-Harness

This integration runs a Meta-Harness search inside each contiguous benchmark
block of an AgentStream Sequential stream. A complete harness is the pair of
`candidate.py` and `harness_store.json`.

It differs from `integrations/agentstream_online` in three important ways:

- Claude Code runs after a complete train/validation candidate evaluation, not
  after every task;
- a syntactically valid proposal is evaluated but is not automatically
  promoted;
- the validation frontier selects one complete harness, which runs held-out
  test tasks once and is then passed to the next benchmark.

The full experiment contract is documented in
[`AGENTSTREAM_SEQUENTIAL_META_HARNESS_PLAN.md`](../../AGENTSTREAM_SEQUENTIAL_META_HARNESS_PLAN.md).

## Smoke command

The first executable target uses BFCL and Tau2 with four tasks per benchmark:

```bash
cd /mnt/public/users/wanhaiyuan/meta-harness

PYTHONPATH="$PWD:/mnt/public/users/wanhaiyuan/AgentStream/exgentic/src" \
python -m integrations.agentstream_sequential_meta.controller \
  --output-dir /tmp/agentstream-sequential-meta-smoke \
  --benchmarks bfcl,tau2 \
  --num-tasks 4 \
  --train-tasks 2 \
  --validation-tasks 1 \
  --test-tasks 1 \
  --iterations 1 \
  --candidates-per-iteration 1 \
  --seed 44 \
  --base-model anthropic/Claude-Opus-4.8-C \
  --proposer-model Claude-Opus-4.8-C \
  --env-file /mnt/public/users/wanhaiyuan/.secrets/jd-joyrouter.env
```

Pass `--iterations 0` for the no-outer-evolution Sequential control. Use
`--resume` with exactly the same experiment arguments after an interruption.

## Current implementation boundary

The controller implements deterministic splits, isolated candidate state,
batch proposal semantics, validation selection, private test, cross-benchmark
handoff, and safe benchmark-level restart. Fine-grained resume inside a
candidate evaluation and explicit candidate-defined state-schema migrations are
planned follow-ups; an interrupted benchmark currently restarts from its exact
incoming complete harness.

## Tests

```bash
EXGENTIC_LITELLM_CACHING=false \
PYTHONPATH="$PWD:/mnt/public/users/wanhaiyuan/AgentStream/exgentic/src" \
pytest -q integrations/agentstream_sequential_meta/tests
```
