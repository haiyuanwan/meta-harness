# Domain Spec: Online Meta-Harness for AgentStream

## Domain Summary

The target is AgentStream's continual, cross-task agent evaluation. The first
setting is online-only: tasks arrive as unseen test-time episodes and there is
no train/validation phase exposed to the evolving harness. The evaluation unit
is one complete benchmark task, while the persistent harness state spans the
ordered task stream.

The initial stream is sequential: BFCL `multi_turn_base` tasks are followed by
Tau2 `telecom` tasks. The fixed solver/base model is
`anthropic/Claude-Opus-4.8-C`. The proposer is native Claude Code using the
same `Claude-Opus-4.8-C` route. Smoke tests on 2026-08-25 confirmed that this
route supports text, Anthropic tool use, LiteLLM tool use, Claude Code, and an
end-to-end BFCL task. The `Claude-Opus-4.8-C-ns` route is out of scope because
tool-use requests timed out.

The base model, benchmark task definitions, benchmark graders, and tool
surfaces are fixed. The evolvable object is the Python harness around the base
model: system-context construction, use of persistent memory and skills, tool
selection, and deterministic state updates. Model fine-tuning, benchmark
changes, new external tools, and exposing grader answers to the proposer are
out of scope.

The total optimization budget is currently `unknown`. The default pilot is two
BFCL tasks plus two Tau2 tasks, one proposer call after every task, one task at
a time, and no retries except for infrastructure failure.

## Harness and Search Plan

Every candidate is a Python file exporting:

- `AgentHarness`, a subclass of the integration's `OnlineHarnessAgent`;
- `CandidatePolicy`, a subclass of `CandidatePolicyBase`.

The stable integration reuses AgentStream's `HarnessAgentInstance` for model
calls, action conversion, trajectory logging, and `HarnessStore`. It overrides
the original `close()` evolution path, so AgentStream's built-in `run_evolver`
is not called. Candidate policy hooks may transform the system message, select
a subset of existing tools, and update persistent memory/skills at task close.
They may not fabricate tools or access benchmark graders.

After task `t`:

1. Save the exact candidate code and harness state used for task `t`.
2. Save the full agent-visible trajectory and LiteLLM trace.
3. Create a proposer workspace containing only score-free experience.
4. Ask Claude Code to edit the staged `candidate.py` and, optionally, the
   staged harness checkpoint for task `t+1`.
5. Compile/import the candidate, check inheritance and hook contracts, and
   validate that the checkpoint can be loaded.
6. Atomically promote both files if validation passes; otherwise retain the
   previous candidate and state.
7. Store the external benchmark score separately after proposal.

The first baseline is the unmodified AgentStream harness behavior with its
original solver loop, memory/skill injection, and all benchmark tools, but with
the original built-in evolver disabled. Useful stable helpers are candidate
loading/validation, state checkpointing, trajectory discovery, proposer
logging, atomic promotion, and resume bookkeeping.

## Evaluation Plan

The initial pilot is:

- mode: `sequential`;
- order: BFCL `multi_turn_base`, then Tau2 `telecom`;
- tasks: two per benchmark;
- concurrency: one;
- evolution cadence: once after every completed task;
- primary metric: cumulative task reward/success over the online stream;
- secondary metrics: per-benchmark cumulative reward, task status, steps,
  action count, input/output tokens, wall-clock time, proposer time, candidate
  validation failures, and rollback count.

There is deliberately no search-set selection or best-candidate filtering in
this setting. Candidate `H_t` handles task `t`; the score for task `t` is a
reported online outcome, not feedback supplied to the proposer. Claude Code
may inspect only the task, actions, observations, model interactions, prior
candidate code, and persistent state. Benchmark result files, ground truth,
aggregate metrics, and task reward are private.

This is an online AgentStream extension rather than the paper's standard
search-set Meta-Harness protocol. It measures test-time adaptation on a single
ordered stream. Noise is not yet measured (`unknown`); the initial pilot uses
one trial per task. A larger report should compare at least a no-evolution
baseline under the identical task order and seed.

## Experience and Logging

For every task, retain:

- candidate code and state before evolution;
- task identity, benchmark identity, stream position, and model names;
- Exgentic trajectory JSONL;
- full agent-side LiteLLM trace and agent log when present;
- Claude Code stdout/events, stderr, duration, and exit status;
- proposed code and state;
- validation report;
- promoted code and state, or the rollback decision;
- private benchmark metrics in a directory not included in proposer context;
- the original Exgentic session directory, including benchmark-specific
  scoring artifacts, in the private evaluation tree.

The run directory separates `public_history/` (proposer-visible, score-free)
from `private_evaluations/` and `private_metrics/`. `current/` holds the exact
candidate and checkpoint used for the next task. This preserves every code
version and trajectory while preventing accidental score injection through the
normal proposer prompt.

## Open Questions and Unknowns

- Total dollar/token/wall-clock budget: `unknown`; default to the four-task
  pilot before scaling.
- Final task counts and number of random seeds: `unknown`.
- Whether later experiments should compare sequential and interleaved modes:
  `unknown`; initial implementation prioritizes sequential mode.
- Online-evaluation noise and required repeat count: `unknown`; estimate after
  the pilot.
- The proposer/base model may be separated in a later ablation, but the first
  implementation intentionally uses the same fixed model route for both.
