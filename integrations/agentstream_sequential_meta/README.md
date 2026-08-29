# Continual Sequential Meta-Harness

This integration adopts AgentStream's Sequential research protocol while using
an independent benchmark-native runtime. A complete harness is the pair of
`candidate.py` and `harness_store.json`.

`candidate.py` is deliberately not an AgentStream agent. It implements the
small benchmark-neutral `CandidateHarnessBase` protocol and receives generic
tool schemas/results. Solver backends own only public tasks, tools, and stateful
execution; private graders are a separate package loaded only by verifier
workers. BFCL verification calls the official `bfcl_eval` checker package;
BrowseCompPlus verification calls its official citation and judge utilities.
Exgentic is not imported by the scored runtime.
The `harness_store.json` filename is retained for artifact compatibility; its
contents now follow this integration's small JSON schema and do not use
Exgentic's `HarnessStore` class.

It differs from `integrations/agentstream_online` in three important ways:

- Claude Code runs after a complete train/validation candidate evaluation, not
  after every task;
- a syntactically valid proposal is evaluated but is not automatically
  promoted;
- the validation frontier selects one complete harness, which runs held-out
  test tasks once; the pre-test winner checkpoint is passed to the next
  benchmark, while test-updated state remains a private analysis artifact.

The method, experiment contract, and implementation progress are documented in
[`continual_harness_progress.md`](../../docs/continual_harness_progress.md).

## OpenSandbox execution

OpenSandbox is the default execution backend. Each benchmark has two distinct
snapshots: `solver` and `grader`. A solver worker runs up to 10 sequential tasks,
exports minimal JSON artifacts, and is destroyed. Its updated harness state is
committed before a grader sandbox is restored and receives only those artifacts.
This follows Terminal-Bench/Harbor's late-verifier ordering. A grader failure
retries only verification and never reruns model actions or rolls back harness
state. A solver worker-level failure still retries its chunk from the committed
incoming checkpoint.

An opt-in `--execution-backend harbor` path uses the same prepared snapshots
but delegates each chunk's solver/separate-verifier lifecycle and artifact
transfer to Harbor 0.20. The continual controller still owns state checkpoints,
candidate selection, transfer evaluation, and HDA. Harbor's built-in
OpenSandbox environment accepts images but not snapshot IDs, so this integration
provides a small snapshot-backed environment extension. Keep `opensandbox` as
the direct fallback; the Harbor path passed a real three-task BFCL
solver/separate-verifier smoke on 2026-08-26.

The client stack is pinned to the same version used by SEA:

```bash
python -m pip install -r \
  integrations/agentstream_sequential_meta/requirements-opensandbox.txt
```

The defaults mirror SEA's current deployment (`10.119.212.249:8080`, HTTP,
server proxy). They can be overridden with `--opensandbox-domain`,
`--opensandbox-protocol`, `--[no-]opensandbox-use-server-proxy`, or the normal
`OPENSANDBOX_DOMAIN`/`OPENSANDBOX_API_KEY` environment variables.

Prepare BFCL and BrowseCompPlus dependency snapshots before a scored run:

```bash
PYTHONPATH="$PWD" \
python -m integrations.agentstream_sequential_meta.controller \
  --output-dir /tmp/not-used-during-prepare \
  --benchmarks bfcl,browsecompplus \
  --prepare-only \
  --execution-backend harbor \
  --opensandbox-runtime-mode auto \
  --env-file /mnt/public/users/wanhaiyuan/.secrets/jd-joyrouter.env
```

Snapshot identities include the benchmark, runtime role, uploaded meta-harness
source digest, base image, SDK versions, OpenSandbox server scope, and a
solver-recipe revision. The local
manifest stores snapshot IDs and non-secret metadata under
`.meta-harness/opensandbox-runtime-cache-v2.json`; model credentials are supplied
only when a scored sandbox is restored and are never included in the prepared
snapshot. Host `ALL_PROXY`/`HTTP_PROXY`/`HTTPS_PROXY` variables are passed
transiently to both dependency bootstrap and scored workers, so downloads and
model calls use the proxy without persisting proxy values in the snapshot. The
uploaded source bundles are role-specific and allowlisted. In particular, the
solver bundle contains neither `benchmark_graders` nor the grader worker.
Repository traces, tests, Git metadata, secret baselines, and local caches are
excluded.

The BrowseCompPlus solver runtime installs CPU-only PyTorch and pins its native
retrieval package to commit
`7cd697e133ba9150c3c310d10043e327d9f06c41`; the official package is installed
without its broad optional dependency set, and the FAISS-path dependencies are
installed explicitly. Tevatron is pinned to
`dd063104c81a76d6a77c845f667b46b9e5abd625`. It does not install Exgentic
agents or memory implementations. Its task manifest contains only `query_id`
and `query`; answers/evidence remain in the separate grader snapshot.

For BFCL, bootstrap fetches the pinned Gorilla/BFCL commit directly and installs
the official package. The solver snapshot then deletes
`bfcl_eval/data/possible_answer`; only the grader snapshot retains it.

Use `--opensandbox-runtime-mode require` for scored experiments. It fails
before task/model execution if a matching runtime is absent. `auto` prepares a
missing runtime, while `rebuild` explicitly creates a new one.

## Smoke command

The first executable target uses BFCL and BrowseCompPlus with four tasks per benchmark:

```bash
cd /mnt/public/users/wanhaiyuan/meta-harness

PYTHONPATH="$PWD" \
python -m integrations.agentstream_sequential_meta.controller \
  --output-dir /tmp/agentstream-sequential-meta-smoke \
  --benchmarks bfcl,browsecompplus \
  --num-tasks 4 \
  --train-tasks 2 \
  --validation-tasks 1 \
  --test-tasks 1 \
  --iterations 1 \
  --candidates-per-iteration 1 \
  --seed 44 \
  --base-model anthropic/Claude-Opus-4.6-hq \
  --proposer-model Claude-Opus-4.6-hq \
  --execution-backend opensandbox \
  --opensandbox-runtime-mode require \
  --env-file /mnt/public/users/wanhaiyuan/.secrets/jd-joyrouter.env
```

Pass `--iterations 0` for the no-outer-evolution Sequential control. Use
`--resume` with exactly the same experiment arguments after an interruption.
The old host/venv behavior remains available only when explicitly requested
with `--execution-backend local`; use `--execution-backend opensandbox` for the
direct two-stage fallback without Harbor trial artifacts. Local mode preserves late call ordering for smoke
tests, but only OpenSandbox provides process/filesystem separation.

For the preregistered transfer/HDA partition, pass
`--partition-profile transfer-hda`. This enumerates and fingerprints both full
task pools, evolves only on 20 search + 10 validation tasks per benchmark, and
writes H0/H1/H2 without consuming hidden tasks. Add
`--run-transfer-matrix` only when you intentionally want to pay for the full
six-cell hidden evaluation and 10,000-sample paired bootstrap gate.

## Current implementation boundary

The controller implements the generic harness protocol, benchmark-native tool
environments, score-to-memory isolation, deterministic splits, candidate isolation,
batch proposal semantics, validation selection, private test, cross-benchmark
handoff, atomic state writes, and bounded solver/grader retries. OpenSandbox
creates a fresh solver sandbox for each bounded task chunk, commits returned
JSON state after solver completion, destroys the solver, and then creates an
independent grader sandbox. `--sandbox-tasks-per-worker` controls the
chunk size (default 10; use 1 for strict per-task sandbox isolation). Persisting
and resuming the exact task cursor across a controller process restart remains
the next reliability step before a large scored run. The six-cell hidden
transfer matrix, stream-level paired-bootstrap gate, HDA review manifest, and
self-contained HTML report are wired behind the explicit
`--run-transfer-matrix` flag. Controlled B_cc and E_neutral variants are not
constructed automatically: the report stops at HDA approval checkpoint 1, as
required by the method. Official checker/judge payloads and full error text are
stored only under `private_evaluation/`; proposer-visible `public/` artifacts
contain agent-visible rollouts, scores, usage, and sanitized error types.

## Tests

```bash
PYTHONPATH="$PWD" \
pytest -q integrations/agentstream_sequential_meta/tests
```
