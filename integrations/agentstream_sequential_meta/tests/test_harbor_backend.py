from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from integrations.agentstream_sequential_meta.controller import build_parser
from integrations.agentstream_sequential_meta.harbor_backend import (
    HarborOpenSandboxBackend,
    _memory_mb,
)
from integrations.agentstream_sequential_meta.harbor_executor import (
    HarborChunkResult,
    parse_snapshot_reference,
    write_task_definition,
)


def test_harbor_task_uses_distinct_snapshot_environments(tmp_path: Path) -> None:
    task = tmp_path / "task"
    write_task_definition(
        task,
        solver_snapshot_id="solver-id",
        grader_snapshot_id="grader-id",
        agent_timeout_sec=60,
        verifier_timeout_sec=30,
        cpus=2,
        memory_mb=4096,
    )

    config = (task / "task.toml").read_text(encoding="utf-8")
    assert 'docker_image = "snapshot:solver-id"' in config
    assert 'environment_mode = "separate"' in config
    assert 'docker_image = "snapshot:grader-id"' in config
    assert parse_snapshot_reference("snapshot:solver-id") == "solver-id"


def test_harbor_is_an_explicit_execution_backend() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "/tmp/output", "--execution-backend", "harbor"]
    )
    assert args.execution_backend == "harbor"
    assert _memory_mb("16Gi") == 16 * 1024


def test_harbor_verifier_failure_does_not_repeat_committed_solver(
    tmp_path: Path,
) -> None:
    backend = object.__new__(HarborOpenSandboxBackend)
    backend.settings = SimpleNamespace(
        domain="localhost",
        api_key="",
        protocol="http",
        use_server_proxy=False,
        request_timeout_sec=10,
        ready_timeout_sec=10,
        sandbox_timeout_sec=10,
        command_timeout_sec=10,
        cpus=1,
        memory="1Gi",
    )
    backend.provider_env = {"PYTHONPATH": "/opt/meta-harness"}
    backend.ensure_runtime = lambda benchmark, role: SimpleNamespace(  # type: ignore[method-assign]
        snapshot_id=f"{benchmark}-{role}"
    )
    calls = 0

    class FakeExecutor:
        def run_chunk(self, **kwargs):
            nonlocal calls
            calls += 1
            result = tmp_path / "solver-result"
            (result / "evaluation").mkdir(parents=True, exist_ok=True)
            (result / "harness_store.json").write_text(
                json.dumps({"count": 1}), encoding="utf-8"
            )
            (result / "result.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "task_id": "task-1",
                                "split": "test",
                                "status": "awaiting_grader",
                            }
                        ],
                        "grading_artifacts": [
                            {
                                "schema_version": 1,
                                "benchmark": "bfcl",
                                "task_id": "task-1",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            kwargs["on_solver_complete"](result)
            raise RuntimeError("verifier environment failed after solver commit")

    backend._harbor_executor = lambda: FakeExecutor()  # type: ignore[method-assign]
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate", encoding="utf-8")
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps({"count": 0}), encoding="utf-8")
    output = tmp_path / "output.json"

    block = backend.run_block(
        benchmark_slug="bfcl",
        task_ids=["task-1"],
        split_names=["test"],
        candidate_path=candidate,
        input_state_path=incoming,
        output_state_path=output,
        evaluation_dir=tmp_path / "evaluation-output",
        public_dir=None,
        config={},
        base_model="model",
        max_tokens=100,
        embedding_model="unused",
    )

    assert calls == 1
    assert json.loads(output.read_text()) == {"count": 1}
    assert block.rows[0]["status"] == "error"
