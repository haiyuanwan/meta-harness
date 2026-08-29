from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from integrations.agentstream_sequential_meta.opensandbox_backend import (
    OpenSandboxBackend,
    OpenSandboxBackendError,
    OpenSandboxSettings,
    _safe_extract,
    runtime_identity,
    safe_remote_path,
    sequential_task_order,
    source_digest,
    _iter_source_files,
)


def test_sequential_task_order_matches_fixed_selection_then_ordering() -> None:
    tasks = {
        "browsecompplus": [f"q{i}" for i in range(20)],
        "bfcl": [f"b{i}" for i in range(20)],
    }

    first = sequential_task_order(tasks, num_tasks=4, ordering_seed=44)
    second = sequential_task_order(tasks, num_tasks=4, ordering_seed=44)

    assert first == second
    assert [benchmark for benchmark, _ in first] == ["bfcl"] * 4 + [
        "browsecompplus"
    ] * 4
    assert len({task for _, task in first}) == 8


def test_source_digest_ignores_runtime_cache_and_git(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    agentstream = tmp_path / "agentstream"
    (meta / ".meta-harness").mkdir(parents=True)
    (meta / ".git").mkdir()
    integration = meta / "integrations" / "agentstream_sequential_meta"
    integration.mkdir(parents=True)
    agentstream_source = agentstream / "src" / "exgentic"
    agentstream_source.mkdir(parents=True)
    (integration / "candidate.py").write_text("one", encoding="utf-8")
    (agentstream_source / "worker.py").write_text("two", encoding="utf-8")
    before = source_digest(meta)

    (meta / ".meta-harness" / "cache.json").write_text("changed", encoding="utf-8")
    (meta / ".git" / "index").write_text("changed", encoding="utf-8")

    assert source_digest(meta) == before

    (agentstream_source / "worker.py").write_text("reference changed", encoding="utf-8")
    assert source_digest(meta) == before


def test_source_digest_ignores_unuploaded_reference_files(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    reference = tmp_path / "reference"
    integration = meta / "integrations" / "agentstream_sequential_meta"
    integration.mkdir(parents=True)
    source = reference / "src" / "exgentic"
    source.mkdir(parents=True)
    (integration / "candidate.py").write_text("runtime", encoding="utf-8")
    (source / "agent.py").write_text("runtime", encoding="utf-8")
    before = source_digest(meta)

    misc = reference / "misc" / "utils"
    misc.mkdir(parents=True)
    (misc / ".secrets.baseline").write_text("not-uploaded", encoding="utf-8")

    assert source_digest(meta) == before


def test_runtime_identity_changes_with_benchmark(tmp_path: Path) -> None:
    settings = OpenSandboxSettings(runtime_cache_path=tmp_path / "cache.json")

    bfcl = runtime_identity(benchmark="bfcl", source_hash="a" * 64, settings=settings)
    browse = runtime_identity(
        benchmark="browsecompplus", source_hash="a" * 64, settings=settings
    )

    assert bfcl != browse


def test_runtime_identity_changes_with_solver_grader_role(tmp_path: Path) -> None:
    settings = OpenSandboxSettings(runtime_cache_path=tmp_path / "cache.json")

    solver = runtime_identity(
        benchmark="bfcl", role="solver", source_hash="a" * 64, settings=settings
    )
    grader = runtime_identity(
        benchmark="bfcl", role="grader", source_hash="a" * 64, settings=settings
    )

    assert solver != grader


def test_runtime_identity_changes_with_recipe_revision(tmp_path: Path) -> None:
    settings = OpenSandboxSettings(runtime_cache_path=tmp_path / "cache.json")
    legacy = runtime_identity(
        benchmark="bfcl", role="solver", source_hash="a" * 64, settings=settings
    )
    revised = runtime_identity(
        benchmark="bfcl",
        role="solver",
        source_hash="a" * 64,
        settings=settings,
        recipe_revision="solver-v2-litellm",
    )
    assert legacy != revised


def test_solver_source_archive_excludes_private_grader_package() -> None:
    root = Path(__file__).parents[3]
    names = {
        relative.as_posix()
        for _, relative in _iter_source_files("meta-harness", root, "solver")
    }

    assert not any("benchmark_graders" in name for name in names)
    assert not any(name.endswith("sandbox_grader_worker.py") for name in names)
    assert not any(name.endswith("grading.py") for name in names)
    assert any(name.endswith("sandbox_worker.py") for name in names)


def test_source_iterator_prunes_virtualenv(tmp_path: Path) -> None:
    package = tmp_path / "integrations" / "agentstream_sequential_meta"
    package.mkdir(parents=True)
    (package / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
    hidden = tmp_path / ".venv" / "integrations" / "agentstream_sequential_meta"
    hidden.mkdir(parents=True)
    (hidden / "candidate.py").write_text("VALUE = 2\n", encoding="utf-8")

    files = [relative for _, relative in _iter_source_files("meta-harness", tmp_path)]
    assert files == [Path("integrations/agentstream_sequential_meta/candidate.py")]


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"escape"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(OpenSandboxBackendError, match="unsafe path"):
        _safe_extract(archive, tmp_path / "result")


def test_remote_path_validation() -> None:
    assert str(safe_remote_path("/work/result.json")) == "/work/result.json"
    with pytest.raises(ValueError):
        safe_remote_path("../result.json")


def test_settings_reject_invalid_runtime_mode(tmp_path: Path) -> None:
    settings = OpenSandboxSettings(
        runtime_mode="fallback", runtime_cache_path=tmp_path / "cache.json"
    )
    with pytest.raises(ValueError, match="runtime_mode"):
        settings.validate()


def test_runtime_bootstrap_does_not_install_exgentic() -> None:
    backend = object.__new__(OpenSandboxBackend)

    for benchmark in ("bfcl", "browsecompplus"):
        command = backend._bootstrap_command(benchmark)
        assert "https://deb.debian.org" in command
        assert "import exgentic" not in command
        assert "/opt/agentstream" not in command
        assert "pip install --no-cache-dir -e '/opt/agentstream" not in command

    browse = backend._bootstrap_command("browsecompplus")
    assert "7cd697e133ba9150c3c310d10043e327d9f06c41" in browse
    assert "dd063104c81a76d6a77c845f667b46b9e5abd625" in browse
    assert "--no-deps 'git+https://github.com/lilacheden/BrowseComp-Plus/" in browse
    assert "pyserini" not in browse
    assert "'pillow>=12.1.1'" in browse
    assert "'peft>=0.16.0'" in browse
    assert "from searcher.searchers.faiss_searcher import FaissSearcher" in browse
    assert "python /opt/meta-harness/integrations/agentstream_sequential_meta/benchmark_backends/prepare_browsecompplus.py" in browse
    assert "@mac-support-and-packaging" not in browse
    assert "rm -f /opt/benchmark-assets/browsecompplus/data/browsecomp_plus_decrypted.jsonl" in browse
    assert "browsecomp_plus_grader.jsonl" in browse

    bfcl_solver = backend._bootstrap_command("bfcl", "solver")
    bfcl_grader = backend._bootstrap_command("bfcl", "grader")
    assert "pip install --no-cache-dir litellm" in bfcl_solver
    assert "pip install --no-cache-dir litellm" not in bfcl_grader
    assert "rm -rf /opt/benchmark-packages/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer" in bfcl_solver
    assert "rm -rf /opt/benchmark-packages/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer" not in bfcl_grader


def test_bootstrap_env_contains_proxy_but_not_model_credentials() -> None:
    backend = object.__new__(OpenSandboxBackend)
    backend.provider_env = {
        "HTTPS_PROXY": "http://proxy.internal",
        "ALL_PROXY": "socks5://proxy.internal",
        "http_proxy": "http://proxy.internal",
        "ANTHROPIC_API_KEY": "secret",
        "PYTHONPATH": "/opt/meta-harness",
    }

    assert backend._bootstrap_env() == {
        "ALL_PROXY": "socks5://proxy.internal",
        "HTTPS_PROXY": "http://proxy.internal",
        "http_proxy": "http://proxy.internal",
    }


def test_runtime_env_preserves_all_supported_proxy_forms() -> None:
    runtime = OpenSandboxBackend._runtime_env(
        {
            "ALL_PROXY": "socks5://proxy.internal",
            "HTTPS_PROXY": "http://proxy.internal",
            "UNRELATED_SECRET": "drop-me",
        }
    )

    assert runtime["ALL_PROXY"] == "socks5://proxy.internal"
    assert runtime["HTTPS_PROXY"] == "http://proxy.internal"
    assert "UNRELATED_SECRET" not in runtime


def test_task_discovery_does_not_receive_model_credentials() -> None:
    backend = object.__new__(OpenSandboxBackend)
    backend.provider_env = {
        "ANTHROPIC_API_KEY": "secret",
        "ANTHROPIC_BASE_URL": "https://model.internal",
        "HTTPS_PROXY": "http://proxy.internal",
        "PYTHONPATH": "/opt/meta-harness",
    }

    discovery = backend._worker_env("list-tasks")
    scored = backend._worker_env("run-block")

    assert "ANTHROPIC_API_KEY" not in discovery
    assert "ANTHROPIC_BASE_URL" not in discovery
    assert discovery["HTTPS_PROXY"] == "http://proxy.internal"
    assert scored["ANTHROPIC_API_KEY"] == "secret"


def test_run_block_reuses_one_worker_for_task_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    backend = object.__new__(OpenSandboxBackend)
    requests = []

    def fake_run_worker(**kwargs):
        request = kwargs["request"]
        requests.append(request)
        remote = tmp_path / f"remote-{len(requests)}"
        result = remote / "result"
        (result / "evaluation").mkdir(parents=True)
        incoming = json.loads(Path(kwargs["state_path"]).read_text())
        incoming["count"] += len(request["task_ids"])
        (result / "harness_store.json").write_text(json.dumps(incoming))
        rows = [
            {
                "task_id": task_id,
                "split": split,
                "score": 1.0,
                "success": True,
                "status": "success",
            }
            for task_id, split in zip(
                request["task_ids"], request["split_names"], strict=True
            )
        ]
        return {"rows": rows}, remote

    monkeypatch.setattr(backend, "_run_worker", fake_run_worker)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# candidate")
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps({"count": 0}))
    output = tmp_path / "output.json"
    task_ids = [f"q{i}" for i in range(23)]

    block = backend.run_block(
        benchmark_slug="browsecompplus",
        task_ids=task_ids,
        split_names=["train"] * len(task_ids),
        candidate_path=candidate,
        input_state_path=incoming,
        output_state_path=output,
        evaluation_dir=tmp_path / "evaluation",
        public_dir=None,
        config={
            "sandbox_tasks_per_worker": 10,
            "grader_kwargs": {"grader_model": "private-grader-model"},
        },
        base_model="model",
        max_tokens=100,
        embedding_model="unused",
    )

    assert [len(request["task_ids"]) for request in requests] == [10, 10, 3]
    assert all(request["task_attempts"] == 3 for request in requests)
    assert all("grader_kwargs" not in request["config"] for request in requests)
    assert len(block.rows) == 23
    assert all(row["sandbox_attempts"] == 1 for row in block.rows)
    assert json.loads(output.read_text()) == {"count": 23}


def test_run_block_commits_chunk_state_and_public_rows(
    tmp_path: Path,
) -> None:
    backend = object.__new__(OpenSandboxBackend)
    calls: list[list[str]] = []

    def fake_worker(**kwargs):
        request = kwargs["request"]
        task_ids = request["task_ids"]
        calls.append(task_ids)
        temporary = tmp_path / f"remote-{len(calls)}"
        result = temporary / "result"
        for task_id in task_ids:
            (result / "evaluation" / "sessions" / task_id).mkdir(
                parents=True
            )
            (result / "public" / "rollouts" / task_id).mkdir(parents=True)
        state = json.loads(Path(kwargs["state_path"]).read_text(encoding="utf-8"))
        state["session_count"] += len(task_ids)
        (result / "harness_store.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        rows = [
            {
                "task_id": task_id,
                "split": split,
                "score": 1.0,
                "success": True,
                "status": "success",
                "steps": 1,
                "action_count": 1,
                "agent_cost": 0.0,
                "execution_time": 0.1,
                "input_tokens": 1,
                "output_tokens": 1,
                **(
                    {"error": "PrivateBackendError: hidden detail"}
                    if task_id == "b"
                    else {}
                ),
            }
            for task_id, split in zip(
                task_ids, request["split_names"], strict=True
            )
        ]
        return {"rows": rows}, temporary

    backend._run_worker = fake_worker  # type: ignore[method-assign]
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate", encoding="utf-8")
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps({"session_count": 0}), encoding="utf-8")

    block = backend.run_block(
        benchmark_slug="bfcl",
        task_ids=["a", "b"],
        split_names=["train", "validation"],
        candidate_path=candidate,
        input_state_path=incoming,
        output_state_path=tmp_path / "out" / "state.json",
        evaluation_dir=tmp_path / "evaluation",
        public_dir=tmp_path / "public",
        config={},
        base_model="fixed",
        max_tokens=10,
        embedding_model="unused",
    )

    assert calls == [["a", "b"]]
    assert json.loads(block.state_path.read_text())["session_count"] == 2
    assert [row["sandbox_attempts"] for row in block.rows] == [1, 1]
    metrics = json.loads((tmp_path / "public" / "metrics.json").read_text())
    assert [row["task_id"] for row in metrics["tasks"]] == ["a", "b"]
    assert "error" not in metrics["tasks"][1]
    assert metrics["tasks"][1]["error_type"] == "PrivateBackendError"
    private = json.loads(
        (tmp_path / "evaluation" / "task_rows.json").read_text()
    )
    assert private["tasks"][1]["error"] == (
        "PrivateBackendError: hidden detail"
    )


def test_grader_failure_never_reruns_or_rolls_back_solver(
    tmp_path: Path,
) -> None:
    backend = object.__new__(OpenSandboxBackend)
    operations: list[str] = []

    def fake_worker(**kwargs):
        operation = kwargs["operation"]
        operations.append(operation)
        if operation == "grade-artifacts":
            raise OpenSandboxBackendError("verifier unavailable")
        request = kwargs["request"]
        temporary = tmp_path / f"remote-{len(operations)}"
        result = temporary / "result"
        (result / "evaluation").mkdir(parents=True)
        state = json.loads(Path(kwargs["state_path"]).read_text())
        state["count"] += 1
        (result / "harness_store.json").write_text(json.dumps(state))
        artifact = {
            "schema_version": 1,
            "benchmark": "bfcl",
            "task_id": request["task_ids"][0],
            "is_finished": True,
        }
        return {
            "rows": [
                {
                    "task_id": request["task_ids"][0],
                    "split": request["split_names"][0],
                    "status": "awaiting_grader",
                }
            ],
            "grading_artifacts": [artifact],
        }, temporary

    backend._run_worker = fake_worker  # type: ignore[method-assign]
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
        evaluation_dir=tmp_path / "evaluation",
        public_dir=None,
        config={},
        base_model="model",
        max_tokens=100,
        embedding_model="unused",
    )

    assert operations == [
        "run-solver-block",
        "grade-artifacts",
        "grade-artifacts",
        "grade-artifacts",
    ]
    assert json.loads(output.read_text()) == {"count": 1}
    assert block.rows[0]["status"] == "grader_error"
