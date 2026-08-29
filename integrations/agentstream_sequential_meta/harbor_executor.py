"""Optional Harbor 0.20 executor for snapshot-backed continual task chunks.

The outer controller still owns evolution, sequential checkpoints, transfer
evaluation, and HDA.  Harbor owns only the lifecycle of one solver chunk and
its separate late verifier.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, override


class HarborExecutorError(RuntimeError):
    pass


def snapshot_reference(snapshot_id: str) -> str:
    if not snapshot_id or any(character.isspace() for character in snapshot_id):
        raise ValueError("OpenSandbox snapshot id must be a non-empty token")
    return f"snapshot:{snapshot_id}"


def parse_snapshot_reference(value: str | None) -> str:
    if value is None or not value.startswith("snapshot:"):
        raise ValueError("Harbor snapshot environment requires snapshot:<id>")
    snapshot_id = value.removeprefix("snapshot:")
    if not snapshot_id or any(character.isspace() for character in snapshot_id):
        raise ValueError("invalid OpenSandbox snapshot reference")
    return snapshot_id


def write_task_definition(
    task_dir: Path,
    *,
    solver_snapshot_id: str,
    grader_snapshot_id: str,
    agent_timeout_sec: int,
    verifier_timeout_sec: int,
    cpus: int,
    memory_mb: int,
) -> None:
    """Materialize the small local Harbor task consumed by ``Trial.create``."""
    if min(agent_timeout_sec, verifier_timeout_sec, cpus, memory_mb) <= 0:
        raise ValueError("Harbor task timeouts and resources must be positive")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "environment").mkdir(exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "instruction.md").write_text(
        "Run the fixed continual-harness solver chunk supplied by the controller.\n",
        encoding="utf-8",
    )
    task_toml = f'''schema_version = "1.3"

[metadata]
executor = "continual-harness-harbor"

[agent]
timeout_sec = {float(agent_timeout_sec)}

[environment]
docker_image = "{snapshot_reference(solver_snapshot_id)}"
workdir = "/work"
cpus = {cpus}
memory_mb = {memory_mb}

[verifier]
timeout_sec = {float(verifier_timeout_sec)}
environment_mode = "separate"

[verifier.environment]
docker_image = "{snapshot_reference(grader_snapshot_id)}"
workdir = "/work"
cpus = {cpus}
memory_mb = {memory_mb}
'''
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")


try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.environments.opensandbox import (
        OpenSandboxEnvironment,
        _is_transient_sdk_error,
    )
    from harbor.models.agent.context import AgentContext
    from harbor.models.verifier.result import VerifierResult
    from harbor.trial.trial import Trial
    from harbor.utils.env import resolve_env_vars
    from harbor.verifier.base import BaseVerifier
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover - exercised only without the optional extra
    BaseAgent = object  # type: ignore[assignment,misc]
    BaseEnvironment = object  # type: ignore[assignment,misc]
    OpenSandboxEnvironment = object  # type: ignore[assignment,misc]
    BaseVerifier = object  # type: ignore[assignment,misc]
    Trial = None  # type: ignore[assignment]
    VerifierResult = None  # type: ignore[assignment]
    AgentContext = Any  # type: ignore[assignment,misc]
    resolve_env_vars = None  # type: ignore[assignment]
    retry = retry_if_exception = stop_after_attempt = wait_exponential = None
    _is_transient_sdk_error = None


if Trial is not None:

    class SnapshotOpenSandboxEnvironment(OpenSandboxEnvironment):
        """Harbor OpenSandbox environment restored from a prepared snapshot."""

        def __init__(
            self,
            *args: Any,
            snapshot_volumes: dict[str, list[dict[str, Any]]] | None = None,
            **kwargs: Any,
        ) -> None:
            self._snapshot_volumes = {
                str(snapshot_id): list(volumes)
                for snapshot_id, volumes in (snapshot_volumes or {}).items()
            }
            super().__init__(*args, **kwargs)

        @override
        def _validate_definition(self) -> None:
            parse_snapshot_reference(self.task_env_config.docker_image)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(_is_transient_sdk_error),
            reraise=True,
        )
        async def _create_sandbox(self, sdk: dict[str, Any]) -> Any:
            snapshot_id = parse_snapshot_reference(
                self.task_env_config.docker_image
            )
            volume_definitions = [
                *self._volumes,
                *self._snapshot_volumes.get(snapshot_id, []),
            ]
            sandbox = await sdk["Sandbox"].create(
                snapshot_id=snapshot_id,
                timeout=timedelta(seconds=self._sandbox_timeout_sec),
                ready_timeout=timedelta(seconds=self._ready_timeout_sec),
                env=dict(self._persistent_env),
                metadata={**self._metadata, "session_id": self.session_id},
                resource=self._build_resource(),
                network_policy=self._build_network_policy(sdk),
                extensions=self._build_extensions(),
                entrypoint=self._entrypoint,
                volumes=(
                    [sdk["Volume"](**volume) for volume in volume_definitions]
                    if volume_definitions
                    else None
                ),
                connection_config=self._build_connection_config(sdk),
                skip_health_check=True,
            )
            if not self._skip_health_check:
                try:
                    await self._wait_until_ready(sandbox)
                except BaseException:
                    await self._safe_kill(sandbox)
                    raise
            return sandbox


    class ContinualChunkAgent(BaseAgent):
        """Trusted Harbor agent that invokes the fixed solver worker once."""

        def __init__(
            self,
            *args: Any,
            candidate_path: str,
            state_path: str,
            solver_request: dict[str, Any],
            grader_request: dict[str, Any],
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.candidate_path = Path(candidate_path)
            self.state_path = Path(state_path)
            self.solver_request = dict(solver_request)
            self.grader_request = dict(grader_request)

        @staticmethod
        @override
        def name() -> str:
            return "continual-chunk"

        @override
        def version(self) -> str:
            return "1"

        @override
        async def setup(self, environment: BaseEnvironment) -> None:
            request_path = self.logs_dir / "solver-request.json"
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(
                json.dumps(self.solver_request, ensure_ascii=False), encoding="utf-8"
            )
            await environment.upload_file(self.candidate_path, "/work/candidate.py")
            await environment.upload_file(
                self.state_path, "/work/harness_store.json"
            )
            await environment.upload_file(request_path, "/work/request.json")

        @override
        async def run(
            self,
            instruction: str,
            environment: BaseEnvironment,
            context: AgentContext,
        ) -> None:
            del instruction
            command = (
                "python -m integrations.agentstream_sequential_meta.sandbox_worker "
                "run-solver-block --request /work/request.json"
            )
            execution = await environment.exec(command, cwd="/work")
            if execution.return_code != 0:
                raise HarborExecutorError(
                    "solver worker failed: "
                    + (execution.stderr or execution.stdout or "no output")
                )

            solver_result_dir = self.logs_dir / "solver_result"
            await environment.download_dir("/work/result", solver_result_dir)
            result_path = solver_result_dir / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            artifacts = payload.get("grading_artifacts")
            if not isinstance(artifacts, list):
                raise HarborExecutorError("solver returned invalid grading artifacts")

            grading_request = {
                **self.grader_request,
                "grading_artifacts": artifacts,
            }
            grading_path = self.logs_dir / "grading-request.json"
            grading_path.write_text(
                json.dumps(grading_request, ensure_ascii=False), encoding="utf-8"
            )
            await environment.upload_file(
                grading_path, "/logs/artifacts/grading-request.json"
            )
            rows = payload.get("rows", [])
            context.metadata = {
                "solver_rows": len(rows) if isinstance(rows, list) else 0,
                "grading_artifacts": len(artifacts),
            }
            if isinstance(rows, list):
                context.n_input_tokens = sum(
                    int(row.get("input_tokens", 0) or 0)
                    for row in rows
                    if isinstance(row, dict)
                )
                context.n_output_tokens = sum(
                    int(row.get("output_tokens", 0) or 0)
                    for row in rows
                    if isinstance(row, dict)
                )
                context.cost_usd = sum(
                    float(row.get("agent_cost", 0.0) or 0.0)
                    for row in rows
                    if isinstance(row, dict)
                )


    class ContinualArtifactVerifier(BaseVerifier):
        """Run the fixed grader worker inside Harbor's separate environment."""

        @override
        async def verify(self) -> VerifierResult:
            merged_env = {
                **self.task.config.verifier.env,
                **(self.verifier_env or {}),
                **self.override_env,
            }
            env = resolve_env_vars(merged_env) if merged_env else None
            command = (
                "cp /logs/artifacts/grading-request.json /work/request.json && "
                "python -m integrations.agentstream_sequential_meta."
                "sandbox_grader_worker grade-artifacts --request /work/request.json"
            )
            execution = await self.environment.exec(command, cwd="/work", env=env)
            if execution.return_code != 0:
                raise HarborExecutorError(
                    "grader worker failed: "
                    + (execution.stderr or execution.stdout or "no output")
                )
            target = self.trial_paths.verifier_dir / "grader_result"
            await self.environment.download_dir("/work/result", target)
            payload = json.loads(
                (target / "result.json").read_text(encoding="utf-8")
            )
            results = payload.get("grade_results")
            if not isinstance(results, list):
                raise HarborExecutorError("grader returned invalid results")
            scores = [
                float(item["score"].get("score", 0.0) or 0.0)
                for item in results
                if isinstance(item, dict) and isinstance(item.get("score"), dict)
            ]
            reward = sum(scores) / len(results) if results else 0.0
            return VerifierResult(rewards={"reward": reward})


@dataclass(frozen=True)
class HarborChunkResult:
    solver_result_dir: Path | None
    grader_result_path: Path | None
    exception_type: str | None
    exception_message: str | None
    reward: float | None


class HarborTrialExecutor:
    """Programmatic Harbor adapter for one continual solver chunk."""

    def __init__(
        self,
        *,
        domain: str,
        api_key: str,
        protocol: str,
        use_server_proxy: bool,
        request_timeout_sec: int,
        ready_timeout_sec: int,
        sandbox_timeout_sec: int,
        cpus: int,
        memory_mb: int,
    ) -> None:
        self.environment_kwargs = {
            "domain": domain,
            "api_key": api_key,
            "protocol": protocol,
            "use_server_proxy": use_server_proxy,
            "request_timeout_sec": request_timeout_sec,
            "ready_timeout_sec": ready_timeout_sec,
            "sandbox_timeout_sec": sandbox_timeout_sec,
            "metadata": {"purpose": "continual-harness-harbor"},
        }
        self.cpus = cpus
        self.memory_mb = memory_mb

    def run_chunk(
        self,
        *,
        task_root: Path,
        trials_dir: Path,
        trial_name: str,
        solver_snapshot_id: str,
        grader_snapshot_id: str,
        candidate_path: Path,
        state_path: Path,
        solver_request: dict[str, Any],
        grader_request: dict[str, Any],
        agent_env: dict[str, str],
        verifier_env: dict[str, str],
        agent_timeout_sec: int,
        verifier_timeout_sec: int,
        on_solver_complete: Callable[[Path], None] | None = None,
        solver_volumes: list[dict[str, Any]] | None = None,
        grader_volumes: list[dict[str, Any]] | None = None,
    ) -> HarborChunkResult:
        if Trial is None:
            raise HarborExecutorError(
                "Harbor executor requires harbor[opensandbox]==0.20.0"
            )
        if task_root.exists():
            shutil.rmtree(task_root)
        write_task_definition(
            task_root,
            solver_snapshot_id=solver_snapshot_id,
            grader_snapshot_id=grader_snapshot_id,
            agent_timeout_sec=agent_timeout_sec,
            verifier_timeout_sec=verifier_timeout_sec,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
        )

        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            TrialConfig,
            VerifierConfig,
        )

        environment_kwargs = {
            **self.environment_kwargs,
            "snapshot_volumes": {
                solver_snapshot_id: list(solver_volumes or []),
                grader_snapshot_id: list(grader_volumes or []),
            },
        }
        config = TrialConfig(
            task=TaskConfig(path=task_root),
            trial_name=trial_name,
            trials_dir=trials_dir,
            agent=AgentConfig(
                import_path=(
                    "integrations.agentstream_sequential_meta.harbor_executor:"
                    "ContinualChunkAgent"
                ),
                kwargs={
                    "candidate_path": str(candidate_path),
                    "state_path": str(state_path),
                    "solver_request": solver_request,
                    "grader_request": grader_request,
                },
                env=agent_env,
            ),
            verifier=VerifierConfig(
                import_path=(
                    "integrations.agentstream_sequential_meta.harbor_executor:"
                    "ContinualArtifactVerifier"
                ),
                env=verifier_env,
            ),
            environment=EnvironmentConfig(
                import_path=(
                    "integrations.agentstream_sequential_meta.harbor_executor:"
                    "SnapshotOpenSandboxEnvironment"
                ),
                kwargs=environment_kwargs,
                override_cpus=self.cpus,
                override_memory_mb=self.memory_mb,
            ),
        )

        async def execute():
            from harbor.trial.hooks import TrialEvent

            trial = await Trial.create(config)
            if on_solver_complete is not None:
                async def commit_solver_output(_event: Any) -> None:
                    on_solver_complete(trial.paths.agent_dir / "solver_result")

                trial.add_hook(TrialEvent.VERIFICATION_START, commit_solver_output)
            return await trial.run()

        result = asyncio.run(execute())
        trial_dir = trials_dir / trial_name
        solver_dir = trial_dir / "agent" / "solver_result"
        grader_path = trial_dir / "verifier" / "grader_result" / "result.json"
        rewards = result.verifier_result.rewards if result.verifier_result else None
        return HarborChunkResult(
            solver_result_dir=solver_dir if solver_dir.is_dir() else None,
            grader_result_path=grader_path if grader_path.is_file() else None,
            exception_type=(
                result.exception_info.exception_type
                if result.exception_info is not None
                else None
            ),
            exception_message=(
                result.exception_info.exception_message
                if result.exception_info is not None
                else None
            ),
            reward=(float(rewards["reward"]) if rewards and "reward" in rewards else None),
        )
