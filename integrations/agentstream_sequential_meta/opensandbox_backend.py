"""OpenSandbox execution backend for benchmark-native Sequential Meta-Harness.

The controller and proposer remain on the host.  Benchmark discovery and each
candidate evaluation block run in a fresh OpenSandbox restored from a
benchmark-specific dependency snapshot. Only the explicit candidate and
generic JSON harness checkpoint cross the sandbox boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

RUNTIME_CONTRACT = "native-meta-harness-opensandbox-v5-mounted-assets"
SOLVER_RECIPE_REVISION = "solver-v2-litellm"
BROWSECOMPPLUS_SOLVER_RECIPE_REVISION = "browsecompplus-solver-v5-mounted-assets"
BROWSECOMPPLUS_ASSET_REVISION = "browsecompplus-qwen3-8b-assets-v1"
MANIFEST_SCHEMA_VERSION = 2
DEFAULT_IMAGE = "python:3.12-bookworm"
DEFAULT_DOMAIN = "10.119.212.249:8080"
DEFAULT_CACHE_PATH = Path(".meta-harness/opensandbox-runtime-cache-v2.json")
DEFAULT_ASSETS_PATH = Path(".meta-harness/opensandbox-assets-v1")
_RUNTIME_MODES = frozenset({"auto", "require", "rebuild"})
_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".meta-harness",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


class OpenSandboxBackendError(RuntimeError):
    """OpenSandbox setup, execution, or artifact-transfer failure."""


class RuntimeCacheMissError(OpenSandboxBackendError):
    """A required benchmark runtime snapshot is not present."""


@dataclass(frozen=True)
class OpenSandboxSettings:
    domain: str = DEFAULT_DOMAIN
    api_key: str = ""
    protocol: str = "http"
    use_server_proxy: bool = True
    request_timeout_sec: int = 600
    ready_timeout_sec: int = 1800
    sandbox_timeout_sec: int = 7200
    command_timeout_sec: int = 3600
    snapshot_ready_timeout_sec: int = 1800
    image: str = DEFAULT_IMAGE
    runtime_mode: str = "auto"
    runtime_cache_path: Path = DEFAULT_CACHE_PATH
    runtime_assets_root: Path | None = None
    cpus: int = 4
    memory: str = "16Gi"

    def validate(self) -> None:
        if self.runtime_mode not in _RUNTIME_MODES:
            raise ValueError(
                f"runtime_mode must be one of {sorted(_RUNTIME_MODES)}"
            )
        if not self.domain.strip():
            raise ValueError("OpenSandbox domain must not be empty")
        if self.protocol not in {"http", "https"}:
            raise ValueError("OpenSandbox protocol must be http or https")
        if any(
            value <= 0
            for value in (
                self.request_timeout_sec,
                self.ready_timeout_sec,
                self.sandbox_timeout_sec,
                self.command_timeout_sec,
                self.snapshot_ready_timeout_sec,
                self.cpus,
            )
        ):
            raise ValueError("OpenSandbox timeouts and cpus must be positive")
        if not self.image.strip() or not self.memory.strip():
            raise ValueError("OpenSandbox image and memory must not be empty")
        if (
            self.runtime_assets_root is not None
            and not self.runtime_assets_root.is_absolute()
        ):
            raise ValueError("runtime_assets_root must be an absolute path")

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["api_key"] = "<configured>" if self.api_key else ""
        value["runtime_cache_path"] = str(self.runtime_cache_path)
        value["runtime_assets_root"] = (
            str(self.runtime_assets_root)
            if self.runtime_assets_root is not None
            else None
        )
        return value


@dataclass(frozen=True)
class RuntimeSnapshot:
    benchmark: str
    role: str
    identity: str
    snapshot_id: str
    source_digest: str
    image: str
    created_at: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RuntimeSnapshot:
        return cls(
            benchmark=str(value["benchmark"]),
            role=str(value["role"]),
            identity=str(value["identity"]),
            snapshot_id=str(value["snapshot_id"]),
            source_digest=str(value["source_digest"]),
            image=str(value["image"]),
            created_at=str(value["created_at"]),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _is_runtime_source(label: str, relative: Path, role: str = "all") -> bool:
    if label != "meta-harness":
        return False
    if len(relative.parts) < 3 or relative.parts[0] != "integrations":
        return False
    package = relative.parts[1]
    package_relative = Path(*relative.parts[2:])
    filename = relative.name
    if package == "agentstream_sequential_meta":
        common = len(package_relative.parts) == 1 and filename in {
            "__init__.py",
        }
        if role == "solver":
            return (
                common
                or package_relative.parts[0] == "benchmark_backends"
                or filename
                in {
                    "candidate.py",
                    "candidate_contract.py",
                    "harness_protocol.py",
                    "model_runtime.py",
                    "sandbox_evaluation.py",
                    "sandbox_worker.py",
                }
            )
        if role == "grader":
            return (
                common
                or package_relative.parts[0]
                in {"benchmark_graders", "benchmark_backends"}
                or filename in {"grading.py", "sandbox_grader_worker.py"}
            )
        return _is_runtime_source(label, relative, "solver") or _is_runtime_source(
            label, relative, "grader"
        )
    return False


def _iter_source_files(label: str, root: Path, role: str = "all"):
    selected: list[tuple[Path, Path]] = []
    for directory, dirnames, filenames in os.walk(root, topdown=True):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _EXCLUDED_PARTS
            and not (directory_path / name).is_symlink()
        )
        for name in sorted(filenames):
            path = directory_path / name
            relative = path.relative_to(root)
            if path.is_symlink() or not path.is_file():
                continue
            if not _is_runtime_source(label, relative, role):
                continue
            selected.append((path, relative))
    yield from sorted(selected, key=lambda item: item[1].as_posix())


def source_digest(meta_harness_root: Path, role: str = "all") -> str:
    """Hash the exact source trees uploaded into benchmark runtimes."""

    digest = hashlib.sha256()
    for label, root in (("meta-harness", meta_harness_root),):
        for path, relative in _iter_source_files(label, root, role):
            digest.update(label.encode())
            digest.update(b"\0")
            digest.update(relative.as_posix().encode())
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def runtime_identity(
    *,
    benchmark: str,
    source_hash: str,
    settings: OpenSandboxSettings,
    role: str = "solver",
    recipe_revision: str | None = None,
) -> str:
    payload = {
        "contract": RUNTIME_CONTRACT,
        "benchmark": benchmark,
        "role": role,
        "source_digest": source_hash,
        "image": settings.image,
        "domain": settings.domain,
        "protocol": settings.protocol,
        "use_server_proxy": settings.use_server_proxy,
        "opensandbox_sdk": _package_version("opensandbox"),
        "harbor": _package_version("harbor"),
    }
    if recipe_revision is not None:
        payload["recipe_revision"] = recipe_revision
    if benchmark == "browsecompplus":
        payload["asset_revision"] = BROWSECOMPPLUS_ASSET_REVISION
        payload["runtime_assets_root"] = (
            str(settings.runtime_assets_root)
            if settings.runtime_assets_root is not None
            else None
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _derive_seed(master_seed: int, slug: str) -> int:
    digest = hashlib.md5(f"{master_seed}_{slug}".encode()).hexdigest()
    return int(digest, 16) % (2**31)


def sequential_task_order(
    task_ids: dict[str, list[str]], num_tasks: int, ordering_seed: int
) -> list[tuple[str, str]]:
    """Use AgentStream's published fixed-selection Sequential ordering rule."""

    selected: dict[str, list[str]] = {}
    for slug in sorted(task_ids):
        ids = [str(item) for item in task_ids[slug]]
        selection_rng = random.Random(_derive_seed(42, slug))
        selection_rng.shuffle(ids)
        chosen = ids[:num_tasks]
        if ordering_seed != 42:
            order_rng = random.Random(_derive_seed(ordering_seed, slug))
            order_rng.shuffle(chosen)
        selected[slug] = chosen
    return [
        (slug, task_id)
        for slug in sorted(selected)
        for task_id in selected[slug]
    ]


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise OpenSandboxBackendError(
                    f"unsafe path in sandbox artifact: {member.name}"
                )
            if member.issym() or member.islnk():
                raise OpenSandboxBackendError(
                    f"links are not allowed in sandbox artifact: {member.name}"
                )
        handle.extractall(root, filter="data")


def _execution_text(execution: Any, stream: str) -> str:
    logs = getattr(execution, "logs", None)
    entries = getattr(logs, stream, []) if logs is not None else []
    return "".join(str(getattr(entry, "text", entry)) for entry in entries)


class OpenSandboxBackend:
    """Prepare benchmark snapshots and run isolated evaluation workers."""

    def __init__(
        self,
        *,
        settings: OpenSandboxSettings,
        meta_harness_root: Path,
        provider_env: dict[str, str],
    ) -> None:
        settings.validate()
        self.settings = settings
        self.meta_harness_root = meta_harness_root.resolve()
        self.provider_env = self._runtime_env(provider_env)
        self._source_digests = {
            role: source_digest(self.meta_harness_root, role)
            for role in ("solver", "grader")
        }
        self._source_archives: dict[str, Path] = {}
        self._snapshots: dict[tuple[str, str], RuntimeSnapshot] = {}

    @property
    def source_hash(self) -> str:
        return source_digest(self.meta_harness_root)

    def public_config(self) -> dict[str, Any]:
        return {
            **self.settings.public_dict(),
            "source_digest": self.source_hash,
            "role_source_digests": dict(self._source_digests),
            "runtime_contract": RUNTIME_CONTRACT,
            "solver_recipe_revisions": {
                "default": SOLVER_RECIPE_REVISION,
                "browsecompplus": BROWSECOMPPLUS_SOLVER_RECIPE_REVISION,
            },
        }

    @staticmethod
    def _recipe_revision(benchmark: str, role: str) -> str | None:
        # Solver model-runtime dependencies are versioned independently so a
        # solver-only recipe fix does not invalidate private grader snapshots.
        if role != "solver":
            return None
        if benchmark == "browsecompplus":
            return BROWSECOMPPLUS_SOLVER_RECIPE_REVISION
        return SOLVER_RECIPE_REVISION

    @staticmethod
    def _runtime_env(provider_env: dict[str, str]) -> dict[str, str]:
        allowed = {
            "ALL_PROXY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "all_proxy",
            "LITELLM_LOCAL_MODEL_COST_MAP",
        }
        env = {key: value for key, value in provider_env.items() if key in allowed}
        env.update(
            {
                "PYTHONPATH": "/opt/meta-harness",
                "BROWSECOMPPLUS_ASSETS_DIR": (
                    "/opt/benchmark-assets/browsecompplus"
                ),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }
        )
        return env

    def _bootstrap_env(self) -> dict[str, str]:
        """Return transient dependency-download proxy variables only.

        These values are attached to the bootstrap command rather than the
        sandbox creation environment, so they are not persisted in the runtime
        snapshot. Model credentials are deliberately excluded.
        """

        proxy_keys = {
            "ALL_PROXY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "all_proxy",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        }
        return {
            key: value
            for key, value in self.provider_env.items()
            if key in proxy_keys
        }

    def _worker_env(self, operation: str) -> dict[str, str]:
        env = dict(self.provider_env)
        if operation == "list-tasks":
            # Discovery only reads benchmark data. Do not send solver
            # credentials to a sandbox that cannot legitimately use them.
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_BASE_URL", None)
        return env

    def _connection_config(self):
        try:
            from opensandbox.config import ConnectionConfig
        except ImportError as exc:
            raise OpenSandboxBackendError(
                "OpenSandbox backend requires harbor[opensandbox]==0.20.0"
            ) from exc
        return ConnectionConfig(
            api_key=self.settings.api_key or None,
            domain=self.settings.domain,
            protocol=self.settings.protocol,
            request_timeout=timedelta(seconds=self.settings.request_timeout_sec),
            use_server_proxy=self.settings.use_server_proxy,
        )

    def _read_manifest(self) -> dict[str, Any]:
        path = self.settings.runtime_cache_path
        if not path.is_file():
            return {"schema_version": MANIFEST_SCHEMA_VERSION, "entries": {}}
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise OpenSandboxBackendError(
                f"unsupported OpenSandbox runtime manifest: {path}"
            )
        if not isinstance(data.get("entries"), dict):
            raise OpenSandboxBackendError(f"invalid runtime manifest: {path}")
        return data

    def _write_manifest(self, data: dict[str, Any]) -> None:
        path = self.settings.runtime_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def runtime_volumes(
        self,
        benchmark: str,
        role: str,
        *,
        read_only: bool,
    ) -> list[dict[str, Any]]:
        """Return role-separated host mounts for benchmark runtime assets."""

        root = getattr(self.settings, "runtime_assets_root", None)
        if benchmark != "browsecompplus" or root is None:
            return []
        if role not in {"solver", "grader"}:
            raise ValueError("runtime role must be solver or grader")
        host_path = (root / benchmark / role).resolve()
        if root.resolve() not in host_path.parents:
            raise OpenSandboxBackendError("runtime asset path escaped its root")
        host_path.mkdir(parents=True, exist_ok=True)
        host_path.chmod(0o700)
        return [
            {
                "name": f"native-harness-{benchmark}-{role}-assets",
                "host": {"path": str(host_path)},
                "mountPath": "/opt/benchmark-assets/browsecompplus",
                "readOnly": read_only,
            }
        ]

    def _lookup_snapshot(self, benchmark: str, role: str) -> RuntimeSnapshot | None:
        identity = runtime_identity(
            benchmark=benchmark,
            role=role,
            source_hash=self._source_digests[role],
            settings=self.settings,
            recipe_revision=self._recipe_revision(benchmark, role),
        )
        value = self._read_manifest()["entries"].get(identity)
        return RuntimeSnapshot.from_dict(value) if value else None

    def _archive_sources(self, role: str) -> Path:
        existing = self._source_archives.get(role)
        if existing is not None and existing.is_file():
            return existing
        directory = Path(tempfile.mkdtemp(prefix="native-harness-sandbox-source-"))
        archive = directory / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for label, root in (("meta-harness", self.meta_harness_root),):
                for path, relative in _iter_source_files(label, root, role):
                    handle.add(path, arcname=str(Path(label) / relative), recursive=False)
        self._source_archives[role] = archive
        return archive

    def prepare(self, benchmarks: list[str]) -> dict[str, RuntimeSnapshot]:
        return {
            f"{benchmark}:{role}": self.ensure_runtime(benchmark, role)
            for benchmark in benchmarks
            for role in ("solver", "grader")
        }

    def ensure_runtime(self, benchmark: str, role: str = "solver") -> RuntimeSnapshot:
        if role not in {"solver", "grader"}:
            raise ValueError("runtime role must be solver or grader")
        key = (benchmark, role)
        cached = self._snapshots.get(key)
        if cached is not None and self.settings.runtime_mode != "rebuild":
            return cached
        existing = self._lookup_snapshot(benchmark, role)
        if existing is not None and self.settings.runtime_mode != "rebuild":
            self._snapshots[key] = existing
            return existing
        if self.settings.runtime_mode == "require":
            raise RuntimeCacheMissError(
                f"no OpenSandbox runtime snapshot for {benchmark!r}; "
                "run once with --opensandbox-runtime-mode auto"
            )
        snapshot = self._build_runtime(benchmark, role)
        manifest = self._read_manifest()
        manifest["entries"][snapshot.identity] = asdict(snapshot)
        manifest["updated_at"] = _utc_now()
        self._write_manifest(manifest)
        self._snapshots[key] = snapshot
        return snapshot

    def _bootstrap_command(self, benchmark: str, role: str = "solver") -> str:
        if benchmark not in {"bfcl", "browsecompplus"}:
            raise OpenSandboxBackendError(
                f"No native runtime bootstrap for {benchmark!r}"
            )
        if role not in {"solver", "grader"}:
            raise ValueError("runtime role must be solver or grader")
        commands = [
            "set -euo pipefail",
            (
                "retry_cmd() { attempt=1; while ! \"$@\"; do "
                "if [ \"$attempt\" -ge 5 ]; then return 1; fi; "
                "sleep $((attempt * 2)); attempt=$((attempt + 1)); done; }"
            ),
            "export DEBIAN_FRONTEND=noninteractive",
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
            "retry_cmd apt-get update",
            (
                "retry_cmd apt-get install -y --no-install-recommends "
                "git ca-certificates curl build-essential"
            ),
            "rm -rf /var/lib/apt/lists/*",
            "mkdir -p /opt/meta-harness /opt/benchmark-assets",
            "tar -xzf /tmp/native-meta-source.tar.gz -C /opt",
            "retry_cmd python -m pip install --no-cache-dir --upgrade pip",
            (
                "retry_cmd python -m pip install --no-cache-dir "
                "python-dotenv 'pydantic>=2'"
            ),
        ]
        if role == "solver":
            commands.append("retry_cmd python -m pip install --no-cache-dir litellm")
        if benchmark == "bfcl":
            commands.extend(self._bfcl_bootstrap_commands(role))
        else:
            commands.extend(self._browsecompplus_bootstrap_commands(role))
        commands.extend(
            [
                "python -c \"print('native-runtime-ready')\"",
                f"printf '%s\\n' {RUNTIME_CONTRACT} > /opt/RUNTIME_CONTRACT",
            ]
        )
        return "; ".join(commands)

    @staticmethod
    def _bfcl_bootstrap_commands(role: str = "solver") -> list[str]:
        gorilla_repo = "/opt/benchmark-packages/gorilla"
        bfcl_repo = f"{gorilla_repo}/berkeley-function-call-leaderboard"
        revision = "7ad0134c665944819f88bc50862108d94015968b"
        commands = [
            f"mkdir -p {gorilla_repo}",
            f"git -C {gorilla_repo} init",
            (
                f"git -C {gorilla_repo} remote add origin "
                "https://github.com/ShishirPatil/gorilla.git"
            ),
            f"retry_cmd git -C {gorilla_repo} fetch --depth 1 origin {revision}",
            f"git -C {gorilla_repo} checkout --force {revision}",
            (
                "retry_cmd python -m pip install --no-cache-dir "
                f"-e {bfcl_repo} soundfile"
            ),
        ]
        if role == "solver":
            commands.extend(
                [
                    f"rm -rf {bfcl_repo}/bfcl_eval/data/possible_answer",
                    (
                        f"test ! -e {bfcl_repo}/bfcl_eval/data/possible_answer"
                    ),
                ]
            )
        commands.append("python -c \"import bfcl_eval; print('bfcl-native-ready')\"")
        return commands

    @staticmethod
    def _browsecompplus_bootstrap_commands(role: str = "solver") -> list[str]:
        assets = "/opt/benchmark-assets/browsecompplus"
        full_data = f"{assets}/data/browsecomp_plus_decrypted.jsonl"
        solver_data = f"{assets}/data/browsecomp_plus_solver.jsonl"
        grader_data = f"{assets}/data/browsecomp_plus_grader.jsonl"
        index_dir = f"{assets}/indexes/qwen3-embedding-8b"
        model_dir = f"{assets}/models/Qwen3-Embedding-8B"
        tokenizer_dir = f"{assets}/models/Qwen3-0.6B-tokenizer"
        corpus_dir = f"{assets}/corpus"
        revision = "7cd697e133ba9150c3c310d10043e327d9f06c41"
        tevatron_revision = "dd063104c81a76d6a77c845f667b46b9e5abd625"
        role_data = grader_data if role == "grader" else solver_data
        common = [
            (
                "retry_cmd python -m pip install --no-cache-dir --no-deps "
                "'git+https://github.com/lilacheden/BrowseComp-Plus/"
                f"@{revision}'"
            ),
            "retry_cmd python -m pip install --no-cache-dir 'datasets>=4.0.0'",
            (
                f"mkdir -p {assets}/data {assets}/topics-qrels "
                f"{assets}/indexes {assets}/models"
            ),
            (
                f"if [ ! -f {role_data} ]; then "
                f"cd {assets} && retry_cmd python -m scripts_build_index.decrypt_dataset "
                f"--output {full_data} "
                f"--generate-tsv {assets}/topics-qrels/queries.tsv && "
                "python /opt/meta-harness/integrations/"
                "agentstream_sequential_meta/benchmark_backends/"
                "prepare_browsecompplus.py "
                f"--input {full_data} --solver-output {solver_data} "
                f"--grader-output {grader_data}; fi"
            ),
        ]
        if role == "grader":
            return common + [
                (
                    "retry_cmd python -m pip install --no-cache-dir "
                    "litellm numpy openai tqdm"
                ),
                f"rm -f {full_data} {solver_data}",
                f"test -f {grader_data}",
                "python -c \"import scripts_evaluation; print('browse-grader-ready')\"",
            ]
        return common + [
            (
                "retry_cmd python -m pip install --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu"
            ),
            (
                "retry_cmd python -m pip install --no-cache-dir "
                "'transformers>=4.53.2,<5' 'pillow>=12.1.1' "
                "'peft>=0.16.0' safetensors faiss-cpu hf_transfer "
                "huggingface_hub"
            ),
            (
                "retry_cmd python -m pip install --no-cache-dir --no-deps "
                "'git+https://github.com/texttron/tevatron.git"
                f"@{tevatron_revision}'"
            ),
            (
                f"if ! compgen -G '{index_dir}/corpus.shard*_of_4.pkl' "
                "> /dev/null; then "
                f"cd {assets}/indexes && retry_cmd "
                "hf download Tevatron/browsecomp-plus-indexes "
                "--repo-type=dataset --include='qwen3-embedding-8b/*' "
                "--local-dir .; fi"
            ),
            (
                f"if [ ! -f {model_dir}/config.json ]; then "
                "retry_cmd hf download "
                f"Qwen/Qwen3-Embedding-8B --local-dir {model_dir}; fi"
            ),
            (
                f"if [ ! -f {corpus_dir}/state.json ]; then "
                "retry_cmd python -c \"from datasets import load_dataset; "
                "load_dataset('Tevatron/browsecomp-plus-corpus', "
                f"split='train', cache_dir='{assets}/.cache/datasets')"
                f".save_to_disk('{corpus_dir}')\"; fi"
            ),
            (
                f"if [ ! -f {tokenizer_dir}/tokenizer_config.json ]; then "
                "retry_cmd python -c \"from transformers import AutoTokenizer; "
                "AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', "
                "cache_dir='/tmp/qwen-tokenizer-cache').save_pretrained("
                f"'{tokenizer_dir}')\"; "
                "rm -rf /tmp/qwen-tokenizer-cache; fi"
            ),
            (
                f"python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('{tokenizer_dir}', "
                "local_files_only=True); "
                "from searcher.searchers.faiss_searcher import FaissSearcher; "
                "import scripts_evaluation; "
                "print('browsecompplus-native-ready')\""
            ),
            f"rm -f {full_data} {grader_data}",
            f"test -f {solver_data}",
        ]

    def _create_sandbox(
        self,
        *,
        benchmark: str,
        snapshot_id: str | None,
        env: dict[str, str],
        role: str,
        assets_read_only: bool,
    ):
        try:
            from opensandbox import Sandbox
            from opensandbox.models.sandboxes import Volume
        except ImportError as exc:
            raise OpenSandboxBackendError(
                "OpenSandbox backend requires harbor[opensandbox]==0.20.0"
            ) from exc
        volume_definitions = self.runtime_volumes(
            benchmark, role, read_only=assets_read_only
        )
        kwargs = {
            "timeout": timedelta(seconds=self.settings.sandbox_timeout_sec),
            "ready_timeout": timedelta(seconds=self.settings.ready_timeout_sec),
            "env": env,
            "metadata": {
                "purpose": "native-sequential-meta-harness",
                "runtime_contract": RUNTIME_CONTRACT,
                "runtime_role": role,
            },
            "resource": {
                "cpu": str(self.settings.cpus),
                "memory": self.settings.memory,
            },
            "volumes": (
                [Volume(**volume) for volume in volume_definitions]
                if volume_definitions
                else None
            ),
            "connection_config": self._connection_config(),
        }
        if snapshot_id is None:
            return Sandbox.create(self.settings.image, **kwargs)
        return Sandbox.create(snapshot_id=snapshot_id, **kwargs)

    def _build_runtime(self, benchmark: str, role: str) -> RuntimeSnapshot:
        import asyncio

        async def build() -> RuntimeSnapshot:
            from opensandbox import SandboxManager
            from opensandbox.models.execd import RunCommandOpts

            sandbox = await self._create_sandbox(
                benchmark=benchmark,
                snapshot_id=None,
                role=role,
                assets_read_only=False,
                env={
                    "DEBIAN_FRONTEND": "noninteractive",
                    "PYTHONPATH": "/opt/meta-harness",
                },
            )
            try:
                archive = self._archive_sources(role)
                await sandbox.files.write_file(
                    "/tmp/native-meta-source.tar.gz",
                    archive.open("rb"),
                    mode=600,
                )
                execution = await sandbox.commands.run(
                    self._bootstrap_command(benchmark, role),
                    opts=RunCommandOpts(
                        working_directory="/",
                        timeout=timedelta(seconds=self.settings.command_timeout_sec),
                        envs=self._bootstrap_env(),
                    ),
                )
                if execution.exit_code != 0:
                    raise OpenSandboxBackendError(
                        f"runtime bootstrap failed for {benchmark} "
                        f"(exit {execution.exit_code})\n"
                        f"stdout:\n{_execution_text(execution, 'stdout')}\n"
                        f"stderr:\n{_execution_text(execution, 'stderr')}"
                    )
                identity = runtime_identity(
                    benchmark=benchmark,
                    role=role,
                    source_hash=self._source_digests[role],
                    settings=self.settings,
                    recipe_revision=self._recipe_revision(benchmark, role),
                )
                pending = await sandbox.create_snapshot(
                    name=f"native-harness-{benchmark}-{role}-{identity[:16]}"
                )
                snapshot_id = str(getattr(pending, "id", ""))
                if not snapshot_id:
                    raise OpenSandboxBackendError("OpenSandbox returned no snapshot id")
                manager = await SandboxManager.create(
                    connection_config=self._connection_config()
                )
                try:
                    deadline = time.monotonic() + self.settings.snapshot_ready_timeout_sec
                    current = pending
                    while True:
                        status = getattr(current, "status", None)
                        state = str(getattr(status, "state", "") or "").casefold()
                        if state in {"ready", "completed", "succeeded"}:
                            break
                        if state in {"failed", "error", "deleted", "terminated"}:
                            raise OpenSandboxBackendError(
                                f"snapshot {snapshot_id} entered state {state!r}"
                            )
                        if time.monotonic() >= deadline:
                            raise OpenSandboxBackendError(
                                f"snapshot {snapshot_id} did not become ready"
                            )
                        await asyncio.sleep(1)
                        current = await manager.get_snapshot(snapshot_id)
                finally:
                    await manager.close()
                return RuntimeSnapshot(
                    benchmark=benchmark,
                    role=role,
                    identity=identity,
                    snapshot_id=snapshot_id,
                    source_digest=self._source_digests[role],
                    image=self.settings.image,
                    created_at=_utc_now(),
                )
            finally:
                await sandbox.kill()
                await sandbox.close()

        return asyncio.run(build())

    def _run_worker(
        self,
        *,
        benchmark: str,
        operation: str,
        request: dict[str, Any],
        candidate_path: Path | None = None,
        state_path: Path | None = None,
    ) -> tuple[dict[str, Any], Path]:
        import asyncio

        role = "grader" if operation == "grade-artifacts" else "solver"
        snapshot = self.ensure_runtime(benchmark, role)

        async def execute() -> tuple[dict[str, Any], Path]:
            from opensandbox.models.execd import RunCommandOpts

            worker_env = self._worker_env(operation)
            sandbox = await self._create_sandbox(
                benchmark=benchmark,
                snapshot_id=snapshot.snapshot_id,
                env=worker_env,
                role=role,
                assets_read_only=True,
            )
            local_dir = Path(tempfile.mkdtemp(prefix="native-opensandbox-result-"))
            try:
                await sandbox.commands.run("rm -rf /work/job /work/result /work/result.tar.gz")
                await sandbox.files.write_file(
                    "/work/request.json",
                    json.dumps(request, ensure_ascii=False),
                    mode=600,
                )
                if candidate_path is not None:
                    await sandbox.files.write_file(
                        "/work/candidate.py", candidate_path.open("rb"), mode=600
                    )
                if state_path is not None:
                    await sandbox.files.write_file(
                        "/work/harness_store.json", state_path.open("rb"), mode=600
                    )
                worker_module = (
                    "sandbox_grader_worker"
                    if role == "grader"
                    else "sandbox_worker"
                )
                command = (
                    "python -m integrations.agentstream_sequential_meta."
                    f"{worker_module} {operation} --request /work/request.json"
                )
                execution = await sandbox.commands.run(
                    command,
                    opts=RunCommandOpts(
                        working_directory="/work",
                        timeout=timedelta(seconds=self.settings.command_timeout_sec),
                    ),
                )
                if execution.exit_code != 0:
                    raise OpenSandboxBackendError(
                        f"sandbox worker failed for {benchmark}/{operation} "
                        f"(exit {execution.exit_code})\n"
                        f"stdout:\n{_execution_text(execution, 'stdout')}\n"
                        f"stderr:\n{_execution_text(execution, 'stderr')}"
                    )
                pack = await sandbox.commands.run(
                    "tar -czf /work/result.tar.gz -C /work result"
                )
                if pack.exit_code != 0:
                    raise OpenSandboxBackendError(
                        f"failed to package sandbox result: {_execution_text(pack, 'stderr')}"
                    )
                archive = local_dir / "result.tar.gz"
                archive.write_bytes(await sandbox.files.read_bytes("/work/result.tar.gz"))
                _safe_extract(archive, local_dir)
                result_path = local_dir / "result" / "result.json"
                if not result_path.is_file():
                    raise OpenSandboxBackendError("sandbox result.json is missing")
                return json.loads(result_path.read_text(encoding="utf-8")), local_dir
            finally:
                await sandbox.kill()
                await sandbox.close()

        return asyncio.run(execute())

    def list_tasks(self, benchmark: str, config: dict[str, Any]) -> list[str]:
        payload, temporary = self._run_worker(
            benchmark=benchmark,
            operation="list-tasks",
            request={"benchmark": benchmark, "config": config},
        )
        try:
            task_ids = payload.get("task_ids")
            if not isinstance(task_ids, list):
                raise OpenSandboxBackendError("list-tasks returned invalid task_ids")
            return [str(item) for item in task_ids]
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def get_task_order(
        self,
        configs: dict[str, dict[str, Any]],
        num_tasks: int,
        ordering_seed: int,
    ) -> list[tuple[str, str]]:
        all_ids = self.get_task_inventory(configs)
        too_short = {
            benchmark: len(ids)
            for benchmark, ids in all_ids.items()
            if len(ids) < num_tasks
        }
        if too_short:
            raise OpenSandboxBackendError(
                f"benchmarks contain fewer than {num_tasks} tasks: {too_short}"
            )
        return sequential_task_order(all_ids, num_tasks, ordering_seed)

    def get_task_inventory(
        self, configs: dict[str, dict[str, Any]]
    ) -> dict[str, list[str]]:
        return {
            benchmark: self.list_tasks(benchmark, configs[benchmark])
            for benchmark in sorted(configs)
        }

    def run_block(self, **kwargs: Any):
        # Imported lazily to avoid a controller/backend circular import at module load.
        from .controller import BlockRun
        from .grading import merge_grade

        benchmark = str(kwargs["benchmark_slug"])
        output_state = Path(kwargs["output_state_path"])
        evaluation_dir = Path(kwargs["evaluation_dir"])
        public_dir_value = kwargs.get("public_dir")
        public_dir = Path(public_dir_value) if public_dir_value is not None else None
        task_ids = [str(item) for item in kwargs["task_ids"]]
        split_names = [str(item) for item in kwargs["split_names"]]
        if len(task_ids) != len(split_names):
            raise ValueError("task_ids and split_names must have the same length")

        output_state.parent.mkdir(parents=True, exist_ok=True)
        temporary_state = output_state.with_name(f".{output_state.name}.next")
        shutil.copy2(Path(kwargs["input_state_path"]), temporary_state)
        os.replace(temporary_state, output_state)
        if evaluation_dir.exists():
            shutil.rmtree(evaluation_dir)
        evaluation_dir.mkdir(parents=True)
        if public_dir is not None:
            if public_dir.exists():
                shutil.rmtree(public_dir)
            (public_dir / "rollouts").mkdir(parents=True)

        config = dict(kwargs["config"])
        tasks_per_worker = int(config.get("sandbox_tasks_per_worker", 10))
        if tasks_per_worker < 1:
            raise ValueError("sandbox_tasks_per_worker must be positive")

        rows: list[dict[str, Any]] = []
        candidate_path = Path(kwargs["candidate_path"])
        for chunk_start in range(0, len(task_ids), tasks_per_worker):
            chunk_task_ids = task_ids[chunk_start : chunk_start + tasks_per_worker]
            chunk_split_names = split_names[
                chunk_start : chunk_start + tasks_per_worker
            ]
            committed = False
            last_error: Exception | None = None
            chunk_rows: list[dict[str, Any]] = []
            grading_artifacts: list[dict[str, Any]] = []
            solver_attempts = 0
            for sandbox_attempt in range(1, 4):
                solver_attempts = sandbox_attempt
                temporary: Path | None = None
                request = {
                    "benchmark_slug": benchmark,
                    "task_ids": chunk_task_ids,
                    "split_names": chunk_split_names,
                    # Solver workers receive no verifier configuration.
                    "config": {
                        key: value
                        for key, value in config.items()
                        if key != "grader_kwargs"
                    },
                    "base_model": kwargs["base_model"],
                    "max_tokens": kwargs["max_tokens"],
                    "embedding_model": kwargs["embedding_model"],
                    "task_attempts": 3,
                    "public": public_dir is not None,
                }
                try:
                    payload, temporary = self._run_worker(
                        benchmark=benchmark,
                        operation="run-solver-block",
                        request=request,
                        candidate_path=candidate_path,
                        state_path=output_state,
                    )
                    raw_rows = payload.get("rows")
                    raw_artifacts = payload.get("grading_artifacts", [])
                    if not isinstance(raw_rows, list) or len(raw_rows) != len(
                        chunk_task_ids
                    ):
                        raise OpenSandboxBackendError(
                            "sandbox task chunk returned invalid rows"
                        )
                    if not isinstance(raw_artifacts, list):
                        raise OpenSandboxBackendError(
                            "sandbox task chunk returned invalid grading artifacts"
                        )
                    chunk_rows = [dict(item) for item in raw_rows]
                    grading_artifacts = [dict(item) for item in raw_artifacts]
                    awaiting = sum(
                        row.get("status") == "awaiting_grader" for row in chunk_rows
                    )
                    if awaiting != len(grading_artifacts):
                        raise OpenSandboxBackendError(
                            "solver rows and grading artifacts are misaligned"
                        )
                    result_root = temporary / "result"
                    remote_state = result_root / "harness_store.json"
                    remote_evaluation = result_root / "evaluation"
                    if not remote_state.is_file() or not remote_evaluation.is_dir():
                        raise OpenSandboxBackendError(
                            "sandbox task is missing state or evaluation artifacts"
                        )
                    shutil.copytree(
                        remote_evaluation, evaluation_dir, dirs_exist_ok=True
                    )
                    shutil.copy2(remote_state, temporary_state)
                    os.replace(temporary_state, output_state)
                    if public_dir is not None:
                        remote_rollouts = result_root / "public" / "rollouts"
                        if remote_rollouts.is_dir():
                            shutil.copytree(
                                remote_rollouts,
                                public_dir
                                / "rollouts"
                                / f"chunk_{chunk_start:04d}",
                                dirs_exist_ok=True,
                            )
                    committed = True
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if sandbox_attempt >= 3:
                        break
                finally:
                    if temporary is not None:
                        shutil.rmtree(temporary, ignore_errors=True)
            if not committed:
                for task_id, split_name in zip(
                    chunk_task_ids, chunk_split_names, strict=True
                ):
                    rows.append(
                        {
                            "task_id": task_id,
                            "split": split_name,
                            "score": 0.0,
                            "success": False,
                            "status": "error",
                            "steps": 0,
                            "action_count": 0,
                            "agent_cost": 0.0,
                            "benchmark_cost": 0.0,
                            "execution_time": 0.0,
                            "model_calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "grader_input_tokens": 0,
                            "grader_output_tokens": 0,
                            "attempts": 3,
                            "sandbox_attempts": 3,
                            "retryable": True,
                            "error": (
                                f"{type(last_error).__name__}: {last_error}"
                            ),
                        }
                    )
                continue

            # The solver sandbox is already killed by _run_worker, and its state
            # was committed above. Only now may a private verifier be started.
            grade_results: list[dict[str, Any]] = []
            grader_sandbox_attempts = 0
            grader_error: Exception | None = None
            if grading_artifacts:
                for grader_sandbox_attempt in range(1, 4):
                    grader_sandbox_attempts = grader_sandbox_attempt
                    temporary = None
                    try:
                        grade_payload, temporary = self._run_worker(
                            benchmark=benchmark,
                            operation="grade-artifacts",
                            request={
                                "benchmark_slug": benchmark,
                                "grading_artifacts": grading_artifacts,
                                "config": config,
                                "grader_attempts": 3,
                            },
                        )
                        raw_results = grade_payload.get("grade_results")
                        if not isinstance(raw_results, list) or len(raw_results) != len(
                            grading_artifacts
                        ):
                            raise OpenSandboxBackendError(
                                "grader returned invalid result count"
                            )
                        grade_results = [dict(item) for item in raw_results]
                        grader_error = None
                        break
                    except Exception as exc:
                        grader_error = exc
                    finally:
                        if temporary is not None:
                            shutil.rmtree(temporary, ignore_errors=True)

            result_iter = iter(grade_results)
            for row in chunk_rows:
                row["sandbox_attempts"] = solver_attempts
                if row.get("status") != "awaiting_grader":
                    rows.append(row)
                    continue
                if grader_error is not None:
                    row.update(
                        {
                            "score": 0.0,
                            "success": False,
                            "status": "grader_error",
                            "retryable": True,
                            "grader_sandbox_attempts": grader_sandbox_attempts,
                            "error": f"{type(grader_error).__name__}: {grader_error}",
                        }
                    )
                    rows.append(row)
                    continue
                result = next(result_iter)
                row["grader_sandbox_attempts"] = grader_sandbox_attempts
                row["grader_attempts"] = int(result.get("grader_attempts", 0))
                if "score" in result:
                    row = merge_grade(row, dict(result["score"]))
                    score_dir = evaluation_dir / "private_scores"
                    score_dir.mkdir(parents=True, exist_ok=True)
                    (score_dir / f"{row['task_id']}.json").write_text(
                        json.dumps(result["score"], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    row.update(
                        {
                            "score": 0.0,
                            "success": False,
                            "status": "grader_error",
                            "retryable": True,
                            "error": str(result.get("error", "private grader failed")),
                        }
                    )
                rows.append(row)

        if public_dir is not None:
            public_rows = []
            for row in rows:
                public_row = {
                    key: value for key, value in row.items() if key != "error"
                }
                if "error" in row:
                    public_row["error_type"] = str(row["error"]).split(
                        ":", 1
                    )[0]
                public_rows.append(public_row)
            (public_dir / "metrics.json").write_text(
                json.dumps(
                    {"tasks": public_rows}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        (evaluation_dir / "task_rows.json").write_text(
            json.dumps({"tasks": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return BlockRun(rows=rows, state_path=output_state)


def safe_remote_path(path: str) -> PurePosixPath:
    """Validate a future remote artifact path (kept public for unit tests)."""

    value = PurePosixPath(path)
    if not value.is_absolute() or value == PurePosixPath("/") or ".." in value.parts:
        raise ValueError("remote path must be a safe absolute path")
    return value
