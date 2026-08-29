"""BrowseCompPlus backend built directly on its official Python package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..harness_protocol import HarnessStep, ToolCall, ToolResult, ToolSpec
from .base import BenchmarkBackend, TaskEnvironment


def _default_assets_dir() -> Path:
    configured = os.environ.get("BROWSECOMPPLUS_ASSETS_DIR")
    if configured:
        return Path(configured)
    return Path("/opt/benchmark-assets/browsecompplus")


class BrowseCompPlusBackend(BenchmarkBackend):
    def __init__(
        self,
        *,
        assets_dir: str | None = None,
        searcher_type: str = "faiss",
        searcher_model_name: str = "Qwen/Qwen3-Embedding-8B",
        normalize_search: bool = True,
        top_k_docs: int = 5,
        max_snippet_tokens: int = 512,
        full_doc_max_tokens: int = 2048,
        include_get_document: bool = True,
        max_interactions: int = 100,
    ) -> None:
        self.assets_dir = Path(assets_dir) if assets_dir else _default_assets_dir()
        self.searcher_type = searcher_type
        self.searcher_model_name = searcher_model_name
        self.normalize_search = normalize_search
        self.top_k_docs = top_k_docs
        self.max_snippet_tokens = max_snippet_tokens
        self.full_doc_max_tokens = full_doc_max_tokens
        self.include_get_document = include_get_document
        self.max_interactions = max_interactions
        self._tasks: dict[str, dict[str, Any]] | None = None
        self._searcher: Any = None
        self._tokenizer: Any = None

    def _ensure_tasks(self) -> None:
        if self._tasks is not None:
            return
        path = self.assets_dir / "data" / "browsecomp_plus_solver.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"BrowseCompPlus dataset is missing at {path}; prepare benchmark assets first"
            )
        tasks: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            task_id = str(raw["query_id"])
            tasks[task_id] = {
                "query_id": task_id,
                "query": raw["query"],
            }
        self._tasks = tasks

    def _ensure_searcher(self) -> None:
        if self._searcher is not None:
            return
        import safetensors  # noqa: F401
        import torch  # noqa: F401

        mounted_model = self.assets_dir / "models" / "Qwen3-Embedding-8B"
        mounted_corpus = self.assets_dir / "corpus"
        has_mounted_model = (mounted_model / "config.json").is_file()
        has_mounted_corpus = (mounted_corpus / "state.json").is_file()
        cache_root: Path | None = None
        old_cache_env: dict[str, str | None] = {}
        try:
            if self.searcher_type == "faiss" and not has_mounted_model:
                # Qwen3-Embedding-8B and the BrowseCompPlus corpus together need
                # more than 20 GiB on first use.  Both are fully materialized in
                # memory by the upstream searcher, so retaining their Hub caches
                # for the lifetime of this one-task sandbox only wastes overlay
                # space.  Keep the two downloads separate and reclaim each as
                # soon as the corresponding object has loaded.
                cache_root = Path(
                    tempfile.mkdtemp(prefix="browsecompplus-hf-", dir="/tmp")
                )
                cache_env = {
                    "HF_HOME": str(cache_root / "model"),
                    "HF_DATASETS_CACHE": str(cache_root / "dataset"),
                }
                for key, value in cache_env.items():
                    old_cache_env[key] = os.environ.get(key)
                    os.environ[key] = value

            from searcher.searchers import SearcherType

            searcher_class = SearcherType.get_searcher_class(self.searcher_type)
            if self.searcher_type == "faiss" and has_mounted_corpus:
                upstream_searcher_class = searcher_class

                class MountedCorpusFaissSearcher(upstream_searcher_class):
                    def _load_dataset(inner_self) -> None:
                        from datasets import load_from_disk

                        dataset = load_from_disk(str(mounted_corpus))
                        inner_self.docid_to_text = {
                            row["docid"]: row["text"] for row in dataset
                        }

                searcher_class = MountedCorpusFaissSearcher
            elif cache_root is not None:
                upstream_searcher_class = searcher_class
                model_cache = cache_root / "model"
                dataset_cache = cache_root / "dataset"

                class DiskBoundedFaissSearcher(upstream_searcher_class):
                    def _load_dataset(inner_self) -> None:
                        shutil.rmtree(model_cache, ignore_errors=True)
                        super()._load_dataset()
                        shutil.rmtree(dataset_cache, ignore_errors=True)

                searcher_class = DiskBoundedFaissSearcher

            parser = argparse.ArgumentParser(add_help=False)
            searcher_class.parse_args(parser)
            if self.searcher_type == "bm25":
                searcher_args: dict[str, Any] = {
                    "index_path": str(self.assets_dir / "indexes" / "bm25")
                }
            else:
                model_dir = self.searcher_model_name.lower().split("/")[-1]
                index_dir = self.assets_dir / "indexes" / model_dir
                if not index_dir.is_dir():
                    raise FileNotFoundError(
                        f"BrowseCompPlus index is missing: {index_dir}"
                    )
                searcher_args = {
                    "index_path": str(index_dir / "corpus.shard*_of_4.pkl"),
                    "model_name": (
                        str(mounted_model)
                        if has_mounted_model
                        else self.searcher_model_name
                    ),
                    "normalize": self.normalize_search,
                }
            cli: list[str] = []
            for key, value in searcher_args.items():
                flag = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        cli.append(flag)
                else:
                    cli.extend([flag, str(value)])
            self._searcher = searcher_class(parser.parse_args(cli))
        finally:
            for key, value in old_cache_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if cache_root is not None:
                shutil.rmtree(cache_root, ignore_errors=True)

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            mounted_tokenizer = (
                self.assets_dir / "models" / "Qwen3-0.6B-tokenizer"
            )
            if (mounted_tokenizer / "tokenizer_config.json").is_file():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    mounted_tokenizer, local_files_only=True
                )
                return
            cache_dir = Path(
                tempfile.mkdtemp(prefix="browsecompplus-tokenizer-", dir="/tmp")
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-0.6B", cache_dir=cache_dir
                )
            finally:
                shutil.rmtree(cache_dir, ignore_errors=True)

    def list_tasks(self) -> list[str]:
        self._ensure_tasks()
        assert self._tasks is not None
        return list(self._tasks)

    def open_task(self, task_id: str, *, attempt_id: str) -> TaskEnvironment:
        del attempt_id
        self._ensure_tasks()
        self._ensure_searcher()
        self._ensure_tokenizer()
        assert self._tasks is not None
        if task_id not in self._tasks:
            raise KeyError(f"Unknown BrowseCompPlus task id: {task_id}")
        return BrowseCompPlusEnvironment(
            instance=dict(self._tasks[task_id]),
            searcher=self._searcher,
            tokenizer=self._tokenizer,
            top_k_docs=self.top_k_docs,
            max_snippet_tokens=self.max_snippet_tokens,
            full_doc_max_tokens=self.full_doc_max_tokens,
            include_get_document=self.include_get_document,
            max_interactions=self.max_interactions,
        )

    def close(self) -> None:
        if self._searcher is not None:
            close = getattr(self._searcher, "close", None)
            if callable(close):
                close()


class BrowseCompPlusEnvironment(TaskEnvironment):
    def __init__(
        self,
        *,
        instance: dict[str, Any],
        searcher: Any,
        tokenizer: Any,
        top_k_docs: int,
        max_snippet_tokens: int,
        full_doc_max_tokens: int,
        include_get_document: bool,
        max_interactions: int,
    ) -> None:
        self.instance = instance
        self.searcher = searcher
        self.tokenizer = tokenizer
        self.top_k_docs = top_k_docs
        self.max_snippet_tokens = max_snippet_tokens
        self.full_doc_max_tokens = full_doc_max_tokens
        self.include_get_document = include_get_document
        self.max_interactions = max_interactions
        self._done = False
        self._response: dict[str, Any] | None = None
        self._retrieved_docids: set[str] = set()
        self._tool_counts: dict[str, int] = {}
        self._interactions = 0

    @property
    def task_id(self) -> str:
        return str(self.instance["query_id"])

    @property
    def task(self) -> str:
        return str(self.instance["query"])

    @property
    def tools(self) -> list[ToolSpec]:
        tools = [
            ToolSpec(
                name="search",
                description="Search the corpus for documents relevant to a query.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
        if self.include_get_document:
            tools.append(
                ToolSpec(
                    name="get_document",
                    description="Retrieve a complete document by document id.",
                    parameters={
                        "type": "object",
                        "properties": {"docid": {"type": "string"}},
                        "required": ["docid"],
                    },
                )
            )
        tools.append(
            ToolSpec(
                name="submit",
                description="Submit the final answer and finish the task.",
                parameters={
                    "type": "object",
                    "properties": {
                        "exact_answer": {
                            "type": "string",
                            "description": "A succinct final answer.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Evidence-based explanation with [docid] citations.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence from 0 to 100.",
                        },
                    },
                    "required": ["exact_answer", "explanation", "confidence"],
                },
                is_finish=True,
            )
        )
        return tools

    def done(self) -> bool:
        return self._done

    def step(self, step: HarnessStep) -> tuple[list[ToolResult], int]:
        calls = list(step.tool_calls)
        if not calls and step.final_text:
            calls = [
                ToolCall(
                    id="final-text",
                    name="submit",
                    arguments={
                        "exact_answer": step.final_text,
                        "explanation": "",
                        "confidence": 0.0,
                    },
                )
            ]
        results: list[ToolResult] = []
        executed = 0
        available = {tool.name for tool in self.tools}
        for call in calls:
            if self._done:
                break
            if call.name not in available:
                results.append(
                    ToolResult(call.id, call.name, f"Unknown tool: {call.name}", True)
                )
                continue
            self._interactions += 1
            self._tool_counts[call.name] = self._tool_counts.get(call.name, 0) + 1
            if self._interactions > self.max_interactions:
                self._done = True
                results.append(
                    ToolResult(
                        call.id,
                        call.name,
                        "Maximum interaction count reached.",
                        True,
                    )
                )
                break
            try:
                if call.name == "search":
                    content = self._search(str(call.arguments["query"]))
                elif call.name == "get_document":
                    content = self._get_document(str(call.arguments["docid"]))
                else:
                    self._response = {
                        "exact_answer": str(call.arguments["exact_answer"]),
                        "explanation": str(call.arguments["explanation"]),
                        "confidence": float(call.arguments["confidence"]),
                    }
                    self._done = True
                    content = "Answer submitted."
                results.append(ToolResult(call.id, call.name, content))
                executed += 1
            except (KeyError, TypeError, ValueError) as exc:
                results.append(ToolResult(call.id, call.name, str(exc), True))
        return results, executed

    def _truncate(self, text: str, limit: int) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if limit > 0 and len(tokens) > limit:
            return self.tokenizer.decode(tokens[:limit], skip_special_tokens=True)
        return text

    def _search(self, query: str) -> str:
        candidates = self.searcher.search(query, self.top_k_docs)
        results = []
        for candidate in candidates:
            docid = str(candidate["docid"])
            self._retrieved_docids.add(docid)
            item = {
                "docid": docid,
                "snippet": self._truncate(
                    str(candidate.get("text", "")), self.max_snippet_tokens
                ),
            }
            if candidate.get("score") is not None:
                item["score"] = candidate["score"]
            results.append(item)
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _get_document(self, docid: str) -> str:
        try:
            document = self.searcher.get_document(docid)
        except Exception:
            document = None
        if document is None:
            return json.dumps({"error": f"Document {docid} not found"})
        result = dict(document)
        result["text"] = self._truncate(
            str(result.get("text", "")), self.full_doc_max_tokens
        )
        self._retrieved_docids.add(str(result.get("docid", docid)))
        return json.dumps(result, ensure_ascii=False, indent=2)

    def grading_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": "browsecompplus",
            "task_id": self.task_id,
            "is_finished": self._done and self._response is not None,
            "response": self._response,
            "retrieved_docids": sorted(self._retrieved_docids),
            "tool_call_counts": dict(self._tool_counts),
        }
