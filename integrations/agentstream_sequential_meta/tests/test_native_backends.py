from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType

from integrations.agentstream_sequential_meta.benchmark_backends import bfcl
from integrations.agentstream_sequential_meta.benchmark_backends.browsecompplus import (
    BrowseCompPlusBackend,
    BrowseCompPlusEnvironment,
)
from integrations.agentstream_sequential_meta.benchmark_backends.prepare_browsecompplus import (
    create_split_datasets,
)
from integrations.agentstream_sequential_meta.benchmark_graders import bfcl as bfcl_grader
from integrations.agentstream_sequential_meta.benchmark_graders.browsecompplus import (
    BrowseCompPlusGrader,
)
from integrations.agentstream_sequential_meta.harness_protocol import (
    HarnessStep,
    ToolCall,
)


def _fake_bfcl_symbols():
    class ModelStyle:
        OPENAI_COMPLETIONS = "openai"

    def convert_to_tool(functions, *_):
        return [
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "parameters": item["parameters"],
                },
            }
            for item in functions
        ]

    def execute_multi_turn_func_call(**kwargs):
        return [f"executed:{item}" for item in kwargs["func_call_list"]], None

    def multi_turn_checker(model_turns, ground_truth, *_):
        return {
            "valid": bool(model_turns[0])
            and len(model_turns) == len(ground_truth)
        }

    return {
        "ModelStyle": ModelStyle,
        "GORILLA_TO_OPENAPI": {},
        "convert_to_tool": convert_to_tool,
        "execute_multi_turn_func_call": execute_multi_turn_func_call,
        "multi_turn_checker": multi_turn_checker,
    }


def test_bfcl_native_environment_handles_multi_turn_tools(monkeypatch) -> None:
    monkeypatch.setattr(bfcl, "_symbols", _fake_bfcl_symbols)
    prompt = {
        "id": "bfcl-1",
        "question": [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        "function": [
            {
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            }
        ],
        "initial_config": {},
        "involved_classes": [],
    }
    environment = bfcl.BFCLEnvironment(
        subset="multi_turn_base",
        prompt_entry=prompt,
        attempt_id="attempt",
    )

    first_results, first_count = environment.step(
        HarnessStep(
            tool_calls=(
                ToolCall("call-1", "lookup", {"key": "a"}),
                ToolCall("finish-1", "finish", {}),
            )
        )
    )

    assert first_count == 2
    assert not environment.done()
    assert any(item.name == "environment" and item.content == "second" for item in first_results)

    environment.step(
        HarnessStep(
            tool_calls=(
                ToolCall("call-2", "lookup", {"key": "b"}),
                ToolCall("finish-2", "finish", {}),
            )
        )
    )

    assert environment.done()
    artifact = environment.grading_artifact()
    assert artifact["is_finished"] is True
    assert artifact["turn_calls"] == [[['lookup(key=\'a\')']], [['lookup(key=\'b\')']]]
    assert "ground_truth" not in json.dumps(artifact)
    monkeypatch.setattr(bfcl_grader, "_symbols", _fake_bfcl_symbols)
    grader = object.__new__(bfcl_grader.BFCLGrader)
    grader.subset = "multi_turn_base"
    grader._prompts = {"bfcl-1": prompt}
    grader._answers = {"bfcl-1": {"ground_truth": [["a"], ["b"]]}}
    assert grader.grade(artifact)["score"] == 1.0


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(tokens)


class FakeSearcher:
    def search(self, query, k):
        assert query == "query"
        assert k == 2
        return [
            {"docid": "d1", "text": "one two three", "score": 1.0},
            {"docid": "d2", "text": "four five", "score": 0.5},
        ]

    def get_document(self, docid):
        return {"docid": docid, "text": "complete document"}


def test_browse_faiss_reclaims_one_task_hub_caches(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, Path] = {}

    class FakeFaissSearcher:
        @classmethod
        def parse_args(cls, parser) -> None:
            parser.add_argument("--index-path")
            parser.add_argument("--model-name")
            parser.add_argument("--normalize", action="store_true")

        def __init__(self, args) -> None:
            self.args = args
            self._load_model()
            self._load_dataset()

        def _load_model(self) -> None:
            model_cache = Path(os.environ["HF_HOME"])
            model_cache.mkdir(parents=True)
            (model_cache / "weights").write_text("loaded", encoding="utf-8")
            observed["model"] = model_cache

        def _load_dataset(self) -> None:
            assert not observed["model"].exists()
            dataset_cache = Path(os.environ["HF_DATASETS_CACHE"])
            dataset_cache.mkdir(parents=True)
            (dataset_cache / "corpus").write_text("loaded", encoding="utf-8")
            observed["dataset"] = dataset_cache

    class FakeSearcherType:
        @staticmethod
        def get_searcher_class(searcher_type):
            assert searcher_type == "faiss"
            return FakeFaissSearcher

    searcher_package = ModuleType("searcher")
    searchers_module = ModuleType("searcher.searchers")
    searchers_module.SearcherType = FakeSearcherType
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "searcher", searcher_package)
    monkeypatch.setitem(sys.modules, "searcher.searchers", searchers_module)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    assets = tmp_path / "assets"
    (assets / "indexes" / "qwen3-embedding-8b").mkdir(parents=True)
    backend = BrowseCompPlusBackend(assets_dir=str(assets))
    backend._ensure_searcher()

    assert isinstance(backend._searcher, FakeFaissSearcher)
    assert not observed["model"].exists()
    assert not observed["dataset"].exists()
    assert "HF_HOME" not in os.environ
    assert "HF_DATASETS_CACHE" not in os.environ


def test_browse_faiss_uses_mounted_model_and_corpus(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeFaissSearcher:
        @classmethod
        def parse_args(cls, parser) -> None:
            parser.add_argument("--index-path")
            parser.add_argument("--model-name")
            parser.add_argument("--normalize", action="store_true")

        def __init__(self, args) -> None:
            self.args = args
            self.docid_to_text = None
            self._load_dataset()

        def _load_dataset(self) -> None:
            raise AssertionError("mounted corpus should replace the Hub loader")

    class FakeSearcherType:
        @staticmethod
        def get_searcher_class(searcher_type):
            assert searcher_type == "faiss"
            return FakeFaissSearcher

    searcher_package = ModuleType("searcher")
    searchers_module = ModuleType("searcher.searchers")
    searchers_module.SearcherType = FakeSearcherType
    datasets_module = ModuleType("datasets")
    datasets_module.load_from_disk = lambda path: [
        {"docid": "doc-1", "text": f"loaded:{path}"}
    ]
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "searcher", searcher_package)
    monkeypatch.setitem(sys.modules, "searcher.searchers", searchers_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    assets = tmp_path / "assets"
    (assets / "indexes" / "qwen3-embedding-8b").mkdir(parents=True)
    model = assets / "models" / "Qwen3-Embedding-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    corpus = assets / "corpus"
    corpus.mkdir()
    (corpus / "state.json").write_text("{}", encoding="utf-8")

    backend = BrowseCompPlusBackend(assets_dir=str(assets))
    backend._ensure_searcher()

    assert backend._searcher.args.model_name == str(model)
    assert backend._searcher.docid_to_text == {
        "doc-1": f"loaded:{corpus}"
    }
    assert "HF_HOME" not in os.environ
    assert "HF_DATASETS_CACHE" not in os.environ


def _install_fake_browse_grader(monkeypatch) -> None:
    litellm = ModuleType("litellm")
    litellm.completion = lambda **kwargs: {
        "choices": [{"message": {"content": "judge-output"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    litellm.completion_cost = lambda **kwargs: 0.25
    package = ModuleType("scripts_evaluation")
    grader = ModuleType("scripts_evaluation.evaluate_with_openai")
    grader.GRADER_TEMPLATE = "{question}|{response}|{correct_answer}"
    grader.compute_citation_metrics = lambda citations, positives: {
        "recall": len(set(citations) & set(positives)) / len(positives)
    }
    grader.extract_citations_from_response = lambda response: ["d1"]
    grader.parse_judge_response = lambda response: {
        "correct": response == "judge-output",
        "parse_error": False,
        "extracted_final_answer": "answer",
    }
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setitem(sys.modules, "scripts_evaluation", package)
    monkeypatch.setitem(
        sys.modules, "scripts_evaluation.evaluate_with_openai", grader
    )


def test_browse_solver_artifact_is_scored_by_separate_private_grader(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_browse_grader(monkeypatch)
    environment = BrowseCompPlusEnvironment(
        instance={
            "query_id": "browse-1",
            "query": "question",
        },
        searcher=FakeSearcher(),
        tokenizer=FakeTokenizer(),
        top_k_docs=2,
        max_snippet_tokens=2,
        full_doc_max_tokens=10,
        include_get_document=True,
        max_interactions=10,
    )

    search_results, _ = environment.step(
        HarnessStep(tool_calls=(ToolCall("s1", "search", {"query": "query"}),))
    )
    assert "secret-answer" not in search_results[0].content
    assert "one two" in search_results[0].content

    environment.step(
        HarnessStep(
            tool_calls=(
                ToolCall(
                    "submit-1",
                    "submit",
                    {
                        "exact_answer": "answer",
                        "explanation": "evidence [d1]",
                        "confidence": 90,
                    },
                ),
            )
        )
    )

    artifact = environment.grading_artifact()
    assert "secret-answer" not in json.dumps(artifact)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "browsecomp_plus_grader.jsonl").write_text(
        json.dumps(
            {
                "query_id": "browse-1",
                "query": "question",
                "answer": "secret-answer",
                "evidence_docs": ["d1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    score = BrowseCompPlusGrader(
        assets_dir=str(tmp_path), grader_model="fixed-grader"
    ).grade(artifact)
    assert score["score"] == 1.0
    assert score["session_metrics"]["retrieval_recall"] == 1.0
    assert score["session_metrics"]["citation_metrics_positives"] == {
        "recall": 1.0
    }
    assert score["grader_usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "cost": 0.25,
    }
    assert "gold_answer" not in json.dumps(score)


def test_dataset_split_keeps_solver_file_free_of_private_fields(tmp_path: Path) -> None:
    source = tmp_path / "full.jsonl"
    solver = tmp_path / "solver.jsonl"
    grader = tmp_path / "grader.jsonl"
    source.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "question",
                "answer": "answer",
                "gold_docs": [{"docid": "g1", "text": "private"}],
                "evidence_docs": [{"docid": "e1", "text": "private"}],
                "negative_docs": [],
                "unexpected_private_field": "must-not-survive",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    create_split_datasets(source, solver, grader)

    public = json.loads(solver.read_text(encoding="utf-8"))
    assert public == {"query_id": "q1", "query": "question"}
    assert "answer" not in solver.read_text(encoding="utf-8")
    result = json.loads(grader.read_text(encoding="utf-8"))
    assert set(result) == {
        "query_id",
        "query",
        "answer",
        "gold_docs",
        "evidence_docs",
        "negative_docs",
    }
    assert result["gold_docs"] == ["g1"]
    assert result["evidence_docs"] == ["e1"]
    assert "private" not in grader.read_text(encoding="utf-8")
