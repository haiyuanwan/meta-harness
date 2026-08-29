"""BFCL checker, intentionally excluded from solver runtimes."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .base import PrivateGrader, validate_artifact

NATIVE_MODEL_NAME = "meta-harness-native-fc"


@lru_cache(maxsize=1)
def _symbols() -> dict[str, Any]:
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    return {
        "Language": Language,
        "ast_checker": ast_checker,
        "multi_turn_checker": multi_turn_checker,
        "load_dataset_entry": load_dataset_entry,
        "load_ground_truth_entry": load_ground_truth_entry,
    }


def _is_multi_turn(subset: str) -> bool:
    return subset.startswith("multi_turn_")


def _is_relevance(subset: str) -> bool:
    return subset in {"irrelevance", "live_irrelevance", "live_relevance"}


def _language(subset: str) -> Any:
    language = _symbols()["Language"]
    if subset == "simple_java":
        return language.JAVA
    if subset == "simple_javascript":
        return language.JAVASCRIPT
    return language.PYTHON


class BFCLGrader(PrivateGrader):
    def __init__(self, *, subset: str = "multi_turn_base") -> None:
        self.subset = subset
        symbols = _symbols()
        self._prompts = {
            str(item["id"]): item for item in symbols["load_dataset_entry"](subset)
        }
        self._answers = (
            {}
            if _is_relevance(subset)
            else {
                str(item["id"]): item
                for item in symbols["load_ground_truth_entry"](subset)
            }
        )

    def grade(self, artifact: dict[str, Any]) -> dict[str, Any]:
        validate_artifact(artifact, "bfcl")
        task_id = artifact["task_id"]
        if artifact.get("subset") != self.subset:
            raise ValueError("BFCL subset mismatch")
        if task_id not in self._prompts:
            raise KeyError(f"Unknown BFCL task id: {task_id}")
        flat_actions = list(artifact.get("flat_actions", []))
        if not artifact.get("is_finished", False):
            return {
                "score": 0.0,
                "success": False,
                "is_finished": False,
                "session_metrics": {
                    "completed_turns": int(artifact.get("completed_turns", 0)),
                    "action_count": len(flat_actions),
                },
            }
        if _is_relevance(self.subset):
            success = (
                len(flat_actions) == 0
                if "irrelevance" in self.subset
                else len(flat_actions) > 0
            )
            checker = {"valid": success}
        elif task_id not in self._answers:
            raise ValueError(f"Missing BFCL answer for task {task_id}")
        elif _is_multi_turn(self.subset):
            checker = _symbols()["multi_turn_checker"](
                artifact.get("turn_calls", []),
                self._answers[task_id]["ground_truth"],
                self._prompts[task_id],
                self.subset,
                f"{NATIVE_MODEL_NAME}_{artifact.get('attempt_id', 'grader')}_score",
            )
        else:
            checker = _symbols()["ast_checker"](
                self._prompts[task_id]["function"],
                flat_actions,
                self._answers[task_id]["ground_truth"],
                _language(self.subset),
                self.subset,
                NATIVE_MODEL_NAME,
            )
        success = bool(checker.get("valid", False))
        return {
            "score": float(success),
            "success": success,
            "is_finished": True,
            "session_metrics": {
                "accuracy": float(success),
                "action_count": len(flat_actions),
            },
            "private_grader_details": checker,
        }
