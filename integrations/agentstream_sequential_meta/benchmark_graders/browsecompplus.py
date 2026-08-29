"""BrowseCompPlus answer judge, intentionally absent from solver runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import PrivateGrader, validate_artifact


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class BrowseCompPlusGrader(PrivateGrader):
    def __init__(
        self,
        *,
        assets_dir: str | None = None,
        grader_model: str = "anthropic/Claude-Opus-4.6-hq",
    ) -> None:
        root = Path(
            assets_dir
            or os.environ.get(
                "BROWSECOMPPLUS_ASSETS_DIR", "/opt/benchmark-assets/browsecompplus"
            )
        )
        path = root / "data" / "browsecomp_plus_grader.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"BrowseCompPlus private grader data missing: {path}")
        self.grader_model = grader_model
        self._tasks = {
            str(item["query_id"]): item
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
        }

    def grade(self, artifact: dict[str, Any]) -> dict[str, Any]:
        validate_artifact(artifact, "browsecompplus")
        task_id = artifact["task_id"]
        instance = self._tasks.get(task_id)
        if instance is None:
            raise KeyError(f"Unknown BrowseCompPlus task id: {task_id}")
        response = artifact.get("response")
        retrieved = {str(item) for item in artifact.get("retrieved_docids", [])}
        evidence = {str(item) for item in instance.get("evidence_docs", [])}
        metrics: dict[str, Any] = {
            "retrieval_recall": (
                len(retrieved & evidence) / len(evidence) if evidence else None
            ),
            "retrieved_document_count": len(retrieved),
            "tool_call_counts": dict(artifact.get("tool_call_counts", {})),
        }
        if not artifact.get("is_finished", False) or response is None:
            return {
                "score": 0.0,
                "success": False,
                "is_finished": False,
                "session_metrics": metrics,
            }

        import litellm
        from scripts_evaluation.evaluate_with_openai import (
            GRADER_TEMPLATE,
            compute_citation_metrics,
            extract_citations_from_response,
            parse_judge_response,
        )

        response_text = json.dumps(response, ensure_ascii=False)
        prompt = GRADER_TEMPLATE.format(
            question=instance["query"],
            response=response_text,
            correct_answer=instance["answer"],
        )
        judge = litellm.completion(
            model=self.grader_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            max_retries=2,
            temperature=0,
        )
        choice = _field(judge, "choices", [])[0]
        message = _field(choice, "message", {})
        parsed = parse_judge_response(str(_field(message, "content", "")))
        success = bool(parsed.get("correct", False)) and not bool(
            parsed.get("parse_error", False)
        )
        citations = extract_citations_from_response(response_text)
        metrics.update(
            {
                "accuracy": float(success),
                "citation_count": len(citations),
                "citation_metrics_positives": compute_citation_metrics(
                    citations, sorted(evidence)
                ),
                "confidence": response["confidence"],
            }
        )
        usage = _field(judge, "usage", {}) or {}
        grader_cost = 0.0
        try:
            grader_cost = float(
                litellm.completion_cost(completion_response=judge) or 0.0
            )
        except Exception:
            pass
        return {
            "score": float(success),
            "success": success,
            "is_finished": True,
            "session_metrics": metrics,
            "private_grader_details": {
                "parse_error": bool(parsed.get("parse_error", False)),
                "extracted_final_answer": parsed.get("extracted_final_answer"),
            },
            "grader_usage": {
                "input_tokens": int(_field(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(_field(usage, "completion_tokens", 0) or 0),
                "cost": grader_cost,
            },
        }
