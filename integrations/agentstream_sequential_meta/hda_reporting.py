"""Prepare the HDA significance review without crossing approval checkpoints."""

from __future__ import annotations

import difflib
import hashlib
import html
import json
from pathlib import Path
from typing import Any

PAIR_PRIORITY = (
    (
        "bfcl_to_browse_transfer",
        "H0",
        "H1",
        "browsecompplus",
        "BFCL → BrowseCompPlus transfer",
    ),
    (
        "bfcl_backward_transfer",
        "H1",
        "H2",
        "bfcl",
        "BrowseCompPlus evolution → BFCL backward transfer/forgetting",
    ),
    (
        "bfcl_in_domain_learning",
        "H0",
        "H1",
        "bfcl",
        "BFCL in-domain learning",
    ),
    (
        "browse_in_domain_learning",
        "H1",
        "H2",
        "browsecompplus",
        "BrowseCompPlus in-domain learning",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _compute_summary(cell: dict[str, Any]) -> dict[str, float]:
    tasks = [task for stream in cell["streams"] for task in stream["tasks"]]
    if not tasks:
        return {
            "tasks": 0,
            "mean_model_calls": 0.0,
            "mean_tokens": 0.0,
            "mean_steps": 0.0,
        }
    return {
        "tasks": len(tasks),
        "mean_model_calls": sum(float(row.get("model_calls", 0)) for row in tasks)
        / len(tasks),
        "mean_tokens": sum(
            float(row.get("input_tokens", 0))
            + float(row.get("output_tokens", 0))
            for row in tasks
        )
        / len(tasks),
        "mean_steps": sum(float(row.get("steps", 0)) for row in tasks)
        / len(tasks),
    }


def _candidate_pair(
    transfer_result: dict[str, Any],
) -> tuple[str, str, str, str, str] | None:
    deltas = transfer_result["deltas"]
    for specification in PAIR_PRIORITY:
        delta_name = specification[0]
        if bool(deltas[delta_name]["significant"]):
            return specification
    return None


def _checkpoint_info(output_dir: Path, name: str) -> dict[str, Any]:
    root = output_dir / "checkpoints" / name
    candidate = root / "candidate.py"
    state = root / "harness_store.json"
    return {
        "name": name,
        "candidate_path": str(candidate),
        "state_path": str(state),
        "candidate_sha256": _sha256(candidate),
        "state_sha256": _sha256(state),
    }


def prepare_hda_review(
    *, output_dir: Path, transfer_result: dict[str, Any]
) -> dict[str, Any]:
    selected = _candidate_pair(transfer_result)
    manifest: dict[str, Any] = {
        "method": "Harness Delta Attribution v2.0",
        "method_semantics": {
            "T": "test-time scaling",
            "O": "detectable overfitting (lower bound)",
            "G": "generalizable improvement (upper bound)",
        },
        "statistical_unit": (
            "paired hidden stream; tasks within a stream share evolving state"
        ),
        "approvals": {
            "compute_matching_plan": False,
            "neutralization_diff": False,
        },
        "all_deltas": transfer_result["deltas"],
    }
    diff_text = ""
    state_diff_text = ""
    if selected is None:
        manifest.update(
            {
                "status": "within_noise",
                "selected_pair": None,
                "decision": (
                    "No priority delta passed the 95% paired-bootstrap gate; "
                    "stop before B_cc/E_neutral and emit no attribution bar."
                ),
            }
        )
    else:
        delta_name, baseline, evolved, benchmark, label = selected
        baseline_info = _checkpoint_info(output_dir, baseline)
        evolved_info = _checkpoint_info(output_dir, evolved)
        baseline_source = Path(baseline_info["candidate_path"]).read_text(
            encoding="utf-8"
        )
        evolved_source = Path(evolved_info["candidate_path"]).read_text(
            encoding="utf-8"
        )
        diff_text = "".join(
            difflib.unified_diff(
                baseline_source.splitlines(keepends=True),
                evolved_source.splitlines(keepends=True),
                fromfile=f"{baseline}/candidate.py",
                tofile=f"{evolved}/candidate.py",
            )
        )
        baseline_state = json.dumps(
            json.loads(Path(baseline_info["state_path"]).read_text()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).splitlines(keepends=True)
        evolved_state = json.dumps(
            json.loads(Path(evolved_info["state_path"]).read_text()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).splitlines(keepends=True)
        state_diff_text = "".join(
            difflib.unified_diff(
                baseline_state,
                evolved_state,
                fromfile=f"{baseline}/harness_store.json",
                tofile=f"{evolved}/harness_store.json",
            )
        )
        baseline_compute = _compute_summary(
            transfer_result["cells"][f"{baseline}__{benchmark}"]
        )
        evolved_compute = _compute_summary(
            transfer_result["cells"][f"{evolved}__{benchmark}"]
        )
        no_extra_calls = (
            evolved_compute["mean_model_calls"]
            <= baseline_compute["mean_model_calls"]
        )
        manifest.update(
            {
                "status": "awaiting_compute_plan_approval",
                "selected_pair": {
                    "label": label,
                    "delta_name": delta_name,
                    "baseline": baseline_info,
                    "evolved": evolved_info,
                    "evaluation_set": f"{benchmark} hidden streams",
                    "gate": transfer_result["deltas"][delta_name],
                    "compute": {
                        "meter": "model_calls per task (tokens reported as sensitivity)",
                        "baseline": baseline_compute,
                        "evolved": evolved_compute,
                        "evolved_uses_no_more_model_calls": no_extra_calls,
                    },
                    "candidate_diff_path": str(
                        output_dir / "hda" / "candidate_pair.diff"
                    ),
                    "state_diff_path": str(
                        output_dir / "hda" / "candidate_pair_state.diff"
                    ),
                },
                "checkpoint_1": {
                    "required": True,
                    "approved": False,
                    "proposal": (
                        "B_cc = B and T = 0 without re-evaluation"
                        if no_extra_calls
                        else (
                            "Human must approve an item-level sample-and-aggregate "
                            "compute matcher before any B_cc evaluation"
                        )
                    ),
                },
                "checkpoint_2": {
                    "required": True,
                    "approved": False,
                    "proposal": (
                        "Inspect the exact B→E code/state delta, discover any "
                        "answer bypass or benchmark shortcut, then approve the "
                        "minimal E→E_neutral diff. No shortcut is declared here."
                    ),
                },
            }
        )

    hda_dir = output_dir / "hda"
    hda_dir.mkdir(parents=True, exist_ok=True)
    (hda_dir / "candidate_pair.diff").write_text(diff_text, encoding="utf-8")
    (hda_dir / "candidate_pair_state.diff").write_text(
        state_diff_text, encoding="utf-8"
    )
    _write_json(hda_dir / "gate_manifest.json", manifest)
    _write_html_report(output_dir, transfer_result, manifest, diff_text)
    return manifest


def _write_html_report(
    output_dir: Path,
    transfer_result: dict[str, Any],
    manifest: dict[str, Any],
    diff_text: str,
) -> None:
    cell_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.3f}</td><td>{}</td></tr>".format(
            html.escape(str(cell["checkpoint"])),
            html.escape(str(cell["benchmark"])),
            float(cell["mean_score"]),
            html.escape(json.dumps(cell["stream_scores"])),
        )
        for cell in transfer_result["cells"].values()
    )
    delta_rows = "".join(
        "<tr><td>{}</td><td>{:+.3f}</td><td>[{:+.3f}, {:+.3f}]</td>"
        "<td>{:.4f}</td><td>{}</td></tr>".format(
            html.escape(name),
            float(value["delta"]),
            float(value["ci_95"][0]),
            float(value["ci_95"][1]),
            float(value.get("p_value", 0.0)),
            "PASS" if value["significant"] else "STOP",
        )
        for name, value in transfer_result["deltas"].items()
    )
    selected = manifest.get("selected_pair")
    selected_html = (
        "<p>No priority pair passed the gate. HDA stops here; no T/O/G bar is valid.</p>"
        if selected is None
        else (
            f"<p><strong>Selected:</strong> {html.escape(selected['label'])}</p>"
            f"<p><strong>Status:</strong> {html.escape(manifest['status'])}</p>"
            f"<pre>{html.escape(diff_text or '(candidate.py unchanged; inspect state diff)')}</pre>"
        )
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sequential Harness Transfer + HDA</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 20px;color:#172033}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ccd3df;padding:8px;text-align:left}}
th{{background:#eef2f7}}pre{{background:#111827;color:#e5e7eb;padding:16px;overflow:auto;white-space:pre-wrap}}
.stop{{color:#9b1c1c}}.note{{background:#fff7d6;padding:12px;border-left:4px solid #d9a400}}
</style></head><body>
<h1>Sequential Harness Transfer + HDA</h1>
<p class="note">HDA semantics: T = test-time scaling; O = detectable overfitting (lower bound); G = generalizable improvement (upper bound). B_cc and E_neutral require separate human approvals.</p>
<p>Statistical unit: paired hidden stream. Tasks inside one stream are not treated as independent because harness state evolves across them.</p>
<h2>Six-cell hidden transfer matrix</h2>
<table><tr><th>Checkpoint</th><th>Benchmark</th><th>Mean</th><th>Stream scores</th></tr>{cell_rows}</table>
<h2>Paired-bootstrap gate</h2>
<table><tr><th>Delta</th><th>Estimate</th><th>95% CI</th><th>p</th><th>Gate</th></tr>{delta_rows}</table>
<h2>HDA review</h2>{selected_html}
<p>No attribution bar is emitted until both approvals and controlled variant scores exist.</p>
</body></html>"""
    report = output_dir / "report" / "agentstream_transfer_hda.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(document, encoding="utf-8")
