from integrations.agentstream_online.proposer import build_proposer_prompt


def test_proposer_prompt_hides_scores_and_names_allowed_evidence() -> None:
    prompt = build_proposer_prompt(3, "bfcl", "task-7")

    assert "benchmark: bfcl" in prompt
    assert "task_id: task-7" in prompt
    assert "not given" in prompt
    assert "reward" in prompt
    assert "trajectory.jsonl" in prompt
    assert "grader output" in prompt
