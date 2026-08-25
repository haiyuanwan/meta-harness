from integrations.agentstream_sequential_meta.proposer import build_proposer_prompt


def test_prompt_exposes_search_feedback_but_protects_test() -> None:
    prompt = build_proposer_prompt(
        benchmark='bfcl',
        iteration=2,
        candidate_number=1,
        base_model='anthropic/Claude-Opus-4.8-C',
    )

    assert 'benchmark: bfcl' in prompt
    assert 'scores' in prompt
    assert 'rollouts' in prompt
    assert 'Do not edit incoming_harness_store.json' in prompt
    assert 'private_test' in prompt
    assert 'verifier' in prompt
