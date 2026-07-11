# Text Classification

Minimal text classification reference experiment for Meta-Harness. The outer loop writes candidate memory systems to `agents/`; the inner loop evaluates them on the datasets in `config.yaml`.

## Quick Start

Install:

```bash
cd reference_examples/text_classification
uv sync
```

Run one evolve iteration:

```bash
uv run python meta_harness.py --iterations 1
```

Run one memory system on one dataset:

```bash
RUN=logs/manual
OUT="$RUN/Symptom2Disease/fewshot_all/gpt-oss-120b"
mkdir -p "$OUT"
PYTHONPATH=.. uv run python -m text_classification.inner_loop \
  --memory fewshot_all \
  --dataset Symptom2Disease \
  --model openrouter/openai/gpt-oss-120b \
  --val-output "$OUT/val.json" \
  --save-memory "$OUT/memory.json"
uv run python benchmark.py --logs-dir "$RUN" --results
```

By default this uses the model in `config.yaml` (`openrouter/openai/gpt-oss-120b`). To target another provider or any OpenAI-compatible endpoint, override `--model` and optionally `--api-base`.

Print the benchmark summary:

```bash
uv run python benchmark.py --results
```

Evolution uses validation results only. Finalize a named run explicitly after
selection is complete:

```bash
uv run python meta_harness.py --run-name my-run --test
```

Test results are written under `logs/my-run/results/` and printed at the end.
Finalization permanently blocks more evolution under that run name. The test
data is included in this public repository, so this is operational isolation,
not access control.

Run provider-free tests:

```bash
PYTHONPATH=.. uv run python -m unittest discover -s tests -v
```

## Layout Notes

- `agents/`: the kept baselines plus the write target for generated candidates.
- `.claude/skills/meta-harness/SKILL.md`: main proposer prior used by `meta_harness.py`.

## Runtime And Cost

The release default uses OpenRouter (`openrouter/openai/gpt-oss-120b`). If you want a different provider or your own OpenAI-compatible endpoint, pass `--model` and optionally `--api-base`, or change `config.yaml`. The paper experiments used a local `vllm` deployment of `gpt-oss-120b`, MXFP4 quantized, with `max-model-len=32768`. API-backed runs may differ in quality from that setup and may be better.

## Release Notes

- `config.yaml` is the source of truth for datasets, models, and active memory systems.
- The public release includes the MCE paper datasets for this experiment under `data/`, so there is no runtime clone step.
- `inner_loop.py` still uses package-mode imports, so the single-candidate command above keeps `PYTHONPATH=..` when run from this directory.
- `benchmark.py` is the sweep/orchestration layer used by `meta_harness.py`; `inner_loop.py` is the single memory-system evaluator that `benchmark.py` dispatches.
