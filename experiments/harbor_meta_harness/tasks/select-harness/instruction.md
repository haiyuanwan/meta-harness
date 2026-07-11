# Improve a Harbor harness

`/app/harness.py` is an editable Python Harbor agent.
Improve it into a broadly useful terminal harness.
You may call `evaluate_harness` at most four times.
Each call validates the source, then scores it on a hidden task suite.
Do not encode benchmark-specific data, validator behavior, test paths, or task-specific output paths in the harness.
Leave the best general harness in `/app/harness.py`.
