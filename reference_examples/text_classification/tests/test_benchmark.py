from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from text_classification import benchmark


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_result(
        base: Path, seed: int, data: dict, memory: str = "unit-memory"
    ) -> None:
        path = (
            benchmark.run_dir(base, "unit-dataset", memory, "unit-model", seed)
            / "val.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_load_results_aggregates_configured_seeds(self) -> None:
        first = {
            "accuracy": 0.5,
            "correct": 5,
            "total": 10,
            "memory_context_chars": 100,
            "runtime_seconds": 1.25,
            "llm_calls": 2,
            "llm_input_tokens": 10,
            "llm_output_tokens": 5,
            "llm_total_tokens": 15,
            "seed": 42,
        }
        second = {
            "accuracy": 1.0,
            "correct": 10,
            "total": 10,
            "memory_context_chars": 200,
            "runtime_seconds": 2.75,
            "llm_calls": 3,
            "llm_input_tokens": 20,
            "llm_output_tokens": 10,
            "llm_total_tokens": 30,
            "seed": 123,
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark, "SEEDS", [123, 42]),
        ):
            base = Path(tmp)
            self._write_result(base, 42, first)
            self._write_result(base, 123, second)

            result = benchmark.load_results(base)[
                ("unit-model", "unit-dataset", "unit-memory")
            ]

        self.assertEqual(result["seeds"], [42, 123])
        self.assertNotIn("seed", result)
        self.assertEqual((result["correct"], result["total"]), (15, 20))
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["memory_context_chars"], 150)
        self.assertEqual(result["runtime_seconds"], 4.0)
        self.assertEqual(result["llm_calls"], 5)
        self.assertEqual(result["llm_input_tokens"], 30)
        self.assertEqual(result["llm_output_tokens"], 15)
        self.assertEqual(result["llm_total_tokens"], 45)

    def test_load_results_omits_incomplete_seed_group(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark, "SEEDS", [42, 123]),
        ):
            base = Path(tmp)
            self._write_result(base, 42, {"accuracy": 0.5})

            self.assertEqual(benchmark.load_results(base), {})

    def test_pareto_dominates_equal_cost_lower_accuracy(self) -> None:
        points = [("better", 90.0, 100), ("worse", 80.0, 100)]

        self.assertEqual(
            benchmark.compute_pareto_frontier(points),
            [("better", 90.0, 100)],
        )

    def test_worker_commands_import_package_from_unrelated_cwd(self) -> None:
        model = {"model": "provider/unit-model"}
        memory_systems = [("no_memory", "agents/no_memory.py")]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark, "SEEDS", [42]),
            patch.object(benchmark, "get_dataset_sizes", return_value=(1, 1, 1)),
        ):
            root = Path(tmp)
            logs_dir = root / "logs"
            results_dir = root / "results"
            val_runs, _, _ = benchmark.build_val_runs(
                logs_dir, memory_systems, ["unit-dataset"], [model]
            )

            memory_file = (
                benchmark.run_dir(logs_dir, "unit-dataset", "no_memory", "unit-model")
                / "memory.json"
            )
            memory_file.parent.mkdir(parents=True, exist_ok=True)
            memory_file.write_text("")
            test_runs, _, _ = benchmark.build_test_runs(
                logs_dir,
                results_dir,
                memory_systems,
                ["unit-dataset"],
                [model],
            )

            for _, command in [val_runs[0], test_runs[0]]:
                self.assertEqual(command[:9], benchmark._inner_loop_command())

            env = os.environ.copy()
            env["PYTHONPATH"] = val_runs[0][1][1].removeprefix("PYTHONPATH=")
            probe = [
                sys.executable,
                "-c",
                "import importlib.util; assert "
                "importlib.util.find_spec('text_classification.inner_loop')",
            ]
            completed = subprocess.run(
                probe,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_build_test_runs_retries_corrupt_result(self) -> None:
        model = {"model": "provider/unit-model"}
        memory_systems = [("no_memory", "agents/no_memory.py")]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark, "SEEDS", [42]),
            patch.object(benchmark, "get_dataset_sizes", return_value=(1, 1, 1)),
        ):
            root = Path(tmp)
            memory_file = (
                benchmark.run_dir(
                    root / "logs", "unit-dataset", "no_memory", "unit-model"
                )
                / "memory.json"
            )
            memory_file.parent.mkdir(parents=True)
            memory_file.write_text("")
            test_file = (
                benchmark.run_dir(
                    root / "results", "unit-dataset", "no_memory", "unit-model"
                )
                / "test.json"
            )
            test_file.parent.mkdir(parents=True)
            test_file.write_text("null")

            self.assertEqual(benchmark.load_results(root / "results", "test.json"), {})

            runs, pending, done = benchmark.build_test_runs(
                root / "logs",
                root / "results",
                memory_systems,
                ["unit-dataset"],
                [model],
            )

        self.assertEqual((pending, done), (1, 0))
        self.assertEqual(len(runs), 1)
        self.assertIn("--force", runs[0][1])

    async def test_main_propagates_worker_failure_and_uses_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "run" / "logs"
            results_dir = root / "run" / "heldout"
            argv = [
                "benchmark.py",
                "--test",
                "--logs-dir",
                str(logs_dir),
                "--results-dir",
                str(results_dir),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark.os, "chdir"),
                patch.object(
                    benchmark,
                    "discover_all_memory_systems",
                    return_value=[("no_memory", "agents/no_memory.py")],
                ),
                patch.object(
                    benchmark,
                    "build_test_runs",
                    return_value=([("failed", ["false"])], 1, 0),
                ) as build_runs,
                patch.object(
                    benchmark,
                    "run_all_jobs",
                    AsyncMock(return_value=[("failed", False)]),
                ),
                patch.object(benchmark, "print_results"),
                patch.object(benchmark, "print_summary"),
                patch.object(benchmark, "print_missing"),
            ):
                exit_code = await benchmark.main()

            self.assertEqual(exit_code, 1)
            self.assertEqual(build_runs.call_args.args[1], results_dir.resolve())


if __name__ == "__main__":
    unittest.main()
