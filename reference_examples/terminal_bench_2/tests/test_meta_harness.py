from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import meta_harness


ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL = ROOT / "scripts" / "run_eval.sh"


class AgentClassValidationTests(unittest.TestCase):
    def test_accepts_terminus2_subclass(self) -> None:
        agent_class = meta_harness.validate_agent_class(
            "agents.baseline_kira:AgentHarness"
        )

        self.assertEqual(agent_class.__name__, "AgentHarness")

    def test_rejects_missing_exact_class(self) -> None:
        with self.assertRaisesRegex(AttributeError, "AgentHarness"):
            meta_harness.validate_agent_class("anthropic_caching:AgentHarness")

    def test_rejects_non_class(self) -> None:
        with self.assertRaisesRegex(TypeError, "is not a class"):
            meta_harness.validate_agent_class("anthropic_caching:add_anthropic_caching")

    def test_rejects_class_that_is_not_terminus2(self) -> None:
        with self.assertRaisesRegex(TypeError, "must subclass Terminus2"):
            meta_harness.validate_agent_class("pathlib:Path")

    def test_run_eval_rejects_invalid_class_before_harbor(self) -> None:
        result = subprocess.run(
            [str(RUN_EVAL), "pathlib:Path", "full", "1", "1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must subclass Terminus2", result.stderr)
        self.assertNotIn("harbor run", result.stdout)


class TimeoutWiringTests(unittest.TestCase):
    def test_python_runner_uses_harbor_timeout(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(meta_harness, "HARBOR_TIMEOUT_SECONDS", 12345),
            mock.patch.object(
                meta_harness.subprocess, "run", return_value=completed
            ) as run,
        ):
            _, result = meta_harness.harbor_run(
                "agents.baseline_kira:AgentHarness", "test-job"
            )

        self.assertTrue(result)
        self.assertEqual(run.call_args.kwargs["timeout"], 12345)
        self.assertEqual(run.call_args.kwargs["env"]["HARBOR_TIMEOUT_SECONDS"], "12345")

    def test_python_runner_does_not_complete_on_timeout(self) -> None:
        timed_out = subprocess.CompletedProcess([], 124, "", "timed out")
        with mock.patch.object(meta_harness.subprocess, "run", return_value=timed_out):
            _, result = meta_harness.harbor_run(
                "agents.baseline_kira:AgentHarness", "test-job"
            )

        self.assertIsNone(result)

    def test_shell_runner_uses_harbor_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            capture = tmp_path / "timeout-args"
            fake_timeout = tmp_path / "timeout"
            fake_timeout.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
            )
            fake_timeout.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "CAPTURE": str(capture),
                    "HARBOR_TIMEOUT_SECONDS": "12345",
                    "PATH": f"{tmp_path}:{env['PATH']}",
                }
            )

            result = subprocess.run(
                [
                    str(RUN_EVAL),
                    "agents.baseline_kira:AgentHarness",
                    "full",
                    "1",
                    "1",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            timeout_args = capture.read_text().splitlines()
            self.assertEqual(
                timeout_args[:3], ["--signal=TERM", "--kill-after=60", "12345"]
            )
            self.assertEqual(timeout_args[3:6], ["uv", "run", "harbor"])


if __name__ == "__main__":
    unittest.main()
