"""Report/provenance tests. Mock timings are fixtures, never benchmark evidence."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "benchmark_fuzzy_distance", Path(__file__).with_name("benchmark_fuzzy_distance.py")
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkFuzzyDistanceTests(unittest.TestCase):
    def test_report_retains_raw_samples_identity_and_correct_scope(self):
        commands = []

        def fake_run(command):
            commands.append(command)
            if command[1:3] == ["--version", "--verbose"]:
                return "rustc fixture compiler"
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return "fixture-head\n"
            if command[:3] == ["git", "status", "--porcelain"]:
                return " M fixture\n"
            if "distance-tests" in command[0]:
                return "test result: ok. 4 passed\n"
            if "distance-bench" in command[0]:
                return "\n".join(json.dumps({
                    "fixture": "one-edit", "chars": 64, "limit": 1,
                    "sample": index, "iterations": 5,
                    "before_ns_per_call": before, "after_ns_per_call": after,
                }) for index, (before, after) in enumerate([(4, 2), (6, 3)]))
            return ""

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(BENCH.shutil, "which", return_value="rustc"), patch.object(
                BENCH, "run", side_effect=fake_run
            ), patch("sys.argv", ["bench", "--output", str(output), "--iterations", "5"]):
                BENCH.main()
            report = json.loads(output.read_text())
        self.assertEqual(report["baseline_commit"], BENCH.BASE)
        self.assertEqual(report["head_commit"], "fixture-head")
        self.assertTrue(report["working_tree_status"])
        self.assertIn("not end-to-end", report["scope"])
        self.assertEqual(len(report["source_sha256"]), 3)
        self.assertEqual(len(report["samples"]), 2)
        self.assertEqual(report["summary"][0]["before_median_ns"], 5)
        self.assertEqual(report["summary"][0]["after_median_ns"], 2.5)
        self.assertEqual(report["summary"][0]["ratio_before_over_after"], 2)
        self.assertTrue(any("--test" in command and str(BENCH.SOURCES[2]) in command
                            for command in commands))
        self.assertTrue(any("-O" in command for command in commands))

    def test_missing_compiler_does_not_emit_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(BENCH.shutil, "which", return_value=None), patch(
                "sys.argv", ["bench", "--output", str(output)]
            ), self.assertRaises(SystemExit):
                BENCH.main()
            self.assertFalse(output.exists())

    def test_failed_native_tests_do_not_emit_evidence(self):
        def failed_run(command):
            if "distance-tests" in command[0]:
                raise subprocess.CalledProcessError(101, command, stderr="regression")
            return "fixture"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(BENCH.shutil, "which", return_value="rustc"), patch.object(
                BENCH, "run", side_effect=failed_run
            ), patch("sys.argv", ["bench", "--output", str(output)]), self.assertRaises(
                subprocess.CalledProcessError
            ):
                BENCH.main()
            self.assertFalse(output.exists())

    def test_nonpositive_iterations_are_rejected(self):
        with patch("sys.argv", ["bench", "--output", "unused.json", "--iterations", "0"]), \
                self.assertRaises(SystemExit):
            BENCH.main()


if __name__ == "__main__":
    unittest.main()
