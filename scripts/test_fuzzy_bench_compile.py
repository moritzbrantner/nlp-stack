"""Check the harness=false compilation mode that standalone --test misses.

CI provides the pinned rustc. Python-only environments skip this native check;
the native evidence job still separately executes the real distance tests.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUSTC = shutil.which("rustc")


@unittest.skipUnless(RUSTC, "native benchmark compile check requires rustc")
class FuzzyBenchCompileTests(unittest.TestCase):
    def test_cargo_harness_false_mode_has_no_warnings(self):
        source = ROOT / "crates/text/text-index/benches/fuzzy_distance.rs"
        with tempfile.TemporaryDirectory(prefix="nlp-fuzzy-compile-") as directory:
            result = subprocess.run(
                [RUSTC, "--edition=2021", "--cfg", "test", "--deny", "warnings",
                 "--emit=metadata", str(source), "-o", str(Path(directory) / "bench.rmeta")],
                cwd=ROOT, text=True, capture_output=True, check=False, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
