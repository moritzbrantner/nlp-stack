"""Verification must reject an unprepared dependency state without repairing it."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LockedVerificationTests(unittest.TestCase):
    def test_gates_reject_stale_lockfile_without_modifying_it(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for script in ("check-preflight.sh", "check-pages.sh"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "scripts").mkdir()
                (root / "src").mkdir()
                (root / "src/lib.rs").write_text("", encoding="utf-8")
                manifest = root / "Cargo.toml"
                manifest.write_text('[package]\nname = "locked-gate-fixture"\nversion = "0.1.0"\n', encoding="utf-8")
                subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=root, check=True, capture_output=True)
                original = (root / "Cargo.lock").read_bytes()
                manifest.write_text(manifest.read_text(encoding="utf-8").replace("0.1.0", "0.2.0"), encoding="utf-8")
                shutil.copyfile(repository / "scripts" / script, root / "scripts" / script)
                result = subprocess.run(["bash", str(root / "scripts" / script)], cwd=repository, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--locked", result.stderr)
                self.assertEqual((root / "Cargo.lock").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
