from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_text_core_a2_boundary import check_contract


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TextCoreA2BoundaryTests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        core = root / "crates" / "text" / "text-core"
        src = core / "src"
        src.mkdir(parents=True)
        (core / "Cargo.toml").write_text(
            "[dependencies]\nserde.workspace = true\nmedia-core.workspace = true\n",
            encoding="utf-8",
        )
        (src / "lib.rs").write_text(
            "pub use media_core::{AnalysisEvent, DetectError, Result, Timebase, Timestamp};\n",
            encoding="utf-8",
        )
        (src / "contracts.rs").write_text(
            "pub struct TextDocumentContract {}\npub struct TextSegmentContract {}\n",
            encoding="utf-8",
        )
        return temporary, root

    def test_current_repository_respects_a2_ceiling(self) -> None:
        self.assertEqual(check_contract(REPOSITORY_ROOT), [])

    def test_new_dependency_is_rejected(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        cargo = root / "crates" / "text" / "text-core" / "Cargo.toml"
        cargo.write_text(cargo.read_text(encoding="utf-8") + "tokio = \"1\"\n", encoding="utf-8")

        self.assertIn(
            "text-core dependency surface grew beyond the A2 baseline: tokio",
            check_contract(root),
        )

    def test_cross_domain_import_cannot_spread_to_new_source_file(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        src = root / "crates" / "text" / "text-core" / "src"
        (src / "new_kernel_module.rs").write_text(
            "use media_core::Timestamp;\n",
            encoding="utf-8",
        )

        self.assertIn(
            "media-core usage spread outside grandfathered A2 debt: src/new_kernel_module.rs",
            check_contract(root),
        )

    def test_new_parallel_contract_type_is_rejected(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        contracts = root / "crates" / "text" / "text-core" / "src" / "contracts.rs"
        contracts.write_text(
            contracts.read_text(encoding="utf-8") + "pub struct ExtraContract {}\n",
            encoding="utf-8",
        )

        self.assertIn(
            "text-core gained new parallel *Contract types during A2: ExtraContract",
            check_contract(root),
        )

    def test_removing_legacy_debt_remains_allowed(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        core = root / "crates" / "text" / "text-core"
        (core / "Cargo.toml").write_text("[dependencies]\nserde.workspace = true\n", encoding="utf-8")
        (core / "src" / "lib.rs").write_text("", encoding="utf-8")
        (core / "src" / "contracts.rs").unlink()

        self.assertEqual(check_contract(root), [])


if __name__ == "__main__":
    unittest.main()
