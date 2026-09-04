from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from check_text_core_a2_boundary import check_contract


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TextCoreA2BoundaryTests(unittest.TestCase):
    def _write_debt(
        self,
        root: Path,
        *,
        dependencies: list[str],
        source_files: dict[str, list[str]],
        mirror_contracts: dict[str, str],
    ) -> None:
        scripts = root / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "text_core_a2_debt.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "crossDomainDependencies": dependencies,
                    "crossDomainSourceFiles": source_files,
                    "mirrorContracts": mirror_contracts,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

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
        self._write_debt(
            root,
            dependencies=["media-core"],
            source_files={"media_core": ["src/lib.rs"]},
            mirror_contracts={
                "TextDocumentContract": "src/contracts.rs",
                "TextSegmentContract": "src/contracts.rs",
            },
        )
        return temporary, root

    def test_current_repository_matches_exact_a2_debt_ledger(self) -> None:
        self.assertEqual(check_contract(REPOSITORY_ROOT), [])

    def test_new_dependency_is_rejected(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        cargo = root / "crates" / "text" / "text-core" / "Cargo.toml"
        cargo.write_text(cargo.read_text(encoding="utf-8") + "tokio = \"1\"\n", encoding="utf-8")

        self.assertIn(
            "text-core dependency surface grew beyond the A2 boundary: tokio",
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
            "media_core source debt does not match ledger: "
            "declared src/lib.rs; actual src/lib.rs, src/new_kernel_module.rs",
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
            "text-core gained unapproved mirror *Contract types during A2: ExtraContract",
            check_contract(root),
        )

    def test_existing_mirror_contract_cannot_spread_to_new_source_file(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        src = root / "crates" / "text" / "text-core" / "src"
        (src / "new_kernel_module.rs").write_text(
            "pub struct TextDocumentContract {}\n",
            encoding="utf-8",
        )

        self.assertIn(
            "TextDocumentContract debt does not match ledger: "
            "declared src/contracts.rs; actual src/contracts.rs, src/new_kernel_module.rs",
            check_contract(root),
        )

    def test_removing_debt_requires_shrinking_the_ledger(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        core = root / "crates" / "text" / "text-core"
        (core / "Cargo.toml").write_text("[dependencies]\nserde.workspace = true\n", encoding="utf-8")
        (core / "src" / "lib.rs").write_text("", encoding="utf-8")
        (core / "src" / "contracts.rs").unlink()

        errors = check_contract(root)
        self.assertTrue(
            any("cross-domain dependency debt does not match ledger" in error for error in errors)
        )
        self.assertTrue(any("TextDocumentContract debt does not match ledger" in error for error in errors))

    def test_removing_legacy_debt_passes_after_ledger_shrinks(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        core = root / "crates" / "text" / "text-core"
        (core / "Cargo.toml").write_text("[dependencies]\nserde.workspace = true\n", encoding="utf-8")
        (core / "src" / "lib.rs").write_text("", encoding="utf-8")
        (core / "src" / "contracts.rs").unlink()
        self._write_debt(
            root,
            dependencies=[],
            source_files={},
            mirror_contracts={},
        )

        self.assertEqual(check_contract(root), [])


if __name__ == "__main__":
    unittest.main()
