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
        legacy_span_constructors: list[str] | None = None,
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
                    "legacyTextSpanConstructors": legacy_span_constructors or [],
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

    SEMANTIC_PROVENANCE_ENUM = (
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n"
        "/// Variants describing annotation provenance.\n"
        "pub enum AnnotationProvenance {\n"
        "    /// Directly observed.\n"
        "    Observed,\n"
        "    Heuristic,\n"
        "    Model,\n"
        "    Derived,\n"
        "    Imported,\n"
        "}\n"
    )

    def _write_provenance(self, root: Path, source: str) -> None:
        provenance = root / "crates" / "text" / "text-core" / "src" / "provenance.rs"
        provenance.write_text(source, encoding="utf-8")

    def test_current_repository_provenance_is_semantic(self) -> None:
        errors = check_contract(REPOSITORY_ROOT)
        self.assertFalse(
            [error for error in errors if "AnnotationProvenance" in error],
            errors,
        )

    def test_semantic_provenance_set_is_accepted(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        self._write_provenance(root, self.SEMANTIC_PROVENANCE_ENUM)

        self.assertEqual(check_contract(root), [])

    def test_runtime_provenance_variants_are_rejected(self) -> None:
        for variant in (
            "Onnx",
            "Candle",
            "CudaOxide",
            "Cuda",
            "Tokenizer",
            "Wgpu",
            "Metal",
            "External",
            "TensorRt",
        ):
            with self.subTest(variant):
                temporary, root = self._fixture()
                self.addCleanup(temporary.cleanup)
                self._write_provenance(
                    root,
                    self.SEMANTIC_PROVENANCE_ENUM.replace(
                        "    Imported,\n",
                        f"    Imported,\n    /// Runtime-specific.\n    {variant},\n",
                    ),
                )

                errors = check_contract(root)
                self.assertTrue(
                    any(
                        error.startswith(
                            "text-core AnnotationProvenance contains concrete "
                            "runtime/backend variants"
                        )
                        and variant in error
                        for error in errors
                    ),
                    errors,
                )

    def test_unapproved_or_missing_semantic_variants_are_rejected(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        self._write_provenance(
            root,
            self.SEMANTIC_PROVENANCE_ENUM.replace("    Imported,\n", "    Guessed,\n"),
        )

        self.assertIn(
            "text-core AnnotationProvenance must have exactly the semantic variants "
            "Derived, Heuristic, Imported, Model, Observed; unexpected Guessed; "
            "missing Imported",
            check_contract(root),
        )

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

    def test_analyzer_pipeline_framework_cannot_return(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        lib = root / "crates" / "text" / "text-core" / "src" / "lib.rs"
        lib.write_text(
            lib.read_text(encoding="utf-8") + "pub trait TextAnalyzer {}\n",
            encoding="utf-8",
        )

        self.assertTrue(
            any(
                error.startswith(
                    "text-core regained forbidden analyzer/pipeline framework types: TextAnalyzer"
                )
                for error in check_contract(root)
            )
        )

    def test_analyzer_pipeline_aliases_and_reexports_cannot_return(self) -> None:
        exposures = {
            "type alias": "pub type TextPipeline = Vec<u8>;\n",
            "enum": "pub enum TextPipeline { Empty }\n",
            "restricted struct": "pub(crate) struct TextPipeline;\n",
            "raw identifier": "pub struct r#TextPipeline;\n",
            "raw identifier re-export": "pub use crate::inner::r#TextPipeline;\n",
            "re-export": "pub use crate::inner::TextPipeline;\n",
            "aliased re-export": "pub use crate::inner::Runner as TextPipeline;\n",
            "grouped re-export": "pub use crate::inner::{\n    Other,\n    TextPipeline,\n};\n",
        }
        for label, source in exposures.items():
            with self.subTest(label):
                temporary, root = self._fixture()
                self.addCleanup(temporary.cleanup)
                lib = root / "crates" / "text" / "text-core" / "src" / "lib.rs"
                lib.write_text(lib.read_text(encoding="utf-8") + source, encoding="utf-8")

                self.assertTrue(
                    any(
                        error.startswith(
                            "text-core regained forbidden analyzer/pipeline framework types: "
                            "TextPipeline"
                        )
                        for error in check_contract(root)
                    )
                )

    def test_public_glob_reexport_is_rejected(self) -> None:
        for source in ("pub use media_core::*;\n", "pub(crate) use media_core::{Timestamp, inner::*};\n"):
            with self.subTest(source):
                temporary, root = self._fixture()
                self.addCleanup(temporary.cleanup)
                lib = root / "crates" / "text" / "text-core" / "src" / "lib.rs"
                lib.write_text(lib.read_text(encoding="utf-8") + source, encoding="utf-8")

                self.assertIn(
                    "text-core must not use public glob re-exports (their exported names "
                    "cannot be checked against the A2 boundary): src/lib.rs",
                    check_contract(root),
                )

    def test_reexport_of_unrelated_names_is_not_flagged(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        lib = root / "crates" / "text" / "text-core" / "src" / "lib.rs"
        lib.write_text(
            lib.read_text(encoding="utf-8") + "pub use crate::inner::TextSpan;\n",
            encoding="utf-8",
        )

        self.assertEqual(check_contract(root), [])

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

    def test_text_span_return_type_is_not_counted_as_direct_construction(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        consumer = root / "crates" / "text" / "text-analysis" / "src"
        consumer.mkdir(parents=True)
        (consumer / "lib.rs").write_text(
            "fn span_for_text(text: &str) -> TextSpan {\n"
            "    TextSpan::from_byte_range(text, 0, text.len()).unwrap()\n"
            "}\n",
            encoding="utf-8",
        )

        self.assertEqual(check_contract(root), [])

    def test_direct_text_span_literal_is_guarded_regardless_of_field_order(self) -> None:
        temporary, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        consumer = root / "crates" / "text" / "text-analysis" / "src"
        consumer.mkdir(parents=True)
        (consumer / "lib.rs").write_text(
            "let span = TextSpan {\n"
            "    char_start: 0,\n"
            "    byte_end: 1,\n"
            "    byte_start: 0,\n"
            "    char_end: 1,\n"
            "};\n",
            encoding="utf-8",
        )

        self.assertIn(
            "legacy direct TextSpan construction debt does not match ledger: "
            "declared <none>; actual crates/text/text-analysis/src/lib.rs",
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
