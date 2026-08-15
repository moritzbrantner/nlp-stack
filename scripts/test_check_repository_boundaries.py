#!/usr/bin/env python3
"""Focused controlled-fault tests for NLP ownership and dependency boundaries."""

from __future__ import annotations

import copy
import unittest

from check_repository_boundaries import bun_manifests, validate
from repository_split import OWNERSHIP_PATH, cargo_metadata, load_json


class RepositoryBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = cargo_metadata()
        self.ownership = load_json(OWNERSHIP_PATH)
        self.bun_packages = bun_manifests()

    def errors(
        self,
        metadata: dict | None = None,
        ownership: dict | None = None,
        bun_packages: dict | None = None,
    ) -> list[str]:
        return validate(
            metadata or self.metadata,
            ownership or self.ownership,
            bun_packages=bun_packages or self.bun_packages,
        )

    def test_live_workspace_has_exact_approved_ownership(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_missing_and_duplicate_records_fail(self) -> None:
        missing = copy.deepcopy(self.ownership)
        missing["packages"].pop()
        self.assertTrue(any("unclassified" in error or "exactly" in error for error in self.errors(ownership=missing)))
        duplicate = copy.deepcopy(self.ownership)
        duplicate["packages"].append(copy.deepcopy(duplicate["packages"][0]))
        self.assertTrue(any("classified more than once" in error for error in self.errors(ownership=duplicate)))

    def test_wrong_owner_fails(self) -> None:
        ownership = copy.deepcopy(self.ownership)
        ownership["packages"][0]["target_repository"] = "rust-packages"
        self.assertTrue(any("wrong target repository" in error for error in self.errors(ownership=ownership)))

    def test_source_inventory_drift_fails(self) -> None:
        ownership = copy.deepcopy(self.ownership)
        source_record = next(
            record
            for record in ownership["packages"]
            if record.get("provenance", {}).get("kind") != "destination-authored"
        )
        source_record["current_domain"] = "unreviewed-domain"
        self.assertTrue(any("source ownership records" in error for error in self.errors(ownership=ownership)))

    def test_adapter_must_wrap_a_workspace_library(self) -> None:
        ownership = copy.deepcopy(self.ownership)
        adapter = next(record for record in ownership["packages"] if record["package_kind"] == "CLI")
        adapter["wrapped_library"] = "missing-library"
        self.assertTrue(any("invalid wrapped_library" in error for error in self.errors(ownership=ownership)))

    def test_synthetic_path_escape_fails(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["packages"][0]["dependencies"].append({"name": "outside", "path": "/tmp/outside"})
        self.assertTrue(any("escapes repository" in error for error in self.errors(metadata=metadata)))

    def test_synthetic_cross_capability_edge_fails(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["packages"][0]["dependencies"].append(
            {
                "name": "moenarch-audio-analysis-core",
                "kind": None,
                "path": None,
                "req": "^0.1.0",
                "source": "registry+https://github.com/rust-lang/crates.io-index",
            }
        )
        self.assertTrue(any("forbidden NLP dependency" in error for error in self.errors(metadata=metadata)))

    def test_foundation_dependency_must_be_exact_registry_version(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        dependency = next(
            dependency
            for package in metadata["packages"]
            for dependency in package["dependencies"]
            if dependency["name"] == "moenarch-runtime-core"
        )
        dependency["req"] = "^0.2.1"
        dependency["path"] = "/tmp/foundation"
        errors = self.errors(metadata=metadata)
        self.assertTrue(any("must use =0.2.1" in error for error in errors), errors)
        self.assertTrue(any("must resolve from the registry" in error for error in errors), errors)

    def test_moving_git_branch_with_resolved_hash_fails(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["packages"][0]["dependencies"].append(
            {
                "name": "remote",
                "source": "git+https://example.invalid/repository?branch=main#" + "a" * 40,
            }
        )
        self.assertTrue(any("non-immutable Git" in error for error in self.errors(metadata=metadata)))

    def test_exact_git_revision_and_resolved_hash_is_allowed(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["packages"][0]["dependencies"].append(
            {
                "name": "remote",
                "source": "git+https://example.invalid/repository?rev=" + "b" * 40 + "#" + "a" * 40,
            }
        )
        self.assertFalse(any("Git dependency" in error for error in self.errors(metadata=metadata)))

    def test_text_apps_cannot_restore_compatibility_ui_dependency(self) -> None:
        bun_packages = copy.deepcopy(self.bun_packages)
        _, app = bun_packages["@moritzbrantner/text-core-app"]
        app["dependencies"]["@moritzbrantner/video-analysis-ui"] = "workspace:*"
        errors = self.errors(bun_packages=bun_packages)
        self.assertTrue(any("compatibility UI facade" in error for error in errors), errors)

    def test_npm_publication_requires_a_separate_plan(self) -> None:
        ownership = copy.deepcopy(self.ownership)
        wrapper = next(record for record in ownership["packages"] if record["package_kind"] == "npm wrapper")
        wrapper["automatic_publish_eligible"] = True
        errors = self.errors(ownership=ownership)
        self.assertTrue(any("publication must remain separately authorized" in error for error in errors), errors)

    def test_platform_packages_ownership_check_cannot_be_removed(self) -> None:
        ownership = copy.deepcopy(self.ownership)
        ownership["platform_packages_ownership_checked"] = False
        self.assertTrue(any("platform_packages_ownership_checked" in error for error in self.errors(ownership=ownership)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
