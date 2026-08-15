#!/usr/bin/env python3
"""Focused controlled-fault tests for the non-publishing bootstrap plan."""

from __future__ import annotations

import copy
import unittest

from check_release_plan import validate
from repository_split import OWNERSHIP_PATH, RELEASE_PLAN_PATH, cargo_metadata, load_json


class ReleasePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_json(RELEASE_PLAN_PATH)
        self.ownership = load_json(OWNERSHIP_PATH)
        self.metadata = cargo_metadata()

    def errors(self, plan: dict, ownership: dict | None = None, metadata: dict | None = None) -> list[str]:
        return validate(plan, ownership or self.ownership, metadata or self.metadata)

    def test_live_nonpublishing_plan_is_valid(self) -> None:
        self.assertEqual(self.errors(self.plan), [])

    def test_nonpublishing_plan_names_current_and_next_release_owners(self) -> None:
        self.assertEqual(self.plan["active_release_owner"], "moritzbrantner/rust-packages")
        self.assertTrue(
            all(
                package["intended_next_release_owner"] == "moritzbrantner/nlp-stack"
                for package in self.plan["packages"] + self.plan["npm_packages"]
            )
        )

    def test_active_release_owner_cannot_move_during_bootstrap(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["active_release_owner"] = "moritzbrantner/nlp-stack"
        self.assertTrue(any("wrong active release owner" in error for error in self.errors(plan)))

    def test_cargo_publication_cannot_be_smuggled_into_bootstrap(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["packages"][0]["publish"] = True
        self.assertTrue(any("publication is not authorized" in error for error in self.errors(plan)))

    def test_npm_publication_cannot_be_smuggled_into_bootstrap(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["npm_packages"][0]["publish"] = True
        self.assertTrue(any("npm publication is not authorized" in error for error in self.errors(plan)))

    def test_version_change_is_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["packages"][0]["new_version"] = "9.9.9"
        self.assertTrue(any("retain version" in error for error in self.errors(plan)))

    def test_forged_equal_versions_are_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["packages"][0]["old_version"] = "9.9.9"
        plan["packages"][0]["new_version"] = "9.9.9"
        self.assertTrue(any("workspace version" in error for error in self.errors(plan)))

    def test_required_checks_cannot_be_deleted(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["required_checks"] = []
        self.assertTrue(any("complete bootstrap gate set" in error for error in self.errors(plan)))

    def test_required_checks_bind_clean_bun_and_wasm_smokes(self) -> None:
        self.assertIn("bun install --frozen-lockfile", self.plan["required_checks"])
        self.assertIn("bun run text-wasm:test:all", self.plan["required_checks"])
        self.assertIn("bun run text-app:build", self.plan["required_checks"])

    def test_real_internal_dependency_cannot_be_deleted(self) -> None:
        plan = copy.deepcopy(self.plan)
        package = next(item for item in plan["packages"] if item["release_dependencies"])
        package["release_dependencies"] = []
        self.assertTrue(any("do not match workspace metadata" in error for error in self.errors(plan)))

    def test_ownership_source_version_is_bound(self) -> None:
        ownership = copy.deepcopy(self.ownership)
        cargo_record = next(record for record in ownership["packages"] if record["ecosystem"] == "cargo")
        cargo_record["source_version"] = "9.9.9"
        errors = self.errors(self.plan, ownership=ownership)
        self.assertTrue(any("ownership source_version" in error for error in errors), errors)

    def test_metadata_and_plan_cannot_jointly_forge_source_version(self) -> None:
        plan = copy.deepcopy(self.plan)
        metadata = copy.deepcopy(self.metadata)
        name = plan["packages"][0]["name"]
        plan["packages"][0]["old_version"] = "9.9.9"
        plan["packages"][0]["new_version"] = "9.9.9"
        next(package for package in metadata["packages"] if package["name"] == name)["version"] = "9.9.9"
        errors = self.errors(plan, metadata=metadata)
        self.assertTrue(any("ownership source_version" in error for error in errors), errors)

    def test_wrong_owner_and_missing_package_are_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["packages"][0]["intended_next_release_owner"] = "moritzbrantner/rust-packages"
        plan["packages"].pop()
        errors = self.errors(plan)
        self.assertTrue(any("52 Cargo packages" in error for error in errors), errors)
        self.assertTrue(any("wrong intended next release owner" in error for error in errors), errors)

    def test_wrong_dependency_order_is_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        dependent = next(package for package in plan["packages"] if package["release_dependencies"])
        dependency = dependent["release_dependencies"][0]
        plan["dependency_order"].remove(dependency)
        plan["dependency_order"].append(dependency)
        self.assertTrue(any("wrong dependency order" in error for error in self.errors(plan)))

    def test_npm_workspace_dependency_is_bound(self) -> None:
        plan = copy.deepcopy(self.plan)
        app = next(package for package in plan["npm_packages"] if package["name"] == "@moritzbrantner/text-core-app")
        app["workspace_dependencies"] = []
        self.assertTrue(any("workspace_dependencies" in error for error in self.errors(plan)))

    def test_platform_packages_review_cannot_be_removed(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["platform_packages_ownership_checked"] = False
        self.assertTrue(any("platform-packages ownership" in error for error in self.errors(plan)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
