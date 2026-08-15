#!/usr/bin/env python3
"""Focused controlled-fault tests for the non-publishing bootstrap plan."""

from __future__ import annotations

import copy
import re
import tomllib
import unittest
from pathlib import Path

from check_release_plan import (
    NLP_WAVE_1,
    NLP_WAVE_1_CONSUMER_CHECKS,
    NLP_WAVE_1_VERSIONS,
    validate,
    validate_control_binding,
    validate_release_manifest,
)
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
        self.assertTrue(
            any("ownership source_version" in error for error in self.errors(plan))
        )

    def test_required_checks_cannot_be_deleted(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["required_checks"] = []
        self.assertTrue(any("complete bootstrap gate set" in error for error in self.errors(plan)))

    def test_required_checks_bind_clean_bun_and_wasm_smokes(self) -> None:
        self.assertIn("bun install --frozen-lockfile", self.plan["required_checks"])
        self.assertIn("bun run text-wasm:test:all", self.plan["required_checks"])
        self.assertIn("bun run text-app:build", self.plan["required_checks"])
        agent_checks = tomllib.loads(
            (OWNERSHIP_PATH.parents[2] / ".agent-loop.toml").read_text(
                encoding="utf-8"
            )
        )["verification"]["commands"]
        self.assertIn(
            "python3 scripts/check_release_plan.py --check releases/nlp-wave-1.toml",
            agent_checks,
        )

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


class CheckedReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_json(RELEASE_PLAN_PATH)
        self.ownership = load_json(OWNERSHIP_PATH)
        self.metadata = cargo_metadata()
        ownership = {
            package["current_package_name"]: package
            for package in self.ownership["packages"]
        }
        metadata = {package["name"]: package for package in self.metadata["packages"]}
        packages = []
        for name, version in NLP_WAVE_1:
            dependencies = sorted(
                dependency["name"]
                for dependency in metadata[name]["dependencies"]
                if dependency["kind"] != "dev" and dependency["name"] in metadata
            )
            packages.append(
                {
                    "name": name,
                    "version": version,
                    "owner": "moritzbrantner/nlp-stack",
                    "manifest_path": ownership[name]["manifest_path"],
                    "dependencies": dependencies,
                    "tag": f"{name}-v{version}",
                }
            )
        checks = tomllib.loads(
            (OWNERSHIP_PATH.parents[2] / ".agent-loop.toml").read_text(
                encoding="utf-8"
            )
        )["verification"]["commands"]
        self.manifest = {
            "schema_version": 1,
            "repository": "moritzbrantner/nlp-stack",
            "issue": 2,
            "source_sha": "a" * 40,
            "registry": "crates.io",
            "dependency_order": [name for name, _version in NLP_WAVE_1],
            "expected_tags": [package["tag"] for package in packages],
            "required_checks": checks,
            "required_consumer_checks": list(NLP_WAVE_1_CONSUMER_CHECKS),
            "packages": packages,
            "github_releases": [],
        }

    def errors(self, manifest: dict) -> list[str]:
        return validate_release_manifest(
            manifest,
            self.ownership,
            self.metadata,
            Path("releases/nlp-wave-1.toml"),
        )

    def test_exact_nlp_wave_is_valid(self) -> None:
        self.assertEqual(self.errors(self.manifest), [])

    def test_wrong_version_is_rejected(self) -> None:
        self.manifest["packages"][0]["version"] = "0.9.9"
        errors = self.errors(self.manifest)
        self.assertTrue(any("versions" in error for error in errors), errors)

    def test_wrong_destination_issue_is_rejected(self) -> None:
        self.manifest["issue"] = 3
        errors = self.errors(self.manifest)
        self.assertTrue(any("destination issue 2" in error for error in errors), errors)

    def test_wrong_order_is_rejected(self) -> None:
        self.manifest["packages"][0], self.manifest["packages"][1] = (
            self.manifest["packages"][1],
            self.manifest["packages"][0],
        )
        self.manifest["dependency_order"] = [
            package["name"] for package in self.manifest["packages"]
        ]
        errors = self.errors(self.manifest)
        self.assertTrue(any("package order" in error for error in errors), errors)

    def test_missing_consumer_gate_is_rejected(self) -> None:
        self.manifest["required_consumer_checks"].pop()
        errors = self.errors(self.manifest)
        self.assertTrue(any("consumer checks" in error for error in errors), errors)

    def test_downstream_consumer_gate_pins_every_known_repository(self) -> None:
        script = (
            OWNERSHIP_PATH.parents[2]
            / "scripts/check_nlp_wave_1_downstream_consumers.sh"
        ).read_text(encoding="utf-8")
        repositories = (
            "native-whisperx",
            "media-similarity",
            "youtube-corpus",
            "document-search",
            "philosophy-extractor",
            "video-analysis-studio",
            "stutter-tracker",
            "rust-packages",
        )
        for repository in repositories:
            self.assertRegex(
                script,
                rf'clone_pinned "{re.escape(repository)}" "[0-9a-f]{{40}}"',
            )
        self.assertNotRegex(script, r"clone_pinned .*\bmain\b")

    def test_postpublication_consumer_is_registry_only(self) -> None:
        script = (
            OWNERSHIP_PATH.parents[2]
            / "scripts/check_nlp_wave_1_registry_consumer.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('startswith("registry+")', script)
        self.assertNotIn("[patch.crates-io]", script)
        self.assertNotIn("--config", script)

    def test_source_and_control_binding_are_exact(self) -> None:
        errors = validate_control_binding(
            self.manifest,
            Path("releases/nlp-wave-1.toml"),
            "b" * 40,
            False,
            ["Cargo.toml", "releases/nlp-wave-1.toml"],
        )
        self.assertTrue(any("ancestor" in error for error in errors), errors)
        self.assertTrue(any("only by its manifest" in error for error in errors), errors)

    def test_workspace_versions_allow_only_the_exact_nlp_wave(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        for package in metadata["packages"]:
            package["version"] = NLP_WAVE_1_VERSIONS[package["name"]]
        self.assertEqual(validate(self.plan, self.ownership, metadata), [])

        metadata["packages"][0]["version"] = "9.9.9"
        errors = validate(self.plan, self.ownership, metadata)
        self.assertTrue(any("authorized source or wave version" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
