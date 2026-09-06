#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_CONVENTION_REFS = {
    "AGENT-001",
    "AGENT-002",
    "AGENT-003",
    "AGENT-007",
    "AGENT-008",
    "AGENT-009",
    "TEST-002",
    "TEST-005",
    "TEST-006",
    "TEST-007",
    "RUST-001",
    "RUST-002",
    "RUST-003",
}
HANDOFF_TIER = ["package:check"]
HANDOFF_COMMAND = ["bun", "run", "check"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_contract(root: Path) -> list[str]:
    errors: list[str] = []

    tooling_path = root / ".coding-tooling.json"
    if not tooling_path.is_file():
        errors.append("missing .coding-tooling.json")
    else:
        tooling = _load_json(tooling_path)
        if tooling.get("schemaVersion") != 1:
            errors.append(".coding-tooling.json must use schemaVersion 1")
        refs = set(tooling.get("conventionRefs", []))
        missing = sorted(REQUIRED_CONVENTION_REFS - refs)
        if missing:
            errors.append(f"missing required convention refs: {', '.join(missing)}")
        if tooling.get("tiers", {}).get("handoff") != HANDOFF_TIER:
            errors.append("coding-tooling handoff tier must contain only package:check")
        if (
            tooling.get("capabilityCommands", {})
            .get(".", {})
            .get("package:check")
            != HANDOFF_COMMAND
        ):
            errors.append("coding-tooling package:check must delegate to bun run check")

    source_path = root / ".coding-tooling.source-deps.json"
    if not source_path.is_file():
        errors.append("missing .coding-tooling.source-deps.json")
    else:
        source = _load_json(source_path)
        cargo = source.get("cargo", {})
        patches = cargo.get("patches", [])
        if source.get("schemaVersion") != 2:
            errors.append("local-only source dependencies must use schemaVersion 2")
        if cargo.get("localOnly") is not True:
            errors.append("foundation source dependencies must remain localOnly")
        if not patches:
            errors.append("foundation source dependency list must not be empty")
        revisions = {patch.get("rev") for patch in patches}
        if len(revisions) != 1 or any(not isinstance(rev, str) or len(rev) != 40 for rev in revisions):
            errors.append("all foundation patches must share one exact 40-character revision")
        for patch in patches:
            if not patch.get("localPath"):
                errors.append(f"source patch {patch.get('package', '<unknown>')} is missing localPath")

    environment_path = root / ".repository-environment.toml"
    if not environment_path.is_file():
        errors.append("missing .repository-environment.toml")
    else:
        environment = environment_path.read_text(encoding="utf-8")
        if "schema_version = 1" not in environment:
            errors.append("repository environment must use schema_version 1")
        if 'track = "latest-stable"' not in environment:
            errors.append("repository environment must track latest-stable")

    codex_environment_path = root / "scripts" / "codex-environment.sh"
    if not codex_environment_path.is_file():
        errors.append("missing scripts/codex-environment.sh")

    readiness_path = root / "scripts" / "check-agent-readiness.sh"
    if not readiness_path.is_file():
        errors.append("missing scripts/check-agent-readiness.sh")
    else:
        readiness = readiness_path.read_text(encoding="utf-8")
        if "environment verify" not in readiness:
            errors.append("agent readiness must verify the semantic repository environment")

    agents_path = root / "AGENTS.md"
    agents = agents_path.read_text(encoding="utf-8") if agents_path.is_file() else ""
    if "scripts/check-agent-readiness.sh" not in agents:
        errors.append("AGENTS.md must point implementations at the readiness canary")
    if "scripts/codex-environment.sh" not in agents:
        errors.append("AGENTS.md must point fresh environments at environment-v1 setup")
    if "coding-agent-conventions" not in agents:
        errors.append("AGENTS.md must keep shared convention ownership explicit")

    if (root / ".agent-loop.toml").exists():
        errors.append("legacy .agent-loop.toml must not duplicate the coding-tooling handoff gate")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        parser.error("--check is required")

    root = Path(__file__).resolve().parents[1]
    errors = check_contract(root)
    if errors:
        for error in errors:
            print(f"agent-readiness: {error}")
        return 1
    print("agent-readiness contract: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
