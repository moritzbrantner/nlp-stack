#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise SystemExit(f"expected exactly one migration match in {path}: {old!r}")
    target.write_text(source.replace(old, new), encoding="utf-8")


replace_once(
    "docs/agents/READINESS.md",
    "Implementations still validate the narrowest affected scope first and use `scripts/check-preflight.sh` for handoff; `.agent-loop.toml` remains the exact-head handoff gate.",
    "Implementations still validate the narrowest affected scope first. Exact-head handoff is the `handoff` tier in `.coding-tooling.json`; run `coding-tooling run --tier handoff --strict --json`. The tier delegates to the repository-owned `bun run check` gate, so command ownership is not duplicated in a second agent-loop config.",
)

replace_once(
    "scripts/publish_release.py",
    "side effect. The public interface is the no-argument CLI configured in\n``.agent-loop.toml``; ``run_release`` accepts an effects adapter so tests can\nreplace only network and process boundaries.",
    "side effect. The public interface is the no-argument CLI; current exact-head\nhandoff verification is declared by the `handoff` tier in `.coding-tooling.json`.\n``run_release`` accepts an effects adapter so tests can replace only network and\nprocess boundaries.",
)
replace_once(
    "scripts/publish_release.py",
    "import re\nimport subprocess",
    "import re\nimport shlex\nimport subprocess",
)
replace_once(
    "scripts/publish_release.py",
    "class ReleaseError(RuntimeError):\n    \"\"\"A fail-closed release validation or external-operation failure.\"\"\"\n\n\ndef _effect_failure",
    "class ReleaseError(RuntimeError):\n    \"\"\"A fail-closed release validation or external-operation failure.\"\"\"\n\n\ndef configured_handoff_checks(root: Path) -> list[str]:\n    \"\"\"Load the one repository-owned exact-head handoff command.\"\"\"\n\n    path = root / \".coding-tooling.json\"\n    try:\n        config = json.loads(path.read_text(encoding=\"utf-8\"))\n    except (OSError, UnicodeError, json.JSONDecodeError) as error:\n        raise ReleaseError(f\"cannot load coding-tooling handoff config: {error}\") from error\n    if config.get(\"schemaVersion\") != 1:\n        raise ReleaseError(\".coding-tooling.json must use schemaVersion 1\")\n    if config.get(\"tiers\", {}).get(\"handoff\") != [\"package:check\"]:\n        raise ReleaseError(\"coding-tooling handoff tier must contain only package:check\")\n    command = (\n        config.get(\"capabilityCommands\", {})\n        .get(\".\", {})\n        .get(\"package:check\")\n    )\n    if (\n        not isinstance(command, list)\n        or not command\n        or any(not isinstance(part, str) or not part for part in command)\n    ):\n        raise ReleaseError(\"coding-tooling package:check must be a non-empty argv array\")\n    return [shlex.join(command)]\n\n\ndef _effect_failure",
)
replace_once(
    "scripts/publish_release.py",
    "    config = tomllib.loads((root / \".agent-loop.toml\").read_text(encoding=\"utf-8\"))\n    configured_checks = config.get(\"verification\", {}).get(\"commands\")\n    if manifest.get(\"required_checks\") != configured_checks:\n        raise ReleaseError(\"required_checks must exactly match .agent-loop.toml\")",
    "    configured_checks = configured_handoff_checks(root)\n    if manifest.get(\"required_checks\") != configured_checks:\n        raise ReleaseError(\"required_checks must exactly match the coding-tooling handoff gate\")",
)

replace_once(
    "scripts/test_publish_release.py",
    "CONFIG_COMMANDS = tomllib.loads(\n    (SCRIPT.parents[1] / \".agent-loop.toml\").read_text(encoding=\"utf-8\")\n)[\"verification\"][\"commands\"]",
    "CONFIG_COMMANDS = publish_release.configured_handoff_checks(SCRIPT.parents[1])",
)
replace_once(
    "scripts/test_publish_release.py",
    "    (root / \".agent-loop.toml\").write_text(\n        (SCRIPT.parents[1] / \".agent-loop.toml\").read_text(encoding=\"utf-8\"),\n        encoding=\"utf-8\",\n    )",
    "    (root / \".coding-tooling.json\").write_text(\n        (SCRIPT.parents[1] / \".coding-tooling.json\").read_text(encoding=\"utf-8\"),\n        encoding=\"utf-8\",\n    )",
)

replace_once(
    "scripts/test_check_release_plan.py",
    "from check_release_plan import (\n",
    "from publish_release import configured_handoff_checks\n\nfrom check_release_plan import (\n",
)
replace_once(
    "scripts/test_check_release_plan.py",
    "        checks = tomllib.loads(\n            (OWNERSHIP_PATH.parents[2] / \".agent-loop.toml\").read_text(\n                encoding=\"utf-8\"\n            )\n        )[\"verification\"][\"commands\"]",
    "        checks = configured_handoff_checks(OWNERSHIP_PATH.parents[2])",
)

legacy = ROOT / ".agent-loop.toml"
if not legacy.is_file():
    raise SystemExit("expected legacy .agent-loop.toml before migration")
legacy.unlink()

# The GitHub-only migration helper and its workflow are self-removing. They are
# not part of the resulting repository architecture.
(ROOT / ".github/workflows/apply-handoff-migration.yml").unlink()
Path(__file__).unlink()
