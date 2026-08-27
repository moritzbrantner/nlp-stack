#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

tooling_dir="${CODING_TOOLING_DIR:-$root/../coding-tooling}"
mode="${1:-quick}"

case "$mode" in
  quick) ;;
  --with-source) ;;
  *)
    printf '%s\n' "usage: scripts/check-agent-readiness.sh [--with-source]" >&2
    exit 2
    ;;
esac

run_tooling() {
  if command -v coding-tooling >/dev/null 2>&1; then
    coding-tooling "$@"
    return
  fi
  if [[ -f "$tooling_dir/src/cli.ts" ]]; then
    bun "$tooling_dir/src/cli.ts" "$@"
    return
  fi
  printf '%s\n' "coding-tooling is required. Install it or set CODING_TOOLING_DIR to its checkout." >&2
  exit 2
}

resolve_skills_root() {
  if [[ -n "${CODING_AGENT_SKILLS_ROOT:-}" && -d "$CODING_AGENT_SKILLS_ROOT" ]]; then
    printf '%s\n' "$CODING_AGENT_SKILLS_ROOT"
    return
  fi
  if [[ -d "$root/../coding-agent-skills" ]]; then
    printf '%s\n' "$root/../coding-agent-skills"
    return
  fi

  local registry="${MOENARCH_ENVIRONMENT_REGISTRY:-${XDG_CONFIG_HOME:-$HOME/.config}/moenarch/environment.toml}"
  if [[ -f "$registry" ]]; then
    local registered
    registered="$(python3 - "$registry" <<'PY'
from pathlib import Path
import sys
import tomllib

path = Path(sys.argv[1])
try:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
entry = data.get("components", {}).get("coding-agent-skills", {})
value = entry.get("path", "") if isinstance(entry, dict) else ""
print(value if isinstance(value, str) else "")
PY
)"
    if [[ -n "$registered" && -d "$registered" ]]; then
      printf '%s\n' "$registered"
      return
    fi
  fi

  printf '%s\n' "coding-agent-skills is required. Set CODING_AGENT_SKILLS_ROOT, register it in the Moenarch environment, or keep it as a sibling checkout." >&2
  exit 2
}

python3 scripts/check_agent_readiness_contract.py --check
run_tooling conventions resolve --root "$root" --config "$root/.coding-tooling.json" --json

skills_root="$(resolve_skills_root)"
run_tooling agent-capabilities validate --root "$skills_root" --json
run_tooling agent-capabilities profile standard --root "$skills_root" --json

if [[ "$mode" == "--with-source" ]]; then
  source_status="$(run_tooling source-deps status --config "$root/.coding-tooling.source-deps.json" --json)"
  printf '%s\n' "$source_status"
  was_active="$(printf '%s' "$source_status" | python3 -c 'import json,sys; print("true" if json.load(sys.stdin).get("data", {}).get("active") else "false")')"
  activated_here=false

  cleanup() {
    if [[ "$activated_here" == "true" ]]; then
      run_tooling source-deps deactivate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null || true
    fi
  }
  trap cleanup EXIT

  run_tooling source-deps activate --config "$root/.coding-tooling.source-deps.json" --json
  if [[ "$was_active" != "true" ]]; then
    activated_here=true
  fi

  cargo metadata --format-version 1 --no-deps >/dev/null

  if [[ "$activated_here" == "true" ]]; then
    run_tooling source-deps deactivate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null
    activated_here=false
  fi
  trap - EXIT
fi

printf '%s\n' "agent readiness: passed"
