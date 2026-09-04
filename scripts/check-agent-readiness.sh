#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

tooling_dir="${CODING_TOOLING_DIR:-$root/../coding-tooling}"
mode="${1:-quick}"
minimum_free_gib="${AGENT_MIN_FREE_GIB:-8}"
target_dir="${CARGO_TARGET_DIR:-$root/target}"
target_parent="$(dirname "$target_dir")"
mkdir -p "$target_parent"

case "$mode" in
  quick) profile="default" ;;
  --with-source) profile="source-development" ;;
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
    if ! command -v bun >/dev/null 2>&1; then
      printf '%s\n' "bun is required to run the sibling coding-tooling checkout" >&2
      exit 2
    fi
    bun "$tooling_dir/src/cli.ts" "$@"
    return
  fi
  printf '%s\n' "coding-tooling is required. Install it or set CODING_TOOLING_DIR to its checkout." >&2
  exit 2
}

registered_component_path() {
  local registry="$1"
  local component="$2"
  python3 - "$registry" "$component" <<'PY'
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
header = f"[components.{sys.argv[2]}]"
active = False
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line.startswith("[") and line.endswith("]"):
        active = line == header
        continue
    if not active:
        continue
    match = re.match(r'^path\s*=\s*("(?:[^"\\]|\\.)*")\s*$', line)
    if match:
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            break
        if isinstance(value, str):
            print(value)
        break
PY
}

resolve_skills_root() {
  if [[ -n "${CODING_AGENT_SKILLS_ROOT:-}" && -d "$CODING_AGENT_SKILLS_ROOT" ]]; then
    printf '%s\n' "$CODING_AGENT_SKILLS_ROOT"
    return
  fi

  local registry="${MOENARCH_ENVIRONMENT_REGISTRY:-${XDG_CONFIG_HOME:-$HOME/.config}/moenarch/environment.toml}"
  if [[ -f "$registry" ]]; then
    local registered
    registered="$(registered_component_path "$registry" "coding-agent-skills")"
    if [[ -n "$registered" && -d "$registered" ]]; then
      printf '%s\n' "$registered"
      return
    fi
  fi

  if [[ -d "$root/../coding-agent-skills" ]]; then
    printf '%s\n' "$root/../coding-agent-skills"
    return
  fi

  printf '%s\n' "coding-agent-skills is required. Set CODING_AGENT_SKILLS_ROOT, register it in the Moenarch environment, or keep it as a sibling checkout." >&2
  exit 2
}

if ! [[ "$minimum_free_gib" =~ ^[0-9]+$ ]]; then
  printf 'AGENT_MIN_FREE_GIB must be a non-negative integer, got %s\n' "$minimum_free_gib" >&2
  exit 2
fi

free_kib="$(df -Pk "$target_parent" | awk 'NR == 2 { print $4 }')"
required_kib="$((minimum_free_gib * 1024 * 1024))"
if [[ -z "$free_kib" || "$free_kib" -lt "$required_kib" ]]; then
  printf 'insufficient free disk for Cargo target: require %s GiB at %s\n' "$minimum_free_gib" "$target_parent" >&2
  df -Ph "$target_parent" >&2 || true
  exit 1
fi

python3 scripts/check_agent_readiness_contract.py --check
run_tooling conventions resolve --root "$root" --json

skills_root="$(resolve_skills_root)"
run_tooling agent-capabilities validate --root "$skills_root" --json
run_tooling agent-capabilities profile standard --root "$skills_root" --json

activated_here=false
cleanup() {
  if [[ "$activated_here" == "true" ]]; then
    run_tooling source-deps deactivate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$mode" == "--with-source" ]]; then
  source_status="$(run_tooling source-deps status --config "$root/.coding-tooling.source-deps.json" --json)"
  printf '%s\n' "$source_status"
  was_active="$(printf '%s' "$source_status" | python3 -c 'import json,sys; print("true" if json.load(sys.stdin).get("data", {}).get("active") else "false")')"
  run_tooling source-deps activate --config "$root/.coding-tooling.source-deps.json" --json
  if [[ "$was_active" != "true" ]]; then
    activated_here=true
  fi
fi

receipt="$(mktemp)"
trap 'rm -f "$receipt"; cleanup' EXIT
if ! run_tooling environment verify --profile "$profile" --json > "$receipt"; then
  cat "$receipt" >&2
  exit 1
fi

fingerprint="$(python3 - "$receipt" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    receipt = json.load(handle)
if receipt.get('status') != 'passed':
    raise SystemExit(f"environment verification did not pass: {receipt.get('status')}")
data = receipt.get('data', {})
expected = data.get('expectedFingerprint')
verified = data.get('verifiedFingerprint')
if not expected or verified != expected:
    raise SystemExit('environment fingerprint was not verified')
print(verified)
PY
)"

if [[ "$mode" == "quick" ]]; then
  cargo metadata --locked --format-version 1 --no-deps >/dev/null
else
  cargo metadata --format-version 1 --no-deps >/dev/null
fi

if [[ "$activated_here" == "true" ]]; then
  run_tooling source-deps deactivate --config "$root/.coding-tooling.source-deps.json" --json >/dev/null
  activated_here=false
fi
rm -f "$receipt"
trap - EXIT

free_gib="$((free_kib / 1024 / 1024))"
printf 'agent readiness: passed (profile=%s, fingerprint=%s, free-disk=%sGiB)\n' "$profile" "${fingerprint:0:12}" "$free_gib"
