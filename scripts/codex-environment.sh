#!/usr/bin/env bash
set -euo pipefail

mode="${1:-setup}"
if [[ "$mode" != "setup" && "$mode" != "maintenance" ]]; then
  printf 'usage: %s [setup|maintenance]\n' "$0" >&2
  exit 2
fi

root="$(git rev-parse --show-toplevel)"
config="$root/.repository-environment.toml"

if [[ ! -f "$config" ]]; then
  printf 'missing environment-v1 config: %s\n' "$config" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  printf 'environment-v1 requires Python 3 with tomllib as a bootstrap prerequisite\n' >&2
  exit 2
fi
if ! python3 - <<'PY' >/dev/null 2>&1
import tomllib
PY
then
  printf 'environment-v1 requires Python 3.11+ or another Python 3 providing tomllib\n' >&2
  exit 2
fi

run_privileged() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

publish_path() {
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    printf '%s\n' "$1" >> "$GITHUB_PATH"
  fi
}

if [[ "$mode" == "setup" ]] && command -v apt-get >/dev/null 2>&1; then
  mapfile -t apt_packages < <(python3 - "$config" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as handle:
    data = tomllib.load(handle)
for package in data.get('system', {}).get('apt', []):
    print(package)
PY
  )
  missing_apt_packages=()
  for package in "${apt_packages[@]}"; do
    if command -v dpkg >/dev/null 2>&1 && dpkg -s "$package" >/dev/null 2>&1; then
      continue
    fi
    missing_apt_packages+=("$package")
  done
  if (( ${#missing_apt_packages[@]} )); then
    run_privileged apt-get update
    run_privileged apt-get install -y --no-install-recommends "${missing_apt_packages[@]}"
  fi
fi

desired_bun="$(python3 - "$root/package.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.is_file():
    value = json.loads(path.read_text()).get('packageManager', '')
    if value.startswith('bun@'):
        print(value.split('@', 1)[1])
PY
)"
if [[ -n "$desired_bun" ]]; then
  if ! [[ "$desired_bun" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf 'Bun packageManager must use an exact version, got %s\n' "$desired_bun" >&2
    exit 2
  fi
  if ! command -v bun >/dev/null 2>&1; then
    printf 'Bun %s is required but is not installed; provision the exact version with a trusted pinned environment mechanism before running this script\n' "$desired_bun" >&2
    exit 2
  fi
  if [[ "$(bun --version)" != "$desired_bun" ]]; then
    printf 'Bun preflight mismatch: expected %s, got %s\n' "$desired_bun" "$(bun --version)" >&2
    exit 1
  fi
  if [[ -d "$HOME/.bun/bin" ]]; then
    export PATH="$HOME/.bun/bin:$PATH"
    publish_path "$HOME/.bun/bin"
  fi
fi

desired_node="$(python3 - "$root/.node-version" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.is_file():
    print(path.read_text().strip())
PY
)"
if [[ -n "$desired_node" ]]; then
  if ! [[ "$desired_node" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '.node-version must use an exact version, got %s\n' "$desired_node" >&2
    exit 2
  fi
  if ! command -v node >/dev/null 2>&1; then
    printf 'Node %s is required but is not installed; provision the exact version with a trusted pinned environment mechanism before running this script\n' "$desired_node" >&2
    exit 2
  fi
  observed_node="$(node --version)"
  observed_node="${observed_node#v}"
  if [[ "$observed_node" != "$desired_node" ]]; then
    printf 'Node preflight mismatch: expected %s, got %s\n' "$desired_node" "$observed_node" >&2
    exit 1
  fi
fi

rust_toolchain="$(python3 - "$root/rust-toolchain.toml" <<'PY'
import pathlib, sys, tomllib
path = pathlib.Path(sys.argv[1])
if path.is_file():
    value = tomllib.loads(path.read_text()).get('toolchain', {}).get('channel', '')
    print(value)
PY
)"
if [[ -n "$rust_toolchain" ]]; then
  if ! [[ "$rust_toolchain" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf 'Rust toolchain must use an exact version, got %s\n' "$rust_toolchain" >&2
    exit 2
  fi
  if ! command -v rustup >/dev/null 2>&1; then
    printf 'rustup is required to provision Rust %s; install rustup through a trusted pinned environment mechanism before running this script\n' "$rust_toolchain" >&2
    exit 2
  fi
  if [[ -d "$HOME/.cargo/bin" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    publish_path "$HOME/.cargo/bin"
  fi
  if ! rustup run "$rust_toolchain" rustc --version >/dev/null 2>&1; then
    rustup toolchain install "$rust_toolchain" --profile minimal
  fi
  mapfile -t rust_components < <(python3 - "$root/rust-toolchain.toml" <<'PY'
import pathlib, sys, tomllib
path = pathlib.Path(sys.argv[1])
if path.is_file():
    for component in tomllib.loads(path.read_text()).get('toolchain', {}).get('components', []):
        print(component)
PY
  )
  installed_rust_components="$(rustup component list --toolchain "$rust_toolchain" --installed)"
  for component in "${rust_components[@]}"; do
    if printf '%s\n' "$installed_rust_components" | grep -Eq "^${component}(-[^ ]+)? \\(installed\\)$"; then
      continue
    fi
    rustup component add --toolchain "$rust_toolchain" "$component"
  done
fi

receipt="$(cd "$root" && git rev-parse --git-path environment-v1-maintenance.sha256)"
if [[ "$receipt" != /* ]]; then
  receipt="$root/$receipt"
fi
mapfile -t reconciliation_state < <(python3 - "$root" "$config" <<'PY'
import hashlib, pathlib, subprocess, sys, tomllib

root = pathlib.Path(sys.argv[1])
config = pathlib.Path(sys.argv[2])
with config.open('rb') as handle:
    data = tomllib.load(handle)
commands = data.get('maintenance', {}).get('commands', [])
cacheable_commands = {
    'bun install --frozen-lockfile',
    'cargo fetch --locked',
}
if not isinstance(commands, list) or any(command not in cacheable_commands for command in commands):
    print('uncacheable')
    raise SystemExit(0)

tracked = subprocess.run(
    ['git', '-C', str(root), 'ls-files', '-z'],
    check=True,
    stdout=subprocess.PIPE,
).stdout.decode().split('\0')
special = {
    '.repository-environment.toml',
    'scripts/codex-environment.sh',
    '.node-version',
    'rust-toolchain.toml',
    'Cargo.lock',
    'bun.lock',
    'bun.lockb',
    'bunfig.toml',
    '.npmrc',
    '.cargo/config',
    '.cargo/config.toml',
}
selected = {
    relative
    for relative in tracked
    if relative and (pathlib.PurePosixPath(relative).name in {'package.json', 'Cargo.toml'} or relative in special)
}
for relative in special:
    if (root / relative).is_file():
        selected.add(relative)

fingerprint = hashlib.sha256()
for relative in sorted(selected):
    file_path = root / relative
    if not file_path.is_file():
        continue
    fingerprint.update(relative.encode())
    fingerprint.update(b'\0')
    fingerprint.update(file_path.read_bytes())
    fingerprint.update(b'\0')
print('cacheable')
print(fingerprint.hexdigest())
PY
)

reconciliation_fingerprint=""
skip_reconciliation=0
if [[ "${reconciliation_state[0]:-}" == "cacheable" ]]; then
  reconciliation_fingerprint="${reconciliation_state[1]:-}"
  if [[ "$mode" == "maintenance" && -n "$reconciliation_fingerprint" && -f "$receipt" && "$(cat "$receipt")" == "$reconciliation_fingerprint" ]]; then
    skip_reconciliation=1
  fi
else
  rm -f "$receipt"
fi

mapfile -t environment_commands < <(python3 - "$config" "$mode" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as handle:
    data = tomllib.load(handle)
for command in data.get(sys.argv[2], {}).get('commands', []):
    print(command)
PY
)
if (( skip_reconciliation )); then
  printf 'environment-v1: maintenance inputs unchanged; skipping %d reconciliation command(s)\n' "${#environment_commands[@]}"
else
  for command in "${environment_commands[@]}"; do
    (cd "$root" && bash -c "$command")
  done
fi

if [[ -n "$desired_bun" && "$(bun --version)" != "$desired_bun" ]]; then
  printf 'Bun preflight mismatch: expected %s, got %s\n' "$desired_bun" "$(bun --version)" >&2
  exit 1
fi
if [[ -n "$desired_node" ]]; then
  observed_node="$(node --version)"
  observed_node="${observed_node#v}"
  if [[ "$observed_node" != "$desired_node" ]]; then
    printf 'Node preflight mismatch: expected %s, got %s\n' "$desired_node" "$observed_node" >&2
    exit 1
  fi
fi
if [[ -n "$rust_toolchain" ]]; then
  observed_rust="$(rustup run "$rust_toolchain" rustc --version | awk '{print $2}')"
  if [[ "$observed_rust" != "$rust_toolchain" ]]; then
    printf 'Rust preflight mismatch: expected %s, got %s\n' "$rust_toolchain" "$observed_rust" >&2
    exit 1
  fi
fi

if [[ -n "$reconciliation_fingerprint" ]]; then
  mkdir -p "$(dirname "$receipt")"
  printf '%s\n' "$reconciliation_fingerprint" > "$receipt"
fi
