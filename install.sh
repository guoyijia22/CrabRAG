#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./install.sh"
  echo "Creates .venv, installs Python dependencies, runs bun install, and copies config/.env.example to config/.env when missing."
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PORTABLE_PYTHON="$ROOT/runtime/python/python"
REQUIREMENTS="$ROOT/requirements.txt"
ENV_EXAMPLE="$ROOT/config/.env.example"
ENV_FILE="$ROOT/config/.env"
PORTABLE_BUN="$ROOT/runtime/bun/bun"

log() {
  printf '[CrabRAG] %s\n' "$1"
}

fail() {
  printf '[CrabRAG] ERROR: %s\n' "$1" >&2
  exit 1
}

linux_install_hint() {
  cat >&2 <<'EOF'
Install Python 3.10+ and venv support first.
Ubuntu/Debian:
  sudo apt update && sudo apt install -y python3 python3-venv python3-pip
CentOS/Rocky/AlmaLinux:
  sudo dnf install -y python3 python3-pip
Older CentOS:
  sudo yum install -y python3 python3-pip
EOF
}

python_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

find_python() {
  if [[ -x "$PORTABLE_PYTHON" ]] && python_ok "$PORTABLE_PYTHON"; then
    printf '%s\n' "$PORTABLE_PYTHON"
    return 0
  fi
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  linux_install_hint
  return 1
}

resolve_bun_for_install() {
  if [[ -x "$PORTABLE_BUN" ]]; then
    printf '%s\n' "$PORTABLE_BUN"
    return 0
  fi
  if command -v bun >/dev/null 2>&1; then
    command -v bun
    return 0
  fi
  fail "bun was not found. Install Bun with: curl -fsSL https://bun.sh/install | bash"
}

use_safe_pip_index_if_needed() {
  local configured_index
  configured_index="$("$VENV_PYTHON" -m pip config get global.index-url 2>/dev/null || true)"
  local effective_index="${PIP_INDEX_URL:-$configured_index}"
  if [[ "$effective_index" == http://* && -z "${PIP_TRUSTED_HOST:-}" ]]; then
    export PIP_INDEX_URL="https://pypi.org/simple"
    log "Detected an insecure HTTP pip index. Using https://pypi.org/simple for this install. Set PIP_INDEX_URL to override."
  fi
}

log "Repository root: $ROOT"

mkdir -p "$ROOT/docs" "$ROOT/data" "$ROOT/logs" "$ROOT/runtime/models"

if [[ ! -f "$ENV_EXAMPLE" ]]; then
  fail "Missing config/.env.example."
fi

if [ -f "$ENV_FILE" ]; then
  log "Keeping existing config/.env."
else
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  log "Created config/.env from config/.env.example."
fi

PYTHON_BIN="$(find_python)" || fail "Python 3.10+ was not found."

if [[ ! -x "$VENV_PYTHON" ]]; then
  log "Creating Python virtual environment in .venv."
  # Equivalent command: python3 -m venv .venv
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    linux_install_hint
    fail "Failed to create .venv."
  fi
else
  log "Reusing existing .venv."
fi

use_safe_pip_index_if_needed

log "Installing Python dependencies."
"$VENV_PYTHON" -m pip install --upgrade pip
# Equivalent command: pip install -r requirements.txt
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"

BUN_BIN="$(resolve_bun_for_install)"
for optional in node npm pnpm; do
  if command -v "$optional" >/dev/null 2>&1; then
    log "$optional detected: $(command -v "$optional")"
  else
    log "$optional not found; it is optional for this bundled frontend."
  fi
done

log "Installing JavaScript dependencies with Bun."
cd "$ROOT"
# Equivalent command: bun install
if [[ -f "$ROOT/bun.lock" ]]; then
  "$BUN_BIN" install --frozen-lockfile
else
  "$BUN_BIN" install
fi

log "Running smoke check."
"$VENV_PYTHON" "$ROOT/scripts/check_env.py"

log "Install completed. Run ./run.sh and open http://127.0.0.1:3003/."
