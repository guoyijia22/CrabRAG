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
PORTABLE_BUN_DIR="$ROOT/runtime/bun"
PACKAGE_JSON="$ROOT/package.json"
RELEASE_MANIFEST="$ROOT/release-manifest.json"
BUN_VERSION="1.3.14"
BUN_RELEASE_BASE_URL="https://github.com/oven-sh/bun/releases/download/bun-v1.3.14"
BUN_LINUX_X64_SHA256="951ee2aee855f08595aeec6225226a298d3fea83a3dcd6465c09cbccdf7e848f"
BUN_LINUX_AARCH64_SHA256="a27ffb63a8310375836e0d6f668ae17fa8d8d18b88c37c821c65331973a19a3b"

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

bun_version_ok() {
  [[ -x "$1" ]] && [[ "$("$1" --version 2>/dev/null || true)" == "$BUN_VERSION" ]]
}

resolve_bun_for_install() {
  if bun_version_ok "$PORTABLE_BUN"; then
    printf '%s\n' "$PORTABLE_BUN"
    return 0
  fi
  local system_bun
  system_bun="$(command -v bun 2>/dev/null || true)"
  if [[ -n "$system_bun" ]] && bun_version_ok "$system_bun"; then
    printf '%s\n' "$system_bun"
    return 0
  fi
  install_portable_bun
  if bun_version_ok "$PORTABLE_BUN"; then
    printf '%s\n' "$PORTABLE_BUN"
    return 0
  fi
  fail "Failed to install project-local Bun $BUN_VERSION."
}

install_portable_bun() {
  local arch bun_archive bun_sha256
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64)
      bun_archive="bun-linux-x64.zip"
      bun_sha256="$BUN_LINUX_X64_SHA256"
      ;;
    aarch64|arm64)
      bun_archive="bun-linux-aarch64.zip"
      bun_sha256="$BUN_LINUX_AARCH64_SHA256"
      ;;
    *) fail "Unsupported CPU architecture for automatic Bun download: $arch" ;;
  esac
  log "Downloading verified project-local Bun $BUN_VERSION to runtime/bun." >&2
  mkdir -p "$PORTABLE_BUN_DIR"
  if ! "$VENV_PYTHON" - "$BUN_RELEASE_BASE_URL" "$bun_archive" "$bun_sha256" "$PORTABLE_BUN" <<'PY'
from pathlib import Path
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

release_base_url = sys.argv[1]
archive_name = sys.argv[2]
expected_sha256 = sys.argv[3]
target = Path(sys.argv[4])
url = f"{release_base_url}/{archive_name}"
with tempfile.TemporaryDirectory() as temp_dir:
    archive = Path(temp_dir) / "bun.zip"
    urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded Bun archive checksum mismatch")
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if name.endswith("/bun") or name == "bun")
        extracted = Path(zf.extract(member, temp_dir))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
  then
    fail "Failed to download project-local Bun. Check network access to GitHub releases or install Bun manually."
  fi
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
if [[ -f "$PACKAGE_JSON" ]]; then
  # Equivalent command: bun install
  if [[ -f "$RELEASE_MANIFEST" ]]; then
    [[ -f "$ROOT/bun.lock" ]] || fail "Release package is missing bun.lock."
    "$BUN_BIN" install --production --frozen-lockfile
  elif [[ -f "$ROOT/bun.lock" ]]; then
    "$BUN_BIN" install --frozen-lockfile
  else
    "$BUN_BIN" install
  fi
else
  log "Skipping JavaScript dependency install because this release uses the bundled gateway."
fi

log "Running smoke check."
"$VENV_PYTHON" "$ROOT/scripts/check_env.py"

log "Install completed. Run ./run.sh and open http://127.0.0.1:3003/."
