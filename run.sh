#!/usr/bin/env bash
set -Eeuo pipefail

WEB_PORT="${WEB_PORT:-3003}"
API_PORT="${API_PORT:-8001}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./run.sh"
  echo "Optional environment variables: WEB_PORT=3003 API_PORT=8001"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT/.venv/bin/python"
PORTABLE_PYTHON="$ROOT/runtime/python/python"
PORTABLE_BUN="$ROOT/runtime/bun/bun"
RUN_STATE="$ROOT/data/run.json"
RUN_STATE_WRITTEN="0"
API_PID=""
WEB_PID=""

log() {
  printf '[CrabRAG] %s\n' "$1"
}

fail() {
  printf '[CrabRAG] ERROR: %s\n' "$1" >&2
  exit 1
}

cleanup() {
  for pid in "$WEB_PID" "$API_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      log "Stopping process $pid."
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  if [[ "$RUN_STATE_WRITTEN" == "1" ]]; then
    rm -f "$RUN_STATE"
  fi
}
trap cleanup EXIT INT TERM

if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
elif [[ -x "$PORTABLE_PYTHON" ]]; then
  PYTHON_BIN="$PORTABLE_PYTHON"
else
  fail "Python runtime not found. Run ./install.sh first."
fi

if [[ -x "$PORTABLE_BUN" ]]; then
  BUN_BIN="$PORTABLE_BUN"
elif command -v bun >/dev/null 2>&1; then
  BUN_BIN="$(command -v bun)"
else
  fail "Bun was not found. Install Bun with: curl -fsSL https://bun.sh/install | bash"
fi

port_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

port_free "$API_PORT" || fail "API port $API_PORT is already in use."
port_free "$WEB_PORT" || fail "Web port $WEB_PORT is already in use."

export CRABRAG_ROOT="$ROOT"
export ELCQA_ROOT="$ROOT"
export CRABRAG_ENV_FILE="$ROOT/config/.env"
export ELCQA_ENV_FILE="$CRABRAG_ENV_FILE"
export RAG_BASE_URL="http://127.0.0.1:$API_PORT"
export PORT="$WEB_PORT"
export CRABRAG_INTERNAL_TOKEN="${CRABRAG_INTERNAL_TOKEN:-$("$PYTHON_BIN" -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export PYTHONUTF8=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"

log "Starting API on http://127.0.0.1:$API_PORT"
"$PYTHON_BIN" -m uvicorn services.rag_api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID="$!"

sleep 2

log "Starting web gateway on http://127.0.0.1:$WEB_PORT"
"$BUN_BIN" server/gateway.js &
WEB_PID="$!"

mkdir -p "$(dirname "$RUN_STATE")"
"$PYTHON_BIN" - "$RUN_STATE" "$ROOT" "$WEB_PORT" "$API_PORT" "$API_PID" "$WEB_PID" <<'PY'
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

path = Path(sys.argv[1])

def start_identity(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return fields[19]

api_pid = int(sys.argv[5])
web_pid = int(sys.argv[6])
payload = {
    "schema_version": 1,
    "project_root": str(Path(sys.argv[2]).resolve()),
    "web_port": int(sys.argv[3]),
    "api_port": int(sys.argv[4]),
    "pids": [api_pid, web_pid],
    "processes": [
        {"pid": api_pid, "role": "api", "start_identity": start_identity(api_pid)},
        {"pid": web_pid, "role": "web", "start_identity": start_identity(web_pid)},
    ],
    "started_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload), encoding="utf-8")
os.replace(temporary, path)
PY
RUN_STATE_WRITTEN="1"

log "CrabRAG is starting. Open http://127.0.0.1:$WEB_PORT/. Press Ctrl+C to stop."
while kill -0 "$API_PID" >/dev/null 2>&1 && kill -0 "$WEB_PID" >/dev/null 2>&1; do
  sleep 1
done

wait "$API_PID" || true
wait "$WEB_PID" || true
fail "One of the CrabRAG processes exited."
