#!/usr/bin/env bash
set -euo pipefail

export QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.openai.com/v1}"
export LLM_API_KEY="${LLM_API_KEY:-}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"
INGEST_ON_STARTUP="${INGEST_ON_STARTUP:-1}"
INGEST_USE_RAY="${INGEST_USE_RAY:-0}"
RECREATE_COLLECTION="${RECREATE_COLLECTION:-0}"
WAIT_FOR_LLM="${WAIT_FOR_LLM:-0}"
SERVICE_TIMEOUT_SECONDS="${SERVICE_TIMEOUT_SECONDS:-1800}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

wait_http() {
  local name="$1"
  local url="$2"
  local timeout_seconds="$3"
  local auth_token_env="${4:-}"

  python - "$name" "$url" "$timeout_seconds" "$auth_token_env" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

name, url, timeout_seconds, auth_token_env = sys.argv[1:5]
deadline = time.monotonic() + float(timeout_seconds)
last_error = None

while True:
    headers = {}
    if auth_token_env:
        token = os.getenv(auth_token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            if 200 <= response.status < 300:
                print(f"{name} is ready: {url}", flush=True)
                sys.exit(0)
            last_error = f"HTTP {response.status}"
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        last_error = exc

    if time.monotonic() >= deadline:
        print(
            f"Timed out waiting for {name} at {url}. Last error: {last_error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(f"Waiting for {name} at {url}...", flush=True)
    time.sleep(5)
PY
}

wait_tcp() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout_seconds="$4"

  python - "$name" "$host" "$port" "$timeout_seconds" <<'PY'
import socket
import sys
import time

name, host, port, timeout_seconds = sys.argv[1:5]
deadline = time.monotonic() + float(timeout_seconds)
last_error = None

while True:
    try:
        with socket.create_connection((host, int(port)), timeout=5):
            print(f"{name} is ready: {host}:{port}", flush=True)
            sys.exit(0)
    except OSError as exc:
        last_error = exc

    if time.monotonic() >= deadline:
        print(
            f"Timed out waiting for {name} at {host}:{port}. "
            f"Last error: {last_error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(f"Waiting for {name} at {host}:{port}...", flush=True)
    time.sleep(5)
PY
}

QDRANT_READY_URL="${QDRANT_READY_URL:-${QDRANT_URL%/}/collections}"
LLM_READY_PATH="${LLM_READY_PATH:-${LLM_HEALTH_PATH:-/models}}"
LLM_READY_URL="${LLM_READY_URL:-${LLM_BASE_URL%/}${LLM_READY_PATH}}"

wait_http "Qdrant" "$QDRANT_READY_URL" "$SERVICE_TIMEOUT_SECONDS"
wait_tcp "PostgreSQL" "$POSTGRES_HOST" "$POSTGRES_PORT" "$SERVICE_TIMEOUT_SECONDS"

if is_true "$INGEST_ON_STARTUP"; then
  ingest_args=()
  if is_true "$RECREATE_COLLECTION"; then
    ingest_args+=(--recreate)
  fi

  if is_true "$INGEST_USE_RAY"; then
    python scripts/ingest_docs.py "${ingest_args[@]}"
  else
    RAY_ENABLED=0 python scripts/ingest_docs.py "${ingest_args[@]}"
  fi
else
  echo "Skipping document ingest because INGEST_ON_STARTUP=${INGEST_ON_STARTUP}"
fi

if is_true "$WAIT_FOR_LLM"; then
  wait_http "LLM" "$LLM_READY_URL" "$SERVICE_TIMEOUT_SECONDS" LLM_API_KEY
else
  echo "Starting API before LLM readiness because WAIT_FOR_LLM=${WAIT_FOR_LLM}"
fi

exec uvicorn app.main:app --host "$API_HOST" --port "$API_PORT"
