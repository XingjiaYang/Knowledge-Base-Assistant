#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- up --build
fi

is_false() {
  case "${1,,}" in
    0|false|no|n|off) return 0 ;;
    *) return 1 ;;
  esac
}

is_gpu_error_log() {
  grep -Eiq \
    "could not select device driver|capabilities:.*gpu|nvidia-container|nvidia runtime|no nvidia|gpu.*not|driver.*gpu|unknown runtime.*nvidia" \
    "$1"
}

run_compose() {
  local output_file="$1"
  shift

  set +e
  docker compose "$@" 2>&1 | tee "$output_file"
  local status="${PIPESTATUS[0]}"
  set -e
  return "$status"
}

if is_false "${CUDA:-TRUE}"; then
  echo "CUDA=FALSE; starting with CPU Compose override." >&2
  exec docker compose -f compose.yaml -f compose.cpu.yaml "$@"
fi

log_file="$(mktemp -t kba-compose-gpu.XXXXXX.log)"
trap 'rm -f "$log_file"' EXIT

if run_compose "$log_file" "$@"; then
  exit 0
fi

if ! is_gpu_error_log "$log_file"; then
  exit 1
fi

echo >&2
echo "GPU Compose startup failed before the app could run; retrying on CPU." >&2
echo "Set up NVIDIA Container Toolkit/GPU access to use CUDA, or set CUDA=FALSE." >&2
echo >&2

CUDA=FALSE exec docker compose -f compose.yaml -f compose.cpu.yaml "$@"
