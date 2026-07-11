#!/usr/bin/env bash
set -euo pipefail

update_services=(
  ray-worker-embedding-ingest-cpu
  docs-updater
)

stop_failed_update() {
  status=$?
  trap - ERR
  echo "Document update failed; active online models and index remain unchanged." >&2
  docker compose --profile docs-update stop "${update_services[@]}" || true
  exit "$status"
}

trap stop_failed_update ERR
docker compose --profile docs-update up \
  -d \
  --no-deps \
  --force-recreate \
  "${update_services[@]}"
docker compose --profile docs-update wait "${update_services[@]}"
trap - ERR

echo "Document update completed without stopping online GPU model workers."
