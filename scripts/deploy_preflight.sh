#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

: "${UEH_CALIBRATION_CACHE_HOST_PATH:?Set UEH_CALIBRATION_CACHE_HOST_PATH to the approved calibration cache}"
cache_path="$(realpath "$UEH_CALIBRATION_CACHE_HOST_PATH")"

if [[ ! -f "$cache_path" || ! -r "$cache_path" ]]; then
  echo "Calibration cache must be a readable regular file." >&2
  exit 1
fi

mode="$(stat -c '%a' "$cache_path")"
mode_value=$((8#$mode))
if (( mode_value & 0007 )); then
  echo "Calibration cache is readable by other host users; restrict its permissions before deployment." >&2
  exit 1
fi

case "$cache_path" in
  "$project_dir"/*)
    echo "Keep the restricted calibration cache outside the project and Docker build context." >&2
    exit 1
    ;;
esac

expected_sha="5edef6b238a0723954b3c678d6b8192a3af672afb227245332daa58ad2a74cea"
actual_sha="$(sha256sum "$cache_path" | cut -d ' ' -f 1)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Calibration cache SHA-256 does not match the approved F3.1/H.1-R6 input." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker Engine is not installed or not available in PATH." >&2
  exit 2
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is not available through 'docker compose'." >&2
  exit 2
fi

UEH_CALIBRATION_CACHE_HOST_PATH="$cache_path" docker compose config --quiet
echo "Preflight passed. Cache digest is approved and Compose configuration resolves."
echo "Before starting, verify container UID/GID 10001 can read the mounted cache."
