#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

: "${UEH_CALIBRATION_CACHE_HOST_PATH:?Set UEH_CALIBRATION_CACHE_HOST_PATH to the approved calibration cache}"
: "${UEH_SERVICE_SAMPLES_HOST_PATH:?Set UEH_SERVICE_SAMPLES_HOST_PATH to the approved adult service samples}"
cache_path="$(realpath "$UEH_CALIBRATION_CACHE_HOST_PATH")"
samples_path="$(realpath "$UEH_SERVICE_SAMPLES_HOST_PATH")"

for input_path in "$cache_path" "$samples_path"; do
  if [[ ! -f "$input_path" || ! -r "$input_path" ]]; then
    echo "Runtime inputs must be readable regular files: $input_path" >&2
    exit 1
  fi
  mode="$(stat -c '%a' "$input_path")"
  mode_value=$((8#$mode))
  if (( mode_value & 0007 )); then
    echo "Runtime inputs are readable by other host users; restrict permissions: $input_path" >&2
    exit 1
  fi
  case "$input_path" in
    "$project_dir"/*)
      echo "Keep runtime inputs outside the project and Docker build context." >&2
      exit 1
      ;;
  esac
done

expected_sha="5edef6b238a0723954b3c678d6b8192a3af672afb227245332daa58ad2a74cea"
actual_sha="$(sha256sum "$cache_path" | cut -d ' ' -f 1)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Calibration cache SHA-256 does not match the approved F3.1/H.1-R6 input." >&2
  exit 1
fi
expected_samples_sha="340f27d51b7638ab105b65708a98757141eacf4c1094fea23f46cbd796ae94c5"
actual_samples_sha="$(sha256sum "$samples_path" | cut -d ' ' -f 1)"
if [[ "$actual_samples_sha" != "$expected_samples_sha" ]]; then
  echo "Adult service samples SHA-256 does not match the approved 2025 input." >&2
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

UEH_CALIBRATION_CACHE_HOST_PATH="$cache_path" UEH_SERVICE_SAMPLES_HOST_PATH="$samples_path" docker compose config --quiet
echo "Preflight passed. Both runtime input digests are approved and Compose configuration resolves."
echo "Before starting, verify container UID/GID 10001 can read both mounted secrets."
