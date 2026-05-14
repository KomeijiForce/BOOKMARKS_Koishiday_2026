#!/usr/bin/env bash
set -euo pipefail

ARTIFACT="${1:-MyGO!!!!!}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" python -m bookmarks.benchmark \
  --artifact "${ARTIFACT}" \
  "$@"
