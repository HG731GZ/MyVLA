#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -eq 0 ]]; then
    echo "Usage: bash ${SCRIPT_DIR}/prepare_environment_suite.sh <suite|all> [options]" >&2
    echo "   or: bash ${SCRIPT_DIR}/prepare_environment_suite.sh --suite <suite|all> [options]" >&2
    exit 1
fi

if [[ "$1" == --* ]]; then
    exec "$PYTHON_BIN" -u "${SCRIPT_DIR}/generate_environment_suite.py" "$@"
fi

SUITE="$1"
shift
exec "$PYTHON_BIN" -u "${SCRIPT_DIR}/generate_environment_suite.py" \
    --suite "$SUITE" \
    "$@"
