#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CASE_DIR [MPI_RANKS] [runner options]" >&2
    exit 2
fi
CASE_DIR="$1"
shift
NPROCS="${1:-6}"
if [[ $# -gt 0 ]]; then shift; fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/cfd_case.py" run "$CASE_DIR" --ranks "$NPROCS" "$@"
