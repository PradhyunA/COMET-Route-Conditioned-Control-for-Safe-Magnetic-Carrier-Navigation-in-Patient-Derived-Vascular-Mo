#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamag_matplotlib}"
mkdir -p "$MPLCONFIGDIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -dimsu)
fi
exec "${KEEP_AWAKE[@]}" "$PYTHON_BIN" -u scripts/run_constrained_cem_pilot.py \
  --objective "${OBJECTIVE:-default}" --steps "${STEPS:-2000}" \
  --workers "${WORKERS:-1}" --seed "${SEED:-20260611}" "$@"
