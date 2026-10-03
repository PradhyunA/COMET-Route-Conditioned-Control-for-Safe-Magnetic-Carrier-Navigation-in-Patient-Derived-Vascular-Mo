#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Keep each worker single-threaded; trial-level parallelism is controlled below.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamag_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

WORKERS="${WORKERS:-6}"
SEED="${SEED:-20260609}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="pets_cem_mpc_45trials_${STAMP}"
OUTPUT_DIR="outputs/publication_benchmark_1mm"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"

echo "PETS-CEM-MPC 45-case benchmark"
echo "Cases: 3 flow scales x (3 canonical + 12 random) = 45"
echo "Controller: pets_cem_mpc only"
echo "Workers: $WORKERS"
echo "Seed: $SEED"
echo "Output: $RUN_DIR"

KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -dimsu)
fi

"${KEEP_AWAKE[@]}" "$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --vessel-file data/real_vessel_segments_cm_corrected.pkl \
  --flow-scales 1,3,5 \
  --n-random-trials 12 \
  --seed "$SEED" \
  --controllers pets_cem_mpc \
  --steps 4000 \
  --profile spion_carrier_1mm \
  --device-mode moving_adaptive_enclosing_gantry \
  --workers "$WORKERS" \
  2>&1 | tee "$LOG_FILE"

echo "Finished. Results: $RUN_DIR"
echo "Summary CSV: $RUN_DIR/paired_results.csv"
