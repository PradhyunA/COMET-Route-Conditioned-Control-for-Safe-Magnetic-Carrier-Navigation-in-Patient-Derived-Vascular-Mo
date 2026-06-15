#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="comet_abdominal_aorta_ablations_${STAMP}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
  --output-dir outputs/comet_abdominal_aorta_ablations \
  --run-id "$RUN_ID" \
  --vessel-file data/real_vessel_segments_cm_corrected.pkl \
  --flow-scales 1,3,5 \
  --n-random-trials 0 \
  --seed 20260612 \
  --controllers comet,comet_no_authority_reserve,comet_no_commitment_memory,comet_no_downstream_value,comet_no_terminal_precision \
  --steps 2000 \
  --profile spion_carrier_500um \
  --device-mode moving_adaptive_enclosing_gantry \
  --workers 8
