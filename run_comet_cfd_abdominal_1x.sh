#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamag_cfd_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-cfd/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "CFD environment missing. Create it using the commands in docs/cfd_validation.md."
  exit 1
fi

CFD_FILE="${CFD_FILE:-data/cfd/abdominal_aorta_steady.npz}"
WORKERS="${WORKERS:-4}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-20260611}"
RUN_ID="${RUN_ID:-comet_cfd_abdominal_1x_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/comet_cfd_abdominal_1x}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

if [[ ! -f "$CFD_FILE" ]]; then
  echo "Missing CFD field: $CFD_FILE. Import the Run 4 ZIP using docs/cfd_validation.md."
  exit 1
fi
if [[ -e "$RUN_DIR/run_config.json" || -e "$RUN_DIR/paired_results.csv" ]]; then
  echo "Run already exists: $RUN_DIR. Choose a new RUN_ID to preserve its results."
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "COMET abdominal CFD: 5 frozen routes at the supplied 1x CFD velocity = 5 cases"
echo "500 um carrier | dt 0.02 s | $STEPS steps | 0.5 mm target tolerance | no stagnation early stop"
echo "Uses the supplied steady CFD field with velocity multiplier 1.0."
echo "Results: $RUN_DIR"

"$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
  --output-dir "$OUTPUT_ROOT" --run-id "$RUN_ID" \
  --vessel-file data/real_vessel_segments_cm_corrected.pkl \
  --cfd-file "$CFD_FILE" \
  --rerun-from-csv benchmark_manifests/icem_45case_500um/cm_corrected.csv \
  --rerun-pair-ids canonical_001_inlet_to_outlet,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002 \
  --rerun-flow-scales 1 --n-random-trials 0 --controllers comet \
  --seed "$SEED" --steps "$STEPS" --dt 0.02 --tolerance 0.0005 \
  --wall-clearance-threshold-mm 0.7 --profile spion_carrier_500um \
  --device-mode moving_adaptive_enclosing_gantry --workers "$WORKERS" \
  --disable-early-stop 2>&1 | tee "$RUN_DIR/run.log"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = list(csv.DictReader((root / 'paired_results.csv').open()))
config = json.loads((root / 'run_config.json').read_text())
assert len(rows) == 5, f'Expected 5 cases, found {len(rows)}'
assert len({(float(row['flow_scale']), row['pair_id']) for row in rows}) == 5
assert {row['controller_key'] for row in rows} == {'comet'}
assert {float(row['flow_scale']) for row in rows} == {1.}
assert config['flow_model'] == 'steady_cfd'
assert not config['early_stopping']['enabled']
assert not config['route_authority_filter_enabled']
provenance_files = list(root.glob('flow_1x/comet/*_cfd_flow.json'))
assert len(provenance_files) == 5, 'Missing per-trial CFD provenance'
for path in provenance_files:
    provenance = json.loads(path.read_text())
    assert provenance['source_zip_sha256'] == config['cfd_metadata']['source_sha256']
    assert provenance['velocity_multiplier'] == 1.0
    assert provenance['queries'] > 0, f'No CFD queries recorded: {path}'
print(f'Completed all 5 abdominal CFD cases at 1x: {root / "paired_results.csv"}')
PY
