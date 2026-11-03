#!/usr/bin/env bash
# Physics sensitivity sweep: far-field core gain x gravity on the frozen
# 45-case manifest (3 anatomies x 5 routes x 1/3/5 flow).
#
# Conditions are "name:mu_r:density:gravity_dir". mu_r "preset" keeps the
# actuator preset's core gain, density "none" disables gravity and
# gravity_dir "optimized" picks a table orientation per route. The paper
# used mu_r=70 with no gravity.
#
# Overrides: ACTUATOR (paper | navion_calibrated), CONTROLLERS, CONDITIONS,
# FLOWS, WORKERS, PARALLEL, STEPS, RUN_ID,
# OUTPUT_ROOT, PYTHON_BIN, EARLY_STOP_FLAG (set to --disable-early-stop to
# turn stagnation early stopping off).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dynamag_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
ACTUATOR="${ACTUATOR:-paper}"
CONTROLLERS="${CONTROLLERS:-comet}"
CONDITIONS="${CONDITIONS:-mu70_nog:70:none:0,-1,0 mu70_gneg_y:70:3542:0,-1,0 mu70_gpos_y:70:3542:0,1,0 mu7_nog:7:none:0,-1,0 mu7_gneg_y:7:3542:0,-1,0 mu2_nog:2:none:0,-1,0 mu2_gneg_y:2:3542:0,-1,0}"
FLOWS="${FLOWS:-1,3,5}"
WORKERS="${WORKERS:-5}"
PARALLEL="${PARALLEL:-2}"
STEPS="${STEPS:-2000}"
EARLY_STOP_FLAG="${EARLY_STOP_FLAG:-}"
RUN_ID="${RUN_ID:-physics_sensitivity_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/physics_sensitivity}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run already exists: $RUN_DIR. Choose a new RUN_ID."
  exit 1
fi
mkdir -p "$RUN_DIR"

ROUTES=canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002
ANATOMIES=(
  "abdominal|data/real_vessel_segments_cm_corrected.pkl|benchmark_manifests/icem_45case_500um/cm_corrected.csv|canonical_001_inlet_to_outlet"
  "aortic_root|data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl|benchmark_manifests/icem_45case_500um/40Left85Right.csv|canonical_001_medium_40Left85Right_endpoint03"
  "pulmonary|data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl|benchmark_manifests/icem_45case_500um/pulmonary_0082_H_PULM_H.csv|canonical_001_inlet_to_outlet"
)

JOBS="$RUN_DIR/jobs.txt"
: > "$JOBS"
for cond in $CONDITIONS; do
  IFS=: read -r name mu density gdir <<< "$cond"
  for anatomy in "${ANATOMIES[@]}"; do
    IFS='|' read -r label vessel manifest first_route <<< "$anatomy"
    echo "$name $mu $density $gdir $label $vessel $manifest $first_route,$ROUTES" >> "$JOBS"
  done
done

run_job() {
  local name=$1 mu=$2 density=$3 gdir=$4 label=$5 vessel=$6 manifest=$7 routes=$8
  local physics_args=()
  if [[ "$mu" != "preset" ]]; then
    physics_args+=(--core-mu-r "$mu")
  fi
  if [[ "$density" != "none" ]]; then
    physics_args+=(--carrier-density-kg-m3 "$density")
    if [[ "$gdir" == "optimized" ]]; then
      physics_args+=(--patient-positioning optimized)
    else
      physics_args+=(--gravity-direction "$gdir")
    fi
  fi
  mkdir -p "$RUN_DIR/$name"
  "$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
    --output-dir "$RUN_DIR/$name" --run-id "$label" \
    --vessel-file "$vessel" \
    --rerun-from-csv "$manifest" --rerun-pair-ids "$routes" \
    --rerun-flow-scales "$FLOWS" --n-random-trials 0 \
    --controllers "$CONTROLLERS" --seed 20260611 --steps "$STEPS" --dt 0.02 \
    --tolerance 0.0005 --wall-clearance-threshold-mm 0.7 \
    --profile spion_carrier_500um --device-mode moving_adaptive_enclosing_gantry \
    --workers "$WORKERS" --actuator "$ACTUATOR" ${physics_args[@]+"${physics_args[@]}"} $EARLY_STOP_FLAG \
    > "$RUN_DIR/$name/$label.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $name $label"
}
export -f run_job
export RUN_DIR PYTHON_BIN ACTUATOR CONTROLLERS FLOWS WORKERS STEPS EARLY_STOP_FLAG

echo "Sweep: $RUN_DIR"
echo "Actuator: $ACTUATOR | controllers: $CONTROLLERS | flows: $FLOWS | steps: $STEPS | early stop flag: ${EARLY_STOP_FLAG:-<default>}"
xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _ < "$JOBS"
echo "Sweep finished: $RUN_DIR"
