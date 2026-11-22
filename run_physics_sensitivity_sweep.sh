#!/usr/bin/env bash
# Benchmark runner for the COMET paper: runs the frozen five-anatomy,
# 75-case manifest (5 routes x 3 flow scales per anatomy) for each
# controller and condition, one process per anatomy.
#
# The defaults reproduce the main benchmark (400 mT/m array, gravity,
# optimized patient positioning, shared-route protocol) for COMET:
#   ./run_physics_sensitivity_sweep.sh
# Baselines:      CONTROLLERS=nmpc_tuned,nmpc_default,mpc_tuned,acbc_tuned
# 300/600 mT/m:   ACTUATOR=high_gradient_300 CONDITIONS=grad300_gopt:preset:3542:optimized
# Replicates:     SEED=101 START_JITTER_MM=0.25
# Steady CFD:     CFD=1 PYTHON_BIN=.venv-cfd/bin/python (needs data/cfd/*.npz)
#
# Conditions are "name:mu_r:density:gravity_dir". mu_r "preset" keeps the
# actuator preset's core gain, density "none" disables gravity and
# gravity_dir "optimized" picks a table orientation per route.
#
# Overrides: ACTUATOR, PROTOCOL (paper | shared_route), CONTROLLERS, CONDITIONS,
# FLOWS, WORKERS, PARALLEL, STEPS, RUN_ID, SEED, START_JITTER_MM, OUTPUT_ROOT,
# PYTHON_BIN, EARLY_STOP_FLAG (--disable-early-stop), CFD, ONLY_ANATOMY.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dynamag_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
ACTUATOR="${ACTUATOR:-high_gradient_400}"
CONTROLLERS="${CONTROLLERS:-comet_v4}"
CONDITIONS="${CONDITIONS:-grad400_gopt:preset:3542:optimized}"
FLOWS="${FLOWS:-1,3,5}"
WORKERS="${WORKERS:-5}"
PARALLEL="${PARALLEL:-2}"
STEPS="${STEPS:-2000}"
EARLY_STOP_FLAG="${EARLY_STOP_FLAG:-}"
PROTOCOL="${PROTOCOL:-shared_route}"
SEED="${SEED:-20260611}"
START_JITTER_MM="${START_JITTER_MM:-0}"
CFD="${CFD:-0}"
RUN_ID="${RUN_ID:-physics_sensitivity_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/physics_sensitivity}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run already exists: $RUN_DIR. Choose a new RUN_ID."
  exit 1
fi
mkdir -p "$RUN_DIR"

ROUTES=canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002
# The five-anatomy, 75-case benchmark manifest used in the paper.
# ONLY_ANATOMY limits a run to one label.
MANIFESTS=benchmark_manifests/five_anatomy_500um
ANATOMIES=(
  "abdominal|data/vessels/abdominal_0033_H_ABAO_AAA/real_vessel_segments.pkl|cm_corrected.csv|canonical_001_inlet_to_outlet"
  "aortic_root|data/vessels/aortic_root_0248_H_AOCERE_CAS_40Left85Right/real_vessel_segments.pkl|40Left85Right.csv|canonical_001_medium_40Left85Right_endpoint03"
  "pulmonary|data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments.pkl|pulmonary_0082_H_PULM_H.csv|canonical_001_inlet_to_outlet"
)
ANATOMIES+=(
  "thoracic_aorta|data/vessels/aorta_0024_H_AO_H/real_vessel_segments.pkl|0024_H_AO_H.csv|canonical_001_root_to_rcca_0024"
  "aortofemoral|data/vessels/aortofemoral_0049_H_ABAO_AIOD/real_vessel_segments.pkl|0049_H_ABAO_AIOD.csv|canonical_001_aorta_to_femoral_0049"
)

JOBS="$RUN_DIR/jobs.txt"
: > "$JOBS"
for cond in $CONDITIONS; do
  IFS=: read -r name mu density gdir <<< "$cond"
  for anatomy in "${ANATOMIES[@]}"; do
    IFS='|' read -r label vessel manifest first_route <<< "$anatomy"
    [[ -n "${ONLY_ANATOMY:-}" && "$label" != "$ONLY_ANATOMY" ]] && continue
    cfd_file=none
    if [[ "$CFD" == 1 ]]; then
      case "$label" in
        abdominal) cfd_file=data/cfd/abdominal_aorta_steady.npz ;;
        pulmonary) cfd_file=data/cfd/pulmonary_steady.npz ;;
        *) continue ;;
      esac
    fi
    echo "$name $mu $density $gdir $label $vessel $MANIFESTS/$manifest $first_route,$ROUTES $cfd_file" >> "$JOBS"
  done
done

run_job() {
  local name=$1 mu=$2 density=$3 gdir=$4 label=$5 vessel=$6 manifest=$7 routes=$8 cfd_file=$9
  local physics_args=()
  if [[ "$cfd_file" != "none" ]]; then
    physics_args+=(--cfd-file "$cfd_file")
  fi
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
    --controllers "$CONTROLLERS" --seed "$SEED" --start-jitter-mm "$START_JITTER_MM" --steps "$STEPS" --dt 0.02 \
    --tolerance 0.0005 --wall-clearance-threshold-mm 0.7 \
    --profile spion_carrier_500um --device-mode moving_adaptive_enclosing_gantry \
    --workers "$WORKERS" --actuator "$ACTUATOR" --protocol "$PROTOCOL" ${physics_args[@]+"${physics_args[@]}"} $EARLY_STOP_FLAG \
    > "$RUN_DIR/$name/$label.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $name $label"
}
export -f run_job
export RUN_DIR PYTHON_BIN ACTUATOR PROTOCOL CONTROLLERS FLOWS WORKERS STEPS EARLY_STOP_FLAG SEED START_JITTER_MM

echo "Sweep: $RUN_DIR"
echo "Actuator: $ACTUATOR | protocol: $PROTOCOL | controllers: $CONTROLLERS | flows: $FLOWS | steps: $STEPS | early stop flag: ${EARLY_STOP_FLAG:-<default>}"
xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _ < "$JOBS"
echo "Sweep finished: $RUN_DIR"
