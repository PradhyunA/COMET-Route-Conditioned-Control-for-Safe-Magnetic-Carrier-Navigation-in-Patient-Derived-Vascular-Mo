#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Parallelize whole trials while keeping each numerical worker single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamag_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

CONTROLLER_KEY="constrained_cem_mpc_tuned_recovery"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-20260611}"
STEPS="${STEPS:-2000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/publication_benchmark_500um/${CONTROLLER_KEY}_noearlystop_45case_3anatomy_${STAMP}}"
MANIFEST_ROOT="$ROOT_DIR/benchmark_manifests/icem_45case_500um"

COMMON_PAIR_IDS="canonical_001_inlet_to_outlet,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002"
AORTIC_PAIR_IDS="canonical_001_medium_40Left85Right_endpoint03,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002"
ANATOMIES="${ANATOMIES:-cm_corrected,40Left85Right,pulmonary_0082_H_PULM_H}"

KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -dimsu)
fi

selected_anatomy() {
  local anatomy_id="$1"
  [[ ",$ANATOMIES," == *",$anatomy_id,"* ]]
}

run_anatomy() {
  local anatomy_id="$1"
  local vessel_file="$2"
  local manifest="$3"
  local pair_ids="$4"
  local run_id="${CONTROLLER_KEY}_${anatomy_id}"
  local run_dir="$OUTPUT_ROOT/$run_id"
  local paired_csv="$run_dir/paired_results.csv"

  if [[ -f "$paired_csv" ]]; then
    local completed
    completed="$($PYTHON_BIN -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$paired_csv")"
    if [[ "$completed" == "15" ]]; then
      echo "[$(date '+%F %T')] Skipping completed $anatomy_id block (15/15)"
      return
    fi
  fi

  mkdir -p "$run_dir"
  echo
  echo "[$(date '+%F %T')] Starting $anatomy_id (15 cases)"
  "${KEEP_AWAKE[@]}" "$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
    --output-dir "$OUTPUT_ROOT" \
    --run-id "$run_id" \
    --vessel-file "$vessel_file" \
    --rerun-from-csv "$manifest" \
    --rerun-pair-ids "$pair_ids" \
    --rerun-flow-scales 1,3,5 \
    --controllers "$CONTROLLER_KEY" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --dt 0.02 \
    --tolerance 0.0005 \
    --wall-clearance-threshold-mm 0.7 \
    --profile spion_carrier_500um \
    --device-mode moving_adaptive_enclosing_gantry \
    --workers "$WORKERS" \
    --disable-early-stop \
    2>&1 | tee "$run_dir/run.log"

  "$PYTHON_BIN" -c 'import csv,json,sys; rows=list(csv.DictReader(open(sys.argv[1]))); cfg=json.load(open(sys.argv[2])); assert len(rows)==15, f"expected 15 rows, got {len(rows)}"; assert {r["controller_key"] for r in rows}=={sys.argv[3]}; assert not cfg["early_stopping"]["enabled"], "early stopping was not disabled"; print(f"Validated {len(rows)} completed no-early-stop cases: {sys.argv[1]}")' \
    "$paired_csv" "$run_dir/run_config.json" "$CONTROLLER_KEY"
}

echo "Constrained CEM-MPC tuned-recovery benchmark"
echo "Protocol: 3 anatomies x 3 flow scales x 5 frozen routes = 45 cases"
echo "Carrier: 500 um | dt: 0.02 s | max steps: $STEPS | tolerance: 0.5 mm"
echo "Clearance metric: 0.7 mm | early stopping: DISABLED"
echo "Workers: $WORKERS | seed: $SEED"
echo "Output root: $OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"

if selected_anatomy "cm_corrected"; then
  run_anatomy \
    "cm_corrected" \
    "data/real_vessel_segments_cm_corrected.pkl" \
    "$MANIFEST_ROOT/cm_corrected.csv" \
    "$COMMON_PAIR_IDS"
fi

if selected_anatomy "40Left85Right"; then
  run_anatomy \
    "40Left85Right" \
    "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl" \
    "$MANIFEST_ROOT/40Left85Right.csv" \
    "$AORTIC_PAIR_IDS"
fi

if selected_anatomy "pulmonary_0082_H_PULM_H"; then
  run_anatomy \
    "pulmonary_0082_H_PULM_H" \
    "data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl" \
    "$MANIFEST_ROOT/pulmonary_0082_H_PULM_H.csv" \
    "$COMMON_PAIR_IDS"
fi

TOTAL_CASES="$($PYTHON_BIN -c 'import csv,glob,os,sys; print(sum(sum(1 for _ in csv.DictReader(open(p))) for p in glob.glob(os.path.join(sys.argv[1],"*","paired_results.csv"))))' "$OUTPUT_ROOT")"
if [[ "$TOTAL_CASES" == "45" ]]; then
  echo "[$(date '+%F %T')] Validated full benchmark: 45 cases across 3 anatomies"
else
  echo "[$(date '+%F %T')] Partial benchmark contains $TOTAL_CASES/45 cases"
  echo "Resume by reusing this OUTPUT_ROOT with the same command."
fi
echo "Results: $OUTPUT_ROOT"
