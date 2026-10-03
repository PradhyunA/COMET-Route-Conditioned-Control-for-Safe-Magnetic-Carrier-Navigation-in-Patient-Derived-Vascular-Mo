#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

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

CONTROLLER_KEY="smc_dob_inspired"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-20260611}"
STEPS="${STEPS:-2000}"
SMOKE_STEPS="${SMOKE_STEPS:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/publication_benchmark_500um/smc_dob_inspired_noearlystop_45case_3anatomy_${STAMP}}"
SMOKE_ROOT="${SMOKE_ROOT:-outputs/smoke_tests/smc_dob_inspired_${STAMP}}"
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

run_block() {
  local output_root="$1"
  local anatomy_id="$2"
  local vessel_file="$3"
  local manifest="$4"
  local pair_ids="$5"
  local flow_scales="$6"
  local steps="$7"
  local workers="$8"
  local run_prefix="$9"
  local run_id="${run_prefix}_${anatomy_id}"
  local run_dir="$output_root/$run_id"
  mkdir -p "$run_dir"

  "${KEEP_AWAKE[@]}" "$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
    --output-dir "$output_root" \
    --run-id "$run_id" \
    --vessel-file "$vessel_file" \
    --rerun-from-csv "$manifest" \
    --rerun-pair-ids "$pair_ids" \
    --rerun-flow-scales "$flow_scales" \
    --controllers "$CONTROLLER_KEY" \
    --seed "$SEED" \
    --steps "$steps" \
    --dt 0.02 \
    --tolerance 0.0005 \
    --wall-clearance-threshold-mm 0.7 \
    --profile spion_carrier_500um \
    --device-mode moving_adaptive_enclosing_gantry \
    --workers "$workers" \
    --disable-early-stop \
    2>&1 | tee "$run_dir/run.log"
}

run_smoke() {
  echo "Running SMC-DOB-inspired unit tests"
  "$PYTHON_BIN" -m pytest \
    tests/test_smc_dob_controller.py \
    tests/test_pid_controller.py \
    tests/test_publication_benchmark_metrics.py \
    -q

  mkdir -p "$SMOKE_ROOT"
  echo "Running three-anatomy physics smoke tests ($SMOKE_STEPS steps each)"
  run_block \
    "$SMOKE_ROOT" \
    "cm_corrected" \
    "data/real_vessel_segments_cm_corrected.pkl" \
    "$MANIFEST_ROOT/cm_corrected.csv" \
    "canonical_001_inlet_to_outlet" \
    "1" \
    "$SMOKE_STEPS" \
    "1" \
    "smoke_${CONTROLLER_KEY}"
  run_block \
    "$SMOKE_ROOT" \
    "40Left85Right" \
    "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl" \
    "$MANIFEST_ROOT/40Left85Right.csv" \
    "canonical_001_medium_40Left85Right_endpoint03" \
    "1" \
    "$SMOKE_STEPS" \
    "1" \
    "smoke_${CONTROLLER_KEY}"
  run_block \
    "$SMOKE_ROOT" \
    "pulmonary_0082_H_PULM_H" \
    "data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl" \
    "$MANIFEST_ROOT/pulmonary_0082_H_PULM_H.csv" \
    "canonical_001_inlet_to_outlet" \
    "1" \
    "$SMOKE_STEPS" \
    "1" \
    "smoke_${CONTROLLER_KEY}"

  "$PYTHON_BIN" -c 'import csv,glob,os,sys; paths=glob.glob(os.path.join(sys.argv[1],"*","paired_results.csv")); rows=[r for p in paths for r in csv.DictReader(open(p))]; assert len(paths)==3 and len(rows)==3, f"expected 3 smoke cases, found {len(paths)} files/{len(rows)} rows"; assert {r["controller_key"] for r in rows}=={"smc_dob_inspired"}; print("Validated 3/3 SMC-DOB-inspired anatomy smoke cases")' "$SMOKE_ROOT"
}

run_full_anatomy() {
  local anatomy_id="$1"
  local vessel_file="$2"
  local manifest="$3"
  local pair_ids="$4"
  local run_id="${CONTROLLER_KEY}_${anatomy_id}"
  local paired_csv="$OUTPUT_ROOT/$run_id/paired_results.csv"

  if [[ -f "$paired_csv" ]]; then
    local completed
    completed="$($PYTHON_BIN -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$paired_csv")"
    if [[ "$completed" == "15" ]]; then
      echo "[$(date '+%F %T')] Skipping completed $anatomy_id block (15/15)"
      return
    fi
  fi

  echo "[$(date '+%F %T')] Starting $anatomy_id (15 cases)"
  run_block \
    "$OUTPUT_ROOT" \
    "$anatomy_id" \
    "$vessel_file" \
    "$manifest" \
    "$pair_ids" \
    "1,3,5" \
    "$STEPS" \
    "$WORKERS" \
    "$CONTROLLER_KEY"

  "$PYTHON_BIN" -c 'import csv,json,sys; rows=list(csv.DictReader(open(sys.argv[1]))); cfg=json.load(open(sys.argv[2])); assert len(rows)==15, f"expected 15 rows, got {len(rows)}"; assert {r["controller_key"] for r in rows}=={"smc_dob_inspired"}; assert not cfg["early_stopping"]["enabled"]; variant=cfg["variants"]["smc_dob_inspired"]; assert variant["kind"]=="smc_dob_inspired"; assert "not an exact reproduction" in variant["label"]; print(f"Validated {len(rows)} SMC-DOB-inspired cases")' "$paired_csv" "$OUTPUT_ROOT/$run_id/run_config.json"
}

echo "SMC-DOB-inspired published-style baseline"
echo "Adapted from Bhattacharjee et al. (2026); NOT an exact reproduction"
echo "Protocol: 3 anatomies x 3 flow scales x 5 frozen Dijkstra routes = 45 cases"
echo "Shared model: 500 um carrier | dt 0.02 s | max steps $STEPS | tolerance 0.5 mm"
echo "Allocation: non-CEM bounded SLSQP current allocator"
echo "COMET branch probability/downstream value/terminal precision: disabled by design"
echo "Workers: $WORKERS | seed: $SEED"
echo "Output root: $OUTPUT_ROOT"

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  run_smoke
fi
if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  echo "Smoke-only mode complete. Results: $SMOKE_ROOT"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
if selected_anatomy "cm_corrected"; then
  run_full_anatomy \
    "cm_corrected" \
    "data/real_vessel_segments_cm_corrected.pkl" \
    "$MANIFEST_ROOT/cm_corrected.csv" \
    "$COMMON_PAIR_IDS"
fi
if selected_anatomy "40Left85Right"; then
  run_full_anatomy \
    "40Left85Right" \
    "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl" \
    "$MANIFEST_ROOT/40Left85Right.csv" \
    "$AORTIC_PAIR_IDS"
fi
if selected_anatomy "pulmonary_0082_H_PULM_H"; then
  run_full_anatomy \
    "pulmonary_0082_H_PULM_H" \
    "data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl" \
    "$MANIFEST_ROOT/pulmonary_0082_H_PULM_H.csv" \
    "$COMMON_PAIR_IDS"
fi

TOTAL_CASES="$($PYTHON_BIN -c 'import csv,glob,os,sys; print(sum(sum(1 for _ in csv.DictReader(open(p))) for p in glob.glob(os.path.join(sys.argv[1],"*","paired_results.csv"))))' "$OUTPUT_ROOT")"
if [[ "$TOTAL_CASES" == "45" ]]; then
  "$PYTHON_BIN" scripts/summarize_smc_dob_benchmark.py "$OUTPUT_ROOT"
  echo "[$(date '+%F %T')] Finished all 45 SMC-DOB-inspired cases"
else
  echo "[$(date '+%F %T')] Partial benchmark contains $TOTAL_CASES/45 cases"
  echo "Resume with OUTPUT_ROOT='$OUTPUT_ROOT' and the same command."
fi
echo "Results: $OUTPUT_ROOT"
