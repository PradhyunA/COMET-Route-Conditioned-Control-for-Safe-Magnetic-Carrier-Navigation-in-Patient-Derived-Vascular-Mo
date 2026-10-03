#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Keep numerical kernels single-threaded; the runner parallelizes whole trials.
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

# OBJECTIVE=tuned is the recommended definitive comparison. Use
# OBJECTIVE=default only to repeat the completed default-objective baseline
# without its former stagnation stopping rule.
OBJECTIVE="${OBJECTIVE:-tuned}"
case "$OBJECTIVE" in
  tuned)
    CONTROLLER_KEY="icem_mpc_tuned_shared"
    OBJECTIVE_LABEL="shared tuned MPC objective"
    ;;
  default)
    CONTROLLER_KEY="icem_mpc_default_shared"
    OBJECTIVE_LABEL="default shared MPC objective"
    ;;
  *)
    echo "OBJECTIVE must be 'tuned' or 'default'; got '$OBJECTIVE'" >&2
    exit 2
    ;;
esac

WORKERS="${WORKERS:-6}"
SEED="${SEED:-20260611}"
STEPS="${STEPS:-2000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/publication_benchmark_500um/${CONTROLLER_KEY}_noearlystop_45case_3anatomy_${STAMP}}"
MANIFEST_ROOT="$ROOT_DIR/benchmark_manifests/icem_45case_500um"

COMMON_PAIR_IDS="canonical_001_inlet_to_outlet,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002"
AORTIC_PAIR_IDS="canonical_001_medium_40Left85Right_endpoint03,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002"

KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -dimsu)
fi

mkdir -p "$OUTPUT_ROOT"

run_anatomy() {
  local anatomy_id="$1"
  local vessel_file="$2"
  local manifest="$3"
  local pair_ids="$4"
  local run_id="${CONTROLLER_KEY}_${anatomy_id}"
  local run_dir="$OUTPUT_ROOT/$run_id"
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
    --tolerance 0.0005 \
    --wall-clearance-threshold-mm 0.7 \
    --profile spion_carrier_500um \
    --device-mode moving_adaptive_enclosing_gantry \
    --workers "$WORKERS" \
    --disable-early-stop \
    2>&1 | tee "$run_dir/run.log"

  "$PYTHON_BIN" -c 'import csv,json,sys; rows=list(csv.DictReader(open(sys.argv[1]))); cfg=json.load(open(sys.argv[2])); assert len(rows)==15, f"expected 15 rows, got {len(rows)}"; assert not cfg["early_stopping"]["enabled"], "early stopping was not disabled"; assert {r["controller_key"] for r in rows}=={sys.argv[3]}; print(f"Validated {len(rows)} completed no-early-stop cases: {sys.argv[1]}")' "$run_dir/paired_results.csv" "$run_dir/run_config.json" "$CONTROLLER_KEY"
}

echo "iCEM-MPC published-baseline benchmark"
echo "Variant: $CONTROLLER_KEY ($OBJECTIVE_LABEL)"
echo "Protocol: 3 anatomies x 3 flow scales x 5 frozen routes = 45 cases"
echo "Carrier: 500 um | dt: 0.02 s | max steps: $STEPS | tolerance: 0.5 mm"
echo "iCEM blitz: H=30, populations=40/32/25, elites=10, iterations=3, beta=3"
echo "Stagnation early stopping: DISABLED"
echo "Workers: $WORKERS | seed: $SEED"
echo "Output root: $OUTPUT_ROOT"

run_anatomy \
  "cm_corrected" \
  "data/real_vessel_segments_cm_corrected.pkl" \
  "$MANIFEST_ROOT/cm_corrected.csv" \
  "$COMMON_PAIR_IDS"

run_anatomy \
  "40Left85Right" \
  "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl" \
  "$MANIFEST_ROOT/40Left85Right.csv" \
  "$AORTIC_PAIR_IDS"

run_anatomy \
  "pulmonary_0082_H_PULM_H" \
  "data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl" \
  "$MANIFEST_ROOT/pulmonary_0082_H_PULM_H.csv" \
  "$COMMON_PAIR_IDS"

"$PYTHON_BIN" -c 'import csv,glob,os,sys; paths=glob.glob(os.path.join(sys.argv[1],"*","paired_results.csv")); count=sum(len(list(csv.DictReader(open(p)))) for p in paths); assert len(paths)==3 and count==45, f"expected 3 summaries/45 rows, got {len(paths)}/{count}"; print(f"Validated full benchmark: {count} cases across {len(paths)} anatomies")' "$OUTPUT_ROOT"

echo "[$(date '+%F %T')] Finished all 45 iCEM-MPC cases"
echo "Results: $OUTPUT_ROOT"
