#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# This resumes the interrupted 2026-08-25 run. The abdominal and aortic
# blocks were complete; the pulmonary block is rerun as a clean 15-case block
# and combined with copies of those completed results in a new root directory.
SOURCE_ROOT="${SOURCE_ROOT:-outputs/publication_benchmark_500um/icem_mpc_tuned_shared_noearlystop_45case_3anatomy_20260825_235555}"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-20260611}"
STEPS="${STEPS:-2000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/publication_benchmark_500um/icem_mpc_tuned_shared_noearlystop_45case_3anatomy_resume_${STAMP}}"
MANIFEST_ROOT="$ROOT_DIR/benchmark_manifests/icem_45case_500um"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamag_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -dimsu)
fi

SOURCE_CM="$SOURCE_ROOT/icem_mpc_tuned_shared_cm_corrected"
SOURCE_AORTIC="$SOURCE_ROOT/icem_mpc_tuned_shared_40Left85Right"
if [[ ! -f "$SOURCE_CM/paired_results.csv" || ! -f "$SOURCE_AORTIC/paired_results.csv" ]]; then
  echo "Completed source blocks not found under $SOURCE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cp -R "$SOURCE_CM" "$OUTPUT_ROOT/"
cp -R "$SOURCE_AORTIC" "$OUTPUT_ROOT/"

echo "Resuming iCEM shared-tuned benchmark"
echo "Preserved: abdominal 15/15 and aortic-root 15/15"
echo "Rerunning: pulmonary 15/15"
echo "Output root: $OUTPUT_ROOT"

PULMONARY_RUN_ID="icem_mpc_tuned_shared_pulmonary_0082_H_PULM_H"
PULMONARY_DIR="$OUTPUT_ROOT/$PULMONARY_RUN_ID"
mkdir -p "$PULMONARY_DIR"

"${KEEP_AWAKE[@]}" "$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
  --output-dir "$OUTPUT_ROOT" \
  --run-id "$PULMONARY_RUN_ID" \
  --vessel-file data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl \
  --rerun-from-csv "$MANIFEST_ROOT/pulmonary_0082_H_PULM_H.csv" \
  --rerun-pair-ids canonical_001_inlet_to_outlet,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002 \
  --rerun-flow-scales 1,3,5 \
  --controllers icem_mpc_tuned_shared \
  --seed "$SEED" \
  --steps "$STEPS" \
  --tolerance 0.0005 \
  --wall-clearance-threshold-mm 0.7 \
  --profile spion_carrier_500um \
  --device-mode moving_adaptive_enclosing_gantry \
  --workers "$WORKERS" \
  --disable-early-stop \
  2>&1 | tee "$PULMONARY_DIR/run.log"

"$PYTHON_BIN" -c 'import csv,glob,os,sys; root=sys.argv[1]; expected={"icem_mpc_tuned_shared_cm_corrected":15,"icem_mpc_tuned_shared_40Left85Right":15,"icem_mpc_tuned_shared_pulmonary_0082_H_PULM_H":15}; paths=glob.glob(os.path.join(root,"*","paired_results.csv")); got={os.path.basename(os.path.dirname(p)):sum(1 for _ in csv.DictReader(open(p))) for p in paths}; assert got==expected, f"expected {expected}, got {got}"; print(f"Validated consolidated benchmark: {sum(got.values())} cases across 3 anatomies")' "$OUTPUT_ROOT"

echo "Finished. Consolidated results: $OUTPUT_ROOT"
