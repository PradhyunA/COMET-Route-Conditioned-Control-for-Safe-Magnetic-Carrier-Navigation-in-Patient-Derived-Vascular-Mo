#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m pip install -q matplotlib pandas || true
mkdir -p finaloutputs
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="finaloutputs/comet_no_flow_compensation_ablation_${STAMP}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR"
echo "===== COMET NO-FLOW-COMPENSATION ABLATION RUN STARTED at $(date) =====" | tee "${LOGDIR}/RUN_STATUS.log"
echo "Output root: $OUT" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Profile: spion_carrier_500um" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Requested by: Reviewer 1 (Task 2 - set -k_f*v_flow term in Eq.11 to zero)" | tee -a "${LOGDIR}/RUN_STATUS.log"

for item in \
  "cm_corrected:data/real_vessel_segments_cm_corrected.pkl" \
  "40Left85Right:data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl" \
  "pulmonary_0082_H_PULM_H:data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl"
do
  NET="${item%%:*}"
  VESSEL="${item#*:}"
  RUN_ID="comet_no_flow_compensation_${NET}_flows1_3_5"
  RUN_DIR="${OUT}/${RUN_ID}"
  echo "===== STARTING $RUN_ID at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"
  python -u scripts/run_publication_benchmark.py \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --vessel-file "$VESSEL" \
    --flow-scales 1,3,5 \
    --n-random-trials 2 \
    --seed 20260611 \
    --controllers comet,comet_no_flow_compensation \
    --steps 2000 \
    --profile spion_carrier_500um \
    --device-mode moving_adaptive_enclosing_gantry \
    --workers 6 \
    2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log"
  echo "===== BENCHMARK $RUN_ID FINISHED at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"
  python -u scripts/generate_publication_figures_1mm.py "$RUN_DIR" \
    2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log" || true
done

echo "===== COMET NO-FLOW-COMPENSATION ABLATION RUN FINISHED at $(date). OUT=$OUT =====" | tee -a "${LOGDIR}/RUN_STATUS.log"
