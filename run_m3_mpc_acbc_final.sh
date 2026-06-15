#!/usr/bin/env bash
set -euo pipefail

# Activate normal Python venv
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m pip install -q matplotlib pandas || true

mkdir -p finaloutputs

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="finaloutputs/m3pro_mpc_acbc_remaining_500um_${STAMP}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR"

echo "===== M3 PRO MPC/ACBC FINAL RUN STARTED at $(date) =====" | tee "${LOGDIR}/RUN_STATUS.log"
echo "Output root: $OUT" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Profile: spion_carrier_500um" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Routes: 3 canonical + 2 random" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Workers: 8" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Python: $(which python)" | tee -a "${LOGDIR}/RUN_STATUS.log"

NET="40Left85Right"
VESSEL="data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl"
RUN_ID="m3pro_${NET}_flow5_mpc_acbc"
RUN_DIR="${OUT}/${RUN_ID}"

echo "===== STARTING $RUN_ID at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"

caffeinate -dimsu python -u scripts/run_publication_benchmark.py \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --vessel-file "$VESSEL" \
  --flow-scales 5 \
  --n-random-trials 2 \
  --seed 20260611 \
  --controllers mpc_tuned,acbc_tuned \
  --steps 2000 \
  --profile spion_carrier_500um \
  --device-mode moving_adaptive_enclosing_gantry \
  --workers 8 \
  2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log"

echo "===== BENCHMARK $RUN_ID FINISHED at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"

caffeinate -dimsu python -u scripts/generate_publication_figures_1mm.py "$RUN_DIR" \
  2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log" || true


NET="pulmonary_0082_H_PULM_H"
VESSEL="data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl"
RUN_ID="m3pro_${NET}_flows1_3_5_mpc_acbc"
RUN_DIR="${OUT}/${RUN_ID}"

echo "===== STARTING $RUN_ID at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"

caffeinate -dimsu python -u scripts/run_publication_benchmark.py \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --vessel-file "$VESSEL" \
  --flow-scales 1,3,5 \
  --n-random-trials 2 \
  --seed 20260611 \
  --controllers mpc_tuned,acbc_tuned \
  --steps 2000 \
  --profile spion_carrier_500um \
  --device-mode moving_adaptive_enclosing_gantry \
  --workers 8 \
  2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log"

echo "===== BENCHMARK $RUN_ID FINISHED at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"

caffeinate -dimsu python -u scripts/generate_publication_figures_1mm.py "$RUN_DIR" \
  2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log" || true

echo "===== M3 PRO MPC/ACBC FINAL RUN FINISHED at $(date). OUT=$OUT =====" | tee -a "${LOGDIR}/RUN_STATUS.log"
