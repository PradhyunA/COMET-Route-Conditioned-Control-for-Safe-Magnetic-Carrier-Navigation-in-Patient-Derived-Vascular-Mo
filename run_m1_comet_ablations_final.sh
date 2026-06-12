#!/usr/bin/env bash
set -euo pipefail

cd ~/DynaMag-Magnetic-Nanoparticle-Steering-Simulation-3

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m pip install -q matplotlib pandas || true

mkdir -p finaloutputs

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="finaloutputs/m1pro_comet_ablations_500um_${STAMP}"
LOGDIR="${OUT}/logs"
mkdir -p "$LOGDIR"

echo "===== M1 PRO COMET/ABLATIONS FINAL RUN STARTED at $(date) =====" | tee "${LOGDIR}/RUN_STATUS.log"
echo "Output root: $OUT" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Profile: spion_carrier_500um" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Routes: 3 canonical + 2 random" | tee -a "${LOGDIR}/RUN_STATUS.log"
echo "Workers: 6" | tee -a "${LOGDIR}/RUN_STATUS.log"

for item in \
  "cm_corrected:data/real_vessel_segments_cm_corrected.pkl" \
  "40Left85Right:data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl" \
  "pulmonary_0082_H_PULM_H:data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl"
do
  NET="${item%%:*}"
  VESSEL="${item#*:}"
  RUN_ID="m1pro_${NET}_flows1_3_5_comet_ablations"
  RUN_DIR="${OUT}/${RUN_ID}"

  echo "===== STARTING $RUN_ID at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"

  caffeinate -dimsu python -u scripts/run_publication_benchmark.py \
    --output-dir "$OUT" \
    --run-id "$RUN_ID" \
    --vessel-file "$VESSEL" \
    --flow-scales 1,3,5 \
    --n-random-trials 2 \
    --seed 20260611 \
    --controllers comet,comet_no_authority_reserve,comet_no_commitment_memory,comet_no_downstream_value,comet_no_terminal_precision,comet_no_flow_scaled_reserve \
    --steps 2000 \
    --profile spion_carrier_500um \
    --device-mode moving_adaptive_enclosing_gantry \
    --workers 6 \
    2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log"

  echo "===== BENCHMARK $RUN_ID FINISHED at $(date) =====" | tee -a "${LOGDIR}/${RUN_ID}.log" "${LOGDIR}/RUN_STATUS.log"

  caffeinate -dimsu python -u scripts/generate_publication_figures_1mm.py "$RUN_DIR" \
    2>&1 | tee -a "${LOGDIR}/${RUN_ID}.log" || true
done

echo "===== M1 PRO COMET/ABLATIONS FINAL RUN FINISHED at $(date). OUT=$OUT =====" | tee -a "${LOGDIR}/RUN_STATUS.log"
