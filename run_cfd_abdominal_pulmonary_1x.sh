#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamag_cfd_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-cfd/bin/python}"
WORKERS="${WORKERS:-4}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-20260611}"
RUN_ID="${RUN_ID:-cfd_abdominal_pulmonary_1x_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cfd_1x_four_controllers}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
CONTROLLERS="comet,constrained_cem_mpc_tuned_recovery,acbc_tuned,mpc_tuned"
PAIR_IDS="canonical_001_inlet_to_outlet,canonical_002_random_to_publication_target,canonical_003_publication_start_to_random,random_0001,random_0002"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "CFD environment missing. See docs/cfd_validation.md."
  exit 1
fi
for field in data/cfd/abdominal_aorta_steady.npz data/cfd/pulmonary_steady.npz; do
  if [[ ! -f "$field" ]]; then
    echo "Missing CFD field: $field. See docs/cfd_validation.md."
    exit 1
  fi
done
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR. Choose a fresh RUN_ID."
  exit 1
fi
mkdir -p "$RUN_DIR"
echo "1x steady CFD: abdominal + pulmonary, 5 frozen routes each, 4 controllers = 40 cases"
echo "Controllers: $CONTROLLERS"
echo "500 um carrier | dt 0.02 s | $STEPS steps | 0.5 mm tolerance | no stagnation early stop"
echo "Output: $RUN_DIR"

for anatomy in abdominal pulmonary; do
  if [[ "$anatomy" == abdominal ]]; then
    vessel_file=data/real_vessel_segments_cm_corrected.pkl
    cfd_file=data/cfd/abdominal_aorta_steady.npz
    manifest=benchmark_manifests/icem_45case_500um/cm_corrected.csv
  else
    vessel_file=data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl
    cfd_file=data/cfd/pulmonary_steady.npz
    manifest=benchmark_manifests/icem_45case_500um/pulmonary_0082_H_PULM_H.csv
  fi
  mkdir -p "$RUN_DIR/$anatomy"
  "$PYTHON_BIN" -u scripts/run_publication_benchmark.py \
    --output-dir "$RUN_DIR" --run-id "$anatomy" \
    --vessel-file "$vessel_file" --cfd-file "$cfd_file" \
    --rerun-from-csv "$manifest" --rerun-pair-ids "$PAIR_IDS" \
    --rerun-flow-scales 1 --n-random-trials 0 --controllers "$CONTROLLERS" \
    --seed "$SEED" --steps "$STEPS" --dt 0.02 --tolerance 0.0005 \
    --wall-clearance-threshold-mm 0.7 --profile spion_carrier_500um \
    --device-mode moving_adaptive_enclosing_gantry --workers "$WORKERS" \
    --disable-early-stop 2>&1 | tee "$RUN_DIR/$anatomy/run.log"
done

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import csv, json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
controllers = {'comet', 'constrained_cem_mpc_tuned_recovery', 'acbc_tuned', 'mpc_tuned'}
pairs = {'canonical_001_inlet_to_outlet', 'canonical_002_random_to_publication_target',
         'canonical_003_publication_start_to_random', 'random_0001', 'random_0002'}
combined = []
summary = {'run_directory': str(root.resolve()), 'flow_scale': 1.0,
           'expected_cases': 40, 'anatomies': {}}
for anatomy in ['abdominal', 'pulmonary']:
    directory = root / anatomy
    config = json.loads((directory / 'run_config.json').read_text())
    rows = list(csv.DictReader((directory / 'paired_results.csv').open()))
    assert len(rows) == 20, f'{anatomy}: expected 20 cases, found {len(rows)}'
    assert {(r['pair_id'], r['controller_key']) for r in rows} == {
        (p, c) for p in pairs for c in controllers}
    assert all(float(r['flow_scale']) == 1.0 for r in rows)
    assert config['flow_model'] == 'steady_cfd'
    assert not config['early_stopping']['enabled']
    manifest = {r['pair_id']: r for r in csv.DictReader(Path(config['rerun_from_csv']).open())
                if float(r['flow_scale']) == 1.0}
    for row in rows:
        trial = directory / 'flow_1x' / row['controller_key'] / row['pair_id']
        provenance = json.loads(Path(str(trial) + '_cfd_flow.json').read_text())
        assert provenance['source_zip_sha256'] == config['cfd_metadata']['source_sha256']
        assert provenance['velocity_multiplier'] == 1.0 and provenance['queries'] > 0
        with np.load(Path(str(trial) + '_data.npz'), allow_pickle=False) as data:
            assert len(data['trajectory']) == int(row['steps_taken']) + 1
            assert np.all(np.isfinite(data['trajectory']))
            for side in ['start', 'target']:
                expected = [float(row[f'{side}_{axis}_m']) for axis in 'xyz']
                np.testing.assert_allclose(data[side], expected, rtol=0, atol=1e-12)
        for side in ['start', 'target']:
            for axis in 'xyz':
                key = f'{side}_{axis}_m'
                assert abs(float(row[key]) - float(manifest[row['pair_id']][key])) < 1e-12
        combined.append({'anatomy': anatomy, **row})
    summary['anatomies'][anatomy] = {
        c: {'cases': 5, 'reached': sum(r['reached_target'] == 'True' for r in rows
                                    if r['controller_key'] == c)}
        for c in sorted(controllers)}
with (root / 'combined_results.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
    writer.writeheader()
    writer.writerows(combined)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(f'Completed and verified all 40 cases: {root / "combined_results.csv"}')
print(json.dumps(summary['anatomies'], indent=2))
PY
