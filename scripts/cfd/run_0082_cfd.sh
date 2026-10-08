#!/usr/bin/env bash

CASE_DIR="$1"
NPROCS="${2:-6}"

if [ -z "$CASE_DIR" ]; then
    echo "Usage: ./run_0082_cfd.sh /path/to/0082_H_PULM_H/Simulations/0084_0001 [cores]"
    exit 1
fi

SVSOLVER="${SVSOLVER:-svsolver}"
MPIEXEC="${MPIEXEC:-mpiexec}"

cd "$CASE_DIR" || exit 1

if [ ! -f solver.inp ]; then
    echo "ERROR: solver.inp not found in:"
    pwd
    exit 1
fi

cp -n solver.inp solver.inp.original 2>/dev/null || true

python3 - <<'PY'
from pathlib import Path
import re

p = Path("solver.inp")
s = p.read_text()

s = re.sub(
    r"Number of Timesteps:\s*\d+",
    "Number of Timesteps: 8125",
    s,
)

s = re.sub(
    r"Time Step Size:\s*[0-9.eE+-]+",
    "Time Step Size: 0.0002375",
    s,
)

s = re.sub(
    r"Number of Timesteps between Restarts:\s*\d+",
    "Number of Timesteps between Restarts: 401",
    s,
)

p.write_text(s)
Path("numstart.dat").write_text("0\n")
PY

echo
echo "=========================================="
echo "0082_H_PULM_H CFD"
echo "=========================================="

grep -E \
'Number of Timesteps:|Time Step Size:|Number of Timesteps between Restarts:' \
solver.inp

echo
echo "MPI ranks: $NPROCS"
echo "Case: $(pwd)"
echo "Started: $(date)"
echo

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu \
        "$MPIEXEC" -n "$NPROCS" \
        "$SVSOLVER" \
        2>&1 | tee "svsolver_8125_${NPROCS}core.log"
else
    "$MPIEXEC" -n "$NPROCS" \
        "$SVSOLVER" \
        2>&1 | tee "svsolver_8125_${NPROCS}core.log"
fi

echo
echo "=========================================="
echo "CFD PROCESS ENDED"
echo "=========================================="
echo "$(date)"
