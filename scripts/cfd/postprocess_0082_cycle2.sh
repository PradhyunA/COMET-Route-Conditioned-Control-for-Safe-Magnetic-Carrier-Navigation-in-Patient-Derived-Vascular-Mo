#!/usr/bin/env bash

CASE_DIR="$1"

if [ -z "$CASE_DIR" ]; then
    echo "Usage: ./postprocess_0082_cycle2.sh /path/to/0082_H_PULM_H/Simulations/0084_0001"
    exit 1
fi

SVPOST="${SVPOST:-svpost}"

cd "$CASE_DIR" || exit 1

OUT="$CASE_DIR/post_cycle2_4010_8020"
mkdir -p "$OUT"

echo
echo "=========================================="
echo "POSTPROCESSING 0082 SECOND CARDIAC CYCLE"
echo "=========================================="
echo "Start:     4010"
echo "Stop:      8020"
echo "Increment: 401"
echo

"$SVPOST" \
    -indir "$CASE_DIR" \
    -outdir "$OUT" \
    -start 4010 \
    -stop 8020 \
    -incr 401 \
    -vtu "all_results.vtu" \
    2>&1 | tee svpost_cycle2.log

echo
echo "=========================================="
echo "VTU OUTPUT"
echo "=========================================="

find "$OUT" -maxdepth 1 -type f -name '*.vtu' | sort -V

echo
echo "VTU count:"
find "$OUT" -maxdepth 1 -type f -name '*.vtu' | wc -l

echo
echo "Output directory:"
echo "$OUT"
