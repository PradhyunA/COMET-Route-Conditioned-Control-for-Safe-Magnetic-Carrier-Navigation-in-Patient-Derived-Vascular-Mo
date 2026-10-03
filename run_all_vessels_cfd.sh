#!/bin/bash
# =============================================================================
# run_all_vessels_cfd.sh
# =============================================================================
# Full CFD pipeline for all three DynaMag vessel models.
# Run from the DynaMag-Magnetic-Nanoparticle-Steering-Simulation repo root
# inside WSL2 after OpenFOAM is installed and sourced.
#
# Expects your .pkl files at the paths below — edit VESSEL_DIR if different.
# Output .npz files land in data/cfd/ and are ready for patched_vessel_network.py
#
# Estimated wall time (8-core CPU):
#   Meshing per vessel:        5–15 min
#   simpleFoam per vessel:     10–30 min
#   Total for all three:       ~1–2 hours (runs serially)
#
# Usage:
#   source /opt/openfoam12/etc/bashrc
#   chmod +x run_all_vessels_cfd.sh
#   ./run_all_vessels_cfd.sh
# =============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CORE_DIR="$REPO_ROOT/dynamag_simulation/core"
DATA_DIR="$REPO_ROOT/data"
CFD_DIR="$REPO_ROOT/data/cfd"
CASES_DIR="$REPO_ROOT/cfd_cases"

mkdir -p "$CFD_DIR"
mkdir -p "$CASES_DIR"

# ---------------------------------------------------------------------------
# Edit these paths to match your actual pkl filenames
# ---------------------------------------------------------------------------
declare -A VESSELS
VESSELS["abdominal_aorta"]="$DATA_DIR/aortasegments.pkl"
VESSELS["aortic_root"]="$DATA_DIR/rootsegments.pkl"
VESSELS["pulmonary"]="$DATA_DIR/pulmonarysegments.pkl"

# If your pkls have different names, check with:
#   ls data/*.pkl
# and update the paths above.

# ---------------------------------------------------------------------------
# Helper: mesh + solve + build interpolator for one vessel
# ---------------------------------------------------------------------------
run_vessel_cfd() {
    local NAME="$1"
    local PKL="$2"
    local CASE="$CASES_DIR/${NAME}_cfd"
    local NPZ="$CFD_DIR/${NAME}_steady.npz"

    echo ""
    echo "========================================================"
    echo " Vessel: $NAME"
    echo " PKL:    $PKL"
    echo " Case:   $CASE"
    echo " Output: $NPZ"
    echo "========================================================"

    if [ ! -f "$PKL" ]; then
        echo "  [SKIP] PKL not found: $PKL"
        return
    fi

    if [ -f "$NPZ" ]; then
        echo "  [SKIP] Interpolator already exists: $NPZ"
        echo "         Delete it to force a rebuild."
        return
    fi

    # --- Step A: Generate OpenFOAM case directory ---
    echo ""
    echo "  [A] Generating OpenFOAM case..."
    python3 "$CORE_DIR/generate_openfoam_case.py" \
        --pkl "$PKL" \
        --output "$CASE" \
        --solver simpleFoam \
        --flow-scale 1.0

    # --- Step B: Build background hex mesh ---
    echo ""
    echo "  [B] Building background mesh..."
    cd "$CASE"

    # Compute bounding box from the segment PKL and build a blockMesh around it.
    # This avoids reparsing the large ASCII multi-region STL.
    python3 - <<PYEOF
import pickle, numpy as np, pathlib, textwrap

with open("$PKL", "rb") as f:
    segments = pickle.load(f)

lo = np.array([np.inf, np.inf, np.inf], dtype=float)
hi = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
for s in segments:
    start = np.asarray(s["start"], dtype=float)
    end = np.asarray(s["end"], dtype=float)
    radius = float(s["radius"])
    lo = np.minimum(lo, np.minimum(start, end) - radius)
    hi = np.maximum(hi, np.maximum(start, end) + radius)
pad = 0.005  # 5 mm padding around vessel

# blockMeshDict: coarse background hex mesh for snappyHexMesh
content = textwrap.dedent(f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1.0;
vertices
(
    ({lo[0]-pad:.6f} {lo[1]-pad:.6f} {lo[2]-pad:.6f})
    ({hi[0]+pad:.6f} {lo[1]-pad:.6f} {lo[2]-pad:.6f})
    ({hi[0]+pad:.6f} {hi[1]+pad:.6f} {lo[2]-pad:.6f})
    ({lo[0]-pad:.6f} {hi[1]+pad:.6f} {lo[2]-pad:.6f})
    ({lo[0]-pad:.6f} {lo[1]-pad:.6f} {hi[2]+pad:.6f})
    ({hi[0]+pad:.6f} {lo[1]-pad:.6f} {hi[2]+pad:.6f})
    ({hi[0]+pad:.6f} {hi[1]+pad:.6f} {hi[2]+pad:.6f})
    ({lo[0]-pad:.6f} {hi[1]+pad:.6f} {hi[2]+pad:.6f})
);
blocks ( hex (0 1 2 3 4 5 6 7) (20 20 20) simpleGrading (1 1 1) );
boundary
(
    allBoundary {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(1 2 6 5)(3 0 4 7)); }}
);
""")
pathlib.Path("system/blockMeshDict").write_text(content)

print(f"  blockMeshDict written  bbox: {lo} → {hi}")
PYEOF

    blockMesh > blockMesh.log 2>&1
    echo "    blockMesh done"

    # --- Step C: Extract surface features for snappyHexMesh ---
    echo ""
    echo "  [C] Extracting surface features..."
    surfaceFeatureExtract > surfaceFeatureExtract.log 2>&1 || \
        echo "    surfaceFeatureExtract skipped (feature edges optional)"

    # --- Step D: snappyHexMesh ---
    echo ""
    echo "  [D] Running snappyHexMesh (this takes a few minutes)..."
    snappyHexMesh -overwrite > snappyHexMesh.log 2>&1
    echo "    snappyHexMesh done"

    # --- Step E: Check mesh quality ---
    echo ""
    echo "  [E] Checking mesh..."
    checkMesh > checkMesh.log 2>&1
    grep -E "Mesh OK|FAILED|cells:|faces:|Max non-ortho" checkMesh.log || true

    # --- Step F: Run simpleFoam ---
    echo ""
    echo "  [F] Running simpleFoam (this may take 10–30 min)..."
    simpleFoam > simpleFoam.log 2>&1
    echo "    simpleFoam done"

    # Check convergence
    FINAL_RES=$(grep "Solving for U" simpleFoam.log | tail -1)
    echo "    Final residual line: $FINAL_RES"

    # --- Step G: Export final timestep to VTK ---
    echo ""
    echo "  [G] Converting to VTK..."
    foamToVTK -latestTime > foamToVTK.log 2>&1
    VTU=$(find VTK -name "*.vtu" | sort | tail -1)
    echo "    VTU file: $VTU"

    if [ -z "$VTU" ]; then
        echo "  [ERROR] No VTU file found. Check simpleFoam.log"
        return 1
    fi

    # --- Step H: Build interpolator ---
    echo ""
    echo "  [H] Building CFD interpolator..."
    cd "$REPO_ROOT"
    python3 "$CORE_DIR/build_cfd_interpolator.py" \
        --input "$CASE/$VTU" \
        --output "$NPZ" \
        --mode steady

    echo ""
    echo "  [DONE] $NAME → $NPZ"
    cd "$REPO_ROOT"
}

# ---------------------------------------------------------------------------
# Run all three vessels
# ---------------------------------------------------------------------------
echo ""
echo "DynaMag CFD pipeline — all three vessels"
echo "Repo root: $REPO_ROOT"
echo "Output:    $CFD_DIR"
echo ""

# Check OpenFOAM is sourced
if ! command -v simpleFoam &> /dev/null; then
    echo "ERROR: simpleFoam not found."
    echo "Run:  source /opt/openfoam12/etc/bashrc"
    echo "Then re-run this script."
    exit 1
fi

simpleFoam --version 2>&1 | head -1

run_vessel_cfd "abdominal_aorta" "${VESSELS[abdominal_aorta]}"
run_vessel_cfd "aortic_root"     "${VESSELS[aortic_root]}"
run_vessel_cfd "pulmonary"       "${VESSELS[pulmonary]}"

echo ""
echo "========================================================"
echo " All vessels processed."
echo " Interpolators written to: $CFD_DIR/"
ls -lh "$CFD_DIR/"*.npz 2>/dev/null || echo "  (none found — check logs above)"
echo ""
echo " Next step: run verify_cfd_interpolator.py against"
echo " one of these .npz files, then update your benchmark"
echo " scripts to use RealVesselNetworkCFD."
echo "========================================================"
