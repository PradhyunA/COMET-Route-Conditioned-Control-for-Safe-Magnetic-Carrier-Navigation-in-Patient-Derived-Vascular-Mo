#!/usr/bin/env bash
# Linux x86_64 Ubuntu 20.04/22.04; keeps the solver and MPI in the same environment.
set -euo pipefail
ROOT="${1:-/workspace/dynamag/toolchain}"
JOBS="${BUILD_JOBS:-8}"
SV_COMMIT=47de08e103078587264ad56089a387435a46d3c0
VTK_VERSION=8.2.0
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Setup requires Ubuntu x86_64 (Runpod CPU pod or a Linux VM).' >&2
    exit 2
fi
if [[ "$(id -u)" != 0 ]]; then
    echo 'Run setup as root or with sudo; subsequent CFD runs do not need root.' >&2
    exit 2
fi
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git cmake make \
    gcc-9 g++-9 gfortran-9 libopenmpi-dev openmpi-bin python3 \
    zlib1g-dev libexpat1-dev libgl1-mesa-dev libxt-dev libx11-dev

# GCC 9 avoids newer-compiler incompatibilities in the legacy VTK 8 source.
if [[ ! -f "$ROOT/vtk-install/lib/cmake/vtk-8.2/VTKConfig.cmake" ]]; then
    if [[ ! -d "$ROOT/VTK-$VTK_VERSION" ]]; then
        curl --fail --location --retry 3 \
            "https://www.vtk.org/files/release/8.2/VTK-$VTK_VERSION.tar.gz" \
            -o "$ROOT/VTK-$VTK_VERSION.tar.gz"
        tar -xzf "$ROOT/VTK-$VTK_VERSION.tar.gz" -C "$ROOT"
    fi
    cmake -S "$ROOT/VTK-$VTK_VERSION" -B "$ROOT/vtk-build" \
        -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/vtk-install" \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
        -DVTK_USE_SYSTEM_EXPAT=ON -DVTK_USE_SYSTEM_ZLIB=ON \
        -DVTK_Group_Rendering=OFF -DVTK_Group_StandAlone=OFF \
        -DVTK_Group_Imaging=OFF -DVTK_Group_Views=OFF -DVTK_Group_Qt=OFF \
        -DVTK_RENDERING_BACKEND=None -DVTK_WRAP_PYTHON=OFF \
        -DModule_vtkChartsCore=ON -DModule_vtkCommonCore=ON \
        -DModule_vtkCommonDataModel=ON -DModule_vtkCommonExecutionModel=ON \
        -DModule_vtkFiltersCore=ON -DModule_vtkFiltersFlowPaths=ON \
        -DModule_vtkFiltersModeling=ON -DModule_vtkIOLegacy=ON -DModule_vtkIOXML=ON
    cmake --build "$ROOT/vtk-build" --parallel "$JOBS"
    cmake --install "$ROOT/vtk-build"
fi
if [[ ! -d "$ROOT/svSolver/.git" ]]; then
    git clone https://github.com/SimVascular/svSolver.git "$ROOT/svSolver"
fi
git -C "$ROOT/svSolver" checkout --detach "$SV_COMMIT"
cmake -S "$ROOT/svSolver" -B "$ROOT/sv-build" \
    -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 \
    -DCMAKE_Fortran_COMPILER=gfortran-9 -DCMAKE_BUILD_TYPE=Release \
    -DSV_USE_LOCAL_VTK=ON -DSV_VTK_LOCAL_PATH="$ROOT/vtk-install" \
    "-DSV_ADDITIONAL_CMAKE_ARGS:STRING=-DCMAKE_Fortran_COMPILER:FILEPATH=/usr/bin/gfortran-9;-DSV_BUILD_ADDITIONAL_NOMPI_VERSION:BOOL=OFF;-DSV_USE_MPI:BOOL=ON;-DSV_USE_DUMMY_MPI:BOOL=OFF"
cmake --build "$ROOT/sv-build" --parallel "$JOBS"
BIN="$ROOT/sv-build/svSolver-build/bin"
test -d "$BIN"
mkdir -p "$ROOT/bin"
for name in svsolver svpost svpre; do
    # Use the MPI binary itself, not a generated wrapper with unset SV_HOME.
    candidate="$(python3 - "$BIN" "$name" <<'PY'
import pathlib, sys
for p in sorted(pathlib.Path(sys.argv[1]).glob(sys.argv[2] + '*')):
    if p.is_file() and 'nompi' not in p.name:
        with p.open('rb') as stream:
            if stream.read(4) == b'\x7fELF':
                print(p)
                break
PY
)"
    test -n "$candidate" || { echo "Missing $name binary in $BIN" >&2; exit 1; }
    ln -sfn "$candidate" "$ROOT/bin/$name"
done
cat > "$ROOT/env.sh" <<EOF
export SVSOLVER="$ROOT/bin/svsolver"
export SVPOST="$ROOT/bin/svpost"
export FLOWSOLVER_CONFIG="$BIN"
export PATH="$ROOT/bin:\$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EOF
# Verify dynamic dependencies and an actual two-rank MPI launch before declaring setup ready.
ldd "$ROOT/bin/svsolver" > "$ROOT/solver-ldd.txt"
if grep -q 'not found' "$ROOT/solver-ldd.txt"; then cat "$ROOT/solver-ldd.txt"; exit 1; fi
grep -q 'libmpi' "$ROOT/solver-ldd.txt"
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpiexec -n 2 python3 -c 'import os; print("MPI rank", os.environ["OMPI_COMM_WORLD_RANK"])'
python3 - "$ROOT" "$SV_COMMIT" <<'PY'
import json, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
info = {"svsolver_commit": sys.argv[2], "vtk_version": "8.2.0",
        "compiler": subprocess.check_output(["gcc-9", "--version"], text=True).splitlines()[0],
        "mpi": subprocess.check_output(["mpiexec", "--version"], text=True).splitlines()[0]}
(root / "setup.json").write_text(json.dumps(info, indent=2) + "\n")
# VM setup may run under sudo; the notebook user must be able to maintain this tree.
if os.environ.get("SUDO_UID"):
    uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    for parent, dirs, files in os.walk(root):
        os.chown(parent, uid, gid)
        for name in dirs + files:
            os.lchown(pathlib.Path(parent) / name, uid, gid)
print("Solver setup complete:", root / "env.sh")
PY
