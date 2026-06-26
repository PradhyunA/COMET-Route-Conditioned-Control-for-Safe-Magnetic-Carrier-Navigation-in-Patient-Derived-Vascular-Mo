"""
build_cfd_interpolator.py
=========================
Offline script: reads raw OpenFOAM velocity field output (.vtu or .vtk),
extracts cell-centre coordinates and velocity vectors, and saves a compact
.npz archive that CFDFlowField / PulsatileCFDFlowField can load at runtime.

Run once per vessel model and flow condition.  The resulting .npz is small
(~50–200 MB for typical vascular meshes) and loads in < 1 second.

Usage
-----
Steady (simpleFoam):
    python build_cfd_interpolator.py \\
        --input  openfoam_case/postProcessing/U_final.vtu \\
        --output data/abdominal_aorta_steady.npz \\
        --mode   steady

Pulsatile (pimpleFoam, 20 phase snapshots):
    python build_cfd_interpolator.py \\
        --input  "openfoam_case/VTK/phase_*.vtu" \\
        --output data/abdominal_aorta_pulsatile.npz \\
        --mode   pulsatile \\
        --period 1.0

The --input argument for pulsatile mode accepts a glob pattern.
Snapshots are sorted by filename, so name them with zero-padded indices
(phase_00.vtu, phase_01.vtu, ...) to ensure correct temporal ordering.

Requirements
------------
    pip install pyvista vtk numpy

pyvista wraps VTK and reads .vtu / .vtk / .vtp files directly.
The script extracts cell-centre data, not point data — OpenFOAM writes
cell-centred velocity fields by default.  If your solver writes
point-centred data, pass --point-data.

Coordinate alignment
--------------------
The .npz is saved in the same coordinate frame as the input .vtu file.
This must match the coordinate frame of your vessel .pkl file.  If you
derived the OpenFOAM geometry from the same VTP that generated the .pkl,
the frames are identical and no transform is needed.

If they differ (e.g. the CFD geometry was scaled or translated during
meshing), pass --scale and --translate to apply a rigid transform:
    --scale 0.001          # convert mm → m
    --translate "0,0,0.5"  # shift origin 0.5 m in z
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np


def load_vtu_steady(vtu_path: str, use_point_data: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single .vtu snapshot.

    Returns
    -------
    points   : (M, 3) float64   cell-centre (or point) coordinates [m]
    velocity : (M, 3) float64   velocity vectors [m/s]
    """
    try:
        import pyvista as pv
    except ImportError:
        sys.exit("pyvista is required: pip install pyvista")

    mesh = pv.read(vtu_path)

    if use_point_data:
        pts = np.array(mesh.points, dtype=np.float64)
        if "U" in mesh.point_data:
            vel = np.array(mesh.point_data["U"], dtype=np.float64)
        elif "velocity" in mesh.point_data:
            vel = np.array(mesh.point_data["velocity"], dtype=np.float64)
        else:
            raise KeyError(
                f"No 'U' or 'velocity' array in point data of {vtu_path}. "
                f"Available: {list(mesh.point_data.keys())}"
            )
    else:
        # Cell-centred: compute cell centres from pyvista
        cell_centres = mesh.cell_centers()
        pts = np.array(cell_centres.points, dtype=np.float64)

        if "U" in mesh.cell_data:
            vel = np.array(mesh.cell_data["U"], dtype=np.float64)
        elif "velocity" in mesh.cell_data:
            vel = np.array(mesh.cell_data["velocity"], dtype=np.float64)
        elif "U" in cell_centres.point_data:
            vel = np.array(cell_centres.point_data["U"], dtype=np.float64)
        else:
            raise KeyError(
                f"No 'U' or 'velocity' array in cell data of {vtu_path}. "
                f"Available cell arrays: {list(mesh.cell_data.keys())}"
            )

    if vel.ndim == 1:
        # Scalar field accidentally loaded — shouldn't happen for U
        raise ValueError(f"Velocity field in {vtu_path} is scalar, not vector.")

    return pts, vel


def apply_transform(
    points: np.ndarray,
    velocity: np.ndarray,
    scale: float,
    translate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply scale + translate to coordinates; scale velocity magnitudes."""
    points = points * scale + translate
    velocity = velocity * scale   # scale m/s consistently if coords are scaled
    return points, velocity


def build_steady(
    input_path: str,
    output_path: str,
    use_point_data: bool = False,
    scale: float = 1.0,
    translate: np.ndarray | None = None,
) -> None:
    print(f"[build] Loading steady field: {input_path}")
    pts, vel = load_vtu_steady(input_path, use_point_data)

    if translate is None:
        translate = np.zeros(3, dtype=np.float64)

    if scale != 1.0 or np.any(translate != 0):
        pts, vel = apply_transform(pts, vel, scale, translate)

    print(f"[build] {len(pts):,} cells  velocity range "
          f"{np.linalg.norm(vel, axis=1).min():.4f} – "
          f"{np.linalg.norm(vel, axis=1).max():.4f} m/s")

    np.savez_compressed(
        output_path,
        points=pts.astype(np.float32),
        velocity=vel.astype(np.float32),
    )
    size_mb = Path(output_path + ".npz").stat().st_size / 1e6 if not output_path.endswith(".npz") else Path(output_path).stat().st_size / 1e6
    print(f"[build] Saved → {output_path}  ({size_mb:.1f} MB)")


def build_pulsatile(
    glob_pattern: str,
    output_path: str,
    period: float = 1.0,
    use_point_data: bool = False,
    scale: float = 1.0,
    translate: np.ndarray | None = None,
) -> None:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        sys.exit(f"[build] No files matched pattern: {glob_pattern}")

    print(f"[build] Found {len(files)} phase snapshots")

    if translate is None:
        translate = np.zeros(3, dtype=np.float64)

    all_velocity = []
    pts_ref = None

    for i, f in enumerate(files):
        print(f"[build]   phase {i:02d}/{len(files)-1}: {f}")
        pts, vel = load_vtu_steady(f, use_point_data)

        if scale != 1.0 or np.any(translate != 0):
            pts, vel = apply_transform(pts, vel, scale, translate)

        if pts_ref is None:
            pts_ref = pts
        else:
            # Sanity: all snapshots must have the same mesh topology
            if pts.shape != pts_ref.shape:
                raise ValueError(
                    f"Phase {i} has {pts.shape[0]} cells but phase 0 has "
                    f"{pts_ref.shape[0]} cells.  All snapshots must share "
                    f"the same mesh."
                )

        all_velocity.append(vel.astype(np.float32))

    velocity_stack = np.stack(all_velocity, axis=0)   # (N, M, 3)

    print(f"[build] Stacked velocity array: {velocity_stack.shape}  "
          f"({velocity_stack.nbytes / 1e6:.0f} MB uncompressed)")

    np.savez_compressed(
        output_path,
        points=pts_ref.astype(np.float32),
        velocity=velocity_stack,
        period=np.float32(period),
    )
    out = output_path if output_path.endswith(".npz") else output_path + ".npz"
    size_mb = Path(out).stat().st_size / 1e6
    print(f"[build] Saved → {out}  ({size_mb:.1f} MB compressed)")


def _parse_translate(s: str) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--translate must be 'x,y,z'")
    return np.array(parts, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CFD interpolator .npz from OpenFOAM .vtu output."
    )
    parser.add_argument("--input", required=True,
                        help="Path to .vtu file (steady) or glob pattern (pulsatile).")
    parser.add_argument("--output", required=True,
                        help="Output .npz path (extension added if missing).")
    parser.add_argument("--mode", choices=["steady", "pulsatile"], default="steady",
                        help="steady = single snapshot; pulsatile = N phase files.")
    parser.add_argument("--period", type=float, default=1.0,
                        help="Cardiac cycle period in seconds (pulsatile only).")
    parser.add_argument("--point-data", action="store_true",
                        help="Read point data instead of cell-centred data.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Coordinate scale factor (e.g. 0.001 to convert mm → m).")
    parser.add_argument("--translate", type=_parse_translate, default=None,
                        metavar="X,Y,Z",
                        help="Coordinate offset in metres, applied after scale.")
    args = parser.parse_args()

    out = args.output if args.output.endswith(".npz") else args.output + ".npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    translate = args.translate if args.translate is not None else np.zeros(3)

    if args.mode == "steady":
        build_steady(
            args.input, out, args.point_data, args.scale, translate
        )
    else:
        build_pulsatile(
            args.input, out, args.period, args.point_data, args.scale, translate
        )


if __name__ == "__main__":
    main()
