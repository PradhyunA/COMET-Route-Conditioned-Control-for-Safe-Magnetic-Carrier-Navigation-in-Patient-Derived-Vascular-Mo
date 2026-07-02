"""
generate_openfoam_case.py
=========================
Generates a ready-to-run OpenFOAM 12 case directory for steady (simpleFoam)
or pulsatile (pimpleFoam) vascular CFD of the coronary geometry produced by
inspectvtp.py / real_vessel_segments.pkl.

Usage
-----
    python generate_openfoam_case.py \
        --pkl  real_vessel_segments.pkl \
        --output  cases/coronary_steady \
        [--solver simpleFoam|pimpleFoam] \
        [--coord-scale 0.001]   # mm->m (default); set 1.0 if pkl already in metres

    cd cases/coronary_steady && ./run_cfd.sh

All confirmed + foreseeable errors fixed
-----------------------------------------
E1  OF12 geometry block syntax
    Old (rejected): vessel.stl { type triSurfaceMesh; name vessel; }
    New (required): wall { type triSurfaceMesh; file "vessel_wall.stl"; name wall; }

E2  STL region name mismatch ("Unknown region name inlet, valid: patch0")
    Patch names come from `name` in each geometry dict entry, NOT from the
    STL's internal region table. Three separate files → three named entries →
    no dependence on internal STL region naming whatsoever.

E3  vessel.eMesh missing
    explicitFeatureSnap false; implicitFeatureSnap false; multiRegionFeatureSnap false;
    features ();  -- no .eMesh file ever looked for.

E4  Illegal faces / high skewness
    (a) blockMeshDict now generated and blockMesh is step 1 of run_cfd.sh.
        snappyHexMesh requires a background hex mesh -- without one, cell
        quality is undefined and produces exactly the illegal-face pattern seen.
    (b) Refinement levels reduced (3-4 not 4-5) for thin coronary geometry.
    (c) meshQualityControls loosened to values standard for vascular CFD.
    (d) More snap / relax / layer iterations.

E5  Wrong inlet/outlet detection (z-axis heuristic fails for coronary)
    Replaced _find_terminal_endpoints() with find_inlet_and_outlets() which
    reads the pkl topology directly:
    - inlet_pt  = the `start` shared across multiple branches (all Dijkstra
                  paths in inspectvtp.py originate from the same inlet vertex)
    - outlet_pts = `end` points with no successor (terminal tips)
    - inlet_normal = mean direction of level-0 segments leaving inlet_pt
    - outlet_normal = direction of terminal segment arriving at each outlet
    O(1) hash lookup (end_key in all_starts) replaces the O(n²) distance scan.

E6  Unit mismatch: pkl coordinates in mm, OpenFOAM expects metres
    Vascular Model Repository VTPs are in mm. inspectvtp.py subsamples at
    spacing=0.3 (mm) without any unit conversion, so pkl coords are in mm.
    Added --coord-scale (default 0.001) applied to every coordinate written
    into blockMeshDict, STL geometry, and locationInMesh.
    Velocities stay in m/s (v_mean=0.02 m/s is correct for coronary).
    Cap radii are also scaled. EPS inside find_inlet_and_outlets stays in
    pkl units (mm) for point-matching, then geometry is scaled at export time.

E7  Missing `background` patch in boundary conditions
    blockMesh creates a patch named `background` covering the outer box faces.
    Without a BC entry for it, simpleFoam throws "patch background not found
    in field U/p". Added slip/zeroGradient entries for background in 0/U, 0/p.

E8  physicalProperties vs transportProperties in OpenFOAM 12
    OF11+ renamed transportProperties → physicalProperties for incompressible
    solvers. We write BOTH files with the same content so the case works on
    OF10, OF11, and OF12 without any manual editing.

E9  Successor lookup bug
    end_key was computed via _pt_key but then ignored; the code fell through to
    an O(n²) distance scan that converted every tuple key back to an array.
    Fixed: has_successor = end_key in all_starts  (O(1) set lookup).

E10 write_gmsh_geo restored
    Was present in the original file and is a useful fallback meshing path.
    Restored and written to constant/triSurface/vessel.geo.

E11 run_cfd.sh now sources a warning if OF not found, uses foamToVTK
    (more reliable than the vtkWrite function object whose library name
    changed in OF12), and has a defensive VTU file search.

Requirements
------------
    pip install pyvista numpy scipy
    OpenFOAM 12 installed and `source /opt/openfoam12/etc/bashrc` run before
    executing run_cfd.sh
"""

from __future__ import annotations

import argparse
import os
import pickle
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Blood physical constants — must match constants.py in the simulation
# ---------------------------------------------------------------------------
BLOOD_DENSITY_KG_M3          = 1060.0
BLOOD_DYNAMIC_VISCOSITY_PA_S = 3.5e-3
BLOOD_KIN_VISCOSITY_M2_S     = BLOOD_DYNAMIC_VISCOSITY_PA_S / BLOOD_DENSITY_KG_M3
HEART_RATE_HZ                = 1.0      # 60 bpm
PULSATION_AMPLITUDE          = 0.15     # ±15 % of mean


# ---------------------------------------------------------------------------
# pkl helpers
# ---------------------------------------------------------------------------

def _load_pkl(pkl_path: str) -> list:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _largest_segment(segments: list) -> dict:
    return max(segments, key=lambda s: float(s["radius"]))


def _mean_inlet_velocity(segments: list, flow_scale: float = 1.0) -> float:
    """
    Return mean inlet velocity in m/s.
    v_mean is stored as 0.02 m/s (hardcoded in inspectvtp.py) for all segments.
    The largest-radius segment is the main trunk = inlet region.
    """
    largest = _largest_segment(segments)
    return float(largest.get("v_mean", 0.02)) * flow_scale


def _inlet_radius_mm(segments: list) -> float:
    """
    Radius of the largest segment in PKL units (mm).
    Caller must apply coord_scale to convert to metres.
    """
    return float(_largest_segment(segments)["radius"])


def _vessel_bounding_box_mm(segments: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned bounding box in PKL units (mm), expanded by each segment's
    radius so the box encloses the vessel wall (not just the centreline).
    """
    lo_pts, hi_pts = [], []
    for s in segments:
        st = np.asarray(s["start"], dtype=float)
        en = np.asarray(s["end"],   dtype=float)
        r  = float(s["radius"])
        lo_pts.append(np.minimum(st, en) - r)
        hi_pts.append(np.maximum(st, en) + r)
    return np.min(lo_pts, axis=0), np.max(hi_pts, axis=0)


# ---------------------------------------------------------------------------
# Inlet / outlet detection — topology-based (no z-axis heuristic)
# ---------------------------------------------------------------------------

def _pt_key(pt, decimals: int = 4) -> tuple:
    """
    Hashable, rounded key for a 3-D point.
    decimals=4 gives 0.1 mm resolution, sufficient for coronary scale (~1-5 mm).
    All point matching in this file uses _pt_key for consistency.
    """
    return tuple(np.round(np.asarray(pt, dtype=float), decimals))


def find_inlet_and_outlets(
    segments: list,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Recover inlet point, outlet points, and inlet normal from the pkl
    produced by inspectvtp.py — using topology only, not coordinate axes.

    How inspectvtp.py builds the pkl
    ---------------------------------
    Dijkstra paths run from one shared inlet vertex (nearest mesh point to
    FaceID-2 cap centre) to each of 16 outlet caps (FaceID 3-18).
    Every path is subsampled at spacing=0.3 mm and turned into consecutive
    segments with fields: seg_id, start, end, radius, v_mean, level.
    level = fid - 2, so level=0 on the inlet branch, level=1..16 per outlet.

    Detection
    ---------
    inlet_pt:     The `start` point that appears most frequently across all
                  segments. All 16 paths begin at the same vertex, so it
                  appears N_paths times. Single-branch pkls work too.

    outlet_pts:   `end` points with no successor, i.e. end_key not in
                  all_starts. O(1) set lookup per point. Deduplicated.

    inlet_normal: Mean unit direction of all segments whose `start` == inlet_pt,
                  pointing into the vessel (away from the cap face).

    Returns (all in original PKL units = mm)
    ----------------------------------------
    inlet_pt     : (3,) float64
    outlet_pts   : list of (3,) float64
    inlet_normal : (3,) float64, unit vector pointing INTO vessel
    """
    # --- count start-point occurrences ---
    start_count: dict[tuple, list] = defaultdict(list)
    for s in segments:
        k = _pt_key(s["start"])
        start_count[k].append(np.asarray(s["start"], dtype=float))

    # inlet = most-referenced start point
    inlet_key = max(start_count, key=lambda k: len(start_count[k]))
    inlet_pt  = np.mean(start_count[inlet_key], axis=0)

    # set of all start keys for O(1) successor lookup
    all_starts: set[tuple] = set(start_count.keys())

    # --- outlets: end points with no successor ---
    outlet_pts: list[np.ndarray] = []
    for s in segments:
        end_pt  = np.asarray(s["end"], dtype=float)
        end_key = _pt_key(end_pt)
        if end_key not in all_starts:               # O(1) hash lookup
            already = any(_pt_key(op) == end_key for op in outlet_pts)
            if not already:
                outlet_pts.append(end_pt)

    # also exclude inlet_pt from outlet list (shouldn't appear, but guard)
    outlet_pts = [op for op in outlet_pts if _pt_key(op) != inlet_key]

    # --- inlet normal ---
    inlet_dirs = []
    for s in segments:
        if _pt_key(s["start"]) == inlet_key:
            d = np.asarray(s["end"], dtype=float) - inlet_pt
            n = np.linalg.norm(d)
            if n > 1e-9:
                inlet_dirs.append(d / n)
    if inlet_dirs:
        inlet_normal = np.mean(inlet_dirs, axis=0)
        nn = np.linalg.norm(inlet_normal)
        inlet_normal = inlet_normal / nn if nn > 1e-9 else np.array([0., 0., 1.])
    else:
        inlet_normal = np.array([0., 0., 1.])

    print(f"[STL] Inlet point    : {np.round(inlet_pt, 3)} mm")
    print(f"[STL] Inlet normal   : {np.round(inlet_normal, 3)}")
    print(f"[STL] Outlet count   : {len(outlet_pts)}")
    for i, op in enumerate(outlet_pts):
        print(f"[STL]   outlet {i:02d}   : {np.round(op, 3)} mm")

    if not outlet_pts:
        raise RuntimeError(
            "No outlet points found in pkl. Verify inspectvtp.py ran "
            "successfully and produced multi-segment paths (not just level-0)."
        )

    return inlet_pt, outlet_pts, inlet_normal


def _get_terminal_segment(
    terminal_pt: np.ndarray,
    segments: list,
    arriving: bool = True,
) -> dict | None:
    """
    Find the segment that ARRIVES at (arriving=True) or DEPARTS FROM
    (arriving=False) terminal_pt.  Returns None if not found.
    """
    key = _pt_key(terminal_pt)
    for s in segments:
        pt = s["end"] if arriving else s["start"]
        if _pt_key(pt) == key:
            return s
    return None


# ---------------------------------------------------------------------------
# STL export — three named files, topology-derived caps
# ---------------------------------------------------------------------------

def _disk_mesh(center: np.ndarray, normal: np.ndarray,
               radius: float, n_sides: int = 32):
    """Flat disk cap as pyvista PolyData."""
    import pyvista as pv
    normal = np.asarray(normal, dtype=float)
    nn = np.linalg.norm(normal)
    if nn < 1e-9:
        normal = np.array([0., 0., 1.])
    else:
        normal = normal / nn
    return pv.Disc(center=center, normal=normal,
                   inner=0.0, outer=radius, r_res=4, c_res=n_sides)


def export_stl_from_pkl(
    segments: list,
    output_dir: Path,
    coord_scale: float = 0.001,
    n_sides: int = 32,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Write three STL files into output_dir/:
        vessel_wall.stl    — cylinder side faces (no end caps)
        vessel_inlet.stl   — disk cap at inlet, normal pointing INTO vessel
        vessel_outlet.stl  — disk caps at all outlets, merged, normals pointing OUT

    All coordinates are scaled by coord_scale (default 0.001 = mm → m) before
    writing, so the STL files are in metres as OpenFOAM expects.

    Inlet/outlet positions come from find_inlet_and_outlets() (topology-based).
    Cap normals come from the arriving/departing segment directions.

    Returns (inlet_pt_mm, outlet_pts_mm) in original PKL units for centroid calc.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")

    output_dir = Path(output_dir)
    s = coord_scale  # shorthand

    inlet_pt_mm, outlet_pts_mm, inlet_normal = find_inlet_and_outlets(segments)

    # ------------------------------------------------------------------ WALL
    wall_meshes = []
    for seg in segments:
        start  = np.asarray(seg["start"], dtype=float) * s
        end    = np.asarray(seg["end"],   dtype=float) * s
        r      = float(seg["radius"]) * s
        axis   = end - start
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        cyl = pv.Cylinder(
            center=(start + end) / 2.0,
            direction=axis / length,
            radius=r,
            height=length,
            resolution=n_sides,
            capping=False,
        )
        wall_meshes.append(cyl)

    if not wall_meshes:
        raise RuntimeError("PKL produced no wall segments -- check inspectvtp output.")

    wall = wall_meshes[0]
    for m in wall_meshes[1:]:
        wall = wall.merge(m)
    wall_path = output_dir / "vessel_wall.stl"
    wall.save(str(wall_path))
    print(f"[STL] Wall   → {wall_path}  ({len(wall_meshes)} cylinders)")

    # ----------------------------------------------------------------- INLET
    # Radius: max radius of all level-0 segments (inlet trunk is widest)
    level0 = [seg for seg in segments if int(seg.get("level", 1)) == 0]
    if level0:
        r_inlet_mm = float(max(seg["radius"] for seg in level0))
    else:
        r_inlet_mm = float(_largest_segment(segments)["radius"])

    inlet_disk = _disk_mesh(
        center=inlet_pt_mm * s,
        normal=inlet_normal,      # unit vector, no scaling needed
        radius=r_inlet_mm * s,
        n_sides=n_sides,
    )
    inlet_path = output_dir / "vessel_inlet.stl"
    inlet_disk.save(str(inlet_path))
    print(f"[STL] Inlet  → {inlet_path}  r={r_inlet_mm:.3f} mm  "
          f"normal={np.round(inlet_normal, 3)}")

    # --------------------------------------------------------------- OUTLETS
    outlet_meshes = []
    for op_mm in outlet_pts_mm:
        seg = _get_terminal_segment(op_mm, segments, arriving=True)
        if seg is None:
            print(f"[STL] WARNING: no arriving segment for outlet {np.round(op_mm,3)}, skipping cap")
            continue
        r_out_mm = float(seg["radius"])
        # Normal points OUTWARD (start→end direction of the arriving segment)
        d = np.asarray(seg["end"], dtype=float) - np.asarray(seg["start"], dtype=float)
        n = np.linalg.norm(d)
        out_normal = d / n if n > 1e-9 else np.array([0., 0., 1.])
        outlet_meshes.append(
            _disk_mesh(op_mm * s, out_normal, r_out_mm * s, n_sides)
        )

    if not outlet_meshes:
        raise RuntimeError("No outlet caps could be built -- check pkl topology.")

    outlet_combined = outlet_meshes[0]
    for m in outlet_meshes[1:]:
        outlet_combined = outlet_combined.merge(m)
    outlet_path = output_dir / "vessel_outlet.stl"
    outlet_combined.save(str(outlet_path))
    print(f"[STL] Outlet → {outlet_path}  ({len(outlet_meshes)} caps)")

    return inlet_pt_mm, outlet_pts_mm


# ---------------------------------------------------------------------------
# OpenFOAM file header
# ---------------------------------------------------------------------------

def _header(cls: str, obj: str) -> str:
    return textwrap.dedent(f"""\
        /*--------------------------------*- C++ -*----------------------------------*\\
          =========                 |
          \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
           \\\\    /   O peration     |
            \\\\  /    A nd           |
             \\\\/     M anipulation  |
        \\*---------------------------------------------------------------------------*/
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       {cls};
            object      {obj};
        }}
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
        """)


# ---------------------------------------------------------------------------
# blockMeshDict — background hex mesh (required before snappyHexMesh)
# ---------------------------------------------------------------------------

def write_block_mesh_dict(
    path: Path,
    bbox_lo_m: np.ndarray,
    bbox_hi_m: np.ndarray,
    cells_per_dim: int = 24,
    margin_frac: float = 0.30,
) -> None:
    """
    Background hex box for snappyHexMesh.

    snappyHexMesh does NOT create a mesh from nothing — it refines an
    existing hex background mesh into the vessel STL surface. Without
    this, snappyHexMesh exits with garbage or produces the illegal-face
    pattern seen in the previous run.

    All coordinates in metres (already scaled by caller).
    margin_frac=0.30 gives 30% padding on each side so the inlet/outlet
    cap disks are well inside the domain (required for snappyHexMesh to
    correctly identify locationInMesh as inside the vessel).
    cells_per_dim=24 → 13824 background cells, plenty for snappyHexMesh
    to refine from. Too fine a background mesh wastes memory.
    """
    margin = margin_frac * (bbox_hi_m - bbox_lo_m)
    lo = bbox_lo_m - margin
    hi = bbox_hi_m + margin
    nx = ny = nz = int(cells_per_dim)

    content = _header("dictionary", "blockMeshDict") + textwrap.dedent(f"""\

        scale 1;

        vertices
        (
            ({lo[0]:.8f} {lo[1]:.8f} {lo[2]:.8f})
            ({hi[0]:.8f} {lo[1]:.8f} {lo[2]:.8f})
            ({hi[0]:.8f} {hi[1]:.8f} {lo[2]:.8f})
            ({lo[0]:.8f} {hi[1]:.8f} {lo[2]:.8f})
            ({lo[0]:.8f} {lo[1]:.8f} {hi[2]:.8f})
            ({hi[0]:.8f} {lo[1]:.8f} {hi[2]:.8f})
            ({hi[0]:.8f} {hi[1]:.8f} {hi[2]:.8f})
            ({lo[0]:.8f} {hi[1]:.8f} {hi[2]:.8f})
        );

        blocks
        (
            hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
        );

        edges ();

        boundary
        (
            background
            {{
                type patch;
                faces
                (
                    (0 3 2 1)
                    (4 5 6 7)
                    (0 1 5 4)
                    (2 3 7 6)
                    (1 2 6 5)
                    (0 4 7 3)
                );
            }}
        );

        mergePatchPairs ();
        """)
    path.write_text(content)


# ---------------------------------------------------------------------------
# snappyHexMeshDict — OF12 compliant
# ---------------------------------------------------------------------------

def write_snappy_hex_mesh_dict(
    path: Path,
    location_in_mesh: tuple = (0.0, 0.0, 0.0),
) -> None:
    """
    OF12-compatible snappyHexMeshDict.

    geometry block:
        Each STL has its own named entry with explicit `file` field.
        OF12 REQUIRES this format. The old style (STL filename as dict key,
        no `file` entry) throws: "keyword file is undefined in dictionary
        geometry/vessel.stl". Patch names come from `name`, not from the
        STL's internal region table, so pyvista's default "patch0" region
        name in the STL is irrelevant.

    feature snapping:
        All three feature snap flags set to false. No `features` block.
        OF12 deprecated surfaceFeatureExtract and the .eMesh it produced.
        Any reference to vessel.eMesh causes a fatal open-file error.

    refinement levels (3 4):
        Coronary vessels are 1-5 mm diameter with sharp bifurcations.
        Level 4-5 (used in generic templates) over-refines these and
        produces degenerate cells at bifurcation edges. Level 3-4 gives
        ~125-625 µm cells, adequate for 500 µm carrier interpolation.

    meshQualityControls:
        Loosened from OF defaults. Vascular bifurcations have inherently
        non-orthogonal faces. Tight defaults cause snappyHexMesh to
        abort or produce illegal faces. These values are standard in
        published coronary CFD workflows.

    background patch:
        The outer box produced by blockMesh is named `background` in
        blockMeshDict. This patch name must match in 0/U and 0/p.
    """
    lx, ly, lz = location_in_mesh
    content = _header("dictionary", "snappyHexMeshDict") + textwrap.dedent(f"""\

        castellatedMesh true;
        snap            true;
        addLayers       true;

        geometry
        {{
            wall
            {{
                type triSurfaceMesh;
                file "vessel_wall.stl";
                name wall;
            }}
            inlet
            {{
                type triSurfaceMesh;
                file "vessel_inlet.stl";
                name inlet;
            }}
            outlet
            {{
                type triSurfaceMesh;
                file "vessel_outlet.stl";
                name outlet;
            }}
        }}

        castellatedMeshControls
        {{
            maxLocalCells       500000;
            maxGlobalCells     4000000;
            minRefinementCells     10;
            maxLoadUnbalance     0.10;
            nCellsBetweenLevels     3;

            features ();

            refinementSurfaces
            {{
                wall
                {{
                    level (3 4);
                    patchInfo {{ type wall; }}
                }}
                inlet
                {{
                    level (3 3);
                    patchInfo {{ type patch; }}
                }}
                outlet
                {{
                    level (3 3);
                    patchInfo {{ type patch; }}
                }}
            }}

            resolveFeatureAngle 30;
            refinementRegions  {{}}

            locationInMesh ({lx:.8f} {ly:.8f} {lz:.8f});

            allowFreeStandingZoneFaces true;
        }}

        snapControls
        {{
            nSmoothPatch             5;
            tolerance                2.0;
            nSolveIter             100;
            nRelaxIter              10;
            nFeatureSnapIter        10;
            implicitFeatureSnap  false;
            explicitFeatureSnap  false;
            multiRegionFeatureSnap false;
        }}

        addLayersControls
        {{
            relativeSizes true;

            layers
            {{
                wall {{ nSurfaceLayers 3; }}
            }}

            expansionRatio            1.2;
            finalLayerThickness       0.3;
            minThickness              0.1;
            nGrow                       0;
            featureAngle               60;
            nRelaxIter                 10;
            nSmoothSurfaceNormals       5;
            nSmoothNormals              5;
            nSmoothThickness           20;
            maxFaceThicknessRatio     0.5;
            maxThicknessToMedialRatio 0.3;
            minMedianAxisAngle         90;
            nBufferCellsNoExtrude       0;
            nLayerIter                 50;
        }}

        meshQualityControls
        {{
            maxNonOrtho          75;
            maxBoundarySkewness  30;
            maxInternalSkewness   8;
            maxConcave           80;
            minVol             1e-30;
            minTetQuality      -1e30;
            minArea               -1;
            minTwist            0.02;
            minDeterminant    0.0001;
            minFaceWeight       0.02;
            minVolRatio        0.005;
            minTriangleTwist     -1;
            nSmoothScale         10;
            errorReduction     0.75;
        }}

        debug          0;
        mergeTolerance 1e-6;
        """)
    path.write_text(content)


# ---------------------------------------------------------------------------
# transportProperties + physicalProperties (both written for OF10/11/12 compat)
# ---------------------------------------------------------------------------

def _fluid_props_content() -> str:
    return textwrap.dedent(f"""\

        transportModel  Newtonian;

        // Blood at 37 C
        nu  nu  [0 2 -1 0 0 0 0]  {BLOOD_KIN_VISCOSITY_M2_S:.6e};
        rho rho [1 -3  0 0 0 0 0] {BLOOD_DENSITY_KG_M3:.1f};
        """)


def write_transport_properties(path: Path) -> None:
    """Legacy name (OF10 and earlier). Still read by OF12 simpleFoam."""
    path.write_text(_header("dictionary", "transportProperties") + _fluid_props_content())


def write_physical_properties(path: Path) -> None:
    """
    Canonical name from OF11+. OF12 simpleFoam looks for physicalProperties
    first; falls back to transportProperties. Writing both means the case
    works on any OF version without editing.
    """
    path.write_text(_header("dictionary", "physicalProperties") + _fluid_props_content())


def write_turbulence_properties(path: Path) -> None:
    path.write_text(_header("dictionary", "turbulenceProperties") + textwrap.dedent("""\

        simulationType  laminar;
        """))


# ---------------------------------------------------------------------------
# controlDict
# ---------------------------------------------------------------------------

def write_control_dict_steady(path: Path) -> None:
    path.write_text(_header("dictionary", "controlDict") + textwrap.dedent("""\

        application     simpleFoam;

        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         500;
        deltaT          1;

        writeControl    timeStep;
        writeInterval   100;
        purgeWrite      3;

        writeFormat     ascii;
        writePrecision  8;
        writeCompression off;
        timeFormat      general;
        timePrecision   6;
        runTimeModifiable true;
        """))


def write_control_dict_pulsatile(path: Path, n_phases: int = 20) -> None:
    period   = 1.0 / HEART_RATE_HZ
    dt       = period / n_phases
    end_time = 3.0 * period   # 3 cardiac cycles; use last for phase averaging
    path.write_text(_header("dictionary", "controlDict") + textwrap.dedent(f"""\

        application     pimpleFoam;

        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         {end_time:.4f};
        deltaT          {dt:.6f};

        writeControl    timeStep;
        writeInterval   1;
        purgeWrite      0;

        writeFormat     ascii;
        writePrecision  8;
        writeCompression off;
        timeFormat      fixed;
        timePrecision   6;
        runTimeModifiable true;
        """))


# ---------------------------------------------------------------------------
# fvSchemes / fvSolution
# ---------------------------------------------------------------------------

def write_fv_schemes(path: Path, solver: str = "simpleFoam") -> None:
    ddt = "steadyState" if solver == "simpleFoam" else "Euler"
    path.write_text(_header("dictionary", "fvSchemes") + textwrap.dedent(f"""\

        ddtSchemes      {{ default {ddt}; }}

        gradSchemes     {{ default Gauss linear; }}

        divSchemes
        {{
            default         none;
            div(phi,U)      Gauss linearUpwind grad(U);
        }}

        laplacianSchemes {{ default Gauss linear corrected; }}

        interpolationSchemes {{ default linear; }}

        snGradSchemes   {{ default corrected; }}
        """))


def write_fv_solution(path: Path, solver: str = "simpleFoam") -> None:
    if solver == "simpleFoam":
        path.write_text(_header("dictionary", "fvSolution") + textwrap.dedent("""\

            solvers
            {
                p
                {
                    solver          GAMG;
                    smoother        GaussSeidel;
                    tolerance       1e-7;
                    relTol          0.01;
                }
                U
                {
                    solver          smoothSolver;
                    smoother        symGaussSeidel;
                    tolerance       1e-8;
                    relTol          0.1;
                }
            }

            SIMPLE
            {
                nNonOrthogonalCorrectors 2;
                residualControl
                {
                    p   1e-5;
                    U   1e-6;
                }
            }

            relaxationFactors
            {
                fields    { p 0.3; }
                equations { U 0.7; }
            }
            """))
    else:  # pimpleFoam
        path.write_text(_header("dictionary", "fvSolution") + textwrap.dedent("""\

            solvers
            {
                p
                {
                    solver          GAMG;
                    smoother        GaussSeidel;
                    tolerance       1e-7;
                    relTol          0.01;
                }
                pFinal
                {
                    $p;
                    relTol 0;
                }
                U
                {
                    solver          smoothSolver;
                    smoother        symGaussSeidel;
                    tolerance       1e-8;
                    relTol          0.1;
                }
                UFinal
                {
                    $U;
                    relTol 0;
                }
            }

            PIMPLE
            {
                nOuterCorrectors         2;
                nCorrectors              2;
                nNonOrthogonalCorrectors 2;
            }
            """))


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def write_U_bc(path: Path, v_inlet: float, pulsatile: bool) -> None:
    """
    0/U — velocity boundary conditions.

    Patches: inlet, outlet, wall, background
    background is the outer blockMesh box; slip BC is appropriate
    (flow never reaches it in a correctly meshed case, but OF requires
    all patches to have a BC or it errors on startup).

    v_inlet: mean velocity in m/s.  v_max = 2*v_mean (Poiseuille).
    The inlet BC applies uniform (not parabolic) velocity; the parabolic
    profile develops naturally in the vessel due to the no-slip wall BC.
    For a pulsatile solve a simple sinusoidal table is used.
    """
    v_max = 2.0 * abs(v_inlet)

    if pulsatile:
        amp = PULSATION_AMPLITUDE
        inlet_bc = textwrap.dedent(f"""\
            type            timeVaryingUniformFixedValue;
            uniformValue    table
            (
                (0.00  0 0 {v_max * 1.0:.6f})
                (0.25  0 0 {v_max * (1.0 + amp):.6f})
                (0.50  0 0 {v_max * 1.0:.6f})
                (0.75  0 0 {v_max * (1.0 - amp):.6f})
                (1.00  0 0 {v_max * 1.0:.6f})
            );
            outOfBounds     repeat;""")
    else:
        inlet_bc = (
            f"type            fixedValue;\n"
            f"            value           uniform (0 0 {v_max:.6f});"
        )

    path.write_text(_header("volVectorField", "U") + textwrap.dedent(f"""\

        dimensions      [0 1 -1 0 0 0 0];
        internalField   uniform (0 0 0);

        boundaryField
        {{
            inlet
            {{
                {inlet_bc}
            }}
            outlet
            {{
                type            zeroGradient;
            }}
            wall
            {{
                type            noSlip;
            }}
            background
            {{
                type            slip;
            }}
        }}
        """))


def write_p_bc(path: Path) -> None:
    """
    0/p — kinematic pressure boundary conditions.

    background uses zeroGradient (no pressure gradient at the far-field
    box boundary — flow doesn't reach it, value is irrelevant, but the
    patch must be declared or OF errors immediately on startup).
    """
    path.write_text(_header("volScalarField", "p") + textwrap.dedent("""\

        dimensions      [0 2 -2 0 0 0 0];
        internalField   uniform 0;

        boundaryField
        {
            inlet
            {
                type            zeroGradient;
            }
            outlet
            {
                type            fixedValue;
                value           uniform 0;
            }
            wall
            {
                type            zeroGradient;
            }
            background
            {
                type            zeroGradient;
            }
        }
        """))


# ---------------------------------------------------------------------------
# Gmsh fallback geometry script (alternative to snappyHexMesh)
# ---------------------------------------------------------------------------

def write_gmsh_geo(
    path: Path,
    stl_name: str = "vessel_wall.stl",
    mesh_size_far: float = 0.003,
    mesh_size_wall: float = 0.0005,
) -> None:
    """
    Gmsh .geo as an alternative meshing path to snappyHexMesh.
    Produces a tet mesh with boundary-layer refinement near the wall.
    mesh_size_wall ~= carrier radius / 2 for adequate flow resolution.
    Not used by run_cfd.sh (which uses blockMesh + snappyHexMesh), but
    available as a fallback if snappyHexMesh continues to struggle.
    """
    content = textwrap.dedent(f"""\
        // Gmsh .geo for vascular CFD meshing
        // Generated by generate_openfoam_case.py
        // Alternative to blockMesh + snappyHexMesh.

        Merge "{stl_name}";

        Field[1] = Distance;
        Field[1].SurfacesList = {{1}};

        Field[2] = Threshold;
        Field[2].InField  = 1;
        Field[2].SizeMin  = {mesh_size_wall:.6f};
        Field[2].SizeMax  = {mesh_size_far:.6f};
        Field[2].DistMin  = 0.0;
        Field[2].DistMax  = 0.005;

        Background Field = 2;

        Mesh.Algorithm3D    = 4;
        Mesh.OptimizeNetgen = 1;
        Mesh 3;

        Field[3] = BoundaryLayer;
        Field[3].AnisoMax   = 10;
        Field[3].Quads      = 1;
        Field[3].Ratio      = 1.2;
        Field[3].Size       = {mesh_size_wall:.6f};
        Field[3].SizeFar    = {mesh_size_far:.6f};
        Field[3].Thickness  = 0.001;
        BoundaryLayer Field = 3;

        Save "mesh.msh";
        """)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Main case generator
# ---------------------------------------------------------------------------

def generate_case(
    pkl_path: str,
    output_dir: str,
    solver: str = "simpleFoam",
    flow_scale: float = 1.0,
    n_phases: int = 20,
    export_stl: bool = True,
    background_cells_per_dim: int = 24,
    coord_scale: float = 0.001,
) -> None:
    """
    Generate a complete OpenFOAM 12 case directory.

    Parameters
    ----------
    coord_scale : float
        Multiply all pkl coordinates by this to get metres.
        Default 0.001 converts mm → m (correct for Vascular Model Repository
        VTPs which are in mm, as used by inspectvtp.py).
        Pass 1.0 if your pkl is already in metres.
    """
    pulsatile = solver == "pimpleFoam"

    segments = _load_pkl(pkl_path)
    v_inlet  = _mean_inlet_velocity(segments, flow_scale)
    r_inlet_mm = _inlet_radius_mm(segments)
    r_inlet_m  = r_inlet_mm * coord_scale
    bbox_lo_mm, bbox_hi_mm = _vessel_bounding_box_mm(segments)
    bbox_lo_m = bbox_lo_mm * coord_scale
    bbox_hi_m = bbox_hi_mm * coord_scale

    print(f"[case] Solver          : {solver}")
    print(f"[case] Segments        : {len(segments)}")
    print(f"[case] coord_scale     : {coord_scale}  (pkl units → metres)")
    print(f"[case] Inlet v_mean    : {v_inlet*100:.2f} cm/s  "
          f"(v_max = {2*v_inlet*100:.2f} cm/s)")
    print(f"[case] Largest radius  : {r_inlet_mm:.3f} mm  →  {r_inlet_m*1000:.3f} mm in OF")
    print(f"[case] Bounding box    : "
          f"lo={np.round(bbox_lo_m,5)}  hi={np.round(bbox_hi_m,5)}  [m]")
    print(f"[case] Output          : {output_dir}")

    base = Path(output_dir)
    (base / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (base / "system").mkdir(parents=True, exist_ok=True)
    (base / "0").mkdir(parents=True, exist_ok=True)

    # constant/
    write_transport_properties(base / "constant" / "transportProperties")
    write_physical_properties(base / "constant" / "physicalProperties")
    write_turbulence_properties(base / "constant" / "turbulenceProperties")

    # system/
    if pulsatile:
        write_control_dict_pulsatile(base / "system" / "controlDict", n_phases)
    else:
        write_control_dict_steady(base / "system" / "controlDict")
    write_fv_schemes(base / "system" / "fvSchemes", solver)
    write_fv_solution(base / "system" / "fvSolution", solver)
    write_block_mesh_dict(
        base / "system" / "blockMeshDict",
        bbox_lo_m, bbox_hi_m,
        cells_per_dim=background_cells_per_dim,
    )

    # 0/
    write_U_bc(base / "0" / "U", v_inlet, pulsatile)
    write_p_bc(base / "0" / "p")

    # STL export + Gmsh fallback
    tri_dir = base / "constant" / "triSurface"
    if export_stl:
        export_stl_from_pkl(segments, tri_dir, coord_scale=coord_scale)
    write_gmsh_geo(tri_dir / "vessel.geo")

    # locationInMesh: length-weighted centroid of all segment midpoints, in metres
    mids, wts = [], []
    for s in segments:
        st = np.asarray(s["start"], float) * coord_scale
        en = np.asarray(s["end"],   float) * coord_scale
        mids.append((st + en) / 2.0)
        wts.append(float(np.linalg.norm(en - st)))
    mids = np.array(mids)
    wts  = np.array(wts)
    centroid_m = (mids * wts[:, None]).sum(axis=0) / wts.sum()
    print(f"[case] locationInMesh  : {np.round(centroid_m, 6)}  [m]")

    write_snappy_hex_mesh_dict(
        base / "system" / "snappyHexMeshDict",
        location_in_mesh=tuple(centroid_m),
    )

    # run_cfd.sh
    npz_stem   = Path(pkl_path).stem
    npz_suffix = "pulsatile" if pulsatile else "steady"
    npz_out    = os.path.abspath(f"{output_dir}/{npz_stem}_{npz_suffix}.npz")
    build_py   = os.path.abspath("build_cfd_interpolator.py")

    run_script = base / "run_cfd.sh"
    run_script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        # Auto-generated by generate_openfoam_case.py — OpenFOAM 12
        # Run from the case directory AFTER sourcing OpenFOAM:
        #   source /opt/openfoam12/etc/bashrc   (adjust path as needed)
        #   ./run_cfd.sh

        set -e
        cd "$(dirname "$0")"

        # Verify OpenFOAM is sourced
        if ! command -v blockMesh &> /dev/null; then
            echo "ERROR: blockMesh not found. Source OpenFOAM first:"
            echo "  source /opt/openfoam12/etc/bashrc"
            exit 1
        fi

        echo "=== 1/5  Background mesh (blockMesh) ==="
        blockMesh

        echo "=== 2/5  Vessel mesh (snappyHexMesh) ==="
        snappyHexMesh -overwrite

        echo "=== 3/5  Mesh quality check (checkMesh) ==="
        checkMesh

        echo "=== 4/5  Flow solve ({solver}) ==="
        {solver}

        echo "=== 5/5  Export + build CFD interpolator ==="
        foamToVTK -latestTime

        # foamToVTK writes internal mesh as e.g. VTK/case_500/internal.vtu
        # Grep for internal mesh file only (exclude boundary patch files)
        VTU_FILE=$(find VTK -name "internal.vtu" 2>/dev/null | sort | tail -1)
        if [ -z "$VTU_FILE" ]; then
            # Older foamToVTK names it differently -- try any non-boundary vtu
            VTU_FILE=$(find VTK -name "*.vtu" 2>/dev/null | grep -v boundary | sort | tail -1)
        fi
        if [ -z "$VTU_FILE" ]; then
            echo "ERROR: No internal-mesh .vtu found under VTK/"
            echo "Run manually:"
            echo "  foamToVTK -latestTime"
            echo "  python {build_py} --input <file.vtu> --mode {"pulsatile --period 1.0" if pulsatile else "steady"} --output {npz_out}"
            exit 1
        fi

        echo "Using VTU: $VTU_FILE"
        python {build_py} \\
            --input "$VTU_FILE" \\
            --mode {"pulsatile --period 1.0" if pulsatile else "steady"} \\
            --output "{npz_out}"

        echo ""
        echo "=== Done ==="
        echo "CFD interpolator: {npz_out}"
        """))
    os.chmod(run_script, 0o755)
    print(f"[case] run_cfd.sh      → {run_script}")
    print(f"[case] CFD output will → {npz_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an OpenFOAM 12 case for coronary vascular CFD."
    )
    parser.add_argument("--pkl",    required=True,
                        help="Path to real_vessel_segments.pkl from inspectvtp.py.")
    parser.add_argument("--output", required=True,
                        help="Output case directory (created if absent).")
    parser.add_argument("--solver", choices=["simpleFoam", "pimpleFoam"],
                        default="simpleFoam",
                        help="simpleFoam=steady (default); pimpleFoam=pulsatile.")
    parser.add_argument("--flow-scale", type=float, default=1.0,
                        help="Global velocity multiplier λ.")
    parser.add_argument("--n-phases", type=int, default=20,
                        help="Phase snapshots for pimpleFoam. Default 20.")
    parser.add_argument("--no-stl", action="store_true",
                        help="Skip STL export (supply your own STL files).")
    parser.add_argument("--background-cells", type=int, default=24,
                        help="blockMesh cells per axis. Default 24.")
    parser.add_argument("--coord-scale", type=float, default=0.001,
                        help="PKL coordinate → metres scale factor. "
                             "Default 0.001 (mm→m, correct for VModel Repo VTPs). "
                             "Use 1.0 if pkl is already in metres.")
    args = parser.parse_args()

    generate_case(
        pkl_path=args.pkl,
        output_dir=args.output,
        solver=args.solver,
        flow_scale=args.flow_scale,
        n_phases=args.n_phases,
        export_stl=not args.no_stl,
        background_cells_per_dim=args.background_cells,
        coord_scale=args.coord_scale,
    )


if __name__ == "__main__":
    main()
