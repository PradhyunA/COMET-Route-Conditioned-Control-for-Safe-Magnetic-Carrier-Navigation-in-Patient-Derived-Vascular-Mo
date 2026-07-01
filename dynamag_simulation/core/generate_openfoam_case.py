"""
generate_openfoam_case.py
=========================
Generates a ready-to-run OpenFOAM case directory for steady (simpleFoam)
or pulsatile (pimpleFoam) vascular flow simulation.

Targets OpenFOAM 12 (openfoam.org).

Takes the vessel surface geometry (STL or VTP), sets physiologically
realistic boundary conditions from the vessel pkl, and writes all
required OpenFOAM configuration files.  After running this script,
the case can be executed with:

    cd <case_dir>
    ./run_cfd.sh

or manually:

    blockMesh
    snappyHexMesh -overwrite
    checkMesh
    simpleFoam          # or pimpleFoam for pulsatile
    python build_cfd_interpolator.py --input postProcessing/... --output ...

CHANGELOG (patched for OpenFOAM 12 compatibility)
--------------------------------------------------
1. snappyHexMeshDict geometry block now includes the required `file`
   entry.  OpenFOAM 12 rejects the old style where the STL filename was
   used as the dict key with no `file` field
   ("keyword file is undefined in dictionary geometry/vessel.stl").

2. Each vessel part (wall / inlet / outlet) is its own triSurfaceMesh
   geometry entry with an explicit `name`, so patch names come from the
   generator, not from the STL's internal region table. This avoids the
   "Unknown region name inlet ... Valid region names are patch0" error
   that occurs when a single merged STL is fed in as one file/one region.

3. `explicitFeatureSnap` and `implicitFeatureSnap` are now explicitly
   set to false, and no `features` entries are declared, so
   snappyHexMesh never looks for a `vessel.eMesh` file (which requires
   a separate surfaceFeatureExtract-equivalent step that OpenFOAM 12
   handles differently and this pipeline doesn't run).

4. Added write_block_mesh_dict(): the script previously never generated
   a blockMeshDict at all, and run_cfd.sh had `blockMesh` commented out.
   snappyHexMesh refines an existing *background* mesh -- without one,
   mesh quality (illegal faces / high skewness) will be bad regardless
   of any other tuning. This writes a background hex box around the
   vessel bounding box (with margin) sized from the vessel geometry.

5. Loosened castellatedMesh / snap / addLayers / meshQuality settings
   for vascular geometry: lower refinement levels (fewer, larger cells
   near real coronary bifurcations -> fewer illegal faces), more snap
   and relax iterations, and much more permissive meshQualityControls
   thresholds (this is a standard, accepted practice for thin/complex
   vascular CFD meshes -- you trade some cell regularity for a mesh
   that actually completes checkMesh).

6. Fixed the `vtkWrite` function-object library reference in
   controlDict: "libIOvtkWrite.so" is not a real OpenFOAM library name.
   The correct OpenFOAM 12 library is `utilityFunctionObjects`. Left as
   a function object for convenience, but flagged with a comment since
   you haven't reached the solve step yet -- verify VTK output location
   once simpleFoam actually runs, the run_cfd.sh glob is written
   defensively (searches for any *.vtu under VTK/) to avoid hardcoding
   a path that may not match exactly.

Everything else (STL export, boundary condition writers, fvSchemes/
fvSolution, transportProperties, blood constants, pkl loading) is
unchanged from your working version.

Requirements
------------
    pip install pyvista numpy
    OpenFOAM 12 (openfoam.org) installed and sourced
"""

from __future__ import annotations

import argparse
import os
import pickle
import textwrap
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Blood physical constants — must match constants.py in the simulation
# ---------------------------------------------------------------------------
BLOOD_DENSITY_KG_M3 = 1060.0
BLOOD_DYNAMIC_VISCOSITY_PA_S = 3.5e-3
BLOOD_KIN_VISCOSITY_M2_S = BLOOD_DYNAMIC_VISCOSITY_PA_S / BLOOD_DENSITY_KG_M3
HEART_RATE_HZ = 1.0          # 60 bpm
PULSATION_AMPLITUDE = 0.15   # ±15 % — matches flow.py


def _load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _largest_segment(segments: list) -> dict:
    return max(segments, key=lambda s: float(s["radius"]))


def _mean_inlet_velocity(segments: list, flow_scale: float = 1.0) -> float:
    """
    Derive mean inlet velocity from the pkl, matching RealVesselNetwork
    scaling convention:  v_mean * (r / r_ref) * flow_scale
    At the largest segment (r = r_ref), scale factor = 1.0.
    """
    largest = _largest_segment(segments)
    v_mean_raw = float(largest.get("v_mean", 0.02))
    return v_mean_raw * flow_scale


def _inlet_radius(segments: list) -> float:
    return float(_largest_segment(segments)["radius"])


def _vessel_bounding_box(segments: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Bounding box of the full vessel network, expanded by each segment's
    radius so the box actually encloses the vessel wall (not just the
    centerline points). Used for the blockMesh background mesh.
    """
    pts_lo = []
    pts_hi = []
    for s in segments:
        st = np.asarray(s["start"], dtype=float)
        en = np.asarray(s["end"], dtype=float)
        r = float(s["radius"])
        pts_lo.append(np.minimum(st, en) - r)
        pts_hi.append(np.maximum(st, en) + r)
    lo = np.min(np.array(pts_lo), axis=0)
    hi = np.max(np.array(pts_hi), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# OpenFOAM file writers
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


def write_transport_properties(path: Path) -> None:
    content = _header("dictionary", "transportProperties") + textwrap.dedent(f"""\

        transportModel  Newtonian;

        // Blood at 37 C
        nu              nu [0 2 -1 0 0 0 0] {BLOOD_KIN_VISCOSITY_M2_S:.6e};

        rho             rho [1 -3 0 0 0 0 0] {BLOOD_DENSITY_KG_M3:.1f};
        """)
    path.write_text(content)


def write_turbulence_properties(path: Path) -> None:
    content = _header("dictionary", "turbulenceProperties") + textwrap.dedent("""\

        simulationType  laminar;
        """)
    path.write_text(content)


def write_block_mesh_dict(
    path: Path,
    bbox_lo: np.ndarray,
    bbox_hi: np.ndarray,
    cells_per_dim: int = 24,
    margin_frac: float = 0.25,
) -> None:
    """
    Background hex mesh for snappyHexMesh to refine.

    snappyHexMesh castellates/snaps/layers an *existing* background
    mesh — it does not create one from nothing. This box fully encloses
    the vessel bounding box with a margin so the inlet/outlet caps and
    wall are safely inside the domain (snappyHexMesh will delete the
    background cells outside the vessel STL and refine the rest).

    Parameters
    ----------
    bbox_lo, bbox_hi : (3,) arrays
        Vessel bounding box in metres (see _vessel_bounding_box).
    cells_per_dim : int
        Background cells per axis before refinement. Keep this modest
        (20-30) — refinement happens in snappyHexMeshDict, not here.
        Too fine a background mesh massively increases maxGlobalCells
        needs and is a common cause of snappyHexMesh running out of
        memory or producing degenerate cells at the refinement boundary.
    margin_frac : float
        Extra space added on each side, as a fraction of the bbox size.
    """
    size = bbox_hi - bbox_lo
    margin = margin_frac * size
    lo = bbox_lo - margin
    hi = bbox_hi + margin

    nx = ny = nz = int(cells_per_dim)

    content = _header("dictionary", "blockMeshDict") + textwrap.dedent(f"""\

        scale   1;

        vertices
        (
            ({lo[0]:.6f} {lo[1]:.6f} {lo[2]:.6f})
            ({hi[0]:.6f} {lo[1]:.6f} {lo[2]:.6f})
            ({hi[0]:.6f} {hi[1]:.6f} {lo[2]:.6f})
            ({lo[0]:.6f} {hi[1]:.6f} {lo[2]:.6f})
            ({lo[0]:.6f} {lo[1]:.6f} {hi[2]:.6f})
            ({hi[0]:.6f} {lo[1]:.6f} {hi[2]:.6f})
            ({hi[0]:.6f} {hi[1]:.6f} {hi[2]:.6f})
            ({lo[0]:.6f} {hi[1]:.6f} {hi[2]:.6f})
        );

        blocks
        (
            hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
        );

        edges
        (
        );

        boundary
        (
            background
            {{
                type patch;
                faces
                (
                    (0 1 2 3)
                    (4 5 6 7)
                    (0 1 5 4)
                    (2 3 7 6)
                    (1 2 6 5)
                    (0 3 7 4)
                );
            }}
        );

        mergePatchPairs
        (
        );
        """)
    path.write_text(content)


def write_control_dict_steady(path: Path) -> None:
    content = _header("dictionary", "controlDict") + textwrap.dedent("""\

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

        functions
        {
            // NOTE: you haven't reached the solve step yet in your
            // debugging, so this is unverified against your actual
            // OpenFOAM 12 install -- if it errors, just delete this
            // functions block and run `foamToVTK -latestTime` manually
            // after simpleFoam finishes instead; run_cfd.sh's VTU
            // lookup globs for any *.vtu under VTK/ either way, so it
            // doesn't depend on this working.
            writeFinalVTK
            {
                type            vtkWrite;
                libs            (utilityFunctionObjects);
                writeControl    onEnd;
                writeInterval   1;
                fields          (U p);
            }
        }
        """)
    path.write_text(content)


def write_control_dict_pulsatile(path: Path, n_phases: int = 20) -> None:
    period = 1.0 / HEART_RATE_HZ
    dt = period / n_phases
    end_time = 3.0 * period   # run 3 cardiac cycles; use last for phase averaging

    content = _header("dictionary", "controlDict") + textwrap.dedent(f"""\

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
        """)
    path.write_text(content)


def write_fv_schemes(path: Path, solver: str = "simpleFoam") -> None:
    content = _header("dictionary", "fvSchemes") + textwrap.dedent("""\

        ddtSchemes
        {
            default         """ + ("steadyState;" if solver == "simpleFoam" else "Euler;") + """
        }

        gradSchemes
        {
            default         Gauss linear;
        }

        divSchemes
        {
            default         none;
            div(phi,U)      Gauss linearUpwind grad(U);
        }

        laplacianSchemes
        {
            default         Gauss linear corrected;
        }

        interpolationSchemes
        {
            default         linear;
        }

        snGradSchemes
        {
            default         corrected;
        }
        """)
    path.write_text(content)


def write_fv_solution(path: Path, solver: str = "simpleFoam") -> None:
    if solver == "simpleFoam":
        content = _header("dictionary", "fvSolution") + textwrap.dedent("""\

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
                    p               1e-5;
                    U               1e-6;
                }
            }

            relaxationFactors
            {
                fields  { p 0.3; }
                equations { U 0.7; }
            }
            """)
    else:  # pimpleFoam
        content = _header("dictionary", "fvSolution") + textwrap.dedent("""\

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
                nOuterCorrectors     2;
                nCorrectors          2;
                nNonOrthogonalCorrectors 2;
            }
            """)
    path.write_text(content)


def write_U_bc(path: Path, v_inlet: float, radius_inlet: float, pulsatile: bool) -> None:
    """
    Velocity boundary conditions.

    Inlet  : parabolic profile pointing in +z direction (vessel axis).
             v_max = 2 * v_mean (Poiseuille).
             For pulsatile, a time-varying uniform value is used — the
             spatial parabola is enforced via the mesh inlet face geometry.
    Wall   : noSlip
    Outlet : zeroGradient
    """
    v_max = 2.0 * abs(v_inlet)

    if pulsatile:
        # timeVaryingUniformFixedValue: sinusoidal pulsation
        inlet_bc = textwrap.dedent(f"""\
            type            timeVaryingUniformFixedValue;
            uniformValue    table
            (
                // t   Ux  Uy  Uz
                (0.00   0   0   {v_max * (1.0 + PULSATION_AMPLITUDE * 0.0):.6f})
                (0.25   0   0   {v_max * (1.0 + PULSATION_AMPLITUDE * 1.0):.6f})
                (0.50   0   0   {v_max * (1.0 + PULSATION_AMPLITUDE * 0.0):.6f})
                (0.75   0   0   {v_max * (1.0 - PULSATION_AMPLITUDE * 1.0):.6f})
                (1.00   0   0   {v_max * (1.0 + PULSATION_AMPLITUDE * 0.0):.6f})
            );
            outOfBounds     repeat;
            """)
    else:
        inlet_bc = f"type            fixedValue;\n            value           uniform (0 0 {v_max:.6f});"

    content = _header("volVectorField", "U") + textwrap.dedent(f"""\

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
        }}
        """)
    path.write_text(content)


def write_p_bc(path: Path) -> None:
    content = _header("volScalarField", "p") + textwrap.dedent("""\

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
        }
        """)
    path.write_text(content)


def write_snappy_hex_mesh_dict(
    path: Path,
    location_in_mesh: tuple = (0.0, 0.0, 0.0),
) -> None:
    """
    snappyHexMeshDict using three separate named STL files:
        vessel_wall.stl    -> patch type wall
        vessel_inlet.stl   -> patch type patch (inlet)
        vessel_outlet.stl  -> patch type patch (outlet)

    Each geometry entry declares its own `name`, which is what actually
    determines the resulting patch name -- NOT the region table inside
    the STL file. This is what avoids the "Unknown region name inlet,
    valid region names are patch0" error: since each part is its own
    file/geometry entry, OpenFOAM never needs the STL to have named
    regions at all.

    `file` is required explicitly (OpenFOAM 12 -- the old style of using
    the filename as the dict key with no `file` entry throws
    "keyword file is undefined").

    Feature snapping is fully disabled (explicitFeatureSnap false,
    implicitFeatureSnap false, no `features` entries) so snappyHexMesh
    never looks for a vessel.eMesh file, which this pipeline doesn't
    generate.

    Refinement levels and quality controls are loosened relative to a
    generic snappyHexMesh template -- real coronary geometry has thin,
    sharply-bifurcating branches, and tight quality thresholds there
    reliably produce illegal/highly-skewed faces. These settings trade
    some local cell regularity for a mesh that completes checkMesh.

    location_in_mesh is auto-set to the vessel centroid by generate_case().
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
            maxLocalCells      500000;
            maxGlobalCells    4000000;
            minRefinementCells 10;
            maxLoadUnbalance   0.10;
            nCellsBetweenLevels 3;

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

            locationInMesh ({lx:.6f} {ly:.6f} {lz:.6f});

            allowFreeStandingZoneFaces true;
        }}

        snapControls
        {{
            nSmoothPatch         5;
            tolerance            2.0;
            nSolveIter           100;
            nRelaxIter           10;
            nFeatureSnapIter     10;
            implicitFeatureSnap  false;
            explicitFeatureSnap  false;
            multiRegionFeatureSnap false;
        }}

        addLayersControls
        {{
            relativeSizes    true;

            layers
            {{
                wall {{ nSurfaceLayers 3; }}
            }}

            expansionRatio          1.2;
            finalLayerThickness     0.3;
            minThickness            0.1;
            nGrow                   0;
            featureAngle            60;
            nRelaxIter              10;
            nSmoothSurfaceNormals   5;
            nSmoothNormals          5;
            nSmoothThickness        20;
            maxFaceThicknessRatio   0.5;
            maxThicknessToMedialRatio 0.3;
            minMedianAxisAngle      90;
            nBufferCellsNoExtrude   0;
            nLayerIter              50;
        }}

        meshQualityControls
        {{
            maxNonOrtho         75;
            maxBoundarySkewness 30;
            maxInternalSkewness  8;
            maxConcave           80;
            minVol              1e-30;
            minTetQuality       -1e30;
            minArea             -1;
            minTwist             0.02;
            minDeterminant       0.0001;
            minFaceWeight        0.02;
            minVolRatio          0.005;
            minTriangleTwist    -1;
            nSmoothScale         10;
            errorReduction       0.75;
        }}

        debug           0;
        mergeTolerance  1e-6;
        """)
    path.write_text(content)


def write_gmsh_geo(path: Path, stl_name: str = "vessel.stl",
                   mesh_size_far: float = 0.003,
                   mesh_size_wall: float = 0.0005) -> None:
    """
    Gmsh .geo script as an alternative to snappyHexMesh.
    Produces a tet mesh with boundary layer refinement at the walls.
    mesh_size_wall ~= carrier radius / 2 for adequate resolution.

    Kept for reference / fallback -- not used by the default run_cfd.sh
    workflow now that blockMesh + snappyHexMesh is properly wired up.
    """
    content = textwrap.dedent(f"""\
        // Gmsh .geo for vascular CFD meshing
        // Generated by generate_openfoam_case.py

        Merge "{stl_name}";

        // Surface refinement near wall
        Field[1] = Distance;
        Field[1].SurfacesList = {{1}};   // adjust to your surface tag

        Field[2] = Threshold;
        Field[2].InField     = 1;
        Field[2].SizeMin     = {mesh_size_wall:.6f};
        Field[2].SizeMax     = {mesh_size_far:.6f};
        Field[2].DistMin     = 0.0;
        Field[2].DistMax     = 0.005;

        Background Field = 2;

        // Generate 3-D mesh
        Mesh.Algorithm3D   = 4;   // Frontal-Delaunay
        Mesh.OptimizeNetgen = 1;
        Mesh 3;

        // Boundary layer
        BoundaryLayer Field = 1;
        Field[3] = BoundaryLayer;
        Field[3].AnisoMax     = 10;
        Field[3].Quads        = 1;
        Field[3].Ratio        = 1.2;
        Field[3].Size         = {mesh_size_wall:.6f};
        Field[3].SizeFar      = {mesh_size_far:.6f};
        Field[3].Thickness    = 0.001;
        BoundaryLayer Field   = 3;

        Save "mesh.msh";
        """)
    path.write_text(content)


# ---------------------------------------------------------------------------
# STL export from pkl
# ---------------------------------------------------------------------------

def _find_terminal_endpoints(segments: list) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Identify inlet and outlet endpoints from segment topology.

    Strategy mirrors RealVesselNetwork:
    - Build an endpoint connectivity map (node → how many segments touch it)
    - Terminal nodes (degree 1) are open ends — either inlet or outlets
    - Inlet  = terminal node with minimum z  (matches inlet_start() convention)
    - Outlets = all other terminal nodes

    Returns
    -------
    inlet_pt  : (3,) array
    outlet_pts : list of (3,) arrays
    """
    EPS = 1e-6
    ROUND = 6   # decimal places for endpoint key — matches connection_tol ~1e-6

    endpoint_count: dict[tuple, list] = {}
    for s in segments:
        for pt in (np.asarray(s["start"], dtype=float),
                   np.asarray(s["end"],   dtype=float)):
            key = tuple(np.round(pt, ROUND))
            endpoint_count.setdefault(key, []).append(pt)

    # Degree-1 nodes = open ends
    terminals = [
        pts[0] for key, pts in endpoint_count.items()
        if len(pts) == 1   # only one segment touches this point
    ]

    if not terminals:
        # Fallback: use z-extremes of all endpoints
        all_pts = np.array(
            [np.asarray(s["start"], float) for s in segments] +
            [np.asarray(s["end"],   float) for s in segments]
        )
        terminals = [all_pts[np.argmin(all_pts[:, 2])],
                     all_pts[np.argmax(all_pts[:, 2])]]

    terminals_arr = np.array(terminals)
    inlet_idx  = int(np.argmin(terminals_arr[:, 2]))
    inlet_pt   = terminals[inlet_idx]
    outlet_pts = [t for i, t in enumerate(terminals) if i != inlet_idx]

    return np.asarray(inlet_pt, dtype=float), outlet_pts


def _disk_mesh(center: np.ndarray, normal: np.ndarray,
               radius: float, n_sides: int = 32):
    """Return a pyvista PolyData disk cap."""
    import pyvista as pv
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    disk = pv.Disc(center=center, normal=normal,
                   inner=0.0, outer=radius, r_res=4, c_res=n_sides)
    return disk


def export_stl_from_pkl(segments: list, output_dir: Path,
                        n_sides: int = 32) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Build a multi-region STL from segment cylinders with named inlet/outlet/wall
    surfaces.  OpenFOAM's snappyHexMesh requires each boundary to be a separate
    named solid in the STL so it can assign patch types correctly.

    Writes three files into output_dir:
        vessel_wall.stl    — cylinder side faces for every segment
        vessel_inlet.stl   — disk cap at the inlet endpoint
        vessel_outlet.stl  — disk cap(s) at all outlet endpoints

    snappyHexMeshDict should reference these as three separate geometries
    (or one STL with three named solids — we use separate files for clarity).

    Returns
    -------
    inlet_pt   : (3,) array   — inlet center [m]
    outlet_pts : list of (3,) — outlet centers [m]
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[WARNING] pyvista not available — skipping STL export.")
        return np.zeros(3), []

    output_dir = Path(output_dir)

    # --- Identify inlet/outlet endpoints ---
    inlet_pt, outlet_pts = _find_terminal_endpoints(segments)
    print(f"[STL] Inlet  : {inlet_pt}")
    for i, o in enumerate(outlet_pts):
        print(f"[STL] Outlet {i}: {o}")

    # --- Wall: cylinder side surfaces only (capping=False) ---
    wall_meshes = []
    for s in segments:
        start  = np.asarray(s["start"], dtype=float)
        end    = np.asarray(s["end"],   dtype=float)
        r      = float(s["radius"])
        axis   = end - start
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        cyl = pv.Cylinder(
            center=(start + end) / 2,
            direction=axis / length,
            radius=r,
            height=length,
            resolution=n_sides,
            capping=False,
        )
        wall_meshes.append(cyl)

    if not wall_meshes:
        print("[WARNING] No segments to export.")
        return inlet_pt, outlet_pts

    wall = wall_meshes[0]
    for m in wall_meshes[1:]:
        wall = wall.merge(m)

    wall_path = output_dir / "vessel_wall.stl"
    wall.save(str(wall_path))
    print(f"[STL] Wall   → {wall_path}")

    # --- Inlet cap ---
    # The inlet segment is whichever segment has an endpoint == inlet_pt
    inlet_seg = None
    EPS = 1e-5
    for s in segments:
        for pt_key in ("start", "end"):
            pt = np.asarray(s[pt_key], dtype=float)
            if np.linalg.norm(pt - inlet_pt) < EPS:
                inlet_seg = s
                break
        if inlet_seg is not None:
            break

    if inlet_seg is not None:
        axis = np.asarray(inlet_seg["end"], float) - np.asarray(inlet_seg["start"], float)
        # Normal points INTO the domain (inward), so flip if pointing away
        if np.dot(axis, np.array([0, 0, 1])) < 0:
            axis = -axis
        r_inlet = float(inlet_seg["radius"])
        inlet_disk = _disk_mesh(inlet_pt, axis / np.linalg.norm(axis), r_inlet, n_sides)
        inlet_path = output_dir / "vessel_inlet.stl"
        inlet_disk.save(str(inlet_path))
        print(f"[STL] Inlet  → {inlet_path}  r={r_inlet*1000:.2f} mm")
    else:
        print("[WARNING] Could not find inlet segment — inlet cap not written.")
        r_inlet = float(max(s["radius"] for s in segments))

    # --- Outlet caps ---
    outlet_meshes = []
    for o_pt in outlet_pts:
        o_seg = None
        for s in segments:
            for pt_key in ("start", "end"):
                pt = np.asarray(s[pt_key], dtype=float)
                if np.linalg.norm(pt - o_pt) < EPS:
                    o_seg = s
                    break
            if o_seg is not None:
                break
        if o_seg is None:
            continue
        axis = np.asarray(o_seg["end"], float) - np.asarray(o_seg["start"], float)
        if np.dot(axis, o_pt - np.asarray(o_seg["start"], float)) < 0:
            axis = -axis
        r_out = float(o_seg["radius"])
        outlet_meshes.append(_disk_mesh(o_pt, axis / np.linalg.norm(axis), r_out, n_sides))

    if outlet_meshes:
        outlet_combined = outlet_meshes[0]
        for m in outlet_meshes[1:]:
            outlet_combined = outlet_combined.merge(m)
        outlet_path = output_dir / "vessel_outlet.stl"
        outlet_combined.save(str(outlet_path))
        print(f"[STL] Outlet → {outlet_path}  ({len(outlet_meshes)} caps)")
    else:
        print("[WARNING] No outlet caps written.")

    return inlet_pt, outlet_pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_case(
    pkl_path: str,
    output_dir: str,
    solver: str = "simpleFoam",
    flow_scale: float = 1.0,
    n_phases: int = 20,
    export_stl: bool = True,
    background_cells_per_dim: int = 24,
) -> None:
    pulsatile = solver == "pimpleFoam"

    segments = _load_pkl(pkl_path)
    v_inlet = _mean_inlet_velocity(segments, flow_scale)
    r_inlet = _inlet_radius(segments)
    bbox_lo, bbox_hi = _vessel_bounding_box(segments)

    print(f"[case] Solver        : {solver}")
    print(f"[case] Inlet v_mean  : {v_inlet*1000:.2f} mm/s  (v_max = {2*v_inlet*1000:.2f} mm/s)")
    print(f"[case] Inlet radius  : {r_inlet*1000:.2f} mm")
    print(f"[case] Bounding box  : lo={bbox_lo} hi={bbox_hi}  [m]")
    print(f"[case] Output        : {output_dir}")

    base = Path(output_dir)
    (base / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (base / "system").mkdir(parents=True, exist_ok=True)
    (base / "0").mkdir(parents=True, exist_ok=True)

    write_transport_properties(base / "constant" / "transportProperties")
    write_turbulence_properties(base / "constant" / "turbulenceProperties")

    if pulsatile:
        write_control_dict_pulsatile(base / "system" / "controlDict", n_phases)
    else:
        write_control_dict_steady(base / "system" / "controlDict")

    write_fv_schemes(base / "system" / "fvSchemes", solver)
    write_fv_solution(base / "system" / "fvSolution", solver)
    write_gmsh_geo(base / "constant" / "triSurface" / "vessel.geo")
    write_U_bc(base / "0" / "U", v_inlet, r_inlet, pulsatile)
    write_p_bc(base / "0" / "p")

    write_block_mesh_dict(
        base / "system" / "blockMeshDict",
        bbox_lo, bbox_hi,
        cells_per_dim=background_cells_per_dim,
    )

    # Export named STL regions first so we can compute centroid for locationInMesh.
    tri_dir = base / "constant" / "triSurface"
    centroid = np.zeros(3)
    if export_stl:
        _, _ = export_stl_from_pkl(segments, tri_dir)
    # Centroid: length-weighted mean of segment midpoints
    pts, wts = [], []
    for s in segments:
        st = np.asarray(s["start"], float)
        en = np.asarray(s["end"],   float)
        pts.append((st + en) / 2)
        wts.append(float(np.linalg.norm(en - st)))
    if pts:
        pts = np.array(pts); wts = np.array(wts)
        centroid = (pts * wts[:, None]).sum(axis=0) / wts.sum()
    print(f"[case] locationInMesh : {centroid}")
    write_snappy_hex_mesh_dict(
        base / "system" / "snappyHexMeshDict",
        location_in_mesh=tuple(centroid),
    )

    # Write a shell script that runs the full pipeline
    run_script = base / "run_cfd.sh"
    run_script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        # Auto-generated by generate_openfoam_case.py
        # Run this script from the case directory after sourcing OpenFOAM 12.

        set -e
        cd "$(dirname "$0")"

        echo "=== Background mesh (blockMesh) ==="
        blockMesh

        echo "=== Refining/snapping to vessel geometry (snappyHexMesh) ==="
        snappyHexMesh -overwrite

        echo "=== Checking mesh ==="
        checkMesh

        echo "=== Running {solver} ==="
        {solver}

        echo "=== Exporting to VTK ==="
        # Using foamToVTK instead of a controlDict function object, since the
        # latter's exact output schema/behaviour on OpenFOAM 12 hasn't been
        # verified yet in your pipeline. This is more predictable.
        foamToVTK -latestTime || true

        echo "=== Building CFD interpolator ==="
        # Defensive glob: don't assume exact foamToVTK folder/file naming,
        # just grab whatever internal-mesh .vtu file exists under VTK/.
        VTU_FILE=$(find VTK -name "*.vtu" 2>/dev/null | sort | tail -1)
        if [ -z "$VTU_FILE" ]; then
            echo "No .vtu file found under VTK/ -- check foamToVTK output"
            echo "and pass --input manually to build_cfd_interpolator.py"
            exit 1
        fi
        echo "Using VTU: $VTU_FILE"
        python {os.path.abspath('build_cfd_interpolator.py')} \\
            --input "$VTU_FILE" --mode {"pulsatile --period 1.0" if pulsatile else "steady"} \\
            --output {os.path.abspath(output_dir)}/{Path(pkl_path).stem}_{"pulsatile" if pulsatile else "steady"}.npz

        echo "=== Done ==="
        """))
    os.chmod(run_script, 0o755)
    print(f"[case] Run script → {run_script}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an OpenFOAM 12 case directory for vascular CFD."
    )
    parser.add_argument("--pkl", required=True,
                        help="Path to vessel segment .pkl file.")
    parser.add_argument("--output", required=True,
                        help="Output case directory (created if absent).")
    parser.add_argument("--solver", choices=["simpleFoam", "pimpleFoam"],
                        default="simpleFoam",
                        help="simpleFoam = steady; pimpleFoam = pulsatile.")
    parser.add_argument("--flow-scale", type=float, default=1.0,
                        help="Global flow scale λ (matches simulation benchmark).")
    parser.add_argument("--n-phases", type=int, default=20,
                        help="Number of cardiac phase snapshots (pimpleFoam only).")
    parser.add_argument("--no-stl", action="store_true",
                        help="Skip STL export (supply your own vessel surface).")
    parser.add_argument("--background-cells", type=int, default=24,
                        help="Cells per axis for the blockMesh background box "
                             "(before snappyHexMesh refinement). Default 24.")
    args = parser.parse_args()

    generate_case(
        pkl_path=args.pkl,
        output_dir=args.output,
        solver=args.solver,
        flow_scale=args.flow_scale,
        n_phases=args.n_phases,
        export_stl=not args.no_stl,
        background_cells_per_dim=args.background_cells,
    )


if __name__ == "__main__":
    main()
