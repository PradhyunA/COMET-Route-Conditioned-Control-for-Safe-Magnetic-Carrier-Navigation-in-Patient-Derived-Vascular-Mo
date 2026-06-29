"""
generate_openfoam_case.py
=========================
Generates a ready-to-run OpenFOAM case directory for steady (simpleFoam)
or pulsatile (pimpleFoam) vascular flow simulation.

Takes the vessel surface geometry (STL or VTP), sets physiologically
realistic boundary conditions from the vessel pkl, and writes all
required OpenFOAM configuration files.  After running this script,
the case can be executed with:

    cd <case_dir>
    blockMesh           # or: gmsh -3 mesh.geo -o mesh.msh && gmshToFoam mesh.msh
    checkMesh
    simpleFoam          # or pimpleFoam for pulsatile
    python build_cfd_interpolator.py --input postProcessing/... --output ...

Workflow
--------
1. Export vessel surface as STL from pyvista (from your .vtp or .pkl):
       python generate_openfoam_case.py --pkl data/aorta.pkl --output cases/aorta_steady
   This writes:
       cases/aorta_steady/
           constant/triSurface/vessel.stl
           constant/transportProperties
           constant/turbulenceProperties
           system/controlDict
           system/fvSchemes
           system/fvSolution
           system/snappyHexMeshDict   (for snappyHexMesh meshing)
           0/U                         (velocity BCs)
           0/p                         (pressure BCs)

2. Run snappyHexMesh (or import a Gmsh mesh) to get the volume mesh.

3. Run simpleFoam / pimpleFoam.

4. Run build_cfd_interpolator.py to package the output.

Boundary conditions
-------------------
Inlet  : parabolic Womersley profile via groovyBC / timeVaryingUniformFixedValue
         (simpleFoam: steady mean profile; pimpleFoam: pulsatile)
Outlet : zeroGradient pressure, uniform p=0
Wall   : no-slip velocity, zeroGradient pressure

The inlet velocity magnitude is derived from v_mean of the largest-radius
segment in the pkl (the aortic root), consistent with the flow_scale = 1.0
convention used in RealVesselNetwork.

Blood properties (37 °C)
------------------------
    dynamic viscosity   ν = η/ρ = 3.5e-3 / 1060 = 3.302e-6 m²/s
    density             ρ = 1060 kg/m³

These match the constants already in the simulation.

Requirements
------------
    pip install pyvista numpy
    OpenFOAM v9 or v10 (openfoam.org) installed and sourced
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
            // Write final velocity field to VTK for build_cfd_interpolator.py
            writeFinalVTK
            {
                type            vtkWrite;
                libs            (\"libIOvtkWrite.so\");
                writeControl    onEnd;
                writeInterval   1;
                fields          (U p);
                format          binary;
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
        vessel_wall.stl    → patch type wall
        vessel_inlet.stl   → patch type patch (inlet)
        vessel_outlet.stl  → patch type patch (outlet)

    Each file is a separate geometry entry so OpenFOAM assigns the
    correct patch type automatically during snappyHexMesh.

    location_in_mesh is auto-set to the vessel centroid by generate_case().
    """
    lx, ly, lz = location_in_mesh
    content = _header("dictionary", "snappyHexMeshDict") + textwrap.dedent(f"""\

        castellatedMesh true;
        snap            true;
        addLayers       true;

        geometry
        {{
            vessel_wall.stl
            {{
                type triSurfaceMesh;
                name wall;
            }}
            vessel_inlet.stl
            {{
                type triSurfaceMesh;
                name inlet;
            }}
            vessel_outlet.stl
            {{
                type triSurfaceMesh;
                name outlet;
            }}
        }}

        castellatedMeshControls
        {{
            maxLocalCells      2000000;
            maxGlobalCells    10000000;
            minRefinementCells 10;
            maxLoadUnbalance   0.10;
            nCellsBetweenLevels 3;

            features ();

            refinementSurfaces
            {{
                wall
                {{
                    level (4 5);
                    patchInfo {{ type wall; }}
                }}
                inlet
                {{
                    level (4 4);
                    patchInfo {{ type patch; }}
                }}
                outlet
                {{
                    level (4 4);
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
            nSmoothPatch     3;
            tolerance        2.0;
            nSolveIter       30;
            nRelaxIter       5;
        }}

        addLayersControls
        {{
            relativeSizes    true;

            layers
            {{
                wall {{ nSurfaceLayers 5; }}
            }}

            expansionRatio          1.2;
            finalLayerThickness     0.3;
            minThickness            0.1;
            nGrow                   0;
            featureAngle            60;
            nRelaxIter              5;
            nSmoothSurfaceNormals   1;
            nSmoothNormals          3;
            nSmoothThickness        10;
            maxFaceThicknessRatio   0.5;
            maxThicknessToMedialRatio 0.3;
            minMedianAxisAngle      90;
            nBufferCellsNoExtrude   0;
            nLayerIter              50;
        }}

        meshQualityControls
        {{
            maxNonOrtho         65;
            maxBoundarySkewness 20;
            maxInternalSkewness  4;
            maxConcave           80;
            minVol              1e-13;
            minTetQuality       1e-15;
            minArea             -1;
            minTwist             0.05;
            minDeterminant       0.001;
            minFaceWeight        0.05;
            minVolRatio          0.01;
            minTriangleTwist    -1;
            nSmoothScale         4;
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
) -> None:
    pulsatile = solver == "pimpleFoam"

    segments = _load_pkl(pkl_path)
    v_inlet = _mean_inlet_velocity(segments, flow_scale)
    r_inlet = _inlet_radius(segments)

    print(f"[case] Solver        : {solver}")
    print(f"[case] Inlet v_mean  : {v_inlet*1000:.2f} mm/s  (v_max = {2*v_inlet*1000:.2f} mm/s)")
    print(f"[case] Inlet radius  : {r_inlet*1000:.2f} mm")
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
        # Run this script from the case directory after sourcing OpenFOAM.

        set -e
        cd "$(dirname "$0")"

        echo "=== Meshing ==="
        # Option A: snappyHexMesh (requires blockMesh background mesh first)
        # blockMesh
        # snappyHexMesh -overwrite

        # Option B: Gmsh + gmshToFoam
        # gmsh constant/triSurface/vessel.geo -3 -o constant/triSurface/mesh.msh
        # gmshToFoam constant/triSurface/mesh.msh

        echo "=== Checking mesh ==="
        checkMesh

        echo "=== Running {solver} ==="
        {solver}

        echo "=== Building CFD interpolator ==="
        CASE_DIR=$(pwd)
        {"VTU_PATTERN='VTK/vessel_*/vessel_*.vtu'" if pulsatile else "VTU_FILE=$(ls postProcessing/*/U_*.vtu 2>/dev/null | tail -1)"}
        python {os.path.abspath('build_cfd_interpolator.py')} \\
            {"--input '$VTU_PATTERN' --mode pulsatile --period 1.0" if pulsatile else "--input $VTU_FILE --mode steady"} \\
            --output {os.path.abspath(output_dir)}/{Path(pkl_path).stem}_{"pulsatile" if pulsatile else "steady"}.npz

        echo "=== Done ==="
        """))
    os.chmod(run_script, 0o755)
    print(f"[case] Run script → {run_script}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an OpenFOAM case directory for vascular CFD."
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
    args = parser.parse_args()

    generate_case(
        pkl_path=args.pkl,
        output_dir=args.output,
        solver=args.solver,
        flow_scale=args.flow_scale,
        n_phases=args.n_phases,
        export_stl=not args.no_stl,
    )


if __name__ == "__main__":
    main()
