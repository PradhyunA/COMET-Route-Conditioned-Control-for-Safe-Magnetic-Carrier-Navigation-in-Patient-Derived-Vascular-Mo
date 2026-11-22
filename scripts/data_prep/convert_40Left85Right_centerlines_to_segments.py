"""
Convert 40Left85Right VMTK centerlines into DynaMag vessel segment PKL format.

Input:
    data/vessels/aortic_root_0248_H_AOCERE_CAS_40Left85Right/centerlines_40Left85Right.vtp

Output:
    data/vessels/aortic_root_0248_H_AOCERE_CAS_40Left85Right/real_vessel_segments.pkl

Important:
    SCALE_TO_METERS = 0.01, meaning:
        1 raw VTP coordinate unit = 1 cm = 0.01 m
"""

import os
import pickle
import numpy as np
import vtk


INPUT_VTP = "data/vessels/aortic_root_0248_H_AOCERE_CAS_40Left85Right/centerlines_40Left85Right.vtp"
OUTPUT_PKL = "data/vessels/aortic_root_0248_H_AOCERE_CAS_40Left85Right/real_vessel_segments.pkl"

# Raw VTP unit -> meters.
# Your VTP is interpreted as centimeters.
SCALE_TO_METERS = 0.01

# Default mean flow speed for synthetic/project vessel segments.
DEFAULT_V_MEAN = 0.02  # m/s


def get_radius_array(poly):
    pd = poly.GetPointData()

    for name in [
        "MaximumInscribedSphereRadius",
        "Radius",
        "radius",
    ]:
        arr = pd.GetArray(name)
        if arr is not None:
            return arr, name

    raise RuntimeError(
        "No radius array found. Expected MaximumInscribedSphereRadius or Radius."
    )


def convert():
    if not os.path.exists(INPUT_VTP):
        raise FileNotFoundError(f"Missing input file: {INPUT_VTP}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(INPUT_VTP)
    reader.Update()
    poly = reader.GetOutput()

    n_points = poly.GetNumberOfPoints()
    n_lines = poly.GetNumberOfLines()
    n_cells = poly.GetNumberOfCells()

    print(f"Reading: {INPUT_VTP}")
    print(f"Points: {n_points}")
    print(f"Lines: {n_lines}")
    print(f"Cells: {n_cells}")
    print(f"Raw bounds: {poly.GetBounds()}")

    radius_arr, radius_name = get_radius_array(poly)
    print(f"Using radius array: {radius_name}")

    segments = []
    seg_id = 0

    lines = poly.GetLines()
    lines.InitTraversal()

    id_list = vtk.vtkIdList()
    line_index = 0

    while lines.GetNextCell(id_list):
        num_ids = id_list.GetNumberOfIds()

        if num_ids < 2:
            line_index += 1
            continue

        for j in range(num_ids - 1):
            pid0 = id_list.GetId(j)
            pid1 = id_list.GetId(j + 1)

            p0_raw = np.array(poly.GetPoint(pid0), dtype=np.float32)
            p1_raw = np.array(poly.GetPoint(pid1), dtype=np.float32)

            p0 = p0_raw * SCALE_TO_METERS
            p1 = p1_raw * SCALE_TO_METERS

            r0 = float(radius_arr.GetTuple1(pid0)) * SCALE_TO_METERS
            r1 = float(radius_arr.GetTuple1(pid1)) * SCALE_TO_METERS
            radius = float(0.5 * (r0 + r1))

            length = float(np.linalg.norm(p1 - p0))

            if length <= 1e-12:
                continue

            segments.append(
                {
                    "seg_id": seg_id,
                    "start": p0.astype(np.float32),
                    "end": p1.astype(np.float32),
                    "radius": radius,
                    "v_mean": DEFAULT_V_MEAN,
                    "level": int(line_index),
                }
            )

            seg_id += 1

        line_index += 1

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(segments, f)

    starts = np.array([s["start"] for s in segments], dtype=np.float32)
    ends = np.array([s["end"] for s in segments], dtype=np.float32)
    pts = np.vstack([starts, ends])

    radii = np.array([s["radius"] for s in segments], dtype=float)
    lengths = np.array(
        [np.linalg.norm(s["end"] - s["start"]) for s in segments],
        dtype=float,
    )

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print()
    print(f"Saved: {OUTPUT_PKL}")
    print(f"Number of segments: {len(segments)}")
    print(f"Bounding box min (m): {bbox_min}")
    print(f"Bounding box max (m): {bbox_max}")
    print(f"Bounding box size (m): {bbox_size}")
    print(f"Bounding box size (cm): {bbox_size * 100.0}")
    print(f"Max spatial extent (cm): {bbox_size.max() * 100.0}")
    print(f"Total segment length (m): {lengths.sum()}")
    print(f"Total segment length (cm): {lengths.sum() * 100.0}")
    print(f"Radius range (m): {radii.min()} to {radii.max()}")
    print(f"Radius range (cm): {radii.min() * 100.0} to {radii.max() * 100.0}")
    print(f"Mean radius (cm): {radii.mean() * 100.0}")


if __name__ == "__main__":
    convert()
