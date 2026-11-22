#!/usr/bin/env python3

import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_vtp(path: str):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTP or file has no points: {path}")

    return poly


def get_points(poly):
    points_vtk = poly.GetPoints().GetData()
    return vtk_to_numpy(points_vtk).astype(float)


def get_radius_array(poly):
    point_data = poly.GetPointData()

    preferred_names = [
        "MaximumInscribedSphereRadius",
        "Radius",
        "radius",
    ]

    for name in preferred_names:
        arr = point_data.GetArray(name)
        if arr is not None:
            return name, vtk_to_numpy(arr).astype(float)

    names = []
    for i in range(point_data.GetNumberOfArrays()):
        names.append(point_data.GetArrayName(i))

    return None, None


def compute_polyline_length(poly, pts):
    total = 0.0
    lines = poly.GetLines()
    lines.InitTraversal()

    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n < 2:
            continue

        for i in range(n - 1):
            p0 = pts[id_list.GetId(i)]
            p1 = pts[id_list.GetId(i + 1)]
            total += np.linalg.norm(p1 - p0)

    return total


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python inspect_centerlines_vtp.py path/to/centerlines.vtp")
        sys.exit(1)

    path = sys.argv[1]
    poly = read_vtp(path)
    pts = get_points(poly)

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_size = bbox_max - bbox_min

    num_points = poly.GetNumberOfPoints()
    num_lines = poly.GetNumberOfLines()
    num_cells = poly.GetNumberOfCells()

    total_length = compute_polyline_length(poly, pts)

    radius_name, radii = get_radius_array(poly)

    print("File:", path)
    print("Number of points:", num_points)
    print("Number of lines:", num_lines)
    print("Number of cells:", num_cells)

    print("Bounding box min:", bbox_min)
    print("Bounding box max:", bbox_max)
    print("Bounding box size:", bbox_size)
    print("Bounding box size (mm):", bbox_size * 1000.0)
    print("Max spatial extent (mm):", np.max(bbox_size) * 1000.0)

    print("Total centerline length:", total_length)
    print("Total centerline length (mm):", total_length * 1000.0)

    if radii is not None:
        print("Radius array:", radius_name)
        print("Radius range (mm):", radii.min() * 1000.0, "to", radii.max() * 1000.0)
        print("Mean radius (mm):", radii.mean() * 1000.0)
    else:
        print("Radius array: NOT FOUND")
        print("Available point-data arrays:")
        point_data = poly.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            print("  -", point_data.GetArrayName(i))


if __name__ == "__main__":
    main()
