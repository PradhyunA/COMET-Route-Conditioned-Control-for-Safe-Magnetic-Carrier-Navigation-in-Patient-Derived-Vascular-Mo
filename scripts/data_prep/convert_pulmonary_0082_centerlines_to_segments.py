import pickle
import numpy as np
from pathlib import Path

try:
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.vtkCommonCore import vtkIdList
except Exception:
    from vtk import vtkXMLPolyDataReader, vtkIdList


CENTERLINES = Path("data/vessels/pulmonary_0082_H_PULM_H/centerlines_0082_H_PULM_H.vtp")
OUT = Path("data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments.pkl")

# SimVascular/VMTK geometry is usually in millimeters.
# Simulator expects meters.
COORD_SCALE = 0.01  # SimVascular centerlines are in cm
RADIUS_SCALE = 0.01

reader = vtkXMLPolyDataReader()
reader.SetFileName(str(CENTERLINES))
reader.Update()
poly = reader.GetOutput()

points = poly.GetPoints()
lines = poly.GetLines()
point_data = poly.GetPointData()

if points is None or lines is None:
    raise RuntimeError("No points or lines found in centerline VTP.")

radius_array = None
for name in ["MaximumInscribedSphereRadius", "Radius", "radius"]:
    arr = point_data.GetArray(name)
    if arr is not None:
        radius_array = arr
        print("Using radius array:", name)
        break

if radius_array is None:
    print("WARNING: No radius array found. Using fallback radius=1.0 mm.")

print("Reading:", CENTERLINES)
print("Points:", poly.GetNumberOfPoints())
print("Lines:", poly.GetNumberOfLines())
print("Cells:", poly.GetNumberOfCells())
print("Raw bounds:", poly.GetBounds())

segments = []
seg_id = 0
ids = vtkIdList()

lines.InitTraversal()
while lines.GetNextCell(ids):
    n = ids.GetNumberOfIds()
    if n < 2:
        continue

    for i in range(n - 1):
        p0_id = ids.GetId(i)
        p1_id = ids.GetId(i + 1)

        p0_raw = np.array(points.GetPoint(p0_id), dtype=float)
        p1_raw = np.array(points.GetPoint(p1_id), dtype=float)

        p0 = p0_raw * COORD_SCALE
        p1 = p1_raw * COORD_SCALE

        if radius_array is not None:
            r0 = float(radius_array.GetTuple1(p0_id))
            r1 = float(radius_array.GetTuple1(p1_id))
            radius = 0.5 * (r0 + r1) * RADIUS_SCALE
        else:
            radius = 1.0 * RADIUS_SCALE

        length = float(np.linalg.norm(p1 - p0))
        if length <= 1e-12:
            continue

        segments.append({
            "seg_id": seg_id,
            "start": p0,
            "end": p1,
            "radius": radius,
            "v_mean": 0.02,
            "level": 0,
        })
        seg_id += 1

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "wb") as f:
    pickle.dump(segments, f)

pts = []
radii = []
lengths = []

for s in segments:
    start = np.asarray(s["start"], dtype=float)
    end = np.asarray(s["end"], dtype=float)
    radius = float(s["radius"])

    pts.append(start)
    pts.append(end)
    radii.append(radius)
    lengths.append(np.linalg.norm(end - start))

pts = np.asarray(pts)
radii = np.asarray(radii)
lengths = np.asarray(lengths)

bbox_min = pts.min(axis=0)
bbox_max = pts.max(axis=0)
bbox_size = bbox_max - bbox_min

print()
print("Saved:", OUT)
print("Number of segments:", len(segments))
print("Bounding box min m:", bbox_min)
print("Bounding box max m:", bbox_max)
print("Bounding box size m:", bbox_size)
print("Bounding box size cm:", bbox_size * 100.0)
print("Bounding box size mm:", bbox_size * 1000.0)
print("Max spatial extent cm:", bbox_size.max() * 100.0)
print("Total segment length m:", lengths.sum())
print("Total segment length cm:", lengths.sum() * 100.0)
print("Radius range cm:", radii.min() * 100.0, "to", radii.max() * 100.0)
print("Mean radius cm:", radii.mean() * 100.0)
