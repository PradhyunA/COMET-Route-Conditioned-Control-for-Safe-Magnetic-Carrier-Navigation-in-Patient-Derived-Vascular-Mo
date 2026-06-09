import pickle
import numpy as np

with open("data/real_vessel_segments.pkl", "rb") as f:
    segments = pickle.load(f)

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

pts = np.array(pts)
radii = np.array(radii)
lengths = np.array(lengths)

bbox_min = pts.min(axis=0)
bbox_max = pts.max(axis=0)
bbox_size = bbox_max - bbox_min

print("Number of segments:", len(segments))
print("Bounding box min (m):", bbox_min)
print("Bounding box max (m):", bbox_max)
print("Bounding box size (m):", bbox_size)
print("Bounding box size (mm):", bbox_size * 1000)

print("Max spatial extent (mm):", np.max(bbox_size) * 1000)
print("Total centerline length (m):", lengths.sum())
print("Total centerline length (mm):", lengths.sum() * 1000)

print("Radius range (mm):", radii.min() * 1000, "to", radii.max() * 1000)
print("Mean radius (mm):", radii.mean() * 1000)