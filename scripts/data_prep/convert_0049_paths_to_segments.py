"""
Convert the VMR 0049_H_ABAO_AIOD SimVascular paths and segmentations into the
DynaMag vessel segment PKL format.

Input:
    data/vessels/aortofemoral_0049_H_ABAO_AIOD/Paths/*.pth
    data/vessels/aortofemoral_0049_H_ABAO_AIOD/Segmentations/*.ctgr

Output:
    data/vessels/aortofemoral_0049_H_ABAO_AIOD/real_vessel_segments.pkl

Same centerline and radius treatment as convert_0024_paths_to_segments.py.
Paths here branch off other branches (aorta -> SMA -> SMA_br1 -> SMA_br1_br1,
graft -> profunda -> circumflex), so paths are added greedily: the next path
is the one whose first point lies deepest inside the lumen already placed, and
it joins that lumen's centerline at the point nearest to its first point
outside the lumen.
"""

import pickle
from pathlib import Path

import numpy as np

from convert_0024_paths_to_segments import (
    COORD_SCALE, DEFAULT_V_MEAN, _parse, _polygon_area, _xyz,
)

ROOT = Path("data/vessels/aortofemoral_0049_H_ABAO_AIOD")
OUT = ROOT / "real_vessel_segments.pkl"

MAIN = "aorta"
# Path files whose names differ from their segmentation's path_name.
PATH_FILE_FOR = {"rt_profunda_femoris": "rt_"}


def read_path(name):
    root = _parse(ROOT / "Paths" / f"{PATH_FILE_FOR.get(name, name)}.pth")
    points = root.find(".//path_points")
    return np.array([_xyz(p.find("pos")) for p in points.findall("path_point")]) * COORD_SCALE


def read_radii(name, n_points):
    """Equivalent-area radius (m) at every path point of the named path."""
    root = _parse(ROOT / "Segmentations" / f"{name}.ctgr")
    ids, radii = [], []
    for contour in root.iter("contour"):
        pts = np.array([_xyz(p) for p in contour.find("contour_points").findall("point")])
        ids.append(int(contour.find("path_point").get("id")))
        radii.append(np.sqrt(_polygon_area(pts * COORD_SCALE) / np.pi))
    order = np.argsort(ids)
    return np.interp(np.arange(n_points), np.asarray(ids)[order], np.asarray(radii)[order])


def main():
    names = sorted(p.stem for p in (ROOT / "Segmentations").glob("*.ctgr"))
    centerlines = {}
    for name in names:
        pts = read_path(name)
        centerlines[name] = (pts, read_radii(name, len(pts)))

    placed_pts = centerlines[MAIN][0]
    placed_r = centerlines[MAIN][1]
    polylines = [centerlines[MAIN]]
    remaining = [name for name in names if name != MAIN]

    while remaining:
        def depth(name):
            d = np.linalg.norm(placed_pts - centerlines[name][0][0], axis=1)
            return float((d - placed_r).min())

        name = min(remaining, key=depth)
        remaining.remove(name)
        pts, radii = centerlines[name]
        gap = np.linalg.norm(pts[:, None, :] - placed_pts[None, :, :], axis=2)
        outside = np.nonzero((gap - placed_r[None, :]).min(axis=1) > 0.0)[0]
        if len(outside) == 0:
            print(f"{name}: never leaves the placed lumen, skipped")
            continue
        j0 = int(outside[0])
        nearest = int(gap[j0].argmin())
        polylines.append((
            np.vstack([placed_pts[nearest], pts[j0:]]),
            np.concatenate([[radii[j0]], radii[j0:]]),
        ))
        print(f"{name}: first point {depth(name) * 1e3:+.2f} mm from placed lumen, "
              f"dropped {j0} points, join length {gap[j0, nearest] * 1e3:.2f} mm")
        placed_pts = np.vstack([placed_pts, pts[j0:]])
        placed_r = np.concatenate([placed_r, radii[j0:]])

    segments = []
    for pts, radii in polylines:
        for i in range(len(pts) - 1):
            if np.linalg.norm(pts[i + 1] - pts[i]) <= 1e-12:
                continue
            segments.append({
                "seg_id": len(segments),
                "start": pts[i].copy(),
                "end": pts[i + 1].copy(),
                "radius": float(0.5 * (radii[i] + radii[i + 1])),
                "v_mean": DEFAULT_V_MEAN,
                "level": 0,
            })

    with open(OUT, "wb") as f:
        pickle.dump(segments, f)

    ends = np.array([s["start"] for s in segments] + [s["end"] for s in segments])
    radii = np.array([s["radius"] for s in segments])
    print("Saved:", OUT)
    print("Segments:", len(segments))
    print("Extent mm:", np.round((ends.max(0) - ends.min(0)) * 1000.0, 1))
    print("Radius mm: min %.2f median %.2f max %.2f" % tuple(1000.0 * np.percentile(radii, [0, 50, 100])))


if __name__ == "__main__":
    main()
