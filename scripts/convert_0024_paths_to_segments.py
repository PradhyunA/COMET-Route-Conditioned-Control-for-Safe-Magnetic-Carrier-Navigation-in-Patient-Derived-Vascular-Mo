"""
Convert the VMR 0024_H_AO_H SimVascular paths and segmentations into the
DynaMag vessel segment PKL format.

Input:
    data/vessels/aorta_0024_H_AO_H/Paths/*.pth
    data/vessels/aorta_0024_H_AO_H/Segmentations/*_final.ctgr

Output:
    data/vessels/aorta_0024_H_AO_H/real_vessel_segments.pkl

The centerline is the dense path_points of each SimVascular path. The radius
at each lofted contour is the equivalent-area radius sqrt(A / pi) of the
contour polygon, interpolated linearly along the path between contours and
held constant past the first and last contour. SimVascular units are cm.

Each branch (RSA, RCCA, LCCA, LSA) is joined to the aorta at the aorta
centerline point nearest to the first branch point outside the aortic lumen,
so flow and routes run continuously from the aortic root into the branches.
No vtk/vmtk install is needed.
"""

import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("data/vessels/aorta_0024_H_AO_H")
OUT = ROOT / "real_vessel_segments.pkl"

COORD_SCALE = 0.01  # SimVascular cm -> m
DEFAULT_V_MEAN = 0.02  # m/s, matches the other benchmark anatomies
MAIN = "Aorta"
BRANCHES = ("RSA", "RCCA", "LCCA", "LSA")


def _parse(path):
    # SimVascular files hold a <format/> element next to the main element.
    text = path.read_text()
    text = text[text.index("?>") + 2:] if text.lstrip().startswith("<?xml") else text
    return ET.fromstring(f"<root>{text}</root>")


def _xyz(elem):
    return np.array([float(elem.get(k)) for k in ("x", "y", "z")])


def read_path(name):
    root = _parse(ROOT / "Paths" / f"{name}.pth")
    points = root.find(".//path_points")
    return np.array([_xyz(p.find("pos")) for p in points.findall("path_point")]) * COORD_SCALE


def _polygon_area(points):
    centred = points - points.mean(axis=0)
    return 0.5 * np.linalg.norm(np.cross(centred, np.roll(centred, -1, axis=0)).sum(axis=0))


def read_radii(name, n_points):
    """Equivalent-area radius (m) at every path point of the named path."""
    root = _parse(ROOT / "Segmentations" / f"{name}_final.ctgr")
    ids, radii = [], []
    for contour in root.iter("contour"):
        pts = np.array([_xyz(p) for p in contour.find("contour_points").findall("point")])
        ids.append(int(contour.find("path_point").get("id")))
        radii.append(np.sqrt(_polygon_area(pts * COORD_SCALE) / np.pi))
    order = np.argsort(ids)
    return np.interp(np.arange(n_points), np.asarray(ids)[order], np.asarray(radii)[order])


def main():
    centerlines = {}
    for name in (MAIN,) + BRANCHES:
        pts = read_path(name)
        centerlines[name] = (pts, read_radii(name, len(pts)))

    aorta_pts, aorta_r = centerlines[MAIN]
    polylines = [(aorta_pts, aorta_r)]
    for name in BRANCHES:
        pts, radii = centerlines[name]
        dist = np.linalg.norm(pts[:, None, :] - aorta_pts[None, :, :], axis=2)
        nearest = dist.argmin(axis=1)
        outside = np.nonzero(dist[np.arange(len(pts)), nearest] > aorta_r[nearest])[0]
        if len(outside) == 0:
            raise RuntimeError(f"Branch {name} never leaves the aortic lumen")
        j0 = int(outside[0])
        junction = aorta_pts[nearest[j0]]
        polylines.append((
            np.vstack([junction, pts[j0:]]),
            np.concatenate([[radii[j0]], radii[j0:]]),
        ))
        print(f"{name}: joined at aorta point {nearest[j0]} (dropped {j0} points inside the aorta)")

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
