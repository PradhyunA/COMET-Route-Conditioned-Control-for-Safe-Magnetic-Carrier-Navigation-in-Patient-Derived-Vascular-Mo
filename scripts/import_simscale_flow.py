"""Convert an ASCII OpenFOAM ZIP to a steady SI CFD field (no VTK required)."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRANSLATE = str.maketrans({"(": " ", ")": " "})


def read_text(archive, name):
    if name not in archive.namelist():
        name += ".gz"
    raw = archive.read(name)
    text = (gzip.decompress(raw) if name.endswith(".gz") else raw).decode("ascii")
    if not re.search(r"format\s+ascii\s*;", text):
        raise ValueError(f"Only ASCII OpenFOAM files are supported: {name}")
    return text


def list_body(text, field=False):
    pattern = (r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(" if field
               else r"\n\s*(\d+)\s*\n\(\s*\n")
    match = re.search(pattern, text)
    if match is None:
        raise ValueError("Expected a nonuniform ASCII list")
    end = text.find("\n)", match.end())
    if end < 0:
        raise ValueError("Unterminated ASCII list")
    return int(match[1]), text[match.end():end]


def parse_array(text, vector=False, field=False, dtype=float):
    count, body = list_body(text, field)
    values = np.fromstring(body.translate(TRANSLATE), sep=" ", dtype=dtype)
    if vector:
        values = values.reshape(-1, 3)
    if len(values) != count:
        raise ValueError("OpenFOAM list count mismatch")
    return values


def parse_faces(text):
    count, body = list_body(text)
    sizes = np.fromiter((int(m[1]) for m in re.finditer(r"(\d+)\(", body)), dtype=np.int32)
    vertices = np.fromstring(re.sub(r"\d+\(", "", body).replace(")", " "), sep=" ", dtype=np.int32)
    offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    if len(sizes) != count or np.any(sizes < 3) or offsets[-1] != len(vertices):
        raise ValueError("Invalid OpenFOAM faces")
    return sizes, offsets, vertices


def parse_patches(text):
    result = {}
    for name, body in re.findall(r"(\w+)\s*\{([^{}]*startFace[^{}]*)\}", text):
        result[name] = dict(type=re.search(r"type\s+(\w+);", body)[1],
                            start=int(re.search(r"startFace\s+(\d+);", body)[1]),
                            count=int(re.search(r"nFaces\s+(\d+);", body)[1]))
    if not result:
        raise ValueError("No boundary patches found")
    return result


def boundary_values(text, name, count, vector=False):
    match = re.search(r"\b" + re.escape(name) + r"\s*\{([^{}]*)\}", text)
    if match is None:
        raise ValueError(f"Missing boundary field: {name}")
    body = match[1]
    uniform = re.search(r"\bvalue\s+uniform\s+([^;]+);", body)
    if uniform:
        value = np.fromstring(uniform[1].translate(TRANSLATE), sep=" ")
        return np.broadcast_to(value, (count, 3)).copy() if vector else np.full(count, value[0])
    start = re.search(r"\bvalue\s+nonuniform", body)
    if start is None:
        raise ValueError(f"Unsupported boundary field: {name}")
    values = parse_array("internalField " + body[start.start() + len("value "):], vector, True)
    if len(values) != count:
        raise ValueError("Boundary field count mismatch")
    return values


def face_geometry(points, sizes, offsets, vertices):
    """Area-weighted polygon centroids, area vectors, and bounding radii."""
    centres = np.empty((len(sizes), 3))
    areas = np.empty_like(centres)
    radii = np.empty(len(sizes))
    for start in range(0, len(sizes), 100000):
        stop = min(start + 100000, len(sizes))
        for size in np.unique(sizes[start:stop]):
            rows = np.flatnonzero(sizes[start:stop] == size) + start
            polygon = points[vertices[offsets[rows, None] + np.arange(size)]]
            first = polygon[:, 0, :].copy()
            area_vector = np.zeros((len(rows), 3))
            moment = np.zeros_like(area_vector)
            total_area = np.zeros(len(rows))
            for edge in range(1, int(size) - 1):
                a = polygon[:, edge, :].copy()
                b = polygon[:, edge + 1, :].copy()
                edge_a = a - first
                edge_b = b - first
                vector = np.cross(edge_a, edge_b) * 0.5
                area = np.linalg.norm(vector, axis=1)
                area_vector += vector
                triangle_centre = (first + a + b) / 3.0
                moment += area[:, None] * triangle_centre
                total_area += area
            if np.any(total_area <= 0):
                raise ValueError("Zero-area mesh face")
            centres[rows] = moment / total_area[:, None]
            areas[rows] = area_vector
            radii[rows] = np.linalg.norm(polygon - centres[rows, None], axis=2).max(axis=1)
    if np.any(centres < points.min(axis=0) - 1e-12) or np.any(centres > points.max(axis=0) + 1e-12):
        raise ValueError("Computed face centroid lies outside the mesh bounds")
    return centres, areas, radii


def cell_geometry(fc, sf, fr, owner, neighbour, count):
    """Volume-weighted face-pyramid centroids, retaining OpenFOAM cell order."""
    internal = len(neighbour)
    face_count = np.bincount(owner, minlength=count) + np.bincount(neighbour, minlength=count)
    if np.any(face_count == 0):
        raise ValueError("Mesh cell has no faces")
    estimate = np.empty((count, 3))
    for axis in range(3):
        estimate[:, axis] = (np.bincount(owner, weights=fc[:, axis], minlength=count)
                            + np.bincount(neighbour, weights=fc[:internal, axis], minlength=count)) / face_count
    volumes = np.zeros(count)
    moments = np.zeros((count, 3))
    for labels, faces, sign in [(owner, slice(None), 1), (neighbour, slice(0, internal), -1)]:
        pyramid = sign * np.einsum("ij,ij->i", sf[faces], fc[faces] - estimate[labels]) / 3
        volumes += np.bincount(labels, weights=pyramid, minlength=count)
        for axis in range(3):
            centroid = 0.75 * fc[faces, axis] + 0.25 * estimate[labels, axis]
            moments[:, axis] += np.bincount(labels, weights=pyramid * centroid, minlength=count)
    if np.any(volumes <= 0) or not np.all(np.isfinite(volumes)):
        bad = np.flatnonzero(volumes <= 0)
        raise ValueError(f"Non-positive mesh cell volumes: {len(bad)}/{count}; "
                         f"minimum={volumes.min():g}; first cells={bad[:10].tolist()}")
    centres = moments / volumes[:, None]
    radii = np.zeros(count)
    np.maximum.at(radii, owner, np.linalg.norm(fc - centres[owner], axis=1) + fr)
    np.maximum.at(radii, neighbour, np.linalg.norm(fc[:internal] - centres[neighbour], axis=1) + fr[:internal])
    return centres, radii, volumes


def check_geometry(points, segments):
    ends = np.array([s[key] for s in segments for key in ["start", "end"]])
    span_ratio = np.ptp(points, axis=0) / np.ptp(ends, axis=0)
    if np.any(span_ratio < 0.8) or np.any(span_ratio > 1.6):
        raise ValueError(f"CFD/centreline scale mismatch (axis span ratios {span_ratio}). Rerun CFD at the correct size.")
    mids = np.array([(np.asarray(s["start"]) + s["end"]) / 2 for s in segments], dtype=float)
    radius = np.array([s["radius"] for s in segments])
    distance, _ = cKDTree(points).query(mids)
    score = float(np.percentile(distance / radius, 95))
    if score > 0.75:
        raise ValueError(f"CFD mesh is not aligned with centreline (p95 distance/radius {score:g})")
    return {"mesh_to_centreline_span_ratio": span_ratio.tolist(), "p95_nearest_vertex_distance_over_radius": score}


def import_case(zip_path, vessel_file, output, time_name=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(vessel_file, "rb") as stream:
        segments = pickle.load(stream)
    with zipfile.ZipFile(zip_path) as archive:
        if time_name is None:
            times = {n.split("/")[0] for n in archive.namelist() if re.fullmatch(r"[\d.eE+-]+/U(?:\.gz)?", n)}
            time_name = max(times, key=float)
        control = read_text(archive, "system/controlDict")
        if not re.search(r"application\s+(simscaleSimpleFoam|simpleFoam);", control):
            raise ValueError("This importer requires a steady simpleFoam solution")
        print(f"Reading SI mesh and velocity at solver iteration {time_name}", flush=True)
        points = parse_array(read_text(archive, "constant/polyMesh/points"), True)
        alignment = check_geometry(points, segments)
        sizes, offsets, vertices = parse_faces(read_text(archive, "constant/polyMesh/faces"))
        owner = parse_array(read_text(archive, "constant/polyMesh/owner"), dtype=np.int32)
        neighbour = parse_array(read_text(archive, "constant/polyMesh/neighbour"), dtype=np.int32)
        u_text = read_text(archive, f"{time_name}/U")
        if not re.search(r"dimensions\s+\[0 1 -1 0 0 0 0\]", u_text):
            raise ValueError("Expected velocity dimensions m/s")
        u = parse_array(u_text, True, True)
        patches = parse_patches(read_text(archive, "constant/polyMesh/boundary"))
        if (len(owner) != len(sizes) or owner.min() < 0 or owner.max() != len(u) - 1
                or neighbour.min() < 0 or neighbour.max() >= len(u)
                or vertices.min() < 0 or vertices.max() >= len(points)
                or not np.all(np.isfinite(u))):
            raise ValueError("Invalid mesh connectivity or velocity values")
        print(f"Computing centroids for {len(u):,} cells", flush=True)
        fc, sf, fr = face_geometry(points, sizes, offsets, vertices)
        cc, cr, volume = cell_geometry(fc, sf, fr, owner, neighbour, len(u))
        phi_text = read_text(archive, f"{time_name}/phi")
        flux_dimension = re.search(r"dimensions\s+\[([^]]+)\]", phi_text)[1]
        if flux_dimension not in {"1 0 -1 0 0 0 0", "0 3 -1 0 0 0 0"}:
            raise ValueError(f"Unsupported flux dimensions: {flux_dimension}")
        phi = np.empty(len(owner))
        phi[:len(neighbour)] = parse_array(phi_text, field=True)
        boundary_rows, boundary_u, patch_report = [], [], []
        for name, patch in patches.items():
            rows = np.arange(patch["start"], patch["start"] + patch["count"])
            flux = boundary_values(phi_text, name, len(rows))
            phi[rows] = flux
            values = boundary_values(u_text, name, len(rows), True)
            boundary_rows.append(rows)
            boundary_u.append(values)
            patch_report.append({"name": name, **patch, "net_outward_flux": float(flux.sum()),
                                 "centre_m": fc[rows].mean(axis=0).tolist()})
        inlet_flux = -sum(min(0, p["net_outward_flux"]) for p in patch_report)
        outlet_flux = sum(max(0, p["net_outward_flux"]) for p in patch_report)
        imbalance = abs(inlet_flux - outlet_flux) / max(inlet_flux, 1e-30)
        if inlet_flux <= 0 or imbalance > 0.01:
            raise ValueError(f"No inlet or excessive mass/volume imbalance: {imbalance:g}")
        residual = np.bincount(owner, weights=phi, minlength=len(u)) - np.bincount(neighbour, weights=phi[:len(neighbour)], minlength=len(u))
        speed = np.linalg.norm(u, axis=1)
        metadata = {
            "schema_version": 1, "source_zip": str(Path(zip_path).resolve()),
            "source_sha256": hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
            "vessel_file": str(Path(vessel_file).resolve()),
            "vessel_sha256": hashlib.sha256(Path(vessel_file).read_bytes()).hexdigest(),
            "solver_iteration": time_name, "flow_mode": "steady", "coordinate_scale": 1.0,
            "coordinate_unit": "m", "velocity_unit": "m/s", "n_cells": len(cc),
            "n_boundary_samples": sum(len(rows) for rows in boundary_rows),
            "bounds_m": [points.min(0).tolist(), points.max(0).tolist()], "alignment": alignment,
            "total_fluid_volume_m3": float(volume.sum()), "minimum_cell_volume_m3": float(volume.min()),
            "speed_percentiles_m_s": dict(zip(["min", "p50", "p95", "p99", "max"], np.percentile(speed, [0, 50, 95, 99, 100]).tolist())),
            "flux_dimensions": flux_dimension, "boundary_flux_imbalance_fraction": imbalance,
            "cell_residual_L1_over_inlet_flux": float(np.abs(residual).sum() / inlet_flux),
            "boundary_patches": patch_report, "interpolation": "8-neighbour inverse-distance-squared",
            "geometry_centroids": "area-weighted faces and volume-weighted face pyramids",
            "domain_support": "nearest-sample bounding radius times 2; bounded local extrapolation",
            "solver_residual_history_available": False, "mesh_independence_verified": False,
            "validation_limits": "Steady exported flow; original centreline wall model. 3x/5x multipliers are sensitivity fields, not independent CFD solves.",
        }
        rows = np.concatenate(boundary_rows)
        np.savez_compressed(output, points_m=np.concatenate([cc, fc[rows]]),
                            velocity_m_s=np.concatenate([u, np.concatenate(boundary_u)]),
                            support_radius_m=np.r_[cr, fr[rows]], metadata_json=json.dumps(metadata))
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {output}; physical coordinate scale 1; flux imbalance {imbalance:.3g}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--vessel-file", default="data/vessels/abdominal_0033_H_ABAO_AAA/real_vessel_segments.pkl")
    parser.add_argument("--output", default="data/cfd/abdominal_aorta_steady.npz")
    parser.add_argument("--time", dest="time_name")
    import_case(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
