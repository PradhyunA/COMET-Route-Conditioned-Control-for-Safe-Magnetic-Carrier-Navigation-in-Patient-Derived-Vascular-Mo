from __future__ import annotations

"""
Fast route candidate finder for the mapped 0050_H_CERE_H vessel.

Goal:
    Find start/target pairs that are likely to stress branch selection:
        - long routes
        - large trunk start
        - distal/small-radius target
        - many nearby branches / competing paths

This is for designing an adversarial branch-disambiguation benchmark BEFORE
running MPC/ACBC, not for cherry-picking after seeing controller results.
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np


DEFAULT_VESSEL_FILE = "data/0050_H_CERE_H/real_vessel_segments.pkl"
DEFAULT_OUTPUT_JSON = "outputs/visualization_0050/acbc_favorable_routes.json"


@dataclass
class SegmentLite:
    seg_id: int
    start: np.ndarray
    end: np.ndarray
    midpoint: np.ndarray
    direction: np.ndarray
    length: float
    radius: float
    level: int


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def load_segments(path: str) -> list[SegmentLite]:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    segments: list[SegmentLite] = []

    for item in raw:
        start = np.asarray(item["start"], dtype=float)
        end = np.asarray(item["end"], dtype=float)
        vec = end - start
        length = float(np.linalg.norm(vec))

        if length <= 1e-12:
            continue

        segments.append(
            SegmentLite(
                seg_id=int(item.get("seg_id", len(segments))),
                start=start,
                end=end,
                midpoint=0.5 * (start + end),
                direction=unit(vec),
                length=length,
                radius=float(item["radius"]),
                level=int(item.get("level", 0)),
            )
        )

    return segments


def endpoint_clusters_fast(
    segments: list[SegmentLite],
    voxel_size_m: float = 0.003,
) -> list[dict]:
    endpoint_records = []

    for seg in segments:
        endpoint_records.append((seg.start, seg.radius, seg.seg_id))
        endpoint_records.append((seg.end, seg.radius, seg.seg_id))

    buckets: dict[tuple[int, int, int], list[tuple[np.ndarray, float, int]]] = {}

    for point, radius, seg_id in endpoint_records:
        key = tuple(np.floor(point / voxel_size_m).astype(int).tolist())
        buckets.setdefault(key, []).append((point, radius, seg_id))

    clusters = []

    for key, records in buckets.items():
        pts = np.array([r[0] for r in records], dtype=float)
        radii = np.array([r[1] for r in records], dtype=float)
        seg_ids = [r[2] for r in records]

        clusters.append(
            {
                "center": pts.mean(axis=0),
                "count": int(len(records)),
                "radius_mean": float(radii.mean()),
                "radius_max": float(radii.max()),
                "seg_ids": seg_ids[:20],
                "voxel_key": key,
            }
        )

    return clusters


def nearest_segment_density(
    point: np.ndarray,
    segment_midpoints: np.ndarray,
    radius_m: float,
) -> int:
    d = np.linalg.norm(segment_midpoints - point, axis=1)
    return int(np.sum(d <= radius_m))


def direction_diversity_approx(
    point: np.ndarray,
    segment_midpoints: np.ndarray,
    segment_dirs: np.ndarray,
    radius_m: float,
    max_dirs: int = 40,
) -> float:
    d = np.linalg.norm(segment_midpoints - point, axis=1)
    idx = np.where(d <= radius_m)[0]

    if len(idx) < 2:
        return 0.0

    if len(idx) > max_dirs:
        idx = idx[np.argsort(d[idx])[:max_dirs]]

    dirs = segment_dirs[idx]
    dots = np.abs(dirs @ dirs.T)
    n = len(dirs)

    if n <= 1:
        return 0.0

    upper = dots[np.triu_indices(n, k=1)]
    diversity = 1.0 - float(np.mean(upper))
    return diversity


def route_branch_score(
    start: np.ndarray,
    target: np.ndarray,
    segment_midpoints: np.ndarray,
    segment_dirs: np.ndarray,
    samples: int = 32,
    search_radius_m: float = 0.008,
) -> dict:
    pts = np.linspace(start, target, samples)

    densities = []
    diversities = []

    for p in pts:
        density = nearest_segment_density(
            p,
            segment_midpoints=segment_midpoints,
            radius_m=search_radius_m,
        )
        diversity = direction_diversity_approx(
            p,
            segment_midpoints=segment_midpoints,
            segment_dirs=segment_dirs,
            radius_m=search_radius_m,
        )

        densities.append(density)
        diversities.append(diversity)

    densities = np.array(densities, dtype=float)
    diversities = np.array(diversities, dtype=float)

    return {
        "branch_hotspots": int(np.sum(densities >= 8)),
        "mean_nearby_segments": float(np.mean(densities)),
        "max_nearby_segments": float(np.max(densities)),
        "mean_direction_diversity": float(np.mean(diversities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument(
        "--voxel-mm",
        type=float,
        default=3.0,
        help="Endpoint clustering voxel size in mm.",
    )
    parser.add_argument("--min-distance-cm", type=float, default=6.0)
    parser.add_argument("--max-distance-cm", type=float, default=20.0)

    parser.add_argument(
        "--n-inlet-candidates",
        type=int,
        default=12,
        help="Number of large-radius candidate starts.",
    )
    parser.add_argument(
        "--n-outlet-candidates",
        type=int,
        default=80,
        help="Number of small/distal candidate targets.",
    )

    args = parser.parse_args()

    print(f"Loading vessel: {args.vessel_file}", flush=True)
    segments = load_segments(args.vessel_file)
    print(f"Loaded segments: {len(segments)}", flush=True)

    segment_midpoints = np.array([s.midpoint for s in segments], dtype=float)
    segment_dirs = np.array([s.direction for s in segments], dtype=float)

    print("Clustering endpoints quickly...", flush=True)
    clusters = endpoint_clusters_fast(
        segments,
        voxel_size_m=args.voxel_mm / 1000.0,
    )
    print(f"Endpoint clusters: {len(clusters)}", flush=True)

    clusters_by_radius = sorted(
        clusters,
        key=lambda c: c["radius_max"],
        reverse=True,
    )

    clusters_by_distal = sorted(
        clusters,
        key=lambda c: (c["radius_max"], c["count"]),
    )

    inlet_candidates = clusters_by_radius[: args.n_inlet_candidates]
    outlet_candidates = clusters_by_distal[: args.n_outlet_candidates]

    print(f"Inlet candidates: {len(inlet_candidates)}", flush=True)
    print(f"Outlet candidates: {len(outlet_candidates)}", flush=True)
    print("Scoring candidate routes...", flush=True)

    candidates = []

    for inlet_idx, inlet in enumerate(inlet_candidates):
        for outlet in outlet_candidates:
            start = np.asarray(inlet["center"], dtype=float)
            target = np.asarray(outlet["center"], dtype=float)

            chord_m = float(np.linalg.norm(target - start))
            chord_cm = chord_m * 100.0

            if chord_cm < args.min_distance_cm or chord_cm > args.max_distance_cm:
                continue

            b = route_branch_score(
                start=start,
                target=target,
                segment_midpoints=segment_midpoints,
                segment_dirs=segment_dirs,
            )

            score = (
                2.5 * chord_cm
                + 4.0 * b["branch_hotspots"]
                + 7.0 * b["mean_direction_diversity"]
                + 0.7 * b["mean_nearby_segments"]
                + 1.5 * (inlet["radius_max"] * 1000.0)
                - 1.2 * (outlet["radius_max"] * 1000.0)
            )

            candidates.append(
                {
                    "score": float(score),
                    "start_m": start.tolist(),
                    "target_m": target.tolist(),
                    "start_mm": (start * 1000.0).tolist(),
                    "target_mm": (target * 1000.0).tolist(),
                    "chord_distance_cm": float(chord_cm),
                    "start_radius_mm": float(inlet["radius_max"] * 1000.0),
                    "target_radius_mm": float(outlet["radius_max"] * 1000.0),
                    "start_cluster_count": int(inlet["count"]),
                    "target_cluster_count": int(outlet["count"]),
                    **b,
                }
            )

        print(
            f"  scored inlet {inlet_idx + 1}/{len(inlet_candidates)}",
            flush=True,
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[: args.top_k]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "vessel_file": args.vessel_file,
                "n_segments": len(segments),
                "n_endpoint_clusters": len(clusters),
                "selection_logic": {
                    "goal": (
                        "Find long, branch-dense, distal-outlet routes likely "
                        "to stress branch selection."
                    ),
                    "min_distance_cm": args.min_distance_cm,
                    "max_distance_cm": args.max_distance_cm,
                    "voxel_mm": args.voxel_mm,
                },
                "top_candidates": top,
            },
            f,
            indent=2,
        )

    print()
    print(f"Saved: {args.output_json}")
    print()

    if not top:
        print("No candidates found. Try lowering --min-distance-cm.")
        return

    for i, c in enumerate(top[:10], start=1):
        print(f"Candidate {i}")
        print(f"  score: {c['score']:.2f}")
        print(f"  chord distance: {c['chord_distance_cm']:.2f} cm")
        print(f"  start mm:  {np.round(c['start_mm'], 2)}")
        print(f"  target mm: {np.round(c['target_mm'], 2)}")
        print(f"  start radius: {c['start_radius_mm']:.2f} mm")
        print(f"  target radius: {c['target_radius_mm']:.2f} mm")
        print(f"  branch hotspots: {c['branch_hotspots']}")
        print(f"  mean nearby segments: {c['mean_nearby_segments']:.2f}")
        print(f"  direction diversity: {c['mean_direction_diversity']:.3f}")
        print()


if __name__ == "__main__":
    main()
